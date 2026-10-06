"""Stage 1 of a two-node pipeline-parallel LoRA fine-tune (run on the worker node).

Holds decoder layers SPLIT..end plus the final norm and LM head. For each
micro-batch from the master it runs forward and backward, accumulates the
adapter gradients, and returns the gradient at its input. When the master
sends "step", it applies one optimizer update over the accumulated gradients.
Each micro-batch is itself MICRO_BATCH_SIZE (master-side env var) examples
padded to a common length, with the padding mask sent alongside `h`.

Receiving and sending run on background threads, so the worker can accept the
next micro-batch while it computes the current one and while it returns the
previous gradient.

Checkpointed and resumed the same way as the master (workers/kaggle/checkpoint.py):
on start, it loads its own latest checkpoint if one exists and continues.

Uses dynamic fp16 loss scaling (workers/kaggle/amp.py), owned here: the scale
is applied once at `loss * scale` and that single factor is what's baked into
every gradient in the backward chain, including the master's half once it
continues backprop from the gradient this worker returns. So this worker
reports the scale it used with every micro-batch reply, and the master must
unscale with that reported value, not a value of its own -- an independently
adjusted scale on the master's side would silently diverge from what's
actually in the gradients it receives.

Needs cp_wire.py next to it.
"""
import os
import sys
import tempfile
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig

from cp_wire import start_reader, start_writer

sys.path.insert(0, "/kaggle/working")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from workers.kaggle import checkpoint as checkpoint_mod  # noqa: E402
from workers.kaggle.amp import DynamicLossScaler  # noqa: E402
from workers.kaggle.model_shard import load_stage  # noqa: E402

MODEL_ID = "NousResearch/Meta-Llama-3.1-8B"
SPLIT = 16
LR = 2e-4
CHECKPOINT_URI = os.environ.get("CHECKPOINT_URI", "local:///kaggle/working/checkpoints")
JOB_ID = os.environ.get("CHECKPOINT_JOB_ID", "pipeline-lora-demo") + "-worker"


def log(*args):
    print("[worker]", *args, file=sys.stderr, flush=True)


log("loading model")
# Load only this stage's tensors (layers SPLIT..end plus the LM head and norm),
# reading only the checkpoint shards that contain them. The embedding table and
# the other half's layers are never read or allocated (compute-pool#5).
# The embedding stays a meta module so peft's tied-weights check still finds it.
# See workers/kaggle/model_shard.py.
GPUS = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [torch.device("cpu")]
model = load_stage(MODEL_ID, range(SPLIT, AutoConfig.from_pretrained(MODEL_ID).num_hidden_layers),
                   include_embed=False, include_head=True, devices=GPUS)
torch.cuda.empty_cache()
dev = next(model.model.layers[0].parameters()).device  # first layer this stage runs
peft_model = get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ),
)
core = peft_model.base_model.model.model
head = peft_model.base_model.model.lm_head
layers = core.layers
params = [p for p in peft_model.parameters() if p.requires_grad]
for p in params:
    p.data = p.data.float()
opt = torch.optim.AdamW(params, lr=LR)
scaler = DynamicLossScaler()


def build_4d_mask(attention_mask, dtype, device):
    """Combine the causal mask with the batch's padding mask into the additive
    4D form the decoder layers expect directly -- see the matching comment in
    pipeline_master.py (compute-pool#6)."""
    seq_len = attention_mask.shape[1]
    min_value = torch.finfo(dtype).min
    causal = torch.triu(torch.full((seq_len, seq_len), min_value, device=device, dtype=dtype), diagonal=1)
    pad = (1.0 - attention_mask.to(dtype=dtype))[:, None, None, :] * min_value
    return causal[None, None, :, :] + pad


def forward_stage(h, attention_mask=None):
    h = h.to(dev)
    pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    pe = core.rotary_emb(h, pos)
    mask = build_4d_mask(attention_mask.to(h.device), h.dtype, h.device) if attention_mask is not None else None
    for layer in layers:
        # Layers may sit on different GPUs of this node, so move the inputs to each one.
        d = next(layer.parameters()).device
        out = layer(
            h.to(d),
            attention_mask=mask.to(d) if mask is not None else None,
            position_ids=pos.to(d),
            position_embeddings=tuple(t.to(d) for t in pe),
        )
        h = out[0] if isinstance(out, tuple) else out
    return h


def loss_fn(h, labels):
    logits = head(core.norm(h)).float()
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1).to(logits.device)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


ckpt_store = checkpoint_mod.open_store(CHECKPOINT_URI, JOB_ID)


def save_ckpt(step):
    tmp = tempfile.mkdtemp(prefix="cp-pipeline-ckpt-")
    path = os.path.join(tmp, "adapters_worker.pt")
    torch.save({n: p.detach().cpu() for n, p in peft_model.named_parameters() if p.requires_grad}, path)
    # Confirmed as a real (non-fatal) gap on a live cluster run: without
    # saving the scaler's state, a resumed run restarts it at init_scale and
    # has to re-earn any growth from scratch.
    ckpt_store.save(step, {"adapters_worker.pt": path}, meta={"step": step, "scaler": scaler.state_dict()})


def resume():
    tmp = tempfile.mkdtemp(prefix="cp-pipeline-resume-")
    loaded = ckpt_store.load_latest(tmp)
    if not loaded:
        return 0
    step, meta = loaded
    state = torch.load(os.path.join(tmp, "adapters_worker.pt"), map_location=dev)
    peft_model.load_state_dict(state, strict=False)
    if "scaler" in meta:
        scaler.load_state_dict(meta["scaler"])
    log(f"resumed from checkpoint at step {step} (loss scale {scaler.scale:.0f})")
    return step


PROFILE = os.environ.get("PROFILE") == "1"
prof = None
inbox = start_reader(sys.stdin.buffer)
outbox, writer = start_writer(sys.stdout.buffer)
accumulated = 0
steps = resume()
log("ready")

while True:
    msg = inbox.get()
    if msg is None or msg["cmd"] == "stop":
        break

    if msg["cmd"] == "eval":
        with torch.no_grad():
            h = msg["h"].to(dev)
            loss = loss_fn(forward_stage(h), msg["labels"])
        outbox.put({"loss": loss.detach()})
        continue

    if msg["cmd"] == "ckpt":
        # Master-driven, not self-timed: the master only writes its own
        # checkpoint for this step after this ack, so a crash between the two
        # saves can no longer leave them at different steps (compute-pool#15).
        save_ckpt(msg["step"])
        outbox.put({"ckpt_done": True})
        continue

    if msg["cmd"] == "mb":
        if PROFILE and steps == 4 and accumulated == 0 and prof is None:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        # scaler.scale only changes at the step boundary below, so every
        # micro-batch in this step snapshots (and reports) the same value.
        step_scale = scaler.scale
        h = msg["h"].to(dev).requires_grad_(True)
        loss = loss_fn(forward_stage(h, msg["mask"]), msg["labels"])
        (loss * step_scale).backward()
        outbox.put({"loss": loss.detach(), "grad": h.grad.detach(), "scale": step_scale})
        accumulated += 1
        if accumulated == msg["per_step"]:
            # Update as soon as the step's last gradient is sent, without waiting for the master.
            finite = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
            if finite:
                for p in params:
                    if p.grad is not None:
                        p.grad.div_(step_scale * accumulated)
                opt.step()
            opt.zero_grad(set_to_none=True)
            scaler.update(finite)
            accumulated = 0
            outbox.put({"finite": finite})
            steps += 1
            if prof is not None and steps == 5:
                prof.__exit__(None, None, None)
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25), file=sys.stderr, flush=True)
                prof.export_chrome_trace("/kaggle/working/trace_worker.json")
                prof = None

outbox.put(None)
writer.join()
save_ckpt(steps)
log("stopped after", steps, "optimizer steps")
sys.stderr.flush()
# The stdin reader thread is still blocked on a read; exit hard to avoid a shutdown crash.
os._exit(0)
