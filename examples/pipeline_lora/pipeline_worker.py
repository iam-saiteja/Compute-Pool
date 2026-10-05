"""Stage 1 of a two-node pipeline-parallel LoRA fine-tune (run on the worker node).

Holds decoder layers SPLIT..end plus the final norm and LM head. For each
micro-batch from the master it runs forward and backward, accumulates the
adapter gradients, and returns the gradient at its input. When the master
sends "step", it applies one optimizer update over the accumulated gradients.

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
from transformers import AutoModelForCausalLM

from cp_wire import start_reader, start_writer

sys.path.insert(0, "/kaggle/working")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from workers.kaggle import checkpoint as checkpoint_mod  # noqa: E402
from workers.kaggle.amp import DynamicLossScaler  # noqa: E402

MODEL_ID = "NousResearch/Meta-Llama-3.1-8B"
SPLIT = 16
LR = 2e-4
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "25"))
CHECKPOINT_URI = os.environ.get("CHECKPOINT_URI", "local:///kaggle/working/checkpoints")
JOB_ID = os.environ.get("CHECKPOINT_JOB_ID", "pipeline-lora-demo") + "-worker"


def log(*args):
    print("[worker]", *args, file=sys.stderr, flush=True)


log("loading model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto", max_memory={0: "13GiB", 1: "13GiB"}
)
model.config.use_cache = False
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
layers = core.layers[SPLIT:]
params = [p for p in peft_model.parameters() if p.requires_grad]
for p in params:
    p.data = p.data.float()
opt = torch.optim.AdamW(params, lr=LR)
scaler = DynamicLossScaler()


def forward_stage(h):
    h = h.to(core.embed_tokens.weight.device)
    pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    pe = core.rotary_emb(h, pos)
    for layer in layers:
        out = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe)
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
    state = torch.load(os.path.join(tmp, "adapters_worker.pt"), map_location=core.embed_tokens.weight.device)
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
            h = msg["h"].to(core.embed_tokens.weight.device)
            loss = loss_fn(forward_stage(h), msg["labels"])
        outbox.put({"loss": loss.detach()})
        continue

    if msg["cmd"] == "mb":
        if PROFILE and steps == 4 and accumulated == 0 and prof is None:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        # scaler.scale only changes at the step boundary below, so every
        # micro-batch in this step snapshots (and reports) the same value.
        step_scale = scaler.scale
        h = msg["h"].to(core.embed_tokens.weight.device).requires_grad_(True)
        loss = loss_fn(forward_stage(h), msg["labels"])
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
            if steps % CKPT_EVERY == 0:
                save_ckpt(steps)

outbox.put(None)
writer.join()
save_ckpt(steps)
log("stopped after", steps, "optimizer steps")
sys.stderr.flush()
# The stdin reader thread is still blocked on a read; exit hard to avoid a shutdown crash.
os._exit(0)
