"""Stage 0 of a two-node pipeline-parallel LoRA fine-tune (run on the master node).

Holds the embedding and decoder layers 0..SPLIT-1 and drives training. Each
optimizer step is MICROBATCHES micro-batches: the master sends all of them
before waiting for gradients, so the worker computes one while the master runs
forward on the next. Gradients are accumulated and applied once per step.

Checkpointed every CKPT_EVERY steps to CHECKPOINT_URI (workers/kaggle/checkpoint.py),
and resumed automatically on restart: rerunning this script after an interruption
continues from the last saved step instead of starting over. If the worker
disconnects mid-run, progress is saved before exiting, so the fix is just to
rerun -- it is not a crash that loses work.

The fp16 loss scale is owned by the worker (workers/kaggle/amp.py), since the
scale it applies at `loss * scale` is the single factor baked into every
gradient in the backward chain, including this side's once it continues
backprop from the gradient the worker returns. This file unscales with the
scale value the worker reports per micro-batch, not an independent value of
its own.

Needs cp_wire.py and pipeline_worker.py next to it.
Run on the master from /kaggle/working:
    python3 pipeline_master.py
"""
import os
import shlex
import subprocess
import sys
import tempfile
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from cp_wire import recv, spawn, start_writer

# workers/kaggle/checkpoint.py: /kaggle/working on the real target (this file's
# directory, once shipped there), the repo root for local/dev testing.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/kaggle/working")
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from workers.kaggle import checkpoint as checkpoint_mod  # noqa: E402

MODEL_ID = "NousResearch/Meta-Llama-3.1-8B"
SPLIT = 16
MAX_LEN = 256
MICROBATCHES = int(os.environ.get("MICROBATCHES", "2"))
MICRO_BATCH_SIZE = int(os.environ.get("MICRO_BATCH_SIZE", "4"))
STEPS = int(os.environ.get("STEPS", "150"))
LR = 2e-4
LOG_EVERY = 10
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "25"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "25"))
EVAL_HOLDOUT = int(os.environ.get("EVAL_HOLDOUT", "20"))
DATASET_ROWS = 800
WORKER_HOST = "node1"
CHECKPOINT_URI = os.environ.get("CHECKPOINT_URI", "local:///kaggle/working/checkpoints")
JOB_ID = os.environ.get("CHECKPOINT_JOB_ID", "pipeline-lora-demo") + "-master"


def log(*args):
    print("[master]", *args, flush=True)


# peft refuses to run with the torchao 0.10 that Kaggle images ship, and LoRA does not need it.
subprocess.run(["pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run(["ssh", "-o", "BatchMode=yes", WORKER_HOST, "pip uninstall -y -q torchao"], check=False)

subprocess.run(
    ["scp", "-o", "ConnectTimeout=5",
     os.path.join(HERE, "pipeline_worker.py"), os.path.join(HERE, "cp_wire.py"),
     f"{WORKER_HOST}:/kaggle/working/"],
    check=True,
)
_worker_env = " ".join(
    f"{k}={shlex.quote(os.environ.get(k, default))}"
    for k, default in [("PROFILE", "0"), ("CHECKPOINT_URI", CHECKPOINT_URI),
                        ("CHECKPOINT_JOB_ID", os.environ.get("CHECKPOINT_JOB_ID", "pipeline-lora-demo")),
                        ("CKPT_EVERY", str(CKPT_EVERY))]
)
worker = spawn(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=20",
                WORKER_HOST, f"cd /kaggle/working && {_worker_env} exec python3 -u pipeline_worker.py"])
w_in, w_out = worker.stdin, worker.stdout
w_q, w_writer = start_writer(w_in)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

log("loading model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto", max_memory={0: "13GiB", 1: "13GiB"}
)
model.config.use_cache = False
# This stage only ever runs layers 0..SPLIT-1 (forward_stage0 below) -- drop
# the rest before LoRA wraps the model, so neither their weights nor unused
# LoRA adapters for them sit in GPU memory or the optimizer's state for the
# whole run (compute-pool#5). See the matching comment in pipeline_worker.py
# for what this does and does not fix.
#
# lm_head is NOT dropped even though this stage never calls it: deleting the
# worker's equivalent (embed_tokens) broke peft's get_peft_model() on a live
# cluster run (it calls model.get_input_embeddings() internally for a
# tied-weights check, which needs the module to exist structurally). lm_head
# is the output-embeddings analogue of that same mixin, so it's kept rather
# than risking the same failure a second time.
model.model.layers = model.model.layers[:SPLIT]
torch.cuda.empty_cache()
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
layers = core.layers
params = [p for p in peft_model.parameters() if p.requires_grad]
for p in params:
    p.data = p.data.float()
opt = torch.optim.AdamW(params, lr=LR)
dev = core.embed_tokens.weight.device

ckpt_store = checkpoint_mod.open_store(CHECKPOINT_URI, JOB_ID)


def save_ckpt(step):
    tmp = tempfile.mkdtemp(prefix="cp-pipeline-ckpt-")
    path = os.path.join(tmp, "adapters_master.pt")
    torch.save({n: p.detach().cpu() for n, p in peft_model.named_parameters() if p.requires_grad}, path)
    ckpt_store.save(step, {"adapters_master.pt": path}, meta={"step": step})


def resume():
    tmp = tempfile.mkdtemp(prefix="cp-pipeline-resume-")
    loaded = ckpt_store.load_latest(tmp)
    if not loaded:
        return 0
    step, _meta = loaded
    state = torch.load(os.path.join(tmp, "adapters_master.pt"), map_location=dev)
    peft_model.load_state_dict(state, strict=False)
    log(f"resumed from checkpoint at step {step}")
    return step


def build_4d_mask(attention_mask, dtype, device):
    """Combine the causal mask with the batch's padding mask into the additive
    4D form the decoder layers expect directly -- calling them one at a time
    (instead of the full model forward) skips transformers' usual mask-prep
    step, so it's redone here by hand rather than relying on an internal
    helper whose name/signature varies across transformers versions."""
    seq_len = attention_mask.shape[1]
    min_value = torch.finfo(dtype).min
    causal = torch.triu(torch.full((seq_len, seq_len), min_value, device=device, dtype=dtype), diagonal=1)
    pad = (1.0 - attention_mask.to(dtype=dtype))[:, None, None, :] * min_value
    return causal[None, None, :, :] + pad


def forward_stage0(input_ids, attention_mask=None):
    h = core.embed_tokens(input_ids)
    pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    pe = core.rotary_emb(h, pos)
    mask = build_4d_mask(attention_mask, h.dtype, h.device) if attention_mask is not None else None
    for layer in layers:
        out = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
        h = out[0] if isinstance(out, tuple) else out
    return h


def encode(example):
    prompt = f"### Instruction:\n{example['instruction']}\n"
    if example["input"]:
        prompt += f"### Input:\n{example['input']}\n"
    prompt += "### Response:\n"
    p_ids = tok(prompt)["input_ids"]
    f_ids = tok(prompt + example["output"] + tok.eos_token)["input_ids"][:MAX_LEN]
    cut = min(len(p_ids), len(f_ids))
    return f_ids, [-100] * cut + f_ids[cut:]


def collate(batch):
    """Right-pad a batch of (ids, labels) pairs to the same length and build
    the matching attention mask -- real multi-example batching needs
    rectangular tensors, not the variable-length lists encode() produces
    (compute-pool#6). Pad positions carry label -100 (ignored by the loss) and
    mask 0 (ignored by attention), so they cannot affect the result."""
    max_len = max(len(ids) for ids, _ in batch)
    pad_id = tok.pad_token_id
    input_ids, labels, mask = [], [], []
    for ids, lab in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        mask.append([1] * len(ids) + [0] * pad)
    return torch.tensor(input_ids, device=dev), torch.tensor(labels), torch.tensor(mask, device=dev)


log("loading dataset")
ds = load_dataset("tatsu-lab/alpaca", split="train").select(range(DATASET_ROWS))
all_examples = [e for e in (encode(ex) for ex in ds) if any(label != -100 for label in e[1])]
n_holdout = min(EVAL_HOLDOUT, max(1, len(all_examples) // 10))
held_out, examples = all_examples[:n_holdout], all_examples[n_holdout:]
log(f"{len(examples)} training examples, {len(held_out)} held out for eval; "
    f"{STEPS} steps x {MICROBATCHES} micro-batches x {MICRO_BATCH_SIZE} examples/micro-batch")


@torch.no_grad()
def evaluate():
    """Mean loss on held_out, which training never sees. Forward-only on both
    sides: no backward, no optimizer step, no loss-scale needed."""
    total = 0.0
    for ids, labels in held_out:
        h = forward_stage0(torch.tensor([ids], device=dev))
        w_q.put({"cmd": "eval", "h": h, "labels": torch.tensor([labels])})
        reply = recv(w_out)
        total += float(reply["loss"])
    return total / len(held_out)


start_step = resume()
if start_step >= STEPS:
    log(f"checkpoint is already at step {start_step} >= STEPS={STEPS}; nothing to do")
    w_q.put({"cmd": "stop"})
    w_q.put(None)
    w_writer.join()
    worker.wait()
    sys.exit(0)

try:
    log(f"held-out loss before this run: {evaluate():.4f}")
except (EOFError, BrokenPipeError, OSError) as exc:
    log(f"worker connection lost before training started ({exc!r}). Nothing was trained; rerun.")
    sys.exit(1)

started = time.time()
t_forward = t_wait = t_backward = t_step = 0.0
ema_loss = None
PROFILE = os.environ.get("PROFILE") == "1"
prof = None
last_completed_step = start_step
try:
    for step in range(start_step + 1, STEPS + 1):
        if PROFILE and step == start_step + 5:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        base = (step - 1) * MICROBATCHES * MICRO_BATCH_SIZE
        t = time.time()
        hs = []
        for j in range(MICROBATCHES):
            batch = [examples[(base + j * MICRO_BATCH_SIZE + k) % len(examples)] for k in range(MICRO_BATCH_SIZE)]
            input_ids, labels, mask = collate(batch)
            h = forward_stage0(input_ids, mask)
            hs.append(h)
            w_q.put({"cmd": "mb", "h": h.detach(), "labels": labels, "mask": mask, "per_step": MICROBATCHES})
        t_forward += time.time() - t

        losses = []
        step_scale = None
        for h in hs:
            t = time.time()
            reply = recv(w_out)
            t_wait += time.time() - t
            t = time.time()
            h.backward(reply["grad"].to(device=h.device, dtype=h.dtype))
            t_backward += time.time() - t
            losses.append(float(reply["loss"]))
            step_scale = reply["scale"]  # the worker's scale; every mb this step reports the same value

        t = time.time()
        finite = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
        if finite:
            for p in params:
                if p.grad is not None:
                    p.grad.div_(step_scale * MICROBATCHES)
            opt.step()
        else:
            log(f"step {step}: non-finite gradient, skipping update")
        opt.zero_grad(set_to_none=True)
        t_step += time.time() - t
        if prof is not None and step == start_step + 6:
            prof.__exit__(None, None, None)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25), file=sys.stderr, flush=True)
            prof.export_chrome_trace("/kaggle/working/trace_master.json")
            prof = None

        ack = recv(w_out)
        if ack["finite"] != finite:
            log(f"step {step}: master and worker disagree on gradient finiteness; they may have diverged")

        # Updated every step regardless of LOG_EVERY, so the smoothing isn't
        # biased by which steps happen to be printed.
        step_loss = sum(losses) / len(losses)
        ema_loss = step_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * step_loss

        if step == start_step + 1 or step % LOG_EVERY == 0:
            n = step - start_step
            per_step = (time.time() - started) / n
            log(f"step {step}/{STEPS}  loss {step_loss:.4f} (smoothed {ema_loss:.4f})  "
                f"scale {step_scale:.0f}  {per_step:.2f}s/step")
            log(f"  per step avg: master forward+send {t_forward / n:.2f}s, waiting on worker {t_wait / n:.2f}s, "
                f"master backward {t_backward / n:.2f}s, optimizer {t_step / n:.2f}s")
        last_completed_step = step
        if step % CKPT_EVERY == 0:
            # Coordinated save: the worker writes and acks its own checkpoint
            # for this step before the master writes its own, so a crash
            # between the two can no longer leave them at different steps
            # (compute-pool#15).
            w_q.put({"cmd": "ckpt", "step": step})
            recv(w_out)  # {"ckpt_done": True}
            save_ckpt(step)
        if step % EVAL_EVERY == 0:
            log(f"step {step}: held-out loss {evaluate():.4f}")
except (EOFError, BrokenPipeError, OSError) as exc:
    # The worker went away mid-run (connection lost, OOM-killed, Kaggle session
    # ended, ...). Do NOT save here: a save without the worker's ack would put
    # this side ahead of the worker's last coordinated checkpoint, the exact
    # desync compute-pool#15 fixes. Resume from the last coordinated checkpoint
    # instead; the steps since then are discarded, which keeps both sides at
    # the same step.
    last_ckpt = (last_completed_step // CKPT_EVERY) * CKPT_EVERY
    log(f"worker connection lost ({exc!r}) at step {last_completed_step}/{STEPS}.")
    log(f"discarded steps since the last checkpoint. rerun this script to resume from step {last_ckpt}.")
    sys.exit(1)

# Must run before the worker is told to stop. Training already finished and
# checkpointed above, so a lost connection here only costs this one number,
# not the run -- log a warning and still shut down cleanly.
try:
    log(f"held-out loss after this run: {evaluate():.4f}")
except (EOFError, BrokenPipeError, OSError) as exc:
    log(f"worker connection lost while computing the final held-out loss ({exc!r}); skipping it.")

w_q.put({"cmd": "stop"})
w_q.put(None)
w_writer.join()
worker.wait()
save_ckpt(STEPS)
log(f"done: {STEPS} steps in {time.time() - started:.0f}s; adapters saved via {CHECKPOINT_URI}")
