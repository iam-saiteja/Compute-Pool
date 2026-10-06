"""N-stage pipeline-parallel LoRA fine-tune, run on the master node (stage 0).

The model's decoder layers are split into N contiguous ranges, one per node. This
process holds stage 0 (the embedding and the first range) and drives training. Each
micro-batch goes forward through every stage, and its gradient comes back through
them in reverse. Gradients are accumulated, then applied once per optimizer step,
after every stage has confirmed its gradients are finite.

N comes from the cluster: one stage per online node, unless PIPELINE_STAGES is set.
Stage k runs on node k. A two-node cluster is the original two-stage pipeline.

Checkpointed every CKPT_EVERY steps. Every stage saves its own adapters, and the
master asks each stage to save before it saves its own, so a crash cannot leave the
stages at different steps (compute-pool#15). Rerunning with the same CHECKPOINT_JOB_ID
resumes. Progress since the last checkpoint is discarded on a lost connection.

The loss scale is owned by the last stage (workers/kaggle/amp.py). The master never
scales on its own: it passes the scale the last stage reported to every stage.

Needs cp_wire.py and pipeline_worker.py next to it. Run on the master from /kaggle/working:
    python3 pipeline_master.py
"""
import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoTokenizer

from cp_wire import recv, spawn, start_writer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/kaggle/working")
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from workers.kaggle import checkpoint as checkpoint_mod  # noqa: E402
from workers.kaggle.model_shard import load_stage  # noqa: E402
from workers.kaggle.pipeline import Stage, split_layers  # noqa: E402

MODEL_ID = "NousResearch/Meta-Llama-3.1-8B"
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
CHECKPOINT_URI = os.environ.get("CHECKPOINT_URI", "local:///kaggle/working/checkpoints")
JOB_ID = os.environ.get("CHECKPOINT_JOB_ID", "pipeline-lora-demo") + "-stage0"


def log(*args):
    print("[master]", *args, flush=True)


def default_stages():
    """One stage per online node in the cluster registry, at least two."""
    try:
        nodes = json.load(open("/etc/compute-pool/nodes.json"))["nodes"]
        return max(2, sum(1 for n in nodes if n.get("status") == "online"))
    except Exception:
        return 2


N_STAGES = int(os.environ.get("PIPELINE_STAGES", default_stages()))
WORKER_HOSTS = [f"node{k}" for k in range(1, N_STAGES)]
LAYERS = split_layers(AutoConfig.from_pretrained(MODEL_ID).num_hidden_layers, N_STAGES)

# peft refuses to run with the torchao 0.10 that Kaggle images ship, and LoRA does not need it.
subprocess.run(["pip", "uninstall", "-y", "-q", "torchao"], check=False)

class StageDisconnected(Exception):
    """A call to stage k failed. Carries k, so the caller knows which stage to reconnect."""

    def __init__(self, k, exc):
        super().__init__(f"stage {k}: {exc!r}")
        self.k = k
        self.exc = exc


workers, w_out, w_q = {}, {}, {}


def connect_stage(k, kill_first=False):
    """(Re)launch stage k's worker process on its node and (re)open its channel.
    kill_first=True is for a reconnect: a stray process from before the drop may still be
    running (compute-pool#16), and a second one talking over the same stdin/stdout would
    corrupt the protocol, so any previous one is killed first."""
    host = WORKER_HOSTS[k - 1]
    if kill_first:
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                         "pkill -9 -f pipeline_worker.py 2>/dev/null; true"], check=False)
    subprocess.run(
        ["scp", "-o", "ConnectTimeout=10",
         os.path.join(HERE, "pipeline_worker.py"), os.path.join(HERE, "cp_wire.py"), f"{host}:/kaggle/working/"],
        check=True,
    )
    env = " ".join(f"{k2}={shlex.quote(v)}" for k2, v in [
        ("STAGE_INDEX", str(k)), ("N_STAGES", str(N_STAGES)),
        ("CHECKPOINT_URI", CHECKPOINT_URI),
        ("CHECKPOINT_JOB_ID", os.environ.get("CHECKPOINT_JOB_ID", "pipeline-lora-demo")),
    ])
    proc = spawn(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=20",
                  host, f"pip uninstall -y -q torchao 2>/dev/null; cd /kaggle/working && {env} exec python3 -u pipeline_worker.py"])
    workers[k] = proc
    w_out[k] = proc.stdout
    w_q[k], _ = start_writer(proc.stdin)


for k in range(1, N_STAGES):
    connect_stage(k)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

log(f"{N_STAGES} stages: master holds layers {LAYERS[0].start}..{LAYERS[0].stop - 1}; "
    + ", ".join(f"{h} holds {LAYERS[k].start}..{LAYERS[k].stop - 1}" for k, h in enumerate(WORKER_HOSTS, start=1)))
log("loading model")
GPUS = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [torch.device("cpu")]
model = load_stage(MODEL_ID, LAYERS[0], include_embed=True, include_head=False, devices=GPUS)
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
params = [p for p in peft_model.parameters() if p.requires_grad and p.device.type != "meta"]
for p in params:
    p.data = p.data.float()
stage0 = Stage(core, None, first=True, last=False, params=params, lr=LR)


def call(k, msg):
    """Send one request to stage k and wait for its reply."""
    try:
        w_q[k].put(msg)
        return recv(w_out[k])
    except (EOFError, BrokenPipeError, OSError) as exc:
        raise StageDisconnected(k, exc) from exc


def save_ckpt(step):
    tmp = tempfile.mkdtemp(prefix="cp-pipeline-ckpt-")
    path = os.path.join(tmp, "adapters.pt")
    torch.save({n: p.detach().cpu() for n, p in peft_model.named_parameters() if p.requires_grad}, path)
    ckpt_store.save(step, {"adapters.pt": path}, meta={"step": step})


def resume():
    tmp = tempfile.mkdtemp(prefix="cp-pipeline-resume-")
    loaded = ckpt_store.load_latest(tmp)
    if not loaded:
        return 0
    step, _meta = loaded
    state = torch.load(os.path.join(tmp, "adapters.pt"), map_location=params[0].device)
    peft_model.load_state_dict(state, strict=False)
    log(f"resumed from checkpoint at step {step}")
    return step


ckpt_store = checkpoint_mod.open_store(CHECKPOINT_URI, JOB_ID)


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
    """Right-pad a batch of (ids, labels) to one length and build the attention mask."""
    max_len = max(len(ids) for ids, _ in batch)
    pad_id = tok.pad_token_id
    input_ids, labels, mask = [], [], []
    for ids, lab in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        mask.append([1] * len(ids) + [0] * pad)
    return torch.tensor(input_ids, device=stage0.dev), torch.tensor(labels), torch.tensor(mask, device=stage0.dev)


log("loading dataset")
ds = load_dataset("tatsu-lab/alpaca", split="train").select(range(DATASET_ROWS))
all_examples = [e for e in (encode(ex) for ex in ds) if any(label != -100 for label in e[1])]
n_holdout = min(EVAL_HOLDOUT, max(1, len(all_examples) // 10))
held_out, examples = all_examples[:n_holdout], all_examples[n_holdout:]
# Length-bucket the training order (seeded): see compute-pool#14.
_order = list(range(len(examples)))
random.Random(0).shuffle(_order)
_bucketed = []
for _c in range(0, len(_order), 64):
    _bucketed += sorted(_order[_c:_c + 64], key=lambda i: len(examples[i][0]))
examples = [examples[i] for i in _bucketed]
log(f"{len(examples)} training examples, {len(held_out)} held out for eval; "
    f"{STEPS} steps x {MICROBATCHES} micro-batches x {MICRO_BATCH_SIZE} examples/micro-batch")


def run_micro(j, ids, labels, mask, train):
    """One micro-batch through every stage. Returns (loss, scale). Training also runs the backward pass."""
    h = stage0.forward(j, ids, mask, train=train)
    for k in range(1, N_STAGES - 1):
        h = call(k, {"cmd": "fwd", "mb": j, "h": h, "mask": mask, "train": train})["h"]
    last = call(N_STAGES - 1, {"cmd": "fwd_loss", "mb": j, "h": h, "mask": mask, "labels": labels, "train": train})
    if not train:
        return last["loss"], None
    grad = last["grad"]
    for k in range(N_STAGES - 2, 0, -1):
        grad = call(k, {"cmd": "bwd", "mb": j, "grad": grad})["grad"]
    stage0.backward(j, grad)
    return last["loss"], last["scale"]


@torch.no_grad()
def evaluate():
    """Mean loss on held_out, which training never sees. Forward-only through every stage."""
    total = 0.0
    for ids, labels in held_out:
        loss, _ = run_micro(0, torch.tensor([ids], device=stage0.dev), torch.tensor([labels]),
                            None, train=False)
        total += float(loss)
    return total / len(held_out)


start_step = resume()
if start_step >= STEPS:
    log(f"checkpoint is already at step {start_step} >= STEPS={STEPS}; nothing to do")
    sys.exit(0)

try:
    log(f"held-out loss before this run: {evaluate():.4f}")
except StageDisconnected as exc:
    log(f"worker connection lost before training started ({exc.exc!r}). Nothing was trained; rerun.")
    sys.exit(1)

MAX_RECONNECTS = int(os.environ.get("MAX_RECONNECTS", "5"))


def recover_from_disconnect(failed_k):
    """A stage dropped mid-run. Reconnect it, roll every other live stage back to the
    last coordinated checkpoint, and reload the master's own state the same way, so
    every stage ends up at the same step and the run can continue instead of ending
    (compute-pool#16)."""
    log(f"stage {failed_k} disconnected; reconnecting it and rolling every stage back "
        f"to the last checkpoint")
    connect_stage(failed_k, kill_first=True)
    for other_k in range(1, N_STAGES):
        if other_k == failed_k:
            continue
        try:
            call(other_k, {"cmd": "reload"})
        except StageDisconnected as exc2:
            log(f"stage {exc2.k} also dropped while rolling back; reconnecting it too")
            connect_stage(exc2.k, kill_first=True)
    return resume()  # the master reloads its own last checkpoint the same way


started = time.time()
ema_loss = None
last_completed_step = start_step
reconnects = 0
step = start_step
while True:
    try:
        for step in range(step + 1, STEPS + 1):
            base = (step - 1) * MICROBATCHES * MICRO_BATCH_SIZE
            losses, scale = [], None
            for j in range(MICROBATCHES):
                batch = [examples[(base + j * MICRO_BATCH_SIZE + k) % len(examples)] for k in range(MICRO_BATCH_SIZE)]
                ids, labels, mask = collate(batch)
                loss, scale = run_micro(j, ids, labels, mask, train=True)
                losses.append(float(loss))

            # Every stage must agree the gradients are finite before any of them steps.
            finite = [stage0.grads_finite()] + [call(k, {"cmd": "check"})["finite"] for k in range(1, N_STAGES)]
            finite_all = all(finite)
            if not finite_all:
                log(f"step {step}: non-finite gradient on some stage, skipping update on all stages")
            new_scale = None
            for k in range(1, N_STAGES):
                reply = call(k, {"cmd": "apply", "apply": finite_all, "n_micro": MICROBATCHES,
                                 "scale": scale, "finite": finite_all})
                if k == N_STAGES - 1:
                    new_scale = reply["scale"]
            stage0.apply(finite_all, MICROBATCHES, scale, finite_all)

            step_loss = sum(losses) / len(losses)
            ema_loss = step_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * step_loss
            if step == start_step + 1 or step % LOG_EVERY == 0:
                n = step - start_step
                per_step = (time.time() - started) / n
                log(f"step {step}/{STEPS}  loss {step_loss:.4f} (smoothed {ema_loss:.4f})  "
                    f"scale {scale:.0f}  {per_step:.2f}s/step")
            last_completed_step = step
            if step % CKPT_EVERY == 0:
                # The stages save first and ack, then the master saves its own.
                for k in range(1, N_STAGES):
                    call(k, {"cmd": "ckpt", "step": step})
                save_ckpt(step)
            if step % EVAL_EVERY == 0:
                log(f"step {step}: held-out loss {evaluate():.4f}")
        break  # every step completed
    except StageDisconnected as exc:
        reconnects += 1
        if reconnects > MAX_RECONNECTS:
            last_ckpt = (last_completed_step // CKPT_EVERY) * CKPT_EVERY
            log(f"stage {exc.k} disconnected ({exc.exc!r}) at step {last_completed_step}/{STEPS}, "
                f"and reconnecting failed too many times ({MAX_RECONNECTS}).")
            log(f"discarded steps since the last checkpoint. rerun this script to resume from step {last_ckpt}.")
            sys.exit(1)
        try:
            step = recover_from_disconnect(exc.k)
            last_completed_step = step
            ema_loss = None  # the reloaded weights are the last checkpoint's, not mid-step
        except Exception as exc2:
            # scp/ssh failing, or another stage also found broken while rolling back: log it
            # and let the outer loop try again from the same point, up to MAX_RECONNECTS.
            log(f"reconnecting after stage {exc.k}'s disconnect failed too ({exc2!r}); will try again")

try:
    log(f"held-out loss after this run: {evaluate():.4f}")
except StageDisconnected as exc:
    log(f"worker connection lost while computing the final held-out loss ({exc.exc!r}); skipping it.")

try:
    for k in range(1, N_STAGES):
        call(k, {"cmd": "ckpt", "step": STEPS})
    save_ckpt(STEPS)
except StageDisconnected as exc:
    # Training already finished and was checkpointed inside the loop, so a lost connection
    # here only costs this very last save, not the run.
    log(f"worker connection lost while saving the final checkpoint ({exc.exc!r}); "
        f"the last CKPT_EVERY save still has the adapters.")
for k in range(1, N_STAGES):
    w_q[k].put({"cmd": "stop"})
    w_q[k].put(None)
for proc in workers.values():
    proc.wait()
log(f"done: {STEPS} steps in {time.time() - started:.0f}s; adapters saved via {CHECKPOINT_URI}")
