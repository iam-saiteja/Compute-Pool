"""Stage k (k >= 1) of an N-stage pipeline-parallel LoRA fine-tune, run on a worker node.

The master (stage 0) routes every micro-batch through the stages in order. This process
only ever talks to the master, over the SSH link it was started with. Each message
asks for one thing:
  fwd       run this stage's layers, return the output (middle stages)
  fwd_loss  run the last stage, compute the loss, backpropagate, return the gradient
            for the stage before it (last stage only)
  bwd       backpropagate a gradient through this stage, return the one for upstream
  check     report whether this stage's gradients are finite
  apply     divide the accumulated gradients, step if the whole pipeline agreed, clear
  ckpt      save this stage's adapters for a step
  stop      exit

Stage index, stage count and checkpoint settings come from the environment, set by the
master when it launches this process. The model code lives in workers/kaggle/pipeline.py.

Needs cp_wire.py next to it.
"""
import os
import sys
import tempfile

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig

from cp_wire import start_reader, start_writer

sys.path.insert(0, "/kaggle/working")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from workers.kaggle import checkpoint as checkpoint_mod  # noqa: E402
from workers.kaggle.amp import DynamicLossScaler  # noqa: E402
from workers.kaggle.model_shard import load_stage  # noqa: E402
from workers.kaggle.chain import HalfPipeline  # noqa: E402
from workers.kaggle.pipeline import make_halves, split_layers  # noqa: E402

MODEL_ID = "NousResearch/Meta-Llama-3.1-8B"
LR = 2e-4
N_STAGES = int(os.environ["N_STAGES"])
STAGE = int(os.environ["STAGE_INDEX"])
LAST = STAGE == N_STAGES - 1
CHECKPOINT_URI = os.environ.get("CHECKPOINT_URI", "local:///kaggle/working/checkpoints")
JOB_ID = os.environ.get("CHECKPOINT_JOB_ID", "pipeline-lora-demo") + f"-stage{STAGE}"


def log(*args):
    print(f"[worker stage {STAGE}]", *args, file=sys.stderr, flush=True)


n_layers = AutoConfig.from_pretrained(MODEL_ID).num_hidden_layers
LAYERS = split_layers(n_layers, N_STAGES)[STAGE]
GPUS = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [torch.device("cpu")]

log(f"loading layers {LAYERS.start}..{LAYERS.stop - 1}{' + head' if LAST else ''}")
model = load_stage(MODEL_ID, LAYERS, include_embed=False, include_head=LAST, devices=GPUS)
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
head = peft_model.base_model.model.lm_head
params = [p for p in peft_model.parameters() if p.requires_grad and p.device.type != "meta"]
for p in params:
    p.data = p.data.float()
scaler = DynamicLossScaler() if LAST else None
# This node's layers, split into two halves that run in parallel on its two GPUs.
front, back = make_halves(core, head, last=LAST, lr=LR, scaler=scaler)
ckpt_store = checkpoint_mod.open_store(CHECKPOINT_URI, JOB_ID)


def save_ckpt(step):
    tmp = tempfile.mkdtemp(prefix="cp-pipeline-ckpt-")
    path = os.path.join(tmp, "adapters.pt")
    torch.save({n: p.detach().cpu() for n, p in peft_model.named_parameters() if p.requires_grad}, path)
    meta = {"step": step}
    if scaler is not None:
        # The scale is the loss scale this stage's gradients were computed under.
        meta["scaler"] = scaler.state_dict()
    ckpt_store.save(step, {"adapters.pt": path}, meta=meta)


def resume():
    tmp = tempfile.mkdtemp(prefix="cp-pipeline-resume-")
    loaded = ckpt_store.load_latest(tmp)
    if not loaded:
        return 0
    step, meta = loaded
    state = torch.load(os.path.join(tmp, "adapters.pt"), map_location=params[0].device)
    peft_model.load_state_dict(state, strict=False)
    if scaler is not None and "scaler" in meta:
        scaler.load_state_dict(meta["scaler"])
    log(f"resumed from checkpoint at step {step}")
    return step


inbox = start_reader(sys.stdin.buffer)
outbox, writer = start_writer(sys.stdout.buffer)
resume()
pipe = HalfPipeline(front, back, reply=outbox.put)
log("ready")

while True:
    msg = inbox.get()
    if msg is None or msg["cmd"] == "stop":
        break
    cmd = msg["cmd"]
    if cmd in ("fwd", "fwd_loss", "bwd"):
        # Queued; the reply goes out when this request finishes. Replies carry the request id.
        pipe.submit(msg)
    elif cmd == "check":
        outbox.put({"id": msg["id"], "finite": pipe.check()})
    elif cmd == "apply":
        # The master only sends control messages once every reply for the step is back,
        # so nothing is in flight here.
        new_scale = pipe.apply(msg["apply"], msg["n_micro"], msg["scale"], msg["finite"])
        outbox.put({"id": msg["id"], "scale": new_scale})
    elif cmd == "ckpt":
        save_ckpt(msg["step"])
        outbox.put({"id": msg["id"], "ckpt_done": True})

outbox.put(None)
writer.join()
log("stopped")
sys.stderr.flush()
# The stdin reader thread is still blocked on a read; exit hard to avoid a shutdown crash.
os._exit(0)
