"""Stage 0 of a two-node pipeline-parallel LoRA fine-tune (run on the master node).

Holds the embedding and decoder layers 0..SPLIT-1 and drives training. Each
optimizer step is MICROBATCHES micro-batches: the master sends all of them
before waiting for gradients, so the worker computes one while the master runs
forward on the next. Gradients are accumulated and applied once per step.

Needs cp_wire.py and pipeline_worker.py next to it.
Run on the master from /kaggle/working:
    python3 pipeline_master.py
"""
import os
import subprocess
import sys
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from cp_wire import recv, spawn, start_writer

MODEL_ID = "NousResearch/Meta-Llama-3.1-8B"
SPLIT = 16
MAX_LEN = 256
MICROBATCHES = int(os.environ.get("MICROBATCHES", "2"))
STEPS = 150
LR = 2e-4
LOSS_SCALE = 1024.0
LOG_EVERY = 10
CKPT_EVERY = 25
DATASET_ROWS = 800
CKPT = "/kaggle/working/ckpt_master.pt"
WORKER_HOST = "node1"
HERE = os.path.dirname(os.path.abspath(__file__))


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
worker = spawn(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=20",
                WORKER_HOST, "cd /kaggle/working && exec python3 -u pipeline_worker.py"])
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
layers = core.layers[:SPLIT]
params = [p for p in peft_model.parameters() if p.requires_grad]
for p in params:
    p.data = p.data.float()
opt = torch.optim.AdamW(params, lr=LR)
dev = core.embed_tokens.weight.device


def forward_stage0(input_ids):
    h = core.embed_tokens(input_ids)
    pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    pe = core.rotary_emb(h, pos)
    for layer in layers:
        out = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe)
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


def save_ckpt():
    torch.save({n: p.detach().cpu() for n, p in peft_model.named_parameters() if p.requires_grad}, CKPT)


log("loading dataset")
ds = load_dataset("tatsu-lab/alpaca", split="train").select(range(DATASET_ROWS))
examples = [e for e in (encode(ex) for ex in ds) if any(label != -100 for label in e[1])]
log(f"{len(examples)} usable examples; {STEPS} steps x {MICROBATCHES} micro-batches")

started = time.time()
t_forward = t_wait = t_backward = t_step = 0.0
for step in range(1, STEPS + 1):
    base = (step - 1) * MICROBATCHES
    t = time.time()
    hs = []
    for j in range(MICROBATCHES):
        ids, labels = examples[(base + j) % len(examples)]
        h = forward_stage0(torch.tensor([ids], device=dev))
        hs.append(h)
        w_q.put({"cmd": "mb", "h": h.detach().cpu(), "labels": torch.tensor([labels])})
    t_forward += time.time() - t

    losses, worker_compute = [], []
    for h in hs:
        t = time.time()
        reply = recv(w_out)
        t_wait += time.time() - t
        t = time.time()
        h.backward(reply["grad"].to(device=h.device, dtype=h.dtype))
        t_backward += time.time() - t
        losses.append(reply["loss"])
        worker_compute.append(reply["compute_s"])

    t = time.time()
    finite = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
    if finite:
        for p in params:
            if p.grad is not None:
                p.grad.div_(LOSS_SCALE * MICROBATCHES)
        opt.step()
    else:
        log(f"step {step}: non-finite gradient, skipping update")
    opt.zero_grad(set_to_none=True)
    t_step += time.time() - t

    ack = recv(w_out)
    if ack["finite"] != finite:
        log(f"step {step}: master and worker disagree on gradient finiteness; they may have diverged")

    if step == 1 or step % LOG_EVERY == 0:
        per_step = (time.time() - started) / step
        n = step
        log(f"step {step}/{STEPS}  loss {sum(losses) / len(losses):.4f}  {per_step:.2f}s/step")
        log(f"  per step avg: master forward+send {t_forward / n:.2f}s, waiting on worker {t_wait / n:.2f}s, "
            f"master backward {t_backward / n:.2f}s, optimizer {t_step / n:.2f}s; "
            f"worker compute (last micro-batch) {worker_compute[-1]:.2f}s")
    if step % CKPT_EVERY == 0:
        save_ckpt()

w_q.put({"cmd": "stop"})
w_q.put(None)
w_writer.join()
worker.wait()
save_ckpt()
log(f"done: {STEPS} steps in {time.time() - started:.0f}s; adapters saved to {CKPT}")
