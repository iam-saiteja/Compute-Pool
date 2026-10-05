"""Stage 0 of a two-node pipeline-parallel LoRA fine-tune (run on the master node).

Holds the embedding and decoder layers 0..SPLIT-1, drives the training loop,
and ships activations to the worker over SSH. The worker (pipeline_worker.py)
runs the rest of the model and returns the gradient at the split point.

Each step costs one round trip over the tunnel plus the transfer of the
hidden states (batch 1 x MAX_LEN x hidden, fp16) in both directions.

Run on the master from /kaggle/working:
    python3 pipeline_master.py
"""
import io
import os
import struct
import subprocess
import sys
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "NousResearch/Meta-Llama-3.1-8B"
SPLIT = 16
MAX_LEN = 256
STEPS = 300
LR = 2e-4
LOSS_SCALE = 1024.0
LOG_EVERY = 10
DATASET_ROWS = 800
CKPT = "/kaggle/working/ckpt_master.pt"
WORKER_HOST = "node1"
WORKER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_worker.py")


def log(*args):
    print("[master]", *args, flush=True)


def read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("worker closed the connection; see its log above")
        buf.extend(chunk)
    return bytes(buf)


def recv(stream):
    n = struct.unpack("<Q", read_exact(stream, 8))[0]
    return torch.load(io.BytesIO(read_exact(stream, n)), weights_only=False)


def send(stream, obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


subprocess.run(["scp", "-o", "ConnectTimeout=5", WORKER_FILE, f"{WORKER_HOST}:/kaggle/working/pipeline_worker.py"], check=True)
worker = subprocess.Popen(
    ["ssh", "-o", "BatchMode=yes", WORKER_HOST, "cd /kaggle/working && exec python3 -u pipeline_worker.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
)
w_in, w_out = worker.stdin, worker.stdout

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
    labels = [-100] * cut + f_ids[cut:]
    return f_ids, labels


log("loading dataset")
ds = load_dataset("tatsu-lab/alpaca", split="train").select(range(DATASET_ROWS))
examples = [encode(ex) for ex in ds]
examples = [e for e in examples if any(label != -100 for label in e[1])]
log(f"{len(examples)} usable examples; training {STEPS} steps")

dev = core.embed_tokens.weight.device
started = time.time()
for step in range(1, STEPS + 1):
    ids, labels = examples[(step - 1) % len(examples)]
    h = forward_stage0(torch.tensor([ids], device=dev))
    send(w_in, {"cmd": "step", "h": h.detach().cpu(), "labels": torch.tensor([labels])})
    reply = recv(w_out)

    h.backward(reply["grad"].to(device=h.device, dtype=h.dtype))
    finite = reply["finite"] and all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
    if finite:
        for p in params:
            if p.grad is not None:
                p.grad.div_(LOSS_SCALE)
        opt.step()
    else:
        log(f"step {step}: non-finite gradient, skipping update")
    opt.zero_grad(set_to_none=True)

    if step == 1 or step % LOG_EVERY == 0:
        per_step = (time.time() - started) / step
        log(f"step {step}/{STEPS}  loss {reply['loss']:.4f}  {per_step:.2f}s/step")
    if step % 50 == 0:
        torch.save({n: p.detach().cpu() for n, p in peft_model.named_parameters() if p.requires_grad}, CKPT)

send(w_in, {"cmd": "stop"})
worker.wait()
torch.save({n: p.detach().cpu() for n, p in peft_model.named_parameters() if p.requires_grad}, CKPT)
log(f"done: {STEPS} steps in {time.time() - started:.0f}s; adapters saved to {CKPT}")
