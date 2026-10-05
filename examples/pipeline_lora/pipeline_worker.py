"""Stage 1 of a two-node pipeline-parallel LoRA fine-tune (run on the worker node).

Holds decoder layers SPLIT..end plus the final norm and LM head. For each
micro-batch from the master it runs forward and backward, accumulates the
adapter gradients, and returns the gradient at its input. When the master
sends "step", it applies one optimizer update over the accumulated gradients.

Receiving and sending run on background threads, so the worker can accept the
next micro-batch while it computes the current one and while it returns the
previous gradient.

Needs cp_wire.py next to it.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from cp_wire import start_reader, start_writer

MODEL_ID = "NousResearch/Meta-Llama-3.1-8B"
SPLIT = 16
LOSS_SCALE = 1024.0
LR = 2e-4
CKPT_EVERY = 25
CKPT = "/kaggle/working/ckpt_worker.pt"


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


def save_ckpt():
    torch.save({n: p.detach().cpu() for n, p in peft_model.named_parameters() if p.requires_grad}, CKPT)


inbox = start_reader(sys.stdin.buffer)
outbox, writer = start_writer(sys.stdout.buffer)
accumulated = 0
steps = 0
log("ready")

while True:
    msg = inbox.get()
    if msg is None or msg["cmd"] == "stop":
        break

    if msg["cmd"] == "mb":
        t0 = time.time()
        h = msg["h"].to(core.embed_tokens.weight.device).requires_grad_(True)
        loss = loss_fn(forward_stage(h), msg["labels"])
        (loss * LOSS_SCALE).backward()
        grad = h.grad.detach().cpu()
        outbox.put({"loss": loss.item(), "grad": grad, "compute_s": time.time() - t0})
        accumulated += 1
        continue

    if msg["cmd"] == "step":
        finite = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
        if msg["apply"] and finite and accumulated:
            for p in params:
                if p.grad is not None:
                    p.grad.div_(LOSS_SCALE * accumulated)
            opt.step()
        opt.zero_grad(set_to_none=True)
        accumulated = 0
        outbox.put({"finite": finite})
        steps += 1
        if steps % CKPT_EVERY == 0:
            save_ckpt()

outbox.put(None)
writer.join()
save_ckpt()
log("stopped after", steps, "optimizer steps")
sys.stderr.flush()
# The stdin reader thread is still blocked on a read; exit hard to avoid a shutdown crash.
os._exit(0)
