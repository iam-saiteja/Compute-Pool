"""Stage 1 of a two-node pipeline-parallel LoRA fine-tune (run on the worker node).

Holds decoder layers SPLIT..end plus the final norm and LM head. Receives the
hidden states at the split point, computes the loss, backpropagates locally,
and returns the gradient with respect to its input so the master can finish
the backward pass through its own layers.

Launched by pipeline_master.py over SSH; protocol is length-prefixed torch
blobs on stdin/stdout, so nothing else may be printed to stdout.
"""
import io
import struct
import sys

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

MODEL_ID = "NousResearch/Meta-Llama-3.1-8B"
SPLIT = 16
LOSS_SCALE = 1024.0
LR = 2e-4
CKPT = "/kaggle/working/ckpt_worker.pt"


def log(*args):
    print("[worker]", *args, file=sys.stderr, flush=True)


def read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("master closed the connection")
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


stream_in = sys.stdin.buffer
stream_out = sys.stdout.buffer
steps = 0
log("ready")

while True:
    msg = recv(stream_in)
    if msg["cmd"] == "stop":
        break

    h = msg["h"].to(core.embed_tokens.weight.device).requires_grad_(True)
    loss = loss_fn(forward_stage(h), msg["labels"])
    (loss * LOSS_SCALE).backward()

    finite = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
    if finite:
        for p in params:
            if p.grad is not None:
                p.grad.div_(LOSS_SCALE)
        opt.step()
    else:
        log("non-finite gradient, skipping optimizer step")
    opt.zero_grad(set_to_none=True)

    send(stream_out, {"loss": loss.item(), "grad": h.grad.detach().cpu(), "finite": finite})
    steps += 1
    if steps % 50 == 0:
        save_ckpt()

save_ckpt()
log("stopped after", steps, "steps")
