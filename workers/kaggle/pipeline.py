"""N-stage pipeline-parallel training of a causal LM, one Stage per node.

The decoder layers are split into contiguous ranges, one range per stage. Stage 0
(the master) holds the embedding and the first range. The last stage holds the
final norm and the LM head, and computes the loss.

A micro-batch goes forward through every stage in order. Its gradient comes back
through the same stages in reverse. Each stage keeps the autograd graph of a
micro-batch until that gradient arrives, then backpropagates through it.

The fp16 loss scale is owned by the last stage (it applies `loss * scale`), and every
gradient in the chain carries that one factor. Each stage divides its own gradients
by the scale the master reports, so no stage ever scales on its own.
"""
import torch
import torch.nn.functional as F


def split_layers(n_layers, n_stages):
    """Contiguous, near-equal ranges of decoder layers, one per stage."""
    base, extra = divmod(n_layers, n_stages)
    ranges, start = [], 0
    for s in range(n_stages):
        size = base + (1 if s < extra else 0)
        ranges.append(range(start, start + size))
        start += size
    return ranges


def build_4d_mask(attention_mask, dtype, device):
    """Causal mask combined with padding, in the additive 4D form the decoder layers take.
    Layers are called directly, so transformers' own mask preparation is skipped."""
    seq_len = attention_mask.shape[1]
    min_value = torch.finfo(dtype).min
    causal = torch.triu(torch.full((seq_len, seq_len), min_value, device=device, dtype=dtype), diagonal=1)
    pad = (1.0 - attention_mask.to(dtype=dtype))[:, None, None, :] * min_value
    return causal[None, None, :, :] + pad


class Stage:
    def __init__(self, core, head, first, last, params, lr, scaler=None):
        """core: the LlamaModel-like module (embed_tokens if first, layers, rotary_emb, norm if last).
        head: the LM head (only used on the last stage). params: this stage's trainable parameters.
        scaler: the DynamicLossScaler, on the last stage only."""
        self.core, self.head = core, head
        self.first, self.last = first, last
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=lr)
        self.scaler = scaler
        self.pending = {}  # micro-batch id -> (input leaf or None, output tensor)
        self.dev = next(core.layers[0].parameters()).device

    def _run_layers(self, h, attention_mask):
        mask = build_4d_mask(attention_mask.to(h.device), h.dtype, h.device) if attention_mask is not None else None
        pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        pe = self.core.rotary_emb(h, pos)
        for layer in self.core.layers:
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

    def forward(self, mb, x, attention_mask=None, train=True):
        """Non-final stage. Stage 0 takes token ids; the others take the previous stage's output.
        Returns this stage's output, detached, to send downstream."""
        if self.first:
            h = self.core.embed_tokens(x.to(self.core.embed_tokens.weight.device))
            leaf = None
        else:
            leaf = x.to(self.dev).detach().requires_grad_(train)
            h = leaf
        out = self._run_layers(h, attention_mask)
        if train:
            self.pending[mb] = (leaf, out)
        return out.detach()

    def forward_loss(self, mb, x, attention_mask, labels, train=True):
        """Final stage. Returns (loss, gradient for the upstream stage, scale used).
        In training it also backpropagates here, so nothing is kept for later."""
        leaf = x.to(self.dev).detach().requires_grad_(train)
        out = self._run_layers(leaf, attention_mask)
        logits = self.head(self.core.norm(out.to(self.head.weight.device))).float()
        shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1).to(logits.device)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        if not train:
            return loss.detach(), None, None
        scale = self.scaler.scale
        (loss * scale).backward()
        return loss.detach(), leaf.grad.detach(), scale

    def backward(self, mb, grad_out):
        """Backpropagate a micro-batch's gradient through this stage. Returns the gradient
        for the upstream stage, or None on stage 0, which has token ids as input."""
        leaf, out = self.pending.pop(mb)
        out.backward(grad_out.to(device=out.device, dtype=out.dtype))
        return leaf.grad.detach() if leaf is not None else None

    def grads_finite(self):
        return all(torch.isfinite(p.grad).all().item() for p in self.params if p.grad is not None)

    def apply(self, apply_update, n_micro, scale, finite_all):
        """End of a step. Divide the accumulated gradients by scale * n_micro, step if the
        whole pipeline agreed the gradients are finite, then clear state. On the last
        stage, also advance the loss scaler from the same agreed value."""
        if apply_update:
            for p in self.params:
                if p.grad is not None:
                    p.grad.div_(scale * n_micro)
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.pending.clear()
        if self.last and self.scaler is not None:
            return self.scaler.update(finite_all)
        return None

    def trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.named_params()}

    def named_params(self):
        """This stage's trainable parameters, skipping parts it never loaded (meta tensors)."""
        mods = [("", self.core)] + ([("lm_head.", self.head)] if self.last else [])
        return [(prefix + n, p) for prefix, m in mods for n, p in m.named_parameters()
                if p.requires_grad and p.device.type != "meta"]


class Part(torch.nn.Module):
    """A slice of a node's layers, shaped like the core module that Stage expects."""

    def __init__(self, layers, rotary_emb, norm=None):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.rotary_emb = rotary_emb
        if norm is not None:
            self.norm = norm


def _trainable(*modules):
    return [p for m in modules for p in m.parameters() if p.requires_grad and p.device.type != "meta"]


def make_halves(core, head, last, lr, scaler=None):
    """Split a node's layers into a front half and a back half, each a Stage.
    The front half runs on the node's first GPU and the back half on its second, so the
    two can work on different micro-batches at the same time (chain.HalfPipeline).
    Returns (front Stage, back Stage)."""
    layers = list(core.layers)
    if len(layers) < 2:
        raise ValueError("a node needs at least two layers to split across its GPUs")
    cut = len(layers) // 2
    front_part = Part(layers[:cut], core.rotary_emb)
    back_part = Part(layers[cut:], core.rotary_emb, core.norm if last else None)
    front = Stage(front_part, None, first=False, last=False, params=_trainable(front_part), lr=lr)
    back = Stage(back_part, head if last else None, first=False, last=last,
                 params=_trainable(back_part, head) if last else _trainable(back_part),
                 lr=lr, scaler=scaler if last else None)
    return front, back
