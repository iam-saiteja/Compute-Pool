"""Check that running micro-batches concurrently through a 3-stage chain still gives the
gradients of one full model over the same micro-batches.

Each stage runs in its own thread with a FIFO inbox, standing in for a worker process.
Run with the project's venv:
    python tests/test_pipeline_overlap.py
"""
import os
import queue
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, LlamaConfig  # noqa: E402

from workers.kaggle.amp import DynamicLossScaler  # noqa: E402
from workers.kaggle.chain import Channel, HalfPipeline, run_micro_batches  # noqa: E402
from workers.kaggle.model_shard import load_stage  # noqa: E402
from workers.kaggle.pipeline import Stage, split_layers  # noqa: E402

N_LAYERS = 4
TOL = 1e-4


class Remote:
    """A stage that serves requests in order, in its own thread: a worker process, in miniature."""

    def __init__(self, stage):
        self.stage = stage
        self.inbox = queue.Queue()
        self.outbox = queue.Queue()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            msg = self.inbox.get()
            if msg["cmd"] == "fwd":
                reply = {"h": self.stage.forward(msg["mb"], msg["h"], msg["mask"], train=True)}
            elif msg["cmd"] == "fwd_loss":
                loss, grad, scale = self.stage.forward_loss(msg["mb"], msg["h"], msg["mask"], msg["labels"])
                reply = {"loss": loss, "grad": grad, "scale": scale}
            elif msg["cmd"] == "bwd":
                reply = {"grad": self.stage.backward(msg["mb"], msg["grad"])}
            reply["id"] = msg["id"]
            self.outbox.put(reply)

    def channel(self):
        return Channel(send=self.inbox.put, recv=self.outbox.get)


def _checkpoint(root):
    cfg = LlamaConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=N_LAYERS,
                      num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=False)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(root, max_shard_size="40KB")
    return root


def _checkpoint6(root):
    cfg = LlamaConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=False)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(root, max_shard_size="40KB")
    return root


def test_concurrent_micro_batches_match_full_model():
    with tempfile.TemporaryDirectory() as root:
        ckpt = _checkpoint(root)
        ranges = split_layers(N_LAYERS, 3)
        cpu = [torch.device("cpu")]
        scaler = DynamicLossScaler(init_scale=1024.0)
        stages = []
        for s, r in enumerate(ranges):
            first, last = s == 0, s == len(ranges) - 1
            m = load_stage(ckpt, r, include_embed=first, include_head=last, devices=cpu, dtype=torch.float32)
            params = [p for p in m.parameters() if p.requires_grad and p.device.type != "meta"]
            stages.append(Stage(m.model, m.lm_head, first=first, last=last, params=params, lr=1e-3,
                                scaler=scaler if last else None))
        stage0 = stages[0]
        remotes = [Remote(st) for st in stages[1:]]
        channels = {k + 1: r.channel() for k, r in enumerate(remotes)}

        # Three micro-batches, one padded, run concurrently through the chain.
        mbs = [
            (0, torch.tensor([[1, 5, 9, 33, 7]]), None),
            (1, torch.tensor([[4, 8, 2, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]])),
            (2, torch.tensor([[3, 7, 11, 19, 23]]), None),
        ]
        batches, refs_inputs = [], []
        for j, ids, mask in mbs:
            labels = ids.clone()
            if mask is not None:
                labels[mask == 0] = -100
            batches.append((j, ids, labels, mask))
            refs_inputs.append((ids, mask, labels))

        results = run_micro_batches(stage0, channels, batches)
        scale = results[0][1]
        assert all(sc == scale for _, sc in results)

        full = AutoModelForCausalLM.from_pretrained(root, dtype=torch.float32)
        full.zero_grad()
        ref_loss_sum = 0.0
        for ids, mask, labels in refs_inputs:
            loss = full(input_ids=ids, attention_mask=mask if mask is not None else torch.ones_like(ids),
                        labels=labels).loss
            loss.backward()
            ref_loss_sum += loss.item()
        assert abs(sum(float(l) for l, _ in results) - ref_loss_sum) < TOL

        checked = 0
        for stage, r in zip(stages, ranges):
            for p_name, p in stage.named_params():
                if p_name.startswith("layers."):
                    idx = int(p_name.split(".")[1])
                    full_name = f"model.layers.{r.start + idx}." + p_name.split(".", 2)[2]
                elif p_name.startswith("lm_head."):
                    full_name = p_name
                else:
                    full_name = "model." + p_name
                ref = dict(full.named_parameters())[full_name].grad
                got = p.grad / scale
                assert torch.allclose(got, ref, atol=TOL, rtol=1e-3), (full_name, (got - ref).abs().max().item())
                checked += 1
        assert checked > 10


class _Part(torch.nn.Module):
    """A slice of a node's layers, shaped like the core module that Stage expects."""

    def __init__(self, layers, rotary_emb, norm=None):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.rotary_emb = rotary_emb
        if norm is not None:
            self.norm = norm


class _HalfRemote:
    """A node that runs its two halves in parallel (HalfPipeline), in its own thread group."""

    def __init__(self, front, back):
        self.outbox = queue.Queue()
        self.pipe = HalfPipeline(front, back, reply=self.outbox.put)

    def channel(self):
        return Channel(send=self.pipe.submit, recv=self.outbox.get)


def test_node_halves_in_parallel_match_full_model():
    from workers.kaggle.chain import HalfPipeline  # noqa: F401  (imported above too)
    with tempfile.TemporaryDirectory() as root:
        ckpt = _checkpoint6(root)
        ranges = split_layers(6, 3)            # [2, 2, 2]: each worker node has two halves of one layer
        cpu = [torch.device("cpu")]
        scaler = DynamicLossScaler(init_scale=1024.0)
        stage0_model = load_stage(ckpt, ranges[0], include_embed=True, include_head=False, devices=cpu, dtype=torch.float32)
        stage0 = Stage(stage0_model.model, None, first=True, last=False,
                       params=[p for p in stage0_model.parameters() if p.requires_grad and p.device.type != "meta"],
                       lr=1e-3)
        checked_parts, remotes = [], []
        for s in (1, 2):
            last = s == 2
            m = load_stage(ckpt, ranges[s], include_embed=False, include_head=last, devices=cpu, dtype=torch.float32)
            layers = list(m.model.layers)
            front_part = _Part(layers[:1], m.model.rotary_emb)
            back_part = _Part(layers[1:], m.model.rotary_emb, m.model.norm if last else None)
            front = Stage(front_part, None, first=False, last=False,
                          params=[p for p in front_part.parameters() if p.requires_grad and p.device.type != "meta"], lr=1e-3)
            back = Stage(back_part, m.lm_head if last else None, first=False, last=last,
                         params=[p for p in list(back_part.parameters()) + (list(m.lm_head.parameters()) if last else [])
                                 if p.requires_grad and p.device.type != "meta"],
                         lr=1e-3, scaler=scaler if last else None)
            checked_parts.append((front_part, ranges[s].start, back_part, last, m.lm_head))
            remotes.append(_HalfRemote(front, back))
        channels = {k + 1: r.channel() for k, r in enumerate(remotes)}

        mbs = [
            (0, torch.tensor([[1, 5, 9, 33, 7]]), None),
            (1, torch.tensor([[4, 8, 2, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]])),
            (2, torch.tensor([[3, 7, 11, 19, 23]]), None),
        ]
        batches, refs_inputs = [], []
        for j, ids, mask in mbs:
            labels = ids.clone()
            if mask is not None:
                labels[mask == 0] = -100
            batches.append((j, ids, labels, mask))
            refs_inputs.append((ids, mask, labels))
        results = run_micro_batches(stage0, channels, batches)
        scale = results[0][1]

        full = AutoModelForCausalLM.from_pretrained(root, dtype=torch.float32)
        full.zero_grad()
        for ids, mask, labels in refs_inputs:
            loss = full(input_ids=ids, attention_mask=mask if mask is not None else torch.ones_like(ids),
                        labels=labels).loss
            loss.backward()
        full_grads = dict(full.named_parameters())

        checked = 0
        for front_part, base, back_part, last, head in checked_parts:
            named = []
            for name, p in front_part.named_parameters():
                named.append((f"model.layers.{base + int(name.split('.')[1])}." + name.split('.', 2)[2], p))
            for name, p in back_part.named_parameters():
                if name.startswith("layers."):
                    named.append((f"model.layers.{base + 1 + int(name.split('.')[1])}." + name.split('.', 2)[2], p))
                else:
                    named.append(("model." + name, p))
            if last:
                named += [("lm_head." + n, p) for n, p in head.named_parameters()]
            for full_name, p in named:
                if not p.requires_grad or p.device.type == "meta" or p.grad is None:
                    continue
                ref = full_grads[full_name].grad
                got = p.grad / scale
                assert torch.allclose(got, ref, atol=TOL, rtol=1e-3), (full_name, (got - ref).abs().max().item())
                checked += 1
        assert checked > 10, checked


if __name__ == "__main__":
    results = []
    for name, fn in [("concurrent_micro_batches_match_full_model", test_concurrent_micro_batches_match_full_model),
                     ("node_halves_in_parallel_match_full_model", test_node_halves_in_parallel_match_full_model)]:
        try:
            fn()
            print(f"ok   {name}")
            results.append(True)
        except Exception as e:
            print(f"FAIL {name}: {e!r}")
            results.append(False)
    sys.exit(0 if all(results) else 1)
