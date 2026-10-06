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
from workers.kaggle.chain import Channel, run_micro_batches  # noqa: E402
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
            self.outbox.put(reply)

    def channel(self):
        return Channel(send=self.inbox.put, recv=self.outbox.get)


def _checkpoint(root):
    cfg = LlamaConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=N_LAYERS,
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


if __name__ == "__main__":
    try:
        test_concurrent_micro_batches_match_full_model()
        print("ok   concurrent_micro_batches_match_full_model")
        sys.exit(0)
    except Exception as e:
        print(f"FAIL concurrent_micro_batches_match_full_model: {e!r}")
        sys.exit(1)
