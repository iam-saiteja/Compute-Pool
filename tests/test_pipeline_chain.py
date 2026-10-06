"""Check that a 3-stage pipeline chain gives the same loss and gradients as one full model.

Uses a tiny random Llama, split into three stages with workers/kaggle/model_shard.py,
driven in-process through workers/kaggle/pipeline.py. Includes a padded batch, so the
attention-mask path is checked too. Run with the project's venv:
    python tests/test_pipeline_chain.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, LlamaConfig  # noqa: E402

from workers.kaggle.amp import DynamicLossScaler  # noqa: E402
from workers.kaggle.model_shard import load_stage  # noqa: E402
from workers.kaggle.pipeline import Stage, split_layers  # noqa: E402

N_LAYERS = 4
TOL = 1e-4


def _checkpoint(root):
    cfg = LlamaConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=N_LAYERS,
                      num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=False)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(root, max_shard_size="40KB")
    return root


def _build_chain(ckpt, ranges):
    cpu = [torch.device("cpu")]
    n = len(ranges)
    stages = []
    scaler = DynamicLossScaler(init_scale=1024.0)
    for s, r in enumerate(ranges):
        first, last = s == 0, s == n - 1
        m = load_stage(ckpt, r, include_embed=first, include_head=last, devices=cpu, dtype=torch.float32)
        m.eval()
        params = [p for p in m.parameters() if p.requires_grad and p.device.type != "meta"]
        for p in params:
            p.requires_grad_(True)
        stages.append(Stage(m.model, m.lm_head, first=first, last=last, params=params, lr=1e-3,
                            scaler=scaler if last else None))
    return stages, scaler


def test_three_stage_chain_matches_full_model():
    with tempfile.TemporaryDirectory() as root:
        ckpt = _checkpoint(root)
        ranges = split_layers(N_LAYERS, 3)
        assert [len(r) for r in ranges] == [2, 1, 1], ranges

        ids = torch.tensor([[1, 5, 9, 33, 7], [4, 8, 2, 0, 0]])  # second row padded
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        labels = ids.clone()
        labels[mask == 0] = -100

        full = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32)
        full.zero_grad()
        ref_loss = full(input_ids=ids, attention_mask=mask, labels=labels).loss
        ref_loss.backward()

        stages, scaler = _build_chain(ckpt, ranges)
        s0, s1, s2 = stages
        h0 = s0.forward(0, ids, mask)
        h1 = s1.forward(0, h0, mask)
        loss, g2, scale = s2.forward_loss(0, h1, mask, labels)
        g1 = s1.backward(0, g2)
        s0.backward(0, g1)

        assert abs(loss.item() - ref_loss.item()) < TOL, (loss.item(), ref_loss.item())
        assert scale == 1024.0

        checked = 0
        for stage, r in zip(stages, ranges):
            for name, p in stage.named_params():
                if name.startswith("layers."):
                    idx = int(name.split(".")[1])
                    full_name = f"model.layers.{r.start + idx}." + name.split(".", 2)[2]
                elif name.startswith("lm_head."):
                    full_name = name
                else:
                    full_name = "model." + name
                ref = dict(full.named_parameters())[full_name].grad
                got = p.grad / scale  # the chain carries the loss scale; apply() divides it out
                assert torch.allclose(got, ref, atol=TOL, rtol=1e-3), (full_name, (got - ref).abs().max().item())
                checked += 1
        assert checked > 10, f"only {checked} parameters compared"


def test_middle_stage_can_be_wrapped_by_peft():
    """A middle stage has neither the embedding nor the LM head loaded, so both are meta
    modules. peft must still accept it."""
    with tempfile.TemporaryDirectory() as root:
        ckpt = _checkpoint(root)
        m = load_stage(ckpt, range(1, 3), include_embed=False, include_head=False,
                       devices=[torch.device("cpu")], dtype=torch.float32)
        lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        peft_model = get_peft_model(m, lora)
        trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
        assert trainable and all("lora_" in n for n in trainable)


if __name__ == "__main__":
    results = []
    for name, fn in [("three_stage_chain_matches_full_model", test_three_stage_chain_matches_full_model),
                     ("middle_stage_can_be_wrapped_by_peft", test_middle_stage_can_be_wrapped_by_peft)]:
        try:
            fn()
            print(f"ok   {name}")
            results.append(True)
        except Exception as e:
            print(f"FAIL {name}: {e!r}")
            results.append(False)
    sys.exit(0 if all(results) else 1)
