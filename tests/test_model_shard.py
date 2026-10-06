"""Check that a two-stage split of a tiny Llama, loaded with workers/kaggle/model_shard.py,
reproduces the full model's logits, that the unused parts never get materialized,
and that peft can wrap each stage.

Run with the project's venv:
    python tests/test_model_shard.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, LlamaConfig  # noqa: E402

from workers.kaggle.model_shard import load_stage  # noqa: E402


def _tiny_checkpoint(root):
    cfg = LlamaConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained(root, max_shard_size="40KB")  # small shards, so the test exercises several
    return root


def _stage_forward(model, h_or_ids, is_first, is_last):
    """Same forward the pipeline scripts use, applied to one stage."""
    core = model.model
    if is_first:
        h = core.embed_tokens(h_or_ids)
    else:
        h = h_or_ids
    pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    pe = core.rotary_emb(h, pos)
    for layer in core.layers:
        out = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe)
        h = out[0] if isinstance(out, tuple) else out
    if is_last:
        h = model.lm_head(core.norm(h))
    return h


def test_two_stage_split_matches_full_model():
    with tempfile.TemporaryDirectory() as root:
        ckpt = _tiny_checkpoint(root)
        ids = torch.tensor([[1, 5, 9, 33, 7]])
        full = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32).eval()
        with torch.no_grad():
            ref = full(ids).logits

        cpu = [torch.device("cpu")]
        a = load_stage(ckpt, range(0, 2), include_embed=True, include_head=False, devices=cpu, dtype=torch.float32).eval()
        b = load_stage(ckpt, range(2, 4), include_embed=False, include_head=True, devices=cpu, dtype=torch.float32).eval()

        assert len(a.model.layers) == 2 and len(b.model.layers) == 2
        assert a.model.embed_tokens.weight.device.type != "meta"
        assert b.model.embed_tokens.weight.device.type == "meta", "unused embedding must not be materialized"
        assert a.lm_head.weight.device.type == "meta", "unused LM head must not be materialized"
        assert b.lm_head.weight.device.type != "meta"

        with torch.no_grad():
            h = _stage_forward(a, ids, is_first=True, is_last=False)
            logits = _stage_forward(b, h, is_first=False, is_last=True)
        assert torch.allclose(logits, ref, atol=1e-5), f"max diff {(logits - ref).abs().max().item()}"


def test_each_stage_can_be_wrapped_by_peft():
    with tempfile.TemporaryDirectory() as root:
        ckpt = _tiny_checkpoint(root)
        cpu = [torch.device("cpu")]
        lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        for layers, embed, head in [(range(0, 2), True, False), (range(2, 4), False, True)]:
            m = load_stage(ckpt, layers, include_embed=embed, include_head=head, devices=cpu, dtype=torch.float32)
            peft_model = get_peft_model(m, lora)
            trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
            assert trainable, "peft should add trainable adapters to this stage's layers"
            assert all(".layers.2." in n or ".layers.3." in n or ".layers.0." in n or ".layers.1." in n
                       for n in trainable)


if __name__ == "__main__":
    results = []
    for name, fn in [("two_stage_split_matches_full_model", test_two_stage_split_matches_full_model),
                     ("each_stage_can_be_wrapped_by_peft", test_each_stage_can_be_wrapped_by_peft)]:
        try:
            fn()
            print(f"ok   {name}")
            results.append(True)
        except Exception as e:
            print(f"FAIL {name}: {e!r}")
            results.append(False)
    sys.exit(0 if all(results) else 1)
