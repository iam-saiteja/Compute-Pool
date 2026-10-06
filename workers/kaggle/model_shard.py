"""Load only one stage of a decoder-only model, without ever materializing the rest.

A pipeline stage needs a contiguous range of decoder layers, and sometimes the
embedding table or the LM head. Loading the whole model with from_pretrained
and deleting the unused part afterwards still pays the full download and the
full peak memory. This loads just the tensors the stage needs, reading only the
checkpoint shards that contain them.

Works on a local directory or a Hub repo id. Uses only the stable safetensors
and accelerate primitives, not transformers' loading internals, so it does not
depend on one transformers version.
"""
import json
import os

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"


def _fetch(repo_or_dir, name):
    """Local path for one file, downloading it from the Hub if needed. None if absent."""
    if os.path.isdir(repo_or_dir):
        path = os.path.join(repo_or_dir, name)
        return path if os.path.exists(path) else None
    try:
        return hf_hub_download(repo_or_dir, name)
    except Exception:
        return None


def _stage_key(key, layers, include_embed, include_head):
    """Whether a checkpoint tensor belongs to this stage."""
    if key.startswith("model.layers."):
        return int(key.split(".")[2]) in layers
    if key.startswith("model.embed_tokens."):
        return include_embed
    if key.startswith("lm_head.") or key.startswith("model.norm."):
        return include_head
    return False


def load_stage(repo_or_dir, layers, *, include_embed, include_head, devices, dtype=torch.float16):
    """Build a causal LM in which only the given decoder layers are really loaded.

    layers:   a contiguous range of decoder layer indices, e.g. range(16, 32).
    devices:  torch devices on this node. The embedding goes on the first, the
              head on the last, and the layers are split evenly across all of
              them, so a node with two GPUs keeps a two-GPU split.

    Layers outside `layers` are removed from the returned model, so peft and the
    caller only see this stage's own layers. Indices are not renumbered, so
    checkpoint names still match.
    """
    layers = list(layers)
    if layers != list(range(layers[0], layers[-1] + 1)):
        raise ValueError("layers must be a contiguous range")

    config = AutoConfig.from_pretrained(repo_or_dir)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.config.use_cache = False

    index_path = _fetch(repo_or_dir, INDEX_NAME)
    if index_path:
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        wanted = {k: shard for k, shard in weight_map.items()
                  if _stage_key(k, layers, include_embed, include_head)}
        shard_files = sorted(set(wanted.values()))
    else:
        wanted = None
        shard_files = [SINGLE_NAME]

    n_dev = len(devices)
    layer_dev = {i: devices[pos * n_dev // len(layers)] for pos, i in enumerate(layers)}
    embed_dev, head_dev = devices[0], devices[-1]

    for shard in shard_files:
        with safe_open(_fetch(repo_or_dir, shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if wanted is not None:
                    if key not in wanted:
                        continue
                elif not _stage_key(key, layers, include_embed, include_head):
                    continue
                if key.startswith("model.layers."):
                    dev = layer_dev[int(key.split(".")[2])]
                elif key.startswith("model.embed_tokens."):
                    dev = embed_dev
                else:
                    dev = head_dev
                set_module_tensor_to_device(model, key, dev, value=f.get_tensor(key), dtype=dtype)

    # Rotary buffers are not stored in the checkpoint, so they stay real. Put them on the first device.
    model.model.rotary_emb.to(embed_dev)

    # Keep only this stage's layers. Dropped layers were never materialized, so this frees nothing new.
    model.model.layers = torch.nn.ModuleList(model.model.layers[layers[0]:layers[-1] + 1])
    return model
