"""Compute Pool 4-GPU Distributed Training Example (PyTorch DDP).

This script runs across both Node 0 (Master) and Node 1 (Worker) simultaneously,
coordinating 4x Tesla T4 GPUs (60 GB Total VRAM) in parallel.

Run inside the cluster web terminal with:
    crun python3 /kaggle/working/cluster_4gpu_distributed.py
"""
import os
import sys
import time
import socket
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

MASTER_PORT = os.environ.get("CP_MASTER_PORT", "29500")

def get_node_rank():
    if os.environ.get("NODE_RANK") is not None:
        return int(os.environ["NODE_RANK"])
    # Check if we are on Node 0 or Node 1
    res = subprocess.run("pgrep -f 'chisel server'", shell=True, capture_output=True)
    if res.returncode == 0:
        return 1
    return 0

def init_distributed(node_rank, local_gpu_id):
    # Total 4 GPUs: 2 GPUs on Node 0, 2 GPUs on Node 1
    # Node 0: Global Ranks 0, 1
    # Node 1: Global Ranks 2, 3
    world_size = 4
    global_rank = node_rank * 2 + local_gpu_id

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = MASTER_PORT
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(global_rank)

    # The managed cluster exposes the rendezvous through an SSH tunnel. NCCL
    # needs direct peer-to-peer networking and cannot use this single-port path.
    backend = "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=global_rank
    )
    return global_rank, world_size

def run_worker_process(local_gpu_id, node_rank):
    torch.cuda.set_device(local_gpu_id)
    device = torch.device(f"cuda:{local_gpu_id}")

    global_rank, world_size = init_distributed(node_rank, local_gpu_id)

    gpu_name = torch.cuda.get_device_name(local_gpu_id)
    vram_gb = torch.cuda.get_device_properties(local_gpu_id).total_memory / (1024**3)

    print(
        f"[Node {node_rank} | GPU {local_gpu_id}] -> "
        f"GLOBAL RANK {global_rank}/{world_size}: {gpu_name} ({vram_gb:.1f} GB VRAM) Active!",
        flush=True
    )

    # Synchronize all 4 GPUs
    dist.barrier()

    if global_rank == 0:
        print("\n=======================================================", flush=True)
        print("  ⚡ All 4 GPUs Synchronized Across Both Nodes! ⚡", flush=True)
        print("  Total Cluster VRAM: 4 x 15 GB = 60 GB", flush=True)
        print("=======================================================\n", flush=True)

    # Define a neural network model
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 100),
    ).to(device)

    ddp_model = DDP(model, device_ids=[local_gpu_id] if torch.cuda.is_available() else None)
    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Total batch distributed across all 4 GPUs
    batch_size_per_gpu = 128
    features = torch.randn(batch_size_per_gpu, 1024, device=device)
    labels = torch.randint(0, 100, (batch_size_per_gpu,), device=device)

    # Distributed Training Loop
    t0 = time.time()
    for step in range(1, 6):
        optimizer.zero_grad()
        outputs = ddp_model(features)
        loss = criterion(outputs, labels)
        loss.backward()  # All-reduce gradient exchange across all 4 GPUs!
        optimizer.step()

        # Gather loss across all ranks
        loss_tensor = loss.detach().clone()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

        if global_rank == 0:
            print(f"  Step {step}/5 | Global Loss across 4 GPUs: {loss_tensor.item():.4f}", flush=True)

    dist.barrier()
    elapsed = time.time() - t0

    if global_rank == 0:
        print("\n=======================================================", flush=True)
        print(f"  ✓ 4-GPU Distributed Training Finished in {elapsed:.2f}s!", flush=True)
        print("=======================================================\n", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    node_rank = get_node_rank()
    if node_rank == 0:
        # A killed torch process can leave the reserved rendezvous port occupied.
        # Only clean this explicitly reserved port, before any child is spawned.
        import shutil
        if shutil.which("fuser"):
            subprocess.run(
                ["fuser", "-k", f"{MASTER_PORT}/tcp"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
    import torch.multiprocessing as mp
    # Spawn 2 processes per node (one per local GPU)
    mp.spawn(run_worker_process, args=(node_rank,), nprocs=2, join=True)
