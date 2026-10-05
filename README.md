# Compute Pool

Compute Pool launches GPU workers on multiple Kaggle accounts and schedules independent tasks across them.

The MVP deliberately does not merge remote GPUs into one CUDA device. Each Kaggle session keeps its native CUDA runtime and sees only its own local GPUs. The cluster layer assigns independent work to those GPUs, including experiments, inference jobs, dataset tasks, and ordinary scripts.

## MVP architecture

```text
compute-pool CLI
      |
      +-- Kaggle account 1: master worker, CUDA_VISIBLE_DEVICES=0/1
      |
      +-- Kaggle account 2: remote worker, CUDA_VISIBLE_DEVICES=0/1
```

The interactive cluster shell provides `cp-dispatch`, a small dispatcher. It runs four task processes concurrently: two locally and two over an SSH link to `node1`.

### Node-to-node connection

Kaggle sessions accept no inbound connections and have no fixed address, so the two nodes can't reach each other directly. Each node opens a [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/) (outbound-only, no account needed) and bridges a real `sshd` through it with [chisel](https://github.com/jpillora/chisel). The master generates the cluster's SSH keypair itself and keeps the private key on local disk for the lifetime of the session; only the public key is ever sent over the wire (via [ntfy.sh](https://ntfy.sh) rendezvous topics), so nothing sensitive leaves the master. Once peered, `ssh node1` from the master terminal works exactly like SSH to any other host.

## Quick start

```bash
compute-pool login --slot 1
compute-pool login --slot 2
compute-pool shell --slot cluster            # 2 nodes (default)
compute-pool shell --slot cluster --nodes 3  # N nodes, one Kaggle account per node
```

Each node needs its own account slot (`login --slot N`, up to 8). Before launching, the CLI checks every account has enough GPU hours left for the session. If a worker fails to come online, the whole cluster is stopped and the worker's error is reported.

Inside the master terminal, `cluster-status` shows each node's state from the health monitor. A worker that stops answering SSH for three consecutive checks is marked `down` and drops out of `crun`; it returns automatically when it recovers.

Inside the master terminal:

```bash
# Run command across both nodes:
crun nvidia-smi

# Dispatch 4 parallel tasks across all 4 GPUs:
crun --gpus "python train.py --dataset {task_index}"
# (shortcut: cp-dispatch "python train.py --dataset {task_index}")
```

Each task receives `CUDA_VISIBLE_DEVICES`, `CP_TASK_INDEX`, `CP_TASK_COUNT`, and `CP_NODE_INDEX`.

For a sharded job that should survive a worker dropping out or the master restarting, use `pool-map` instead of `crun --gpus`: it retries failed shards, checkpoints progress, and resumes with the same `--job-id`.

```bash
pool-map "python embed.py --shard {task_index}" --shards 100 --job-id embeddings-run1 --output results.json
```

Cross-node `torch.distributed` (gloo or NCCL) collectives do not work on this fabric: the ranks need direct connections to each other, and Kaggle containers don't accept inbound connections. Use the SSH all-reduce in `examples/cluster_comm/cp_wire.py` instead; `allreduce_logreg.py` shows it end to end. Within one node, multi-GPU training with `torch.distributed` is unaffected.

## Commands

```bash
compute-pool accounts status
compute-pool probe --slot 1
compute-pool jobs submit --script train.py
compute-pool jobs list
compute-pool shell-stop
```

`torch.cuda.device_count()` reports the GPUs on the current Kaggle node. That is expected. The scheduler, not CUDA, is responsible for using all four GPUs.

## Scope

This project is a task scheduler and worker launcher. It does not provide transparent cross-machine CUDA memory, distributed model parallelism, or collective GPU training. Those require a compatible native distributed network and are outside the MVP.
