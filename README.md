# Compute Pool

[![CI](https://github.com/iam-saiteja/Compute-Pool/actions/workflows/ci.yml/badge.svg)](https://github.com/iam-saiteja/Compute-Pool/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ray Cluster](https://img.shields.io/badge/Ray-4--GPU%20Cluster%20Ready-028CF0.svg)](https://ray.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-Distributed%20Ready-EE4C2C.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-4x%20Tesla%20T4%20(60%20GB%20VRAM)-76B900.svg)](https://www.nvidia.com)

> **High-Performance Multi-GPU Infrastructure-as-a-Service (IaaS) & Resource Aggregator**. Pool individual free cloud GPU quotas into a unified, industry-standard 4-GPU compute cluster (**4x Tesla T4 GPUs = 60 GB combined VRAM**) with zero framework lock-in.

---

## ⚡ Pure Infrastructure-as-a-Service (IaaS)

Compute Pool aggregates isolated cloud GPU nodes into a raw, high-throughput compute cluster. You have full root access to build, run, and scale any ML workload using industry-standard tools:

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                   Unified 4-GPU Distributed Cluster Pool                     │
├──────────────────────────────────────┬───────────────────────────────────────┤
│        Node 0 (Master Node)          │         Node 1 (Worker Node)          │
│   • 2x NVIDIA Tesla T4 (30 GB VRAM)  │    • 2x NVIDIA Tesla T4 (30 GB VRAM)  │
│   • 4 vCPUs • 20 GB Disk             │    • 4 vCPUs • 20 GB Disk             │
│   • Root Web Terminal (ttyd)         │    • OpenSSH Server (Port 2222)       │
│   • Node 0 Web File Manager (8080)   │    • Node 1 Web File Manager (8081)   │
├──────────────────────────────────────┴───────────────────────────────────────┤
│  ⚡ Inter-Node Fabric: Bidirectional SSH & TCP Bridge (`ssh node1`, Ed25519) │
│  ⚡ Zero Port Collisions: Default ML ports (6379, 29500) kept clean & free    │
│  ⚡ Opt-In Tooling: Ray Cluster • PyTorch DDP • DeepSpeed • Parallel `crun`  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Highlights & Capabilities

- **4x Tesla T4 GPUs (60 GB Total VRAM)**: Aggregates dual accounts into a unified 4-GPU cluster with 8 vCPUs and ~42 GB RAM.
- **Ray Multi-Node Cluster (`enable-ray`)**: Auto-provisions a unified Ray cluster across all 4 GPUs with auto-scaling, GCS coordination, and zero-copy Plasma object store.
- **DeepSpeed Ready (`enable-deepspeed`)**: Pre-configures passwordless SSH and hostfile (`localhost slots=2`, `node1 slots=2`) for multi-node ZeRO stages.
- **PyTorch DDP (`enable-pytorch`)**: Seamless rendezvous bridge on port `29500` for standard `torchrun` and `torch.distributed`.
- **Parallel Cluster Runner (`crun <cmd>`)**: Automatically syncs code files and executes commands simultaneously across both nodes.
- **Dual Independent Web File Managers**: Dedicated visual FTP web file managers for **Node 0** (`:8080`) and **Node 1** (`:8081`) to inspect, upload, and download artifacts across both disk volumes independently.
- **Native Inter-Node SSH Fabric**: Direct OpenSSH connection with auto-generated Ed25519 keys (`ssh node1`, `scp file.py node1:/kaggle/working/`).
- **Real GPU Hardware Telemetry**: Native, un-mocked `nvidia-smi` output showing exact PCIe bus IDs, temperatures, VRAM, and power draw.
- **Standalone Rust Engine**: Sub-millisecond execution CLI (`compute-pool.exe`) with zero local Python dependencies.
- **Interactive REPL Console**: Built-in interactive command prompt with auto-history, quick actions, and sequential 0-indexed job management.

---

## 📦 Quickstart & Setup

### 1. Build Standalone Binary

```bash
# Clone the repository
git clone https://github.com/iam-saiteja/Compute-Pool.git
cd Compute-Pool

# Build release binary
cargo build --release

# (Optional) Install globally
cargo install --path crates/compute-pool-cli
```

### 2. Authenticate GPU Slots

Authenticate your participating Kaggle account credentials:

```bash
compute-pool login --slot 1
compute-pool login --slot 2
```

### 3. Check Account Quotas

```bash
compute-pool accounts status
```

---

## 🖥️ Launching the Unified 4-GPU Cluster

```bash
# Launch unified 4-GPU cluster and open Master Web Terminal in browser:
compute-pool shell -s cluster --open

# Custom lease duration (in minutes):
compute-pool shell -s cluster -d 180 --open
```

Once launched, Compute Pool outputs the connection endpoints:

```text
Compute Pool -- Unified 4-GPU Cluster
  Cluster Fabric:        2 Nodes (4x Tesla T4 GPUs) Connected & Peered
  Master Web Terminal:   https://your-master-terminal.trycloudflare.com
  Node 0 File Manager:   https://your-node0-files.trycloudflare.com
  Node 1 File Manager:   https://your-node1-files.trycloudflare.com
  Inter-Node SSH:        ssh node1 (from Master terminal)
  Cluster Runner:        crun <command> (e.g. crun nvidia-smi)
  Cluster Helpers:       enable-ray, enable-pytorch, enable-deepspeed
```

---

## 🛠️ In-Terminal Cluster Helpers & Distributed Frameworks

Inside the **Master Web Terminal**, you have access to pre-configured cluster helpers:

### 1. Ray Multi-Node Cluster

Initialize a unified 4-GPU Ray cluster across both nodes:

```bash
enable-ray
```

Verify cluster resources:

```bash
ray status
```

```text
======== Autoscaler status ========
Active:
 1 node_master (Head)
 1 node_worker (Worker)

Resources:
 0.0/8.0 CPU
 0.0/4.0 GPU (60 GB Total VRAM)
 0B/41.63GiB memory
 0B/17.84GiB object_store_memory
```

Use in Python:

```python
import ray, torch

ray.init(address="auto")

@ray.remote(num_gpus=1)
def gpu_task(idx):
    return f"Task {idx} on: {torch.cuda.get_device_name(0)}"

print(ray.get([gpu_task.remote(i) for i in range(4)]))
```

---

### 2. DeepSpeed 4-GPU Training

Configure the multi-node DeepSpeed environment:

```bash
enable-deepspeed
```

Run training with DeepSpeed across all 4 GPUs:

```bash
deepspeed --hostfile /root/.ssh/hostfile train.py --deepspeed_config ds_config.json
```

---

### 3. PyTorch Distributed Data Parallel (DDP / `torchrun`)

Set up PyTorch multi-node rendezvous:

```bash
enable-pytorch
```

Run PyTorch DDP:

```bash
# On Master (Node 0):
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 train.py

# On Worker (Node 1 via ssh node1):
ssh node1 "torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_addr=127.0.0.1 --master_port=29500 train.py"
```

---

### 4. Parallel Command Runner (`crun`)

Execute any bash command or Python script across both nodes in parallel (auto-syncs local code files to Node 1):

```bash
# Check GPU status across both nodes
crun nvidia-smi

# Run training script across both nodes simultaneously
crun python3 /kaggle/working/train.py
```

---

### 5. Direct SSH & File Sync

```bash
# Open interactive shell into Worker (Node 1)
ssh node1

# Copy files directly between nodes
scp dataset.zip node1:/kaggle/working/
```

---

## 🔄 Batch & Sequential Job Management

Submit standalone asynchronous batch jobs to the scheduler:

```bash
# Submit a GPU script for execution
compute-pool jobs submit --script train.py --name resnet-train --gpu

# List jobs in 0-indexed sequential order
compute-pool jobs list

# Inspect logs
compute-pool jobs logs 0

# Stop / Delete jobs
compute-pool jobs stop 0
compute-pool jobs delete 0
```

---

## ⚡ Teardown & Quota Preservation

To immediately terminate all remote containers and preserve cloud GPU quotas:

- Inside Master Terminal: type `halt` or `exit`
- From local CLI: run `compute-pool shell-stop`

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
