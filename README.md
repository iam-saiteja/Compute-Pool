# Compute Pool

> **Provider-compliant distributed compute orchestration system that pools voluntarily shared, unused Kaggle free-tier GPU capacity across multiple accounts into a single high-throughput compute cluster.**

---

## Features

- **Multi-Account GPU Pooling**: Aggregate 2+ Kaggle accounts into a shared pool (~60h/week total quota, **4x Tesla T4 GPUs**, ~60 GB VRAM).
- **Dual-Node Concurrent Interactive Shells**: Boot independent live terminals on **BOTH accounts simultaneously** with a single command (`compute-pool shell-all --web`).
- **Live Hardware Probing**: Real-time `nvidia-smi` kernel probe detecting GPU count, VRAM, CUDA 13.0, and driver versions.
- **Distributed Multi-Node Training**: Train models in parallel across distinct Kaggle accounts simultaneously (`compute-pool distributed run`).
- **Zero-Install Web Terminals**: Secure HTTPS/WSS browser terminals with root bash, 256-color support, and CUDA tools.
- **Instant Lifecycle & Stop Controls**: Cancel jobs and terminate remote GPU containers in `< 0.5s` to preserve quota (`compute-pool job stop`, `compute-pool shell-stop`).
- **Quota-Aware Intelligent Scheduler**: Automatically assigns workloads to the account slot with the most remaining GPU hours.
- **Provider-Compliant Architecture**: 100% official Kaggle API integration with isolated execution environments.

---

## Architecture

```
Compute Pool Orchestration Architecture
├── compute_pool/
│   ├── auth/           # Dual-account credential isolation & multi-slot auth
│   ├── accounts/       # Live quota tracking via Kaggle API (/api/v1/kernels/quota)
│   ├── probe/          # Real-time nvidia-smi GPU hardware detection & caching
│   ├── scheduler/      # Quota-aware job scheduler & slot assigner
│   ├── jobs/           # Job model, state machine & remote Kaggle GPU runner
│   ├── distributed/    # Multi-node parallel training coordinator
│   ├── shell/          # Dual-node & single-node interactive web terminal bridges
│   ├── storage/        # Local JSON persistent state database
│   └── cli.py          # Full-featured Typer CLI
└── tests/              # Comprehensive test suite (18/18 tests passing)
```

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/iam-saiteja/Compute-Pool.git
cd Compute-Pool

# Install dependencies in editable mode
pip install -e .
```

### 2. Authenticate Kaggle Accounts

Authenticate both accounts (Slot 1 and Slot 2):

```bash
compute-pool login --slot 1
compute-pool login --slot 2
```
*Credentials are securely stored locally at `~/.compute-pool/accounts/slot{N}/kaggle.json`.*

### 3. Check Live Pool Status & Quota

```bash
compute-pool accounts status
```

---

## Interactive Remote GPU Terminals

### 1. Launch Dual Interactive Shells on BOTH Accounts Simultaneously (4x Tesla T4 GPUs):
```bash
# Boot interactive terminals on BOTH accounts and automatically open both tabs in browser:
compute-pool shell-all --web

# Or:
compute-pool shell --all --web
```

### 2. Launch on a Specific Single Slot:
```bash
# Launch interactive terminal on Slot 1:
compute-pool shell --slot 1 --web

# Launch interactive terminal on Slot 2:
compute-pool shell --slot 2 --web
```

### 3. Stop / Terminate Active Shells:
```bash
# Stop all active interactive GPU terminals and immediately free GPUs:
compute-pool shell-stop

# Stop only Slot 1:
compute-pool shell-stop --slot 1

# Stop only Slot 2:
compute-pool shell-stop --slot 2
```

---

## Distributed Multi-Node Training

Train across both Kaggle accounts simultaneously in parallel:

```bash
compute-pool distributed run examples/distributed_training.yaml
```

Output benchmark:
```text
Compute Pool -- Distributed Cluster Job (job-24d5ee6b)
  Nodes       : 2 concurrent Kaggle GPU workers (4x Tesla T4 GPUs)
  Node 0      : Slot 1 (saitejathanniru) - 2x Tesla T4
  Node 1      : Slot 2 (thannirusahithya01) - 2x Tesla T4

* Combined Cluster Throughput: ~89,290 samples/sec across 4x Tesla T4 GPUs!
```

---

## Complete CLI Reference

| Command | Description |
|---|---|
| `compute-pool login --slot {1\|2}` | Authenticate Kaggle account credentials for a slot |
| `compute-pool accounts status` | Show live status, quota, and cached GPU specs |
| `compute-pool accounts probe --slot N` | Push live `nvidia-smi` kernel and extract hardware specs |
| `compute-pool accounts stop [--slot N]` | Terminate all active jobs/sessions and idle account(s) |
| `compute-pool shell-all [--web]` | **Boot dual interactive terminals on BOTH accounts (4x T4 GPUs)** |
| `compute-pool shell [--slot 1\|2] [--all] [--web]` | Boot interactive remote GPU terminal(s) |
| `compute-pool shell-stop [--slot N]` | Terminate running shell terminal(s) and release GPU |
| `compute-pool distributed run <spec.yaml>` | Run distributed multi-node parallel training |
| `compute-pool job submit <spec.yaml>` | Submit and schedule a job |
| `compute-pool job run <spec.yaml>` | Submit, schedule, and run immediately |
| `compute-pool job list` | List all historical and active jobs |
| `compute-pool job status <job-id>` | Inspect detailed state and metadata of a job |
| `compute-pool job stop <job-id>` | Cancel a running remote GPU job |
| `compute-pool job stop --all` | Cancel all active jobs |

---

## Testing

Run the full automated test suite:

```bash
pytest
```
*18/18 tests passing across jobs, scheduler, single shell, and dual-cluster shell modules.*

---

## License

MIT
