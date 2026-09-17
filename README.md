# Compute Pool

> **Provider-compliant distributed compute orchestration system that pools voluntarily shared, unused Kaggle free-tier GPU capacity across multiple accounts into a single high-throughput compute cluster.**

---

## Features

- **Multi-Account GPU Pooling**: Aggregate 2+ Kaggle accounts into a shared pool (~60h/week total quota, **4x Tesla T4 GPUs**, ~60 GB VRAM).
- **Live Hardware Probing**: Real-time `nvidia-smi` kernel probe detecting GPU count, VRAM, CUDA 13.0, and driver versions.
- **Distributed Multi-Node Training**: Train models in parallel across distinct Kaggle accounts simultaneously (`compute-pool distributed run`).
- **Interactive Remote GPU Terminal**: Zero-install browser terminal with root bash, 256-color support, CUDA tools, and shell aliases (`compute-pool shell`).
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
│   ├── shell/          # Interactive web terminal bridge (ttyd + Cloudflare tunnel)
│   ├── storage/        # Local JSON persistent state database
│   └── cli.py          # Full-featured Typer CLI
└── tests/              # Comprehensive test suite (17/17 tests passing)
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

Output:
```text
Compute Pool -- Account Status

+-------------------------------------------+
| saitejathanniru   (slot 1)                |
|   Status      : * Connected               |
|   GPU HW      : Tesla T4 x2  15.0 GB VRAM |
|   GPU-h left  : 29.43h / 30.0h            |
+-------------------------------------------+
+-------------------------------------------+
| thannirusahithya01   (slot 2)             |
|   Status      : * Connected               |
|   GPU HW      : Tesla T4 x2  15.0 GB VRAM |
|   GPU-h left  : 29.68h / 30.0h            |
+-------------------------------------------+

  Total pooled quota : 59.11h GPU-hours available
```

### 4. Probe Remote GPU Hardware

```bash
compute-pool accounts probe --slot 1
compute-pool accounts probe --slot 2
```

---

## Interactive Remote GPU Terminal

Boot an interactive root bash terminal inside a live Tesla T4 GPU container:

```bash
# Launch interactive terminal on Slot 1
compute-pool shell --slot 1

# Launch and automatically open in your default browser
compute-pool shell --slot 1 --web

# Launch on Slot 2 with custom duration (e.g. 60 minutes)
compute-pool shell --slot 2 --duration 60 --web
```

### Stopping Terminal Sessions:
```bash
# Stop active terminal on Slot 1 and immediately release GPU:
compute-pool shell-stop --slot 1

# Stop all active terminal sessions:
compute-pool shell-stop
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

## Single Job Submission & Lifecycle

### Submit and Run a Job
```bash
# Submit a YAML spec to the pool queue:
compute-pool job submit examples/hello_gpu.yaml

# Run immediately:
compute-pool job run examples/hello_gpu.yaml
```

### Monitor Jobs
```bash
# List all jobs:
compute-pool job list

# View detailed status of a single job:
compute-pool job status <job-id>
```

### Cancel & Stop Running Jobs
```bash
# Stop a specific running job and terminate its GPU worker:
compute-pool job stop <job-id>
# Or:
compute-pool job cancel <job-id>

# Cancel all running/queued jobs:
compute-pool job stop --all

# Emergency account-wide stop & reset:
compute-pool accounts stop
```

---

## Complete CLI Reference

| Command | Description |
|---|---|
| `compute-pool login --slot {1\|2}` | Authenticate Kaggle account credentials for a slot |
| `compute-pool accounts status` | Show live status, quota, and cached GPU specs |
| `compute-pool accounts probe --slot N` | Push live `nvidia-smi` kernel and extract hardware specs |
| `compute-pool accounts stop [--slot N]` | Terminate all active jobs/sessions and idle account(s) |
| `compute-pool shell [--slot 1\|2] [--web]` | Boot interactive remote GPU terminal |
| `compute-pool shell-stop [--slot N]` | Terminate running shell terminal and release GPU |
| `compute-pool job submit <spec.yaml>` | Submit and schedule a job |
| `compute-pool job run <spec.yaml>` | Submit, schedule, and run immediately |
| `compute-pool job list` | List all historical and active jobs |
| `compute-pool job status <job-id>` | Inspect detailed state and metadata of a job |
| `compute-pool job stop <job-id>` | Cancel a running remote GPU job |
| `compute-pool job stop --all` | Cancel all active jobs |
| `compute-pool distributed run <spec.yaml>` | Run distributed multi-node parallel training |

---

## Testing

Run the full automated test suite:

```bash
pytest
```
*17/17 tests passing across jobs, scheduler, and interactive shell modules.*

---

## License

MIT
