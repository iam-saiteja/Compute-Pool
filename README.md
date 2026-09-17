# Compute Pool

> A provider-compliant distributed compute pool that aggregates voluntarily shared,
> unused Kaggle compute capacity and makes it available to checkpointable ML workloads.

## Quickstart

### 1. Install

`ash
pip install -e .
`

### 2. Authenticate two Kaggle accounts

`ash
compute-pool login --slot 1
compute-pool login --slot 2
`

Each slot stores credentials at `~/.compute-pool/accounts/slot{N}/kaggle.json`.
You can get your Kaggle API key from https://www.kaggle.com/settings → API → Create New Token.

### 3. Check account status

`ash
compute-pool accounts status
`

Output:

`
Slot 1  username: alice   quota: ~30h/wk  kernels-run: 3
Slot 2  username: bob     quota: ~30h/wk  kernels-run: 1
`

### 4. Submit a job

`ash
compute-pool job submit examples/hello_gpu.yaml
`

### 5. Track jobs

`ash
compute-pool job list
compute-pool job status <job-id>
`

## Architecture

`
compute_pool/
├── auth/       — Kaggle credential management (per slot)
├── accounts/   — Multi-account quota tracking
├── scheduler/  — Picks best account for each job
├── jobs/       — Job model + state machine
└── storage/    — Local JSON state store
`

## Design Principles

- **Provider compliance first** — only official Kaggle API, no ToS violations
- **Ephemeral workers** — compute disappears; checkpoint everything
- **Two free tiers** — pool Account A + Account B GPU quota
- **Git history** — every state change is trackable

## License

MIT
