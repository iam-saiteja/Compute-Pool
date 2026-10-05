# Compute Pool Architecture

Written for: contributors working on the platform. Companion to `docs/PRD.md`
(the product requirements); this document records the technical architecture
and the decisions behind it, grounded in what the Kaggle fabric was measured to
actually do.

## 1. The one decision everything follows from

Compute Pool does **not** merge borrowed GPUs into one logical device. It puts a
common scheduler over many independent, ephemeral, heterogeneous resources
(PRD §51). The platform's job is to take a described computation and run it in
whatever execution mode the available fabric can actually support — and to
refuse, clearly, the modes it cannot.

That reframing is what makes "support every kind of distributed workload"
tractable. The paradigms people ask for — independent/map, data-parallel,
pipeline-parallel, sharding, inference serving, tight synchronous training —
differ mainly in **how much and how tightly they must communicate between
workers**. The fabric has a measured ceiling. So the architecture classifies a
workload by its communication profile and routes it to a matching execution
strategy, failing safely when the workload needs more than the fabric can give.

## 2. Measured fabric (the ground truth)

Each node is one Kaggle account running a notebook kernel (2× Tesla T4, 15 GB
each, no bf16), in a container with root but no `/dev/net/tun`, no
`CAP_NET_ADMIN`, outbound internet only, **no inbound connectivity**, and no
fixed address. Nodes reach each other over Cloudflare quick tunnels + chisel +
SSH, with ntfy.sh for rendezvous.

Measured link between two nodes (`examples/cluster_comm/link_bench.py`):

| | value |
|---|---|
| Effective bandwidth (both directions) | ~16 MB/s |
| Round trip, small message | ~70 ms |
| Round trip, 8 MB | ~1.0 s |
| Direct worker-to-worker connections | none (containers refuse inbound) |

Consequences, verified:
- `torch.distributed` gloo/NCCL **cannot** run across nodes: rendezvous works
  through a tunnel, but the backend then dials peers' container IPs and gets
  connection-refused. Encoded as the `sync_collective` strategy being
  infeasible on `KAGGLE_SSH_FABRIC`.
- Pipeline-parallel and periodic-sync data-parallel **do** work; they are
  bandwidth-bound, not impossible.

## 3. Layers

```
                 user: declarative job spec + pool.map
                                │
                        CLI  (Rust)
                                │
                     control plane (Rust)
         ┌──────────────┬───────────────┬──────────────┐
         │              │               │              │
    Job engine      Scheduler        Registry     Provider adapter
   (state, retry)  (strategy pick,  (workers,     (Kaggle now:
                    fail-safe)       health)       launch/stop/logs)
                                │
                        transport (wire)
                                │
                      worker runtime (Python)
         ┌──────────────┬───────────────┬──────────────┐
    execution        checkpoint       heartbeat      the shipped
    strategies       store            / health       job code
```

The Rust control plane already exists in part (`crates/compute-pool-core`,
`crates/compute-pool-cli`): credentials, the node registry, the health loop,
and Kaggle launch via the API. The worker runtime is the new
`workers/kaggle/` package.

## 4. Worker runtime (`workers/kaggle/`)

A flat, install-free package (standard library + torch) copied to each
ephemeral container.

- **`wire.py` — transport.** Length-prefixed torch messages over two byte
  streams (an SSH process's stdin/stdout on one side, the worker's own on the
  other), plus background reader/writer threads that overlap transfer with
  compute, and an SSH-friendly all-reduce. This is the *only* place the runtime
  talks to another node, so a future transport (a relay, Tailscale userspace,
  WebRTC) only has to supply two streams with the same framing. It does not
  assume SSH.
- **`checkpoint.py` — durable state.** `CheckpointStore` writes atomic,
  step-numbered checkpoints to a location outside the worker (`local://` for
  tests and single-node, `s3://` for real runs) and resumes from the latest.
  This is the mechanism behind recovered compute (PRD §35): a replacement
  worker loads the latest checkpoint and continues.
- **`strategy.py` — the paradigm contract.** Each execution strategy declares a
  `CommProfile` (cross-node frequency and bytes, whether it needs direct
  peer-to-peer, whether it tolerates worker loss) and a `feasible(fabric)`
  check. The registry ships `independent`, `intra_node_ddp`,
  `data_parallel_sync`, `pipeline`, and `sync_collective`. The scheduler calls
  `feasible_strategies(fabric)` and fails safely on the rest.

Reference run loops for two strategies already exist and are verified on a live
cluster: `examples/cluster_comm/` (periodic-sync data parallel — logistic
regression, matches single-node to 2e-16) and `examples/pipeline_lora/`
(pipeline-parallel 8B LoRA across two nodes). These are the canonical
implementations the runtime strategies are factored from.

## 5. Scheduling (fail-safe by construction)

Per PRD §16–17, the scheduler is deterministic in V0:

1. Collect available workers from the registry (with detected GPU/VRAM/health).
2. Determine the job's execution strategy — from the job spec, or inferred from
   its communication profile.
3. Check `strategy.feasible(fabric)`. If not feasible, **reject the job with
   the reason** rather than running it in a mode that will hang or silently
   diverge. This is PRD §49: fail safely, don't pretend independent workers are
   one cluster.
4. Filter workers that satisfy the job's resource requirements; rank by
   availability, VRAM, GPU count, utilisation, expected lifetime.
5. Assign; on worker loss, resume from the latest checkpoint on another worker.

## 6. Job spec (what the user writes)

The user describes *what*, the scheduler picks *how* (PRD §13, §20):

```yaml
job:
  name: image-embeddings
  runtime: { framework: pytorch }
  resources: { gpu: true, gpu_memory_gb: 12 }
  execution:
    strategy: independent      # or: auto | data_parallel_sync | pipeline | intra_node_ddp
    checkpointable: true
    max_runtime_hours: 4
  retry: { enabled: true, max_attempts: 5 }
```

`pool.map(fn, items)` is the first-class primitive for the `independent`
strategy and the key V0 workload.

## 7. What maps to what

| Workload the user wants | Strategy | On Kaggle today |
|---|---|---|
| Map / sweeps / batch inference | `independent` | works |
| Multi-GPU in one worker (DDP/FSDP, local NCCL) | `intra_node_ddp` | works |
| Data-parallel small model / LoRA, periodic sync | `data_parallel_sync` | works (SSH all-reduce) |
| Model too big for one node | `pipeline` | works, bandwidth-bound |
| Tight synchronous all-reduce on big models | `sync_collective` | rejected (fail-safe) |

## 8. Roadmap (phases, from PRD §43)

1. **Foundation (this change):** worker runtime, checkpoint store, strategy
   registry, transport abstraction — all locally tested.
2. **Wire it into the control plane:** launch a chosen strategy from the Rust
   bootstrap; ship `workers/kaggle/` to every node; resume from checkpoint
   after worker loss.
3. **Map workloads end to end:** `pool.map`, shard assignment, retry of failed
   shards, result collection — the V0 success scenario (PRD §48).
4. **Security** (issue #1): authenticate the web terminal and file manager;
   cryptographic rendezvous topic names; workload isolation. Deferred by
   project decision, tracked, required before inviting third-party contributors.
5. **More providers** behind the provider adapter (Colab, …).

## 9. Known limits (tracked as GitHub issues)

- Cross-node tight collectives are physically out of reach on this fabric (#2,
  resolved by documenting and the `sync_collective` fail-safe).
- Pipeline throughput is capped by link bandwidth, not GPU (#14).
- Pipeline resume across the session cap, both-nodes-load-full-model, and
  worker-failure recovery are not yet built (#4, #5, #8).
- Health monitor and 3+ node clusters are unverified on a live run (#10).
- Web terminal and file manager are unauthenticated (#1).
