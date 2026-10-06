# Compute Pool

Compute Pool turns several Kaggle accounts into one cluster you can launch, dispatch work to, and tear down from a single CLI. It does not provide GPUs of its own — it pools GPU time you already have access to (Kaggle's free weekly quota per account) and makes using several accounts together as simple as using one.

The MVP deliberately does not merge remote GPUs into one CUDA device. Each Kaggle session keeps its native CUDA runtime and sees only its own local GPUs. The cluster layer assigns work to those GPUs across accounts, and the paradigm that work follows — independent tasks, data-parallel training, or a model split across nodes — is a choice your own code makes, not something the platform hardcodes. See [Ways to use this infrastructure](#ways-to-use-this-infrastructure) below.

> **Platform risk:** Kaggle's policy is one account per person, enforced (phone-number verification, account bans). Pooling GPU quota across multiple accounts you control -- this project's core mechanism -- is the pattern that policy prohibits, not an edge case of it. Every account in a pool is individually at risk of a ban; see [#12](https://github.com/iam-saiteja/Compute-Pool/issues/12) before using this beyond your own private experimentation.

## Architecture

```text
compute-pool CLI (your machine)
      |
      +-- Kaggle account 1: node0 / master -- runs the cluster shell, SSH keypair, registry
      |
      +-- Kaggle account 2: node1 -- peers in over SSH, runs crun/pool-map/your own scripts
      |
      +-- Kaggle account N: node2.. -- same, one Kaggle account per node (up to 8)
```

### Node-to-node connection

Kaggle sessions accept no inbound connections and have no fixed address, so nodes can't reach each other directly. Each node opens a [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/) (outbound-only, no account needed) and bridges a real `sshd` through it with [chisel](https://github.com/jpillora/chisel). The master generates the cluster's SSH keypair itself and keeps the private key on local disk for the lifetime of the session; only the public key is ever sent over the wire (via [ntfy.sh](https://ntfy.sh) rendezvous topics), so nothing sensitive leaves the master. Once peered, `ssh node1` (or `node2`, ...) from the master terminal works exactly like SSH to any other host.

A health loop pings every node every 30 seconds; a node that misses 3 checks in a row is marked `down` and drops out of dispatch, and returns automatically once it answers again.

### Web terminal and file manager password

The web terminal (`ttyd`) and file manager (`filebrowser`) in your browser are protected by a password you set, with the username `compute-pool`.

- **First launch:** `compute-pool shell` asks you to set one if none exists yet. There is no default password.
- **Change it any time:** `compute-pool pwd`. The new password applies to the **next** `compute-pool shell`. A session that is already running keeps the password it started with, so restart it to pick up the change.
- **Where it lives:** `~/.compute_pool/credentials.json`, next to your account keys. It is written into each cluster's bootstrap script, and that kernel is pushed to Kaggle as **private**, so only your account can read it.
- **Allowed characters:** 4 to 64 letters, digits, and `- _ . ! @ # % + =`. Quotes, backslashes and spaces are refused, because the password becomes a string inside the generated script.

### Account policy (asked once)

The first `compute-pool login` or `compute-pool shell` asks you to accept the policy risk, then records the answer in the same credentials file. Declining stores nothing and runs nothing.

What the answer means: Kaggle's policy, as stated in its rules and community posts, is one account per person. Compute Pool pools several accounts, so accounts used this way can be banned. The full research is in [#12](https://github.com/iam-saiteja/Compute-Pool/issues/12). The full terms were not readable by the tool that researched them, so check the current terms yourself.

### Security

What is protected:

- **Node-to-node SSH** is encrypted and key-authenticated. The master generates the keypair and keeps the private key on its own disk; only the public half goes over the wire.
- **Host keys are pinned.** Each worker publishes its SSH host key, and the master writes it to `known_hosts` before connecting. An unknown or changed key is refused. Before this, host checking was off, so someone in the middle of the tunnel could have impersonated a worker.
- **Web terminal and file manager** require the password above. Their URLs are public Cloudflare tunnels, so the password is the only thing between the internet and a shell on your node.
- **Session topics** on ntfy.sh use a 12-character random suffix from a cryptographically secure generator, so they cannot be guessed. Anyone who learns a topic name can read that session's URLs, which is why the password matters.

What is not protected, and what to know:

- The rendezvous over [ntfy.sh](https://ntfy.sh) is public. The secrecy of a session rests on its random topic name.
- Kaggle sees everything that runs on its machines. Do not put secrets you cannot share with Kaggle into a job.
- Pooling accounts can get them banned (see above).

### Performance and latency

Measured on the Kaggle-over-SSH link (see `examples/cluster_comm/link_bench.py`): about 16 MB/s effective bandwidth and about 70 ms round-trip time. A pipeline step moves activations and gradients for every example across that link, so the link, not the GPU, is usually the bottleneck. Practical consequences:

- Work that exchanges data rarely (independent tasks, `pool-map`, periodic all-reduce) runs well over this link.
- Work that exchanges data every micro-batch (pipeline parallelism) is limited by bytes moved. Batches are length-bucketed so padding is not sent (about 7% padding instead of about 42% for random batches at batch size 4).
- Tight collectives that synchronize every step on large tensors do not fit this link.

The cluster launch checks every account's GPU quota in parallel, so a larger cluster does not wait for each check in turn.

## Help, troubleshooting, and asking an AI for help

- Run `compute-pool --help`, or `compute-pool <command> --help`, for the options of each command.
- For a problem with a workload, the **examples are the reference**. Start with the closest one: `examples/cluster_comm/allreduce_logreg.py` for data-parallel, `examples/pipeline_lora/` for model-parallel, `crun`/`pool-map` for independent tasks.
- Logs from a run show which side failed. A `[master] worker connection lost` line means the worker died; rerun the same command with the same job id to resume.

If you ask an AI assistant for help, paste this prompt first so it works from the same facts:

```text
You are helping with Compute Pool, a CLI that pools several Kaggle accounts' GPUs into a
cluster. Facts that matter:
- Kaggle's policy is one account per person; pooling accounts can get them banned. Do not
  suggest tactics that hide multiple accounts.
- Nodes connect over SSH through Cloudflare tunnels; host keys are pinned by the master.
- The link is about 16 MB/s and 70 ms RTT. Pipeline parallelism is bandwidth-bound.
- Check the examples before writing new code: examples/cluster_comm/allreduce_logreg.py
  (data-parallel), examples/pipeline_lora/ (model-parallel with checkpoints),
  workers/kaggle/ (wire transport, checkpoints, pool-map).
- Do not claim torch.distributed NCCL or gloo works across nodes; it does not on this fabric.
Ask me for the exact command and the full error before diagnosing. Then suggest the smallest change.
```

## What this can do

- **Data parallelism:** yes. Each node computes gradients on its share of the data and they are summed with an all-reduce over SSH. `examples/cluster_comm/allreduce_logreg.py` checks the result against a single-node run.
- **Model parallelism (pipeline):** yes. A model is split across two nodes and micro-batches flow through both. `examples/pipeline_lora/` trains an 8B model's LoRA adapters this way, with checkpoint and resume.
- **Several GPUs at the same time:** yes across nodes, since each node runs its own stage. Within a node, the layers are split across its GPUs, but the two GPUs run one after the other for a given micro-batch. Running them in parallel is not done yet (issue #14).
- **Not supported:** tight collectives across nodes on every step (`torch.distributed` gloo/NCCL), and tensor parallelism across nodes. Both need direct node-to-node connections, which Kaggle does not allow.

## How many accounts

The code allows up to **8 account slots** (`MAX_ACCOUNT_SLOTS` in `auth.rs`). Each node uses its own slot, so a cluster is at most 8 nodes. Two things set how useful that is:

- **Independent work** (`crun`, `pool-map`) scales with the number of nodes. The SSH ports and the registry grow with the cluster, and this is the path that scales best.
- **Pipeline work** is two stages in the example. A three-stage pipeline is not implemented, so extra nodes do not speed up a single pipeline.

Eight is a limit in the code, not a tested maximum: only two-node clusters have been run end to end. Three-node failure and recovery is open in [#10](https://github.com/iam-saiteja/Compute-Pool/issues/10).

## Quick start

```bash
compute-pool login --slot 1
compute-pool login --slot 2
compute-pool cluster                         # which accounts the cluster uses (default: all configured)
compute-pool cluster use 1 3                 # use only slots 1 and 3, e.g. to leave slot 2 out
compute-pool shell --slot cluster            # launch the cluster from the chosen accounts
```

Each node uses one configured account. `compute-pool cluster` chooses which accounts, and by default every configured account is used, in slot order. The first chosen account is the master. A cluster needs at least two accounts. Before launching, the CLI checks every account has enough GPU hours left for the session; if any account doesn't, or any worker fails to come online, the whole cluster is stopped and the failing account is reported — you never end up with a partially-billed cluster you didn't ask for.

Inside the master terminal, `cluster-status` shows every node's state:

```bash
$ cluster-status
+----------------------------------------------------------------------+
|                 Compute Pool GPU Cluster Status                      |
+----------------------------------------------------------------------+
[*] node0 (Master): online, 2 GPUs
[*] node1 (Worker): online, 2 GPUs
[*] Online GPUs in cluster: 4
+----------------------------------------------------------------------+
Cluster Tools & Commands:
  • nvidia-smi              -> Local GPU telemetry (master)
  • ssh node1               -> Shell on worker node1 (node<N> for others)
  • crun <command>          -> Run command on every online node
  • crun --gpus '<command>' -> Run one task per online GPU
  • pool-map '<cmd>' --shards N -> Checkpointed shard dispatch (retries, resumable)
  • stop                    -> Terminate cluster session
+----------------------------------------------------------------------+
```

## Ways to use this infrastructure

Compute Pool doesn't lock you into one training paradigm. Pick the one that matches your workload's communication pattern, bring your own training/inference script, and the platform's job is to get it running correctly across every GPU in the cluster.

### 1. Independent tasks (sweeps, batch inference, embeddings) — `crun` / `pool-map`

The simplest and most reliable pattern: no cross-node communication at all, each GPU runs its own shard of the work. Your script doesn't need to know anything about Compute Pool.

```bash
# Run one command on every online node (not per-GPU):
crun nvidia-smi

# One task per online GPU across the whole cluster -- a hyperparameter sweep:
crun --gpus "python train.py --lr {task_index}e-4 --dataset {task_index}"
```

Each task gets `CUDA_VISIBLE_DEVICES`, `CP_TASK_INDEX`, `CP_TASK_COUNT`, and `CP_NODE_INDEX` as environment variables. `crun` syncs any local file your command references to every node first, so you don't need to `scp` your script by hand.

For anything long enough that a worker could disappear mid-run (a Kaggle session ending, an account's quota running out), use `pool-map` instead: it shards a job across every online GPU, **retries a failed shard instead of failing the whole job**, and checkpoints progress so the whole run resumes with the same `--job-id` instead of restarting from zero.

```bash
pool-map "python embed.py --shard {task_index}" --shards 100 --job-id embeddings-run1 --output results.json
# Rerun with the same --job-id any time -- already-completed shards are skipped.
```

This is the paradigm to reach for first. It has no cross-node collective to get wrong, tolerates a node dropping out, and is proven end to end on a live 2-node cluster.

### 2. Data-parallel training with periodic sync — SSH all-reduce

For a model small enough to fit on one GPU (or LoRA adapters on top of a frozen base model), train a replica per node and combine gradients with an all-reduce over the SSH link. `torch.distributed`'s own collectives (gloo/NCCL) don't work on this fabric -- see the limitation below -- so this uses a purpose-built all-reduce over the same wire transport everything else uses.

```bash
python3 examples/cluster_comm/allreduce_logreg.py
```

That example is a minimal, numerically-checked reference (distributed vs. single-node results are compared directly): two nodes each hold half a dataset, compute a local gradient, all-reduce it every iteration, and the result matches a single-node run to floating-point precision. Build your own training loop the same way using `workers/kaggle/wire.py`'s `allreduce_sum_master` / `allreduce_sum_worker`.

### 3. Model-parallel / pipeline training — split a model too large for one GPU across nodes

For a model too large for a single node's GPU memory, split it in half (or more) across nodes and pipeline micro-batches through it: `examples/pipeline_lora/` fine-tunes an 8B-parameter Llama model's LoRA adapters this way, split across two nodes, and is verified end to end on a live cluster.

```bash
# On the master, inside the cluster shell:
python3 examples/pipeline_lora/pipeline_master.py
```

What it demonstrates, all reusable in your own pipeline-parallel script:

- **Checkpoint/resume** (`workers/kaggle/checkpoint.py`): both stages save adapter state periodically and resume automatically; killing a node mid-run costs nothing but the steps since the last checkpoint.
- **Coordinated checkpointing across stages**: the master drives each save, so the two halves of the model can't end up checkpointed at different steps after a crash.
- **Dynamic fp16 loss scaling** (`workers/kaggle/amp.py`) owned by whichever side applies it at the loss, since that scale factor is baked into every gradient downstream of it.
- **Held-out evaluation and EMA-smoothed training loss**, so you get a real learning-progress signal instead of noisy per-step loss.
- **Real multi-example batching** with hand-built padding + attention masks, not batch-size-1 micro-batches.
- **Overlapped transfer and compute** via background reader/writer threads, so the link isn't blocking the GPU any more than the fabric's actual bandwidth/latency requires.

Configure a run with environment variables (all optional, shown with defaults):

```bash
STEPS=200 MICROBATCHES=2 MICRO_BATCH_SIZE=4 CKPT_EVERY=25 EVAL_EVERY=25 \
CHECKPOINT_JOB_ID=my-run CHECKPOINT_URI=local:///kaggle/working/checkpoints \
python3 examples/pipeline_lora/pipeline_master.py
```

Rerun the same command with the same `CHECKPOINT_JOB_ID` any time to resume. `MICROBATCHES` × `MICRO_BATCH_SIZE` is the effective batch size per optimizer step; raising either increases activation memory, so watch `nvidia-smi` and lower them if you see an OOM.

### 4. Inference serving

The same `pool-map`/`crun --gpus` dispatch works for batch inference (shard a dataset of prompts across every GPU, collect results) today. A long-lived, low-latency inference *server* across the cluster — rather than a batch job — isn't built yet; `crun --gpus` can still launch one server process per GPU, you'd just handle request routing yourself.

### Picking a paradigm for your own workload

`workers/kaggle/strategy.py` is the contract: it describes the fabric's measured, actual capabilities (bandwidth, round-trip latency, no direct peer-to-peer, no inbound connections) and each paradigm's communication profile, and tells you whether a given paradigm is feasible on this fabric *before* you build on it -- so a workload that needs more than the fabric can give fails with a clear reason instead of silently starving on the link.

```python
from workers.kaggle import strategy

strategy.feasible_strategies(strategy.KAGGLE_SSH_FABRIC)
# {'independent': ..., 'intra_node_ddp': ..., 'data_parallel_sync': ..., 'pipeline': ...}
# ('sync_collective' -- tight full all-reduce every step -- is NOT in this
# list: it needs direct node-to-node connections, which this fabric has none of.)
```

## Building your own workload on this infrastructure

You don't need to modify Compute Pool to run your own training or inference code on it:

- **No cross-node communication needed?** Just use `crun --gpus`/`pool-map` with your existing script, unmodified. This is almost always the right starting point.
- **Need two nodes to exchange state?** Use `workers/kaggle/wire.py` directly: `spawn()` launches your worker script over SSH, `send`/`recv` move any picklable object (including tensors) across the link with a background reader/writer thread so neither side blocks on the other unnecessarily, and `allreduce_sum_master`/`allreduce_sum_worker` give you a ready-made collective. `examples/cluster_comm/allreduce_logreg.py` and `examples/pipeline_lora/` are both built from exactly this module and are the reference to copy from.
- **Need your job to survive a worker dying?** Use `workers/kaggle/checkpoint.py`'s `open_store("local://..." or "s3://...", job_id)` — atomic save/resume, the same primitive every example above uses.
- **Not sure your communication pattern fits this fabric?** Check it against `workers/kaggle/strategy.py` first (see above) rather than finding out mid-run.

## Commands

```bash
compute-pool login --slot N          # store a Kaggle account's credentials for slot N (asks once to accept the policy)
compute-pool pwd                     # set or change the web terminal / file manager password
compute-pool accounts status         # GPU-hours remaining per configured account
compute-pool probe --slot N          # quick GPU/driver check on one account
compute-pool shell --slot N          # single-node interactive GPU terminal
compute-pool cluster [show|use <slots>|reset]   # choose the accounts a cluster uses
compute-pool shell --slot cluster [--duration MIN]
compute-pool shell-stop [--slot N]   # tear down a session (all, if no slot given)
compute-pool jobs submit --script train.py
compute-pool jobs list
```

`torch.cuda.device_count()` reports only the GPUs on the current Kaggle node — that's expected, each node is its own CUDA runtime. The scheduler and your dispatch choice (`crun`/`pool-map`/your own script), not CUDA, are what use every GPU across the cluster.

## Known limitations

- **No tight synchronous cross-node collectives.** `torch.distributed` (gloo/NCCL) needs direct node-to-node connections; Kaggle containers accept no inbound connections at all, so standard `torch.distributed` process groups across nodes don't work on this fabric regardless of configuration. Use the SSH-based primitives in `workers/kaggle/wire.py` instead (paradigms 2 and 3 above). Multi-GPU training *within* one node via `torch.distributed` is unaffected.
- **Both pipeline stages currently download the full base model** before dropping the half they don't need ([#5](https://github.com/iam-saiteja/Compute-Pool/issues/5)) — saves steady-state memory, not startup time.
- **Examples and clusters of three or more nodes.** `crun` and `pool-map` use every node in the cluster. The two example pipelines and `allreduce_logreg.py` are written for two nodes: they use node0 and node1, and any further node sits idle. Changing that means generalizing their stage or shard layout, which is not done.
- **A third account is needed to test beyond 2 nodes' failure/recovery paths** ([#10](https://github.com/iam-saiteja/Compute-Pool/issues/10)).
- See the [issue tracker](https://github.com/iam-saiteja/Compute-Pool/issues) for the full, current list with reproduction evidence from the live cluster.

## Scope

This is not a managed GPU cloud and does not make several machines look like one CUDA device. It's a CLI that launches a fleet of Kaggle sessions, bridges them into one addressable cluster over SSH, and gives you primitives (checkpointing, a measured-fabric-aware strategy registry, a wire transport) to run independent, data-parallel, or model-parallel workloads across them — choosing which is your code's decision, not a constraint the platform imposes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — setup, what "done" means for a change that touches training code, and how to report a bug with evidence that's actually actionable.
