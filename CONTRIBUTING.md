# Contributing to Compute Pool

Thanks for considering it. This project is young and most of it has only ever been run on a two- or three-node Kaggle cluster, so real-world testing from more setups is some of the most valuable help it can get.

## Before anything else

Read the platform-risk note at the top of the [README](README.md). Kaggle's policy is one account per person; this tool pools several. Any change you make, and any cluster you test it on, carries that risk. Don't test against accounts you aren't willing to risk.

## Setting up

- **Rust**: stable toolchain, `cargo build --workspace`.
- **Python**: a local virtual environment, never a global install. `uv venv && uv pip install torch transformers peft accelerate datasets safetensors huggingface_hub` (CPU-only torch is enough for everything except an actual training run against the real model).

## Running the checks before you open a PR

```bash
cargo build --workspace
cargo test --workspace
pytest tests/ -v          # or run each tests/test_*.py directly
```

Both must pass. CI runs the same two commands on every push and PR.

## What "done" means here

This project's whole history is built on one rule: **a claim about the cluster needs cluster evidence, not just code that looks right.** Local tests catch a lot (several real bugs in this codebase were caught by a local test before they ever reached a Kaggle node), but local tests use a tiny model on CPU; the real target is an 8B model across real, flaky, bandwidth-limited links. A change that touches `examples/` or `workers/kaggle/` should say plainly in its PR description whether it's been run on a real cluster, and if not, exactly what was checked instead (compiled, unit-tested against a tiny model, traced by hand) and what still needs a cluster run to confirm. Don't claim a fix works because the code looks right; several "obviously correct" changes in this project's history turned out not to be (see the closed issues for examples: PEFT breaking when an unused embedding table is dropped, for one).

If you can't test on a real cluster yourself, that's fine: say so, and open a PR with the honest caveat. Someone with cluster access can confirm it.

## Reporting a bug

Open an issue with:
- The exact command you ran and its full output (or the relevant tail of it).
- Whether it's reproducible, and on what (local, single Kaggle node, a cluster of how many nodes).
- If it's a training run, the job ID and `CHECKPOINT_URI`/log lines around the failure, not just the final traceback.

## Proposing a change

- Small fixes: just open a PR.
- Anything that changes behavior people rely on (a command's output, a wire-protocol message, a checkpoint's format): open an issue first, so the approach gets agreed before the work goes in.
- Match the existing style: comments explain *why* a line exists, not what it does; avoid adding a dependency for something a few lines of stdlib already covers; keep the shortest diff that's still correct, not the most defensive one.
- No duplicate commands. If an existing command already does something, extend it or add a flag rather than adding a second name for the same action (`cp-dispatch` was removed for exactly this reason: it was a literal one-line alias for `crun --gpus`).

## Branch protection

`.github/ruleset.json` is the repository ruleset for the default branch: pull requests required, CI must pass, no force pushes, no deletion, linear history. Import it under the repo's Settings, Rules, Rulesets, New ruleset, Import a ruleset. The repository owner can still push directly (the bypass rule), so this only changes what's required of everyone else.

## Using an AI coding assistant on this repo

See [AGENTS.md](AGENTS.md) first. It has the specific gotchas this codebase has already hit once (a few of them twice), so an agent doesn't rediscover them the slow way.

## Code map

- `crates/compute-pool-core/` -- the CLI's logic: account credentials and settings (`auth.rs`), Kaggle API client (`kaggle.rs`), job scheduling (`scheduler.rs`, `job.rs`), the bootstrap scripts the CLI pushes to Kaggle kernels (`shell.rs`).
- `crates/compute-pool-cli/` -- the `compute-pool` binary: argument parsing and the interactive terminal.
- `workers/kaggle/` -- the Python runtime shared by every distributed-training paradigm: `wire.py` (transport), `checkpoint.py` (resumable state), `strategy.py` (what the fabric can support), `model_shard.py` (load only the layers a stage needs), `pipeline.py` (the pipeline-parallel stage abstraction).
- `examples/` -- reference workloads built on the above: `cluster_comm/` (data-parallel all-reduce), `pipeline_lora/` (model-parallel pipeline training).
- `tests/` -- the Python test suite. Every one of them runs without a GPU or a cluster.
