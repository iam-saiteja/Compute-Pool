# AGENTS.md

Instructions for AI coding agents (Claude, Copilot, Codex, or any other) working in this repo. AI assistance built most of this codebase and is welcome here; this file exists so the next agent doesn't have to relearn the same lessons the hard way.

## The one rule that matters most

**Never claim a change works on the cluster unless it was actually run on the cluster.** This codebase's whole history is a record of changes that looked correct and weren't: dropping an unused embedding table broke PEFT's internal tied-weights check in a way no static read would catch; a worker's SSH server surviving a `pkill` was a plausible-sounding assumption that turned out false; two independently-timed checkpoint saves looked fine until a real mid-run crash proved they could diverge. All of these were caught because the person running this agent actually executed the command on a live Kaggle cluster and pasted the real output back.

If you can't run something on a real cluster yourself:
- Say exactly what you *did* verify (compiled, unit-tested against a tiny model, traced by hand) and what you didn't.
- Give the exact command for the user to run, and wait for their output before claiming success.
- Don't mark an issue closed, don't say "this fixes it," and don't write a commit message that implies cluster verification you don't have.

## Setup

```bash
# Rust
cargo build --workspace
cargo test --workspace

# Python -- a local venv only, never a global install
uv venv
uv pip install torch transformers peft accelerate datasets safetensors huggingface_hub
pytest tests/ -v   # or run each tests/test_*.py directly; every one works without a GPU
```

CI (`.github/workflows/ci.yml`) runs the same two test commands. Run them before every commit; several real bugs here (a Windows-specific `subprocess` quirk, a concurrency bug in shard retry, an off-by-one in a `sys.path` computation) were caught by these exact tests before ever reaching a cluster.

## Code map

- `crates/compute-pool-core/` -- credentials and settings (`auth.rs`), the Kaggle API client (`kaggle.rs`), job scheduling (`scheduler.rs`, `job.rs`), and the Python bootstrap scripts the CLI pushes to Kaggle kernels, as Rust string constants (`shell.rs`).
- `crates/compute-pool-cli/` -- the `compute-pool` binary: argument parsing (`clap`) and the interactive terminal (`main.rs`).
- `workers/kaggle/` -- the shared Python runtime: `wire.py` (transport over an SSH pipe), `checkpoint.py` (atomic resumable state), `strategy.py` (what the measured fabric can support), `model_shard.py` (load only the tensors a stage needs), `pipeline.py` (the pipeline-parallel stage abstraction), `amp.py` (dynamic loss scaling).
- `examples/` -- reference workloads: `cluster_comm/` (data-parallel all-reduce, link benchmark), `pipeline_lora/` (N-stage pipeline-parallel LoRA fine-tune).
- `tests/` -- the Python test suite, all CPU-only.

## Gotchas specific to this codebase

- **The bootstrap scripts in `shell.rs` are Python, embedded as Rust raw strings.** `cargo build` does not catch a Python syntax error in them. After editing one, fill its placeholders and run it through `py_compile` (see the pattern used in recent commits touching `shell.rs`) before pushing.
- **`pipeline_master.py` and `pipeline_worker.py` are not import-safe.** They load the real model and open live SSH sessions at module scope, immediately on execution. You cannot `import` them to unit-test control flow. Test the underlying logic (`workers/kaggle/pipeline.py`, `workers/kaggle/model_shard.py`) directly against a tiny random `LlamaConfig` model instead -- see `tests/test_pipeline_chain.py` and `tests/test_model_shard.py` for the pattern.
- **PEFT's `get_peft_model()` calls `model.get_input_embeddings()` internally**, even for a stage that never uses the embedding table. Dropping that module entirely (to save memory) breaks it with a `NotImplementedError` that only appears at runtime, on a real cluster, with the real checkpoint. Keep it loaded; the real memory win is in the decoder layers, which are far larger anyway.
- **The decoder layers are called directly, one at a time, bypassing the model's own `forward()`.** That means transformers' usual attention-mask preparation never runs. Any code building a batch with padding must construct the combined causal + padding mask by hand (see `build_4d_mask` in `workers/kaggle/pipeline.py`) rather than relying on `attention_mask=...` doing the right thing automatically.
- **Checkpointing across stages must be coordinated, not independent.** Two sides checkpointing on their own timer can end up saved at different steps after a crash between the two saves. The fix in this codebase is: the master asks each stage to save and acks before saving its own. Don't reintroduce independent per-side checkpoint timers.
- **A dropped SSH link and a dead remote process look identical from the master's side** (the same `EOFError`/`BrokenPipeError`/`OSError`), and you often cannot tell locally which one happened. The reconnect logic in `pipeline_master.py` handles both uniformly: always kill any stray remote process before relaunching, never assume the old one is gone.
- **Windows `subprocess.run([...], shell=True, executable=X)` does not use `X` as the interpreter the way it does on POSIX.** Use explicit argv (`[bash_path, "-c", command]`) instead. This only matters for running the test suite on a Windows dev machine; the real target is Linux.
- **The fp16 loss scale belongs to one side only** (whichever side applies `loss * scale`), and every downstream gradient has that one factor baked into it via the chain rule. An independently-adjusted scale on the other side silently diverges from what's actually in the gradients it receives. Only one `DynamicLossScaler` should ever decide the scale for a given gradient chain.

## Don't add a second name for something that already exists

`cp-dispatch` was removed because its entire body was `exec crun --gpus "$@"` -- a second name for one action, not a second capability. Before adding a new command, check whether an existing one with a new flag already covers it.

## This project pools Kaggle accounts, which is against Kaggle's policy

Read the platform-risk note in the README before testing anything that uses real Kaggle credentials. Don't suggest ways to make the multi-account pattern less detectable; that is explicitly out of scope and against the project's own stated terms of use for itself.
