"""Checks that examples/pipeline_lora's sys.path insertion actually resolves to
the repo root, so `from workers.kaggle import checkpoint` succeeds when either
pipeline script runs from its own directory. This caught a real off-by-one
(one dirname() call too many) during development; keep it as a regression
check rather than relying on a full 8B model load to notice the same bug.
"""
import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(REPO_ROOT, "examples", "pipeline_lora")


def _extract_sys_path_insert(script_path):
    """Pull out and eval the real sys.path.insert(0, <expr>) line that should
    resolve to the repo root, without executing the rest of the script (which
    imports torch/transformers/peft at module scope). Evaluated with the
    actual file's __file__ and HERE (= dirname(abspath(__file__)), as both
    scripts define it) so this exercises the real expression, not a copy of it."""
    here = os.path.dirname(os.path.abspath(script_path))
    env = {"os": os, "__file__": script_path, "HERE": here}
    with open(script_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("sys.path.insert(0, os.path.dirname"):
                return eval(line[len("sys.path.insert(0, "):-1], env)
    raise AssertionError(f"no 'sys.path.insert(0, os.path.dirname(...))' line found in {script_path}")


def test_master_path_resolves_to_repo_root():
    resolved = _extract_sys_path_insert(os.path.join(PIPELINE_DIR, "pipeline_master.py"))
    assert resolved == REPO_ROOT, f"pipeline_master.py resolves to {resolved}, expected {REPO_ROOT}"


def test_worker_path_resolves_to_repo_root():
    resolved = _extract_sys_path_insert(os.path.join(PIPELINE_DIR, "pipeline_worker.py"))
    assert resolved == REPO_ROOT, f"pipeline_worker.py resolves to {resolved}, expected {REPO_ROOT}"


def test_checkpoint_importable_from_resolved_root():
    sys.path.insert(0, REPO_ROOT)
    mod = importlib.import_module("workers.kaggle.checkpoint")
    assert hasattr(mod, "open_store")


def _run(name, fn):
    try:
        fn()
        print(f"ok   {name}")
        return True
    except Exception as e:
        print(f"FAIL {name}: {e!r}")
        return False


if __name__ == "__main__":
    results = [
        _run("master_path_resolves_to_repo_root", test_master_path_resolves_to_repo_root),
        _run("worker_path_resolves_to_repo_root", test_worker_path_resolves_to_repo_root),
        _run("checkpoint_importable_from_resolved_root", test_checkpoint_importable_from_resolved_root),
    ]
    sys.exit(0 if all(results) else 1)
