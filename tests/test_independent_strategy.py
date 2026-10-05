"""Local tests for the independent/map strategy.

The remote-node path normally spawns a worker over SSH; here it spawns a local
Python subprocess running the same worker_serve() instead, so the dispatch
logic, the wire protocol, retry and checkpoint resume are all exercised
without a cluster. Run with the project's uv venv:
    python tests/test_independent_strategy.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.kaggle import checkpoint  # noqa: E402
from workers.kaggle.strategies import independent  # noqa: E402

independent.SHARD_TIMEOUT_SECONDS = 15  # fail fast in tests instead of the 30 min production default

_WORKER_ENTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "workers", "kaggle", "strategies", "independent.py")


def _local_subprocess_argv(host):
    # Stand in for `ssh host ...`: just run worker_serve() as a local process.
    return [sys.executable, "-u", "-c",
            "import sys; sys.path.insert(0, r'" + os.path.dirname(os.path.dirname(os.path.dirname(_WORKER_ENTRY))) + "'); "
            "from workers.kaggle.strategies import independent as ind; ind.worker_serve()"]


def _nodes():
    return [
        {"name": "node0", "index": 0, "local": True, "gpus": 1},
        {"name": "node1", "index": 1, "local": False, "gpus": 2},
    ]


def test_map_runs_every_shard_across_local_and_remote():
    out = independent.run_map(
        _nodes(), "echo shard-{task_index}-of-{task_count}-on-{node_index}",
        shard_count=6, worker_argv_for=_local_subprocess_argv,
    )
    assert not out["failed"]
    assert set(out["results"]) == set(range(6))
    for i, r in out["results"].items():
        assert r["returncode"] == 0
        assert r["stdout"].strip() in (f"shard-{i}-of-6-on-0", f"shard-{i}-of-6-on-1")


def test_map_retries_then_fails_a_shard_that_always_errors():
    out = independent.run_map(
        _nodes(), "bash -c 'test {task_index} -ne 3'",
        shard_count=5, max_retries=1, worker_argv_for=_local_subprocess_argv,
    )
    assert out["failed"] == [3]
    assert set(out["results"]) == {0, 1, 2, 4}


def test_map_resumes_from_checkpoint():
    with tempfile.TemporaryDirectory() as root:
        store = checkpoint.open_store(f"local://{root}", "resume-job")
        seen = []
        out1 = independent.run_map(
            [{"name": "node0", "index": 0, "local": True, "gpus": 1}],
            "bash -c 'if [ {task_index} -ge 3 ]; then exit 7; fi; echo ok-{task_index}'",
            shard_count=6, checkpoint_store=store, max_retries=0,
            on_progress=lambda idx, code, done, total: seen.append(idx),
        )
        assert set(out1["results"]) == {0, 1, 2}
        assert out1["failed"] == [3, 4, 5]

        # A second store pointed at the same job should see the checkpoint and
        # skip the shards that already succeeded.
        store2 = checkpoint.open_store(f"local://{root}", "resume-job")
        out2 = independent.run_map(
            [{"name": "node0", "index": 0, "local": True, "gpus": 1}],
            "echo ok-{task_index}",
            shard_count=6, checkpoint_store=store2, max_retries=0,
        )
        # the 3 that succeeded before are not re-run; the rest complete now
        assert set(out2["results"]) == {0, 1, 2, 3, 4, 5}
        assert not out2["failed"]


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
        _run("map_runs_every_shard_across_local_and_remote", test_map_runs_every_shard_across_local_and_remote),
        _run("map_retries_then_fails_a_shard_that_always_errors", test_map_retries_then_fails_a_shard_that_always_errors),
        _run("map_resumes_from_checkpoint", test_map_resumes_from_checkpoint),
    ]
    sys.exit(0 if all(results) else 1)
