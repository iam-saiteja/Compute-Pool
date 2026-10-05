"""Local tests for the Kaggle worker runtime.

Run with the project's uv venv:
    python -m pytest tests/test_worker_runtime.py
or without pytest:
    python tests/test_worker_runtime.py
These exercise checkpoint save/load/resume, the transport framing over a pipe,
and strategy feasibility -- everything that does not need a live cluster.
"""
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from workers.kaggle import checkpoint, strategy, wire  # noqa: E402


def test_checkpoint_roundtrip_and_resume():
    with tempfile.TemporaryDirectory() as root:
        store = checkpoint.open_store(f"local://{root}", "job-1")
        assert store.latest_step() is None

        src = os.path.join(root, "src")
        os.makedirs(src)
        with open(os.path.join(src, "model.bin"), "wb") as f:
            f.write(b"weights-at-50")
        store.save(50, {"model.bin": os.path.join(src, "model.bin")}, meta={"loss": 1.23})

        with open(os.path.join(src, "model.bin"), "wb") as f:
            f.write(b"weights-at-100")
        store.save(100, {"model.bin": os.path.join(src, "model.bin")}, meta={"loss": 0.45})

        assert store.latest_step() == 100
        dest = os.path.join(root, "restore")
        step, meta = store.load_latest(dest)
        assert step == 100 and meta["loss"] == 0.45
        with open(os.path.join(dest, "model.bin"), "rb") as f:
            assert f.read() == b"weights-at-100"


def test_wire_framing_over_pipe():
    r, w = os.pipe()
    with os.fdopen(w, "wb") as wf, os.fdopen(r, "rb") as rf:
        wire.send(wf, {"step": 7, "grad": torch.ones(4)})
        wf.flush()
        wf.close()
        msg = wire.recv(rf)
    assert msg["step"] == 7 and bool((msg["grad"] == 1).all())


def test_writer_thread_survives_a_dead_peer():
    """Found on a live cluster: killing the worker mid-run left an unhandled
    BrokenPipeError in the background writer thread. Non-fatal (the main
    thread's own recv() independently detects the disconnect) but it printed
    a spurious traceback. The writer thread must now exit quietly instead."""
    r, w = os.pipe()
    wf, rf = os.fdopen(w, "wb"), os.fdopen(r, "rb")

    captured = []
    orig_hook = threading.excepthook
    threading.excepthook = captured.append
    try:
        q, t = wire.start_writer(wf)
        q.put(torch.ones(4))
        rf.close()  # the peer is gone
        q.put(torch.ones(4))  # should hit BrokenPipeError and exit quietly
        q.put(None)
        t.join(timeout=5)
    finally:
        threading.excepthook = orig_hook

    assert not t.is_alive()
    assert not captured, f"writer thread raised unhandled: {captured}"


def test_strategy_feasibility_on_kaggle_fabric():
    f = strategy.KAGGLE_SSH_FABRIC
    feasible = strategy.feasible_strategies(f)
    # map / intra-node / periodic-sync / pipeline work; tight collectives do not
    assert "independent" in feasible
    assert "intra_node_ddp" in feasible
    assert "data_parallel_sync" in feasible
    assert "pipeline" in feasible
    assert "sync_collective" not in feasible
    ok, reason = strategy.get("sync_collective").feasible(f)
    assert not ok and "direct" in reason


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
        _run("checkpoint_roundtrip_and_resume", test_checkpoint_roundtrip_and_resume),
        _run("wire_framing_over_pipe", test_wire_framing_over_pipe),
        _run("writer_thread_survives_a_dead_peer", test_writer_thread_survives_a_dead_peer),
        _run("strategy_feasibility_on_kaggle_fabric", test_strategy_feasibility_on_kaggle_fabric),
    ]
    sys.exit(0 if all(results) else 1)
