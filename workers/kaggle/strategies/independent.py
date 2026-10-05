"""The independent / map strategy: pool.map over every GPU in the cluster.

This is the PRD's stated most important V0 workload (docs/PRD.md section 20,
section 39): shard a dataset, run one shard per GPU, retry failed shards,
collect results. There is no cross-node communication of training state, so a
worker disappearing mid-run costs only its in-flight shard, not the job: that
shard goes back into the queue and a surviving worker picks it up. Progress is
checkpointed after every completed shard, so the whole job can also resume
across a restart of the driving process.

Local GPUs run each shard as a direct subprocess. Remote nodes run a
persistent shard server reached over the wire transport (workers/kaggle/wire.py)
so the SSH connection is paid for once, not per shard.
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from workers.kaggle import wire  # noqa: E402

SHARD_TIMEOUT_SECONDS = 1800


def _resolve_bash():
    # The real target is Linux, where /bin/bash always exists. These extra
    # candidates only matter for running the test suite on Windows, where
    # shutil.which("bash") can return a mangled MSYS-style path.
    for candidate in (r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash"), "/bin/bash"):
        if candidate and os.path.exists(candidate):
            return candidate
    return "/bin/bash"


_BASH = _resolve_bash()


def _run_local(command, env):
    # Invoke bash directly as argv rather than via shell=True: on Windows,
    # shell=True always uses cmd.exe to interpret the command regardless of
    # `executable`, so executable=<bash path> does not do what it does on
    # POSIX. Explicit argv is correct and identical in behavior on both.
    full_env = os.environ.copy()
    full_env.update(env)
    res = subprocess.run([_BASH, "-c", command], capture_output=True, text=True, env=full_env)
    return res.returncode, res.stdout, res.stderr


def worker_serve(stream_in=None, stream_out=None):
    """Run on a remote node: serve shard requests until told to stop.

    Message in:  {"cmd": "shard", "index": i, "command": str, "env": {...}}
                 {"cmd": "stop"}
    Message out: {"index": i, "returncode": int, "stdout": str, "stderr": str}
    """
    stream_in = stream_in or sys.stdin.buffer
    stream_out = stream_out or sys.stdout.buffer
    inbox = wire.start_reader(stream_in)
    outbox, writer = wire.start_writer(stream_out)
    while True:
        msg = inbox.get()
        if msg is None or msg.get("cmd") == "stop":
            break
        if msg.get("cmd") == "shard":
            try:
                code, out, err = _run_local(msg["command"], msg.get("env", {}))
            except Exception as exc:
                code, out, err = -1, "", f"worker_serve: {exc!r}"
            outbox.put({"index": msg["index"], "returncode": code, "stdout": out, "stderr": err})
    outbox.put(None)
    writer.join()
    sys.stderr.flush()
    os._exit(0)


class _Slot:
    """One execution slot: the local machine, or a remote node's shard server."""

    def __init__(self, name, node_index, local, proc=None, out_q=None, in_q=None):
        self.name = name
        self.node_index = node_index
        self.local = local
        self.proc = proc
        self.out_q = out_q
        self.in_q = in_q
        self.alive = True


def _spawn_remote_slot(node, worker_argv_for):
    proc = wire.spawn(worker_argv_for(node["name"]))
    out_q, _writer = wire.start_writer(proc.stdin)
    in_q = wire.start_reader(proc.stdout)
    return _Slot(node["name"], node["index"], local=False, proc=proc, out_q=out_q, in_q=in_q)


def default_worker_argv(host):
    """Default remote launch: SSH to `host` and serve shards from /kaggle/working."""
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=20",
        host,
        "cd /kaggle/working && exec python3 -u -c "
        "\"import sys; sys.path.insert(0, '/kaggle/working'); "
        "from workers.kaggle.strategies import independent as ind; ind.worker_serve()\"",
    ]


def run_map(nodes, command_template, shard_count, checkpoint_store=None,
            max_retries=2, on_progress=None, worker_argv_for=default_worker_argv):
    """Run `command_template` once per shard, spread across every GPU in `nodes`.

    nodes: online-node dicts {"name", "index", "local", "gpus"} (the cluster
        registry's shape -- see strategy.py and /etc/compute-pool/nodes.json).
    command_template: a str.format()-able shell command; receives task_index,
        task_count, node_index. Each shard also gets those as env vars
        (CP_TASK_INDEX, CP_TASK_COUNT, CP_NODE_INDEX).
    checkpoint_store: a workers.kaggle.checkpoint store, or None to skip
        checkpointing. If it already has a checkpoint for this job, completed
        shards are skipped.
    worker_argv_for: override to replace SSH with something else (tests use
        this to spawn a local Python process as a stand-in for a remote node).

    Returns {"results": {shard_index: {"returncode", "stdout", "stderr"}},
             "failed": [shard_index, ...]}.
    """
    completed = {}
    if checkpoint_store is not None:
        resume_dir = tempfile.mkdtemp(prefix="cp-map-resume-")
        loaded = checkpoint_store.load_latest(resume_dir)
        if loaded:
            _, meta = loaded
            for idx_str, result in meta.get("completed", {}).items():
                completed[int(idx_str)] = result

    pending = queue.Queue()
    for i in range(shard_count):
        if i not in completed:
            pending.put(i)

    slots = []
    for n in nodes:
        for _ in range(max(1, n.get("gpus", 1))):
            if n["local"]:
                slots.append(_Slot(n["name"], n["index"], local=True))
            else:
                slots.append(_spawn_remote_slot(n, worker_argv_for))

    lock = threading.Lock()
    results = dict(completed)
    failed = []
    attempts = {}
    # Shards neither completed nor permanently failed yet. A worker thread
    # seeing an empty queue must keep polling while this is nonzero, because
    # another thread's retry can still put work back in -- it must not exit
    # just because the queue was momentarily empty.
    outstanding = [shard_count - len(completed)]

    def save_progress():
        if checkpoint_store is None:
            return
        tmp = tempfile.mkdtemp(prefix="cp-map-ckpt-")
        with open(os.path.join(tmp, "progress.json"), "w") as f:
            json.dump({"shard_count": shard_count}, f)
        checkpoint_store.save(len(results), {"progress.json": os.path.join(tmp, "progress.json")},
                               meta={"completed": {str(k): v for k, v in results.items()}})

    def resolve_failure(idx):
        """A shard attempt failed (exception, or a nonzero exit code). Retry
        or give up; return True if this shard is now terminally resolved."""
        with lock:
            attempts[idx] = attempts.get(idx, 0) + 1
            if attempts[idx] <= max_retries:
                pending.put(idx)
                return False
            failed.append(idx)
            outstanding[0] -= 1
            return True

    def run_slot(slot):
        while True:
            try:
                idx = pending.get(timeout=0.5)
            except queue.Empty:
                with lock:
                    done = outstanding[0] <= 0
                if done or not slot.alive:
                    return
                continue

            if not slot.alive:
                pending.put(idx)
                return

            cmd = command_template.format(task_index=idx, task_count=shard_count, node_index=slot.node_index)
            env = {"CP_TASK_INDEX": str(idx), "CP_TASK_COUNT": str(shard_count),
                   "CP_NODE_INDEX": str(slot.node_index)}
            try:
                if slot.local:
                    code, out, err = _run_local(cmd, env)
                else:
                    slot.out_q.put({"cmd": "shard", "index": idx, "command": cmd, "env": env})
                    reply = slot.in_q.get(timeout=SHARD_TIMEOUT_SECONDS)
                    if reply is None:
                        raise RuntimeError(f"{slot.name} disconnected")
                    code, out, err = reply["returncode"], reply["stdout"], reply["stderr"]
            except Exception:
                resolve_failure(idx)
                if not slot.local:
                    slot.alive = False
                    return
                continue

            if code != 0:
                resolve_failure(idx)
                continue

            with lock:
                results[idx] = {"returncode": code, "stdout": out, "stderr": err}
                outstanding[0] -= 1
                save_progress()
            if on_progress:
                on_progress(idx, code, len(results), shard_count)

    threads = [threading.Thread(target=run_slot, args=(s,)) for s in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for slot in slots:
        if not slot.local and slot.proc.poll() is None:
            try:
                slot.out_q.put({"cmd": "stop"})
                slot.out_q.put(None)
                slot.proc.wait(timeout=10)
            except Exception:
                slot.proc.kill()

    return {"results": results, "failed": sorted(set(failed))}
