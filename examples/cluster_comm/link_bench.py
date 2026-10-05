"""Measure round-trip time and throughput of the master-worker link.

The worker echoes every message back. The master sends tensors of growing size
and reports the median round trip. Uses the same framing as the pipeline, so
the numbers include serialization.

Master:  python3 link_bench.py                      (local worker, for testing)
         CP_WORKER_HOST=node1 python3 link_bench.py (worker over SSH)
"""
import os
import statistics
import subprocess
import sys
import time

import torch

from cp_wire import recv, send, spawn

WORKER_HOST = os.environ.get("CP_WORKER_HOST")
HERE = os.path.dirname(os.path.abspath(__file__))
SIZES_MB = [0.001, 0.0625, 1, 2, 8]
REPS = 5


def run_worker():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        msg = recv(stdin)
        if msg is None:
            break
        send(stdout, msg)


def run_master():
    if WORKER_HOST:
        subprocess.run(["scp", "-o", "ConnectTimeout=5", os.path.join(HERE, "link_bench.py"),
                        os.path.join(HERE, "cp_wire.py"), f"{WORKER_HOST}:/kaggle/working/"], check=True)
        argv = ["ssh", "-o", "BatchMode=yes", WORKER_HOST,
                "cd /kaggle/working && exec python3 -u link_bench.py worker"]
    else:
        argv = [sys.executable, "-u", os.path.abspath(__file__), "worker"]
    worker = spawn(argv)
    w_in, w_out = worker.stdin, worker.stdout

    print(f"{'size':>8}  {'median round trip':>18}  {'throughput (both ways)':>24}", flush=True)
    for mb in SIZES_MB:
        tensor = torch.zeros(max(1, int(mb * 1024 * 1024 / 4)), dtype=torch.float32)
        times = []
        for _ in range(REPS):
            t0 = time.time()
            send(w_in, tensor)
            recv(w_out)
            times.append(time.time() - t0)
        rtt = statistics.median(times)
        throughput = 2 * tensor.numel() * 4 / rtt / (1024 * 1024)
        print(f"{mb:>6} MB  {rtt * 1000:>15.1f} ms  {throughput:>18.1f} MB/s", flush=True)

    send(w_in, None)
    worker.wait()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        run_worker()
    else:
        run_master()
