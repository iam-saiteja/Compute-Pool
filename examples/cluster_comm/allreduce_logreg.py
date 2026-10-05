"""Two-node data-parallel logistic regression over the SSH link.

Each node holds half of the rows and computes its gradient sum; the sums are
combined with an all-reduce every iteration. The result must match a single
run over all rows, which is what the check at the end verifies.

Master (run on node0):    python3 allreduce_logreg.py
Worker is started by the master: over SSH to CP_WORKER_HOST if set, or as a
local subprocess otherwise (for testing on one machine).
"""
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from cp_wire import allreduce_sum_master, allreduce_sum_worker, recv, send, spawn

N = 20000
D = 50
ITERS = 200
LR = 1.0
WORKER_HOST = os.environ.get("CP_WORKER_HOST")
HERE = os.path.dirname(os.path.abspath(__file__))


def make_data():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(N, D, generator=g, dtype=torch.float64)
    w_true = torch.randn(D, generator=g, dtype=torch.float64)
    y = (torch.rand(N, generator=g, dtype=torch.float64) < torch.sigmoid(X @ w_true)).double()
    return X, y


def grad_sum(X, y, w):
    return X.T @ (torch.sigmoid(X @ w) - y)


def mean_loss(X, y, w):
    z = X @ w
    return (F.softplus(z) - y * z).mean().item()


def run_worker():
    X, y = make_data()
    half = N // 2
    Xw, yw = X[half:], y[half:]
    w = torch.zeros(D, dtype=torch.float64)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    for _ in range(ITERS):
        g = allreduce_sum_worker(grad_sum(Xw, yw, w), stdin, stdout)
        w = w - LR * g / N
    send(stdout, w)
    recv(stdin)


def run_master():
    X, y = make_data()
    half = N // 2
    Xm, ym = X[:half], y[:half]

    w_ref = torch.zeros(D, dtype=torch.float64)
    for _ in range(ITERS):
        w_ref = w_ref - LR * grad_sum(X, y, w_ref) / N

    if WORKER_HOST:
        subprocess.run(["scp", "-o", "ConnectTimeout=5", os.path.join(HERE, "allreduce_logreg.py"),
                        os.path.join(HERE, "cp_wire.py"), f"{WORKER_HOST}:/kaggle/working/"], check=True)
        argv = ["ssh", "-o", "BatchMode=yes", WORKER_HOST,
                "cd /kaggle/working && exec python3 -u allreduce_logreg.py worker"]
    else:
        argv = [sys.executable, "-u", os.path.abspath(__file__), "worker"]
    worker = spawn(argv)
    peers = [(worker.stdout, worker.stdin)]

    w = torch.zeros(D, dtype=torch.float64)
    started = time.time()
    for _ in range(ITERS):
        g = allreduce_sum_master(grad_sum(Xm, ym, w), peers)
        w = w - LR * g / N
    per_iter = (time.time() - started) / ITERS

    w_worker = recv(worker.stdout)
    send(worker.stdin, "done")
    worker.wait()

    print(f"iterations: {ITERS}, {per_iter * 1000:.1f} ms per all-reduce iteration", flush=True)
    print(f"max |w_distributed - w_single_node| = {(w - w_ref).abs().max().item():.2e}", flush=True)
    print(f"max |w_master - w_worker|           = {(w - w_worker).abs().max().item():.2e}", flush=True)
    print(f"loss single-node {mean_loss(X, y, w_ref):.6f}  distributed {mean_loss(X, y, w):.6f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        run_worker()
    else:
        run_master()
