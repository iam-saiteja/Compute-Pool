"""Run several micro-batches through a pipeline at once, so the stages overlap.

Each remote stage processes its requests in arrival order and replies in the same
order. A Channel keeps that order: requests are enqueued under a lock together with
their futures, and one reader thread resolves the futures in order. Each micro-batch
then runs in its own thread. While stage 1 works on micro-batch j+1, stage 2 can work
on micro-batch j, which is the overlap the sequential chain gave up.

Correctness does not depend on timing. A micro-batch's backward request can only be
sent after its own forward replies have come back, so the per-stage graphs are always
ready before they are used.
"""
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor


class Channel:
    def __init__(self, send, recv):
        """send(msg) writes a request; recv() blocks for the next reply, in order."""
        self._send = send
        self._recv = recv
        self._futures = queue.Queue()
        self._lock = threading.Lock()
        self._dead = None
        threading.Thread(target=self._reader, daemon=True).start()

    def request(self, msg):
        fut = Future()
        with self._lock:
            if self._dead is not None:
                raise self._dead
            # Enqueue the future and send the request under one lock, so the reply order matches.
            self._futures.put(fut)
            self._send(msg)
        return fut

    def call(self, msg):
        return self.request(msg).result()

    def _reader(self):
        while True:
            try:
                reply = self._recv()
            except BaseException as exc:  # the connection is gone: fail everything waiting
                with self._lock:
                    self._dead = exc
                    while not self._futures.empty():
                        self._futures.get().set_exception(exc)
                return
            self._futures.get().set_result(reply)


def run_micro(stage0, channels, j, ids, labels, mask, train=True):
    """One micro-batch through every stage. channels[k] talks to stage k (k >= 1).
    Returns (loss, scale). In training it also runs the backward pass through the chain."""
    n = len(channels) + 1
    h = stage0.forward(j, ids, mask, train=train)
    for k in range(1, n - 1):
        h = channels[k].call({"cmd": "fwd", "mb": j, "h": h, "mask": mask, "train": train})["h"]
    last = channels[n - 1].call({"cmd": "fwd_loss", "mb": j, "h": h, "mask": mask,
                                  "labels": labels, "train": train})
    if not train:
        return last["loss"], None
    grad = last["grad"]
    for k in range(n - 2, 0, -1):
        grad = channels[k].call({"cmd": "bwd", "mb": j, "grad": grad})["grad"]
    stage0.backward(j, grad)
    return last["loss"], last["scale"]


def run_micro_batches(stage0, channels, batches, train=True):
    """Run every (j, ids, labels, mask) concurrently. Returns [(loss, scale), ...] in order."""
    if not batches:
        return []
    with ThreadPoolExecutor(max_workers=len(batches)) as ex:
        futs = [ex.submit(run_micro, stage0, channels, j, ids, labels, mask, train)
                for j, ids, labels, mask in batches]
        return [f.result() for f in futs]
