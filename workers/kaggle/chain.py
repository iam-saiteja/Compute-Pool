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
    """Requests and replies to one remote stage, matched by an id each request carries.

    The remote may answer out of order, because a node can run its two halves in
    parallel (HalfPipeline). Each request gets a Future keyed by its id, and a reader
    thread resolves them as replies arrive."""

    def __init__(self, send, recv):
        """send(msg) writes a request; recv() blocks for the next reply, which carries its id."""
        self._send = send
        self._recv = recv
        self._futures = {}
        self._next = 0
        self._lock = threading.Lock()
        self._dead = None
        threading.Thread(target=self._reader, daemon=True).start()

    def request(self, msg):
        fut = Future()
        with self._lock:
            if self._dead is not None:
                raise self._dead
            self._next += 1
            msg = dict(msg, id=self._next)
            self._futures[self._next] = fut
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
                    for fut in self._futures.values():
                        fut.set_exception(exc)
                    self._futures.clear()
                return
            with self._lock:
                fut = self._futures.pop(reply.pop("id"))
            fut.set_result(reply)


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


class HalfPipeline:
    """Serves one node's requests with the node's two GPU halves running in parallel.

    The node's layers are split into a front half (first GPU) and a back half (second
    GPU). Each half has its own thread and queue. Forward runs front, then back. For the
    last node, back also computes the loss. Gradients run back, then front. While the
    back half works on one micro-batch, the front half can start the next.

    reply(payload) is called with each finished reply, which carries the request's id.
    """

    def __init__(self, front, back, reply):
        self.front, self.back = front, back
        self._reply = reply
        self._qF = queue.Queue()
        self._qB = queue.Queue()
        threading.Thread(target=self._front_loop, daemon=True).start()
        threading.Thread(target=self._back_loop, daemon=True).start()

    def submit(self, msg):
        """Queue a fwd, fwd_loss or bwd request. Control requests go through control()."""
        if msg["cmd"] in ("fwd", "fwd_loss"):
            self._qF.put(msg)
        elif msg["cmd"] == "bwd":
            self._qB.put(msg)
        else:
            raise ValueError(f"not a pipeline request: {msg['cmd']}")

    def _front_loop(self):
        while True:
            item = self._qF.get()
            if item["cmd"] == "bwd_front":
                # Back has finished its part of a backward pass. Finish on the front half.
                grad_in = self.front.backward(item["mb"], item["grad"])
                self._reply(dict(item["extra"], id=item["id"], grad=grad_in))
                continue
            out = self.front.forward(item["mb"], item["h"], item.get("mask"), train=item.get("train", True))
            self._qB.put(dict(item, h=out))

    def _back_loop(self):
        while True:
            msg = self._qB.get()
            train = msg.get("train", True)
            if msg["cmd"] == "fwd":
                out = self.back.forward(msg["mb"], msg["h"], msg.get("mask"), train=train)
                self._reply({"id": msg["id"], "h": out})
            elif msg["cmd"] == "fwd_loss":
                loss, grad, scale = self.back.forward_loss(msg["mb"], msg["h"], msg.get("mask"),
                                                           msg["labels"], train=train)
                if not train:
                    self._reply({"id": msg["id"], "loss": loss, "grad": None, "scale": None})
                    continue
                self._qF.put({"cmd": "bwd_front", "mb": msg["mb"], "id": msg["id"], "grad": grad,
                              "extra": {"loss": loss, "scale": scale}})
            elif msg["cmd"] == "bwd":
                grad_mid = self.back.backward(msg["mb"], msg["grad"])
                self._qF.put({"cmd": "bwd_front", "mb": msg["mb"], "id": msg["id"],
                              "grad": grad_mid, "extra": {}})

    def check(self):
        return self.front.grads_finite() and self.back.grads_finite()

    def apply(self, apply_update, n_micro, scale, finite_all):
        self.front.apply(apply_update, n_micro, scale, finite_all)
        return self.back.apply(apply_update, n_micro, scale, finite_all)
