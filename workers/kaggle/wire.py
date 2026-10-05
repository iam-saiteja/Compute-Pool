"""Worker-to-worker transport for Compute Pool.

A length-prefixed message channel over a pair of byte streams (an SSH
process's stdin/stdout on the master side, the worker's own stdin/stdout on
the other). This is the one place the rest of the runtime talks to another
node, so a future transport (a relay, Tailscale userspace, WebRTC) only has
to provide two streams with the same framing.

The master runs a worker over SSH and holds (proc.stdin, proc.stdout); the
worker uses (sys.stdin.buffer, sys.stdout.buffer). Only torch-serialisable
objects cross the wire. Nothing else may be written to a worker's stdout, so
worker logs must go to stderr.
"""
import io
import queue
import struct
import subprocess
import threading

import torch

_HEADER = struct.Struct("<Q")


def read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("peer closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def send(stream, obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    stream.write(_HEADER.pack(len(data)))
    stream.write(data)
    stream.flush()


def recv(stream):
    n = _HEADER.unpack(read_exact(stream, _HEADER.size))[0]
    return torch.load(io.BytesIO(read_exact(stream, n)), weights_only=False)


def spawn(argv):
    """Start a child process wired for send()/recv() on its stdin/stdout."""
    return subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def start_writer(stream):
    """Send queued objects from a background thread. Put None to flush and stop.

    Tensors may still be on a GPU when queued; the copy to host memory happens
    on this thread, so the producer keeps queueing GPU work instead of blocking
    on the device.
    """
    q = queue.Queue()

    def run():
        while True:
            obj = q.get()
            if obj is None:
                break
            send(stream, _to_cpu(obj))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return q, thread


def start_reader(stream):
    """Receive objects on a background thread. A None item marks end of stream."""
    q = queue.Queue()

    def run():
        try:
            while True:
                q.put(recv(stream))
        except EOFError:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()
    return q


def allreduce_sum_master(local, peers):
    """Sum `local` with each worker's tensor and return the total to every worker.

    `peers` is a list of (recv_stream, send_stream) pairs, one per worker.
    Workers call allreduce_sum_worker at the same point.
    """
    total = local.clone()
    for r, _ in peers:
        total += recv(r)
    for _, w in peers:
        send(w, total)
    return total


def allreduce_sum_worker(local, recv_stream, send_stream):
    send(send_stream, local)
    return recv(recv_stream)
