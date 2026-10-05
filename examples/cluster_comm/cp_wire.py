"""Length-prefixed torch messages over a process's stdin/stdout.

Used to talk between the master and a worker started over SSH. Nothing else
may be written to a worker's stdout, so logs go to stderr.
"""
import io
import queue
import struct
import subprocess
import threading

import torch


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
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


def recv(stream):
    n = struct.unpack("<Q", read_exact(stream, 8))[0]
    return torch.load(io.BytesIO(read_exact(stream, n)), weights_only=False)


def spawn(argv):
    return subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    return obj


def start_writer(stream):
    """Send queued objects from a background thread. Put None to flush and stop.

    Tensors may still be on a GPU when queued: the copy to CPU happens here, so
    the producing thread does not wait for the GPU before queueing more work.
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
    """Receive objects in a background thread. A None item marks end of stream."""
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
    """Sum `local` with each worker's tensor and send the total back to every worker.

    `peers` is a list of (stdin, stdout) pairs, one per worker process.
    Workers must call allreduce_sum_worker at the same point in the program.
    """
    total = local.clone()
    for w_in, _ in peers:
        total += recv(w_in)
    for _, w_out in peers:
        send(w_out, total)
    return total


def allreduce_sum_worker(local, w_in, w_out):
    send(w_out, local)
    return recv(w_in)
