"""Checkpoint storage for Compute Pool.

A worker is ephemeral: it can vanish mid-job, so its local disk is never the
only copy of important state. A CheckpointStore writes checkpoints to a
durable location outside the worker and lets a replacement worker resume from
the latest one. That resume-on-another-worker loop is the platform's core
value (recovered compute), so checkpointing is a first-class primitive, not a
per-example concern.

Backends:
  local://<dir>                     a directory (for tests and single-node runs)
  s3://<bucket>/<prefix>            S3-compatible object storage (needs boto3 and
                                    AWS_* / endpoint env vars)

A checkpoint is a directory of files plus a state.json with {step, meta}. Saves
are atomic: write to a temp name, then rename, so a crash mid-save can't corrupt
the latest good checkpoint.
"""
import json
import os
import shutil
import tempfile


def open_store(uri, job_id):
    if uri.startswith("local://"):
        return LocalCheckpointStore(uri[len("local://"):], job_id)
    if uri.startswith("s3://"):
        rest = uri[len("s3://"):]
        bucket, _, prefix = rest.partition("/")
        return S3CheckpointStore(bucket, prefix, job_id)
    raise ValueError(f"unsupported checkpoint URI: {uri!r} (use local:// or s3://)")


class LocalCheckpointStore:
    def __init__(self, root, job_id):
        self.base = os.path.join(root, job_id)
        os.makedirs(self.base, exist_ok=True)

    def latest_step(self):
        steps = [int(n[5:]) for n in os.listdir(self.base) if n.startswith("step-") and n[5:].isdigit()]
        return max(steps) if steps else None

    def save(self, step, files, meta=None):
        """files: {name: local_path}. Written atomically under step-<step>/."""
        tmp = tempfile.mkdtemp(dir=self.base, prefix=f".tmp-step-{step}-")
        try:
            for name, path in files.items():
                shutil.copy2(path, os.path.join(tmp, name))
            with open(os.path.join(tmp, "state.json"), "w") as f:
                json.dump({"step": step, "meta": meta or {}}, f)
            final = os.path.join(self.base, f"step-{step}")
            if os.path.exists(final):
                shutil.rmtree(final)
            os.rename(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return step

    def load(self, step, dest_dir):
        """Copy a checkpoint's files into dest_dir. Returns (step, meta)."""
        src = os.path.join(self.base, f"step-{step}")
        os.makedirs(dest_dir, exist_ok=True)
        meta = {}
        for name in os.listdir(src):
            if name == "state.json":
                with open(os.path.join(src, name)) as f:
                    meta = json.load(f).get("meta", {})
                continue
            shutil.copy2(os.path.join(src, name), os.path.join(dest_dir, name))
        return step, meta

    def load_latest(self, dest_dir):
        step = self.latest_step()
        return self.load(step, dest_dir) if step is not None else None


class S3CheckpointStore:
    """S3-compatible backend. Lazily imports boto3 so local runs need no AWS deps."""

    def __init__(self, bucket, prefix, job_id):
        import boto3  # noqa: F401 -- fail here with a clear message if missing

        self._boto3 = boto3
        self.bucket = bucket
        self.prefix = "/".join(p for p in (prefix.strip("/"), job_id) if p)
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        self.client = boto3.client("s3", endpoint_url=endpoint) if endpoint else boto3.client("s3")

    def _key(self, *parts):
        return "/".join([self.prefix, *parts])

    def latest_step(self):
        resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=self._key() + "/")
        steps = set()
        for obj in resp.get("Contents", []):
            rel = obj["Key"][len(self._key()) + 1:]
            head = rel.split("/", 1)[0]
            if head.startswith("step-") and head[5:].isdigit():
                steps.add(int(head[5:]))
        return max(steps) if steps else None

    def save(self, step, files, meta=None):
        for name, path in files.items():
            self.client.upload_file(path, self.bucket, self._key(f"step-{step}", name))
        body = json.dumps({"step": step, "meta": meta or {}}).encode()
        self.client.put_object(Bucket=self.bucket, Key=self._key(f"step-{step}", "state.json"), Body=body)
        return step

    def load(self, step, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=self._key(f"step-{step}") + "/")
        meta = {}
        for obj in resp.get("Contents", []):
            name = obj["Key"].rsplit("/", 1)[-1]
            if name == "state.json":
                body = self.client.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                meta = json.loads(body).get("meta", {})
                continue
            self.client.download_file(self.bucket, obj["Key"], os.path.join(dest_dir, name))
        return step, meta

    def load_latest(self, dest_dir):
        step = self.latest_step()
        return self.load(step, dest_dir) if step is not None else None
