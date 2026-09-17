"""
Local JSON state store for Compute Pool.

All mutable state (job records) is stored in:
    data/jobs.json

This file is gitignored — only the schema and code are tracked.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from compute_pool.jobs.model import Job

DATA_DIR = Path("data")
JOBS_FILE = DATA_DIR / "jobs.json"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_all_jobs() -> list[Job]:
    _ensure_data_dir()
    if not JOBS_FILE.exists():
        return []
    try:
        raw = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        return [Job.from_dict(d) for d in raw]
    except (json.JSONDecodeError, KeyError):
        return []


def save_all_jobs(jobs: list[Job]) -> None:
    _ensure_data_dir()
    JOBS_FILE.write_text(
        json.dumps([j.to_dict() for j in jobs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def upsert_job(job: Job) -> None:
    jobs = load_all_jobs()
    existing = {j.id: j for j in jobs}
    existing[job.id] = job
    save_all_jobs(list(existing.values()))


def get_job(job_id: str) -> Optional[Job]:
    for j in load_all_jobs():
        if j.id == job_id:
            return j
    return None
