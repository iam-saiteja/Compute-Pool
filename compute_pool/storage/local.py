"""
Local JSON state store for Compute Pool.

All mutable state (job records) is stored in:
    data/jobs.json

This file is gitignored — only the schema and code are tracked.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional, Union

from compute_pool.jobs.model import Job, JobState

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


def next_job_id() -> str:
    """Generate the next sequential 0-indexed job ID ('0', '1', '2', ...)."""
    jobs = load_all_jobs()
    if not jobs:
        return "0"

    indices = []
    for j in jobs:
        m = re.match(r"^(?:job-)?(\d+)$", str(j.id))
        if m:
            indices.append(int(m.group(1)))

    if indices:
        return str(max(indices) + 1)
    return str(len(jobs))


def upsert_job(job: Job) -> None:
    jobs = load_all_jobs()
    existing = {j.id: j for j in jobs}
    existing[job.id] = job
    save_all_jobs(list(existing.values()))


def _normalize_id_match(target: str, candidate_id: str) -> bool:
    target_clean = str(target).strip()
    cand_clean = str(candidate_id).strip()
    if cand_clean == target_clean:
        return True
    if cand_clean == f"job-{target_clean}":
        return True
    if target_clean == f"job-{cand_clean}":
        return True
    return False


def get_job(job_id: Union[str, int]) -> Optional[Job]:
    """Retrieve a job by ID or numeric index (e.g. 0, '0', 'job-0')."""
    target = str(job_id).strip()
    for j in load_all_jobs():
        if _normalize_id_match(target, j.id):
            return j
    return None


def delete_job(job_id: Union[str, int]) -> Optional[Job]:
    """Delete a single job by ID or numeric index. Cleans up artifacts directory."""
    jobs = load_all_jobs()
    target_str = str(job_id).strip()
    target_job = None
    remaining = []

    for j in jobs:
        if _normalize_id_match(target_str, j.id) and target_job is None:
            target_job = j
        else:
            remaining.append(j)

    if target_job:
        save_all_jobs(remaining)
        job_dir = DATA_DIR / "jobs" / str(target_job.id)
        if job_dir.exists() and job_dir.is_dir():
            shutil.rmtree(job_dir, ignore_errors=True)

    return target_job


def clear_jobs(all_jobs: bool = False) -> list[Job]:
    """
    Clear job records from local store.
    - If all_jobs=True: Deletes all job records and all job artifacts.
    - If all_jobs=False: Deletes completed, failed, and cancelled job records.
    Returns list of deleted jobs.
    """
    jobs = load_all_jobs()
    active_states = {JobState.RUNNING, JobState.ASSIGNED, JobState.QUEUED, JobState.SCHEDULING}
    deleted = []
    remaining = []

    for j in jobs:
        if all_jobs or j.state not in active_states:
            deleted.append(j)
            job_dir = DATA_DIR / "jobs" / str(j.id)
            if job_dir.exists() and job_dir.is_dir():
                shutil.rmtree(job_dir, ignore_errors=True)
        else:
            remaining.append(j)

    save_all_jobs(remaining)
    return deleted


def reindex_jobs() -> list[Job]:
    """
    Re-index all existing jobs chronologically from 0 to N.
    Renames local artifact directories to match new IDs.
    Returns the updated job list.
    """
    jobs = load_all_jobs()
    # Sort chronologically
    jobs.sort(key=lambda j: j.created_at or "")

    for idx, j in enumerate(jobs):
        old_id = str(j.id)
        new_id = str(idx)
        if old_id != new_id:
            old_dir = DATA_DIR / "jobs" / old_id
            new_dir = DATA_DIR / "jobs" / new_id
            if old_dir.exists() and not new_dir.exists():
                try:
                    old_dir.rename(new_dir)
                except Exception:
                    pass
            j.id = new_id

    save_all_jobs(jobs)
    return jobs
