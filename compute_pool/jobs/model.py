"""
Job model and state machine for Compute Pool.

States:
    QUEUED → SCHEDULING → ASSIGNED → RUNNING → COMPLETED
                                              ↘ FAILED → RETRYING
    CANCELLED
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Union

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobState(str, Enum):
    QUEUED = "QUEUED"
    SCHEDULING = "SCHEDULING"
    ASSIGNED = "ASSIGNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    CANCELLED = "CANCELLED"


@dataclass
class JobSpec:
    """User-supplied job specification (from YAML)."""
    name: str
    script: str                       # path or inline Python
    gpu: bool = True
    gpu_memory_gb: float = 8.0
    max_runtime_hours: float = 4.0
    checkpointable: bool = True
    max_retries: int = 3


def _default_job_id() -> str:
    try:
        from compute_pool.storage.local import next_job_id
        return next_job_id()
    except Exception:
        return "0"


@dataclass
class Job:
    """Live job record tracked in state store."""
    id: str = field(default_factory=_default_job_id)
    spec: JobSpec = field(default_factory=lambda: JobSpec(name="unnamed", script=""))
    state: JobState = JobState.QUEUED
    assigned_slot: Optional[Union[int, str]] = None          # 1, 2, or "1, 2"
    assigned_username: Optional[str] = None
    kaggle_kernel_slug: Optional[str] = None
    retry_count: int = 0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    error: Optional[str] = None

    def transition(self, new_state: JobState, error: Optional[str] = None) -> None:
        self.state = new_state
        self.updated_at = _now_iso()
        if error:
            self.error = error

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.spec.name,
            "script": self.spec.script,
            "gpu": self.spec.gpu,
            "state": self.state.value,
            "assigned_slot": self.assigned_slot,
            "assigned_username": self.assigned_username,
            "kaggle_kernel_slug": self.kaggle_kernel_slug,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        spec = JobSpec(
            name=d.get("name", "unnamed"),
            script=d.get("script", ""),
            gpu=d.get("gpu", True),
        )
        job = cls(
            id=d["id"],
            spec=spec,
            state=JobState(d.get("state", "QUEUED")),
            assigned_slot=d.get("assigned_slot"),
            assigned_username=d.get("assigned_username"),
            kaggle_kernel_slug=d.get("kaggle_kernel_slug"),
            retry_count=d.get("retry_count", 0),
            created_at=d.get("created_at", _now_iso()),
            updated_at=d.get("updated_at", _now_iso()),
            error=d.get("error"),
        )
        return job
