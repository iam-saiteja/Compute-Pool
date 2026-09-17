"""
Multi-account manager for Compute Pool.

Tracks both Kaggle accounts (slot 1 and slot 2), fetches their
kernel run history to estimate quota usage, and exposes available
capacity for the scheduler.

Kaggle free tier (as of 2025):
  - ~30 GPU-hours / week per account
  - resets weekly
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx
from rich.console import Console

from compute_pool.auth.kaggle_auth import load_credentials, _is_bearer_token

console = Console()

KAGGLE_GPU_WEEKLY_QUOTA_HOURS: float = 30.0  # approximate free-tier cap


@dataclass
class AccountStatus:
    slot: int
    username: str
    connected: bool
    gpu_seconds_used: int = 0
    gpu_seconds_total: int = 108000
    tpu_seconds_used: int = 0
    tpu_seconds_total: int = 72000
    quota_refresh_time: str = ""
    error: Optional[str] = None

    @property
    def gpu_hours_used(self) -> float:
        return round(self.gpu_seconds_used / 3600.0, 2)

    @property
    def gpu_hours_total(self) -> float:
        return round(self.gpu_seconds_total / 3600.0, 1)

    @property
    def gpu_hours_remaining(self) -> float:
        return max(0.0, round((self.gpu_seconds_total - self.gpu_seconds_used) / 3600.0, 2))

    @property
    def has_capacity(self) -> bool:
        return self.connected and self.gpu_hours_remaining > 0.05

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "username": self.username,
            "connected": self.connected,
            "gpu_hours_used": self.gpu_hours_used,
            "gpu_hours_remaining": self.gpu_hours_remaining,
            "gpu_hours_total": self.gpu_hours_total,
            "quota_refresh_time": self.quota_refresh_time,
            "error": self.error,
        }


def _make_auth(username: str, key: str):
    """Return (headers, auth) tuple depending on token type."""
    if _is_bearer_token(key):
        return {"Authorization": f"Bearer {key}"}, None
    return {}, (username, key)


def get_account_status(slot: int) -> AccountStatus:
    """Return live status and exact quota for one account slot."""
    creds = load_credentials(slot)
    if creds is None:
        return AccountStatus(
            slot=slot,
            username="<not configured>",
            connected=False,
            error="No credentials found. Run: compute-pool login --slot {}".format(slot),
        )

    username = creds["username"]
    key = creds["key"]

    headers, auth = _make_auth(username, key)
    try:
        resp = httpx.get(
            "https://www.kaggle.com/api/v1/kernels/quota",
            headers=headers,
            auth=auth,
            timeout=10,
        )
        if resp.status_code == 401:
            return AccountStatus(
                slot=slot,
                username=username,
                connected=False,
                error="Authentication failed (401). Re-run: compute-pool login --slot {}".format(slot),
            )
        if resp.status_code in (200, 204):
            data = resp.json()
            gpu_q = data.get("gpuQuota", {})
            tpu_q = data.get("tpuQuota", {})

            def _get_sec(q_obj, name):
                return int(q_obj.get(name, {}).get("seconds", 0))

            gpu_used = _get_sec(gpu_q, "timeUsed")
            gpu_total = _get_sec(gpu_q, "totalTimeAllowed") or 108000
            tpu_used = _get_sec(tpu_q, "timeUsed")
            tpu_total = _get_sec(tpu_q, "totalTimeAllowed") or 72000
            refresh_time = data.get("quotaRefreshTime", "")

            return AccountStatus(
                slot=slot,
                username=username,
                connected=True,
                gpu_seconds_used=gpu_used,
                gpu_seconds_total=gpu_total,
                tpu_seconds_used=tpu_used,
                tpu_seconds_total=tpu_total,
                quota_refresh_time=refresh_time,
            )
    except httpx.ConnectError:
        return AccountStatus(
            slot=slot,
            username=username,
            connected=False,
            error="Cannot reach kaggle.com",
        )
    except Exception as e:
        return AccountStatus(
            slot=slot,
            username=username,
            connected=False,
            error=str(e),
        )

    return AccountStatus(
        slot=slot,
        username=username,
        connected=False,
        error="Unknown response from Kaggle",
    )


def get_all_statuses() -> list[AccountStatus]:
    return [get_account_status(1), get_account_status(2)]


def best_available_slot(statuses: list[AccountStatus]) -> Optional[AccountStatus]:
    """Return the slot with the most remaining quota."""
    available = [s for s in statuses if s.has_capacity]
    if not available:
        return None
    return max(available, key=lambda s: s.gpu_hours_remaining)

