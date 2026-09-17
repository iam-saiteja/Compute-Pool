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
    kernels_run_this_week: int = 0
    estimated_gpu_hours_used: float = 0.0
    estimated_gpu_hours_remaining: float = KAGGLE_GPU_WEEKLY_QUOTA_HOURS
    error: Optional[str] = None

    @property
    def has_capacity(self) -> bool:
        return self.connected and self.estimated_gpu_hours_remaining > 0.5

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "username": self.username,
            "connected": self.connected,
            "kernels_run_this_week": self.kernels_run_this_week,
            "estimated_gpu_hours_used": round(self.estimated_gpu_hours_used, 2),
            "estimated_gpu_hours_remaining": round(self.estimated_gpu_hours_remaining, 2),
            "error": self.error,
        }


def _make_auth(username: str, key: str):
    """Return (headers, auth) tuple depending on token type."""
    if _is_bearer_token(key):
        return {"Authorization": f"Bearer {key}"}, None
    return {}, (username, key)


def _fetch_kernel_count(username: str, key: str) -> int:
    """
    Fetch number of kernels run this week for quota estimation.
    Uses the Kaggle API kernels list endpoint filtered by the owner.
    """
    headers, auth = _make_auth(username, key)
    try:
        resp = httpx.get(
            "https://www.kaggle.com/api/v1/kernels",
            headers=headers,
            auth=auth,
            params={"ownerSlug": username, "pageSize": 100},
            timeout=15,
        )
        if resp.status_code == 200:
            kernels = resp.json()
            # Count kernels that have a lastRunTime in the past 7 days
            now = datetime.utcnow()
            recent = 0
            for k in kernels:
                last_run = k.get("lastRunTime", "")
                if last_run:
                    try:
                        dt = datetime.fromisoformat(last_run.replace("Z", "+00:00"))
                        age_days = (now - dt.replace(tzinfo=None)).days
                        if age_days < 7:
                            recent += 1
                    except ValueError:
                        pass
            return recent
        return 0
    except Exception:
        return 0


def get_account_status(slot: int) -> AccountStatus:
    """Return live status for one account slot."""
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

    # Verify connectivity
    headers, auth = _make_auth(username, key)
    try:
        resp = httpx.get(
            "https://www.kaggle.com/api/v1/competitions/list",
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
        connected = resp.status_code in (200, 204)
    except httpx.ConnectError:
        return AccountStatus(
            slot=slot,
            username=username,
            connected=False,
            error="Cannot reach kaggle.com",
        )

    kernels_run = _fetch_kernel_count(username, key)
    # Rough heuristic: each GPU kernel run ≈ 2 h average GPU usage
    gpu_hours_used = kernels_run * 2.0
    gpu_hours_remaining = max(0.0, KAGGLE_GPU_WEEKLY_QUOTA_HOURS - gpu_hours_used)

    return AccountStatus(
        slot=slot,
        username=username,
        connected=connected,
        kernels_run_this_week=kernels_run,
        estimated_gpu_hours_used=gpu_hours_used,
        estimated_gpu_hours_remaining=gpu_hours_remaining,
    )


def get_all_statuses() -> list[AccountStatus]:
    return [get_account_status(1), get_account_status(2)]


def best_available_slot(statuses: list[AccountStatus]) -> Optional[AccountStatus]:
    """Return the slot with the most remaining quota."""
    available = [s for s in statuses if s.has_capacity]
    if not available:
        return None
    return max(available, key=lambda s: s.estimated_gpu_hours_remaining)
