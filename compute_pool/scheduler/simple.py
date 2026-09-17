"""
Simple quota-aware scheduler for Compute Pool.

Algorithm (V1 — deterministic):
  1. Get live status of all slots.
  2. Filter slots that are connected and have remaining GPU quota.
  3. Pick the slot with the most remaining hours.
  4. Assign the job and persist.

This runs locally — actual Kaggle kernel dispatch is wired in dispatch.py.
"""
from __future__ import annotations

from rich.console import Console

from compute_pool.accounts.manager import AccountStatus, get_all_statuses, best_available_slot
from compute_pool.jobs.model import Job, JobState
from compute_pool.storage.local import upsert_job

console = Console()


def schedule_job(job: Job) -> Job:
    """
    Assign `job` to the best available account slot.
    Mutates and persists the job; returns updated job.
    """
    console.print(f"  [dim]Fetching account statuses...[/dim]")
    statuses = get_all_statuses()

    for s in statuses:
        icon = "[green]*[/green]" if s.connected else "[red]x[/red]"
        console.print(
            f"  Slot {s.slot} {icon}  {s.username:20s}  "
            f"{s.gpu_hours_remaining:.2f}h remaining"
        )

    job.transition(JobState.SCHEDULING)
    upsert_job(job)

    winner: AccountStatus | None = best_available_slot(statuses)
    if winner is None:
        job.transition(
            JobState.FAILED,
            error="No account has remaining GPU quota. Check `compute-pool accounts status`.",
        )
        upsert_job(job)
        return job

    job.assigned_slot = winner.slot
    job.assigned_username = winner.username
    job.transition(JobState.ASSIGNED)
    upsert_job(job)

    console.print(
        f"\n  [green]-> Job [bold]{job.id}[/bold] assigned to "
        f"slot {winner.slot} ({winner.username})[/green]"
    )
    return job
