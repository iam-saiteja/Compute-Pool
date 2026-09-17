"""
compute-pool CLI

Commands:
    compute-pool login --slot {1|2}       Authenticate a Kaggle account
    compute-pool accounts status          Show both account statuses + quota
    compute-pool job submit <spec.yaml>   Submit a job (schedule + assign)
    compute-pool job list                 List all jobs
    compute-pool job status <job-id>      Show one job's full status
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="compute-pool",
    help="Pool voluntarily shared Kaggle free-tier GPU compute across two accounts.",
    add_completion=False,
)
accounts_app = typer.Typer(help="Account management commands.")
job_app = typer.Typer(help="Job management commands.")
app.add_typer(accounts_app, name="accounts")
app.add_typer(job_app, name="job")

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# compute-pool login
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def login(
    slot: int = typer.Option(..., "--slot", "-s", help="Account slot (1 or 2)"),
):
    """Authenticate a Kaggle account for the given slot."""
    if slot not in (1, 2):
        console.print("[red]Error:[/red] --slot must be 1 or 2.")
        raise typer.Exit(1)

    from compute_pool.auth.kaggle_auth import interactive_login
    try:
        interactive_login(slot)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]Login failed:[/red] {exc}")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# compute-pool accounts status
# ─────────────────────────────────────────────────────────────────────────────

@accounts_app.command("status")
def accounts_status():
    """Show live status and estimated GPU quota for both accounts."""
    from compute_pool.accounts.manager import get_all_statuses

    console.print("\n[bold cyan]Compute Pool — Account Status[/bold cyan]\n")
    statuses = get_all_statuses()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Slot", style="bold", width=6)
    table.add_column("Username", width=22)
    table.add_column("Connected", width=10)
    table.add_column("Kernels (7d)", width=14)
    table.add_column("GPU-h Used", width=12)
    table.add_column("GPU-h Left", width=12)
    table.add_column("Note", width=40)

    for s in statuses:
        connected_str = "[green]Yes[/green]" if s.connected else "[red]No[/red]"
        table.add_row(
            str(s.slot),
            s.username,
            connected_str,
            str(s.kernels_run_this_week),
            f"~{s.estimated_gpu_hours_used:.1f}h",
            f"~{s.estimated_gpu_hours_remaining:.1f}h",
            s.error or "",
        )

    console.print(table)
    console.print(
        "\n[dim]Note: GPU quota is estimated from recent kernel history "
        "(Kaggle does not expose exact remaining hours via API).[/dim]\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# compute-pool job submit
# ─────────────────────────────────────────────────────────────────────────────

@job_app.command("submit")
def job_submit(
    spec_file: Path = typer.Argument(..., help="Path to job YAML spec file"),
):
    """Submit a job to the compute pool."""
    import yaml
    from compute_pool.jobs.model import Job, JobSpec
    from compute_pool.scheduler.simple import schedule_job
    from compute_pool.storage.local import upsert_job

    if not spec_file.exists():
        console.print(f"[red]Error:[/red] File not found: {spec_file}")
        raise typer.Exit(1)

    try:
        raw = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]Error parsing YAML:[/red] {exc}")
        raise typer.Exit(1)

    job_data = raw.get("job", raw)
    spec = JobSpec(
        name=job_data.get("name", spec_file.stem),
        script=job_data.get("script", ""),
        gpu=job_data.get("resources", {}).get("gpu", True),
        gpu_memory_gb=float(job_data.get("resources", {}).get("gpu_memory_gb", 8)),
        max_runtime_hours=float(job_data.get("execution", {}).get("max_runtime_hours", 4)),
        checkpointable=job_data.get("execution", {}).get("checkpointable", True),
        max_retries=job_data.get("retry", {}).get("max_attempts", 3),
    )

    job = Job(spec=spec)
    upsert_job(job)

    console.print(f"\n[bold cyan]Compute Pool — Job Submit[/bold cyan]")
    console.print(f"  Job ID   : [bold]{job.id}[/bold]")
    console.print(f"  Name     : {spec.name}")
    console.print(f"  GPU      : {'Yes' if spec.gpu else 'No'}  ({spec.gpu_memory_gb} GB)")
    console.print(f"  Max time : {spec.max_runtime_hours}h")

    job = schedule_job(job)

    console.print(f"\n  State    : [bold]{job.state.value}[/bold]")
    if job.assigned_slot:
        console.print(f"  Account  : Slot {job.assigned_slot} ({job.assigned_username})")
    if job.error:
        console.print(f"  [red]Error    : {job.error}[/red]")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# compute-pool job list
# ─────────────────────────────────────────────────────────────────────────────

@job_app.command("list")
def job_list():
    """List all submitted jobs."""
    from compute_pool.storage.local import load_all_jobs

    jobs = load_all_jobs()
    if not jobs:
        console.print("\n[dim]No jobs found. Submit one with: compute-pool job submit <spec.yaml>[/dim]\n")
        return

    console.print("\n[bold cyan]Compute Pool — Jobs[/bold cyan]\n")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Job ID", width=14)
    table.add_column("Name", width=20)
    table.add_column("State", width=14)
    table.add_column("Slot", width=6)
    table.add_column("Account", width=20)
    table.add_column("Submitted", width=22)

    STATE_COLORS = {
        "QUEUED": "yellow",
        "SCHEDULING": "cyan",
        "ASSIGNED": "blue",
        "RUNNING": "green",
        "COMPLETED": "bright_green",
        "FAILED": "red",
        "RETRYING": "magenta",
        "CANCELLED": "dim",
    }

    for j in reversed(jobs):
        color = STATE_COLORS.get(j.state.value, "white")
        table.add_row(
            j.id,
            j.spec.name,
            f"[{color}]{j.state.value}[/{color}]",
            str(j.assigned_slot or "-"),
            j.assigned_username or "-",
            j.created_at[:19].replace("T", " "),
        )

    console.print(table)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# compute-pool job status
# ─────────────────────────────────────────────────────────────────────────────

@job_app.command("status")
def job_status(
    job_id: str = typer.Argument(..., help="Job ID (e.g. job-a1b2c3d4)"),
):
    """Show full status for a specific job."""
    from compute_pool.storage.local import get_job

    job = get_job(job_id)
    if job is None:
        console.print(f"[red]Job not found:[/red] {job_id}")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Job: {job.id}[/bold cyan]")
    console.print(f"  Name          : {job.spec.name}")
    console.print(f"  State         : [bold]{job.state.value}[/bold]")
    console.print(f"  Slot          : {job.assigned_slot or 'unassigned'}")
    console.print(f"  Account       : {job.assigned_username or 'unassigned'}")
    console.print(f"  GPU           : {'Yes' if job.spec.gpu else 'No'} ({job.spec.gpu_memory_gb} GB)")
    console.print(f"  Checkpointable: {'Yes' if job.spec.checkpointable else 'No'}")
    console.print(f"  Retries       : {job.retry_count} / {job.spec.max_retries}")
    console.print(f"  Created       : {job.created_at}")
    console.print(f"  Updated       : {job.updated_at}")
    if job.error:
        console.print(f"  [red]Error         : {job.error}[/red]")
    console.print()


if __name__ == "__main__":
    app()
