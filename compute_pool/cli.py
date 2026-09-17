"""
compute-pool CLI

Commands:
    compute-pool login --slot {1|2}         Authenticate a Kaggle account
    compute-pool accounts status            Show both account statuses + quota + cached GPU info
    compute-pool accounts probe --slot N    Push nvidia-smi kernel, show real GPU hardware
    compute-pool job submit <spec.yaml>     Submit a job (schedule + assign)
    compute-pool job list                   List all jobs
    compute-pool job status <job-id>        Show one job's full status
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
dist_app = typer.Typer(help="Distributed multi-node cluster commands.")
app.add_typer(accounts_app, name="accounts")
app.add_typer(job_app, name="job")
app.add_typer(dist_app, name="distributed")

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
    """Show live status, GPU quota, and cached GPU hardware for both accounts."""
    from rich.panel import Panel
    from rich.columns import Columns
    from compute_pool.accounts.manager import get_all_statuses
    from compute_pool.probe import load_cached_gpu_info

    console.print("\n[bold cyan]Compute Pool -- Account Status[/bold cyan]\n")
    statuses = get_all_statuses()

    panels = []
    total_left = 0.0

    for s in statuses:
        if s.connected:
            status_line = "[bold green]* Connected[/bold green]"
            total_left += s.gpu_hours_remaining
        else:
            status_line = "[bold red]x Disconnected[/bold red]"

        bar_filled = int((s.gpu_hours_remaining / (s.gpu_hours_total or 30.0)) * 20)
        bar = "[green]" + "#" * bar_filled + "[/green]" + "[dim]" + "-" * (20 - bar_filled) + "[/dim]"

        # Load cached GPU probe result
        gpu = load_cached_gpu_info(s.slot)
        if gpu and not gpu.error and gpu.gpu_name != "Unknown":
            gpu_line = f"[bold yellow]{gpu.gpu_name}[/bold yellow] x{gpu.gpu_count}  [cyan]{gpu.vram_gb} GB VRAM[/cyan]"
            driver_line = f"  Driver: {gpu.driver_version} | CUDA: {gpu.cuda_version}"
            probed_at = f"  (probed {gpu.probed_at[:10]})"
        else:
            gpu_line = "[dim]Not probed — run: compute-pool accounts probe --slot {}[/dim]".format(s.slot)
            driver_line = ""
            probed_at = ""

        lines = [
            f"[bold]{s.username}[/bold]   (slot {s.slot})",
            "",
            f"  Status      : {status_line}",
            f"  GPU HW      : {gpu_line}",
        ]
        if driver_line:
            lines.append(driver_line)
            lines.append(probed_at)
        lines += [
            "",
            f"  GPU-h used  : {s.gpu_hours_used:.2f}h",
            f"  GPU-h left  : [bold]{s.gpu_hours_remaining:.2f}h[/bold] / {s.gpu_hours_total:.1f}h",
            f"  Quota       : {bar}",
        ]
        if s.quota_refresh_time:
            lines.append(f"  Resets      : {s.quota_refresh_time[:10]}")
        if s.error:
            lines.append(f"\n  [red]{s.error}[/red]")

        color = "green" if s.connected else "red"
        panels.append(Panel("\n".join(lines), border_style=color, expand=True))

    console.print(Columns(panels, equal=True, expand=True))

    # Summary bar
    pool_bar_filled = int((min(total_left, 60.0) / 60.0) * 40)
    pool_bar = "[cyan]" + "#" * pool_bar_filled + "[/cyan]" + "[dim]" + "-" * (40 - pool_bar_filled) + "[/dim]"
    console.print(
        f"\n  [bold]Total pooled quota[/bold] : [bold cyan]{total_left:.2f}h[/bold cyan] GPU-hours available"
    )
    console.print(f"  {pool_bar}")
    console.print(
        "\n  [dim]Exact quota from Kaggle API. "
        "Probe real GPU via: compute-pool accounts probe --slot N[/dim]\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# compute-pool accounts probe
# ─────────────────────────────────────────────────────────────────────────────

@accounts_app.command("probe")
def accounts_probe(
    slot: int = typer.Option(..., "--slot", "-s", help="Account slot to probe (1 or 2)"),
):
    """
    Push a GPU probe kernel to Kaggle and display real hardware info (nvidia-smi).

    This provisions a real Kaggle GPU instance to inspect live hardware.
    Results are cached — run once per account.
    """
    from rich.panel import Panel
    from compute_pool.probe import run_probe

    if slot not in (1, 2):
        console.print("[red]Error:[/red] --slot must be 1 or 2.")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Compute Pool — GPU Probe (Slot {slot})[/bold cyan]")
    console.print("  Executing nvidia-smi kernel on Kaggle GPU worker...\n")

    info = run_probe(slot)

    if info.error:
        console.print(f"\n  [red]Probe failed:[/red] {info.error}\n")
        raise typer.Exit(1)

    lines = [
        f"[bold]{info.username}[/bold]   (slot {slot})",
        "",
        f"  GPU model    : [bold yellow]{info.gpu_name}[/bold yellow]",
        f"  GPU count    : {info.gpu_count}",
        f"  VRAM per GPU : [cyan]{info.vram_gb} GB[/cyan]  ({info.vram_mb} MiB)",
        f"  Driver       : {info.driver_version}",
        f"  CUDA Version : {info.cuda_version}",
        f"  Probed at    : {info.probed_at[:19].replace('T', ' ')} UTC",
    ]
    console.print(Panel("\n".join(lines), border_style="yellow", title="Live GPU Worker Hardware", expand=False))

    if info.raw_smi:
        console.print("\n  [dim]Raw nvidia-smi output:[/dim]")
        for line in info.raw_smi.splitlines()[:30]:
            console.print(f"  [dim]{line}[/dim]")
    console.print()




# ─────────────────────────────────────────────────────────────────────────────
# compute-pool job submit
# ─────────────────────────────────────────────────────────────────────────────

@job_app.command("submit")
def job_submit(
    spec_file: Path = typer.Argument(..., help="Path to job YAML spec file"),
    run: bool = typer.Option(False, "--run", "-r", help="Immediately dispatch and run on remote GPU"),
):
    """Submit a job to the compute pool (and optionally execute on GPU)."""
    import yaml
    from compute_pool.jobs.model import Job, JobSpec
    from compute_pool.scheduler.simple import schedule_job
    from compute_pool.storage.local import upsert_job
    from compute_pool.jobs.runner import run_job_remote

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

    if run and job.assigned_slot:
        run_job_remote(job.id)


# ─────────────────────────────────────────────────────────────────────────────
# compute-pool job run
# ─────────────────────────────────────────────────────────────────────────────

@job_app.command("run")
def job_run(
    job_id: str = typer.Argument(..., help="Job ID to run on remote Kaggle GPU"),
):
    """Run an assigned job on its remote Kaggle GPU worker."""
    from compute_pool.jobs.runner import run_job_remote
    try:
        run_job_remote(job_id)
    except Exception as e:
        console.print(f"[red]Error running job {job_id}:[/red] {e}")
        raise typer.Exit(1)



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

    console.print("\n[bold cyan]Compute Pool -- Jobs[/bold cyan]\n")
    from rich import box
    table = Table(show_header=True, header_style="bold magenta", expand=False, box=box.ASCII)
    table.add_column("Job ID", style="bold", no_wrap=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("State", min_width=10, no_wrap=True)
    table.add_column("Slot", justify="center", no_wrap=True)
    table.add_column("Account", no_wrap=True)
    table.add_column("Submitted", no_wrap=True)

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
        submitted = j.created_at[:16].replace("T", " ")   # "2026-09-17 09:36"
        table.add_row(
            j.id,
            j.spec.name,
            f"[{color}]{j.state.value}[/{color}]",
            str(j.assigned_slot or "-"),
            j.assigned_username or "-",
            submitted,
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


# ─────────────────────────────────────────────────────────────────────────────
# compute-pool job stop / cancel
# ─────────────────────────────────────────────────────────────────────────────

@job_app.command("stop")
@job_app.command("cancel")
def job_cancel(
    job_id: Optional[str] = typer.Argument(None, help="Job ID (e.g. job-a1b2c3d4)"),
    all_jobs: bool = typer.Option(False, "--all", "-a", help="Cancel all running/queued jobs"),
):
    """Cancel and terminate a running remote GPU job on Kaggle."""
    from compute_pool.jobs.runner import stop_remote_job
    from compute_pool.storage.local import load_all_jobs
    from compute_pool.jobs.model import JobState

    if all_jobs:
        jobs = load_all_jobs()
        active = [j for j in jobs if j.state in (JobState.RUNNING, JobState.ASSIGNED, JobState.QUEUED, JobState.SCHEDULING)]
        if not active:
            console.print("[dim]No active jobs found to cancel.[/dim]")
            return
        for j in active:
            stop_remote_job(j.id)
        return

    if not job_id:
        console.print("[red]Error:[/red] Please provide a job ID or use --all.")
        raise typer.Exit(1)

    try:
        stop_remote_job(job_id)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# compute-pool accounts stop
# ─────────────────────────────────────────────────────────────────────────────

@accounts_app.command("stop")
def accounts_stop(
    slot: Optional[int] = typer.Option(None, "--slot", "-s", help="Slot to stop (1, 2, or omitted for all)"),
):
    """Stop all active GPU sessions/kernels on account slot(s) and free quotas."""
    from compute_pool.shell import stop_gpu_shell
    from compute_pool.jobs.runner import stop_remote_job
    from compute_pool.storage.local import load_all_jobs
    from compute_pool.jobs.model import JobState

    # Stop any active shell sessions
    stop_gpu_shell(slot=slot)

    # Cancel active jobs on that slot
    jobs = load_all_jobs()
    for j in jobs:
        if j.state in (JobState.RUNNING, JobState.ASSIGNED, JobState.QUEUED, JobState.SCHEDULING):
            if slot is None or j.assigned_slot == slot:
                stop_remote_job(j.id)

    console.print(f"[bold green]* Account slot(s) idle and compute released.[/bold green]\n")


# ─────────────────────────────────────────────────────────────────────────────
# compute-pool distributed run
# ─────────────────────────────────────────────────────────────────────────────

@dist_app.command("run")
def distributed_run(
    spec_file: Path = typer.Argument(..., help="Path to distributed job YAML spec"),
):
    """Run a distributed training job across both Kaggle accounts in parallel."""
    from compute_pool.distributed.coordinator import run_distributed_job

    if not spec_file.exists():
        console.print(f"[red]Error:[/red] Spec file not found: {spec_file}")
        raise typer.Exit(1)

    try:
        run_distributed_job(spec_file)
    except Exception as e:
        console.print(f"[red]Distributed run error:[/red] {e}")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# compute-pool shell & shell-stop
# ─────────────────────────────────────────────────────────────────────────────

@app.command("shell")
def gpu_shell(
    slot: Optional[int] = typer.Option(None, "--slot", "-s", help="Account slot to launch GPU shell on (1 or 2, omit for both)"),
    all_slots: bool = typer.Option(False, "--all", "-a", help="Launch interactive terminals on BOTH accounts (4x Tesla T4 GPUs)"),
    duration: int = typer.Option(120, "--duration", "-d", help="Max session duration in minutes (default 120)"),
    web: bool = typer.Option(False, "--web", "-w", help="Automatically open Web Terminal(s) in default browser"),
):
    """Boot interactive remote terminal(s) inside live Tesla T4 GPU container(s) with root bash & CUDA."""
    from compute_pool.shell import launch_gpu_shell, launch_dual_gpu_shells

    try:
        if all_slots or slot is None:
            launch_dual_gpu_shells(duration_minutes=duration, open_web=web)
        else:
            if slot not in (1, 2):
                console.print("[red]Error:[/red] --slot must be 1 or 2.")
                raise typer.Exit(1)
            launch_gpu_shell(slot=slot, duration_minutes=duration, open_web=web)
    except (ValueError, RuntimeError, TimeoutError) as exc:
        console.print(f"[red]Shell error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("shell-all")
def gpu_shell_all(
    duration: int = typer.Option(120, "--duration", "-d", help="Max session duration in minutes (default 120)"),
    web: bool = typer.Option(False, "--web", "-w", help="Automatically open Web Terminals in default browser"),
):
    """Boot dual interactive remote terminals across BOTH accounts (4x Tesla T4 GPUs)."""
    from compute_pool.shell import launch_dual_gpu_shells
    try:
        launch_dual_gpu_shells(duration_minutes=duration, open_web=web)
    except (ValueError, RuntimeError, TimeoutError) as exc:
        console.print(f"[red]Shell error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("shell-stop")
def gpu_shell_stop(
    slot: Optional[int] = typer.Option(None, "--slot", "-s", help="Slot to stop (1, 2, or omitted for both)"),
):
    """Stop running interactive GPU shell sessions and immediately free GPUs."""
    from compute_pool.shell import stop_gpu_shell
    stop_gpu_shell(slot=slot)


if __name__ == "__main__":
    app()



