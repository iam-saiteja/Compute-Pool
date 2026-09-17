"""
Remote job runner for Compute Pool.

Dispatches jobs to their assigned Kaggle account as a GPU kernel session,
monitors execution live, and saves output logs and artifacts to data/jobs/<job-id>/.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from compute_pool.auth.kaggle_auth import load_credentials
from compute_pool.jobs.model import Job, JobState
from compute_pool.storage.local import upsert_job, get_job

console = Console()
DATA_DIR = Path("data") / "jobs"


def _get_api_for_slot(slot: int):
    creds = load_credentials(slot)
    if not creds:
        raise ValueError(f"Slot {slot} has no credentials configured.")
    os.environ["KAGGLE_API_TOKEN"] = creds["key"]
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api, creds["username"]


def _sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\-]", "-", text.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:50]


def run_job_remote(job_id: str) -> Job:
    job = get_job(job_id)
    if job is None:
        raise ValueError(f"Job {job_id} not found in state store.")

    if not job.assigned_slot:
        raise ValueError(f"Job {job_id} has not been scheduled to an account slot yet.")

    api, username = _get_api_for_slot(job.assigned_slot)
    kernel_slug = _sanitize_slug(f"{job.spec.name}-{job.id}")
    job.kaggle_kernel_slug = kernel_slug
    job.transition(JobState.RUNNING)
    upsert_job(job)

    console.print(f"\n[bold cyan]Dispatching {job.id} to Kaggle GPU ({username} - Slot {job.assigned_slot})...[/bold cyan]")

    # Prepare push package
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        meta = {
            "id": f"{username}/{kernel_slug}",
            "title": kernel_slug,
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if job.spec.gpu else "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        (tmp_path / "script.py").write_text(job.spec.script)

        try:
            acc_arg = "nvidia-tesla-t4" if job.spec.gpu else None
            api.kernels_push(str(tmp_path), acc=acc_arg)
            console.print(f"  [green]* Kernel pushed to Kaggle GPU worker.[/green]")
        except Exception as e:
            job.transition(JobState.FAILED, error=f"Push failed: {e}")
            upsert_job(job)
            console.print(f"  [red]Push failed: {e}[/red]")
            return job

    kernel_ref = f"{username}/{kernel_slug}"
    console.print(f"  [dim]Running on remote Kaggle GPU worker...[/dim]")

    max_runtime_sec = int(job.spec.max_runtime_hours * 3600)
    deadline = time.time() + max(360, max_runtime_sec)
    dots = 0
    final_status = "UNKNOWN"

    while time.time() < deadline:
        try:
            st = api.kernels_status(kernel_ref)
            status_str = str(getattr(st, "status", st)).upper()
            print(f"\r  Worker status: {status_str} {'.' * (dots % 4)}    ", end="", flush=True)
            dots += 1
            if "COMPLETE" in status_str:
                print()
                final_status = "COMPLETE"
                break
            elif "ERROR" in status_str or "FAILED" in status_str:
                print()
                final_status = "ERROR"
                break
            elif "CANCEL" in status_str:
                print()
                final_status = "CANCELLED"
                break
        except Exception:
            pass
        time.sleep(8)

    job_dir = DATA_DIR / job.id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        api.kernels_output(kernel_ref, path=str(job_dir))
        console.print(f"  [green]* Output & artifacts downloaded to: {job_dir}[/green]")
    except Exception as e:
        console.print(f"  [yellow]Notice: Could not download output files: {e}[/yellow]")

    # Display stdout log
    log_files = list(job_dir.glob("*.log"))
    if log_files:
        raw_log = log_files[0].read_text(encoding="utf-8", errors="replace")
        extracted_lines = []
        for line in raw_log.splitlines():
            clean = line.strip().lstrip(",")
            if clean.startswith("{") and clean.endswith("}"):
                try:
                    item = json.loads(clean)
                    if "data" in item:
                        extracted_lines.append(item["data"])
                except Exception:
                    extracted_lines.append(line)
            else:
                extracted_lines.append(line)
        clean_text = "".join(extracted_lines) if extracted_lines else raw_log

        console.print(f"\n[bold]--- Remote Execution Output ({job.id}) ---[/bold]")
        for line in clean_text.strip().splitlines():
            console.print(f"  {line}")
        console.print(f"[bold]-----------------------------------------[/bold]\n")

    if final_status == "COMPLETE":
        job.transition(JobState.COMPLETED)
        console.print(f"[bold green]* Job {job.id} completed successfully on GPU![/bold green]\n")
    else:
        job.transition(JobState.FAILED, error=f"Execution ended with status: {final_status}")
        console.print(f"[bold red]x Job {job.id} failed ({final_status})[/bold red]\n")

    upsert_job(job)
    return job
