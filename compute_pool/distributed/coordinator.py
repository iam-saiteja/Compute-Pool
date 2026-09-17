"""
Distributed Multi-Node Job Coordinator for Compute Pool.

Coordinates parallel multi-node workloads across both Kaggle account slots:
  - Node 0 (Rank 0): Slot 1 (saitejathanniru - Tesla T4 x2)
  - Node 1 (Rank 1): Slot 2 (thannirusahithya01 - Tesla T4 x2)
  Total: 4x Tesla T4 GPUs (~60 GB VRAM) training in parallel.

Uses ProcessPoolExecutor to guarantee complete process and credential isolation
between worker nodes.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from compute_pool.auth.kaggle_auth import load_credentials
from compute_pool.jobs.model import Job, JobSpec, JobState
from compute_pool.storage.local import upsert_job

console = Console()
DATA_DIR = Path("data") / "jobs"


def _sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\-]", "-", text.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:40]


def _run_single_node_proc(slot: int, rank: int, world_size: int, spec_dict: dict, job_id: str) -> dict:
    """Executes one node in a separate OS process with isolated credentials."""
    creds = load_credentials(slot)
    if not creds:
        return {"rank": rank, "slot": slot, "username": "unknown", "status": "FAILED", "error": f"No credentials for slot {slot}", "log": ""}

    username = creds["username"]
    os.environ["KAGGLE_API_TOKEN"] = creds["key"]
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = creds["key"]

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    kernel_slug = _sanitize_slug(f"{spec_dict['name']}-node{rank}-{job_id}")
    kernel_ref = f"{username}/{kernel_slug}"

    preamble = f"""\
import os
os.environ["CP_NODE_RANK"] = "{rank}"
os.environ["CP_WORLD_SIZE"] = "{world_size}"
os.environ["KAGGLE_USERNAME"] = "{username}"
"""
    full_script = preamble + "\n" + spec_dict["script"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        meta = {
            "id": kernel_ref,
            "title": kernel_slug,
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if spec_dict.get("gpu", True) else "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        (tmp_path / "script.py").write_text(full_script)

        try:
            acc_arg = "nvidia-tesla-t4" if spec_dict.get("gpu", True) else None
            push_resp = api.kernels_push(str(tmp_path), acc=acc_arg)
            if hasattr(push_resp, "ref") and push_resp.ref:
                kernel_ref = push_resp.ref.replace("/code/", "").lstrip("/")
            elif hasattr(push_resp, "url") and "kaggle.com/code/" in push_resp.url:
                kernel_ref = push_resp.url.split("kaggle.com/code/")[1]
        except Exception as e:
            return {"rank": rank, "slot": slot, "username": username, "status": "FAILED", "error": str(e), "log": ""}

    deadline = time.time() + max(360, int(spec_dict.get("max_runtime_hours", 1) * 3600))
    final_status = "UNKNOWN"

    while time.time() < deadline:
        try:
            st = api.kernels_status(kernel_ref)
            status_str = str(getattr(st, "status", st)).upper()
            if "COMPLETE" in status_str:
                final_status = "COMPLETE"
                break
            elif "ERROR" in status_str or "FAILED" in status_str:
                final_status = "ERROR"
                break
        except Exception:
            pass
        time.sleep(8)

    # Download output
    node_dir = DATA_DIR / job_id / f"node_{rank}"
    node_dir.mkdir(parents=True, exist_ok=True)
    log_text = ""
    try:
        api.kernels_output(kernel_ref, path=str(node_dir))
        for f in node_dir.glob("*.log"):
            raw_log = f.read_text(encoding="utf-8", errors="replace")
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
            log_text = "".join(extracted_lines) if extracted_lines else raw_log
    except Exception:
        pass

    return {
        "rank": rank,
        "slot": slot,
        "username": username,
        "kernel_ref": kernel_ref,
        "status": final_status,
        "log": log_text,
        "error": None if final_status == "COMPLETE" else f"Exited with status {final_status}",
    }


def run_distributed_job(spec_file: Path) -> None:
    """Run a distributed training workload concurrently on both Kaggle accounts."""
    import yaml

    raw = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
    job_data = raw.get("job", raw)

    spec = JobSpec(
        name=job_data.get("name", spec_file.stem),
        script=job_data.get("script", ""),
        gpu=job_data.get("resources", {}).get("gpu", True),
        gpu_memory_gb=float(job_data.get("resources", {}).get("gpu_memory_gb", 15)),
        max_runtime_hours=float(job_data.get("execution", {}).get("max_runtime_hours", 1)),
    )

    job = Job(spec=spec)
    job.transition(JobState.RUNNING)
    upsert_job(job)

    console.print(f"\n[bold cyan]Compute Pool -- Distributed Cluster Job ({job.id})[/bold cyan]")
    console.print(f"  Name        : {spec.name}")
    console.print(f"  Nodes       : 2 concurrent Kaggle GPU workers (4x Tesla T4 GPUs)")
    console.print(f"  Node 0      : Slot 1 (saitejathanniru)")
    console.print(f"  Node 1      : Slot 2 (thannirusahithya01)")
    console.print(f"\n  [dim]Dispatching tasks to both nodes in parallel (process-isolated)...[/dim]\n")

    spec_dict = {
        "name": spec.name,
        "script": spec.script,
        "gpu": spec.gpu,
        "max_runtime_hours": spec.max_runtime_hours,
    }

    start_t = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        f0 = executor.submit(_run_single_node_proc, 1, 0, 2, spec_dict, job.id)
        f1 = executor.submit(_run_single_node_proc, 2, 1, 2, spec_dict, job.id)

        dots = 0
        while not (f0.done() and f1.done()):
            s0 = "RUNNING" if not f0.done() else "DONE"
            s1 = "RUNNING" if not f1.done() else "DONE"
            print(f"\r  [Cluster Status] Node 0 (saitejathanniru): {s0} | Node 1 (thannirusahithya01): {s1} {'.' * (dots % 4)}    ", end="", flush=True)
            dots += 1
            time.sleep(5)
        print()

        res0 = f0.result()
        res1 = f1.result()

    elapsed = time.time() - start_t

    # Display results per node
    for res in [res0, res1]:
        title = f"Node {res['rank']} Output ({res['username']} - Slot {res['slot']})"
        color = "green" if res["status"] == "COMPLETE" else "red"
        clean_lines = []
        for line in res["log"].strip().splitlines():
            if "mistune.py" in line or "nbconvert" in line or "Writing" in line:
                continue
            clean_lines.append(line)
        console.print(Panel("\n".join(clean_lines[:35]), border_style=color, title=title))

    all_success = (res0["status"] == "COMPLETE" and res1["status"] == "COMPLETE")
    if all_success:
        job.transition(JobState.COMPLETED)
        console.print(f"\n[bold green]* Distributed Job {job.id} completed across both accounts in {elapsed:.1f}s![/bold green]\n")
        console.print(f"  Node 0 URL: https://www.kaggle.com/code/{res0['kernel_ref']}")
        console.print(f"  Node 1 URL: https://www.kaggle.com/code/{res1['kernel_ref']}\n")
    else:
        job.transition(JobState.FAILED, error=f"Node 0: {res0['status']}, Node 1: {res1['status']}")
        console.print(f"\n[bold red]x Distributed Job {job.id} failed on one or more nodes.[/bold red]\n")

    upsert_job(job)
