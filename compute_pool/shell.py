"""
Interactive Remote GPU Shell for Compute Pool.

Boots interactive terminals inside live Kaggle Tesla T4 GPU containers.
Supports single-slot and dual-node cluster shells with full root bash, CUDA tools & live GPU inspection.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import tempfile
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.panel import Panel

from compute_pool.auth.kaggle_auth import load_credentials
from compute_pool.jobs.model import Job, JobSpec, JobState
from compute_pool.probe import _get_authenticated_api
from compute_pool.storage.local import load_all_jobs, upsert_job

console = Console()

# Remote startup script executed inside the Kaggle GPU container
SHELL_BOOTSTRAP_TEMPLATE = """\
import os
import subprocess
import sys
import time
import re
import urllib.request

SESSION_ID = "__SESSION_ID__"
DURATION_MINUTES = __DURATION_MINUTES__
NODE_LABEL = "__NODE_LABEL__"

print(f"[*] Setting up Compute Pool Interactive GPU Web Terminal ({NODE_LABEL})...", flush=True)

# 1. Install ttyd (fast Web terminal server)
subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 -o /usr/local/bin/ttyd && chmod +x /usr/local/bin/ttyd"
], check=True)

# 2. Install cloudflared (HTTPS/WSS tunnel)
subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
], check=True)

# 3. Configure shell environment with aliases
with open(os.path.expanduser("~/.bashrc"), "a") as f:
    f.write(f"\\nexport PS1='\\\\[\\\\033[01;32m\\\\]compute-pool@{NODE_LABEL}\\\\[\\\\033[00m\\\\]:\\\\[\\\\033[01;34m\\\\]\\\\w\\\\[\\\\033[00m\\\\]\\\\$ '\\n")
    f.write("alias gpus='nvidia-smi'\\n")
    f.write("alias watch-gpu='watch -n 1 nvidia-smi'\\n")

# 4. Start ttyd with bash
ttyd_proc = subprocess.Popen(["/usr/local/bin/ttyd", "-W", "-p", "7681", "bash"])
time.sleep(1)

# 5. Start cloudflared tunnel
cf_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:7681", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

terminal_url = ""
for line in cf_proc.stdout:
    clean = line.strip()
    m = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", clean)
    if m:
        terminal_url = m.group(0)
        print("========================================", flush=True)
        print(f"WEB_TERMINAL: {terminal_url}", flush=True)
        print("========================================", flush=True)
        
        # Publish URL to rendezvous point for instant client pickup
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}", data=terminal_url.encode("utf-8"))
            urllib.request.urlopen(req, timeout=10)
            print("[*] Tunnel URL published to client.", flush=True)
        except Exception as exc:
            print("[!] Rendezvous publish failed:", exc, flush=True)
        break

# 6. Monitor bash shell and stop signals
# If user types 'exit' in terminal or stops session from CLI, terminate immediately
stop_url = f"https://ntfy.sh/{SESSION_ID}-stop/raw?poll=1"
for _ in range(int(DURATION_MINUTES * 60 / 3)):
    # Check if ttyd/bash exited
    if ttyd_proc.poll() is not None:
        print("[*] User exited shell session. Shutting down container...", flush=True)
        break
    
    # Check if stop was requested via ntfy signal
    try:
        req = urllib.request.Request(stop_url)
        with urllib.request.urlopen(req, timeout=2) as r:
            body = r.read().decode("utf-8").strip()
            if body == "STOP":
                print("[*] Received stop signal. Terminating container...", flush=True)
                break
    except Exception:
        pass

    time.sleep(3)

# Force cleanup
subprocess.run(["pkill", "-9", "-f", "cloudflared"], check=False)
subprocess.run(["pkill", "-9", "-f", "ttyd"], check=False)
sys.exit(0)
"""


def _get_shell_slug(slot: int) -> str:
    return f"interactive-gpu-terminal-s{slot}"


def _launch_single_slot_proc(slot: int, duration_minutes: int, session_id: str) -> dict:
    """Worker function executed in an isolated process/thread for slot authentication."""
    creds = load_credentials(slot)
    if not creds:
        return {"slot": slot, "username": "unknown", "status": "FAILED", "error": f"No credentials for slot {slot}"}

    username = creds["username"]
    os.environ["KAGGLE_API_TOKEN"] = creds["key"]
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = creds["key"]

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    kernel_slug = _get_shell_slug(slot)
    kernel_ref = f"{username}/{kernel_slug}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        meta = {
            "id": kernel_ref,
            "title": kernel_slug,
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        
        script = (
            SHELL_BOOTSTRAP_TEMPLATE
            .replace("__SESSION_ID__", session_id)
            .replace("__DURATION_MINUTES__", str(int(duration_minutes)))
            .replace("__NODE_LABEL__", f"node{slot-1}-slot{slot}")
        )
        (tmp_path / "script.py").write_text(script)

        last_err = None
        for attempt in range(3):
            try:
                resp = api.kernels_push(str(tmp_path), acc="nvidia-tesla-t4")
                if isinstance(resp, dict) and resp.get("error"):
                    err_msg = resp.get("error")
                    if "Maximum batch GPU session count" in str(err_msg):
                        # Force stop previous shell to free quota
                        stop_gpu_shell(slot=slot)
                        time.sleep(2)
                        continue
                    return {"slot": slot, "username": username, "status": "FAILED", "error": err_msg}
                return {"slot": slot, "username": username, "status": "QUEUED", "kernel_ref": kernel_ref, "error": None}
            except Exception as exc:
                last_err = exc
                if "409" in str(exc) or "Conflict" in str(exc):
                    time.sleep(2)
                    continue
                return {"slot": slot, "username": username, "status": "FAILED", "error": str(exc)}

        return {"slot": slot, "username": username, "status": "FAILED", "error": str(last_err)}


def launch_gpu_shell(
    slot: int,
    duration_minutes: int = 120,
    open_web: bool = False,
    timeout_seconds: int = 240,
) -> dict[str, str]:
    """Launch an interactive GPU terminal session on a specific slot (1 or 2)."""
    creds = load_credentials(slot)
    if creds is None:
        raise ValueError(
            f"No credentials configured for slot {slot}.\n"
            f"Run: compute-pool login --slot {slot}"
        )

    username = creds["username"]
    session_id = f"cp-shell-s{slot}-{uuid.uuid4().hex[:10]}"
    kernel_slug = _get_shell_slug(slot)

    # Register as an active running job in the state store
    shell_job = Job(
        id=f"job-shell-s{slot}",
        spec=JobSpec(
            name=f"interactive-shell-slot{slot}",
            script="ttyd + cloudflared web terminal",
            gpu=True,
            gpu_memory_gb=15,
        ),
        state=JobState.RUNNING,
        assigned_slot=slot,
        assigned_username=username,
        kaggle_kernel_slug=kernel_slug,
    )
    upsert_job(shell_job)

    console.print(f"\n[bold cyan]Booting Interactive GPU Terminal (Slot {slot}: {username})...[/bold cyan]")
    console.print("  [dim]Provisioning 2x Tesla T4 GPU worker with live Web Terminal...[/dim]")

    res = _launch_single_slot_proc(slot, duration_minutes, session_id)
    if res.get("status") == "FAILED":
        shell_job.transition(JobState.FAILED, error=res.get("error"))
        upsert_job(shell_job)
        raise RuntimeError(f"Failed to push shell kernel: {res.get('error')}")

    console.print("  [dim]Worker queued on Kaggle GPU cluster. Waiting for tunnel connection...[/dim]\n")

    start_time = time.time()
    web_url = ""
    dots = 0

    while time.time() - start_time < timeout_seconds:
        try:
            resp = httpx.get(f"https://ntfy.sh/{session_id}/raw?poll=1", timeout=5)
            if resp.status_code == 200 and resp.text.strip():
                for line in resp.text.splitlines():
                    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                    if m:
                        web_url = m.group(0).strip()
                        break
                if web_url:
                    break
        except Exception:
            pass

        print(f"\r  Connecting to live GPU terminal {'.' * (dots % 4 + 1)}    ", end="", flush=True)
        dots += 1
        time.sleep(3)

    print()

    if not web_url:
        shell_job.transition(JobState.FAILED, error="Connection timeout")
        upsert_job(shell_job)
        raise TimeoutError(
            f"Interactive terminal failed to establish tunnel connection within {timeout_seconds}s.\n"
            f"Check status on Kaggle: https://www.kaggle.com/code/{username}/{kernel_slug}"
        )

    _display_shell_panel(slot, username, web_url, duration_minutes)

    if open_web:
        console.print(f"\n[green]* Opening Web Terminal in your default browser...[/green]")
        try:
            webbrowser.open(web_url)
        except Exception:
            pass

    return {"web": web_url, "kernel_ref": f"{username}/{kernel_slug}"}


def launch_dual_gpu_shells(
    duration_minutes: int = 120,
    open_web: bool = False,
    timeout_seconds: int = 240,
) -> dict[str, dict]:
    """
    Launch interactive GPU terminals across BOTH accounts simultaneously.
    Provides 4x Tesla T4 GPUs across two independent live terminal tabs.
    """
    creds1 = load_credentials(1)
    creds2 = load_credentials(2)
    if not creds1 or not creds2:
        raise ValueError(
            "Both Slot 1 and Slot 2 must be configured for dual shell.\n"
            "Run: compute-pool login --slot 1 and compute-pool login --slot 2"
        )

    session1_id = f"cp-shell-s1-{uuid.uuid4().hex[:10]}"
    session2_id = f"cp-shell-s2-{uuid.uuid4().hex[:10]}"

    console.print("\n[bold cyan]Booting Dual-Node Interactive GPU Terminals (4x Tesla T4 GPUs)...[/bold cyan]")
    console.print(f"  Node 0 : Slot 1 ({creds1['username']}) - 2x Tesla T4 (15 GB each)")
    console.print(f"  Node 1 : Slot 2 ({creds2['username']}) - 2x Tesla T4 (15 GB each)")
    console.print("  [dim]Dispatching both terminal workers in parallel...[/dim]\n")

    # Register jobs
    upsert_job(Job(
        id="job-shell-s1",
        spec=JobSpec(name="interactive-shell-slot1", script="ttyd web terminal", gpu=True, gpu_memory_gb=15),
        state=JobState.RUNNING,
        assigned_slot=1,
        assigned_username=creds1["username"],
        kaggle_kernel_slug=_get_shell_slug(1),
    ))
    upsert_job(Job(
        id="job-shell-s2",
        spec=JobSpec(name="interactive-shell-slot2", script="ttyd web terminal", gpu=True, gpu_memory_gb=15),
        state=JobState.RUNNING,
        assigned_slot=2,
        assigned_username=creds2["username"],
        kaggle_kernel_slug=_get_shell_slug(2),
    ))

    # Push kernels sequentially to avoid KaggleApi env race condition
    res1 = _launch_single_slot_proc(1, duration_minutes, session1_id)
    res2 = _launch_single_slot_proc(2, duration_minutes, session2_id)

    if res1.get("status") == "FAILED" or res2.get("status") == "FAILED":
        err = f"Slot 1: {res1.get('error')} | Slot 2: {res2.get('error')}"
        raise RuntimeError(f"Failed to launch dual shells: {err}")

    console.print("  [dim]Both workers queued on GPU cluster. Waiting for tunnel connections...[/dim]\n")

    start_time = time.time()
    url1, url2 = "", ""
    dots = 0

    while time.time() - start_time < timeout_seconds:
        if not url1:
            try:
                r1 = httpx.get(f"https://ntfy.sh/{session1_id}/raw?poll=1", timeout=4)
                if r1.status_code == 200 and r1.text.strip():
                    m1 = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", r1.text)
                    if m1:
                        url1 = m1.group(0).strip()
            except Exception:
                pass

        if not url2:
            try:
                r2 = httpx.get(f"https://ntfy.sh/{session2_id}/raw?poll=1", timeout=4)
                if r2.status_code == 200 and r2.text.strip():
                    m2 = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", r2.text)
                    if m2:
                        url2 = m2.group(0).strip()
            except Exception:
                pass

        if url1 and url2:
            break

        print(f"\r  Connecting to dual GPU workers {'.' * (dots % 4 + 1)}    ", end="", flush=True)
        dots += 1
        time.sleep(3)

    print()

    if not url1 or not url2:
        raise TimeoutError("One or both GPU terminals timed out establishing tunnels.")

    # Display Dual Cluster Panel
    _display_dual_shell_panel(creds1["username"], creds2["username"], url1, url2, duration_minutes)

    if open_web:
        console.print("\n[green]* Opening both Web Terminals in your default browser...[/green]")
        try:
            webbrowser.open(url1)
            time.sleep(0.5)
            webbrowser.open(url2)
        except Exception:
            pass

    return {
        "node0": {"slot": 1, "username": creds1["username"], "web": url1},
        "node1": {"slot": 2, "username": creds2["username"], "web": url2},
    }


def _display_shell_panel(slot: int, username: str, web_url: str, duration_minutes: int) -> None:
    body = (
        f"[bold green]* GPU Worker Active & Connected[/bold green]\n\n"
        f"  [bold white]Account Slot:[/bold white]   Slot {slot} ({username})\n"
        f"  [bold white]Hardware:[/bold white]       2x Tesla T4 GPUs (15 GB VRAM each)\n"
        f"  [bold white]Max Duration:[/bold white]   {duration_minutes} minutes\n\n"
        f"  [bold yellow]Web Terminal URL:[/bold yellow]\n"
        f"  [bold underline cyan]{web_url}[/bold underline cyan]\n\n"
        f"  [dim]Click the URL above to access full root bash, CUDA drivers & nvidia-smi live.[/dim]"
    )
    console.print(
        Panel(
            body,
            title="[bold green]Compute Pool -- Live Interactive GPU Terminal[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


def _display_dual_shell_panel(user1: str, user2: str, url1: str, url2: str, duration_minutes: int) -> None:
    body = (
        f"[bold green]* Dual GPU Nodes Active & Connected (4x Tesla T4 GPUs / ~60 GB VRAM)[/bold green]\n\n"
        f"  [bold white]Max Duration:[/bold white]   {duration_minutes} minutes\n\n"
        f"+-- [bold yellow]Node 0: Slot 1 ({user1}) - 2x Tesla T4[/bold yellow] -------------------------+\n"
        f"  Web Terminal: [bold underline cyan]{url1}[/bold underline cyan]\n\n"
        f"+-- [bold yellow]Node 1: Slot 2 ({user2}) - 2x Tesla T4[/bold yellow] -------------------------+\n"
        f"  Web Terminal: [bold underline cyan]{url2}[/bold underline cyan]\n\n"
        f"  [dim]Both nodes have independent root bash environments with PyTorch & CUDA 13.0.[/dim]"
    )
    console.print(
        Panel(
            body,
            title="[bold green]Compute Pool -- Dual-Node Interactive Cluster Shells[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


def stop_gpu_shell(slot: int | None = None) -> None:
    """Stop active interactive GPU shell sessions on slot 1, slot 2, or both."""
    slots = [1, 2] if slot is None else [slot]
    for s in slots:
        creds = load_credentials(s)
        if not creds:
            continue
        username = creds["username"]
        key = creds["key"]
        try:
            api = _get_authenticated_api(username, key)
            shell_slugs = [_get_shell_slug(s), "interactive-gpu-terminal", "test-cf-terminal", "test-pinggy-terminal", "interactive-gpu-session"]
            for slug in shell_slugs:
                kernel_ref = f"{username}/{slug}"
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    meta = {
                        "id": kernel_ref,
                        "title": slug,
                        "code_file": "stop.py",
                        "language": "python",
                        "kernel_type": "script",
                        "is_private": "true",
                        "enable_gpu": "false",
                        "enable_tpu": "false",
                        "enable_internet": "false",
                        "dataset_sources": [],
                        "competition_sources": [],
                        "kernel_sources": [],
                        "model_sources": [],
                    }
                    (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
                    (tmp_path / "stop.py").write_text("import sys\nprint('Shell session stopped by user.')\nsys.exit(0)\n")
                    try:
                        api.kernels_push(str(tmp_path))
                    except Exception:
                        pass
            console.print(f"[green]* Slot {s} ({username}) GPU terminal stopped and GPU released.[/green]")
        except Exception as exc:
            console.print(f"[yellow]Notice for Slot {s}: {exc}[/yellow]")

        # Update job state in local store
        for j in load_all_jobs():
            if j.id in [f"job-shell-s{s}", f"job-shell-s1", f"job-shell-s2"] or (j.assigned_slot == s and "shell" in j.spec.name):
                if j.state == JobState.RUNNING:
                    j.transition(JobState.CANCELLED)
                    upsert_job(j)
