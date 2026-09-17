"""
Interactive Remote GPU Shell for Compute Pool.

Boots an interactive terminal inside a live Kaggle Tesla T4 GPU container.
Provides direct Web Terminal access with full root bash, CUDA tools, and live GPU inspection.
"""
from __future__ import annotations

import json
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
from compute_pool.probe import _get_authenticated_api

console = Console()

SHELL_KERNEL_SLUG = "interactive-gpu-terminal"

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

print("[*] Setting up Compute Pool Interactive GPU Web Terminal...", flush=True)

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
    f.write("\\nexport PS1='\\\\[\\\\033[01;32m\\\\]compute-pool@gpu-worker\\\\[\\\\033[00m\\\\]:\\\\[\\\\033[01;34m\\\\]\\\\w\\\\[\\\\033[00m\\\\]\\\\$ '\\n")
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

sys.stdout.flush()

# Keep container alive for the session duration
for _ in range(DURATION_MINUTES * 12):
    time.sleep(5)
"""


def launch_gpu_shell(
    slot: int,
    duration_minutes: int = 120,
    open_web: bool = False,
    timeout_seconds: int = 240,
) -> dict[str, str]:
    """
    Launch an interactive GPU terminal session on the specified Kaggle slot.

    Returns dict with {"web": str, "kernel_ref": str}.
    """
    creds = load_credentials(slot)
    if creds is None:
        raise ValueError(
            f"No credentials configured for slot {slot}.\n"
            f"Run: compute-pool login --slot {slot}"
        )

    username = creds["username"]
    key = creds["key"]
    api = _get_authenticated_api(username, key)
    kernel_ref = f"{username}/{SHELL_KERNEL_SLUG}"
    session_id = f"cp-shell-{uuid.uuid4().hex[:12]}"

    console.print(f"\n[bold cyan]Booting Interactive GPU Terminal (Slot {slot}: {username})...[/bold cyan]")
    console.print("  [dim]Provisioning 2x Tesla T4 GPU worker with live Web Terminal...[/dim]")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        meta = {
            "id": kernel_ref,
            "title": SHELL_KERNEL_SLUG,
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
        )
        (tmp_path / "script.py").write_text(script)

        try:
            api.kernels_push(str(tmp_path), acc="nvidia-tesla-t4")
        except Exception as exc:
            raise RuntimeError(f"Failed to push shell kernel to Kaggle: {exc}") from exc

    console.print("  [dim]Worker queued on Kaggle GPU cluster. Waiting for tunnel connection...[/dim]\n")

    start_time = time.time()
    web_url = ""
    dots = 0

    while time.time() - start_time < timeout_seconds:
        # Check instant rendezvous point
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

    print()  # newline after status dots

    if not web_url:
        raise TimeoutError(
            f"Interactive terminal failed to establish tunnel connection within {timeout_seconds}s.\n"
            f"Check status on Kaggle: https://www.kaggle.com/code/{kernel_ref}"
        )

    # Display Rich interactive connection panel
    _display_shell_panel(slot, username, web_url, duration_minutes)

    if open_web:
        console.print(f"\n[green]* Opening Web Terminal in your default browser...[/green]")
        try:
            webbrowser.open(web_url)
        except Exception:
            pass

    return {
        "web": web_url,
        "kernel_ref": kernel_ref,
    }


def _display_shell_panel(
    slot: int,
    username: str,
    web_url: str,
    duration_minutes: int,
) -> None:
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
