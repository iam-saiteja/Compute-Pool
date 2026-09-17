"""
GPU probe for Compute Pool.

Uses official Kaggle API to push an nvidia-smi probe kernel,
polls until COMPLETE, downloads the real logs, and extracts GPU details.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console

from compute_pool.auth.kaggle_auth import load_credentials

console = Console()

DATA_DIR = Path("data")
GPU_CACHE_FILE = DATA_DIR / "gpu_info.json"
PROBE_SLUG = "compute-pool-gpu-probe"
PROBE_TITLE = "Compute Pool GPU Probe"

PROBE_CODE = """\
import subprocess
print("=== NVIDIA-SMI OUTPUT ===")
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print("NVIDIA_SMI_ERROR:", e)
"""


@dataclass
class GPUInfo:
    slot: int
    username: str
    probed_at: str
    gpu_name: str = "Unknown"
    vram_mb: int = 0
    gpu_count: int = 0
    driver_version: str = "Unknown"
    cuda_version: str = "Unknown"
    raw_smi: str = ""
    error: Optional[str] = None

    @property
    def vram_gb(self) -> float:
        return round(self.vram_mb / 1024.0, 1) if self.vram_mb else 0.0


def _get_authenticated_api(username: str, key: str):
    os.environ["KAGGLE_API_TOKEN"] = key
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def load_cached_gpu_info(slot: int) -> Optional[GPUInfo]:
    DATA_DIR.mkdir(exist_ok=True)
    if not GPU_CACHE_FILE.exists():
        return None
    try:
        all_info = json.loads(GPU_CACHE_FILE.read_text())
        d = all_info.get(str(slot))
        if d:
            return GPUInfo(**d)
    except Exception:
        pass
    return None


def _save_gpu_info(info: GPUInfo) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    all_info: dict = {}
    if GPU_CACHE_FILE.exists():
        try:
            all_info = json.loads(GPU_CACHE_FILE.read_text())
        except Exception:
            pass
    all_info[str(info.slot)] = asdict(info)
    GPU_CACHE_FILE.write_text(json.dumps(all_info, indent=2))


def _parse_smi_log(log_text: str, slot: int, username: str) -> GPUInfo:
    now = datetime.now(timezone.utc).isoformat()
    info = GPUInfo(slot=slot, username=username, probed_at=now, raw_smi=log_text)

    # Find driver and cuda version
    drv_m = re.search(r"Driver Version:\s*([0-9\.]+)", log_text)
    if drv_m:
        info.driver_version = drv_m.group(1)
    cuda_m = re.search(r"CUDA Version:\s*([0-9\.]+)", log_text)
    if cuda_m:
        info.cuda_version = cuda_m.group(1)

    # Find GPUs: e.g. "|   0  Tesla T4   Off | ... |   0MiB /  15360MiB |"
    # Match lines like: |   0  Tesla T4                       Off |
    gpu_matches = re.findall(r"\|\s+(\d+)\s+([A-Za-z0-9\s\-]+?)\s+(?:Off|On|\d+C|\d+W|P\d+)", log_text)
    if gpu_matches:
        info.gpu_count = len(gpu_matches)
        info.gpu_name = gpu_matches[0][1].strip()

    # Find VRAM: e.g. "0MiB /  15360MiB" or "15360MiB"
    vram_matches = re.findall(r"/\s*(\d+)MiB", log_text)
    if vram_matches:
        info.vram_mb = int(vram_matches[0])

    if not gpu_matches and not vram_matches:
        if "NO_GPU" in log_text or "NVIDIA_SMI_ERROR" in log_text or "No such file or directory: 'nvidia-smi'" in log_text:
            info.error = "GPU accelerator not active on account (Kaggle requires one-time phone verification at https://www.kaggle.com/settings to enable free GPU)"
        elif "No running processes" not in log_text and "NVIDIA-SMI" not in log_text:
            info.error = "No GPU detected on worker"
    return info


def run_probe(slot: int) -> GPUInfo:
    creds = load_credentials(slot)
    if creds is None:
        now = datetime.now(timezone.utc).isoformat()
        return GPUInfo(
            slot=slot,
            username="<not configured>",
            probed_at=now,
            error=f"No credentials for slot {slot}. Run: compute-pool login --slot {slot}",
        )

    username = creds["username"]
    key = creds["key"]

    try:
        api = _get_authenticated_api(username, key)
    except Exception as e:
        now = datetime.now(timezone.utc).isoformat()
        return GPUInfo(slot=slot, username=username, probed_at=now, error=f"Auth error: {e}")

    # Prepare temporary directory for pushing kernel
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        meta = {
            "id": f"{username}/{PROBE_SLUG}",
            "title": PROBE_TITLE,
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "false",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        (tmp_path / "script.py").write_text(PROBE_CODE)

        console.print(f"  [dim]Pushing GPU probe kernel to Kaggle (slot {slot}: {username})...[/dim]")
        try:
            api.kernels_push(str(tmp_path), acc="nvidia-tesla-t4")
        except Exception as e:
            now = datetime.now(timezone.utc).isoformat()
            return GPUInfo(slot=slot, username=username, probed_at=now, error=f"Push error: {e}")

    kernel_ref = f"{username}/{PROBE_SLUG}"
    console.print(f"  [dim]Kernel queued on Kaggle GPU. Polling status until complete...[/dim]")

    deadline = time.time() + 360
    dots = 0
    final_status = "UNKNOWN"
    while time.time() < deadline:
        try:
            st = api.kernels_status(kernel_ref)
            status_str = str(getattr(st, "status", st)).upper()
            print(f"\r  Status: {status_str} {'.' * (dots % 4)}    ", end="", flush=True)
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
        time.sleep(6)

    if final_status != "COMPLETE":
        now = datetime.now(timezone.utc).isoformat()
        info = GPUInfo(
            slot=slot,
            username=username,
            probed_at=now,
            error=f"Kernel status: {final_status} (did not complete successfully)",
        )
        _save_gpu_info(info)
        return info

    # Fetch output
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            api.kernels_output(kernel_ref, path=out_dir)
            log_files = list(Path(out_dir).glob("*.log"))
            log_text = ""
            if log_files:
                log_text = log_files[0].read_text(encoding="utf-8", errors="replace")
            else:
                for f in Path(out_dir).glob("*"):
                    log_text += f.read_text(encoding="utf-8", errors="replace") + "\n"

            # Parse JSON line stream
            extracted_lines = []
            for line in log_text.splitlines():
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
            clean_smi = "".join(extracted_lines) if extracted_lines else log_text

            info = _parse_smi_log(clean_smi, slot, username)
            _save_gpu_info(info)
            return info
        except Exception as e:
            now = datetime.now(timezone.utc).isoformat()
            info = GPUInfo(slot=slot, username=username, probed_at=now, error=f"Fetch output error: {e}")
            _save_gpu_info(info)
            return info
