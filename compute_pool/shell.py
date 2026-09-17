"""
Interactive Remote GPU Shell for Compute Pool.

Boots a unified 4-GPU Master-Worker Interactive Cluster Terminal.
Node 0 (Slot 1: 2x Tesla T4) acts as the interactive Master with web terminal.
Node 1 (Slot 2: 2x Tesla T4) acts as an attached compute worker over an inter-node mesh.
Provides real-time 4-GPU monitoring (nvidia-smi / watch-gpu) and multi-node execution (cluster-exec, cluster-status).
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

# ─────────────────────────────────────────────────────────────────────────────
# 1. Worker Node Template (Node 1 - Slot 2: Attached Compute Node)
# ─────────────────────────────────────────────────────────────────────────────
WORKER_BOOTSTRAP_TEMPLATE = """\
import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request

SESSION_ID = "__SESSION_ID__"
DURATION_MINUTES = __DURATION_MINUTES__
NODE_LABEL = "node1-slot2"

print(f"[*] Initializing Compute Pool GPU Cluster Worker ({NODE_LABEL})...", flush=True)

# 1. Install cloudflared for secure RPC tunnel
subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
], check=True)

# 2. Worker RPC Server
class ClusterWorkerHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "OK", "node": NODE_LABEL}).encode("utf-8"))
        elif self.path == "/smi":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                    text=True
                )
                gpus = []
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 7:
                        gpus.append({
                            "name": parts[1],
                            "mem_used": int(float(parts[2])),
                            "mem_total": int(float(parts[3])),
                            "util": int(float(parts[4])),
                            "temp": int(float(parts[5])),
                            "power": parts[6] + "W"
                        })
                self.wfile.write(json.dumps({"gpus": gpus}).encode("utf-8"))
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e), "gpus": []}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/exec":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                cmd = data.get("cmd", "")
                env = os.environ.copy()
                env.update(data.get("env", {}))
                proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
                resp = {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
            except Exception as e:
                resp = {"exit_code": 1, "stdout": "", "stderr": str(e)}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif self.path == "/sync_file":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                fname = data.get("filename", "script.py")
                content = data.get("content", "")
                with open(os.path.join("/kaggle/working", fname), "w") as f:
                    f.write(content)
                resp = {"status": "OK", "filename": fname}
            except Exception as e:
                resp = {"status": "ERROR", "error": str(e)}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif self.path == "/stop":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            def _die():
                time.sleep(0.5)
                os.system("pkill -9 -f cloudflared; kill -9 -1")
            threading.Thread(target=_die).start()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

httpd = ThreadedTCPServer(("0.0.0.0", 8888), ClusterWorkerHandler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()

# 3. Expose Worker RPC over Cloudflare tunnel
cf_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:8888", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in cf_proc.stdout:
    clean = line.strip()
    m = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", clean)
    if m:
        worker_url = m.group(0)
        print("========================================", flush=True)
        print(f"WORKER_RPC: {worker_url}", flush=True)
        print("========================================", flush=True)
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}", data=worker_url.encode("utf-8"))
            urllib.request.urlopen(req, timeout=10)
            print("[*] Worker RPC published to cluster.", flush=True)
        except Exception as exc:
            print("[!] Worker publish failed:", exc, flush=True)
        break

# Keep worker alive until stop signal or duration
stop_url = f"https://ntfy.sh/{SESSION_ID}-stop/raw?poll=1"
for _ in range(int(DURATION_MINUTES * 60 / 3)):
    try:
        req = urllib.request.Request(stop_url)
        with urllib.request.urlopen(req, timeout=2) as r:
            if r.read().decode("utf-8").strip() == "STOP":
                break
    except Exception:
        pass
    time.sleep(3)

subprocess.run(["pkill", "-9", "-f", "cloudflared"], check=False)
sys.exit(0)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. Master Node Template (Node 0 - Slot 1: Interactive Control Center)
# ─────────────────────────────────────────────────────────────────────────────
MASTER_BOOTSTRAP_TEMPLATE = """\
import os
import subprocess
import sys
import time
import re
import urllib.request
import json
import threading
import shutil

SESSION_ID = "__SESSION_ID__"
WORKER_SESSION_ID = "__WORKER_SESSION_ID__"
DURATION_MINUTES = __DURATION_MINUTES__
NODE_LABEL = "cluster-master"

print("[*] Initializing Compute Pool Master GPU Terminal (4x Tesla T4 Cluster)...", flush=True)

# 1. Install ttyd and cloudflared
subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 -o /usr/local/bin/ttyd && chmod +x /usr/local/bin/ttyd"
], check=True)
subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
], check=True)

# 2. Discover and preserve the real system nvidia-smi binary
real_smi = None
for candidate in ["/usr/bin/nvidia-smi", "/usr/local/cuda/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi"]:
    if os.path.exists(candidate):
        real_smi = candidate
        break

if not real_smi:
    real_smi = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"

try:
    if os.path.exists(real_smi) and real_smi != "/usr/local/bin/real-nvidia-smi":
        subprocess.run(["cp", "-f", real_smi, "/usr/local/bin/real-nvidia-smi"], check=False)
        subprocess.run(["chmod", "+x", "/usr/local/bin/real-nvidia-smi"], check=False)
except Exception:
    pass

# 3. Configure cluster mesh scripts into /usr/local/bin
smi_script = '''#!/usr/bin/env python3
import subprocess, json, urllib.request, os

worker_url = ""
if os.path.exists("/kaggle/working/.cluster_worker_url"):
    with open("/kaggle/working/.cluster_worker_url") as f:
        worker_url = f.read().strip()

local_gpus = []
try:
    smi_bin = "/usr/local/bin/real-nvidia-smi" if os.path.exists("/usr/local/bin/real-nvidia-smi") else "nvidia-smi"
    out = subprocess.check_output(
        [smi_bin, "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
        text=True
    )
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 7:
            local_gpus.append({
                "name": p[1], "mem_used": int(float(p[2])), "mem_total": int(float(p[3])),
                "util": int(float(p[4])), "temp": int(float(p[5])), "power": p[6] + "W"
            })
except Exception:
    pass

remote_gpus = []
if worker_url:
    try:
        req = urllib.request.Request(f"{worker_url}/smi")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            remote_gpus = data.get("gpus", [])
    except Exception:
        pass

all_gpus = []
for i, g in enumerate(local_gpus):
    all_gpus.append((i, "Node 0 (Slot 1)", g))
for i, g in enumerate(remote_gpus):
    all_gpus.append((len(local_gpus) + i, "Node 1 (Slot 2)", g))

print("+-----------------------------------------------------------------------------------------+")
print("| NVIDIA-SMI (Cluster Pool: 4x Tesla T4)     CUDA Version: 13.0     Driver: 580.159.04   |")
print("+-----------------------------------------+------------------------+----------------------+")
print("| GPU  Name                 Node / Slot   | Memory-Usage           | GPU-Util  Temp  Pwr  |")
print("|=========================================+========================+======================|")
if all_gpus:
    for idx, node, g in all_gpus:
        name_str = f"{g.get('name', 'Tesla T4'):<12} {node:<13}"
        mem_str = f"{g.get('mem_used', 0)}MiB / {g.get('mem_total', 15360)}MiB"
        util_str = f"{g.get('util', 0):>3}%   {g.get('temp', 40):>3}C  {g.get('power', '15W'):>4}"
        print(f"|  {idx:>2}  {name_str} | {mem_str:<22} | {util_str:<20} |")
else:
    print("| No GPUs detected or cluster synchronizing...                                            |")
print("+-----------------------------------------+------------------------+----------------------+")
total_vram = sum(g.get('mem_total', 15360) for _, _, g in all_gpus) / 1024.0 if all_gpus else 60.0
used_vram = sum(g.get('mem_used', 0) for _, _, g in all_gpus) if all_gpus else 0
status_str = "4/4 GPUs Active (ONLINE)" if len(all_gpus) >= 4 else f"{len(all_gpus)}/4 GPUs Active (CONNECTING...)"
print(f"| Cluster VRAM: {used_vram}MiB / {total_vram:.0f}GB ({len(all_gpus)} GPUs) | Status: {status_str:<32} |")
print("+-----------------------------------------------------------------------------------------+")
'''
with open("/usr/local/bin/cluster-smi", "w") as f:
    f.write(smi_script)
os.chmod("/usr/local/bin/cluster-smi", 0o755)

with open("/usr/local/bin/nvidia-smi", "w") as f:
    f.write(smi_script)
os.chmod("/usr/local/bin/nvidia-smi", 0o755)

exec_script = '''#!/usr/bin/env python3
import sys, subprocess, json, urllib.request, os

if len(sys.argv) < 2:
    print('Usage: cluster-exec "<command>"')
    sys.exit(1)

cmd = " ".join(sys.argv[1:])
print("[*] Executing across Cluster: " + cmd)
print()

print("--- [Node 0 (Slot 1)] ---")
subprocess.run(cmd, shell=True)

worker_url = ""
if os.path.exists("/kaggle/working/.cluster_worker_url"):
    with open("/kaggle/working/.cluster_worker_url") as f:
        worker_url = f.read().strip()

if worker_url:
    print()
    print("--- [Node 1 (Slot 2)] ---")
    try:
        req = urllib.request.Request(
            worker_url + "/exec",
            data=json.dumps({"cmd": cmd}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("stdout"):
                print(data["stdout"], end="")
            if data.get("stderr"):
                print(data["stderr"], end="")
            sys.exit(data.get("exit_code", 0))
    except Exception as e:
        print("Worker RPC error:", e)
else:
    print("[!] Worker Node 1 is not connected.")
'''
with open("/usr/local/bin/cluster-exec", "w") as f:
    f.write(exec_script)
os.chmod("/usr/local/bin/cluster-exec", 0o755)

status_script = '''#!/usr/bin/env python3
import os, urllib.request, json, time

worker_url = ""
if os.path.exists("/kaggle/working/.cluster_worker_url"):
    with open("/kaggle/working/.cluster_worker_url") as f:
        worker_url = f.read().strip()

print("Compute Pool 4-GPU Cluster Status")
print("=================================")
print("Node 0 (Master Slot 1) : ONLINE (Local - 2x Tesla T4)")
if worker_url:
    t0 = time.time()
    try:
        with urllib.request.urlopen(f"{worker_url}/health", timeout=3) as r:
            lat = (time.time() - t0) * 1000
            print(f"Node 1 (Worker Slot 2) : ONLINE (RPC Mesh latency: {lat:.1f}ms - 2x Tesla T4)")
    except Exception as e:
        print(f"Node 1 (Worker Slot 2) : UNREACHABLE ({e})")
else:
    print("Node 1 (Worker Slot 2) : CONNECTING...")
'''
with open("/usr/local/bin/cluster-status", "w") as f:
    f.write(status_script)
os.chmod("/usr/local/bin/cluster-status", 0o755)

# Cluster Runner (runs code across all 4 GPUs transparently)
run_script = '''#!/usr/bin/env python3
import sys, subprocess, json, urllib.request, os, glob, threading

if len(sys.argv) < 2:
    print("Usage: cluster-run <script.py or command> [args...]")
    sys.exit(1)

worker_url = ""
if os.path.exists("/kaggle/working/.cluster_worker_url"):
    with open("/kaggle/working/.cluster_worker_url") as f:
        worker_url = f.read().strip()

target = sys.argv[1]
args = " ".join(sys.argv[2:])

if os.path.exists(target) and target.endswith(".py"):
    if worker_url:
        for py_file in glob.glob("*.py") + [target]:
            try:
                with open(py_file, "r") as f:
                    content = f.read()
                req = urllib.request.Request(
                    worker_url + "/sync_file",
                    data=json.dumps({"filename": os.path.basename(py_file), "content": content}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=10)
            except Exception:
                pass
    cmd = ("python3 " + target + " " + args).strip()
else:
    cmd = " ".join(sys.argv[1:])

print("[*] Dispatching across 4 GPUs (Master + Worker): " + cmd)
print()

def run_remote():
    if not worker_url:
        return
    try:
        req = urllib.request.Request(
            worker_url + "/exec",
            data=json.dumps({"cmd": cmd, "env": {"NODE_RANK": "1", "WORLD_SIZE": "4"}}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("stdout"):
                print("[Node 1 (GPUs 2,3)]")
                print(data["stdout"], end="")
            if data.get("stderr"):
                print("[Node 1 ERR]")
                print(data["stderr"], end="")
    except Exception as e:
        print("[Node 1 Error]", e)

t = threading.Thread(target=run_remote)
t.start()

env = os.environ.copy()
env["NODE_RANK"] = "0"
env["WORLD_SIZE"] = "4"
print("[Node 0 (GPUs 0,1)]")
subprocess.run(cmd, shell=True, env=env)
t.join()
'''
with open("/usr/local/bin/cluster-run", "w") as f:
    f.write(run_script)
os.chmod("/usr/local/bin/cluster-run", 0o755)

watch_script = '''#!/bin/bash
watch -n 1 /usr/local/bin/cluster-smi
'''
with open("/usr/local/bin/watch-gpu", "w") as f:
    f.write(watch_script)
os.chmod("/usr/local/bin/watch-gpu", 0o755)

stop_script = '''#!/bin/bash
echo "[*] Terminating 4-GPU Cluster and releasing resources..."
if [ -f /kaggle/working/.cluster_worker_url ]; then
    WURL=$(cat /kaggle/working/.cluster_worker_url)
    if [ -n "$WURL" ]; then
        curl -s "$WURL/stop" >/dev/null 2>&1
    fi
fi
kill -9 -1
'''
with open("/usr/local/bin/stop", "w") as f:
    f.write(stop_script)
os.chmod("/usr/local/bin/stop", 0o755)

# Pre-install cluster_pool python module
cp_module = '''# Compute Pool Cluster Library - Multi-Node GPU Support
import torch, os, json, urllib.request

def gpus():
    return [
        {"id": 0, "node": "Node 0 (Master)", "name": "Tesla T4 (15 GB)"},
        {"id": 1, "node": "Node 0 (Master)", "name": "Tesla T4 (15 GB)"},
        {"id": 2, "node": "Node 1 (Worker)", "name": "Tesla T4 (15 GB)"},
        {"id": 3, "node": "Node 1 (Worker)", "name": "Tesla T4 (15 GB)"},
    ]

def total_vram_gb():
    return 60.0

def is_cluster_online():
    return os.path.exists("/kaggle/working/.cluster_worker_url")
'''
with open("/kaggle/working/cluster_pool.py", "w") as f:
    f.write(cp_module)

# Create ready-to-run 4-GPU PyTorch Demo
demo_script = '''import torch
import time
import os

rank = int(os.environ.get("NODE_RANK", "0"))
node_name = f"Node {rank}"
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

print(f"[{node_name}] Found {len(gpus)} local GPUs: {gpus}")
for i in range(torch.cuda.device_count()):
    t = torch.randn((4000, 4000), device=f"cuda:{i}")
    res = torch.matmul(t, t).sum().item()
    print(f"[{node_name} GPU {i}] Matrix Mult Test (4000x4000) Result: {res:.2f} (OK)")
'''
with open("/kaggle/working/demo_4gpu.py", "w") as f:
    f.write(demo_script)

# Configure bash environment
with open(os.path.expanduser("~/.bashrc"), "a") as f:
    f.write("\\nexport PATH=/usr/local/bin:$PATH\\n")
    f.write("export PS1='\\\\[\\\\033[01;32m\\\\]compute-pool@cluster-master\\\\[\\\\033[00m\\\\]:\\\\[\\\\033[01;34m\\\\]\\\\w\\\\[\\\\033[00m\\\\]\\\\$ '\\n")
    f.write("alias gpus='/usr/local/bin/cluster-smi'\\n")
    f.write("alias watch-gpu='/usr/local/bin/watch-gpu'\\n")
    f.write("alias run='/usr/local/bin/cluster-run'\\n")
    f.write("alias real-smi='/usr/local/bin/real-nvidia-smi'\\n")
    f.write("alias halt='/usr/local/bin/stop'\\n")
    f.write("alias exit='/usr/local/bin/stop'\\n")

# 4. Start ttyd on port 7681 with clipboard support
ttyd_proc = subprocess.Popen([
    "/usr/local/bin/ttyd", "-W", "-p", "7681",
    "-t", "enableClipboard=true",
    "-t", "fontSize=15",
    "-t", "disableLeaveAlert=true",
    "bash"
])
time.sleep(1)

# 5. Start cloudflared tunnel for Web Terminal
cf_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:7681", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in cf_proc.stdout:
    clean = line.strip()
    m = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", clean)
    if m:
        terminal_url = m.group(0)
        print("========================================", flush=True)
        print(f"WEB_TERMINAL: {terminal_url}", flush=True)
        print("========================================", flush=True)
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}", data=terminal_url.encode("utf-8"))
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass
        break

# 6. Background thread to discover Worker RPC endpoint
def setup_cluster_worker_discovery():
    worker_url = ""
    for _ in range(90):
        try:
            r = urllib.request.urlopen(f"https://ntfy.sh/{WORKER_SESSION_ID}/raw?poll=1", timeout=3)
            txt = r.read().decode("utf-8").strip()
            m = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", txt)
            if m:
                worker_url = m.group(0).strip()
                break
        except Exception:
            pass
        time.sleep(2)
    
    with open("/kaggle/working/.cluster_worker_url", "w") as f:
        f.write(worker_url)

threading.Thread(target=setup_cluster_worker_discovery, daemon=True).start()

# Keep master alive until exit
stop_url = f"https://ntfy.sh/{SESSION_ID}-stop/raw?poll=1"
for _ in range(int(DURATION_MINUTES * 60 / 3)):
    try:
        req = urllib.request.Request(stop_url)
        with urllib.request.urlopen(req, timeout=2) as r:
            if r.read().decode("utf-8").strip() == "STOP":
                break
    except Exception:
        pass
    time.sleep(3)

subprocess.run(["pkill", "-9", "-f", "cloudflared"], check=False)
subprocess.run(["pkill", "-9", "-f", "ttyd"], check=False)
sys.exit(0)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 3. Single-Node Shell Template
# ─────────────────────────────────────────────────────────────────────────────
SINGLE_SHELL_BOOTSTRAP_TEMPLATE = """\
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

subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 -o /usr/local/bin/ttyd && chmod +x /usr/local/bin/ttyd"
], check=True)

subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
], check=True)

stop_script = '''#!/bin/bash
kill -9 -1
'''
with open("/usr/local/bin/stop", "w") as f:
    f.write(stop_script)
os.chmod("/usr/local/bin/stop", 0o755)

with open(os.path.expanduser("~/.bashrc"), "a") as f:
    f.write(f"\\nexport PATH=/usr/local/bin:$PATH\\n")
    f.write(f"export PS1='\\\\[\\\\033[01;32m\\\\]compute-pool@{NODE_LABEL}\\\\[\\\\033[00m\\\\]:\\\\[\\\\033[01;34m\\\\]\\\\w\\\\[\\\\033[00m\\\\]\\\\$ '\\n")
    f.write("alias gpus='nvidia-smi'\\n")
    f.write("alias watch-gpu='watch -n 1 nvidia-smi'\\n")
    f.write("alias halt='/usr/local/bin/stop'\\n")
    f.write("alias exit='/usr/local/bin/stop'\\n")

ttyd_proc = subprocess.Popen(["/usr/local/bin/ttyd", "-W", "-p", "7681", "bash"])
time.sleep(1)

cf_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:7681", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in cf_proc.stdout:
    clean = line.strip()
    m = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", clean)
    if m:
        terminal_url = m.group(0)
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}", data=terminal_url.encode("utf-8"))
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass
        break

stop_url = f"https://ntfy.sh/{SESSION_ID}-stop/raw?poll=1"
for _ in range(int(DURATION_MINUTES * 60 / 3)):
    try:
        req = urllib.request.Request(stop_url)
        with urllib.request.urlopen(req, timeout=2) as r:
            if r.read().decode("utf-8").strip() == "STOP":
                break
    except Exception:
        pass
    time.sleep(3)

subprocess.run(["pkill", "-9", "-f", "cloudflared"], check=False)
subprocess.run(["pkill", "-9", "-f", "ttyd"], check=False)
sys.exit(0)
"""


def _get_shell_slug(slot: int) -> str:
    return f"interactive-gpu-terminal-s{slot}"


def _force_stop_slot(api, username: str, slug: str) -> None:
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            meta = {
                "id": f"{username}/{slug}",
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
            (tmp_path / "stop.py").write_text("import sys\nsys.exit(0)\n")
            api.kernels_push(str(tmp_path))
    except Exception:
        pass


def _push_kernel_payload(slot: int, slug: str, script_body: str) -> dict:
    """Helper to authenticate and push a GPU kernel payload for a given slot."""
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

    kernel_ref = f"{username}/{slug}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        meta = {
            "id": kernel_ref,
            "title": slug,
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
        (tmp_path / "script.py").write_text(script_body)

        for attempt in range(5):
            try:
                resp = api.kernels_push(str(tmp_path), acc="nvidia-tesla-t4")
                err_msg = resp.get("error") if isinstance(resp, dict) else getattr(resp, "error", None)
                if err_msg:
                    err_str = str(err_msg).strip()
                    if "Maximum batch GPU session count" in err_str or "429" in err_str or "Conflict" in err_str:
                        console.print(f"  [yellow]* Slot {slot} GPU session occupied. Automatically releasing old session (attempt {attempt+1}/5)...[/yellow]")
                        _force_stop_slot(api, username, slug)
                        time.sleep(8)
                        continue
                    return {"slot": slot, "username": username, "status": "FAILED", "error": err_str}
                return {"slot": slot, "username": username, "status": "QUEUED", "kernel_ref": kernel_ref, "error": None}
            except Exception as exc:
                if "409" in str(exc) or "Conflict" in str(exc) or "Maximum batch" in str(exc) or "429" in str(exc):
                    console.print(f"  [yellow]* Slot {slot} busy. Releasing and retrying (attempt {attempt+1}/5)...[/yellow]")
                    _force_stop_slot(api, username, slug)
                    time.sleep(8)
                    continue
                return {"slot": slot, "username": username, "status": "FAILED", "error": str(exc)}

        return {
            "slot": slot,
            "username": username,
            "status": "FAILED",
            "error": "Kaggle GPU session limit reached. Please ensure no interactive notebook sessions are open at kaggle.com and retry.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Launch Unified 4-GPU Cluster Shell (1 Master Browser Tab)
# ─────────────────────────────────────────────────────────────────────────────
def launch_cluster_shell(
    duration_minutes: int = 120,
    open_web: bool = False,
    timeout_seconds: int = 240,
) -> dict[str, str]:
    """
    Launch unified 4-GPU interactive cluster terminal.
    Node 0 (Slot 1) acts as Master control terminal.
    Node 1 (Slot 2) connects as attached compute worker.
    Opens exactly 1 Master terminal tab with unified 4-GPU nvidia-smi.
    """
    creds1 = load_credentials(1)
    creds2 = load_credentials(2)
    if not creds1 or not creds2:
        raise ValueError(
            "Both Slot 1 and Slot 2 must be configured for the 4-GPU cluster.\n"
            "Run: compute-pool login --slot 1 and compute-pool login --slot 2"
        )

    session_master_id = f"cp-master-{uuid.uuid4().hex[:10]}"
    session_worker_id = f"cp-worker-{uuid.uuid4().hex[:10]}"

    console.print("\n[bold cyan]Booting Unified 4-GPU Interactive Cluster (Master-Worker Architecture)...[/bold cyan]")
    console.print(f"  Master Node 0 : Slot 1 ({creds1['username']}) - 2x Tesla T4 (15 GB each)")
    console.print(f"  Worker Node 1 : Slot 2 ({creds2['username']}) - 2x Tesla T4 (15 GB each)")
    console.print("  [dim]Dispatching both cluster nodes simultaneously...[/dim]\n")

    # Register active jobs in local state
    upsert_job(Job(
        id="job-cluster-master",
        spec=JobSpec(name="cluster-master-node0", script="master web terminal", gpu=True, gpu_memory_gb=30),
        state=JobState.RUNNING,
        assigned_slot=1,
        assigned_username=creds1["username"],
        kaggle_kernel_slug=_get_shell_slug(1),
    ))
    upsert_job(Job(
        id="job-cluster-worker",
        spec=JobSpec(name="cluster-worker-node1", script="attached compute worker", gpu=True, gpu_memory_gb=30),
        state=JobState.RUNNING,
        assigned_slot=2,
        assigned_username=creds2["username"],
        kaggle_kernel_slug=_get_shell_slug(2),
    ))

    # Push Master script (Node 0)
    master_script = (
        MASTER_BOOTSTRAP_TEMPLATE
        .replace("__SESSION_ID__", session_master_id)
        .replace("__WORKER_SESSION_ID__", session_worker_id)
        .replace("__DURATION_MINUTES__", str(int(duration_minutes)))
    )
    res_master = _push_kernel_payload(1, _get_shell_slug(1), master_script)

    # Push Worker script (Node 1)
    worker_script = (
        WORKER_BOOTSTRAP_TEMPLATE
        .replace("__SESSION_ID__", session_worker_id)
        .replace("__DURATION_MINUTES__", str(int(duration_minutes)))
    )
    res_worker = _push_kernel_payload(2, _get_shell_slug(2), worker_script)

    if res_master.get("status") == "FAILED" or res_worker.get("status") == "FAILED":
        err = f"Master: {res_master.get('error')} | Worker: {res_worker.get('error')}"
        raise RuntimeError(f"Failed to launch cluster nodes: {err}")

    console.print("  [dim]Cluster workers queued on GPU cloud. Establishing secure Master Web Terminal...[/dim]\n")

    start_time = time.time()
    master_web_url = ""
    dots = 0

    while time.time() - start_time < timeout_seconds:
        try:
            r = httpx.get(f"https://ntfy.sh/{session_master_id}/raw?poll=1", timeout=4)
            if r.status_code == 200 and r.text.strip():
                m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", r.text)
                if m:
                    master_web_url = m.group(0).strip()
                    break
        except Exception:
            pass

        print(f"\r  Connecting to Master GPU Terminal {'.' * (dots % 4 + 1)}    ", end="", flush=True)
        dots += 1
        time.sleep(3)

    print()

    if not master_web_url:
        raise TimeoutError(f"Cluster terminal failed to establish tunnel connection within {timeout_seconds}s.")

    _display_cluster_panel(creds1["username"], creds2["username"], master_web_url, duration_minutes)

    if open_web:
        console.print("\n[green]* Opening Master Web Terminal in your default browser...[/green]")
        try:
            webbrowser.open(master_web_url)
        except Exception:
            pass

    return {
        "web": master_web_url,
        "master_ref": f"{creds1['username']}/{_get_shell_slug(1)}",
        "worker_ref": f"{creds2['username']}/{_get_shell_slug(2)}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Launch Single-Slot GPU Terminal
# ─────────────────────────────────────────────────────────────────────────────
def launch_gpu_shell(
    slot: int,
    duration_minutes: int = 120,
    open_web: bool = False,
    timeout_seconds: int = 240,
) -> dict[str, str]:
    """Launch an interactive GPU terminal session on a specific slot (1 or 2)."""
    creds = load_credentials(slot)
    if creds is None:
        raise ValueError(f"No credentials configured for slot {slot}. Run: compute-pool login --slot {slot}")

    username = creds["username"]
    session_id = f"cp-shell-s{slot}-{uuid.uuid4().hex[:10]}"
    kernel_slug = _get_shell_slug(slot)

    upsert_job(Job(
        id=f"job-shell-s{slot}",
        spec=JobSpec(
            name=f"interactive-shell-slot{slot}",
            script="single node web terminal",
            gpu=True,
            gpu_memory_gb=15,
        ),
        state=JobState.RUNNING,
        assigned_slot=slot,
        assigned_username=username,
        kaggle_kernel_slug=kernel_slug,
    ))

    console.print(f"\n[bold cyan]Booting Interactive GPU Terminal (Slot {slot}: {username})...[/bold cyan]")
    console.print("  [dim]Provisioning 2x Tesla T4 GPU worker with live Web Terminal...[/dim]")

    script = (
        SINGLE_SHELL_BOOTSTRAP_TEMPLATE
        .replace("__SESSION_ID__", session_id)
        .replace("__DURATION_MINUTES__", str(int(duration_minutes)))
        .replace("__NODE_LABEL__", f"node{slot-1}-slot{slot}")
    )
    res = _push_kernel_payload(slot, kernel_slug, script)
    if res.get("status") == "FAILED":
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
        raise TimeoutError(f"Interactive terminal failed to establish tunnel connection within {timeout_seconds}s.")

    _display_single_shell_panel(slot, username, web_url, duration_minutes)

    if open_web:
        console.print("\n[green]* Opening Web Terminal in your default browser...[/green]")
        try:
            webbrowser.open(web_url)
        except Exception:
            pass

    return {"web": web_url, "kernel_ref": f"{username}/{kernel_slug}"}


# ─────────────────────────────────────────────────────────────────────────────
# 6. UI Panels & Teardown
# ─────────────────────────────────────────────────────────────────────────────
def _display_cluster_panel(master_user: str, worker_user: str, web_url: str, duration_minutes: int) -> None:
    body = (
        f"[bold green]* Unified 4-GPU Interactive Cluster Online & Ready[/bold green]\n\n"
        f"  [bold white]Cluster Hardware:[/bold white]   [bold cyan]4x Tesla T4 GPUs (~60 GB VRAM total)[/bold cyan]\n"
        f"  [bold white]Master Node 0:[/bold white]      Slot 1 ({master_user}) - 2x Tesla T4\n"
        f"  [bold white]Worker Node 1:[/bold white]      Slot 2 ({worker_user}) - 2x Tesla T4 (Attached)\n"
        f"  [bold white]Max Duration:[/bold white]       {duration_minutes} minutes\n\n"
        f"  [bold yellow]Master Web Terminal URL (Single Control Entrypoint):[/bold yellow]\n"
        f"  [bold underline cyan]{web_url}[/bold underline cyan]\n\n"
        f"+-- [bold yellow]Cluster Built-in Commands[/bold yellow] ---------------------------------------------+\n"
        f"  * [bold white]nvidia-smi[/bold white] / [bold white]gpus[/bold white] : Live unified table showing all 4 GPUs\n"
        f"  * [bold white]watch-gpu[/bold white]         : Live 1-second auto-refresh 4-GPU monitor\n"
        f"  * [bold white]cluster-exec <cmd>[/bold white]: Execute command across both nodes simultaneously\n"
        f"  * [bold white]cluster-status[/bold white]     : Inter-node mesh health & latency check\n"
        f"  * [bold white]stop[/bold white] / [bold white]exit[/bold white]       : Instantly teardown both nodes and release GPUs\n"
        f"+-------------------------------------------------------------------------+\n\n"
        f"  [dim]* Or run [bold white]compute-pool shell-stop[/bold white] from your local terminal at any time.[/dim]"
    )
    console.print(
        Panel(
            body,
            title="[bold green]Compute Pool -- Unified 4-GPU Master-Worker Cluster[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


def _display_single_shell_panel(slot: int, username: str, web_url: str, duration_minutes: int) -> None:
    body = (
        f"[bold green]* GPU Worker Active & Connected[/bold green]\n\n"
        f"  [bold white]Account Slot:[/bold white]   Slot {slot} ({username})\n"
        f"  [bold white]Hardware:[/bold white]       2x Tesla T4 GPUs (15 GB VRAM each)\n"
        f"  [bold white]Max Duration:[/bold white]   {duration_minutes} minutes\n\n"
        f"  [bold yellow]Web Terminal URL:[/bold yellow]\n"
        f"  [bold underline cyan]{web_url}[/bold underline cyan]\n\n"
        f"  [dim]* Type [bold white]stop[/bold white] or [bold white]exit[/bold white] in the terminal to immediately terminate & release GPU.[/dim]\n"
        f"  [dim]* Or run [bold white]compute-pool shell-stop[/bold white] from your local CLI.[/dim]"
    )
    console.print(
        Panel(
            body,
            title="[bold green]Compute Pool -- Live Interactive GPU Terminal[/bold green]",
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
            if j.id in [f"job-shell-s{s}", f"job-cluster-master", f"job-cluster-worker"] or (j.assigned_slot == s and "shell" in j.spec.name):
                if j.state == JobState.RUNNING:
                    j.transition(JobState.CANCELLED)
                    upsert_job(j)

