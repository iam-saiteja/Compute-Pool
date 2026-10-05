use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::auth::{load_all_credentials, load_credentials, MAX_ACCOUNT_SLOTS};
use crate::job::{Job, JobSpec, JobState};
use crate::kaggle::KaggleClient;
use crate::scheduler::get_account_status;
use crate::storage::{load_all_jobs, upsert_job};

pub const WORKER_BOOTSTRAP_TEMPLATE: &str = r#"
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

SESSION_ID = "__SESSION_ID__"
MASTER_SESSION_ID = "__MASTER_SESSION_ID__"
DURATION_MINUTES = __DURATION_MINUTES__
NODE_LABEL = "__NODE_LABEL__"

print(f"[*] Initializing Compute Pool GPU Cluster Worker ({NODE_LABEL})...", flush=True)

def report_failure(exc_type, exc, tb):
    import traceback
    text = "".join(traceback.format_exception(exc_type, exc, tb))[-3000:]
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}-error", data=text.encode("utf-8"))
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc, tb)

sys.excepthook = report_failure

# 1. Download Cloudflared, Chisel, and Filebrowser (user-facing file manager
#    plus the reverse TCP bridge used for the node-to-node SSH fabric below).
subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
], check=True)
subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/jpillora/chisel/releases/download/v1.10.1/chisel_1.10.1_linux_amd64.gz | gzip -d > /usr/local/bin/chisel && chmod +x /usr/local/bin/chisel"
], check=False)
subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/filebrowser/filebrowser/releases/download/v2.32.0/linux-amd64-filebrowser.tar.gz | tar -xz -C /usr/local/bin filebrowser && chmod +x /usr/local/bin/filebrowser"
], check=False)

# Fetch the worker runtime (checkpoint store, wire transport, execution
# strategies) from the public repo so pool-map and checkpointed training are
# available without the user uploading them by hand. Best-effort: if this
# fails (no internet to GitHub, repo renamed), the cluster still works --
# only pool-map and pipeline checkpoint/resume need it.
subprocess.run(["bash", "-c",
    "curl -sL https://github.com/iam-saiteja/Compute-Pool/archive/refs/heads/main.tar.gz "
    "| tar -xz -C /tmp "
    "&& rm -rf /kaggle/working/workers "
    "&& cp -r /tmp/Compute-Pool-main/workers /kaggle/working/workers"
], check=False)

# Preserve the worker's native NVIDIA tools for non-interactive task execution.
# Kaggle's terminal PATH is not necessarily inherited by an SSH session.
gpu_tool_candidates = [
    "/opt/bin/nvidia-smi",
    "/usr/bin/nvidia-smi",
    "/usr/local/nvidia/bin/nvidia-smi",
    "/usr/local/cuda/bin/nvidia-smi",
    "/opt/conda/bin/nvidia-smi",
]
gpu_tool = next((path for path in gpu_tool_candidates if os.path.exists(path)), None)
if not gpu_tool:
    gpu_tool = shutil.which("nvidia-smi")
if not gpu_tool:
    found = subprocess.run(
        ["bash", "-lc", "find /usr /opt /bin /sbin -name nvidia-smi 2>/dev/null | head -1"],
        capture_output=True,
        text=True,
    )
    gpu_tool = found.stdout.strip() or None
gpu_tool_dir = os.path.dirname(gpu_tool) if gpu_tool else None
if gpu_tool_dir:
    os.environ["PATH"] = f"{gpu_tool_dir}:" + os.environ.get("PATH", "")
    print(f"[*] Worker CUDA tools available from {gpu_tool_dir}", flush=True)
else:
    print("[!] Worker NVIDIA tools could not be located; GPU task validation will fail.", flush=True)

# nvidia-smi additionally needs libnvidia-ml.so at runtime -- its directory
# doesn't always match the binary's, so it's found independently.
gpu_lib_candidates = [
    "/usr/local/nvidia/lib64/libnvidia-ml.so",
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so",
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
]
gpu_lib = next((path for path in gpu_lib_candidates if os.path.exists(path)), None)
if not gpu_lib:
    found = subprocess.run(
        ["bash", "-lc", "find /usr /opt -iname 'libnvidia-ml.so*' 2>/dev/null | head -1"],
        capture_output=True,
        text=True,
    )
    gpu_lib = found.stdout.strip() or None
gpu_lib_dir = os.path.dirname(gpu_lib) if gpu_lib else None

# 2. Start a real sshd, then wait for the master's SSH *public* key over ntfy
#    and grant it access. The private half is generated on the master and
#    never leaves it -- only the public key (not sensitive) crosses the wire.
subprocess.run(["bash", "-c", "which sshd || (apt-get update -qq && apt-get install -y -qq openssh-server)"], check=False)
subprocess.run(["bash", "-c", "mkdir -p /var/run/sshd /root/.ssh /kaggle/working && chmod 700 /root/.ssh"], check=False)
subprocess.run(["bash", "-c", "ssh-keygen -A"], check=False)
# PAM's pam_motd module prints a container-provided MOTD (via /etc/update-motd.d)
# on every SSH session, including non-interactive `ssh host cmd` runs, which
# pollutes crun's captured stdout. Drop it; pam_env (needed for the PATH /
# LD_LIBRARY_PATH fix below) stays untouched.
subprocess.run(["bash", "-c", "sed -i '/pam_motd/d' /etc/pam.d/sshd 2>/dev/null || true"], check=False)

# sshd spawns a fresh, minimal environment for each `ssh host cmd` exec session
# -- it does not inherit this process's PATH and does not source .bashrc for
# non-interactive commands. Worse, PAM's pam_env module re-reads /etc/environment
# during session setup *after* sshd applies ~/.ssh/environment, silently
# overwriting PATH with Ubuntu's stock value. Writing /etc/environment directly
# is what actually sticks; ~/.ssh/environment is kept too as a fallback for
# systems where PAM isn't in play.
ssh_path = ":".join(filter(None, [
    gpu_tool_dir,
    "/opt/bin",
    "/usr/local/nvidia/bin",
    "/usr/local/cuda/bin",
    "/opt/conda/bin",
    "/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin",
    "/usr/games", "/usr/local/games", "/snap/bin",
]))
ssh_ld_library_path = ":".join(filter(None, [
    gpu_lib_dir,
    "/usr/local/nvidia/lib64",
    "/usr/local/cuda/lib64",
]))
with open("/root/.ssh/environment", "w") as f:
    f.write(f"PATH={ssh_path}\n")
    f.write(f"LD_LIBRARY_PATH={ssh_ld_library_path}\n")
os.chmod("/root/.ssh/environment", 0o600)
with open("/etc/environment", "w") as f:
    f.write(f'PATH="{ssh_path}"\n')
    f.write(f'LD_LIBRARY_PATH="{ssh_ld_library_path}"\n')

pubkey = ""
for _ in range(150):
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{MASTER_SESSION_ID}-pubkey/raw?poll=1")
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read().decode("utf-8").strip()
            if raw.startswith("ssh-"):
                pubkey = raw
                break
    except Exception:
        pass
    time.sleep(2)

if not pubkey:
    raise RuntimeError("Timed out waiting for the master's SSH public key over ntfy.")

with open("/root/.ssh/authorized_keys", "w") as f:
    f.write(pubkey + "\n")
os.chmod("/root/.ssh/authorized_keys", 0o600)

subprocess.Popen([
    "/usr/sbin/sshd", "-D", "-p", "2222",
    "-o", "PermitRootLogin=yes",
    "-o", "PubkeyAuthentication=yes",
    "-o", "AuthorizedKeysFile=/root/.ssh/authorized_keys",
    "-o", "PasswordAuthentication=no",
    "-o", "StrictModes=no",
    "-o", "AllowTcpForwarding=yes",
    "-o", "GatewayPorts=yes",
    "-o", "PermitUserEnvironment=yes",
    "-o", "TCPKeepAlive=yes",
    "-o", "ClientAliveInterval=15",
])
print("[*] SSH daemon active on port 2222 (master's public key installed).", flush=True)

# 3. Bridge that SSH port back to the master via Chisel over a Cloudflare
#    quick tunnel -- Kaggle sessions accept no inbound connections and have
#    no fixed address, so this is how the master reaches this port at all.
chisel_proc = subprocess.Popen(
    ["/usr/local/bin/chisel", "server", "--port", "8888", "--reverse"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(1)

cf_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:8888", "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
chisel_url = ""
for line in cf_proc.stdout:
    clean = line.strip()
    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", clean)
    if m:
        chisel_url = m.group(0)
        break

# 4. Start File Browser on Worker port 8081, tunneled out for the user's own
#    browser only.
fb_proc = subprocess.Popen([
    "/usr/local/bin/filebrowser", "-r", "/kaggle/working", "-a", "0.0.0.0", "-p", "8081", "--noauth"
])
time.sleep(1)

cf_fb_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:8081", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

worker_files_url = ""
for line in cf_fb_proc.stdout:
    clean = line.strip()
    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", clean)
    if m:
        worker_files_url = m.group(0)
        break

# 5. Publish this worker's tunnel URLs for the master to pick up.
payload = json.dumps({"chisel_url": chisel_url, "files_url": worker_files_url})
for _ in range(5):
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}", data=payload.encode("utf-8"))
        urllib.request.urlopen(req, timeout=10)
        break
    except Exception:
        time.sleep(2)
try:
    req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}-files", data=worker_files_url.encode("utf-8"))
    urllib.request.urlopen(req, timeout=10)
except Exception:
    pass

print(f"[*] Worker registered. Chisel: {chisel_url}, Files: {worker_files_url}", flush=True)

# 6. Keep worker alive until STOP signal
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
subprocess.run(["pkill", "-9", "-f", "chisel"], check=False)
subprocess.run(["pkill", "-9", "-f", "sshd"], check=False)
subprocess.run(["kill", "-9", "-1"], check=False)
sys.exit(0)
"#;

pub const SINGLE_SHELL_BOOTSTRAP_TEMPLATE: &str = r#"
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
    "curl -sL https://github.com/filebrowser/filebrowser/releases/download/v2.32.0/linux-amd64-filebrowser.tar.gz | tar -xz -C /usr/local/bin filebrowser && chmod +x /usr/local/bin/filebrowser"
], check=False)
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
    f.write(f"\nexport PATH=/usr/local/bin:$PATH\n")
    f.write(f"export PS1='\\[\\033[01;32m\\]compute-pool@{NODE_LABEL}\\[\\033[00m\\]:\\[\\033[01;34m\\]\\w\\[\\033[00m\\]\\$ '\n")
    f.write("alias gpus='nvidia-smi'\n")
    f.write("alias watch-gpu='watch -n 1 nvidia-smi'\n")
    f.write("alias halt='/usr/local/bin/stop'\n")
    f.write("alias exit='/usr/local/bin/stop'\n")

ttyd_proc = subprocess.Popen([
    "/usr/local/bin/ttyd", "-W", "-p", "7681",
    "-t", "enableClipboard=true",
    "-t", "fontSize=15",
    "-t", "disableLeaveAlert=true",
    "bash"
])
fb_proc = subprocess.Popen([
    "/usr/local/bin/filebrowser", "-r", "/kaggle/working", "-a", "0.0.0.0", "-p", "8080", "--noauth"
])
time.sleep(1)

cf_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:7681", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in cf_proc.stdout:
    clean = line.strip()
    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", clean)
    if m:
        terminal_url = m.group(0)
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}", data=terminal_url.encode("utf-8"))
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass
        break

cf_fb_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:8080", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in cf_fb_proc.stdout:
    clean = line.strip()
    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", clean)
    if m:
        files_url = m.group(0)
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}-files", data=files_url.encode("utf-8"))
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
subprocess.run(["pkill", "-9", "-f", "filebrowser"], check=False)
sys.exit(0)
"#;

pub const MASTER_BOOTSTRAP_TEMPLATE: &str = r#"
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

SESSION_ID = "__SESSION_ID__"
WORKER_SESSIONS = __WORKER_SESSIONS__
DURATION_MINUTES = __DURATION_MINUTES__
NODE_LABEL = "cluster-master"
REGISTRY_PATH = "/etc/compute-pool/nodes.json"
SSH_PORT_BASE = 2200
PEER_WAIT_SECONDS = 900
HEALTH_INTERVAL_SECONDS = 30
HEALTH_FAILURES_BEFORE_DOWN = 3

print(f"[*] Initializing Compute Pool Master GPU Terminal ({len(WORKER_SESSIONS) + 1} nodes)...", flush=True)

def report_failure(exc_type, exc, tb):
    text = "".join(traceback.format_exception(exc_type, exc, tb))[-3000:]
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}-error", data=text.encode("utf-8"))
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc, tb)

sys.excepthook = report_failure

def notify(topic, text):
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=text.encode("utf-8"))
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

# 1. Download ttyd, filebrowser, cloudflared, chisel -- ttyd/filebrowser/cloudflared
#    serve the user's own browser; chisel carries the node-to-node SSH fabric.
subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 -o /usr/local/bin/ttyd && chmod +x /usr/local/bin/ttyd"
], check=True)
subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/filebrowser/filebrowser/releases/download/v2.32.0/linux-amd64-filebrowser.tar.gz | tar -xz -C /usr/local/bin filebrowser && chmod +x /usr/local/bin/filebrowser"
], check=False)
subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
], check=True)
subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/jpillora/chisel/releases/download/v1.10.1/chisel_1.10.1_linux_amd64.gz | gzip -d > /usr/local/bin/chisel && chmod +x /usr/local/bin/chisel"
], check=False)

# Fetch the worker runtime (checkpoint store, wire transport, execution
# strategies) from the public repo -- see the matching comment in the worker
# template. Best-effort: pool-map and checkpointed training need it, the rest
# of the cluster does not.
subprocess.run(["bash", "-c",
    "curl -sL https://github.com/iam-saiteja/Compute-Pool/archive/refs/heads/main.tar.gz "
    "| tar -xz -C /tmp "
    "&& rm -rf /kaggle/working/workers "
    "&& cp -r /tmp/Compute-Pool-main/workers /kaggle/working/workers"
], check=False)

# 2. Generate the cluster SSH keypair here. The private half never leaves this
#    machine; only the public half is published for the workers to trust.
key_path = "/root/.ssh/cluster_key"
os.makedirs("/root/.ssh", mode=0o700, exist_ok=True)
if not os.path.exists(key_path):
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", key_path, "-C", "compute-pool-cluster"], check=True)
with open(f"{key_path}.pub") as f:
    pubkey = f.read().strip()

def publish_pubkey():
    for _ in range(100):
        notify(f"{SESSION_ID}-pubkey", pubkey)
        time.sleep(3)

threading.Thread(target=publish_pubkey, daemon=True).start()

with open("/root/.ssh/config", "w") as f:
    for i in range(1, len(WORKER_SESSIONS) + 1):
        f.write(f"Host node{i}\n")
        f.write("    HostName 127.0.0.1\n")
        f.write(f"    Port {SSH_PORT_BASE + i}\n")
        f.write("    User root\n")
        f.write(f"    IdentityFile {key_path}\n")
        f.write("    StrictHostKeyChecking no\n")
        f.write("    UserKnownHostsFile /dev/null\n")
        f.write("    LogLevel ERROR\n")
os.chmod("/root/.ssh/config", 0o600)

# 3. Cluster registry: the single source of truth every cluster tool reads.
#    node0 is this master; node1..nodeN are the workers, filled in as they peer.
os.makedirs("/etc/compute-pool", exist_ok=True)
state_lock = threading.RLock()
nodes = []

def count_gpus(smi_output):
    return sum(1 for line in smi_output.splitlines() if line.startswith("GPU "))

def write_registry():
    with state_lock:
        tmp = REGISTRY_PATH + ".tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps({"nodes": nodes}))
        os.replace(tmp, REGISTRY_PATH)

def update_node(name, **fields):
    with state_lock:
        for n in nodes:
            if n["name"] == name:
                n.update(fields)
        write_registry()

try:
    local_gpu_count = count_gpus(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
except FileNotFoundError:
    local_gpu_count = 0
nodes.append({"name": "node0", "index": 0, "local": True, "gpus": local_gpu_count, "status": "online", "error": "", "fails": 0})
for i in range(1, len(WORKER_SESSIONS) + 1):
    nodes.append({"name": f"node{i}", "index": i, "local": False, "gpus": 0, "status": "pending", "error": "", "fails": 0})
write_registry()

def ssh_run(name, cmd, timeout=10):
    try:
        res = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes", name, cmd],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        return res.returncode, res.stdout
    except Exception as exc:
        return -1, str(exc)

def peer_worker(index, session_id):
    name = f"node{index}"
    worker = None
    deadline = time.time() + PEER_WAIT_SECONDS
    while worker is None and time.time() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{session_id}/raw?poll=1"), timeout=5) as r:
                raw = r.read().decode("utf-8").strip()
                if raw.startswith("{") and "chisel_url" in raw:
                    worker = json.loads(raw)
        except Exception:
            pass
        if worker is None:
            time.sleep(3)
    if worker is None:
        update_node(name, status="failed", error="worker never registered (check its error topic)")
        return
    subprocess.Popen(
        ["/usr/local/bin/chisel", "client", "--keepalive", "10s", worker["chisel_url"], f"{SSH_PORT_BASE + index}:127.0.0.1:2222"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    rc, out = -1, ""
    for _ in range(30):
        rc, out = ssh_run(name, "nvidia-smi -L")
        if rc == 0:
            break
        time.sleep(3)
    if rc != 0:
        update_node(name, status="failed", error="tunnel is up but SSH to the worker failed")
        return
    update_node(name, status="online", gpus=count_gpus(out), error="", files_url=worker.get("files_url", ""), fails=0)
    print(f"[*] {name} online ({count_gpus(out)} GPUs)", flush=True)

def health_loop():
    while True:
        time.sleep(HEALTH_INTERVAL_SECONDS)
        with state_lock:
            targets = [n["name"] for n in nodes if not n["local"] and n["status"] in ("online", "down")]
        for name in targets:
            rc, _ = ssh_run(name, "true", timeout=5)
            went_down = False
            came_back = False
            with state_lock:
                node = next(n for n in nodes if n["name"] == name)
                if rc == 0:
                    came_back = node["status"] == "down"
                    node["fails"] = 0
                    node["status"] = "online"
                else:
                    node["fails"] += 1
                    if node["status"] == "online" and node["fails"] >= HEALTH_FAILURES_BEFORE_DOWN:
                        node["status"] = "down"
                        went_down = True
                write_registry()
            if went_down:
                notify(f"{SESSION_ID}-health", f"{name} is DOWN: no SSH response for {HEALTH_FAILURES_BEFORE_DOWN} checks")
            if came_back:
                notify(f"{SESSION_ID}-health", f"{name} recovered")

threading.Thread(target=health_loop, daemon=True).start()
for i, sid in enumerate(WORKER_SESSIONS, start=1):
    threading.Thread(target=peer_worker, args=(i, sid), daemon=True).start()

# 4. crun: runs on every online node. Reads the registry on each call, so a node
#    that goes down drops out of dispatch and comes back when it recovers.
crun_script = '''#!/usr/bin/env python3
import concurrent.futures, json, os, shlex, subprocess, sys

REGISTRY = "/etc/compute-pool/nodes.json"

def print_help():
    print("Compute Pool Cluster Runner (crun)")
    print("Usage:")
    print("  crun <command> [args...]       Run command on every online node")
    print("  crun --gpus '<command>'        Run one task per online GPU across the cluster")
    print("  crun -g '<command>'            Shortcut for --gpus")
    print("")
    print("Examples:")
    print("  crun nvidia-smi")
    print("  crun --gpus 'python3 train.py --fold {task_index}'")
    print("")
    print("Placeholders for --gpus mode:")
    print("  {task_index}  : global task number (0 .. task_count-1)")
    print("  {gpu_index}   : GPU index on its node")
    print("  {node_index}  : node number (0 = master)")
    print("  {task_count}  : total tasks")
    sys.exit(0)

def load_nodes():
    with open(REGISTRY) as f:
        nodes = json.load(f)["nodes"]
    return [n for n in nodes if n.get("status") == "online"]

def run_remote(name, cmd, env=None, timeout=300):
    env_str = ""
    if env:
        env_str = " ".join(f"export {k}={shlex.quote(str(v))};" for k, v in env.items()) + " "
    full_cmd = f"cd /kaggle/working 2>/dev/null; {env_str}{cmd}"
    try:
        res = subprocess.run(["ssh", "-o", "ConnectTimeout=5", name, full_cmd], capture_output=True, text=True, timeout=timeout)
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Remote command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)

def run_local(cmd, env=None):
    local_env = os.environ.copy()
    local_env.update(env or {})
    res = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, env=local_env)
    return res.returncode, res.stdout, res.stderr

def sync_files(tokens, nodes):
    for token in tokens:
        if os.path.isfile(token) and not token.startswith("-"):
            fname = os.path.basename(token)
            for n in nodes:
                if not n["local"]:
                    subprocess.run(["scp", "-o", "ConnectTimeout=5", token, f"{n['name']}:/kaggle/working/{fname}"], capture_output=True)

def describe(code, out, err):
    out = (out or "").strip()
    if err and err.strip():
        out = out + "\\n[stderr]\\n" + err.strip()
        out = out.strip()
    if not out and code == 0:
        out = "[OK: completed with exit code 0]"
    elif not out:
        out = f"[! exited with code {code}]"
    return out

def run_gpus(template, nodes):
    tasks = [(n, g) for n in nodes for g in range(n["gpus"])]
    total = len(tasks)
    if total == 0:
        print("No online GPUs in the cluster registry.")
        return 1
    sync_files(template.split(), nodes)

    def run_task(item):
        idx, (n, g) = item
        cmd = template.format(task_index=idx, gpu_index=g, node_index=n["index"], task_count=total)
        env = dict(CUDA_VISIBLE_DEVICES=str(g), CP_TASK_INDEX=str(idx), CP_TASK_COUNT=str(total), CP_NODE_INDEX=str(n["index"]))
        if n["local"]:
            return (idx, n, g) + run_local(cmd, env)
        return (idx, n, g) + run_remote(n["name"], cmd, env)

    with concurrent.futures.ThreadPoolExecutor(max_workers=total) as pool:
        results = list(pool.map(run_task, enumerate(tasks)))
    failed = 0
    for idx, n, g, code, out, err in sorted(results, key=lambda r: r[0]):
        status = "PASSED" if code == 0 else f"FAILED (exit {code})"
        print(f"=== [Task {idx}: {n['name']} GPU {g}] {status} ===")
        if out and out.strip():
            print(out.strip())
        if err and err.strip():
            print(err.strip(), file=sys.stderr)
        print()
        if code != 0:
            failed += 1
    return 1 if failed else 0

def run_broadcast(shell_cmd, nodes):
    sync_files(shell_cmd.split(), nodes)

    def run_node(n):
        if n["local"]:
            return (n,) + run_local(shell_cmd, {"NODE_RANK": "0"})
        return (n,) + run_remote(n["name"], shell_cmd, {"NODE_RANK": str(n["index"])})

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        results = list(pool.map(run_node, nodes))
    for n, code, out, err in results:
        role = "Master" if n["local"] else "Worker"
        print(f"[{n['name']}: {role} (node {n['index']})]")
        print(describe(code, out, err))
        print()
        print("=" * 80)
        print()
    return 0

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print_help()

nodes = load_nodes()
if not nodes:
    print("No online nodes in the cluster registry.")
    sys.exit(1)

if sys.argv[1] in ("--gpus", "-g", "--dispatch"):
    if len(sys.argv) < 3:
        print("Usage: crun --gpus '<command template>'")
        print("Example: crun --gpus 'python3 train.py --fold {task_index}'")
        sys.exit(1)
    sys.exit(run_gpus(" ".join(sys.argv[2:]), nodes))

sys.exit(run_broadcast(" ".join(sys.argv[1:]), nodes))
'''
with open("/usr/local/bin/crun", "w") as f:
    f.write(crun_script)
os.chmod("/usr/local/bin/crun", 0o755)

# 5. cp-dispatch: shortcut for crun --gpus
cp_dispatch_script = '''#!/bin/bash
exec /usr/local/bin/crun --gpus "$@"
'''
with open("/usr/local/bin/cp-dispatch", "w") as f:
    f.write(cp_dispatch_script)
os.chmod("/usr/local/bin/cp-dispatch", 0o755)

# 5b. pool-map: checkpointed, retrying shard dispatch across every online GPU
#     (workers/kaggle/strategies/independent.py, fetched in step 1). A worker
#     that drops mid-run only loses its in-flight shard, not the job, and the
#     whole run can also resume if restarted.
pool_map_script = '''#!/usr/bin/env python3
import argparse
import json
import sys

REGISTRY = "/etc/compute-pool/nodes.json"


def main():
    sys.path.insert(0, "/kaggle/working")
    try:
        from workers.kaggle import checkpoint as checkpoint_mod
        from workers.kaggle.strategies import independent
    except ImportError:
        print("pool-map needs /kaggle/working/workers/kaggle -- it was not fetched during", file=sys.stderr)
        print("cluster startup. Check internet access to github.com and relaunch the cluster.", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Checkpointed, retrying shard dispatch across every online GPU.")
    ap.add_argument("command", help="shell command template; may use {task_index}/{task_count}/{node_index}")
    ap.add_argument("--shards", type=int, required=True, help="number of shards")
    ap.add_argument("--job-id", default="pool-map-job", help="checkpoint job id (rerun with the same id to resume)")
    ap.add_argument("--checkpoint-uri", default="local:///kaggle/working/checkpoints")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--output", help="write the combined per-shard results as JSON to this path")
    args = ap.parse_args()

    nodes = [n for n in json.load(open(REGISTRY))["nodes"] if n.get("status") == "online"]
    if not nodes:
        print("No online nodes in the cluster registry.", file=sys.stderr)
        sys.exit(1)

    store = checkpoint_mod.open_store(args.checkpoint_uri, args.job_id)

    def progress(idx, code, done, total):
        mark = "ok" if code == 0 else f"FAILED(exit {code})"
        print(f"[{done}/{total}] shard {idx}: {mark}", flush=True)

    out = independent.run_map(nodes, args.command, args.shards, checkpoint_store=store,
                               max_retries=args.max_retries, on_progress=progress)
    print()
    print(f"done: {len(out['results'])}/{args.shards} shards completed, {len(out['failed'])} failed permanently")
    if out["failed"]:
        print("failed shards:", out["failed"])
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"combined results written to {args.output}")
    sys.exit(1 if out["failed"] else 0)


if __name__ == "__main__":
    main()
'''
with open("/usr/local/bin/pool-map", "w") as f:
    f.write(pool_map_script)
os.chmod("/usr/local/bin/pool-map", 0o755)

# 6. cluster-status: reads the registry, so it reports the health loop's view.
status_script = '''#!/usr/bin/env python3
import json

REGISTRY = "/etc/compute-pool/nodes.json"
nodes = json.load(open(REGISTRY))["nodes"]
online_gpus = sum(n.get("gpus", 0) for n in nodes if n.get("status") == "online")

print("+----------------------------------------------------------------------+")
print("|                 Compute Pool GPU Cluster Status                      |")
print("+----------------------------------------------------------------------+")
for n in nodes:
    role = "Master" if n.get("local") else "Worker"
    status = n.get("status", "unknown")
    mark = "*" if status == "online" else "!"
    line = f"[{mark}] {n['name']} ({role}): {status}, {n.get('gpus', 0)} GPUs"
    if n.get("error"):
        line += f" -- {n['error']}"
    print(line)
print(f"[*] Online GPUs in cluster: {online_gpus}")
print("+----------------------------------------------------------------------+")
print("Cluster Tools & Commands:")
print("  • nvidia-smi              -> Local GPU telemetry (master)")
print("  • ssh node1               -> Shell on worker node1 (node<N> for others)")
print("  • crun <command>          -> Run command on every online node")
print("  • crun --gpus '<command>' -> Run one task per online GPU")
print("  • cp-dispatch '<command>' -> Shortcut for crun --gpus")
print("  • pool-map '<cmd>' --shards N -> Checkpointed shard dispatch (retries, resumable)")
print("  • stop                    -> Terminate cluster session")
print("+----------------------------------------------------------------------+")
'''
with open("/usr/local/bin/cluster-status", "w") as f:
    f.write(status_script)
os.chmod("/usr/local/bin/cluster-status", 0o755)

stop_script = '''#!/bin/bash
kill -9 -1
'''
with open("/usr/local/bin/stop", "w") as f:
    f.write(stop_script)
os.chmod("/usr/local/bin/stop", 0o755)

with open(os.path.expanduser("~/.bashrc"), "a") as f:
    f.write(f"\nexport PATH=/usr/local/bin:$PATH\n")
    f.write(f"export PS1='\\[\\033[01;32m\\]compute-pool@{NODE_LABEL}\\[\\033[00m\\]:\\[\\033[01;34m\\]\\w\\[\\033[00m\\]\\$ '\n")
    f.write("alias gpus='nvidia-smi'\n")
    f.write("alias watch-gpu='watch -n 1 nvidia-smi'\n")
    f.write("alias halt='/usr/local/bin/stop'\n")
    f.write("alias exit='/usr/local/bin/stop'\n")

# 8. Web terminal, file manager and their Cloudflare tunnels for the user's browser.
ttyd_proc = subprocess.Popen([
    "/usr/local/bin/ttyd", "-W", "-p", "7681",
    "-t", "enableClipboard=true",
    "-t", "fontSize=15",
    "-t", "disableLeaveAlert=true",
    "bash"
])
fb_proc = subprocess.Popen([
    "/usr/local/bin/filebrowser", "-r", "/kaggle/working", "-a", "0.0.0.0", "-p", "8080", "--noauth"
])
time.sleep(1)

cf_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:7681", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in cf_proc.stdout:
    clean = line.strip()
    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", clean)
    if m:
        notify(SESSION_ID, m.group(0))
        break

cf_fb_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:8080", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in cf_fb_proc.stdout:
    clean = line.strip()
    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", clean)
    if m:
        notify(f"{SESSION_ID}-files", m.group(0))
        break

# 9. Keep the master alive until a STOP signal, then tear everything down.
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
subprocess.run(["pkill", "-9", "-f", "filebrowser"], check=False)
subprocess.run(["pkill", "-9", "-f", "chisel"], check=False)
subprocess.run(["kill", "-9", "-1"], check=False)
sys.exit(0)
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellInfo {
    pub slot: usize,
    pub username: String,
    pub web_url: String,
    pub files_url: String,
    pub kernel_ref: String,
    pub duration_minutes: u32,
}

fn shell_slug_for_slot(slot: usize) -> String {
    format!("interactive-gpu-terminal-s{}", slot)
}

fn random_suffix() -> String {
    (0..8)
        .map(|_| {
            let chars = b"abcdefghijklmnopqrstuvwxyz0123456789";
            chars[fastrand::usize(..chars.len())] as char
        })
        .collect()
}

pub async fn launch_gpu_shell(
    slot: usize,
    duration_minutes: u32,
    timeout_seconds: Option<u64>,
) -> Result<ShellInfo> {
    let creds = load_credentials(slot)?
        .ok_or_else(|| anyhow::anyhow!("No credentials for slot {}. Run `compute-pool login --slot {}`", slot, slot))?;

    let session_id = format!("cp-shell-s{}-{}", slot, random_suffix());
    let slug = shell_slug_for_slot(slot);

    // Register job in local state
    let mut job = Job::new(
        format!("job-shell-s{}", slot),
        JobSpec {
            name: format!("interactive-shell-slot{}", slot),
            script: format!("session_id:{}", session_id),
            gpu: true,
            gpu_memory_gb: 15.0,
            max_runtime_hours: (duration_minutes as f64) / 60.0,
            checkpointable: false,
            max_retries: 0,
        },
    );
    job.state = JobState::Running;
    job.assigned_slot = Some(serde_json::json!(slot));
    job.assigned_username = Some(creds.username.clone());
    job.kaggle_kernel_slug = Some(slug.clone());
    upsert_job(&job)?;

    let script = SINGLE_SHELL_BOOTSTRAP_TEMPLATE
        .replace("__SESSION_ID__", &session_id)
        .replace("__DURATION_MINUTES__", &duration_minutes.to_string())
        .replace("__NODE_LABEL__", &format!("node{}-slot{}", slot - 1, slot));

    let client = KaggleClient::new(&creds.username, &creds.key);
    client.push_kernel(&slug, &script, true).await
        .context("Failed to push shell kernel to Kaggle")?;

    let http_client = reqwest::Client::new();
    let start = Instant::now();
    let re = Regex::new(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")?;
    let mut web_url = String::new();

    while match timeout_seconds {
        Some(t) => start.elapsed() < Duration::from_secs(t),
        None => true,
    } {
        let ntfy_url = format!("https://ntfy.sh/{}/raw?poll=1", session_id);
        if let Ok(resp) = http_client.get(&ntfy_url).send().await {
            if resp.status().is_success() {
                if let Ok(text) = resp.text().await {
                    for line in text.lines() {
                        if let Some(m) = re.find(line) {
                            web_url = m.as_str().to_string();
                            break;
                        }
                    }
                    if !web_url.is_empty() {
                        break;
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(3)).await;
    }

    if web_url.is_empty() {
        anyhow::bail!("Timed out waiting for GPU Web Terminal tunnel connection.");
    }

    // Try fetching files URL
    let mut files_url = String::new();
    for _ in 0..5 {
        let ntfy_files_url = format!("https://ntfy.sh/{}-files/raw?poll=1", session_id);
        if let Ok(resp) = http_client.get(&ntfy_files_url).send().await {
            if resp.status().is_success() {
                if let Ok(text) = resp.text().await {
                    if let Some(m) = re.find(&text) {
                        files_url = m.as_str().to_string();
                        break;
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    Ok(ShellInfo {
        slot,
        username: creds.username.clone(),
        web_url,
        files_url,
        kernel_ref: format!("{}/{}", creds.username, slug),
        duration_minutes,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNodeInfo {
    pub node_index: usize,
    pub slot: usize,
    pub username: String,
    pub files_url: String,
    pub kernel_ref: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterShellInfo {
    pub web_url: String,
    pub nodes: Vec<ClusterNodeInfo>,
    pub duration_minutes: u32,
}

const DEFAULT_CLUSTER_TIMEOUT_SECS: u64 = 900;

async fn read_ntfy(client: &reqwest::Client, topic: &str) -> String {
    match client.get(format!("https://ntfy.sh/{}/raw?poll=1", topic)).send().await {
        Ok(resp) if resp.status().is_success() => resp.text().await.unwrap_or_default(),
        _ => String::new(),
    }
}

async fn read_ntfy_url(client: &reqwest::Client, topic: &str, re: &Regex) -> String {
    let text = read_ntfy(client, topic).await;
    re.find(&text).map(|m| m.as_str().to_string()).unwrap_or_default()
}

async fn ensure_quota(slot: usize, needed_hours: f64) -> Result<()> {
    let status = get_account_status(slot).await;
    if let Some(err) = &status.error {
        anyhow::bail!("Slot {}: {}", slot, err);
    }
    if !status.connected {
        anyhow::bail!("Slot {} ({}): Kaggle account is not reachable -- check its API key", slot, status.username);
    }
    let remaining = status.gpu_hours_remaining();
    if remaining < needed_hours {
        anyhow::bail!(
            "Slot {} ({}): {:.2} GPU hours left, this session needs {:.2}",
            slot,
            status.username,
            remaining,
            needed_hours
        );
    }
    Ok(())
}

pub async fn launch_cluster_shell(
    nodes: usize,
    duration_minutes: u32,
    timeout_seconds: Option<u64>,
) -> Result<ClusterShellInfo> {
    if nodes < 2 {
        anyhow::bail!("A cluster needs at least 2 nodes");
    }
    if nodes > MAX_ACCOUNT_SLOTS {
        anyhow::bail!("At most {} nodes are supported (one Kaggle account per node)", MAX_ACCOUNT_SLOTS);
    }

    let mut creds = Vec::with_capacity(nodes);
    for slot in 1..=nodes {
        let c = load_credentials(slot)?.ok_or_else(|| {
            anyhow::anyhow!("Slot {} not configured. Run `compute-pool login --slot {}`", slot, slot)
        })?;
        creds.push(c);
    }

    let needed_hours = duration_minutes as f64 / 60.0;
    for slot in 1..=nodes {
        ensure_quota(slot, needed_hours).await?;
    }

    let suffix = random_suffix();
    let master_id = format!("cp-master-{}", suffix);
    let worker_ids: Vec<String> = (1..nodes).map(|i| format!("cp-worker{}-{}", i, suffix)).collect();
    let all_slots: Vec<usize> = (1..=nodes).collect();

    for idx in 0..nodes {
        let slot = idx + 1;
        let session = if idx == 0 { master_id.clone() } else { worker_ids[idx - 1].clone() };
        let mut job = Job::new(
            format!("job-cluster-node{}", idx),
            JobSpec {
                name: format!("cluster-node{}", idx),
                script: format!("session_id:{}", session),
                gpu: true,
                gpu_memory_gb: 15.0,
                max_runtime_hours: needed_hours,
                checkpointable: false,
                max_retries: 0,
            },
        );
        job.state = JobState::Running;
        job.assigned_slot = Some(serde_json::json!(slot));
        job.assigned_username = Some(creds[idx].username.clone());
        job.kaggle_kernel_slug = Some(shell_slug_for_slot(slot));
        upsert_job(&job)?;
    }

    let master_script = MASTER_BOOTSTRAP_TEMPLATE
        .replace("__SESSION_ID__", &master_id)
        .replace("__DURATION_MINUTES__", &duration_minutes.to_string())
        .replace("__WORKER_SESSIONS__", &serde_json::to_string(&worker_ids)?);

    let mut scripts = vec![master_script];
    for (i, wid) in worker_ids.iter().enumerate() {
        let node_idx = i + 1;
        scripts.push(
            WORKER_BOOTSTRAP_TEMPLATE
                .replace("__SESSION_ID__", wid)
                .replace("__DURATION_MINUTES__", &duration_minutes.to_string())
                .replace("__MASTER_SESSION_ID__", &master_id)
                .replace("__NODE_LABEL__", &format!("node{}-slot{}", node_idx, node_idx + 1)),
        );
    }

    let mut pushes = tokio::task::JoinSet::new();
    for (idx, script) in scripts.into_iter().enumerate() {
        let slot = idx + 1;
        let client = KaggleClient::new(&creds[idx].username, &creds[idx].key);
        let slug = shell_slug_for_slot(slot);
        let label = if idx == 0 { "master".to_string() } else { format!("worker node{}", idx) };
        pushes.spawn(async move {
            client
                .push_kernel(&slug, &script, true)
                .await
                .with_context(|| format!("Failed to push {} kernel", label))
        });
    }
    let mut push_error: Option<anyhow::Error> = None;
    while let Some(joined) = pushes.join_next().await {
        match joined {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => push_error = push_error.or(Some(e)),
            Err(e) => push_error = push_error.or(Some(anyhow::anyhow!("kernel push task failed: {}", e))),
        }
    }
    if let Some(err) = push_error {
        let _ = stop_slots(&all_slots).await;
        return Err(err);
    }

    let client = reqwest::Client::new();
    let re = Regex::new(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")?;
    let deadline = Duration::from_secs(timeout_seconds.unwrap_or(DEFAULT_CLUSTER_TIMEOUT_SECS));
    let start = Instant::now();

    let mut web_url = String::new();
    let mut ready = vec![false; worker_ids.len()];
    while start.elapsed() < deadline {
        if web_url.is_empty() {
            web_url = read_ntfy_url(&client, &master_id, &re).await;
        }
        for (i, wid) in worker_ids.iter().enumerate() {
            if !ready[i] && read_ntfy(&client, wid).await.contains("chisel_url") {
                ready[i] = true;
            }
        }
        if !web_url.is_empty() && ready.iter().all(|r| *r) {
            break;
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }

    if web_url.is_empty() || ready.iter().any(|r| !*r) {
        let mut reasons = Vec::new();
        if web_url.is_empty() {
            reasons.push("master web terminal did not come online".to_string());
        }
        for (i, wid) in worker_ids.iter().enumerate() {
            if ready[i] {
                continue;
            }
            let reported = read_ntfy(&client, &format!("{}-error", wid)).await;
            let tail: Vec<&str> = reported.trim().lines().rev().take(6).collect();
            let detail = if tail.is_empty() {
                "no response (still booting, or crashed before reporting)".to_string()
            } else {
                tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
            };
            reasons.push(format!("node{} did not come online: {}", i + 1, detail));
        }
        let _ = stop_slots(&all_slots).await;
        anyhow::bail!("Cluster did not come up; stopped all {} accounts. {}", nodes, reasons.join("; "));
    }

    let mut node_infos = Vec::with_capacity(nodes);
    node_infos.push(ClusterNodeInfo {
        node_index: 0,
        slot: 1,
        username: creds[0].username.clone(),
        files_url: read_ntfy_url(&client, &format!("{}-files", master_id), &re).await,
        kernel_ref: format!("{}/{}", creds[0].username, shell_slug_for_slot(1)),
    });
    for (i, wid) in worker_ids.iter().enumerate() {
        let slot = i + 2;
        node_infos.push(ClusterNodeInfo {
            node_index: i + 1,
            slot,
            username: creds[i + 1].username.clone(),
            files_url: read_ntfy_url(&client, &format!("{}-files", wid), &re).await,
            kernel_ref: format!("{}/{}", creds[i + 1].username, shell_slug_for_slot(slot)),
        });
    }

    Ok(ClusterShellInfo {
        web_url,
        nodes: node_infos,
        duration_minutes,
    })
}

pub async fn stop_gpu_shell(slot: Option<usize>) -> Result<()> {
    let slots: Vec<usize> = match slot {
        Some(s) => vec![s],
        None => {
            let mut all: Vec<usize> = load_all_credentials()?.keys().copied().collect();
            all.sort_unstable();
            all
        }
    };
    stop_slots(&slots).await
}

async fn stop_slots(slots: &[usize]) -> Result<()> {
    let stop_script = "import sys\nprint('Shell terminated by user.')\nsys.exit(0)\n";
    let http_client = reqwest::Client::new();
    let all_jobs = load_all_jobs().unwrap_or_default();

    for &s in slots {
        for mut j in all_jobs.clone() {
            let is_slot_match = j.get_slot_number() == Some(s);
            if is_slot_match && (j.spec.name.contains("shell") || j.spec.name.contains("cluster")) {
                if let Some(session_id) = j.spec.script.strip_prefix("session_id:") {
                    let stop_url = format!("https://ntfy.sh/{}-stop", session_id.trim());
                    let _ = http_client.post(&stop_url).body("STOP").send().await;
                }
                if j.state.is_active() {
                    j.transition(JobState::Cancelled, Some("Terminated via shell-stop".to_string()));
                    let _ = upsert_job(&j);
                }
            }
        }

        if let Ok(Some(creds)) = load_credentials(s) {
            let client = KaggleClient::new(&creds.username, &creds.key);
            let slugs = vec![
                shell_slug_for_slot(s),
                "interactive-gpu-terminal".to_string(),
                "interactive-gpu-session".to_string(),
            ];
            for slug in slugs {
                let _ = client.push_kernel(&slug, stop_script, false).await;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_placeholders() {
        let t = SINGLE_SHELL_BOOTSTRAP_TEMPLATE
            .replace("__SESSION_ID__", "cp-test")
            .replace("__DURATION_MINUTES__", "60")
            .replace("__NODE_LABEL__", "node0-slot1");

        assert!(t.contains("SESSION_ID = \"cp-test\""));
        assert!(t.contains("DURATION_MINUTES = 60"));
        assert!(t.contains("NODE_LABEL = \"node0-slot1\""));
    }

    #[test]
    fn test_cluster_templates_fill_all_placeholders() {
        let master = MASTER_BOOTSTRAP_TEMPLATE
            .replace("__SESSION_ID__", "cp-master-test")
            .replace("__DURATION_MINUTES__", "60")
            .replace("__WORKER_SESSIONS__", r#"["cp-worker1-test","cp-worker2-test"]"#);

        assert!(master.contains("SESSION_ID = \"cp-master-test\""));
        assert!(master.contains(r#"WORKER_SESSIONS = ["cp-worker1-test","cp-worker2-test"]"#));
        for placeholder in ["__SESSION_ID__", "__WORKER_SESSIONS__", "__DURATION_MINUTES__"] {
            assert!(!master.contains(placeholder), "unfilled {}", placeholder);
        }

        let worker = WORKER_BOOTSTRAP_TEMPLATE
            .replace("__SESSION_ID__", "cp-worker1-test")
            .replace("__DURATION_MINUTES__", "60")
            .replace("__MASTER_SESSION_ID__", "cp-master-test")
            .replace("__NODE_LABEL__", "node1-slot2");

        assert!(worker.contains("SESSION_ID = \"cp-worker1-test\""));
        assert!(worker.contains("MASTER_SESSION_ID = \"cp-master-test\""));
        assert!(worker.contains("NODE_LABEL = \"node1-slot2\""));
        for placeholder in ["__SESSION_ID__", "__MASTER_SESSION_ID__", "__NODE_LABEL__", "__DURATION_MINUTES__"] {
            assert!(!worker.contains(placeholder), "unfilled {}", placeholder);
        }
    }
}
