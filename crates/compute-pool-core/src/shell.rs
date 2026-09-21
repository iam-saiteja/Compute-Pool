use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::auth::load_credentials;
use crate::job::{Job, JobSpec, JobState};
use crate::kaggle::KaggleClient;
use crate::storage::{load_all_jobs, upsert_job};

pub const WORKER_BOOTSTRAP_TEMPLATE: &str = r#"
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

# 1. Download Cloudflared, Chisel, and Filebrowser
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

# 1b. Extract LUPINE server binary from GHCR image using crane (no Docker required)
# crane is a single-binary container tool from Google that can pull/export OCI images.
# The LUPINE server binary is extracted from the image layer and run directly on this node.
print("[*] Extracting LUPINE GPU-over-IP server...", flush=True)
subprocess.run(["bash", "-c",
    "curl -sfL https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz "
    "| tar -xz -C /usr/local/bin crane 2>/dev/null && chmod +x /usr/local/bin/crane || true"
], check=False)

subprocess.run(["bash", "-c",
    "mkdir -p /opt/lupine && "
    "crane export ghcr.io/lupinemachines/lupine-server:cuda-12.4-ubuntu22.04 /tmp/lupine.tar 2>/dev/null && "
    "tar -xf /tmp/lupine.tar -C /opt/lupine/ 2>/dev/null || true && "
    "find /opt/lupine -name 'lupine-server' -type f -exec chmod +x {} \\; 2>/dev/null || true"
], check=False)

lupine_server_bin = subprocess.run(
    ["bash", "-c", "find /opt/lupine -name 'lupine-server' -type f 2>/dev/null | head -1"],
    capture_output=True, text=True
).stdout.strip()

if lupine_server_bin and os.path.isfile(lupine_server_bin):
    os.chmod(lupine_server_bin, 0o755)
    lupine_lib_dir = subprocess.run(
        ["bash", "-c", "find /opt/lupine -name 'libcuda.so*' -o -name 'libnvidia-ml.so*' 2>/dev/null | head -1 | xargs -I{} dirname {} 2>/dev/null || echo /opt/lupine/lib"],
        capture_output=True, text=True
    ).stdout.strip() or "/opt/lupine/lib"
    lupine_env = os.environ.copy()
    lupine_env["LD_LIBRARY_PATH"] = f"{lupine_lib_dir}:/usr/local/cuda/lib64:" + lupine_env.get("LD_LIBRARY_PATH", "")
    subprocess.Popen(
        [lupine_server_bin, "--port", "14833"],
        env=lupine_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    print("[*] LUPINE GPU server started on Node 1 :14833 (exporting GPUs 0,1)", flush=True)
else:
    print("[!] LUPINE server binary not found — GPU fabric unavailable on Node 1", flush=True)

# Start File Browser on Worker port 8081
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
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}-worker-files", data=worker_files_url.encode("utf-8"))
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass
        break

# 2. Setup SSH Daemon and cluster keys
cluster_priv_key = ""
try:
    subprocess.run(["bash", "-c", "which sshd || (apt-get update -qq && apt-get install -y -qq openssh-server)"], check=False)
    subprocess.run(["bash", "-c", "mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh"], check=False)
    subprocess.run(["bash", "-c", "ssh-keygen -A"], check=False)
    
    subprocess.run(["bash", "-c", "echo 'PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' >> /etc/environment"], check=False)
    subprocess.run(["bash", "-c", "echo 'export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH' >> /root/.bashrc"], check=False)
    
    key_path = "/root/.ssh/cluster_key"
    if not os.path.exists(key_path):
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", key_path, "-C", "compute-pool-cluster"], check=True)
    subprocess.run(["bash", "-c", f"cat {key_path}.pub >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys"], check=True)
    
    with open(key_path, "r") as f:
        cluster_priv_key = f.read().strip()
    
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
        "-o", "ClientAliveInterval=15"
    ])
    print("[*] SSH daemon active on port 2222.", flush=True)
except Exception as e:
    print(f"[!] SSH setup notice: {e}", flush=True)

# 3. HTTP Worker & Exec Daemon on port 8889
class WorkerExecHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "OK", "node": NODE_LABEL}).encode("utf-8"))
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
        elif self.path == "/stop":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            def _die():
                time.sleep(0.5)
                os.system("pkill -9 -f cloudflared; pkill -9 -f chisel; kill -9 -1")
            threading.Thread(target=_die).start()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

httpd = ThreadedTCPServer(("0.0.0.0", 8889), WorkerExecHandler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()

# 4. Start Chisel Server on port 8888 for bidirectional TCP bridging
chisel_proc = subprocess.Popen(
    ["/usr/local/bin/chisel", "server", "--port", "8888", "--reverse"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
time.sleep(1)

# 5. Expose Chisel server via Cloudflare tunnel
cf_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:8888", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

chisel_tunnel_url = ""
for line in cf_proc.stdout:
    clean = line.strip()
    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", clean)
    if m:
        chisel_tunnel_url = m.group(0)
        break

# 6. Expose HTTP Exec server via second Cloudflare tunnel
cf_http_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:8889", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

http_tunnel_url = ""
for line in cf_http_proc.stdout:
    clean = line.strip()
    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", clean)
    if m:
        http_tunnel_url = m.group(0)
        break

# 7. Publish connection payload to ntfy for Master
payload = json.dumps({
    "status": "READY",
    "node": NODE_LABEL,
    "chisel_url": chisel_tunnel_url,
    "http_url": http_tunnel_url,
    "files_url": worker_files_url,
    "ssh_key": cluster_priv_key
})

for _ in range(5):
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}", data=payload.encode("utf-8"))
        urllib.request.urlopen(req, timeout=10)
        break
    except Exception:
        time.sleep(2)

print(f"[*] Worker registered. Chisel: {chisel_tunnel_url}, Files: {worker_files_url}", flush=True)

# 8. Keep worker alive until STOP signal
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

# 1. Download core binaries: ttyd, filebrowser, cloudflared, chisel
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

# 2. Setup crun (cluster runner) script - parallel multi-node command executor
crun_script = '''#!/usr/bin/env python3
import sys, os, subprocess, concurrent.futures, shlex

if len(sys.argv) < 2:
    print("Usage: crun <command> [args...]")
    print("Example: crun nvidia-smi")
    print("Example: crun python3 train.py")
    print("Example: crun \\"pip install tabfm\\"")
    sys.exit(1)

raw_args = sys.argv[1:]
if len(raw_args) == 1:
    shell_cmd = raw_args[0]
else:
    shell_cmd = " ".join(raw_args)

# Auto-sync referenced local files to Node 1 if needed
for token in shell_cmd.split():
    if os.path.isfile(token) and not token.startswith("-"):
        fname = os.path.basename(token)
        subprocess.run(["scp", "-o", "ConnectTimeout=3", token, f"node1:/kaggle/working/{fname}"], capture_output=True)

def run_local():
    env = os.environ.copy()
    env["NODE_RANK"] = "0"
    try:
        res = subprocess.run(shell_cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, env=env)
        out = (res.stdout or "").strip()
        if res.stderr:
            out = (out + "\\n[stderr]\\n" + res.stderr.strip()).strip()
    except Exception as exc:
        return f"[Node 0: Master (Slot 1 - GPUs 0, 1)]\\n[! Runner error: {exc}]"
    if not out and res.returncode == 0:
        out = "[✓ Completed successfully (Exit Code 0)]"
    elif not out:
        out = f"[! Exited with code {res.returncode}]"
    return f"[Node 0: Master (Slot 1 - GPUs 0, 1)]\\n{out}"

def run_remote():
    remote_exec = f"export NODE_RANK=1; {shell_cmd}"
    try:
        res = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=10", "node1", f"bash -c {shlex.quote(remote_exec)}"], capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return "[Node 1: Worker (Slot 2 - GPUs 2, 3)]\\n[! Remote command timed out after 180 seconds]"
    except Exception as exc:
        return f"[Node 1: Worker (Slot 2 - GPUs 2, 3)]\\n[! Runner error: {exc}]"
    if res.returncode == 0 or (res.stdout and "Connection refused" not in res.stderr):
        out = (res.stdout or "").strip()
        if res.stderr:
            out = (out + "\\n[stderr]\\n" + res.stderr.strip()).strip()
        if not out and res.returncode == 0:
            out = "[✓ Completed successfully (Exit Code 0)]"
        elif not out:
            out = f"[! Exited with code {res.returncode}]"
        return f"[Node 1: Worker (Slot 2 - GPUs 2, 3)]\\n{out}"
    # Fallback to worker-exec HTTP daemon
    res = subprocess.run(["/usr/local/bin/worker-exec", remote_exec], capture_output=True, text=True)
    out = (res.stdout or "").strip()
    if res.stderr:
        out = (out + "\\n[stderr]\\n" + res.stderr.strip()).strip()
    if not out and res.returncode == 0:
        out = "[✓ Completed successfully (Exit Code 0)]"
    elif not out:
        out = f"[! Exited with code {res.returncode}]"
    return f"[Node 1: Worker (Slot 2 - GPUs 2, 3)]\\n{out}"

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    f0 = ex.submit(run_local)
    f1 = ex.submit(run_remote)
    out0 = f0.result()
    out1 = f1.result()

print(out0)
print("\\n" + "="*80 + "\\n")
print(out1)
'''
with open("/usr/local/bin/crun", "w") as f:
    f.write(crun_script)
os.chmod("/usr/local/bin/crun", 0o755)

# 3. Setup worker-exec (HTTP fallback runner)
worker_exec_script = '''#!/usr/bin/env python3
import sys, urllib.request, json, os

cmd = " ".join(sys.argv[1:])
if not cmd:
    sys.exit(0)

worker_url = ""
if os.path.exists("/kaggle/working/.cluster_worker_url"):
    with open("/kaggle/working/.cluster_worker_url") as f:
        worker_url = f.read().strip()

if not worker_url:
    sys.stderr.write("Worker node not yet peered.\\n")
    sys.exit(1)

try:
    req = urllib.request.Request(
        f"{worker_url}/exec",
        data=json.dumps({"cmd": cmd}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read().decode("utf-8"))
        if resp.get("stdout"):
            sys.stdout.write(resp["stdout"])
        if resp.get("stderr"):
            sys.stderr.write(resp["stderr"])
        sys.exit(resp.get("exit_code", 0))
except Exception as e:
    sys.stderr.write(f"Remote exec error: {e}\\n")
    sys.exit(1)
'''
with open("/usr/local/bin/worker-exec", "w") as f:
    f.write(worker_exec_script)
os.chmod("/usr/local/bin/worker-exec", 0o755)

# 4. Setup cluster-status script
status_script = '''#!/usr/bin/env python3
import subprocess, sys

print("+----------------------------------------------------------------------+")
print("|            Compute Pool 4-GPU Cluster Infrastructure Status          |")
print("+----------------------------------------------------------------------+")
try:
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    gpus_node0 = [l.strip() for l in smi.splitlines() if l.strip()]
    print(f"[*] Node 0 (Master, Slot 1): Online ({len(gpus_node0)} GPUs: {', '.join(gpus_node0)})")
except Exception as e:
    print(f"[!] Node 0 (Master, Slot 1): Query failed ({e})")

try:
    res = subprocess.run(["ssh", "-o", "ConnectTimeout=3", "node1", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"], capture_output=True, text=True, timeout=5)
    if res.returncode == 0 and res.stdout.strip():
        gpus_node1 = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
        print(f"[*] Node 1 (Worker, Slot 2): Online ({len(gpus_node1)} GPUs: {', '.join(gpus_node1)})")
        print(f"[*] Inter-Node Fabric: Active (SSH & Bidirectional Network Bridge)")
    else:
        # Check HTTP fallback
        res_http = subprocess.run(["/usr/local/bin/worker-exec", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if res_http.returncode == 0 and res_http.stdout.strip():
            gpus_node1 = [l.strip() for l in res_http.stdout.strip().splitlines() if l.strip()]
            print(f"[*] Node 1 (Worker, Slot 2): Online ({len(gpus_node1)} GPUs: {', '.join(gpus_node1)})")
            print(f"[*] Inter-Node Fabric: Active (HTTP RPC Bridge)")
        else:
            print("[!] Node 1 (Worker, Slot 2): Connecting to cluster...")
except Exception as e:
    print(f"[!] Node 1 (Worker, Slot 2): Connecting ({e})")

# LUPINE GPU fabric status
try:
    import socket as _s
    lupine_node0_up = False
    lupine_node1_up = False
    for port, label in [(14833, "node0"), (14834, "node1")]:
        try:
            c = _s.create_connection(("127.0.0.1", port), timeout=1)
            c.close()
            if label == "node0": lupine_node0_up = True
            else: lupine_node1_up = True
        except Exception:
            pass
    if lupine_node0_up or lupine_node1_up:
        both = lupine_node0_up and lupine_node1_up
        gpus_up = (2 if lupine_node0_up else 0) + (2 if lupine_node1_up else 0)
        status = "Active" if both else "Partial"
        print(f"[*] LUPINE GPU Fabric: {status} ({gpus_up}/4 GPUs via IP fabric)")
        print(f"    Node 0 :14833 -> {'UP' if lupine_node0_up else 'DOWN'}   Node 1 :14834 -> {'UP' if lupine_node1_up else 'DOWN'}")
    else:
        print("[*] LUPINE GPU Fabric: Inactive (run: enable-lupine)")
except Exception:
    print("[*] LUPINE GPU Fabric: Unknown")

print("+----------------------------------------------------------------------+")
print("Cluster Tools & Commands:")
print("  • nvidia-smi        -> Real local GPU telemetry (Node 0)")
print("  • ssh node1         -> Direct SSH shell into Worker (Node 1)")
print("  • crun <command>    -> Parallel execution across all 4 GPUs")
print("  • enable-lupine     -> Initialize LUPINE GPU-over-IP fabric (4 GPUs via IP)")
print("  • enable-ray        -> Initialize unified Ray Cluster across all 4 GPUs")
print("  • enable-pytorch    -> Setup PyTorch multi-node rendezvous")
print("  • enable-deepspeed  -> Configure DeepSpeed multi-node hostfile")
print("  • lupine-env <cmd>  -> Run any CUDA program through LUPINE fabric")
print("+----------------------------------------------------------------------+")
'''
with open("/usr/local/bin/cluster-status", "w") as f:
    f.write(status_script)
os.chmod("/usr/local/bin/cluster-status", 0o755)

# 5. Setup helper scripts: enable-ray, enable-pytorch, enable-deepspeed
ray_script = '''#!/bin/bash
echo "[*] Initializing Ray Cluster across all 4 GPUs..."
which ray >/dev/null 2>&1 || (pip install -q "ray[default]" && ssh node1 "pip install -q 'ray[default]'")
pkill -9 -f ray 2>/dev/null || true
ssh node1 "pkill -9 -f ray 2>/dev/null || true"
pkill -9 -f "ssh -f -N -R 6379" 2>/dev/null || true

export RAY_NODE_IP_ADDRESS=127.0.0.1
ray start --head --node-ip-address=127.0.0.1 --port=6379 --ray-client-server-port=10001 --dashboard-port=8265 --disable-usage-stats --num-gpus=2
ssh -f -N -o ServerAliveInterval=10 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes -R 6379:127.0.0.1:6379 -R 10001:127.0.0.1:10001 node1
sleep 1
ssh node1 "export RAY_NODE_IP_ADDRESS=127.0.0.1; ray start --address=127.0.0.1:6379 --node-ip-address=127.0.0.1 --disable-usage-stats --num-gpus=2"
echo ""
echo "[✓] Ray Cluster Active! 4x Tesla T4 GPUs (60 GB Total VRAM) pooled."
echo "    In Python: import ray; ray.init(address='auto')"
echo "    Check cluster: ray status"
'''
with open("/usr/local/bin/enable-ray", "w") as f:
    f.write(ray_script)
os.chmod("/usr/local/bin/enable-ray", 0o755)

pytorch_script = '''#!/bin/bash
set -e
echo "[*] Initializing PyTorch Distributed Environment..."

# Re-running this helper must replace the old reverse tunnel, not stack another one.
pkill -f 'ssh.*-R 29500:127.0.0.1:29500' 2>/dev/null || true
ssh -o ExitOnForwardFailure=yes -f -N \\
    -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \\
    -R 29500:127.0.0.1:29500 node1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=4
export CP_MASTER_PORT=29500
echo "[✓] PyTorch Distributed rendezvous ready on port 29500."
echo "    Use Gloo for the SSH-tunneled cluster; NCCL requires direct peer networking."
'''
with open("/usr/local/bin/enable-pytorch", "w") as f:
    f.write(pytorch_script)
os.chmod("/usr/local/bin/enable-pytorch", 0o755)

deepspeed_script = '''#!/bin/bash
echo "[*] Setting up DeepSpeed multi-node hostfile..."
mkdir -p /root/.ssh
cat << 'EOF' > /root/.ssh/hostfile
localhost slots=2
node1 slots=2
EOF
cat << 'EOF' > /kaggle/working/hostfile
localhost slots=2
node1 slots=2
EOF
echo "[✓] DeepSpeed hostfile configured (4 GPU slots across 2 nodes)."
echo "    Run: deepspeed --hostfile /root/.ssh/hostfile <script.py>"
'''
with open("/usr/local/bin/enable-deepspeed", "w") as f:
    f.write(deepspeed_script)
os.chmod("/usr/local/bin/enable-deepspeed", 0o755)

# 5b. Setup enable-lupine: GPU-over-IP fabric via LUPINE (opt-in, like enable-ray)
lupine_script = '''#!/bin/bash
# enable-lupine: Initialize LUPINE GPU-over-IP fabric across all 4 T4 GPUs
# Uses GHCR anonymous OCI bearer-token pull (curl only, no Docker/crane needed).
# LUPINE server on Node 0 (local :14833) + Node 1 (tunneled -> local :14834)
# Combined: LUPINE_SERVER=127.0.0.1:14833,127.0.0.1:14834 = 4 GPUs visible
echo "========================================================================"
echo "  Initializing LUPINE GPU-over-IP Fabric (4x Tesla T4)"
echo "========================================================================"

# Helper: pull an OCI image from GHCR using anonymous bearer token (curl only)
lupine_pull_from_ghcr() {
    local IMAGE="lupinemachines/lupine-server"
    local TAG="$1"
    local DESTDIR="$2"
    echo "[*] Trying GHCR tag: $TAG"

    # A: Get anonymous pull token
    local TOKEN
    TOKEN=$(curl -sf \
        "https://ghcr.io/token?scope=repository:${IMAGE}:pull&service=ghcr.io" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('token',''))" 2>/dev/null)
    if [ -z "$TOKEN" ]; then
        echo "[!] Could not get GHCR token"
        return 1
    fi

    # B: Fetch manifest (try OCI v1 then Docker v2)
    local MANIFEST
    MANIFEST=$(curl -sf \
        -H "Authorization: Bearer $TOKEN" \
        -H "Accept: application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json" \
        "https://ghcr.io/v2/${IMAGE}/manifests/${TAG}" 2>/dev/null)
    if [ -z "$MANIFEST" ]; then
        echo "[!] No manifest for tag: $TAG (tag may not exist or auth required)"
        return 1
    fi

    # Handle OCI image index (multi-platform) -> resolve linux/amd64
    local MEDIATYPE
    MEDIATYPE=$(echo "$MANIFEST" | python3 -c "import sys,json; print(json.load(sys.stdin).get('mediaType',''))" 2>/dev/null)
    if echo "$MEDIATYPE" | grep -q "index"; then
        local AMD64_DIGEST
        AMD64_DIGEST=$(echo "$MANIFEST" | python3 -c "
import sys,json
m=json.load(sys.stdin)
for mf in m.get('manifests',[]):
    p=mf.get('platform',{})
    if p.get('os')=='linux' and p.get('architecture')=='amd64':
        print(mf.get('digest',''))
        break
" 2>/dev/null)
        if [ -n "$AMD64_DIGEST" ]; then
            MANIFEST=$(curl -sf \
                -H "Authorization: Bearer $TOKEN" \
                -H "Accept: application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json" \
                "https://ghcr.io/v2/${IMAGE}/manifests/${AMD64_DIGEST}" 2>/dev/null)
        fi
    fi

    echo "[*] Manifest OK for $TAG — downloading layers..."
    mkdir -p "$DESTDIR"

    # C: Extract layer digests and download each layer
    local DIGESTS
    DIGESTS=$(echo "$MANIFEST" | python3 -c "
import sys,json
m=json.load(sys.stdin)
for l in m.get('layers', m.get('fsLayers',[])):
    d=l.get('digest',l.get('blobSum',''))
    if d: print(d)
" 2>/dev/null)

    if [ -z "$DIGESTS" ]; then
        echo "[!] No layers found in manifest"
        return 1
    fi

    local N=0
    while IFS= read -r DIGEST; do
        N=$((N+1))
        echo "[*] Downloading layer $N: ${DIGEST:0:24}..."
        curl -sfL \
            -H "Authorization: Bearer $TOKEN" \
            "https://ghcr.io/v2/${IMAGE}/blobs/${DIGEST}" \
            -o "/tmp/lupine_l${N}.tar.gz" 2>/dev/null
        if [ -f "/tmp/lupine_l${N}.tar.gz" ]; then
            tar -xzf "/tmp/lupine_l${N}.tar.gz" -C "$DESTDIR" 2>/dev/null || \
            tar -xf  "/tmp/lupine_l${N}.tar.gz" -C "$DESTDIR" 2>/dev/null || true
            rm -f "/tmp/lupine_l${N}.tar.gz"
        fi
    done <<< "$DIGESTS"

    find "$DESTDIR" -name "lupine-server" -type f -exec chmod +x {} \; 2>/dev/null || true
    local BIN
    BIN=$(find "$DESTDIR" -name "lupine-server" -type f 2>/dev/null | head -1)
    if [ -n "$BIN" ]; then
        echo "[*] Found binary: $BIN"
        return 0
    fi
    echo "[!] lupine-server not found in layers for $TAG"
    return 1
}

# --- Step 1: Extract LUPINE server binary ---
LUPINE_BIN=$(find /opt/lupine -name "lupine-server" -type f 2>/dev/null | head -1)
if [ -z "$LUPINE_BIN" ]; then
    echo "[*] Fetching LUPINE server from GHCR (anonymous OCI pull, curl only)..."
    mkdir -p /opt/lupine
    # Tags in order: verified existing GHCR tags
    for TAG in v0.4.0-cuda-12.4.1-ubuntu22.04-amd64 cuda-12.4.1-ubuntu22.04-amd64 cuda-12.4.1-ubuntu22.04 v0.4.0-cuda-12.6.2-ubuntu24.04-amd64 cuda-12.6.2-ubuntu24.04-amd64; do
        lupine_pull_from_ghcr "$TAG" /opt/lupine && break || true
    done
    LUPINE_BIN=$(find /opt/lupine -name "lupine-server" -type f 2>/dev/null | head -1)
fi

if [ -z "$LUPINE_BIN" ]; then
    echo ""
    echo "[!] LUPINE server binary not found. Diagnostic info:"
    echo "    /opt/lupine contents (top 30):"
    find /opt/lupine 2>/dev/null | head -30 || echo "      (empty)"
    echo ""
    echo "    Manual debug commands:"
    echo "      TOKEN=\$(curl -sf 'https://ghcr.io/token?scope=repository:lupinemachines/lupine-server:pull&service=ghcr.io' | python3 -c \"import sys,json; print(json.load(sys.stdin).get('token',''))\")"
    echo "      echo \${TOKEN:0:60}"
    echo "      curl -sH \"Authorization: Bearer \$TOKEN\" 'https://ghcr.io/v2/lupinemachines/lupine-server/tags/list'"
    exit 1
fi
chmod +x "$LUPINE_BIN"

# Locate shim libraries (OCI image extracts to nested path like /opt/lupine/usr/local/lib)
LUPINE_LIB=$(find /opt/lupine \( -name "libcuda.so*" -o -name "libnvidia-ml.so*" \) 2>/dev/null \
    | head -1 | xargs -I{} dirname {} 2>/dev/null)
if [ -z "$LUPINE_LIB" ]; then
    for C in /opt/lupine/usr/local/lib /opt/lupine/opt/lupine/lib /opt/lupine/lib; do
        [ -d "$C" ] && { LUPINE_LIB="$C"; break; }
    done
fi
LUPINE_LIB="${LUPINE_LIB:-/opt/lupine/usr/local/lib}"
export LD_LIBRARY_PATH="${LUPINE_LIB}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
echo "[*] Server binary: $LUPINE_BIN"
echo "[*] Shim libs dir: $LUPINE_LIB"

# --- Step 2: Start LUPINE server on Node 0 (GPUs 0,1) ---
pkill -f "lupine-server" 2>/dev/null || true
sleep 1
"$LUPINE_BIN" --port 14833 &
LUPINE0_PID=$!
echo "[*] LUPINE server started on Node 0 :14833 [pid=$LUPINE0_PID]"

# --- Step 3: SSH local-forward Node 1 LUPINE port -> local :14834 ---
pkill -f "ssh.*14834" 2>/dev/null || true
sleep 1
ssh -f -N \
    -o ServerAliveInterval=10 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes \
    -o ExitOnForwardFailure=no -o StrictHostKeyChecking=no \
    -L 14834:127.0.0.1:14833 node1 2>/dev/null
echo "[*] LUPINE tunnel: Node 1 GPU server -> local :14834"

# --- Step 4: Fetch LUPINE client shim from running server ---
sleep 3
mkdir -p /opt/lupine/lib
curl -sf "http://127.0.0.1:14833/.well-known/lupine/client/v1/linux/amd64" \
    -o /opt/lupine/lib/libcuda.so.1 2>/dev/null \
    && chmod +x /opt/lupine/lib/libcuda.so.1 \
    && echo "[*] Client shim fetched -> /opt/lupine/lib/libcuda.so.1" \
    || echo "[*] Client shim not needed (libs extracted from image)"

# Write env file
cat > /opt/lupine/env.sh << 'ENVEOF'
export LUPINE_SERVER=127.0.0.1:14833,127.0.0.1:14834
export LD_LIBRARY_PATH=/opt/lupine/lib:/opt/lupine/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
ENVEOF

# --- Step 5: Write verification script ---
cat > /kaggle/working/nccl_allreduce_test.py << 'PYEOF'
#!/usr/bin/env python3
"""4-GPU LUPINE fabric verification. Run: lupine-env python3 /kaggle/working/nccl_allreduce_test.py"""
import os, sys, time, torch

def main():
    n = torch.cuda.device_count()
    print(f"[*] CUDA devices: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"    GPU {i}: {p.name} ({p.total_memory//1024**3} GB)")
    if n < 4:
        print(f"[!] Got {n} GPUs, expected 4. Run enable-lupine first."); sys.exit(1)
    print(f"[OK] {n}/4 GPUs visible via LUPINE fabric")
    tensors = [torch.ones(4096, device=f"cuda:{i}") * float(i+1) for i in range(n)]
    print("[*] Cross-GPU copy test...")
    for i in range(n):
        j = (i+1) % n
        dst = tensors[i].to(f"cuda:{j}")
        torch.cuda.synchronize(j)
        assert abs(dst.sum().item() - tensors[i].sum().item()) < 1, f"copy {i}->{j} FAIL"
        print(f"    [OK] cuda:{i} -> cuda:{j}")
    print("[*] Simulated all_reduce...")
    expected = float(sum(range(1, n+1)) * 4096)
    for i in range(n):
        t = sum(tensors[j].to(f"cuda:{i}") for j in range(n))
        torch.cuda.synchronize(i)
        assert abs(t.sum().item() - expected) < 1, f"reduce GPU{i} FAIL"
        print(f"    [OK] GPU {i}: {t.sum().item():.0f}")
    print(f"\n[OK] 4-GPU LUPINE fabric VERIFIED")
    print(f"     LUPINE_SERVER={os.environ.get('LUPINE_SERVER','not set')}")
    print(f"     Next: lupine-env torchrun --nproc_per_node={n} train.py")
if __name__ == "__main__":
    main()
PYEOF
chmod +x /kaggle/working/nccl_allreduce_test.py
scp -o ConnectTimeout=5 /kaggle/working/nccl_allreduce_test.py node1:/kaggle/working/ 2>/dev/null || true

# --- Step 6: Verify ---
echo ""
echo "========================================================================"
echo "  Verifying GPU visibility via LUPINE fabric..."
echo "========================================================================"
sleep 2
LUPINE_SERVER=127.0.0.1:14833,127.0.0.1:14834 \
    LD_LIBRARY_PATH="${LUPINE_LIB}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
    nvidia-smi -L 2>&1 || true

echo ""
echo "========================================================================"
echo "  LUPINE GPU-over-IP Fabric Active"
echo "========================================================================"
echo "  Node 0 GPUs (0,1) -> LUPINE server :14833 (local)"
echo "  Node 1 GPUs (2,3) -> LUPINE server :14834 (tunneled)"
echo ""
echo "  Verify and test:"
echo "    lupine-env python3 /kaggle/working/nccl_allreduce_test.py"
echo "    lupine-env nvidia-smi -L"
echo "    lupine-env torchrun --nproc_per_node=4 train.py"
echo "========================================================================"
'''
with open("/usr/local/bin/enable-lupine", "w") as f:
    f.write(lupine_script)
os.chmod("/usr/local/bin/enable-lupine", 0o755)

# 5c. Setup lupine-env: wrapper that injects LUPINE env vars into any CUDA program
lupine_env_script = '''#!/bin/bash
# lupine-env: Run any command with the LUPINE GPU-over-IP environment active
# Usage: lupine-env python3 train.py
#        lupine-env torchrun --nproc_per_node=4 train.py
#        lupine-env nvidia-smi -L
if [ $# -eq 0 ]; then
    echo "Usage: lupine-env <command> [args...]"
    echo "Example: lupine-env python3 train.py"
    echo "Example: lupine-env nvidia-smi -L"
    echo ""
    echo "Environment injected:"
    echo "  LUPINE_SERVER=127.0.0.1:14833,127.0.0.1:14834"
    echo "  LD_LIBRARY_PATH=/opt/lupine/lib:/usr/local/cuda/lib64:..."
    exit 0
fi
LUPINE_LIB=$(find /opt/lupine -name "libcuda.so*" -o -name "libnvidia-ml.so*" 2>/dev/null | head -1 | xargs -I{} dirname {} 2>/dev/null || echo "/opt/lupine/lib")
export LUPINE_SERVER=127.0.0.1:14833,127.0.0.1:14834
export LD_LIBRARY_PATH="${LUPINE_LIB}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
exec "$@"
'''
with open("/usr/local/bin/lupine-env", "w") as f:
    f.write(lupine_env_script)
os.chmod("/usr/local/bin/lupine-env", 0o755)

# 6. Setup stop script
stop_script = '''#!/bin/bash
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

# 7. Configure shell environment (.bashrc) - NEVER ALIAS nvidia-smi
with open(os.path.expanduser("~/.bashrc"), "a") as f:
    f.write("\nexport PATH=/usr/local/bin:$PATH\n")
    f.write("export MASTER_ADDR=127.0.0.1\n")
    f.write("export MASTER_PORT=29500\n")
    f.write("export WORLD_SIZE=2\n")
    f.write("export CLUSTER_NODES=2\n")
    f.write("export GPUS_PER_NODE=2\n")
    f.write("export TOTAL_GPUS=4\n")
    f.write("export PS1='\\[\\033[01;32m\\]compute-pool@cluster-master\\[\\033[00m\\]:\\[\\033[01;34m\\]\\w\\[\\033[00m\\]\\$ '\n")
    f.write("alias halt='/usr/local/bin/stop'\n")
    f.write("alias exit='/usr/local/bin/stop'\n")

# 8. Start ttyd Web Terminal and File Browser
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

# 9. Start Cloudflare Tunnel for Web Terminal
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

# 10. Start Cloudflare Tunnel for File Browser
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

# 11. Inter-Node Discovery & Peering Thread
def setup_cluster_peering():
    worker_data = None
    for _ in range(120):
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{WORKER_SESSION_ID}/raw?poll=1")
            with urllib.request.urlopen(req, timeout=5) as r:
                raw = r.read().decode("utf-8").strip()
                if raw.startswith("{") and "chisel_url" in raw:
                    worker_data = json.loads(raw)
                    break
        except Exception:
            pass
        time.sleep(3)

    if not worker_data:
        return

    # Save worker HTTP URL
    http_url = worker_data.get("http_url", "")
    if http_url:
        with open("/kaggle/working/.cluster_worker_url", "w") as f:
            f.write(http_url)

    # Setup SSH Key
    ssh_key = worker_data.get("ssh_key", "")
    if ssh_key:
        os.makedirs("/root/.ssh", mode=0o700, exist_ok=True)
        with open("/root/.ssh/cluster_key", "w") as f:
            f.write(ssh_key + "\n")
        os.chmod("/root/.ssh/cluster_key", 0o600)

        with open("/root/.ssh/config", "w") as f:
            f.write("Host node1 worker\n")
            f.write("    HostName 127.0.0.1\n")
            f.write("    Port 2222\n")
            f.write("    User root\n")
            f.write("    IdentityFile /root/.ssh/cluster_key\n")
            f.write("    StrictHostKeyChecking no\n")
            f.write("    UserKnownHostsFile /dev/null\n")
            f.write("    LogLevel ERROR\n")

    # Connect Chisel Client (bridges port 2222 for SSH - zero port collisions)
    chisel_url = worker_data.get("chisel_url", "")
    if chisel_url:
        subprocess.Popen([
            "/usr/local/bin/chisel", "client", chisel_url,
            "2222:127.0.0.1:2222"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Notify CLI that peering is established
    time.sleep(3)
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}-peered", data=b"PEERED_OK")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

threading.Thread(target=setup_cluster_peering, daemon=True).start()

# 11. Keep Master alive until STOP signal
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterShellInfo {
    pub master_username: String,
    pub worker_username: String,
    pub web_url: String,
    pub files_url: String,
    pub worker_files_url: String,
    pub master_ref: String,
    pub worker_ref: String,
    pub duration_minutes: u32,
}

fn shell_slug_for_slot(slot: usize) -> String {
    format!("interactive-gpu-terminal-s{}", slot)
}

pub async fn launch_gpu_shell(
    slot: usize,
    duration_minutes: u32,
    timeout_seconds: Option<u64>,
) -> Result<ShellInfo> {
    let creds = load_credentials(slot)?
        .ok_or_else(|| anyhow::anyhow!("No credentials for slot {}. Run `compute-pool login --slot {}`", slot, slot))?;

    let random_suffix: String = (0..8).map(|_| {
        let chars = b"abcdefghijklmnopqrstuvwxyz0123456789";
        chars[fastrand::usize(..chars.len())] as char
    }).collect();

    let session_id = format!("cp-shell-s{}-{}", slot, random_suffix);
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
    client.push_kernel(&slug, &script, true, Some("nvidia-tesla-t4")).await
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

pub async fn launch_cluster_shell(
    duration_minutes: u32,
    timeout_seconds: Option<u64>,
) -> Result<ClusterShellInfo> {
    let creds1 = load_credentials(1)?
        .ok_or_else(|| anyhow::anyhow!("Slot 1 not configured. Run `compute-pool login --slot 1`"))?;
    let creds2 = load_credentials(2)?
        .ok_or_else(|| anyhow::anyhow!("Slot 2 not configured. Run `compute-pool login --slot 2`"))?;

    let random_suffix: String = (0..8).map(|_| {
        let chars = b"abcdefghijklmnopqrstuvwxyz0123456789";
        chars[fastrand::usize(..chars.len())] as char
    }).collect();

    let session_master_id = format!("cp-master-{}", random_suffix);
    let session_worker_id = format!("cp-worker-{}", random_suffix);

    // Register jobs in local state
    let mut j_master = Job::new(
        "job-cluster-master".to_string(),
        JobSpec {
            name: "cluster-master-node0".to_string(),
            script: format!("session_id:{}", session_master_id),
            gpu: true,
            gpu_memory_gb: 30.0,
            max_runtime_hours: (duration_minutes as f64) / 60.0,
            checkpointable: false,
            max_retries: 0,
        },
    );
    j_master.state = JobState::Running;
    j_master.assigned_slot = Some(serde_json::json!(1));
    j_master.assigned_username = Some(creds1.username.clone());
    j_master.kaggle_kernel_slug = Some(shell_slug_for_slot(1));
    upsert_job(&j_master)?;

    let mut j_worker = Job::new(
        "job-cluster-worker".to_string(),
        JobSpec {
            name: "cluster-worker-node1".to_string(),
            script: format!("session_id:{}", session_worker_id),
            gpu: true,
            gpu_memory_gb: 30.0,
            max_runtime_hours: (duration_minutes as f64) / 60.0,
            checkpointable: false,
            max_retries: 0,
        },
    );
    j_worker.state = JobState::Running;
    j_worker.assigned_slot = Some(serde_json::json!(2));
    j_worker.assigned_username = Some(creds2.username.clone());
    j_worker.kaggle_kernel_slug = Some(shell_slug_for_slot(2));
    upsert_job(&j_worker)?;

    let master_script = MASTER_BOOTSTRAP_TEMPLATE
        .replace("__SESSION_ID__", &session_master_id)
        .replace("__WORKER_SESSION_ID__", &session_worker_id)
        .replace("__DURATION_MINUTES__", &duration_minutes.to_string());

    let worker_script = WORKER_BOOTSTRAP_TEMPLATE
        .replace("__SESSION_ID__", &session_worker_id)
        .replace("__DURATION_MINUTES__", &duration_minutes.to_string());

    let client1 = KaggleClient::new(&creds1.username, &creds1.key);
    let client2 = KaggleClient::new(&creds2.username, &creds2.key);

    let slug1 = shell_slug_for_slot(1);
    let slug2 = shell_slug_for_slot(2);
    let (m_res, w_res) = tokio::join!(
        client1.push_kernel(&slug1, &master_script, true, Some("nvidia-tesla-t4")),
        client2.push_kernel(&slug2, &worker_script, true, Some("nvidia-tesla-t4"))
    );

    m_res.context("Failed to push Master cluster kernel")?;
    w_res.context("Failed to push Worker cluster kernel")?;

    let http_client = reqwest::Client::new();
    let start = Instant::now();
    let re = Regex::new(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")?;
    let mut web_url = String::new();
    let mut worker_online = false;
    let mut cluster_peered = false;

    while match timeout_seconds {
        Some(t) => start.elapsed() < Duration::from_secs(t),
        None => true,
    } {
        // Poll Master Web Terminal URL
        if web_url.is_empty() {
            let ntfy_url = format!("https://ntfy.sh/{}/raw?poll=1", session_master_id);
            if let Ok(resp) = http_client.get(&ntfy_url).send().await {
                if resp.status().is_success() {
                    if let Ok(text) = resp.text().await {
                        if let Some(m) = re.find(&text) {
                            web_url = m.as_str().to_string();
                        }
                    }
                }
            }
        }

        // Poll Worker Online
        if !worker_online {
            let ntfy_w_url = format!("https://ntfy.sh/{}/raw?poll=1", session_worker_id);
            if let Ok(resp) = http_client.get(&ntfy_w_url).send().await {
                if resp.status().is_success() {
                    if let Ok(text) = resp.text().await {
                        if text.contains("READY") || text.contains("chisel_url") {
                            worker_online = true;
                        }
                    }
                }
            }
        }

        // Poll Peered status
        if !cluster_peered {
            let ntfy_p_url = format!("https://ntfy.sh/{}-peered/raw?poll=1", session_master_id);
            if let Ok(resp) = http_client.get(&ntfy_p_url).send().await {
                if resp.status().is_success() {
                    if let Ok(text) = resp.text().await {
                        if text.contains("PEERED_OK") {
                            cluster_peered = true;
                        }
                    }
                }
            }
        }

        // Both nodes are online and web_url is ready
        if !web_url.is_empty() && (cluster_peered || worker_online) {
            break;
        }

        tokio::time::sleep(Duration::from_secs(3)).await;
    }

    if web_url.is_empty() {
        anyhow::bail!("Timed out waiting for Master Cluster Web Terminal connection.");
    }

    let mut files_url = String::new();
    for _ in 0..5 {
        let ntfy_files_url = format!("https://ntfy.sh/{}-files/raw?poll=1", session_master_id);
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

    let mut worker_files_url = String::new();
    for _ in 0..5 {
        let ntfy_wfiles_url = format!("https://ntfy.sh/{}-worker-files/raw?poll=1", session_worker_id);
        if let Ok(resp) = http_client.get(&ntfy_wfiles_url).send().await {
            if resp.status().is_success() {
                if let Ok(text) = resp.text().await {
                    if let Some(m) = re.find(&text) {
                        worker_files_url = m.as_str().to_string();
                        break;
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    Ok(ClusterShellInfo {
        master_username: creds1.username.clone(),
        worker_username: creds2.username.clone(),
        web_url,
        files_url,
        worker_files_url,
        master_ref: format!("{}/{}", creds1.username, shell_slug_for_slot(1)),
        worker_ref: format!("{}/{}", creds2.username, shell_slug_for_slot(2)),
        duration_minutes,
    })
}

pub async fn stop_gpu_shell(slot: Option<usize>) -> Result<()> {
    let slots_to_stop: Vec<usize> = match slot {
        Some(s) => vec![s],
        None => vec![1, 2],
    };

    let stop_script = "import sys\nprint('Shell terminated by user.')\nsys.exit(0)\n";
    let http_client = reqwest::Client::new();
    let all_jobs = load_all_jobs().unwrap_or_default();

    for s in slots_to_stop {
        // Send ntfy STOP signals to any active sessions for this slot
        for mut j in all_jobs.clone() {
            let is_slot_match = j.get_slot_number() == Some(s)
                || j.id.contains(&format!("s{}", s))
                || (s == 1 && j.id == "job-cluster-master")
                || (s == 2 && j.id == "job-cluster-worker");

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
                let _ = client.push_kernel(&slug, stop_script, false, None).await;
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
}
