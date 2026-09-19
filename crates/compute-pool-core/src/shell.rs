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
    res = subprocess.run(shell_cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, env=env)
    out = res.stdout if res.stdout else res.stderr
    return f"[Node 0: Master (Slot 1 - GPUs 0, 1)]\\n{out.strip()}"

def run_remote():
    remote_exec = f"export NODE_RANK=1; {shell_cmd}"
    res = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "node1", f"bash -c {shlex.quote(remote_exec)}"], capture_output=True, text=True)
    if res.returncode == 0 or (res.stdout and "Connection refused" not in res.stderr):
        out = res.stdout if res.stdout else res.stderr
        return f"[Node 1: Worker (Slot 2 - GPUs 2, 3)]\\n{out.strip()}"
    # Fallback to worker-exec HTTP daemon
    res = subprocess.run(["/usr/local/bin/worker-exec", remote_exec], capture_output=True, text=True)
    out = res.stdout if res.stdout else res.stderr
    return f"[Node 1: Worker (Slot 2 - GPUs 2, 3)]\\n{out.strip()}"

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

print("+----------------------------------------------------------------------+")
print("Cluster Tools & Commands:")
print("  • nvidia-smi        -> Real local GPU telemetry (Node 0)")
print("  • ssh node1         -> Direct SSH shell into Worker (Node 1)")
print("  • crun <command>    -> Parallel execution across all 4 GPUs")
print("  • enable-ray        -> Initialize unified Ray Cluster across all 4 GPUs")
print("  • enable-pytorch    -> Setup PyTorch multi-node rendezvous")
print("  • enable-deepspeed  -> Configure DeepSpeed multi-node hostfile")
print("+----------------------------------------------------------------------+")
'''
with open("/usr/local/bin/cluster-status", "w") as f:
    f.write(status_script)
os.chmod("/usr/local/bin/cluster-status", 0o755)

# 5. Setup helper scripts: enable-ray, enable-pytorch, enable-deepspeed
ray_script = '''#!/bin/bash
echo "[*] Initializing Ray Cluster across all 4 GPUs..."
which ray >/dev/null 2>&1 || (pip install -q "ray[default]" && ssh node1 "pip install -q 'ray[default]'")
ray stop --force >/dev/null 2>&1 || true
ssh node1 "ray stop --force >/dev/null 2>&1 || true"
export RAY_NODE_IP_ADDRESS=127.0.0.1
ray start --head --node-ip-address=127.0.0.1 --port=6379 --ray-client-server-port=10001 --dashboard-port=8265 --disable-usage-stats --num-gpus=2
ssh -f -N -R 6379:127.0.0.1:6379 -R 10001:127.0.0.1:10001 node1
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
echo "[*] Initializing PyTorch Distributed Environment..."
ssh -f -N -R 29500:127.0.0.1:29500 node1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=2
echo "[✓] PyTorch Distributed rendezvous ready on port 29500."
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
