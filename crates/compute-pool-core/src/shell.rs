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

subprocess.run([
    "bash", "-c",
    "curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
], check=True)

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

cf_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:8888", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in cf_proc.stdout:
    clean = line.strip()
    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", clean)
    if m:
        worker_url = m.group(0)
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{SESSION_ID}", data=worker_url.encode("utf-8"))
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

smi_script = '''#!/usr/bin/env python3
import subprocess, json, urllib.request, os

worker_url = ""
if os.path.exists("/kaggle/working/.cluster_worker_url"):
    with open("/kaggle/working/.cluster_worker_url") as f:
        worker_url = f.read().strip()

local_gpus = []
try:
    smi_bin = "nvidia-smi"
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
'''
with open("/usr/local/bin/cluster-smi", "w") as f:
    f.write(smi_script)
os.chmod("/usr/local/bin/cluster-smi", 0o755)

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

with open(os.path.expanduser("~/.bashrc"), "a") as f:
    f.write(f"\nexport PATH=/usr/local/bin:$PATH\n")
    f.write(f"export PS1='\\[\\033[01;32m\\]compute-pool@cluster-master\\[\\033[00m\\]:\\[\\033[01;34m\\]\\w\\[\\033[00m\\]\\$ '\n")
    f.write("alias gpus='/usr/local/bin/cluster-smi'\n")
    f.write("alias nvidia-smi='/usr/local/bin/cluster-smi'\n")
    f.write("alias watch-gpu='watch -n 1 /usr/local/bin/cluster-smi'\n")
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

def setup_cluster_worker_discovery():
    for _ in range(60):
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{WORKER_SESSION_ID}/raw?poll=1")
            with urllib.request.urlopen(req, timeout=5) as r:
                url = r.read().decode("utf-8").strip()
                m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", url)
                if m:
                    worker_url = m.group(0)
                    with open("/kaggle/working/.cluster_worker_url", "w") as f:
                        f.write(worker_url)
                    break
        except Exception:
            pass
        time.sleep(4)

threading.Thread(target=setup_cluster_worker_discovery, daemon=True).start()

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

    while match timeout_seconds {
        Some(t) => start.elapsed() < Duration::from_secs(t),
        None => true,
    } {
        let ntfy_url = format!("https://ntfy.sh/{}/raw?poll=1", session_master_id);
        if let Ok(resp) = http_client.get(&ntfy_url).send().await {
            if resp.status().is_success() {
                if let Ok(text) = resp.text().await {
                    if let Some(m) = re.find(&text) {
                        web_url = m.as_str().to_string();
                        break;
                    }
                }
            }
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

    Ok(ClusterShellInfo {
        master_username: creds1.username.clone(),
        worker_username: creds2.username.clone(),
        web_url,
        files_url,
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
