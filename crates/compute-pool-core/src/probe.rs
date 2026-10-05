use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::auth::load_credentials;
use crate::kaggle::KaggleClient;

pub const PROBE_SLUG: &str = "compute-pool-gpu-probe";

pub const PROBE_CODE: &str = r#"
import subprocess
print("=== NVIDIA-SMI OUTPUT ===")
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print("NVIDIA_SMI_ERROR:", e)
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUInfo {
    pub slot: usize,
    pub username: String,
    pub probed_at: String,
    pub gpu_name: String,
    pub vram_mb: u64,
    pub gpu_count: usize,
    pub driver_version: String,
    pub cuda_version: String,
    pub raw_smi: String,
    pub error: Option<String>,
}

impl GPUInfo {
    pub fn vram_gb(&self) -> f64 {
        if self.vram_mb > 0 {
            (self.vram_mb as f64 / 1024.0 * 10.0).round() / 10.0
        } else {
            0.0
        }
    }
}

pub fn load_cached_gpu_info(slot: usize) -> Option<GPUInfo> {
    let cache_file = PathBuf::from("data").join("gpu_info.json");
    if !cache_file.exists() {
        return None;
    }
    if let Ok(content) = std::fs::read_to_string(&cache_file) {
        if let Ok(map) = serde_json::from_str::<std::collections::HashMap<String, GPUInfo>>(&content) {
            return map.get(&slot.to_string()).cloned();
        }
    }
    None
}

pub fn save_gpu_info(info: &GPUInfo) -> Result<()> {
    let dir = PathBuf::from("data");
    std::fs::create_dir_all(&dir)?;
    let cache_file = dir.join("gpu_info.json");

    let mut map: std::collections::HashMap<String, GPUInfo> = if cache_file.exists() {
        std::fs::read_to_string(&cache_file)
            .ok()
            .and_then(|c| serde_json::from_str(&c).ok())
            .unwrap_or_default()
    } else {
        std::collections::HashMap::new()
    };

    map.insert(info.slot.to_string(), info.clone());
    let json = serde_json::to_string_pretty(&map)?;
    std::fs::write(&cache_file, json)?;
    Ok(())
}

pub fn parse_smi_log(log_text: &str, slot: usize, username: &str) -> GPUInfo {
    let now = chrono::Utc::now().to_rfc3339();
    let mut info = GPUInfo {
        slot,
        username: username.to_string(),
        probed_at: now,
        gpu_name: "Unknown".to_string(),
        vram_mb: 0,
        gpu_count: 0,
        driver_version: "Unknown".to_string(),
        cuda_version: "Unknown".to_string(),
        raw_smi: log_text.to_string(),
        error: None,
    };

    let re_drv = Regex::new(r"Driver Version:\s*([0-9\.]+)").unwrap();
    if let Some(caps) = re_drv.captures(log_text) {
        if let Some(m) = caps.get(1) {
            info.driver_version = m.as_str().to_string();
        }
    }

    let re_cuda = Regex::new(r"CUDA Version:\s*([0-9\.]+)").unwrap();
    if let Some(caps) = re_cuda.captures(log_text) {
        if let Some(m) = caps.get(1) {
            info.cuda_version = m.as_str().to_string();
        }
    }

    let re_gpu = Regex::new(r"\|\s+(\d+)\s+([A-Za-z0-9\s\-]+?)\s+(?:Off|On|\d+C|\d+W|P\d+)").unwrap();
    let matches: Vec<_> = re_gpu.captures_iter(log_text).collect();
    if !matches.is_empty() {
        info.gpu_count = matches.len();
        if let Some(m) = matches[0].get(2) {
            info.gpu_name = m.as_str().trim().to_string();
        }
    }

    let re_vram = Regex::new(r"/\s*(\d+)MiB").unwrap();
    if let Some(caps) = re_vram.captures(log_text) {
        if let Some(m) = caps.get(1) {
            if let Ok(vram) = m.as_str().parse::<u64>() {
                info.vram_mb = vram;
            }
        }
    }

    if info.gpu_count == 0 && info.vram_mb == 0 {
        if log_text.contains("NO_GPU") || log_text.contains("NVIDIA_SMI_ERROR") || log_text.contains("No such file") {
            info.error = Some("GPU accelerator not active on account. Phone verification may be required at https://www.kaggle.com/settings".to_string());
        } else if !log_text.contains("NVIDIA-SMI") {
            info.error = Some("No GPU detected on worker".to_string());
        }
    }

    info
}

pub async fn run_probe(slot: usize) -> Result<GPUInfo> {
    let creds = load_credentials(slot)?
        .ok_or_else(|| anyhow::anyhow!("No credentials configured for Slot {}. Run `compute-pool login --slot {}`", slot, slot))?;

    let client = KaggleClient::new(&creds.username, &creds.key);
    client
        .push_kernel(PROBE_SLUG, PROBE_CODE, true)
        .await
        .context("Failed to push probe kernel to Kaggle")?;

    let kernel_ref = format!("{}/{}", creds.username, PROBE_SLUG);
    let start = Instant::now();
    let deadline = Duration::from_secs(300);

    while start.elapsed() < deadline {
        if let Ok(st) = client.get_kernel_status(&kernel_ref).await {
            if st.contains("COMPLETE") || st.contains("ERROR") || st.contains("FAILED") {
                break;
            }
        }
        tokio::time::sleep(Duration::from_secs(6)).await;
    }

    let tmp_dir = tempfile::tempdir()?;
    let _ = client.download_kernel_output(&kernel_ref, tmp_dir.path()).await;

    let mut log_text = String::new();
    if let Ok(entries) = std::fs::read_dir(tmp_dir.path()) {
        for entry in entries.flatten() {
            if entry.path().extension().map(|e| e == "log").unwrap_or(false) {
                if let Ok(c) = std::fs::read_to_string(entry.path()) {
                    log_text = c;
                    break;
                }
            }
        }
    }

    let info = parse_smi_log(&log_text, slot, &creds.username);
    let _ = save_gpu_info(&info);
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_smi_log() {
        let sample_smi = r#"
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2       |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   43C    P8             10W /  70W  |       0MiB /  15360MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla T4                       Off |   00000000:00:05.0 Off |                    0 |
| N/A   45C    P8             11W /  70W  |       0MiB /  15360MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
        "#;

        let info = parse_smi_log(sample_smi, 1, "testuser");
        assert_eq!(info.gpu_name, "Tesla T4");
        assert_eq!(info.gpu_count, 2);
        assert_eq!(info.vram_mb, 15360);
        assert_eq!(info.driver_version, "535.104.05");
        assert_eq!(info.cuda_version, "12.2");
        assert_eq!(info.vram_gb(), 15.0);
    }
}
