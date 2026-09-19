use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::auth::load_credentials;
use crate::job::{Job, JobSpec, JobState};
use crate::kaggle::KaggleClient;
use crate::storage::{next_job_id, upsert_job};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResult {
    pub rank: usize,
    pub slot: usize,
    pub username: String,
    pub kernel_ref: String,
    pub status: String,
    pub log: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedJobResult {
    pub job_id: String,
    pub elapsed_seconds: f64,
    pub node_results: Vec<NodeResult>,
    pub success: bool,
}

fn sanitize_slug(text: &str) -> String {
    let re = regex::Regex::new(r"[^a-zA-Z0-9\-]").unwrap();
    let s = re.replace_all(&text.to_lowercase(), "-").into_owned();
    let re_dashes = regex::Regex::new(r"-+").unwrap();
    let clean = re_dashes.replace_all(&s, "-").trim_matches('-').to_string();
    if clean.len() > 40 {
        clean[..40].to_string()
    } else {
        clean
    }
}

async fn run_single_node(
    slot: usize,
    rank: usize,
    world_size: usize,
    name: &str,
    script: &str,
    gpu: bool,
    max_runtime_hours: f64,
    job_id: &str,
) -> NodeResult {
    let creds = match load_credentials(slot) {
        Ok(Some(c)) => c,
        _ => {
            return NodeResult {
                rank,
                slot,
                username: "unknown".to_string(),
                kernel_ref: "".to_string(),
                status: "FAILED".to_string(),
                log: "".to_string(),
                error: Some(format!("No credentials for slot {}", slot)),
            };
        }
    };

    let slug = sanitize_slug(&format!("{}-node{}-{}", name, rank, job_id));
    let kernel_ref = format!("{}/{}", creds.username, slug);

    let preamble = format!(
        "import os\nos.environ[\"CP_NODE_RANK\"] = \"{}\"\nos.environ[\"CP_WORLD_SIZE\"] = \"{}\"\nos.environ[\"KAGGLE_USERNAME\"] = \"{}\"\n",
        rank, world_size, creds.username
    );
    let full_script = format!("{}\n{}", preamble, script);

    let client = KaggleClient::new(&creds.username, &creds.key);
    if let Err(e) = client.push_kernel(&slug, &full_script, gpu, Some("nvidia-tesla-t4")).await {
        return NodeResult {
            rank,
            slot,
            username: creds.username,
            kernel_ref,
            status: "FAILED".to_string(),
            log: "".to_string(),
            error: Some(format!("Push failed: {}", e)),
        };
    }

    let start = Instant::now();
    let deadline = Duration::from_secs((max_runtime_hours * 3600.0) as u64 + 300);
    let mut final_status = "UNKNOWN".to_string();

    while start.elapsed() < deadline {
        if let Ok(st) = client.get_kernel_status(&kernel_ref).await {
            if st.contains("COMPLETE") {
                final_status = "COMPLETE".to_string();
                break;
            } else if st.contains("ERROR") || st.contains("FAILED") {
                final_status = "ERROR".to_string();
                break;
            }
        }
        tokio::time::sleep(Duration::from_secs(8)).await;
    }

    let node_dir = PathBuf::from("data").join("jobs").join(job_id).join(format!("node_{}", rank));
    let _ = client.download_kernel_output(&kernel_ref, &node_dir).await;

    let mut log_text = String::new();
    if let Ok(entries) = std::fs::read_dir(&node_dir) {
        for entry in entries.flatten() {
            if entry.path().extension().map(|e| e == "log").unwrap_or(false) {
                if let Ok(c) = std::fs::read_to_string(entry.path()) {
                    log_text = c;
                    break;
                }
            }
        }
    }

    let err = if final_status == "COMPLETE" {
        None
    } else {
        Some(format!("Exited with status {}", final_status))
    };

    NodeResult {
        rank,
        slot,
        username: creds.username,
        kernel_ref,
        status: final_status,
        log: log_text,
        error: err,
    }
}

pub async fn run_distributed_workload(
    name: &str,
    script: &str,
    gpu: bool,
    max_runtime_hours: f64,
) -> Result<DistributedJobResult> {
    let next_id = next_job_id()?;
    let mut job = Job::new(
        next_id.clone(),
        JobSpec {
            name: name.to_string(),
            script: script.to_string(),
            gpu,
            gpu_memory_gb: 30.0,
            max_runtime_hours,
            checkpointable: true,
            max_retries: 3,
        },
    );

    job.state = JobState::Running;
    job.assigned_slot = Some(serde_json::json!([1, 2]));
    job.assigned_username = Some("Multi-Account (Slot 1 + Slot 2)".to_string());
    upsert_job(&job)?;

    let t0 = Instant::now();

    let (r0, r1) = tokio::join!(
        run_single_node(1, 0, 2, name, script, gpu, max_runtime_hours, &next_id),
        run_single_node(2, 1, 2, name, script, gpu, max_runtime_hours, &next_id)
    );

    let elapsed = t0.elapsed().as_secs_f64();
    let all_success = r0.status == "COMPLETE" && r1.status == "COMPLETE";

    if all_success {
        job.transition(JobState::Completed, None);
    } else {
        job.transition(
            JobState::Failed,
            Some(format!("Node 0: {}, Node 1: {}", r0.status, r1.status)),
        );
    }
    upsert_job(&job)?;

    Ok(DistributedJobResult {
        job_id: next_id,
        elapsed_seconds: elapsed,
        node_results: vec![r0, r1],
        success: all_success,
    })
}
