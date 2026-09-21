use anyhow::Result;
use serde::{Deserialize, Serialize};

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

pub async fn run_distributed_workload(
    _name: &str,
    script: &str,
    _gpu: bool,
    _max_runtime_hours: f64,
) -> Result<DistributedJobResult> {
    if !script.contains("torch.distributed")
        || !script.contains("init_process_group")
        || !script.contains("DistributedDataParallel")
    {
        anyhow::bail!(
            "{}{}{}",
            "Distributed batch jobs require torch.distributed.init_process_group and ",
            "DistributedDataParallel. Use `compute-pool shell --slot cluster --open` ",
            "and run the script with `crun` so both nodes share the cluster rendezvous."
        );
    }

    anyhow::bail!(
        "{}{}{}",
        "Distributed batch kernels cannot communicate: Kaggle runs each submitted kernel in ",
        "an isolated network namespace. Start the managed cluster with `compute-pool shell ",
        "--slot cluster --open`, then run the DDP script with `crun`."
    );

}
