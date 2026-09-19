use chrono::Utc;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum JobState {
    Queued,
    Scheduling,
    Assigned,
    Running,
    Completed,
    Failed,
    Retrying,
    Cancelled,
}

impl JobState {
    pub fn as_str(&self) -> &'static str {
        match self {
            JobState::Queued => "QUEUED",
            JobState::Scheduling => "SCHEDULING",
            JobState::Assigned => "ASSIGNED",
            JobState::Running => "RUNNING",
            JobState::Completed => "COMPLETED",
            JobState::Failed => "FAILED",
            JobState::Retrying => "RETRYING",
            JobState::Cancelled => "CANCELLED",
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self,
            JobState::Queued | JobState::Scheduling | JobState::Assigned | JobState::Running | JobState::Retrying
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobSpec {
    pub name: String,
    pub script: String,
    #[serde(default = "default_gpu")]
    pub gpu: bool,
    #[serde(default = "default_gpu_memory")]
    pub gpu_memory_gb: f64,
    #[serde(default = "default_max_runtime")]
    pub max_runtime_hours: f64,
    #[serde(default = "default_true")]
    pub checkpointable: bool,
    #[serde(default = "default_retries")]
    pub max_retries: u32,
}

fn default_gpu() -> bool { true }
fn default_gpu_memory() -> f64 { 8.0 }
fn default_max_runtime() -> f64 { 4.0 }
fn default_true() -> bool { true }
fn default_retries() -> u32 { 3 }

impl Default for JobSpec {
    fn default() -> Self {
        Self {
            name: "unnamed".to_string(),
            script: "".to_string(),
            gpu: true,
            gpu_memory_gb: 8.0,
            max_runtime_hours: 4.0,
            checkpointable: true,
            max_retries: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Job {
    pub id: String,
    #[serde(flatten)]
    pub spec: JobSpec,
    pub state: JobState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assigned_slot: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assigned_username: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kaggle_kernel_slug: Option<String>,
    #[serde(default)]
    pub retry_count: u32,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Job {
    pub fn new(id: String, spec: JobSpec) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            id,
            spec,
            state: JobState::Queued,
            assigned_slot: None,
            assigned_username: None,
            kaggle_kernel_slug: None,
            retry_count: 0,
            created_at: now.clone(),
            updated_at: now,
            error: None,
        }
    }

    pub fn transition(&mut self, new_state: JobState, error: Option<String>) {
        self.state = new_state;
        self.updated_at = Utc::now().to_rfc3339();
        if let Some(err) = error {
            self.error = Some(err);
        }
    }

    pub fn get_slot_number(&self) -> Option<usize> {
        match &self.assigned_slot {
            Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| v as usize),
            Some(serde_json::Value::String(s)) => s.parse::<usize>().ok(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_lifecycle() {
        let spec = JobSpec {
            name: "test-run".to_string(),
            script: "print('hello')".to_string(),
            ..Default::default()
        };
        let mut job = Job::new("0".to_string(), spec);
        assert_eq!(job.state, JobState::Queued);
        assert!(job.state.is_active());

        job.transition(JobState::Running, None);
        assert_eq!(job.state, JobState::Running);

        job.transition(JobState::Completed, None);
        assert_eq!(job.state, JobState::Completed);
        assert!(!job.state.is_active());
    }

    #[test]
    fn test_job_serialization_roundtrip() {
        let spec = JobSpec {
            name: "mlp-training".to_string(),
            script: "import torch".to_string(),
            gpu: true,
            gpu_memory_gb: 15.0,
            max_runtime_hours: 2.0,
            checkpointable: true,
            max_retries: 3,
        };
        let mut job = Job::new("0".to_string(), spec);
        job.assigned_slot = Some(serde_json::json!(1));
        job.assigned_username = Some("alice".to_string());

        let json = serde_json::to_string(&job).unwrap();
        let parsed: Job = serde_json::from_str(&json).unwrap();
        assert_eq!(job.id, parsed.id);
        assert_eq!(job.spec.name, parsed.spec.name);
        assert_eq!(parsed.assigned_slot, Some(serde_json::json!(1)));
        assert_eq!(parsed.assigned_username.as_deref(), Some("alice"));
    }
}
