use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::auth::load_credentials;
use crate::job::{Job, JobState};
use crate::kaggle::KaggleClient;
use crate::storage::upsert_job;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountStatus {
    pub slot: usize,
    pub username: String,
    pub connected: bool,
    pub gpu_seconds_used: i64,
    pub gpu_seconds_total: i64,
    pub tpu_seconds_used: i64,
    pub tpu_seconds_total: i64,
    pub quota_refresh_time: String,
    pub error: Option<String>,
}

impl AccountStatus {
    pub fn gpu_hours_used(&self) -> f64 {
        (self.gpu_seconds_used as f64 / 3600.0 * 100.0).round() / 100.0
    }

    pub fn gpu_hours_total(&self) -> f64 {
        (self.gpu_seconds_total as f64 / 3600.0 * 10.0).round() / 10.0
    }

    pub fn gpu_hours_remaining(&self) -> f64 {
        let rem = (self.gpu_seconds_total - self.gpu_seconds_used) as f64 / 3600.0;
        if rem < 0.0 {
            0.0
        } else {
            (rem * 100.0).round() / 100.0
        }
    }

    pub fn has_capacity(&self) -> bool {
        self.connected && self.gpu_hours_remaining() > 0.05
    }
}

pub async fn get_account_status(slot: usize) -> AccountStatus {
    let creds = match load_credentials(slot) {
        Ok(Some(c)) => c,
        Ok(None) => {
            return AccountStatus {
                slot,
                username: "<not configured>".to_string(),
                connected: false,
                gpu_seconds_used: 0,
                gpu_seconds_total: 108000,
                tpu_seconds_used: 0,
                tpu_seconds_total: 72000,
                quota_refresh_time: "".to_string(),
                error: Some(format!("No credentials found. Run: compute-pool login --slot {}", slot)),
            };
        }
        Err(e) => {
            return AccountStatus {
                slot,
                username: "<error>".to_string(),
                connected: false,
                gpu_seconds_used: 0,
                gpu_seconds_total: 108000,
                tpu_seconds_used: 0,
                tpu_seconds_total: 72000,
                quota_refresh_time: "".to_string(),
                error: Some(e.to_string()),
            };
        }
    };

    let client = KaggleClient::new(&creds.username, &creds.key);
    match client.get_quota().await {
        Ok(quota) => {
            let gpu_used = quota.gpu_quota.as_ref().and_then(|q| q.time_used.as_ref()).map(|u| u.seconds).unwrap_or(0);
            let gpu_total = quota.gpu_quota.as_ref().and_then(|q| q.total_time_allowed.as_ref()).map(|u| u.seconds).unwrap_or(108000);
            let tpu_used = quota.tpu_quota.as_ref().and_then(|q| q.time_used.as_ref()).map(|u| u.seconds).unwrap_or(0);
            let tpu_total = quota.tpu_quota.as_ref().and_then(|q| q.total_time_allowed.as_ref()).map(|u| u.seconds).unwrap_or(72000);
            let refresh = quota.quota_refresh_time.unwrap_or_default();

            AccountStatus {
                slot,
                username: creds.username,
                connected: true,
                gpu_seconds_used: gpu_used,
                gpu_seconds_total: if gpu_total > 0 { gpu_total } else { 108000 },
                tpu_seconds_used: tpu_used,
                tpu_seconds_total: if tpu_total > 0 { tpu_total } else { 72000 },
                quota_refresh_time: refresh,
                error: None,
            }
        }
        Err(e) => AccountStatus {
            slot,
            username: creds.username,
            connected: false,
            gpu_seconds_used: 0,
            gpu_seconds_total: 108000,
            tpu_seconds_used: 0,
            tpu_seconds_total: 72000,
            quota_refresh_time: "".to_string(),
            error: Some(e.to_string()),
        },
    }
}

pub async fn get_all_statuses() -> Vec<AccountStatus> {
    let mut results = Vec::new();
    for slot in 1..=2 {
        results.push(get_account_status(slot).await);
    }
    results
}

pub fn best_available_slot(statuses: &[AccountStatus]) -> Option<AccountStatus> {
    statuses
        .iter()
        .filter(|s| s.has_capacity())
        .max_by(|a, b| {
            a.gpu_hours_remaining()
                .partial_cmp(&b.gpu_hours_remaining())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
}

pub async fn schedule_job(job: &mut Job) -> Result<()> {
    job.transition(JobState::Scheduling, None);
    upsert_job(job)?;

    let statuses = get_all_statuses().await;
    match best_available_slot(&statuses) {
        Some(winner) => {
            job.assigned_slot = Some(serde_json::json!(winner.slot));
            job.assigned_username = Some(winner.username.clone());
            job.transition(JobState::Assigned, None);
            upsert_job(job)?;
            Ok(())
        }
        None => {
            job.transition(
                JobState::Failed,
                Some("No account has remaining GPU quota. Check `compute-pool accounts status`.".to_string()),
            );
            upsert_job(job)?;
            anyhow::bail!("No account has remaining GPU quota");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_account_capacity_calculation() {
        let acc = AccountStatus {
            slot: 1,
            username: "testuser".to_string(),
            connected: true,
            gpu_seconds_used: 36000,
            gpu_seconds_total: 108000,
            tpu_seconds_used: 0,
            tpu_seconds_total: 72000,
            quota_refresh_time: "".to_string(),
            error: None,
        };

        assert_eq!(acc.gpu_hours_used(), 10.0);
        assert_eq!(acc.gpu_hours_total(), 30.0);
        assert_eq!(acc.gpu_hours_remaining(), 20.0);
        assert!(acc.has_capacity());
    }

    #[test]
    fn test_best_available_slot() {
        let s1 = AccountStatus {
            slot: 1,
            username: "user1".to_string(),
            connected: true,
            gpu_seconds_used: 72000, // 10h left
            gpu_seconds_total: 108000,
            tpu_seconds_used: 0,
            tpu_seconds_total: 72000,
            quota_refresh_time: "".to_string(),
            error: None,
        };
        let s2 = AccountStatus {
            slot: 2,
            username: "user2".to_string(),
            connected: true,
            gpu_seconds_used: 36000, // 20h left
            gpu_seconds_total: 108000,
            tpu_seconds_used: 0,
            tpu_seconds_total: 72000,
            quota_refresh_time: "".to_string(),
            error: None,
        };

        let best = best_available_slot(&[s1, s2]).unwrap();
        assert_eq!(best.slot, 2);
    }
}
