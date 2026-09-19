use std::fs;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use regex::Regex;

use crate::job::Job;

pub fn get_data_dir() -> PathBuf {
    PathBuf::from("data")
}

pub fn get_jobs_file() -> PathBuf {
    get_data_dir().join("jobs.json")
}

pub fn load_all_jobs() -> Result<Vec<Job>> {
    load_jobs_from_file(&get_jobs_file())
}

pub fn load_jobs_from_file(path: &Path) -> Result<Vec<Job>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read jobs file: {:?}", path))?;
    if content.trim().is_empty() {
        return Ok(Vec::new());
    }
    let jobs: Vec<Job> = serde_json::from_str(&content)
        .with_context(|| "Failed to parse jobs JSON")?;
    Ok(jobs)
}

pub fn save_all_jobs(jobs: &[Job]) -> Result<()> {
    save_jobs_to_file(&get_jobs_file(), jobs)
}

pub fn save_jobs_to_file(path: &Path, jobs: &[Job]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(jobs)?;
    fs::write(path, json)
        .with_context(|| format!("Failed to save jobs to {:?}", path))?;
    Ok(())
}

pub fn next_job_id() -> Result<String> {
    let jobs = load_all_jobs()?;
    if jobs.is_empty() {
        return Ok("0".to_string());
    }

    let re = Regex::new(r"^(?:job-)?(\d+)$")?;
    let mut indices = Vec::new();

    for j in &jobs {
        if let Some(caps) = re.captures(&j.id) {
            if let Some(m) = caps.get(1) {
                if let Ok(num) = m.as_str().parse::<usize>() {
                    indices.push(num);
                }
            }
        }
    }

    if let Some(max_idx) = indices.into_iter().max() {
        Ok((max_idx + 1).to_string())
    } else {
        Ok(jobs.len().to_string())
    }
}

pub fn upsert_job(job: &Job) -> Result<()> {
    let mut jobs = load_all_jobs()?;
    let mut found = false;
    for j in jobs.iter_mut() {
        if j.id == job.id {
            *j = job.clone();
            found = true;
            break;
        }
    }
    if !found {
        jobs.push(job.clone());
    }
    save_all_jobs(&jobs)?;
    Ok(())
}

pub fn normalize_id_match(target: &str, candidate_id: &str) -> bool {
    let target = target.trim();
    let cand = candidate_id.trim();
    if cand == target {
        return true;
    }
    if cand == format!("job-{}", target) {
        return true;
    }
    if target == format!("job-{}", cand) {
        return true;
    }
    false
}

pub fn get_job(target: &str) -> Result<Option<Job>> {
    let jobs = load_all_jobs()?;
    for j in jobs {
        if normalize_id_match(target, &j.id) {
            return Ok(Some(j));
        }
    }
    Ok(None)
}

pub fn delete_job(target: &str) -> Result<Option<Job>> {
    let jobs = load_all_jobs()?;
    let mut target_job = None;
    let mut remaining = Vec::new();

    for j in jobs {
        if target_job.is_none() && normalize_id_match(target, &j.id) {
            target_job = Some(j);
        } else {
            remaining.push(j);
        }
    }

    if let Some(ref deleted) = target_job {
        save_all_jobs(&remaining)?;
        let job_dir = get_data_dir().join("jobs").join(&deleted.id);
        if job_dir.exists() && job_dir.is_dir() {
            let _ = fs::remove_dir_all(&job_dir);
        }
    }

    Ok(target_job)
}

pub fn clear_jobs(all_jobs: bool) -> Result<Vec<Job>> {
    let jobs = load_all_jobs()?;
    let mut deleted = Vec::new();
    let mut remaining = Vec::new();

    for j in jobs {
        if all_jobs || !j.state.is_active() {
            let job_dir = get_data_dir().join("jobs").join(&j.id);
            if job_dir.exists() && job_dir.is_dir() {
                let _ = fs::remove_dir_all(&job_dir);
            }
            deleted.push(j);
        } else {
            remaining.push(j);
        }
    }

    save_all_jobs(&remaining)?;
    Ok(deleted)
}

pub fn reindex_jobs() -> Result<Vec<Job>> {
    let mut jobs = load_all_jobs()?;
    jobs.sort_by(|a, b| a.created_at.cmp(&b.created_at));

    for (idx, j) in jobs.iter_mut().enumerate() {
        let old_id = j.id.clone();
        let new_id = idx.to_string();
        if old_id != new_id {
            let old_dir = get_data_dir().join("jobs").join(&old_id);
            let new_dir = get_data_dir().join("jobs").join(&new_id);
            if old_dir.exists() && !new_dir.exists() {
                let _ = fs::rename(&old_dir, &new_dir);
            }
            j.id = new_id;
        }
    }

    save_all_jobs(&jobs)?;
    Ok(jobs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::JobSpec;
    use tempfile::NamedTempFile;

    #[test]
    fn test_storage_crud() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        let mut jobs = Vec::new();
        let j0 = Job::new("0".to_string(), JobSpec { name: "j0".to_string(), ..Default::default() });
        let j1 = Job::new("1".to_string(), JobSpec { name: "j1".to_string(), ..Default::default() });
        jobs.push(j0);
        jobs.push(j1);

        save_jobs_to_file(path, &jobs).unwrap();
        let loaded = load_jobs_from_file(path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "0");
        assert_eq!(loaded[1].id, "1");
    }

    #[test]
    fn test_normalize_id() {
        assert!(normalize_id_match("0", "0"));
        assert!(normalize_id_match("0", "job-0"));
        assert!(normalize_id_match("job-0", "0"));
        assert!(normalize_id_match("job-12", "job-12"));
        assert!(!normalize_id_match("1", "0"));
    }
}
