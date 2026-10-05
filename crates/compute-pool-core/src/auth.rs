use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const MAX_ACCOUNT_SLOTS: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Credentials {
    pub username: String,
    pub key: String,
}

pub fn get_credentials_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Failed to resolve user home directory")?;
    Ok(home.join(".compute_pool"))
}

pub fn get_credentials_path() -> Result<PathBuf> {
    Ok(get_credentials_dir()?.join("credentials.json"))
}

pub fn load_all_credentials() -> Result<HashMap<usize, Credentials>> {
    let path = get_credentials_path()?;
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let data = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read credentials file: {:?}", path))?;
    let raw: HashMap<String, Credentials> = serde_json::from_str(&data)
        .with_context(|| "Failed to parse credentials JSON")?;
    
    let mut parsed = HashMap::new();
    for (k, v) in raw {
        if let Ok(slot) = k.parse::<usize>() {
            parsed.insert(slot, v);
        }
    }
    Ok(parsed)
}

pub fn load_credentials(slot: usize) -> Result<Option<Credentials>> {
    let all = load_all_credentials()?;
    Ok(all.get(&slot).cloned())
}

pub fn save_credentials(slot: usize, creds: &Credentials) -> Result<()> {
    let dir = get_credentials_dir()?;
    fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create credentials directory: {:?}", dir))?;
    
    let mut all = load_all_credentials().unwrap_or_default();
    all.insert(slot, creds.clone());

    let raw: HashMap<String, Credentials> = all
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

    let path = get_credentials_path()?;
    let json = serde_json::to_string_pretty(&raw)?;
    fs::write(&path, json)
        .with_context(|| format!("Failed to write credentials to {:?}", path))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credentials_serialization() {
        let creds = Credentials {
            username: "testuser".to_string(),
            key: "testkey123".to_string(),
        };
        let json = serde_json::to_string(&creds).unwrap();
        let deserialized: Credentials = serde_json::from_str(&json).unwrap();
        assert_eq!(creds, deserialized);
    }
}
