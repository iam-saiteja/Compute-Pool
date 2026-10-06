use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const MAX_ACCOUNT_SLOTS: usize = 8;

/// Key under which settings are stored in the same credentials file as the
/// account slots. Slot keys are numeric, so this can never collide with one.
const SETTINGS_KEY: &str = "_settings";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Credentials {
    pub username: String,
    pub key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Settings {
    /// Set once the user has acknowledged the Kaggle multi-account policy risk.
    #[serde(default)]
    pub risk_acknowledged: bool,
    /// Password for the web terminal and file manager on every node. Set by the user.
    #[serde(default)]
    pub web_password: Option<String>,
    /// Accounts the cluster uses. None means every configured account.
    #[serde(default)]
    pub cluster_slots: Option<Vec<usize>>,
}

/// The slots a cluster uses: the chosen ones if set, otherwise every configured one.
/// A chosen slot that is not configured is an error, not silently dropped.
pub fn resolve_cluster_slots(configured: &[usize], selected: Option<&[usize]>) -> Result<Vec<usize>> {
    let mut configured = configured.to_vec();
    configured.sort_unstable();
    let Some(selected) = selected else {
        return Ok(configured);
    };
    let mut chosen = selected.to_vec();
    chosen.sort_unstable();
    chosen.dedup();
    if chosen.is_empty() {
        anyhow::bail!("The cluster needs at least one account. Run `compute-pool cluster reset`.");
    }
    for slot in &chosen {
        if !configured.contains(slot) {
            anyhow::bail!("Slot {} is not configured. Run `compute-pool login --slot {}`.", slot, slot);
        }
    }
    Ok(chosen)
}

/// The slots the next cluster will use, from the stored selection.
pub fn cluster_slots() -> Result<Vec<usize>> {
    let configured: Vec<usize> = load_all_credentials()?.keys().copied().collect();
    resolve_cluster_slots(&configured, load_settings()?.cluster_slots.as_deref())
}

pub fn get_credentials_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Failed to resolve user home directory")?;
    Ok(home.join(".compute_pool"))
}

pub fn get_credentials_path() -> Result<PathBuf> {
    Ok(get_credentials_dir()?.join("credentials.json"))
}

/// The whole file as raw JSON, so slot entries and the settings entry can be
/// read and rewritten without either one clobbering the other.
fn read_raw() -> Result<HashMap<String, serde_json::Value>> {
    let path = get_credentials_path()?;
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let data = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read credentials file: {:?}", path))?;
    serde_json::from_str(&data).with_context(|| "Failed to parse credentials JSON")
}

fn write_raw(raw: &HashMap<String, serde_json::Value>) -> Result<()> {
    let dir = get_credentials_dir()?;
    fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create credentials directory: {:?}", dir))?;
    let path = get_credentials_path()?;
    let json = serde_json::to_string_pretty(raw)?;
    fs::write(&path, json).with_context(|| format!("Failed to write credentials to {:?}", path))?;
    Ok(())
}

pub fn load_all_credentials() -> Result<HashMap<usize, Credentials>> {
    let mut parsed = HashMap::new();
    for (k, v) in read_raw()? {
        if let Ok(slot) = k.parse::<usize>() {
            parsed.insert(slot, serde_json::from_value(v).context("Malformed credentials entry")?);
        }
    }
    Ok(parsed)
}

pub fn load_credentials(slot: usize) -> Result<Option<Credentials>> {
    let all = load_all_credentials()?;
    Ok(all.get(&slot).cloned())
}

pub fn save_credentials(slot: usize, creds: &Credentials) -> Result<()> {
    let mut raw = read_raw()?;
    raw.insert(slot.to_string(), serde_json::to_value(creds)?);
    write_raw(&raw)
}

/// The web password is written into the bootstrap script as a Python string
/// literal, so only characters that are safe inside one are accepted. A quote
/// or backslash here would break the script or inject code into it.
pub fn validate_web_password(pw: &str) -> Result<()> {
    if !(4..=64).contains(&pw.chars().count()) {
        anyhow::bail!("Password must be 4 to 64 characters.");
    }
    if !pw.chars().all(|c| c.is_ascii_alphanumeric() || "-_.!@#%+=".contains(c)) {
        anyhow::bail!("Password may only use letters, digits and - _ . ! @ # % + =");
    }
    Ok(())
}

pub fn load_settings() -> Result<Settings> {
    match read_raw()?.remove(SETTINGS_KEY) {
        Some(v) => serde_json::from_value(v).context("Malformed settings entry"),
        None => Ok(Settings::default()),
    }
}

pub fn save_settings(settings: &Settings) -> Result<()> {
    let mut raw = read_raw()?;
    raw.insert(SETTINGS_KEY.to_string(), serde_json::to_value(settings)?);
    write_raw(&raw)
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

    #[test]
    fn test_web_password_rules_block_script_breaking_characters() {
        assert!(validate_web_password("1234").is_ok());
        assert!(validate_web_password("good-Pass_1.2!").is_ok());
        assert!(validate_web_password("123").is_err(), "too short");
        assert!(validate_web_password("bad\"quote").is_err());
        assert!(validate_web_password("back\\slash").is_err());
        assert!(validate_web_password("new\nline").is_err());
        assert!(validate_web_password("a b c d").is_err(), "spaces are not allowed");
    }

    #[test]
    fn test_cluster_slots_default_to_every_configured_account() {
        assert_eq!(resolve_cluster_slots(&[3, 1, 2], None).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_cluster_slots_use_only_the_chosen_accounts() {
        // The example from the user: three accounts, cluster uses 1 and 3.
        assert_eq!(resolve_cluster_slots(&[1, 2, 3], Some(&[3, 1])).unwrap(), vec![1, 3]);
        assert_eq!(resolve_cluster_slots(&[1, 2, 3], Some(&[1, 1, 3])).unwrap(), vec![1, 3]);
    }

    #[test]
    fn test_cluster_slots_reject_unconfigured_and_empty_selections() {
        assert!(resolve_cluster_slots(&[1, 2], Some(&[1, 5])).is_err(), "slot 5 is not configured");
        assert!(resolve_cluster_slots(&[1, 2], Some(&[])).is_err(), "empty selection");
    }

    #[test]
    fn test_settings_default_when_absent() {
        // Files written before settings existed must still load with safe defaults.
        let s: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(s, Settings::default());
        assert!(!s.risk_acknowledged);
        assert!(s.web_password.is_none());
    }
}
