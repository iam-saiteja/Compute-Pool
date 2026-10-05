use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::path::Path;

const KAGGLE_API_BASE: &str = "https://www.kaggle.com/api/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelPushResponse {
    pub ref_url: Option<String>,
    pub url: Option<String>,
    pub error: Option<String>,
    pub version_number: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuotaTimeItem {
    #[serde(default)]
    pub seconds: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuotaItem {
    #[serde(default)]
    pub time_used: Option<QuotaTimeItem>,
    #[serde(default)]
    pub total_time_allowed: Option<QuotaTimeItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KaggleQuotaResponse {
    #[serde(default)]
    pub gpu_quota: Option<QuotaItem>,
    #[serde(default)]
    pub tpu_quota: Option<QuotaItem>,
    #[serde(default)]
    pub quota_refresh_time: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelStatusResponse {
    pub status: Option<String>,
    pub failure_message: Option<String>,
}

pub struct KaggleClient {
    pub username: String,
    pub key: String,
    client: reqwest::Client,
}

impl KaggleClient {
    pub fn new(username: &str, key: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            username: username.to_string(),
            key: key.to_string(),
            client,
        }
    }

    pub fn is_bearer_token(key: &str) -> bool {
        key.to_ascii_lowercase().starts_with("kgat_")
    }

    fn auth_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if Self::is_bearer_token(&self.key) {
            if let Ok(val) = HeaderValue::from_str(&format!("Bearer {}", self.key)) {
                headers.insert(AUTHORIZATION, val);
            }
        }
        headers
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if Self::is_bearer_token(&self.key) {
            req.headers(self.auth_headers())
        } else {
            req.basic_auth(&self.username, Some(&self.key))
        }
    }

    /// Fetch real-time quota from Kaggle
    pub async fn get_quota(&self) -> Result<KaggleQuotaResponse> {
        let url = format!("{}/kernels/quota", KAGGLE_API_BASE);
        let req = self.client.get(&url);
        let resp = self.apply_auth(req).send().await
            .context("Failed to connect to Kaggle quota endpoint")?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            anyhow::bail!("Authentication failed (401). Check username and API key.");
        }

        let body = resp.text().await.context("Failed to read quota body")?;
        let quota: KaggleQuotaResponse = serde_json::from_str(&body)
            .with_context(|| format!("Failed to parse Kaggle quota response: {}", body))?;
        Ok(quota)
    }

    /// Push a kernel (with script and metadata) to Kaggle
    pub async fn push_kernel(
        &self,
        slug: &str,
        script_code: &str,
        enable_gpu: bool,
    ) -> Result<KernelPushResponse> {
        let url = format!("{}/kernels/push", KAGGLE_API_BASE);
        let kernel_ref = format!("{}/{}", self.username, slug);

        #[derive(Debug, Serialize)]
        #[serde(rename_all = "camelCase")]
        struct PushRequest {
            slug: String,
            new_title: String,
            text: String,
            language: String,
            kernel_type: String,
            is_private: bool,
            enable_gpu: bool,
            enable_internet: bool,
            dataset_data_sources: Vec<String>,
            competition_data_sources: Vec<String>,
            kernel_data_sources: Vec<String>,
            category_ids: Vec<String>,
        }

        let payload = PushRequest {
            slug: kernel_ref.clone(),
            new_title: slug.to_string(),
            text: script_code.to_string(),
            language: "python".to_string(),
            kernel_type: "script".to_string(),
            is_private: true,
            enable_gpu,
            enable_internet: true,
            dataset_data_sources: vec![],
            competition_data_sources: vec![],
            kernel_data_sources: vec![],
            category_ids: vec![],
        };

        let req = self.client.post(&url).json(&payload);
        let resp = self.apply_auth(req).send().await
            .context("Failed to send push request to Kaggle")?;

        let status = resp.status();
        let body = resp.text().await.context("Failed to read push response")?;

        if !status.is_success() {
            anyhow::bail!("Kaggle push failed [{}]: {}", status, body);
        }

        let push_res: KernelPushResponse = serde_json::from_str(&body)
            .unwrap_or(KernelPushResponse {
                ref_url: Some(kernel_ref),
                url: None,
                error: None,
                version_number: None,
            });

        Ok(push_res)
    }

    /// Query the status of a kernel (e.g. COMPLETE, RUNNING, ERROR, etc.)
    pub async fn get_kernel_status(&self, kernel_ref: &str) -> Result<String> {
        let url = format!("{}/kernels/status", KAGGLE_API_BASE);
        let req = if kernel_ref.contains('/') {
            let parts: Vec<&str> = kernel_ref.splitn(2, '/').collect();
            self.client.get(&url).query(&[("userName", parts[0]), ("kernelSlug", parts[1])])
        } else {
            self.client.get(&url).query(&[("userName", self.username.as_str()), ("kernelSlug", kernel_ref)])
        };

        let resp = self.apply_auth(req).send().await
            .context("Failed to query kernel status from Kaggle")?;

        let body = resp.text().await.context("Failed to read kernel status response")?;
        
        // Try parsing JSON or fallback to text status
        if let Ok(parsed) = serde_json::from_str::<KernelStatusResponse>(&body) {
            if let Some(st) = parsed.status {
                return Ok(st.to_uppercase());
            }
        }

        Ok(body.trim().trim_matches('"').to_uppercase())
    }

    /// Download output files of a kernel to a local folder
    pub async fn download_kernel_output(&self, kernel_ref: &str, output_dir: &Path) -> Result<()> {
        let parts: Vec<&str> = kernel_ref.splitn(2, '/').collect();
        let (user, slug) = if parts.len() == 2 {
            (parts[0], parts[1])
        } else {
            (self.username.as_str(), kernel_ref)
        };

        let url = format!("{}/kernels/output", KAGGLE_API_BASE);
        let req = self.client.get(&url).query(&[("userName", user), ("kernelSlug", slug)]);
        let resp = self.apply_auth(req).send().await
            .context("Failed to download kernel output from Kaggle")?;

        if !resp.status().is_success() {
            anyhow::bail!("Output download failed: HTTP {}", resp.status());
        }

        let bytes = resp.bytes().await.context("Failed to read output response bytes")?;
        std::fs::create_dir_all(output_dir)?;

        // Case 1: Check if response is a ZIP file (starts with PK\x03\x04)
        if bytes.starts_with(b"PK\x03\x04") {
            let mut archive = zip::ZipArchive::new(Cursor::new(bytes))?;
            for i in 0..archive.len() {
                let mut file = archive.by_index(i)?;
                let outpath = match file.enclosed_name() {
                    Some(path) => output_dir.join(path),
                    None => continue,
                };

                if file.name().ends_with('/') {
                    std::fs::create_dir_all(&outpath)?;
                } else {
                    if let Some(p) = outpath.parent() {
                        std::fs::create_dir_all(p)?;
                    }
                    let mut outfile = std::fs::File::create(&outpath)?;
                    std::io::copy(&mut file, &mut outfile)?;
                }
            }
            return Ok(());
        }

        // Case 2: Response is Kaggle JSON output format
        if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            let mut full_log = String::new();

            // Extract log string from "log" or "logNullable"
            let log_str = val.get("log")
                .and_then(|v| v.as_str())
                .or_else(|| val.get("logNullable").and_then(|v| v.as_str()));

            if let Some(raw_log) = log_str {
                if let Ok(entries) = serde_json::from_str::<Vec<serde_json::Value>>(raw_log) {
                    for entry in entries {
                        if let Some(data) = entry.get("data").and_then(|v| v.as_str()) {
                            full_log.push_str(data);
                        }
                    }
                } else {
                    full_log.push_str(raw_log);
                }
            }

            if !full_log.is_empty() {
                let log_file = output_dir.join("stdout.log");
                std::fs::write(&log_file, full_log)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_bearer_token() {
        assert!(KaggleClient::is_bearer_token("kgat_12345abcdef"));
        assert!(KaggleClient::is_bearer_token("KGAT_12345ABCDEF"));
        assert!(!KaggleClient::is_bearer_token("0123456789abcdef0123456789abcdef"));
    }

    #[test]
    fn test_client_init() {
        let client = KaggleClient::new("testuser", "testkey");
        assert_eq!(client.username, "testuser");
    }
}
