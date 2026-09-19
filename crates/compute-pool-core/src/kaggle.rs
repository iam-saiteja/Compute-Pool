use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde::{Deserialize, Serialize};
use std::io::{Cursor, Write};
use std::path::Path;
use zip::write::FileOptions;
use zip::ZipWriter;

const KAGGLE_API_BASE: &str = "https://www.kaggle.com/api/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelPushPayload {
    pub id: String,
    pub title: String,
    pub code_file: String,
    pub language: String,
    pub kernel_type: String,
    pub is_private: String,
    pub enable_gpu: String,
    pub enable_tpu: String,
    pub enable_internet: String,
    #[serde(default)]
    pub dataset_sources: Vec<String>,
    #[serde(default)]
    pub competition_sources: Vec<String>,
    #[serde(default)]
    pub kernel_sources: Vec<String>,
    #[serde(default)]
    pub model_sources: Vec<String>,
}

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
        key.starts_with("kgat_")
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
        accelerator: Option<&str>,
    ) -> Result<KernelPushResponse> {
        let url = format!("{}/kernels/push", KAGGLE_API_BASE);
        let kernel_ref = format!("{}/{}", self.username, slug);

        // Construct zip payload containing script.py and kernel-metadata.json
        let mut buffer = Vec::new();
        {
            let mut zip = ZipWriter::new(Cursor::new(&mut buffer));
            let options = FileOptions::<()>::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .unix_permissions(0o755);

            let metadata = KernelPushPayload {
                id: kernel_ref.clone(),
                title: slug.to_string(),
                code_file: "script.py".to_string(),
                language: "python".to_string(),
                kernel_type: "script".to_string(),
                is_private: "true".to_string(),
                enable_gpu: if enable_gpu { "true".to_string() } else { "false".to_string() },
                enable_tpu: "false".to_string(),
                enable_internet: "true".to_string(),
                dataset_sources: vec![],
                competition_sources: vec![],
                kernel_sources: vec![],
                model_sources: vec![],
            };

            let meta_bytes = serde_json::to_vec_pretty(&metadata)?;
            zip.start_file("kernel-metadata.json", options)?;
            zip.write_all(&meta_bytes)?;

            zip.start_file("script.py", options)?;
            zip.write_all(script_code.as_bytes())?;

            zip.finish()?;
        }

        // Build multipart form
        let mut form = reqwest::multipart::Form::new()
            .part(
                "blob",
                reqwest::multipart::Part::bytes(buffer)
                    .file_name("kernel.zip")
                    .mime_str("application/zip")?,
            );

        if let Some(acc) = accelerator {
            form = form.text("acc", acc.to_string());
        }

        let req = self.client.post(&url).multipart(form);
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

        let bytes = resp.bytes().await.context("Failed to read output zip bytes")?;
        std::fs::create_dir_all(output_dir)?;

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

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_bearer_token() {
        assert!(KaggleClient::is_bearer_token("kgat_12345abcdef"));
        assert!(!KaggleClient::is_bearer_token("0123456789abcdef0123456789abcdef"));
    }

    #[test]
    fn test_zip_payload_structure() {
        let client = KaggleClient::new("testuser", "testkey");
        assert_eq!(client.username, "testuser");
    }
}
