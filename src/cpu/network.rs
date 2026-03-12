//! CPU: Networking engine — HTTP fetch, TLS, prefetch cache.
//!
//! This is the minimal CPU work that CANNOT be offloaded:
//! - TCP socket I/O
//! - TLS handshake
//! - HTTP protocol handling
//!
//! Everything else (understanding the response) goes to NPU.

use anyhow::{Result, Context};
use log::info;
use parking_lot::Mutex;
use std::collections::HashMap;
use ureq::tls::{TlsConfig, RootCerts};

pub struct NetworkEngine {
    agent: ureq::Agent,
    prefetch_cache: Mutex<HashMap<String, String>>,
}

impl NetworkEngine {
    pub fn new() -> Self {
        // Rustls with bundled WebPki root certificates
        let tls = TlsConfig::builder()
            .root_certs(RootCerts::WebPki)
            .build();

        let agent = ureq::Agent::config_builder()
            .tls_config(tls)
            .build()
            .new_agent();

        Self {
            agent,
            prefetch_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Fetch a URL, returning raw HTML.
    /// Checks prefetch cache first.
    pub fn fetch(&self, url: &str) -> Result<String> {
        // Check prefetch cache
        {
            let mut cache = self.prefetch_cache.lock();
            if let Some(html) = cache.remove(url) {
                info!("[CPU:NET] Cache hit for {url}");
                return Ok(html);
            }
        }

        info!("[CPU:NET] Fetching {url}");
        let resp = match self.agent.get(url)
            .header("User-Agent", "NeuralBrowser/0.1 (CPU+NPU+GPU)")
            .header("Accept", "text/html,application/xhtml+xml,*/*")
            .call()
        {
            Ok(r) => r,
            Err(e) => {
                log::error!("[CPU:NET] Request error: {e:?}");
                return Err(anyhow::anyhow!("HTTP request failed: {e}"));
            }
        };

        let status = resp.status();
        info!("[CPU:NET] {url} → {status}");

        let body = resp.into_body()
            .read_to_string()
            .context("Failed to read response body")?;

        Ok(body)
    }

    /// Prefetch a URL into cache (NPU predicted the user will click it).
    pub fn prefetch(&self, url: &str) -> Result<()> {
        let html = self.fetch(url)?;
        self.prefetch_cache.lock().insert(url.to_string(), html);
        Ok(())
    }
}
