//! CPU: Networking engine — HTTP fetch, TLS, prefetch cache.
//!
//! This is the minimal CPU work that CANNOT be offloaded:
//! - TCP socket I/O
//! - TLS handshake
//! - HTTP protocol handling
//! - Redirect following (up to 10 hops)
//! - Timeout management
//! - Error page generation for failed requests
//!
//! Everything else (understanding the response) goes to NPU.

use anyhow::{Result, Context};
use log::{info, warn};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::time::Duration;
use ureq::tls::{TlsConfig, RootCerts};

const USER_AGENT: &str = "NeuralBrowser/0.1 (CPU+NPU+GPU)";
const MAX_REDIRECTS: u32 = 10;
const TIMEOUT_CONNECT: Duration = Duration::from_secs(10);
const TIMEOUT_RECV_BODY: Duration = Duration::from_secs(30);
const TIMEOUT_GLOBAL: Duration = Duration::from_secs(45);

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
            .max_redirects(MAX_REDIRECTS)
            .http_status_as_error(false)
            .user_agent(USER_AGENT)
            .timeout_connect(Some(TIMEOUT_CONNECT))
            .timeout_recv_body(Some(TIMEOUT_RECV_BODY))
            .timeout_global(Some(TIMEOUT_GLOBAL))
            .build()
            .new_agent();

        Self {
            agent,
            prefetch_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Fetch a URL, returning raw HTML.
    /// Checks prefetch cache first.
    /// On HTTP errors (4xx, 5xx) or network failures, returns error page HTML.
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
            .header("Accept", "text/html,application/xhtml+xml,*/*")
            .call()
        {
            Ok(r) => r,
            Err(e) => {
                warn!("[CPU:NET] Request error for {url}: {e}");
                let message = friendly_error_message(&e);
                return Ok(generate_error_page(url, &message));
            }
        };

        let status = resp.status();
        let status_code: u16 = status.into();
        info!("[CPU:NET] {url} -> {status_code}");

        // Handle HTTP error status codes gracefully
        if status_code >= 400 {
            let status_text = http_status_text(status_code);
            warn!("[CPU:NET] HTTP {status_code} ({status_text}) for {url}");
            return Ok(generate_error_page(
                url,
                &format!("HTTP {status_code} - {status_text}"),
            ));
        }

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

/// Convert ureq errors into user-friendly messages.
fn friendly_error_message(e: &ureq::Error) -> String {
    match e {
        ureq::Error::Timeout(_) => {
            "Connection timed out. The server took too long to respond.".into()
        }
        ureq::Error::HostNotFound => {
            format!("Could not find the server. Check the URL and your internet connection.")
        }
        ureq::Error::ConnectionFailed => {
            "Connection failed. The server may be down or unreachable.".into()
        }
        ureq::Error::TooManyRedirects => {
            format!("Too many redirects (limit: {MAX_REDIRECTS}). The page may be misconfigured.")
        }
        ureq::Error::Tls(_) => {
            "Secure connection failed. There may be a problem with the site's certificate.".into()
        }
        _ => {
            format!("Network error: {e}")
        }
    }
}

/// Map HTTP status codes to human-readable text.
fn http_status_text(status: u16) -> &'static str {
    match status {
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        410 => "Gone",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Error",
    }
}

/// Generate a simple HTML error page that the parser and NPU can process.
pub fn generate_error_page(url: &str, message: &str) -> String {
    format!(
        r#"<html>
<head><title>Error - Neural Browser</title></head>
<body>
<main>
<h1>Page could not be loaded</h1>
<p>{message}</p>
<p>URL: {url}</p>
<hr>
<p>Press F5 to retry, or F6 to navigate to a different page.</p>
</main>
</body>
</html>"#
    )
}
