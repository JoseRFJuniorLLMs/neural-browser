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

use anyhow::Result;
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
    prefetch_order: Mutex<Vec<String>>,
    /// Cache for fetched image bytes (URL -> raw bytes). Up to 50 entries.
    image_cache: Mutex<HashMap<String, Vec<u8>>>,
    image_order: Mutex<Vec<String>>,
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
            prefetch_order: Mutex::new(Vec::new()),
            image_cache: Mutex::new(HashMap::new()),
            image_order: Mutex::new(Vec::new()),
        }
    }

    /// Fetch a URL, returning raw HTML.
    /// Checks prefetch cache first.
    /// On HTTP errors (4xx, 5xx) or network failures, returns error page HTML.
    pub fn fetch(&self, url: &str) -> Result<String> {
        // SECURITY: Only allow http(s) schemes for browser-level fetch
        if let Ok(parsed) = url::Url::parse(url) {
            match parsed.scheme() {
                "http" | "https" => {}
                scheme => {
                    warn!("[CPU:NET] Blocked scheme: {scheme}://");
                    return Ok(generate_error_page(url, &format!("Blocked: {scheme}:// URLs are not supported")));
                }
            }
        }

        // Check prefetch cache
        {
            let mut cache = self.prefetch_cache.lock();
            if let Some(html) = cache.remove(url) {
                let mut order = self.prefetch_order.lock();
                order.retain(|k| k != url);
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

        // Accumulate raw bytes first, convert to UTF-8 once at the end
        let mut raw_bytes: Vec<u8> = Vec::new();
        let mut binding = resp.into_body();
        let mut reader = binding.as_reader();
        let mut buf = [0u8; 8192];
        loop {
            match std::io::Read::read(&mut reader, &mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    raw_bytes.extend_from_slice(&buf[..n]);
                    if raw_bytes.len() > MAX_BODY_SIZE {
                        warn!("[CPU:NET] Body exceeds {} bytes, truncating", MAX_BODY_SIZE);
                        break;
                    }
                }
                Err(e) => {
                    if raw_bytes.is_empty() {
                        return Err(anyhow::anyhow!("Failed to read response body: {e}"));
                    }
                    warn!("[CPU:NET] Body read error after {} bytes: {e}", raw_bytes.len());
                    break;
                }
            }
        }

        let body = String::from_utf8(raw_bytes)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());

        Ok(body)
    }

    /// Maximum prefetch cache entries to prevent unbounded memory growth.
    const MAX_PREFETCH_ENTRIES: usize = 10;

    /// Prefetch a URL into cache (NPU predicted the user will click it).
    pub fn prefetch(&self, url: &str) -> Result<()> {
        let html = self.fetch(url)?;
        let mut cache = self.prefetch_cache.lock();
        // Evict oldest entries if cache is full (FIFO order)
        if cache.len() >= Self::MAX_PREFETCH_ENTRIES {
            let mut order = self.prefetch_order.lock();
            if let Some(oldest) = order.first().cloned() {
                cache.remove(&oldest);
                order.remove(0);
            }
        }
        cache.insert(url.to_string(), html);
        self.prefetch_order.lock().push(url.to_string());
        Ok(())
    }

    /// Maximum image download size (5 MB).
    const MAX_IMAGE_SIZE: usize = 5 * 1024 * 1024;

    /// Maximum image cache entries.
    const MAX_IMAGE_CACHE_ENTRIES: usize = 50;

    /// Fetch image bytes from a URL. Returns raw bytes on success.
    /// Checks the image cache first; on miss, fetches from network and caches.
    pub fn fetch_image(&self, url: &str) -> Result<Vec<u8>> {
        // Check image cache
        {
            let cache = self.image_cache.lock();
            if let Some(bytes) = cache.get(url) {
                info!("[CPU:NET] Image cache hit for {url}");
                return Ok(bytes.clone());
            }
        }

        info!("[CPU:NET] Fetching image {url}");
        let resp = self.agent.get(url)
            .header("Accept", "image/*,*/*")
            .call()
            .map_err(|e| {
                warn!("[CPU:NET] Image fetch error for {url}: {e}");
                anyhow::anyhow!("Image fetch failed: {}", friendly_error_message(&e))
            })?;

        let status: u16 = resp.status().into();
        if status >= 400 {
            return Err(anyhow::anyhow!("HTTP {status} for image {url}"));
        }

        let mut bytes = Vec::new();
        let mut binding = resp.into_body();
        let mut reader = binding.as_reader();
        let mut buf = [0u8; 8192];
        let mut read_ok = false;
        loop {
            match std::io::Read::read(&mut reader, &mut buf) {
                Ok(0) => { read_ok = true; break; }
                Ok(n) => {
                    bytes.extend_from_slice(&buf[..n]);
                    if bytes.len() > Self::MAX_IMAGE_SIZE {
                        warn!("[CPU:NET] Image exceeds {} bytes, aborting", Self::MAX_IMAGE_SIZE);
                        return Err(anyhow::anyhow!("Image too large (>{} MB)", Self::MAX_IMAGE_SIZE / 1024 / 1024));
                    }
                }
                Err(e) => {
                    if bytes.is_empty() {
                        return Err(anyhow::anyhow!("Failed to read image body: {e}"));
                    }
                    warn!("[CPU:NET] Image read error after {} bytes: {e}", bytes.len());
                    break;
                }
            }
        }

        // Only cache when read completed successfully (no partial images)
        if read_ok {
            let mut cache = self.image_cache.lock();
            if cache.len() >= Self::MAX_IMAGE_CACHE_ENTRIES {
                // Evict oldest entry (FIFO order)
                let mut order = self.image_order.lock();
                if let Some(oldest) = order.first().cloned() {
                    cache.remove(&oldest);
                    order.remove(0);
                }
            }
            cache.insert(url.to_string(), bytes.clone());
            self.image_order.lock().push(url.to_string());
        }

        Ok(bytes)
    }
}

/// Convert ureq errors into user-friendly messages.
fn friendly_error_message(e: &ureq::Error) -> String {
    match e {
        ureq::Error::Timeout(_) => {
            "Connection timed out. The server took too long to respond.".into()
        }
        ureq::Error::HostNotFound => {
            "Could not find the server. Check the URL and your internet connection.".to_string()
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

/// Maximum response body size (10 MB) to prevent memory exhaustion.
const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;

/// Generate a simple HTML error page that the parser and NPU can process.
/// URL and message are HTML-escaped to prevent XSS.
pub fn generate_error_page(url: &str, message: &str) -> String {
    let safe_url = html_escape(url);
    let safe_msg = html_escape(message);
    format!(
        r#"<html>
<head><title>Error - Neural Browser</title></head>
<body>
<main>
<h1>Page could not be loaded</h1>
<p>{safe_msg}</p>
<p>URL: {safe_url}</p>
<hr>
<p>Press F5 to retry, or F6 to navigate to a different page.</p>
</main>
</body>
</html>"#
    )
}

/// Escape HTML special characters to prevent XSS in error pages.
fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(c),
        }
    }
    out
}
