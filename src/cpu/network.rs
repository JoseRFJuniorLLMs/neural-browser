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
            // 0 = never auto-follow. Redirects are followed manually in
            // `follow_redirects` so every hop is scheme- and SSRF-validated.
            .max_redirects(0)
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
        self.fetch_inner(url, 0)
    }

    /// Inner fetch with a search-fallback depth counter.
    ///
    /// `search_depth` guards the Google -> DuckDuckGo fallback: without it a
    /// response that keeps matching the "enable JavaScript" heuristic recurses
    /// until the stack overflows.
    fn fetch_inner(&self, url: &str, search_depth: u32) -> Result<String> {
        // SECURITY: Only allow http(s) schemes for browser-level fetch
        if let Err(reason) = validate_fetch_url(url) {
            warn!("[CPU:NET] Blocked {url}: {reason}");
            return Ok(generate_error_page(url, &reason));
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

        // Follow redirects manually, validating every hop.
        let mut current = url.to_string();
        let mut hops = 0u32;
        let resp = loop {
            let resp = match self.agent.get(&current)
                .header("Accept", "text/html,application/xhtml+xml,*/*")
                .call()
            {
                Ok(r) => r,
                Err(e) => {
                    warn!("[CPU:NET] Request error for {current}: {e}");
                    let message = friendly_error_message(&e);
                    return Ok(generate_error_page(url, &message));
                }
            };

            let code: u16 = resp.status().into();
            info!("[CPU:NET] {current} -> {code}");

            let location = resp.headers().get("location")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());

            match next_hop(code, location.as_deref(), &current) {
                Hop::Done => break resp,
                Hop::Blocked(reason) => {
                    warn!("[CPU:NET] Blocked redirect from {current}: {reason}");
                    return Ok(generate_error_page(url, &reason));
                }
                Hop::Redirect(next) => {
                    hops += 1;
                    if hops > MAX_REDIRECTS {
                        warn!("[CPU:NET] Too many redirects for {url}");
                        return Ok(generate_error_page(
                            url,
                            &format!("Too many redirects (limit: {MAX_REDIRECTS}). The page may be misconfigured."),
                        ));
                    }
                    info!("[CPU:NET] Redirect {hops}/{MAX_REDIRECTS} -> {next}");
                    current = next;
                }
            }
        };

        let status_code: u16 = resp.status().into();

        // Handle HTTP error status codes gracefully
        if status_code >= 400 {
            let status_text = http_status_text(status_code);
            warn!("[CPU:NET] HTTP {status_code} ({status_text}) for {url}");
            return Ok(generate_error_page(
                url,
                &format!("HTTP {status_code} - {status_text}"),
            ));
        }

        // ── Detect charset from Content-Type header ──
        let content_type = resp.headers().get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();
        let header_charset = extract_charset(&content_type);

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

        let body = decode_body(&raw_bytes, header_charset.as_deref());

        // Detect Google "enable JavaScript" redirect page — retry with DuckDuckGo.
        // Only once: the fallback target can itself match the heuristic, and an
        // unguarded retry would recurse until the stack overflows.
        if search_depth == 0
            && (body.contains("enablejs")
                || body.contains("Ative o JavaScript")
                || body.contains("Enable JavaScript"))
        {
            if let Some(query) = extract_google_query(url) {
                let ddg_url = format!(
                    "https://html.duckduckgo.com/html/?q={}",
                    query
                );
                info!("[CPU:NET] Google requires JavaScript, falling back to DuckDuckGo: {ddg_url}");
                return self.fetch_inner(&ddg_url, search_depth + 1);
            }
        }

        Ok(body)
    }

    /// Maximum prefetch cache entries to prevent unbounded memory growth.
    const MAX_PREFETCH_ENTRIES: usize = 10;

    /// Prefetch a URL into cache (NPU predicted the user will click it).
    /// Does NOT cache error pages — only successful responses.
    pub fn prefetch(&self, url: &str) -> Result<()> {
        let html = self.fetch(url)?;
        // Don't cache error pages (generated by our own generate_error_page)
        if html.contains("<title>Error - Neural Browser</title>") {
            return Ok(());
        }
        let mut cache = self.prefetch_cache.lock();
        let mut order = self.prefetch_order.lock();
        // Remove old entry from order if re-prefetching same URL
        if cache.contains_key(url) {
            order.retain(|u| u != url);
        }
        // Evict oldest entries if cache is full (FIFO order)
        while cache.len() >= Self::MAX_PREFETCH_ENTRIES {
            if let Some(oldest) = order.first().cloned() {
                cache.remove(&oldest);
                order.remove(0);
            } else {
                break;
            }
        }
        cache.insert(url.to_string(), html);
        order.push(url.to_string());
        Ok(())
    }

    /// Maximum image download size (5 MB).
    const MAX_IMAGE_SIZE: usize = 5 * 1024 * 1024;

    /// Maximum image cache entries.
    const MAX_IMAGE_CACHE_ENTRIES: usize = 50;

    /// Fetch image bytes from a URL. Returns raw bytes on success.
    /// Checks the image cache first; on miss, fetches from network and caches.
    pub fn fetch_image(&self, url: &str) -> Result<Vec<u8>> {
        // SECURITY: image URLs come straight out of page markup, so they get
        // the same scheme validation as a top-level navigation.
        if let Err(reason) = validate_fetch_url(url) {
            return Err(anyhow::anyhow!("Blocked image URL: {reason}"));
        }

        // Check image cache
        {
            let cache = self.image_cache.lock();
            if let Some(bytes) = cache.get(url) {
                info!("[CPU:NET] Image cache hit for {url}");
                return Ok(bytes.clone());
            }
        }

        info!("[CPU:NET] Fetching image {url}");

        // Same manually-validated redirect chain as `fetch`.
        let mut current = url.to_string();
        let mut hops = 0u32;
        let resp = loop {
            let resp = self.agent.get(&current)
                .header("Accept", "image/*,*/*")
                .call()
                .map_err(|e| {
                    warn!("[CPU:NET] Image fetch error for {current}: {e}");
                    anyhow::anyhow!("Image fetch failed: {}", friendly_error_message(&e))
                })?;

            let code: u16 = resp.status().into();
            let location = resp.headers().get("location")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());

            match next_hop(code, location.as_deref(), &current) {
                Hop::Done => break resp,
                Hop::Blocked(reason) => return Err(anyhow::anyhow!("{reason}")),
                Hop::Redirect(next) => {
                    hops += 1;
                    if hops > MAX_REDIRECTS {
                        return Err(anyhow::anyhow!("Too many redirects for image {url}"));
                    }
                    current = next;
                }
            }
        };

        let status: u16 = resp.status().into();
        if status >= 400 {
            return Err(anyhow::anyhow!("HTTP {status} for image {url}"));
        }

        let mut bytes = Vec::new();
        let mut binding = resp.into_body();
        let mut reader = binding.as_reader();
        let mut buf = [0u8; 8192];
        loop {
            match std::io::Read::read(&mut reader, &mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    bytes.extend_from_slice(&buf[..n]);
                    if bytes.len() > Self::MAX_IMAGE_SIZE {
                        warn!("[CPU:NET] Image exceeds {} bytes, aborting", Self::MAX_IMAGE_SIZE);
                        return Err(anyhow::anyhow!("Image too large (>{} MB)", Self::MAX_IMAGE_SIZE / 1024 / 1024));
                    }
                }
                Err(e) => {
                    warn!("[CPU:NET] Image read error after {} bytes: {e}", bytes.len());
                    return Err(anyhow::anyhow!("Failed to read image body: {e}"));
                }
            }
        }

        // Cache the fully-downloaded image
        {
            let mut cache = self.image_cache.lock();
            let mut order = self.image_order.lock();
            // Remove old entry from order if re-fetching same URL
            if cache.contains_key(url) {
                order.retain(|u| u != url);
            }
            // Evict oldest entries if cache is full (FIFO order)
            while cache.len() >= Self::MAX_IMAGE_CACHE_ENTRIES {
                if let Some(oldest) = order.first().cloned() {
                    cache.remove(&oldest);
                    order.remove(0);
                } else {
                    break;
                }
            }
            cache.insert(url.to_string(), bytes.clone());
            order.push(url.to_string());
        }

        Ok(bytes)
    }
}

/// Outcome of inspecting one HTTP response while following redirects.
enum Hop {
    /// Not a redirect — this response is the final one.
    Done,
    /// Follow this absolute URL next (already validated).
    Redirect(String),
    /// Refuse to follow, with a user-facing reason.
    Blocked(String),
}

/// Decide what to do with a response status + Location header.
///
/// SECURITY: this is the choke point that makes redirect following safe.
/// `ureq`'s automatic follower is disabled precisely so that every hop passes
/// through here, where the scheme is re-validated and a public -> private
/// origin jump (the classic SSRF / cloud-metadata attack) is refused.
fn next_hop(status: u16, location: Option<&str>, current: &str) -> Hop {
    if !matches!(status, 301 | 302 | 303 | 307 | 308) {
        return Hop::Done;
    }
    let location = match location {
        Some(l) if !l.trim().is_empty() => l.trim(),
        // A 3xx with no usable Location is just a normal (if odd) response.
        _ => return Hop::Done,
    };

    // Resolve the target against the current URL (Location may be relative).
    let base = match url::Url::parse(current) {
        Ok(b) => b,
        Err(_) => return Hop::Blocked("Invalid URL in redirect chain".into()),
    };
    let target = match base.join(location) {
        Ok(t) => t,
        Err(_) => return Hop::Blocked(format!("Invalid redirect target: {location}")),
    };

    // Re-validate the scheme on every hop: the initial check on the typed URL
    // says nothing about where the server chooses to send us.
    if let Err(reason) = validate_url(&target) {
        return Hop::Blocked(format!("Blocked redirect: {reason}"));
    }

    // Refuse a public -> private jump. Navigating straight to a private address
    // stays allowed (browsing http://localhost is legitimate); it is only the
    // remote server steering us inside the local network that is refused.
    if !is_private_host(&base) && is_private_host(&target) {
        return Hop::Blocked(format!(
            "Blocked redirect: {} tried to redirect into a private address ({})",
            base.host_str().unwrap_or("site"),
            target.host_str().unwrap_or("?"),
        ));
    }

    Hop::Redirect(target.to_string())
}

/// Validate a URL string for browser-level fetching.
/// Returns a user-facing reason on rejection.
fn validate_fetch_url(url: &str) -> std::result::Result<url::Url, String> {
    let parsed = url::Url::parse(url)
        .map_err(|e| format!("Invalid URL: {e}"))?;
    validate_url(&parsed)?;
    Ok(parsed)
}

/// Shared scheme/host validation for an already-parsed URL.
fn validate_url(parsed: &url::Url) -> std::result::Result<(), String> {
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(format!("{scheme}:// URLs are not supported"));
        }
    }
    if parsed.host_str().unwrap_or("").is_empty() {
        return Err("URL has no host".to_string());
    }
    Ok(())
}

/// Whether a URL points at loopback, a private/link-local range, or a bare
/// hostname with no dot (an intranet name like `router` or `localhost`).
fn is_private_host(u: &url::Url) -> bool {
    use std::net::IpAddr;

    let host = match u.host_str() {
        Some(h) => h,
        None => return false,
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');

    if let Ok(ip) = host.parse::<IpAddr>() {
        return match ip {
            IpAddr::V4(v4) => is_private_v4(v4),
            IpAddr::V6(v6) => is_private_v6(v6),
        };
    }

    let lower = host.to_ascii_lowercase();
    lower == "localhost"
        || lower.ends_with(".localhost")
        || lower.ends_with(".local")
        || lower.ends_with(".internal")
        // No dot at all: a single-label intranet name, not a public site.
        || !lower.contains('.')
}

fn is_private_v4(ip: std::net::Ipv4Addr) -> bool {
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()      // 169.254/16 — cloud metadata lives here
        || ip.is_broadcast()
        || ip.is_documentation()
        || ip.is_unspecified()
        // 100.64/10 carrier-grade NAT
        || (ip.octets()[0] == 100 && (64..=127).contains(&ip.octets()[1]))
}

fn is_private_v6(ip: std::net::Ipv6Addr) -> bool {
    if ip.is_loopback() || ip.is_unspecified() {
        return true;
    }
    let seg = ip.segments();
    // fc00::/7 unique local, fe80::/10 link local
    if (seg[0] & 0xfe00) == 0xfc00 || (seg[0] & 0xffc0) == 0xfe80 {
        return true;
    }
    // IPv4-mapped / IPv4-compatible addresses reuse the v4 rules.
    if let Some(v4) = ip.to_ipv4() {
        return is_private_v4(v4);
    }
    false
}

/// Extract the search query from a Google URL (q= parameter).
/// Returns the raw (still URL-encoded) query string if found.
fn extract_google_query(url: &str) -> Option<String> {
    if let Ok(parsed) = url::Url::parse(url) {
        for (key, value) in parsed.query_pairs() {
            if key == "q" {
                return Some(value.into_owned());
            }
        }
    }
    // Also try extracting from enablejs redirect URLs
    if let Some(pos) = url.find("q=") {
        let rest = &url[pos + 2..];
        let end = rest.find('&').unwrap_or(rest.len());
        let query = &rest[..end];
        if !query.is_empty() {
            return Some(query.to_string());
        }
    }
    None
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

// html_escape: use super::html_escape
use super::html_escape;

/// Extract charset from Content-Type header value.
/// E.g. "text/html; charset=iso-8859-1" → Some("iso-8859-1")
fn extract_charset(content_type: &str) -> Option<String> {
    for part in content_type.split(';') {
        let trimmed = part.trim();
        if let Some(value) = trimmed.strip_prefix("charset=") {
            let charset = value.trim().trim_matches('"').trim_matches('\'');
            if !charset.is_empty() {
                return Some(charset.to_lowercase());
            }
        }
    }
    None
}

/// Decode raw bytes to UTF-8 string, handling different charsets.
///
/// Priority:
/// 1. If Content-Type header specifies a charset, use it
/// 2. Check for <meta charset="..."> in the first 1024 bytes
/// 3. Try UTF-8 (most common)
/// 4. Fall back to lossy UTF-8 conversion
fn decode_body(raw: &[u8], header_charset: Option<&str>) -> String {
    // A byte-order mark outranks every other signal (HTML spec: BOM sniffing
    // wins over both the Content-Type header and <meta charset>). Without this,
    // a UTF-16 document decodes to interleaved NUL bytes and renders as garbage.
    if let Some((encoding, bom_len)) = detect_bom(raw) {
        let (decoded, _, _) = encoding.decode(&raw[bom_len..]);
        info!("[CPU:NET] BOM detected: decoding as {}", encoding.name());
        return decoded.into_owned();
    }

    // Determine charset: header > meta tag > utf-8 default
    let charset = header_charset
        .map(|s| s.to_string())
        .or_else(|| detect_meta_charset(raw))
        .unwrap_or_else(|| "utf-8".to_string());

    // UTF-8 fast path
    if charset == "utf-8" || charset == "utf8" {
        return String::from_utf8(raw.to_vec())
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());
    }

    // Use encoding_rs for other charsets
    if let Some(encoding) = encoding_rs::Encoding::for_label(charset.as_bytes()) {
        let (decoded, _, had_errors) = encoding.decode(raw);
        if had_errors {
            info!("[CPU:NET] Charset {charset}: decoded with some replacement chars");
        } else {
            info!("[CPU:NET] Charset {charset}: decoded successfully");
        }
        decoded.into_owned()
    } else {
        // Unknown charset — fall back to lossy UTF-8
        warn!("[CPU:NET] Unknown charset '{charset}', falling back to UTF-8 lossy");
        String::from_utf8_lossy(raw).into_owned()
    }
}

/// Detect a byte-order mark. Returns the encoding and the BOM length in bytes.
fn detect_bom(raw: &[u8]) -> Option<(&'static encoding_rs::Encoding, usize)> {
    if raw.starts_with(&[0xEF, 0xBB, 0xBF]) {
        Some((encoding_rs::UTF_8, 3))
    } else if raw.starts_with(&[0xFF, 0xFE]) {
        Some((encoding_rs::UTF_16LE, 2))
    } else if raw.starts_with(&[0xFE, 0xFF]) {
        Some((encoding_rs::UTF_16BE, 2))
    } else {
        None
    }
}

/// Detect charset from <meta> tags in the first 1024 bytes of HTML.
/// Looks for:
///   <meta charset="...">
///   <meta http-equiv="Content-Type" content="text/html; charset=...">
fn detect_meta_charset(raw: &[u8]) -> Option<String> {
    // Only scan the head of the document (first 2048 bytes is enough)
    let scan_len = raw.len().min(2048);

    // BOM-less UTF-16 shows up as ASCII interleaved with NUL bytes, where a
    // lossy UTF-8 scan finds nothing. Strip the NULs before pattern matching so
    // `<meta charset>` is still found in such documents.
    let scan: Vec<u8> = if looks_like_utf16(&raw[..scan_len]) {
        raw[..scan_len].iter().copied().filter(|&b| b != 0).collect()
    } else {
        raw[..scan_len].to_vec()
    };

    // Convert to lossy string just for scanning (safe — we only look at ASCII patterns)
    let head = String::from_utf8_lossy(&scan);
    let head_lower = head.to_lowercase();

    // Pattern 1: <meta charset="...">
    if let Some(pos) = head_lower.find("charset=") {
        let rest = &head_lower[pos + 8..];
        // Remove quotes
        let rest = rest.trim_start_matches(|c: char| c == '"' || c == '\'' || c == ' ');
        let end = rest.find(|c: char| c == '"' || c == '\'' || c == ';' || c == '>' || c == ' ')
            .unwrap_or(rest.len());
        let charset = rest[..end].trim();
        if !charset.is_empty() {
            return Some(charset.to_string());
        }
    }

    None
}

/// Heuristic: BOM-less UTF-16 text has a NUL in most byte pairs.
/// Requires a sample of at least 16 bytes to avoid false positives.
fn looks_like_utf16(sample: &[u8]) -> bool {
    if sample.len() < 16 {
        return false;
    }
    let nulls = sample.iter().filter(|&&b| b == 0).count();
    nulls * 3 >= sample.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> url::Url {
        url::Url::parse(s).unwrap()
    }

    #[test]
    fn non_http_schemes_are_rejected() {
        assert!(validate_fetch_url("file:///etc/passwd").is_err());
        assert!(validate_fetch_url("javascript:alert(1)").is_err());
        assert!(validate_fetch_url("gopher://example.com/").is_err());
        assert!(validate_fetch_url("data:text/html,hi").is_err());
    }

    #[test]
    fn http_and_https_are_accepted() {
        assert!(validate_fetch_url("http://example.com").is_ok());
        assert!(validate_fetch_url("https://example.com/path?q=1").is_ok());
    }

    #[test]
    fn non_redirect_status_finishes_the_chain() {
        assert!(matches!(next_hop(200, None, "https://example.com"), Hop::Done));
        assert!(matches!(next_hop(404, None, "https://example.com"), Hop::Done));
        // A 3xx without a Location header has nowhere to go.
        assert!(matches!(next_hop(302, None, "https://example.com"), Hop::Done));
    }

    #[test]
    fn relative_redirect_is_resolved_against_current_url() {
        match next_hop(302, Some("/next"), "https://example.com/a/b") {
            Hop::Redirect(next) => assert_eq!(next, "https://example.com/next"),
            other => panic!("expected redirect, got {}", match other {
                Hop::Done => "done", Hop::Blocked(_) => "blocked", _ => "?",
            }),
        }
    }

    #[test]
    fn redirect_to_dangerous_scheme_is_blocked() {
        // The initial URL check says nothing about where the server sends us.
        assert!(matches!(
            next_hop(302, Some("file:///etc/passwd"), "https://example.com"),
            Hop::Blocked(_)
        ));
        assert!(matches!(
            next_hop(301, Some("gopher://evil.test/"), "https://example.com"),
            Hop::Blocked(_)
        ));
    }

    #[test]
    fn public_site_cannot_redirect_into_private_network() {
        // Cloud metadata endpoint — the classic SSRF target.
        assert!(matches!(
            next_hop(302, Some("http://169.254.169.254/latest/meta-data/"), "https://example.com"),
            Hop::Blocked(_)
        ));
        assert!(matches!(
            next_hop(302, Some("http://127.0.0.1:8080/admin"), "https://example.com"),
            Hop::Blocked(_)
        ));
        assert!(matches!(
            next_hop(302, Some("http://192.168.1.1/"), "https://example.com"),
            Hop::Blocked(_)
        ));
    }

    #[test]
    fn private_host_may_redirect_within_itself() {
        // Browsing a local dev server must keep working.
        assert!(matches!(
            next_hop(302, Some("http://localhost:3000/login"), "http://localhost:3000/"),
            Hop::Redirect(_)
        ));
    }

    #[test]
    fn private_host_detection() {
        assert!(is_private_host(&u("http://localhost/")));
        assert!(is_private_host(&u("http://127.0.0.1/")));
        assert!(is_private_host(&u("http://10.0.0.5/")));
        assert!(is_private_host(&u("http://172.16.3.4/")));
        assert!(is_private_host(&u("http://192.168.0.1/")));
        assert!(is_private_host(&u("http://169.254.169.254/")));
        assert!(is_private_host(&u("http://[::1]/")));
        assert!(is_private_host(&u("http://router/"))); // single-label intranet name
        assert!(is_private_host(&u("http://nas.local/")));

        assert!(!is_private_host(&u("https://example.com/")));
        assert!(!is_private_host(&u("https://8.8.8.8/")));
    }

    #[test]
    fn bom_overrides_everything() {
        // UTF-16LE BOM + "hi"
        let raw = vec![0xFF, 0xFE, b'h', 0x00, b'i', 0x00];
        assert_eq!(decode_body(&raw, Some("iso-8859-1")), "hi");

        // UTF-8 BOM must not leak into the document text.
        let mut utf8 = vec![0xEF, 0xBB, 0xBF];
        utf8.extend_from_slice("olá".as_bytes());
        assert_eq!(decode_body(&utf8, None), "olá");
    }

    #[test]
    fn utf16_without_bom_still_yields_its_meta_charset() {
        // "<meta charset=utf-16>" encoded as UTF-16LE, no BOM.
        let text = "<meta charset=\"utf-16\">padding padding";
        let raw: Vec<u8> = text.encode_utf16()
            .flat_map(|c| c.to_le_bytes())
            .collect();
        assert_eq!(detect_meta_charset(&raw).as_deref(), Some("utf-16"));
    }

    #[test]
    fn ascii_body_is_not_mistaken_for_utf16() {
        let raw = b"<html><head><meta charset=\"iso-8859-1\"></head></html>";
        assert!(!looks_like_utf16(raw));
        assert_eq!(detect_meta_charset(raw).as_deref(), Some("iso-8859-1"));
    }

    #[test]
    fn charset_from_content_type_header() {
        assert_eq!(extract_charset("text/html; charset=iso-8859-1").as_deref(), Some("iso-8859-1"));
        assert_eq!(extract_charset("text/html; charset=\"UTF-8\"").as_deref(), Some("utf-8"));
        assert_eq!(extract_charset("text/html").as_deref(), None);
    }

    #[test]
    fn latin1_body_decodes_via_header_charset() {
        // 0xE7 is 'ç' in ISO-8859-1 and invalid UTF-8.
        let raw = vec![b'a', 0xE7, b'o'];
        assert_eq!(decode_body(&raw, Some("iso-8859-1")), "aço");
    }
}
