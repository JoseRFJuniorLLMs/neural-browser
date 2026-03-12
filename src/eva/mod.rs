//! EVA AI Assistant — native integration with the Neural Browser.
//!
//! Communicates with EVA server via REST API for text chat.
//! EVA can answer questions about pages, summarize content,
//! and explain selected text.

pub mod panel;

use anyhow::Result;
use log::{info, warn};
use std::time::Duration;
use ureq::tls::{RootCerts, TlsConfig};

const DEFAULT_EVA_URL: &str = "http://136.111.0.47:8091";
const TIMEOUT_GLOBAL: Duration = Duration::from_secs(30);

/// EVA AI client — talks to EVA server via REST.
pub struct EvaClient {
    agent: ureq::Agent,
    base_url: String,
}

impl EvaClient {
    /// Create a new EVA client.
    /// Reads `EVA_URL` env var for server address (default: http://136.111.0.47:8091).
    pub fn new() -> Self {
        let base_url = std::env::var("EVA_URL")
            .unwrap_or_else(|_| DEFAULT_EVA_URL.to_string());

        let tls = TlsConfig::builder()
            .root_certs(RootCerts::WebPki)
            .build();

        let agent = ureq::Agent::config_builder()
            .tls_config(tls)
            .http_status_as_error(false)
            .timeout_global(Some(TIMEOUT_GLOBAL))
            .build()
            .new_agent();

        info!("[EVA] Client initialized, server: {base_url}");

        Self { agent, base_url }
    }

    /// Ask EVA a question, optionally with current page context.
    pub fn ask(&self, message: &str, page_context: &str) -> Result<String> {
        info!("[EVA] Asking: {}", truncate_log(message, 80));
        self.chat_request(message, page_context)
    }

    /// Ask EVA to summarize page content.
    pub fn summarize_page(&self, content: &str) -> Result<String> {
        let msg = "Summarize this page content concisely.";
        self.chat_request(msg, content)
    }

    /// Ask EVA to explain selected text in context.
    pub fn explain(&self, selection: &str, page_context: &str) -> Result<String> {
        let msg = format!("Explain this text: \"{selection}\"");
        self.chat_request(&msg, page_context)
    }

    /// Send a chat request to EVA's REST API.
    fn chat_request(&self, message: &str, context: &str) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);

        // Build JSON body manually to avoid serde dependency in this module
        let body = format!(
            r#"{{"cpf":"","message":"{}","context":"{}"}}"#,
            escape_json(message),
            escape_json(context),
        );

        let resp = match self.agent.post(&url)
            .header("Content-Type", "application/json")
            .send(body.as_bytes())
        {
            Ok(r) => r,
            Err(e) => {
                warn!("[EVA] Request failed: {e}");
                return Ok(format!(
                    "EVA is unreachable right now. Error: {}",
                    friendly_error(&e)
                ));
            }
        };

        let status: u16 = resp.status().into();
        if status >= 400 {
            warn!("[EVA] HTTP {status}");
            return Ok(format!("EVA returned an error (HTTP {status}). Try again later."));
        }

        let body_str = resp.into_body()
            .read_to_string()
            .unwrap_or_else(|_| String::from(r#"{"response":"Failed to read EVA response."}"#));

        // Parse response field from JSON manually
        let response = extract_json_field(&body_str, "response")
            .unwrap_or_else(|| body_str.clone());

        info!("[EVA] Response received ({} chars)", response.len());
        Ok(response)
    }
}

/// Escape a string for JSON embedding (minimal: quotes, backslashes, newlines).
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\x20' => {} // skip control chars
            c => out.push(c),
        }
    }
    out
}

/// Extract a string field from a JSON object (simple parser, no serde needed).
fn extract_json_field(json: &str, field: &str) -> Option<String> {
    let pattern = format!("\"{}\"", field);
    let pos = json.find(&pattern)?;
    let after_key = &json[pos + pattern.len()..];

    // Skip whitespace and colon
    let after_colon = after_key.trim_start();
    let after_colon = after_colon.strip_prefix(':')?;
    let after_colon = after_colon.trim_start();

    // Expect opening quote
    let after_colon = after_colon.strip_prefix('"')?;

    // Read until unescaped closing quote
    let mut result = String::new();
    let mut chars = after_colon.chars();
    loop {
        match chars.next() {
            None => break,
            Some('\\') => {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('"') => result.push('"'),
                    Some('\\') => result.push('\\'),
                    Some(c) => { result.push('\\'); result.push(c); }
                    None => break,
                }
            }
            Some('"') => break,
            Some(c) => result.push(c),
        }
    }

    Some(result)
}

/// Convert ureq errors to friendly messages.
fn friendly_error(e: &ureq::Error) -> String {
    match e {
        ureq::Error::Timeout(_) => "Connection timed out.".into(),
        ureq::Error::HostNotFound => "Could not find EVA server.".into(),
        ureq::Error::ConnectionFailed => "Connection to EVA failed.".into(),
        _ => format!("{e}"),
    }
}

/// Truncate a string for log display (UTF-8 safe).
fn truncate_log(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    // Find a valid UTF-8 boundary at or before `max`
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json(r#"hello "world""#), r#"hello \"world\""#);
        assert_eq!(escape_json("line\nnew"), r#"line\nnew"#);
        assert_eq!(escape_json("back\\slash"), r#"back\\slash"#);
    }

    #[test]
    fn test_extract_json_field() {
        let json = r#"{"response":"Hello from EVA!","cpf":""}"#;
        assert_eq!(
            extract_json_field(json, "response"),
            Some("Hello from EVA!".to_string())
        );
        assert_eq!(
            extract_json_field(json, "cpf"),
            Some(String::new())
        );
        assert_eq!(extract_json_field(json, "missing"), None);
    }

    #[test]
    fn test_extract_json_field_with_escapes() {
        let json = r#"{"response":"Line 1\nLine 2"}"#;
        assert_eq!(
            extract_json_field(json, "response"),
            Some("Line 1\nLine 2".to_string())
        );
    }
}
