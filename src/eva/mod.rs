//! AI Assistant — multi-provider integration with Neural Browser.
//!
//! Supports multiple AI backends:
//! - EVA (default, local server at 136.111.0.47:8091)
//! - Claude (Anthropic API)
//! - Gemini (Google AI)
//! - GPT-4 (OpenAI API)
//!
//! Also includes voice response via EVA's native audio model.

pub mod panel;

use anyhow::Result;
use log::{info, warn};
use std::time::Duration;
use ureq::tls::{RootCerts, TlsConfig};

// ── Default endpoints ──

const DEFAULT_EVA_URL: &str = "http://136.111.0.47:8091";
const DEFAULT_CLAUDE_URL: &str = "https://api.anthropic.com/v1/messages";
const DEFAULT_GEMINI_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent";
const DEFAULT_OPENAI_URL: &str = "https://api.openai.com/v1/chat/completions";
const TIMEOUT_GLOBAL: Duration = Duration::from_secs(30);

/// Which AI provider to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AiProvider {
    Eva,
    Claude,
    Gemini,
    Gpt4,
}

impl AiProvider {
    pub fn name(&self) -> &'static str {
        match self {
            AiProvider::Eva => "EVA",
            AiProvider::Claude => "Claude",
            AiProvider::Gemini => "Gemini",
            AiProvider::Gpt4 => "GPT-4",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            AiProvider::Eva => AiProvider::Claude,
            AiProvider::Claude => AiProvider::Gemini,
            AiProvider::Gemini => AiProvider::Gpt4,
            AiProvider::Gpt4 => AiProvider::Eva,
        }
    }
}

/// Multi-AI client — talks to EVA, Claude, Gemini, or GPT-4 via REST.
pub struct AiClient {
    agent: ureq::Agent,
    eva_url: String,
    claude_key: Option<String>,
    gemini_key: Option<String>,
    openai_key: Option<String>,
    // EVA voice endpoint
    eva_voice_url: String,
}

impl AiClient {
    /// Create a new multi-AI client.
    /// Reads env vars for configuration:
    /// - EVA_URL (default: http://136.111.0.47:8091)
    /// - ANTHROPIC_API_KEY for Claude
    /// - GEMINI_API_KEY for Gemini
    /// - OPENAI_API_KEY for GPT-4
    pub fn new() -> Self {
        let eva_url = std::env::var("EVA_URL")
            .unwrap_or_else(|_| DEFAULT_EVA_URL.to_string());
        let eva_voice_url = format!("{}/api/voice", eva_url);

        let claude_key = std::env::var("ANTHROPIC_API_KEY").ok();
        let gemini_key = std::env::var("GEMINI_API_KEY").ok();
        let openai_key = std::env::var("OPENAI_API_KEY").ok();

        let tls = TlsConfig::builder()
            .root_certs(RootCerts::WebPki)
            .build();

        let agent = ureq::Agent::config_builder()
            .tls_config(tls)
            .http_status_as_error(false)
            .timeout_global(Some(TIMEOUT_GLOBAL))
            .build()
            .new_agent();

        info!("[AI] Multi-AI client initialized:");
        info!("[AI]   EVA: {eva_url}");
        info!("[AI]   Claude: {}", if claude_key.is_some() { "configured" } else { "no key" });
        info!("[AI]   Gemini: {}", if gemini_key.is_some() { "configured" } else { "no key" });
        info!("[AI]   GPT-4: {}", if openai_key.is_some() { "configured" } else { "no key" });

        Self {
            agent,
            eva_url,
            claude_key,
            gemini_key,
            openai_key,
            eva_voice_url,
        }
    }

    /// Check which providers are available (have API keys configured).
    pub fn available_providers(&self) -> Vec<AiProvider> {
        let mut providers = vec![AiProvider::Eva]; // EVA always available
        if self.claude_key.is_some() {
            providers.push(AiProvider::Claude);
        }
        if self.gemini_key.is_some() {
            providers.push(AiProvider::Gemini);
        }
        if self.openai_key.is_some() {
            providers.push(AiProvider::Gpt4);
        }
        providers
    }

    /// Ask a question using the specified provider.
    pub fn ask(&self, provider: AiProvider, message: &str, page_context: &str) -> Result<String> {
        info!("[AI:{}] Asking: {}", provider.name(), truncate_log(message, 80));
        match provider {
            AiProvider::Eva => self.eva_request(message, page_context),
            AiProvider::Claude => self.claude_request(message, page_context),
            AiProvider::Gemini => self.gemini_request(message, page_context),
            AiProvider::Gpt4 => self.gpt4_request(message, page_context),
        }
    }

    /// Summarize page content using the specified provider.
    pub fn summarize(&self, provider: AiProvider, content: &str) -> Result<String> {
        self.ask(provider, "Summarize this page content concisely.", content)
    }

    /// Request voice response from EVA (returns status text, audio plays server-side).
    pub fn request_voice(&self, text: &str) -> Result<String> {
        info!("[AI:VOICE] Requesting voice for: {}", truncate_log(text, 60));

        let body = format!(
            r#"{{"text":"{}","voice":"eva"}}"#,
            escape_json(text),
        );

        match self.agent.post(&self.eva_voice_url)
            .header("Content-Type", "application/json")
            .send(body.as_bytes())
        {
            Ok(resp) => {
                let status: u16 = resp.status().into();
                if status >= 400 {
                    Ok("Voice unavailable.".into())
                } else {
                    Ok("🔊 Voice response sent.".into())
                }
            }
            Err(_) => Ok("Voice service unreachable.".into()),
        }
    }

    // ── EVA provider ──

    fn eva_request(&self, message: &str, context: &str) -> Result<String> {
        let url = format!("{}/api/chat", self.eva_url);
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
                warn!("[AI:EVA] Request failed: {e}");
                return Ok(format!("EVA unreachable: {}", friendly_error(&e)));
            }
        };

        let status: u16 = resp.status().into();
        if status >= 400 {
            return Ok(format!("EVA error (HTTP {status})."));
        }

        let body_str = resp.into_body()
            .read_to_string()
            .unwrap_or_else(|_| r#"{"response":"Read error."}"#.into());

        Ok(extract_json_field(&body_str, "response")
            .unwrap_or_else(|| body_str))
    }

    // ── Claude provider (Anthropic API) ──

    fn claude_request(&self, message: &str, context: &str) -> Result<String> {
        let api_key = match &self.claude_key {
            Some(k) => k,
            None => return Ok("Claude API key not configured. Set ANTHROPIC_API_KEY.".into()),
        };

        let system_prompt = if context.is_empty() {
            "You are a helpful browser AI assistant. Answer concisely.".to_string()
        } else {
            format!(
                "You are a helpful browser AI assistant. The user is viewing a webpage. Page content:\n\n{}",
                truncate_context(context, 4000)
            )
        };

        let body = format!(
            "{{\"model\":\"claude-sonnet-4-20250514\",\"max_tokens\":1024,\"system\":\"{}\",\"messages\":[{{\"role\":\"user\",\"content\":\"{}\"}}]}}",
            escape_json(&system_prompt),
            escape_json(message),
        );

        let resp = match self.agent.post(DEFAULT_CLAUDE_URL)
            .header("Content-Type", "application/json")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .send(body.as_bytes())
        {
            Ok(r) => r,
            Err(e) => {
                warn!("[AI:Claude] Request failed: {e}");
                return Ok(format!("Claude unreachable: {}", friendly_error(&e)));
            }
        };

        let status: u16 = resp.status().into();
        if status >= 400 {
            let err_body = resp.into_body().read_to_string().unwrap_or_default();
            warn!("[AI:Claude] HTTP {status}: {}", truncate_log(&err_body, 200));
            return Ok(format!("Claude error (HTTP {status})."));
        }

        let body_str = resp.into_body().read_to_string().unwrap_or_default();

        // Parse Claude response: {"content":[{"type":"text","text":"..."}]}
        extract_json_field(&body_str, "text")
            .map(|t| Ok(t))
            .unwrap_or_else(|| Ok(body_str))
    }

    // ── Gemini provider (Google AI) ──

    fn gemini_request(&self, message: &str, context: &str) -> Result<String> {
        let api_key = match &self.gemini_key {
            Some(k) => k,
            None => return Ok("Gemini API key not configured. Set GEMINI_API_KEY.".into()),
        };

        let prompt = if context.is_empty() {
            message.to_string()
        } else {
            format!(
                "You are a helpful browser AI assistant. Page content:\n{}\n\nUser question: {}",
                truncate_context(context, 4000),
                message
            )
        };

        let url = format!("{}?key={}", DEFAULT_GEMINI_URL, api_key);
        let body = format!(
            "{{\"contents\":[{{\"parts\":[{{\"text\":\"{}\"}}]}}],\"generationConfig\":{{\"maxOutputTokens\":1024}}}}",
            escape_json(&prompt),
        );

        let resp = match self.agent.post(&url)
            .header("Content-Type", "application/json")
            .send(body.as_bytes())
        {
            Ok(r) => r,
            Err(e) => {
                warn!("[AI:Gemini] Request failed: {e}");
                return Ok(format!("Gemini unreachable: {}", friendly_error(&e)));
            }
        };

        let status: u16 = resp.status().into();
        if status >= 400 {
            return Ok(format!("Gemini error (HTTP {status})."));
        }

        let body_str = resp.into_body().read_to_string().unwrap_or_default();

        // Parse Gemini response: {"candidates":[{"content":{"parts":[{"text":"..."}]}}]}
        extract_json_field(&body_str, "text")
            .map(|t| Ok(t))
            .unwrap_or_else(|| Ok(body_str))
    }

    // ── GPT-4 provider (OpenAI API) ──

    fn gpt4_request(&self, message: &str, context: &str) -> Result<String> {
        let api_key = match &self.openai_key {
            Some(k) => k,
            None => return Ok("OpenAI API key not configured. Set OPENAI_API_KEY.".into()),
        };

        let system_msg = if context.is_empty() {
            "You are a helpful browser AI assistant. Answer concisely.".to_string()
        } else {
            format!(
                "You are a helpful browser AI assistant. The user is viewing a webpage. Page content:\n\n{}",
                truncate_context(context, 4000)
            )
        };

        let body = format!(
            "{{\"model\":\"gpt-4o\",\"max_tokens\":1024,\"messages\":[{{\"role\":\"system\",\"content\":\"{}\"}},{{\"role\":\"user\",\"content\":\"{}\"}}]}}",
            escape_json(&system_msg),
            escape_json(message),
        );

        let resp = match self.agent.post(DEFAULT_OPENAI_URL)
            .header("Content-Type", "application/json")
            .header("Authorization", &format!("Bearer {api_key}"))
            .send(body.as_bytes())
        {
            Ok(r) => r,
            Err(e) => {
                warn!("[AI:GPT-4] Request failed: {e}");
                return Ok(format!("GPT-4 unreachable: {}", friendly_error(&e)));
            }
        };

        let status: u16 = resp.status().into();
        if status >= 400 {
            return Ok(format!("GPT-4 error (HTTP {status})."));
        }

        let body_str = resp.into_body().read_to_string().unwrap_or_default();

        // Parse OpenAI response: {"choices":[{"message":{"content":"..."}}]}
        extract_json_field(&body_str, "content")
            .map(|t| Ok(t))
            .unwrap_or_else(|| Ok(body_str))
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

    let after_colon = after_key.trim_start();
    let after_colon = after_colon.strip_prefix(':')?;
    let after_colon = after_colon.trim_start();
    let after_colon = after_colon.strip_prefix('"')?;

    let mut result = String::new();
    let mut chars = after_colon.chars();
    loop {
        match chars.next() {
            None => break,
            Some('\\') => match chars.next() {
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('"') => result.push('"'),
                Some('\\') => result.push('\\'),
                Some(c) => { result.push('\\'); result.push(c); }
                None => break,
            },
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
        ureq::Error::HostNotFound => "Server not found.".into(),
        ureq::Error::ConnectionFailed => "Connection failed.".into(),
        _ => format!("{e}"),
    }
}

/// Truncate a string for log display (UTF-8 safe).
fn truncate_log(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Truncate context for API calls (UTF-8 safe, word boundary).
fn truncate_context(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let truncated = &s[..end];
    if let Some(pos) = truncated.rfind(' ') {
        format!("{}...", &s[..pos])
    } else {
        format!("{}...", truncated)
    }
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
        assert_eq!(extract_json_field(json, "response"), Some("Hello from EVA!".into()));
        assert_eq!(extract_json_field(json, "cpf"), Some(String::new()));
        assert_eq!(extract_json_field(json, "missing"), None);
    }

    #[test]
    fn test_extract_json_field_with_escapes() {
        let json = r#"{"response":"Line 1\nLine 2"}"#;
        assert_eq!(extract_json_field(json, "response"), Some("Line 1\nLine 2".into()));
    }

    #[test]
    fn test_provider_cycling() {
        assert_eq!(AiProvider::Eva.next(), AiProvider::Claude);
        assert_eq!(AiProvider::Claude.next(), AiProvider::Gemini);
        assert_eq!(AiProvider::Gemini.next(), AiProvider::Gpt4);
        assert_eq!(AiProvider::Gpt4.next(), AiProvider::Eva);
    }

    #[test]
    fn test_truncate_context() {
        assert_eq!(truncate_context("short", 100), "short");
        let long = "word1 word2 word3 word4 word5 word6";
        let t = truncate_context(long, 15);
        assert!(t.len() < 20);
        assert!(t.ends_with("..."));
    }
}
