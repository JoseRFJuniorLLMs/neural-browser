//! Semantic History — stores page visits in NietzscheDB for semantic search.
//!
//! Uses the NietzscheDB HTTP REST API (baseserver) to avoid heavy gRPC deps.
//! Collection: `browser_history` (128D, poincare metric).
//!
//! Currently uses zero-vectors for embeddings (no local embedding model yet).
//! Once an embedding model is wired in, KNN search will return semantically
//! similar pages instead of arbitrary results.

use log::{info, warn, error};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

const COLLECTION_NAME: &str = "browser_history";
const DIMENSION: usize = 128;
const METRIC: &str = "poincare";
const DEFAULT_URL: &str = "http://136.111.0.47:8080";
const TIMEOUT: Duration = Duration::from_secs(10);

/// A page visit stored in NietzscheDB.
#[derive(Debug, Clone)]
pub struct PageMemory {
    pub url: String,
    pub title: String,
    pub summary: String,
    pub visited_at: String,
}

/// Semantic history backed by NietzscheDB's HTTP API.
pub struct SemanticMemory {
    agent: ureq::Agent,
    base_url: String,
    /// Monotonic ID counter for vector inserts (baseserver uses u32 IDs).
    next_id: AtomicU32,
}

impl SemanticMemory {
    /// Create a new SemanticMemory, connecting to NietzscheDB.
    /// Reads `NIETZSCHE_URL` env var (default: http://136.111.0.47:8080).
    /// Creates the `browser_history` collection if it doesn't exist.
    pub fn new() -> Self {
        let base_url = std::env::var("NIETZSCHE_URL")
            .unwrap_or_else(|_| DEFAULT_URL.to_string());

        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(TIMEOUT))
            .build()
            .new_agent();

        let mem = Self {
            agent,
            base_url,
            next_id: AtomicU32::new(1),
        };

        // Ensure collection exists
        mem.ensure_collection();

        info!("[MEMORY] Semantic history initialized (NietzscheDB at {})", mem.base_url);
        mem
    }

    /// Check if collection exists, create it if not.
    fn ensure_collection(&self) {
        let url = format!("{}/api/collections", self.base_url);

        // List collections to check if ours exists
        let exists = match self.agent.get(&url).call() {
            Ok(resp) => {
                let status: u16 = resp.status().into();
                if status >= 400 {
                    warn!("[MEMORY] Failed to list collections (HTTP {status})");
                    false
                } else {
                    let body = resp.into_body()
                        .read_to_string()
                        .unwrap_or_default();
                    // Simple check: look for our collection name in the JSON array
                    body.contains(&format!("\"name\":\"{}\"", COLLECTION_NAME))
                        || body.contains(&format!("\"name\": \"{}\"", COLLECTION_NAME))
                }
            }
            Err(e) => {
                warn!("[MEMORY] Cannot reach NietzscheDB: {e}");
                return;
            }
        };

        if exists {
            info!("[MEMORY] Collection '{}' already exists", COLLECTION_NAME);
            // Seed the ID counter from the collection's current count
            self.sync_next_id();
            return;
        }

        // Create collection
        info!("[MEMORY] Creating collection '{}' ({}D, {})", COLLECTION_NAME, DIMENSION, METRIC);
        let body = format!(
            r#"{{"name":"{}","dimension":{},"metric":"{}"}}"#,
            COLLECTION_NAME, DIMENSION, METRIC,
        );

        match self.agent.post(&url)
            .header("Content-Type", "application/json")
            .send(body.as_bytes())
        {
            Ok(resp) => {
                let status: u16 = resp.status().into();
                if status < 300 {
                    info!("[MEMORY] Collection created successfully");
                } else {
                    let err = resp.into_body().read_to_string().unwrap_or_default();
                    warn!("[MEMORY] Failed to create collection (HTTP {status}): {err}");
                }
            }
            Err(e) => {
                error!("[MEMORY] Failed to create collection: {e}");
            }
        }
    }

    /// Sync the next_id counter from the collection's current vector count.
    fn sync_next_id(&self) {
        let url = format!("{}/api/collections/{}/stats", self.base_url, COLLECTION_NAME);
        match self.agent.get(&url).call() {
            Ok(resp) => {
                let status: u16 = resp.status().into();
                if status < 400 {
                    let body = resp.into_body().read_to_string().unwrap_or_default();
                    if let Some(count) = extract_json_u32(&body, "count") {
                        let next = count.saturating_add(1);
                        self.next_id.store(next, Ordering::Relaxed);
                        info!("[MEMORY] Synced ID counter: next_id = {next} (count = {count})");
                    }
                }
            }
            Err(_) => {} // best-effort
        }
    }

    /// Store a page visit in NietzscheDB.
    ///
    /// Metadata fields: url, title, summary, visited_at (ISO 8601 timestamp).
    /// Uses n-gram hash vectors for semantic similarity (lightweight, no model needed).
    pub fn store_page(&self, url: &str, title: &str, summary: &str, content: &str) {
        // Guard: skip useless entries where both title and content are empty/whitespace
        if title.trim().is_empty() && content.trim().is_empty() {
            warn!("[MEMORY] Skipping store_page — title and content are both empty");
            return;
        }

        // Generate a unique u32 ID for this visit.
        //
        // The previous scheme packed `epoch_secs << 16 | seq`, which threw away
        // all but the low 16 bits of the timestamp — those wrap every 65536
        // seconds, so IDs collided (and silently overwrote history) roughly
        // every 18 hours of use.
        //
        // Instead: hash the URL together with the full-resolution timestamp and
        // a per-process counter. Every component contributes to all 32 bits, so
        // the only collisions left are ordinary birthday collisions rather than
        // a guaranteed clash on a fixed cycle.
        let seq = self.next_id.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let mut hasher_input = Vec::with_capacity(url.len() + 16);
        hasher_input.extend_from_slice(url.as_bytes());
        hasher_input.extend_from_slice(&nanos.to_le_bytes());
        hasher_input.extend_from_slice(&seq.to_le_bytes());
        // Fold the 64-bit hash into 32 bits; avoid 0, which the server treats
        // as "unset" in some code paths.
        let h = fnv1a(&hasher_input);
        let id = (((h >> 32) as u32) ^ (h as u32)).max(1);
        let visited_at = timestamp_now();

        // Build semantic vector from page content (n-gram hashing into Poincaré ball)
        let text = format!("{} {} {}", title, summary, content);
        let vec_json = text_to_vector_json(&text, DIMENSION);

        let body = format!(
            r#"{{"vector":{},"id":{},"metadata":{{"url":"{}","title":"{}","summary":"{}","visited_at":"{}"}}}}"#,
            vec_json,
            id,
            escape_json(url),
            escape_json(title),
            escape_json(summary),
            escape_json(&visited_at),
        );

        let endpoint = format!(
            "{}/api/collections/{}/insert",
            self.base_url, COLLECTION_NAME
        );

        match self.agent.post(&endpoint)
            .header("Content-Type", "application/json")
            .send(body.as_bytes())
        {
            Ok(resp) => {
                let status: u16 = resp.status().into();
                if status < 300 {
                    info!("[MEMORY] Stored page: {} (id={})", truncate(url, 60), id);
                } else {
                    let err = resp.into_body().read_to_string().unwrap_or_default();
                    warn!("[MEMORY] Failed to store page (HTTP {status}): {err}");
                }
            }
            Err(e) => {
                warn!("[MEMORY] Failed to store page: {e}");
            }
        }
    }

    /// KNN search for semantically similar pages.
    ///
    /// Uses n-gram hash vectors for lightweight semantic similarity.
    /// Not as good as real embeddings, but works entirely on CPU with zero latency.
    pub fn search_semantic(&self, query: &str, k: usize) -> Vec<PageMemory> {
        let query_vec = text_to_vector_json(query, DIMENSION);

        let body = format!(
            r#"{{"vector":{},"top_k":{}}}"#,
            query_vec, k,
        );

        let endpoint = format!(
            "{}/api/collections/{}/search",
            self.base_url, COLLECTION_NAME
        );

        let resp = match self.agent.post(&endpoint)
            .header("Content-Type", "application/json")
            .send(body.as_bytes())
        {
            Ok(r) => r,
            Err(e) => {
                warn!("[MEMORY] Search failed: {e}");
                return Vec::new();
            }
        };

        let status: u16 = resp.status().into();
        if status >= 400 {
            warn!("[MEMORY] Search returned HTTP {status}");
            return Vec::new();
        }

        let body_str = resp.into_body()
            .read_to_string()
            .unwrap_or_default();

        parse_search_results(&body_str)
    }

    /// Return recently stored pages by peeking the collection.
    ///
    /// Uses the peek endpoint which returns the most recent vectors with metadata.
    pub fn recent_pages(&self, limit: usize) -> Vec<PageMemory> {
        let endpoint = format!(
            "{}/api/collections/{}/peek?limit={}",
            self.base_url, COLLECTION_NAME, limit,
        );

        let resp = match self.agent.get(&endpoint).call() {
            Ok(r) => r,
            Err(e) => {
                warn!("[MEMORY] Recent pages fetch failed: {e}");
                return Vec::new();
            }
        };

        let status: u16 = resp.status().into();
        if status >= 400 {
            warn!("[MEMORY] Recent pages returned HTTP {status}");
            return Vec::new();
        }

        let body_str = resp.into_body()
            .read_to_string()
            .unwrap_or_default();

        parse_peek_results(&body_str)
    }
}

// ── JSON helpers (no serde, matching project style) ──

/// Escape a string for safe JSON embedding.
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

/// Generate a JSON array of N zeros: [0.0, 0.0, ...]
fn zero_vector_json(dim: usize) -> String {
    let mut s = String::with_capacity(dim * 4 + 2);
    s.push('[');
    for i in 0..dim {
        if i > 0 {
            s.push(',');
        }
        s.push_str("0.0");
    }
    s.push(']');
    s
}

/// Generate a semantic vector from text using n-gram hashing (lightweight, no model).
/// Uses FNV-1a hash of unigrams, bigrams, and character trigrams into fixed buckets.
/// Normalized to Poincaré ball (magnitude < 1.0) for NietzscheDB's hyperbolic metric.
fn text_to_vector_json(text: &str, dim: usize) -> String {
    let mut vector = vec![0.0f32; dim];

    if !text.is_empty() {
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();

        // Unigram hashing
        for word in &words {
            let bucket = fnv1a(word.as_bytes()) as usize % dim;
            vector[bucket] += 1.0;
        }

        // Bigram hashing (word pairs capture context)
        for pair in words.windows(2) {
            let combined = format!("{} {}", pair[0], pair[1]);
            let bucket = fnv1a(combined.as_bytes()) as usize % dim;
            vector[bucket] += 0.5;
        }

        // Character trigram hashing (captures subword patterns)
        let chars: Vec<char> = text_lower.chars().collect();
        for trigram in chars.windows(3) {
            let s: String = trigram.iter().collect();
            let bucket = fnv1a(s.as_bytes()) as usize % dim;
            vector[bucket] += 0.2;
        }

        // Normalize to Poincaré ball (magnitude = 0.5, safely inside unit ball)
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            let scale = 0.5 / norm;
            for v in &mut vector {
                *v *= scale;
            }
        }
    }

    // Format as JSON array
    let mut s = String::with_capacity(dim * 10 + 2);
    s.push('[');
    for (i, v) in vector.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{:.6}", v));
    }
    s.push(']');
    s
}

/// FNV-1a hash — deterministic, fast, no randomization.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Extract a u32 field from JSON (simple parser).
fn extract_json_u32(json: &str, field: &str) -> Option<u32> {
    let pattern = format!("\"{}\"", field);
    let pos = json.find(&pattern)?;
    let rest = &json[pos + pattern.len()..];
    // Skip whitespace and colon
    let rest = rest.trim_start();
    let rest = rest.strip_prefix(':')?;
    let rest = rest.trim_start();
    // Parse number
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Extract a string field value from a JSON object.
///
/// Scans the object structurally (rather than searching for `"field"` anywhere
/// in the text) so a key name appearing inside some other string value cannot
/// be mistaken for the field itself.
fn extract_json_string(json: &str, field: &str) -> Option<String> {
    let bytes = json.as_bytes();
    let mut i = skip_ws(bytes, 0);
    if bytes.get(i) != Some(&b'{') {
        return None;
    }
    i += 1;
    loop {
        i = skip_ws(bytes, i);
        match bytes.get(i) {
            Some(b'}') | None => return None,
            Some(b',') => { i += 1; continue; }
            Some(b'"') => {}
            _ => return None,
        }
        let (key, next) = scan_json_string(bytes, i)?;
        i = skip_ws(bytes, next);
        if bytes.get(i) != Some(&b':') {
            return None;
        }
        i = skip_ws(bytes, i + 1);
        if key == field {
            // Only string values are returned; a non-string field yields None.
            if bytes.get(i) == Some(&b'"') {
                return scan_json_string(bytes, i).map(|(v, _)| v);
            }
            return None;
        }
        i = skip_json_value(bytes, i)?;
    }
}

/// Skip ASCII whitespace starting at `i`.
fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() {
        i += 1;
    }
    i
}

/// Read a JSON string starting at the opening quote.
/// Returns the decoded value and the index just past the closing quote.
fn scan_json_string(bytes: &[u8], start: usize) -> Option<(String, usize)> {
    if bytes.get(start) != Some(&b'"') {
        return None;
    }
    let mut out = String::new();
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Some((out, i + 1)),
            b'\\' => {
                i += 1;
                match bytes.get(i)? {
                    b'n' => { out.push('\n'); i += 1; }
                    b'r' => { out.push('\r'); i += 1; }
                    b't' => { out.push('\t'); i += 1; }
                    b'b' => { out.push('\u{0008}'); i += 1; }
                    b'f' => { out.push('\u{000C}'); i += 1; }
                    b'"' => { out.push('"'); i += 1; }
                    b'\\' => { out.push('\\'); i += 1; }
                    b'/' => { out.push('/'); i += 1; }
                    b'u' => {
                        // \uXXXX, including surrogate pairs for astral chars.
                        let hi = parse_hex4(bytes, i + 1)?;
                        i += 5;
                        let ch = if (0xD800..0xDC00).contains(&hi) {
                            // High surrogate: a matching low surrogate must follow.
                            if bytes.get(i) == Some(&b'\\') && bytes.get(i + 1) == Some(&b'u') {
                                let lo = parse_hex4(bytes, i + 2)?;
                                if (0xDC00..0xE000).contains(&lo) {
                                    i += 6;
                                    let cp = 0x10000
                                        + (((hi as u32) - 0xD800) << 10)
                                        + ((lo as u32) - 0xDC00);
                                    char::from_u32(cp).unwrap_or('\u{FFFD}')
                                } else {
                                    '\u{FFFD}'
                                }
                            } else {
                                '\u{FFFD}'
                            }
                        } else if (0xDC00..0xE000).contains(&hi) {
                            // Unpaired low surrogate.
                            '\u{FFFD}'
                        } else {
                            char::from_u32(hi as u32).unwrap_or('\u{FFFD}')
                        };
                        out.push(ch);
                    }
                    other => { out.push(*other as char); i += 1; }
                }
            }
            _ => {
                // Copy one whole UTF-8 sequence so multi-byte characters survive.
                let len = utf8_len(bytes[i]);
                let end = (i + len).min(bytes.len());
                out.push_str(&String::from_utf8_lossy(&bytes[i..end]));
                i = end;
            }
        }
    }
    None
}

/// Byte length of the UTF-8 sequence that starts with `b`.
fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else if b >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

/// Parse exactly four hex digits at `i`.
fn parse_hex4(bytes: &[u8], i: usize) -> Option<u16> {
    let slice = bytes.get(i..i + 4)?;
    let text = std::str::from_utf8(slice).ok()?;
    u16::from_str_radix(text, 16).ok()
}

/// Skip one complete JSON value (object, array, string, number, literal),
/// returning the index just past it. Nested structures are matched properly,
/// and braces inside strings do not end an object.
fn skip_json_value(bytes: &[u8], start: usize) -> Option<usize> {
    let mut i = skip_ws(bytes, start);
    match *bytes.get(i)? {
        b'"' => scan_json_string(bytes, i).map(|(_, next)| next),
        b'{' | b'[' => {
            let mut depth = 0usize;
            while i < bytes.len() {
                match bytes[i] {
                    b'"' => {
                        let (_, next) = scan_json_string(bytes, i)?;
                        i = next;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            None
        }
        _ => {
            // Number, true, false or null: read until a structural character.
            while i < bytes.len() && !matches!(bytes[i], b',' | b'}' | b']') {
                i += 1;
            }
            Some(i)
        }
    }
}

/// Parse search results: [{"id": N, "distance": D, "metadata": {"url": "...", ...}}, ...]
fn parse_search_results(json: &str) -> Vec<PageMemory> {
    parse_items(json)
}

/// Parse peek results: similar format with metadata objects.
fn parse_peek_results(json: &str) -> Vec<PageMemory> {
    parse_items(json)
}

/// Parse an array of objects that contain metadata with url/title/summary/visited_at.
/// Works for both search and peek responses.
///
/// Each `"metadata"` object is delimited by real brace matching, so nested
/// objects and values containing `}` no longer truncate the entry.
fn parse_items(json: &str) -> Vec<PageMemory> {
    let bytes = json.as_bytes();
    let mut results = Vec::new();
    let mut i = 0usize;

    while i < bytes.len() {
        // Find the next `"metadata"` key. Strings are scanned through, so a
        // literal "metadata" inside some other value is skipped.
        if bytes[i] == b'"' {
            let (key, next) = match scan_json_string(bytes, i) {
                Some(v) => v,
                None => break,
            };
            if key == "metadata" {
                let after = skip_ws(bytes, next);
                if bytes.get(after) == Some(&b':') {
                    let val_start = skip_ws(bytes, after + 1);
                    if bytes.get(val_start) == Some(&b'{') {
                        if let Some(val_end) = skip_json_value(bytes, val_start) {
                            let meta = &json[val_start..val_end];
                            let url = extract_json_string(meta, "url").unwrap_or_default();
                            if !url.is_empty() {
                                results.push(PageMemory {
                                    url,
                                    title: extract_json_string(meta, "title").unwrap_or_default(),
                                    summary: extract_json_string(meta, "summary").unwrap_or_default(),
                                    visited_at: extract_json_string(meta, "visited_at").unwrap_or_default(),
                                });
                            }
                            i = val_end;
                            continue;
                        }
                    }
                }
            }
            i = next;
            continue;
        }
        i += 1;
    }

    results
}

/// Current timestamp as ISO 8601 string (UTC, manual — no chrono dep).
fn timestamp_now() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();

    // Convert epoch seconds to ISO 8601 (simplified, good enough for metadata)
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let mins = (time_secs % 3600) / 60;
    let s = time_secs % 60;

    // Days since 1970-01-01 to year/month/day (simplified leap year handling)
    let (year, month, day) = days_to_ymd(days);

    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{mins:02}:{s:02}Z")
}

/// Convert days since epoch to (year, month, day). Handles leap years.
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    days += 719468;
    let era = days / 146097;
    let doe = days - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Truncate a string for display (UTF-8 safe).
pub(crate) fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
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
        assert_eq!(escape_json("hello \"world\""), r#"hello \"world\""#);
        assert_eq!(escape_json("line\nnew"), r#"line\nnew"#);
    }

    #[test]
    fn test_zero_vector_json() {
        let v = zero_vector_json(3);
        assert_eq!(v, "[0.0,0.0,0.0]");
    }

    #[test]
    fn test_extract_json_u32() {
        assert_eq!(extract_json_u32(r#"{"count":42,"dim":128}"#, "count"), Some(42));
        assert_eq!(extract_json_u32(r#"{"count":42,"dim":128}"#, "dim"), Some(128));
        assert_eq!(extract_json_u32(r#"{"count":42}"#, "missing"), None);
    }

    #[test]
    fn test_extract_json_string() {
        let json = r#"{"url":"https://example.com","title":"Test Page"}"#;
        assert_eq!(extract_json_string(json, "url"), Some("https://example.com".into()));
        assert_eq!(extract_json_string(json, "title"), Some("Test Page".into()));
        assert_eq!(extract_json_string(json, "missing"), None);
    }

    #[test]
    fn test_parse_search_results() {
        let json = r#"[{"id":1,"distance":0.5,"metadata":{"url":"https://a.com","title":"A","summary":"Page A","visited_at":"2026-03-13T10:00:00Z"}},{"id":2,"distance":0.7,"metadata":{"url":"https://b.com","title":"B","summary":"Page B","visited_at":"2026-03-13T11:00:00Z"}}]"#;
        let results = parse_search_results(json);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].url, "https://a.com");
        assert_eq!(results[0].title, "A");
        assert_eq!(results[1].url, "https://b.com");
        assert_eq!(results[1].summary, "Page B");
    }

    #[test]
    fn test_parse_empty() {
        assert!(parse_search_results("[]").is_empty());
        assert!(parse_search_results("").is_empty());
    }

    #[test]
    fn test_timestamp_format() {
        let ts = timestamp_now();
        // Should look like 2026-03-13T12:34:56Z
        assert!(ts.len() == 20, "timestamp should be 20 chars: {ts}");
        assert!(ts.ends_with('Z'));
        assert!(ts.contains('T'));
    }

    #[test]
    fn test_days_to_ymd_epoch() {
        // 1970-01-01
        let (y, m, d) = days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn test_days_to_ymd_known_date() {
        // 2026-03-13 = day 20525 since epoch
        let (y, m, d) = days_to_ymd(20525);
        assert_eq!((y, m, d), (2026, 3, 13));
    }

    #[test]
    fn test_text_to_vector_json_dimension() {
        let json = text_to_vector_json("hello world test", 128);
        // Count commas + 1 = number of elements
        let count = json.matches(',').count() + 1;
        assert_eq!(count, 128);
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
    }

    #[test]
    fn test_text_to_vector_poincare_ball() {
        let json = text_to_vector_json("the quick brown fox jumps over the lazy dog", 128);
        // Parse values and check magnitude < 1.0
        let inner = &json[1..json.len()-1]; // strip []
        let values: Vec<f32> = inner.split(',').filter_map(|s| s.parse().ok()).collect();
        let magnitude: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(magnitude < 1.0, "magnitude={magnitude} should be < 1.0 (Poincaré ball)");
        assert!(magnitude > 0.0, "magnitude should be > 0.0 for non-empty text");
    }

    #[test]
    fn test_text_to_vector_empty() {
        let json = text_to_vector_json("", 128);
        // All zeros
        assert!(json.contains("0.000000"));
        // Parse and check
        let inner = &json[1..json.len()-1];
        let values: Vec<f32> = inner.split(',').filter_map(|s| s.parse().ok()).collect();
        assert!(values.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_text_to_vector_deterministic() {
        let v1 = text_to_vector_json("rust programming", 128);
        let v2 = text_to_vector_json("rust programming", 128);
        assert_eq!(v1, v2, "same input should produce same vector");
    }

    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a(b"hello"), fnv1a(b"hello"));
        assert_ne!(fnv1a(b"hello"), fnv1a(b"world"));
    }

    #[test]
    fn json_string_handles_unicode_escapes() {
        let json = r#"{"title":"café — done"}"#;
        assert_eq!(extract_json_string(json, "title").as_deref(), Some("café — done"));
    }

    #[test]
    fn json_string_handles_surrogate_pairs() {
        // U+1F600 GRINNING FACE, encoded as a surrogate pair.
        let json = r#"{"title":"hi 😀"}"#;
        assert_eq!(extract_json_string(json, "title").as_deref(), Some("hi \u{1F600}"));
    }

    #[test]
    fn json_string_handles_escaped_quotes_and_slashes() {
        let json = r#"{"title":"a \"quoted\" \/ path\\end"}"#;
        assert_eq!(
            extract_json_string(json, "title").as_deref(),
            Some("a \"quoted\" / path\\end")
        );
    }

    #[test]
    fn json_key_inside_a_value_is_not_mistaken_for_the_field() {
        // The literal "url" appears inside the title's value first.
        let json = r#"{"title":"how to \"url\" things","url":"https://real.example"}"#;
        assert_eq!(
            extract_json_string(json, "url").as_deref(),
            Some("https://real.example")
        );
    }

    #[test]
    fn metadata_with_nested_objects_is_not_truncated() {
        let json = r#"[{"id":1,"metadata":{"url":"https://a.test","extra":{"depth":2},"title":"A","summary":"s","visited_at":"2026-01-01T00:00:00Z"}}]"#;
        let items = parse_items(json);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].url, "https://a.test");
        // Fields after the nested object used to be lost entirely.
        assert_eq!(items[0].title, "A");
        assert_eq!(items[0].visited_at, "2026-01-01T00:00:00Z");
    }

    #[test]
    fn brace_inside_a_value_does_not_end_the_object() {
        let json = r#"[{"metadata":{"url":"https://a.test/?q={x}","title":"} not the end","summary":"","visited_at":""}}]"#;
        let items = parse_items(json);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].url, "https://a.test/?q={x}");
        assert_eq!(items[0].title, "} not the end");
    }

    #[test]
    fn multiple_results_are_all_parsed() {
        let json = r#"[
            {"id":1,"distance":0.1,"metadata":{"url":"https://a.test","title":"A","summary":"","visited_at":""}},
            {"id":2,"distance":0.2,"metadata":{"url":"https://b.test","title":"B","summary":"","visited_at":""}}
        ]"#;
        let items = parse_items(json);
        assert_eq!(items.len(), 2);
        assert_eq!(items[1].url, "https://b.test");
    }

    #[test]
    fn json_escapes_survive_a_round_trip() {
        let original = "line1\nline2 \"quoted\" \\ backslash";
        let json = format!(r#"{{"title":"{}"}}"#, escape_json(original));
        assert_eq!(extract_json_string(&json, "title").as_deref(), Some(original));
    }
}
