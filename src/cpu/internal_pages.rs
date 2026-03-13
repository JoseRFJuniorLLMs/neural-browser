//! Built-in internal pages served via the `neural://` protocol.
//!
//! These pages provide browser-internal UI (settings, history, about, etc.)
//! and are rendered through the normal NPU pipeline just like any web page.

/// Generate HTML for a `neural://` URL.
///
/// Returns `Some(html)` for recognized internal pages, `None` otherwise.
pub fn generate_internal_page(url: &str, visited_urls: &[String]) -> Option<String> {
    let path = url.strip_prefix("neural://")?;

    // Strip any query string or fragment for matching
    let page = path.split('?').next().unwrap_or(path);
    let page = page.split('#').next().unwrap_or(page);

    match page {
        "settings" => Some(page_settings()),
        "history" => Some(page_history(visited_urls)),
        "about" => Some(page_about()),
        "files" => Some(page_files()),
        "start" => None, // handled by start_page.rs
        _ => Some(page_not_found(url)),
    }
}

// ── Shared CSS ──────────────────────────────────────────────────────────

const INTERNAL_STYLE: &str = r#"
    body {
        font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
        background: #1a1a2e;
        color: #e0e0e0;
        margin: 0;
        padding: 0;
    }
    main {
        max-width: 800px;
        margin: 0 auto;
        padding: 40px 20px;
    }
    h1 {
        color: #64ffda;
        border-bottom: 2px solid #333;
        padding-bottom: 10px;
    }
    h2 {
        color: #82b1ff;
        margin-top: 30px;
    }
    a {
        color: #64ffda;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    ul {
        list-style: none;
        padding-left: 0;
    }
    li {
        padding: 6px 0;
        border-bottom: 1px solid #2a2a3e;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        margin-left: 8px;
    }
    .badge-ok { background: #1b5e20; color: #a5d6a7; }
    .badge-off { background: #4a1a1a; color: #ef9a9a; }
    .badge-info { background: #1a237e; color: #9fa8da; }
    .section {
        background: #16213e;
        border-radius: 8px;
        padding: 20px;
        margin: 16px 0;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    td, th {
        text-align: left;
        padding: 8px 12px;
        border-bottom: 1px solid #2a2a3e;
    }
    th {
        color: #82b1ff;
        font-weight: 600;
    }
    .placeholder {
        text-align: center;
        padding: 60px 20px;
        color: #888;
    }
    .nav-links {
        margin-bottom: 20px;
        padding: 10px 0;
        border-bottom: 1px solid #2a2a3e;
    }
    .nav-links a {
        margin-right: 20px;
        color: #82b1ff;
    }
"#;

fn nav_bar() -> &'static str {
    r#"<div class="nav-links">
        <a href="neural://start">Home</a>
        <a href="neural://settings">Settings</a>
        <a href="neural://history">History</a>
        <a href="neural://about">About</a>
    </div>"#
}

fn wrap_page(title: &str, body: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>{title} - Neural Browser</title>
    <style>{INTERNAL_STYLE}</style>
</head>
<body>
<main>
{nav}
{body}
</main>
</body>
</html>"#,
        title = title,
        nav = nav_bar(),
        body = body,
    )
}

// ── Settings Page ───────────────────────────────────────────────────────

fn page_settings() -> String {
    // Check which AI providers have API keys set
    let openai_status = if std::env::var("OPENAI_API_KEY").is_ok() {
        r#"<span class="badge badge-ok">configured</span>"#
    } else {
        r#"<span class="badge badge-off">not set</span>"#
    };

    let anthropic_status = if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        r#"<span class="badge badge-ok">configured</span>"#
    } else {
        r#"<span class="badge badge-off">not set</span>"#
    };

    let gemini_status = if std::env::var("GEMINI_API_KEY").is_ok()
        || std::env::var("GOOGLE_API_KEY").is_ok()
    {
        r#"<span class="badge badge-ok">configured</span>"#
    } else {
        r#"<span class="badge badge-off">not set</span>"#
    };

    let body = format!(
        r#"<h1>Settings</h1>

<div class="section">
<h2>AI Providers</h2>
<table>
    <tr><th>Provider</th><th>Status</th></tr>
    <tr><td>OpenAI (EVA backend)</td><td>{openai_status}</td></tr>
    <tr><td>Anthropic (Claude)</td><td>{anthropic_status}</td></tr>
    <tr><td>Google Gemini</td><td>{gemini_status}</td></tr>
</table>
<p>Set API keys via environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY</p>
</div>

<div class="section">
<h2>NPU Engine</h2>
<table>
    <tr><th>Setting</th><th>Value</th></tr>
    <tr><td>Runtime</td><td>ONNX Runtime + DirectML</td></tr>
    <tr><td>Ad Blocker</td><td>ML heuristic classifier<span class="badge badge-ok">active</span></td></tr>
    <tr><td>Language Detection</td><td>Statistical + Unicode analysis<span class="badge badge-ok">active</span></td></tr>
</table>
</div>

<div class="section">
<h2>GPU Renderer</h2>
<table>
    <tr><th>Setting</th><th>Value</th></tr>
    <tr><td>Backend</td><td>wgpu (auto-detect: Vulkan / DX12 / Metal)</td></tr>
    <tr><td>Text Engine</td><td>glyphon + cosmic-text</td></tr>
    <tr><td>Theme</td><td>Dark <span class="badge badge-info">only option for now</span></td></tr>
</table>
</div>

<div class="section">
<h2>Keyboard Shortcuts</h2>
<table>
    <tr><th>Key</th><th>Action</th></tr>
    <tr><td>F6 / Ctrl+L</td><td>Focus URL bar</td></tr>
    <tr><td>Enter</td><td>Navigate to URL or search</td></tr>
    <tr><td>Escape</td><td>Cancel URL editing</td></tr>
    <tr><td>F5 / Ctrl+R</td><td>Refresh page</td></tr>
    <tr><td>Alt+Left / Alt+Right</td><td>Back / Forward</td></tr>
    <tr><td>Page Up / Page Down</td><td>Scroll by page</td></tr>
    <tr><td>Home / End</td><td>Scroll to top / bottom</td></tr>
    <tr><td>Ctrl+E / F2</td><td>Toggle EVA AI panel</td></tr>
    <tr><td>Ctrl+S</td><td>Ask EVA to summarize page</td></tr>
</table>
</div>"#,
    );

    wrap_page("Settings", &body)
}

// ── History Page ────────────────────────────────────────────────────────

fn page_history(visited_urls: &[String]) -> String {
    let list = if visited_urls.is_empty() {
        "<p>No pages visited yet. Start browsing to build your history.</p>".to_string()
    } else {
        let mut items = String::new();
        for url in visited_urls.iter().rev() {
            let safe_url = html_escape(url);
            items.push_str(&format!(
                r#"<li><a href="{safe_url}">{safe_url}</a></li>"#
            ));
        }
        format!("<ul>{items}</ul>")
    };

    let count = visited_urls.len();
    let body = format!(
        r#"<h1>Browsing History</h1>
<p>{count} page(s) visited this session.</p>
<div class="section">
{list}
</div>"#
    );

    wrap_page("History", &body)
}

// ── About Page ──────────────────────────────────────────────────────────

fn page_about() -> String {
    let body = r#"<h1>About Neural Browser</h1>

<div class="section">
<h2>Version</h2>
<table>
    <tr><th>Component</th><th>Version</th></tr>
    <tr><td>Neural Browser</td><td>v0.1.0</td></tr>
    <tr><td>Architecture</td><td>CPU + NPU + GPU (three-brain pipeline)</td></tr>
    <tr><td>License</td><td>MIT</td></tr>
</table>
</div>

<div class="section">
<h2>Architecture</h2>
<p>Neural Browser uses a three-processor pipeline inspired by how the human brain processes visual information:</p>
<ul>
    <li><strong>CPU</strong> — Networking, HTML parsing, DOM tree construction, browser history</li>
    <li><strong>NPU</strong> — AI content understanding, ad blocking, language detection (ONNX Runtime + DirectML)</li>
    <li><strong>GPU</strong> — Layout computation, text rasterization, compositing (wgpu + glyphon)</li>
</ul>
</div>

<div class="section">
<h2>Technology</h2>
<table>
    <tr><th>Layer</th><th>Technology</th></tr>
    <tr><td>Language</td><td>Rust</td></tr>
    <tr><td>Networking</td><td>ureq + rustls</td></tr>
    <tr><td>HTML Parsing</td><td>html5ever</td></tr>
    <tr><td>AI Runtime</td><td>ONNX Runtime + DirectML</td></tr>
    <tr><td>Rendering</td><td>wgpu</td></tr>
    <tr><td>Text</td><td>glyphon + cosmic-text</td></tr>
    <tr><td>Windowing</td><td>winit</td></tr>
</table>
</div>

<div class="section">
<h2>Credits</h2>
<p>Built with Rust and open-source libraries. Powered by the wgpu graphics API and ONNX Runtime for neural inference.</p>
</div>"#;

    wrap_page("About", body)
}

// ── Files Page (Placeholder) ────────────────────────────────────────────

fn page_files() -> String {
    let body = r#"<h1>Local Files</h1>
<div class="section placeholder">
    <h2>File browser coming soon</h2>
    <p>This feature will allow you to browse and open local HTML files.</p>
</div>"#;

    wrap_page("Files", body)
}

// ── 404 Page ────────────────────────────────────────────────────────────

fn page_not_found(url: &str) -> String {
    let safe_url = html_escape(url);
    let body = format!(
        r#"<h1>Page Not Found</h1>
<div class="section">
    <p>The internal page <code>{safe_url}</code> does not exist.</p>
    <p>Try one of these:</p>
    <ul>
        <li><a href="neural://start">neural://start</a> — Start page</li>
        <li><a href="neural://settings">neural://settings</a> — Settings</li>
        <li><a href="neural://history">neural://history</a> — History</li>
        <li><a href="neural://about">neural://about</a> — About</li>
    </ul>
</div>"#,
    );

    wrap_page("Not Found", &body)
}

// ── Utility ─────────────────────────────────────────────────────────────

/// Escape HTML special characters to prevent XSS.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_page_generates_html() {
        let html = generate_internal_page("neural://settings", &[]);
        assert!(html.is_some());
        let html = html.unwrap();
        assert!(html.contains("<title>Settings - Neural Browser</title>"));
        assert!(html.contains("AI Providers"));
        assert!(html.contains("Keyboard Shortcuts"));
    }

    #[test]
    fn test_history_page_empty() {
        let html = generate_internal_page("neural://history", &[]).unwrap();
        assert!(html.contains("No pages visited yet"));
        assert!(html.contains("0 page(s) visited"));
    }

    #[test]
    fn test_history_page_with_urls() {
        let urls = vec![
            "https://example.com".to_string(),
            "https://rust-lang.org".to_string(),
        ];
        let html = generate_internal_page("neural://history", &urls).unwrap();
        assert!(html.contains("example.com"));
        assert!(html.contains("rust-lang.org"));
        assert!(html.contains("2 page(s) visited"));
    }

    #[test]
    fn test_about_page() {
        let html = generate_internal_page("neural://about", &[]).unwrap();
        assert!(html.contains("v0.1.0"));
        assert!(html.contains("CPU + NPU + GPU"));
        assert!(html.contains("MIT"));
    }

    #[test]
    fn test_files_placeholder() {
        let html = generate_internal_page("neural://files", &[]).unwrap();
        assert!(html.contains("File browser coming soon"));
    }

    #[test]
    fn test_start_returns_none() {
        // neural://start is handled by start_page.rs, not here
        let result = generate_internal_page("neural://start", &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_unknown_page_shows_404() {
        let html = generate_internal_page("neural://nonexistent", &[]).unwrap();
        assert!(html.contains("Page Not Found"));
        assert!(html.contains("neural://nonexistent"));
    }

    #[test]
    fn test_not_neural_protocol_returns_none() {
        let result = generate_internal_page("https://example.com", &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_history_escapes_html() {
        let urls = vec!["https://example.com/<script>".to_string()];
        let html = generate_internal_page("neural://history", &urls).unwrap();
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn test_all_pages_are_valid_html() {
        let pages = ["neural://settings", "neural://history", "neural://about", "neural://files"];
        for page in &pages {
            let html = generate_internal_page(page, &[]).unwrap();
            assert!(html.contains("<!DOCTYPE html>"), "Missing DOCTYPE in {page}");
            assert!(html.contains("<html>"), "Missing <html> in {page}");
            assert!(html.contains("</html>"), "Missing </html> in {page}");
            assert!(html.contains("<head>"), "Missing <head> in {page}");
            assert!(html.contains("</body>"), "Missing </body> in {page}");
        }
    }
}
