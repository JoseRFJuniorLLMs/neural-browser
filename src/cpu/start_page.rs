//! Built-in start page shown when the browser first opens.

/// The internal URL for the start page.
pub const START_PAGE_URL: &str = "neural://start";

/// Returns the HTML for the built-in welcome/start page.
pub fn start_page_html() -> &'static str {
    r#"<!DOCTYPE html>
<html>
<head>
    <title>Neural Browser</title>
</head>
<body>
<main>
    <h1>Neural Browser</h1>
    <p>A browser with three brains: CPU + NPU + GPU working together.</p>

    <h2>Architecture</h2>
    <ul>
        <li>CPU — Networking, HTML parsing, DOM tree construction</li>
        <li>NPU — AI content understanding, ad blocking, language detection (ONNX + DirectML)</li>
        <li>GPU — Layout, text rasterization, compositing (wgpu + glyphon)</li>
    </ul>

    <h2>Keyboard Shortcuts</h2>
    <ul>
        <li>F6 — Focus the URL bar (type a URL and press Enter)</li>
        <li>Enter — Navigate to the typed URL</li>
        <li>Escape — Cancel URL editing</li>
        <li>F5 — Refresh the current page</li>
        <li>Page Up / Page Down — Scroll by page</li>
        <li>Home — Scroll to top</li>
    </ul>

    <h2>Try These Pages</h2>
    <ul>
        <li><a href="https://example.com">example.com</a> — Simple test page</li>
        <li><a href="https://en.wikipedia.org/wiki/Web_browser">Wikipedia: Web Browser</a> — Content-heavy article</li>
        <li><a href="https://news.ycombinator.com">Hacker News</a> — Link-heavy page</li>
        <li><a href="https://rust-lang.org">rust-lang.org</a> — The Rust programming language</li>
    </ul>

    <h2>How It Works</h2>
    <p>When you navigate to a URL, the pipeline flows like this:</p>
    <ol>
        <li>CPU fetches the page over HTTPS (ureq + rustls)</li>
        <li>CPU parses HTML into a lightweight DOM tree</li>
        <li>NPU extracts semantic content blocks (headings, paragraphs, images, code)</li>
        <li>NPU classifies and blocks ads and trackers using ML heuristics</li>
        <li>NPU detects the page language</li>
        <li>GPU computes layout and renders text via glyphon</li>
    </ol>

    <p>Press F6 to enter a URL and start browsing.</p>
</main>
</body>
</html>"#
}
