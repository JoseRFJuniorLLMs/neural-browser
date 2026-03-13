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
        <li>F6 / Ctrl+L — Focus the URL bar (type a URL or search query)</li>
        <li>Enter — Navigate to URL or search Google</li>
        <li>Escape — Cancel URL editing</li>
        <li>F5 / Ctrl+R — Refresh the current page</li>
        <li>Alt+Left / Alt+Right — Back / Forward</li>
        <li>Page Up / Page Down / Space — Scroll by page (Shift+Space = up)</li>
        <li>Arrow Up / Arrow Down — Scroll by line</li>
        <li>Home / End — Scroll to top / bottom</li>
        <li>Ctrl+C — Copy URL to clipboard</li>
        <li>Ctrl+V — Paste from clipboard</li>
        <li>Ctrl+W — Close window</li>
        <li>Ctrl+E / F2 — Toggle AI assistant panel</li>
        <li>Tab (in panel) — Switch AI provider (EVA → Claude → Gemini → GPT-4)</li>
        <li>Ctrl+S — Ask AI to summarize current page</li>
        <li>Ctrl+D — Reading mode (high-relevance content only + AI summary)</li>
        <li>Ctrl+T — Translate page to Portuguese</li>
        <li>Ctrl+Plus / Ctrl+Minus — Zoom in / out (30% to 300%)</li>
        <li>Ctrl+0 — Reset zoom to 100%</li>
        <li>Ctrl+Scroll — Zoom with mouse wheel</li>
        <li>F11 — Toggle fullscreen</li>
        <li>Ctrl+Shift+V (in panel) — Voice response via EVA</li>
    </ul>

    <h2>Internal Pages</h2>
    <ul>
        <li><a href="neural://settings">neural://settings</a> — Browser settings and AI provider status</li>
        <li><a href="neural://history">neural://history</a> — Browsing history this session</li>
        <li><a href="neural://about">neural://about</a> — About Neural Browser</li>
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

    <h2>AI Features</h2>
    <ul>
        <li><strong>Multi-AI</strong> — EVA, Claude, Gemini, GPT-4 (Tab to switch in panel)</li>
        <li><strong>Smart Search</strong> — Type a question in the URL bar, AI answers directly</li>
        <li><strong>Reading Mode</strong> — Ctrl+D strips noise, shows only high-relevance content</li>
        <li><strong>Auto-Translate</strong> — Ctrl+T translates page via active AI provider</li>
        <li><strong>Voice Response</strong> — Ctrl+Shift+V: EVA reads the last AI response aloud</li>
        <li><strong>Proactive Insights</strong> — When AI panel is open, auto-suggests questions about the page</li>
    </ul>

    <p>Press F6 to enter a URL and start browsing.</p>
</main>
</body>
</html>"#
}
