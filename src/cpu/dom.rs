//! CPU: Minimal DOM parser using a simple state machine.
//!
//! Produces a flat DomTree that CPU can cheaply build and NPU can analyze.
//! We do NOT build a full browser DOM -- that's wasted CPU work.
//! Instead, we extract just enough structure for the NPU to understand the page.
//!
//! Features:
//! - HTML entity decoding (&amp; &lt; &gt; &quot; &nbsp; &#NNN; &#xHHH;)
//! - Self-closing / void element handling
//! - Comment stripping (<!-- -->)
//! - Script/style content skipping
//! - Whitespace collapsing in text nodes

use std::collections::HashMap;

/// Lightweight DOM node -- just enough for NPU analysis.
#[derive(Debug, Clone)]
pub struct DomNode {
    pub id: usize,
    pub tag: String,
    pub attrs: HashMap<String, String>,
    pub text: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub depth: usize,
}

/// Flat DOM tree -- array of nodes, root is index 0.
#[derive(Debug, Clone)]
pub struct DomTree {
    pub nodes: Vec<DomNode>,
}

impl DomTree {
    pub fn text_content(&self) -> String {
        let mut out = String::new();
        for node in &self.nodes {
            if !node.text.is_empty() {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(&node.text);
            }
        }
        out
    }

    /// Extract all nodes with a specific tag.
    pub fn by_tag(&self, tag: &str) -> Vec<&DomNode> {
        self.nodes.iter().filter(|n| n.tag == tag).collect()
    }

    /// Get all links (href from <a> tags).
    pub fn links(&self) -> Vec<(&str, &str)> {
        self.nodes
            .iter()
            .filter(|n| n.tag == "a")
            .filter_map(|n| {
                n.attrs.get("href").map(|href| (href.as_str(), n.text.as_str()))
            })
            .collect()
    }

    /// Get all image sources.
    pub fn images(&self) -> Vec<&str> {
        self.nodes
            .iter()
            .filter(|n| n.tag == "img")
            .filter_map(|n| n.attrs.get("src").map(|s| s.as_str()))
            .collect()
    }
}

/// Tags whose content should be completely skipped (not added as text).
const RAW_TEXT_TAGS: &[&str] = &["script", "style"];

/// Void elements that never have closing tags.
const VOID_ELEMENTS: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img",
    "input", "link", "meta", "param", "source", "track", "wbr",
];

/// Maximum nesting depth to prevent stack-like overflow with deeply nested HTML.
const MAX_NESTING_DEPTH: usize = 256;

/// Maximum number of DOM nodes to prevent memory exhaustion on huge pages.
const MAX_DOM_NODES: usize = 100_000;

/// Parse raw HTML into a flat DomTree.
pub fn parse_html(html: &str) -> DomTree {
    let mut nodes = Vec::new();
    let mut stack: Vec<usize> = Vec::new(); // parent stack

    let bytes = html.as_bytes();
    let len = bytes.len();
    let mut pos = 0;
    let mut current_text = String::new();

    while pos < len {
        if bytes[pos] == b'<' {
            // Flush accumulated text to current parent
            flush_text(&mut current_text, &mut nodes, &stack);

            pos += 1;
            if pos >= len {
                break;
            }

            // ── Check for comment: <!-- ... --> ──
            if pos + 2 < len && &bytes[pos..pos + 3] == b"!--" {
                pos += 3;
                // Scan for -->
                while pos + 2 < len {
                    if bytes[pos] == b'-' && bytes[pos + 1] == b'-' && bytes[pos + 2] == b'>' {
                        pos += 3;
                        break;
                    }
                    pos += 1;
                }
                continue;
            }

            // ── Read the tag content up to '>' (UTF-8 safe) ──
            let mut tag_content = String::new();
            let mut in_quote = false;
            let mut quote_char = '"';
            while pos < len {
                // Decode the next UTF-8 character from the slice
                let ch = match html[pos..].chars().next() {
                    Some(c) => c,
                    None => break,
                };
                if !in_quote && ch == '>' {
                    pos += ch.len_utf8();
                    break;
                }
                if !in_quote && (ch == '"' || ch == '\'') {
                    in_quote = true;
                    quote_char = ch;
                } else if in_quote && ch == quote_char {
                    in_quote = false;
                }
                tag_content.push(ch);
                pos += ch.len_utf8();
            }

            let tag_content = tag_content.trim().to_string();

            // Skip doctype and processing instructions (but NOT comments, handled above)
            if tag_content.starts_with('!')
                || tag_content.starts_with('?')
            {
                continue;
            }

            if tag_content.starts_with('/') {
                // ── Closing tag ──
                let closing_tag = tag_content[1..].trim().to_lowercase();
                // Pop until we find the matching tag
                while let Some(parent_id) = stack.last() {
                    if nodes[*parent_id].tag == closing_tag {
                        stack.pop();
                        break;
                    } else {
                        stack.pop(); // Auto-close mismatched tags
                    }
                }
            } else {
                // ── Opening tag ──
                let self_closing = tag_content.ends_with('/');
                let content = if self_closing {
                    &tag_content[..tag_content.len() - 1]
                } else {
                    &tag_content
                };

                let mut parts = content.split_whitespace();
                let tag_name = parts
                    .next()
                    .unwrap_or("div")
                    .to_lowercase();
                let depth = stack.len();

                // Parse attributes
                let mut attrs = HashMap::new();
                let attr_str: String = parts.collect::<Vec<_>>().join(" ");
                parse_attrs(&attr_str, &mut attrs);

                let id = nodes.len();
                let parent = stack.last().copied();

                nodes.push(DomNode {
                    id,
                    tag: tag_name.clone(),
                    attrs,
                    text: String::new(),
                    parent,
                    children: Vec::new(),
                    depth,
                });

                if let Some(parent_id) = parent {
                    nodes[parent_id].children.push(id);
                }

                let is_void = VOID_ELEMENTS.contains(&tag_name.as_str());

                // If this is a raw-text element (script/style), skip all content
                // until the matching closing tag.
                if !is_void && RAW_TEXT_TAGS.contains(&tag_name.as_str()) {
                    let close_tag = format!("</{tag_name}");
                    if let Some(found) = find_case_insensitive(&html[pos..], &close_tag) {
                        pos += found;
                        // Skip past the '>'
                        while pos < len && bytes[pos] != b'>' {
                            pos += 1;
                        }
                        if pos < len {
                            pos += 1; // skip '>'
                        }
                    }
                    // Do NOT push to stack -- we already consumed the close tag
                    continue;
                }

                if !self_closing && !is_void && stack.len() < MAX_NESTING_DEPTH {
                    stack.push(id);
                }

                if nodes.len() >= MAX_DOM_NODES {
                    break;
                }
            }
        } else {
            // Accumulate text character by character (handle multi-byte UTF-8)
            let ch = html[pos..].chars().next().unwrap_or('\0');
            current_text.push(ch);
            pos += ch.len_utf8();
        }
    }

    // Flush any remaining text
    flush_text(&mut current_text, &mut nodes, &stack);

    DomTree { nodes }
}

/// Flush accumulated text, decode entities, collapse whitespace, attach to parent.
fn flush_text(
    current_text: &mut String,
    nodes: &mut Vec<DomNode>,
    stack: &[usize],
) {
    if current_text.is_empty() {
        return;
    }

    // Decode HTML entities first, then collapse whitespace
    let decoded = decode_entities(current_text);
    let collapsed = collapse_whitespace(&decoded);

    if !collapsed.is_empty() {
        if let Some(&parent_id) = stack.last() {
            if let Some(parent) = nodes.get_mut(parent_id) {
                if parent.text.is_empty() {
                    parent.text = collapsed;
                } else {
                    parent.text.push(' ');
                    parent.text.push_str(&collapsed);
                }
            }
        } else {
            // Text outside any tag -- create a text node
            let id = nodes.len();
            nodes.push(DomNode {
                id,
                tag: "#text".into(),
                attrs: HashMap::new(),
                text: collapsed,
                parent: None,
                children: Vec::new(),
                depth: 0,
            });
        }
    }

    current_text.clear();
}

/// Collapse runs of whitespace (spaces, tabs, newlines) into single spaces,
/// and trim leading/trailing whitespace.
fn collapse_whitespace(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut prev_ws = true; // treat start as whitespace to trim leading
    for ch in s.chars() {
        if ch.is_ascii_whitespace() {
            if !prev_ws {
                result.push(' ');
            }
            prev_ws = true;
        } else {
            result.push(ch);
            prev_ws = false;
        }
    }
    // Trim trailing space
    if result.ends_with(' ') {
        result.pop();
    }
    result
}

/// Decode common HTML entities: named (&amp; etc.), decimal (&#NNN;), hex (&#xHH;).
fn decode_entities(s: &str) -> String {
    // Fast path: no entities at all
    if !s.contains('&') {
        return s.to_string();
    }

    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '&' {
            let mut entity = String::new();
            let mut found_semi = false;
            // Collect up to 12 chars until ';' or limit
            for _ in 0..12 {
                match chars.peek() {
                    Some(&';') => {
                        chars.next(); // consume ';'
                        found_semi = true;
                        break;
                    }
                    Some(_) => {
                        entity.push(chars.next().unwrap());
                    }
                    None => break,
                }
            }
            if found_semi {
                if let Some(decoded) = resolve_entity(&entity) {
                    result.push(decoded);
                    continue;
                }
                // Not a known entity — output literally
                result.push('&');
                result.push_str(&entity);
                result.push(';');
            } else {
                result.push('&');
                result.push_str(&entity);
            }
            continue;
        }

        result.push(ch);
    }

    result
}

/// Resolve a single HTML entity name (without & and ;) to a char.
fn resolve_entity(entity: &str) -> Option<char> {
    // Numeric entities
    if let Some(rest) = entity.strip_prefix('#') {
        if let Some(hex) = rest.strip_prefix('x').or_else(|| rest.strip_prefix('X')) {
            let code = u32::from_str_radix(hex, 16).ok()?;
            return char::from_u32(code);
        } else {
            let code: u32 = rest.parse().ok()?;
            return char::from_u32(code);
        }
    }

    // Named entities (most common ones)
    match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        "nbsp" => Some('\u{00A0}'),
        "ndash" => Some('\u{2013}'),
        "mdash" => Some('\u{2014}'),
        "lsquo" => Some('\u{2018}'),
        "rsquo" => Some('\u{2019}'),
        "ldquo" => Some('\u{201C}'),
        "rdquo" => Some('\u{201D}'),
        "bull" => Some('\u{2022}'),
        "hellip" => Some('\u{2026}'),
        "copy" => Some('\u{00A9}'),
        "reg" => Some('\u{00AE}'),
        "trade" => Some('\u{2122}'),
        "laquo" => Some('\u{00AB}'),
        "raquo" => Some('\u{00BB}'),
        "deg" => Some('\u{00B0}'),
        "times" => Some('\u{00D7}'),
        "divide" => Some('\u{00F7}'),
        "euro" => Some('\u{20AC}'),
        "pound" => Some('\u{00A3}'),
        "yen" => Some('\u{00A5}'),
        "cent" => Some('\u{00A2}'),
        _ => None,
    }
}

/// Case-insensitive search for a substring in a haystack.
/// Returns the byte offset of the match start, or None.
/// Zero-allocation: compares byte-by-byte without lowercasing the entire haystack.
fn find_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    let needle_bytes: Vec<u8> = needle.bytes().map(|b| b.to_ascii_lowercase()).collect();
    if needle_bytes.is_empty() {
        return Some(0);
    }
    haystack.as_bytes()
        .windows(needle_bytes.len())
        .position(|w| w.iter().zip(&needle_bytes).all(|(a, b)| a.to_ascii_lowercase() == *b))
}

/// Parse HTML attributes from a string like: class="foo" id="bar" href="..."
fn parse_attrs(s: &str, attrs: &mut HashMap<String, String>) {
    let mut chars = s.chars().peekable();

    loop {
        // Skip whitespace
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }

        if chars.peek().is_none() {
            break;
        }

        // Read attribute name
        let mut name = String::new();
        while let Some(&ch) = chars.peek() {
            if ch == '=' || ch.is_whitespace() || ch == '>' {
                break;
            }
            name.push(ch);
            chars.next();
        }

        if name.is_empty() {
            break;
        }

        // Skip whitespace
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }

        // Check for =
        if chars.peek() == Some(&'=') {
            chars.next(); // consume '='

            // Skip whitespace
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }

            // Read value
            let mut value = String::new();
            if chars.peek() == Some(&'"') || chars.peek() == Some(&'\'') {
                let quote = chars.next().unwrap();
                for ch in chars.by_ref() {
                    if ch == quote {
                        break;
                    }
                    value.push(ch);
                }
            } else {
                // Unquoted value
                while let Some(&ch) = chars.peek() {
                    if ch.is_whitespace() || ch == '>' {
                        break;
                    }
                    value.push(ch);
                    chars.next();
                }
            }

            attrs.insert(name.to_lowercase(), decode_entities(&value));
        } else {
            // Boolean attribute (no value)
            attrs.insert(name.to_lowercase(), String::new());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_html() {
        let html = r#"<html><head><title>Test</title></head><body><h1>Hello</h1><p>World</p></body></html>"#;
        let dom = parse_html(html);
        assert!(dom.nodes.len() >= 5);
        assert_eq!(dom.nodes[0].tag, "html");
    }

    #[test]
    fn test_parse_attrs() {
        let html = r#"<a href="https://example.com" class="link">Click</a>"#;
        let dom = parse_html(html);
        let links: Vec<_> = dom.by_tag("a");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].attrs.get("href").unwrap(), "https://example.com");
    }

    #[test]
    fn test_links_extraction() {
        let html = r#"<a href="/page1">One</a><a href="/page2">Two</a>"#;
        let dom = parse_html(html);
        let links = dom.links();
        assert_eq!(links.len(), 2);
    }

    #[test]
    fn test_void_elements() {
        let html = r#"<p>Before<br>After</p><img src="test.png">"#;
        let dom = parse_html(html);
        let brs = dom.by_tag("br");
        assert_eq!(brs.len(), 1);
    }

    #[test]
    fn test_html_entities() {
        let html = r#"<p>A &amp; B &lt; C &gt; D &quot;E&quot;</p>"#;
        let dom = parse_html(html);
        let p = dom.by_tag("p");
        assert_eq!(p.len(), 1);
        assert!(p[0].text.contains("A & B < C > D \"E\""));
    }

    #[test]
    fn test_numeric_entities() {
        let html = r#"<p>&#65; &#x42;</p>"#;
        let dom = parse_html(html);
        let p = dom.by_tag("p");
        assert_eq!(p.len(), 1);
        assert!(p[0].text.contains('A'));
        assert!(p[0].text.contains('B'));
    }

    #[test]
    fn test_nbsp_entity() {
        let html = r#"<p>Hello&nbsp;World</p>"#;
        let dom = parse_html(html);
        let p = dom.by_tag("p");
        assert_eq!(p.len(), 1);
        // &nbsp; becomes \u{00A0} (non-breaking space)
        assert!(p[0].text.contains("Hello\u{00A0}World"));
    }

    #[test]
    fn test_comments_stripped() {
        let html = r#"<p>Before<!-- this is a comment -->After</p>"#;
        let dom = parse_html(html);
        let p = dom.by_tag("p");
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].text, "Before After");
        assert!(!p[0].text.contains("comment"));
    }

    #[test]
    fn test_script_content_skipped() {
        let html = r#"<p>Hello</p><script>var x = 1; alert("hi");</script><p>World</p>"#;
        let dom = parse_html(html);
        let ps = dom.by_tag("p");
        assert_eq!(ps.len(), 2);
        assert_eq!(ps[0].text, "Hello");
        assert_eq!(ps[1].text, "World");
        // Script tag node exists but has no text content
        let scripts = dom.by_tag("script");
        assert_eq!(scripts.len(), 1);
        assert!(scripts[0].text.is_empty());
    }

    #[test]
    fn test_style_content_skipped() {
        let html = r#"<style>body { color: red; }</style><p>Content</p>"#;
        let dom = parse_html(html);
        let ps = dom.by_tag("p");
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].text, "Content");
        let styles = dom.by_tag("style");
        assert_eq!(styles.len(), 1);
        assert!(styles[0].text.is_empty());
    }

    #[test]
    fn test_whitespace_collapse() {
        let html = "<p>  Hello   \n\t  World  </p>";
        let dom = parse_html(html);
        let p = dom.by_tag("p");
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].text, "Hello World");
    }

    #[test]
    fn test_self_closing_tags() {
        let html = r#"<p>Before<br/>After</p><img src="a.png" />"#;
        let dom = parse_html(html);
        let brs = dom.by_tag("br");
        assert_eq!(brs.len(), 1);
        let imgs = dom.by_tag("img");
        assert_eq!(imgs.len(), 1);
    }

    #[test]
    fn test_nested_elements() {
        let html = r#"<div><ul><li>Item 1</li><li>Item 2</li></ul></div>"#;
        let dom = parse_html(html);
        let divs = dom.by_tag("div");
        assert_eq!(divs.len(), 1);
        let lis = dom.by_tag("li");
        assert_eq!(lis.len(), 2);
        assert_eq!(lis[0].text, "Item 1");
        assert_eq!(lis[1].text, "Item 2");
        assert_eq!(divs[0].depth, 0);
        assert_eq!(lis[0].depth, 2);
    }

    #[test]
    fn test_deeply_nested() {
        let html = r#"<div><div><div><div><p>Deep</p></div></div></div></div>"#;
        let dom = parse_html(html);
        let ps = dom.by_tag("p");
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].depth, 4);
        assert_eq!(ps[0].text, "Deep");
    }

    #[test]
    fn test_malformed_unclosed_tags() {
        let html = r#"<p>One<p>Two<p>Three"#;
        let dom = parse_html(html);
        let ps = dom.by_tag("p");
        assert_eq!(ps.len(), 3);
    }

    #[test]
    fn test_malformed_extra_closing() {
        let html = r#"<p>Text</p></p></div>"#;
        let dom = parse_html(html);
        let ps = dom.by_tag("p");
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].text, "Text");
    }

    #[test]
    fn test_empty_html() {
        let dom = parse_html("");
        assert_eq!(dom.nodes.len(), 0);
    }

    #[test]
    fn test_text_only() {
        let dom = parse_html("Hello world");
        let text = dom.text_content();
        assert!(text.contains("Hello world"));
    }

    #[test]
    fn test_comment_and_doctype_skipped() {
        let html = r#"<!DOCTYPE html><!-- comment --><p>Content</p>"#;
        let dom = parse_html(html);
        let ps = dom.by_tag("p");
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].text, "Content");
    }

    #[test]
    fn test_boolean_attributes() {
        let html = r#"<input disabled readonly type="text">"#;
        let dom = parse_html(html);
        let inputs = dom.by_tag("input");
        assert_eq!(inputs.len(), 1);
        assert!(inputs[0].attrs.contains_key("disabled"));
        assert!(inputs[0].attrs.contains_key("readonly"));
        assert_eq!(inputs[0].attrs.get("type").unwrap(), "text");
    }

    #[test]
    fn test_single_quoted_attributes() {
        let html = r#"<a href='https://example.com' class='link'>Click</a>"#;
        let dom = parse_html(html);
        let links = dom.by_tag("a");
        assert_eq!(links[0].attrs.get("href").unwrap(), "https://example.com");
        assert_eq!(links[0].attrs.get("class").unwrap(), "link");
    }

    #[test]
    fn test_images_extraction() {
        let html = r#"<img src="a.png"><img src="b.jpg"><p>Text</p>"#;
        let dom = parse_html(html);
        let imgs = dom.images();
        assert_eq!(imgs.len(), 2);
        assert_eq!(imgs[0], "a.png");
        assert_eq!(imgs[1], "b.jpg");
    }

    #[test]
    fn test_text_content_aggregation() {
        let html = r#"<h1>Title</h1><p>Paragraph one.</p><p>Paragraph two.</p>"#;
        let dom = parse_html(html);
        let text = dom.text_content();
        assert!(text.contains("Title"));
        assert!(text.contains("Paragraph one."));
        assert!(text.contains("Paragraph two."));
    }

    #[test]
    fn test_entity_in_attribute() {
        let html = r#"<a href="/search?q=a&amp;b">Link</a>"#;
        let dom = parse_html(html);
        let links = dom.by_tag("a");
        assert_eq!(links[0].attrs.get("href").unwrap(), "/search?q=a&b");
    }

    #[test]
    fn test_multiline_comment() {
        let html = "<p>A</p><!--\nmultiline\ncomment\n--><p>B</p>";
        let dom = parse_html(html);
        let ps = dom.by_tag("p");
        assert_eq!(ps.len(), 2);
        assert_eq!(ps[0].text, "A");
        assert_eq!(ps[1].text, "B");
    }

    #[test]
    fn test_inline_script_with_html_inside() {
        let html = r#"<script>if (a < b && c > d) { document.write("<p>nope</p>"); }</script><p>Real</p>"#;
        let dom = parse_html(html);
        let ps = dom.by_tag("p");
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].text, "Real");
    }
}
