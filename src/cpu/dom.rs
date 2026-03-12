//! CPU: Minimal DOM parser using html5ever.
//!
//! Produces a flat DomTree that CPU can cheaply build and NPU can analyze.
//! We do NOT build a full browser DOM — that's wasted CPU work.
//! Instead, we extract just enough structure for the NPU to understand the page.

use std::collections::HashMap;

/// Lightweight DOM node — just enough for NPU analysis.
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

/// Flat DOM tree — array of nodes, root is index 0.
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

/// Parse raw HTML into a flat DomTree.
/// Uses a simple tag-based parser (no full html5ever tree builder needed for MVP).
pub fn parse_html(html: &str) -> DomTree {
    let mut nodes = Vec::new();
    let mut stack: Vec<usize> = Vec::new(); // parent stack

    // Simple state machine parser
    let mut chars = html.chars().peekable();
    let mut current_text = String::new();

    while let Some(ch) = chars.next() {
        if ch == '<' {
            // Flush text to current parent
            let trimmed = current_text.trim().to_string();
            if !trimmed.is_empty() {
                if let Some(&parent_id) = stack.last() {
                    if let Some(parent) = nodes.get_mut(parent_id) {
                        let parent: &mut DomNode = parent;
                        if parent.text.is_empty() {
                            parent.text = trimmed;
                        } else {
                            parent.text.push(' ');
                            parent.text.push_str(&trimmed);
                        }
                    }
                } else {
                    // Text outside any tag — create a text node
                    let id = nodes.len();
                    nodes.push(DomNode {
                        id,
                        tag: "#text".into(),
                        attrs: HashMap::new(),
                        text: trimmed,
                        parent: None,
                        children: Vec::new(),
                        depth: 0,
                    });
                }
            }
            current_text.clear();

            // Read tag
            let mut tag_content = String::new();
            let mut in_quote = false;
            let mut quote_char = '"';
            for ch in chars.by_ref() {
                if !in_quote && ch == '>' {
                    break;
                }
                if !in_quote && (ch == '"' || ch == '\'') {
                    in_quote = true;
                    quote_char = ch;
                } else if in_quote && ch == quote_char {
                    in_quote = false;
                }
                tag_content.push(ch);
            }

            let tag_content = tag_content.trim().to_string();

            if tag_content.starts_with('!') || tag_content.starts_with('?') {
                // Comment or doctype — skip
                continue;
            }

            if tag_content.starts_with('/') {
                // Closing tag — pop stack
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
                // Opening tag
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

                // Void elements and self-closing tags don't push to stack
                let void_elements = [
                    "area", "base", "br", "col", "embed", "hr", "img",
                    "input", "link", "meta", "param", "source", "track", "wbr",
                ];
                if !self_closing && !void_elements.contains(&tag_name.as_str()) {
                    stack.push(id);
                }
            }
        } else {
            current_text.push(ch);
        }
    }

    DomTree { nodes }
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

            attrs.insert(name.to_lowercase(), value);
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
}
