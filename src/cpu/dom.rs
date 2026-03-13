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

/// Where a script comes from.
#[derive(Debug, Clone)]
pub enum ScriptSource {
    /// Inline script content embedded in `<script>...</script>`.
    Inline(String),
    /// External script loaded via `<script src="...">`.
    External(String),
}

/// A script found during HTML parsing.
#[derive(Debug, Clone)]
pub struct ScriptInfo {
    /// The source of this script (inline code or external URL).
    pub source: ScriptSource,
    /// The `type` attribute, if any (e.g. "module", "text/javascript").
    pub script_type: Option<String>,
    /// Whether this script has `defer` attribute.
    pub defer: bool,
    /// Whether this script has `async` attribute.
    pub is_async: bool,
}

/// Flat DOM tree -- array of nodes, root is index 0.
#[derive(Debug, Clone)]
pub struct DomTree {
    pub nodes: Vec<DomNode>,
    /// Scripts collected during parsing (in document order).
    pub scripts: Vec<ScriptInfo>,
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

    // ── Mutable DOM APIs (for JavaScript engine) ──

    /// Create a new detached element node. Returns its node ID.
    pub fn create_element(&mut self, tag: &str) -> usize {
        let id = self.nodes.len();
        self.nodes.push(DomNode {
            id,
            tag: tag.to_lowercase(),
            attrs: HashMap::new(),
            text: String::new(),
            parent: None,
            children: Vec::new(),
            depth: 0,
        });
        id
    }

    /// Check if `potential_ancestor` is an ancestor of `node` by walking up the parent chain.
    fn is_ancestor(&self, potential_ancestor: usize, node: usize) -> bool {
        let mut current = Some(node);
        while let Some(id) = current {
            if id == potential_ancestor { return true; }
            current = self.nodes.get(id).and_then(|n| n.parent);
        }
        false
    }

    /// Append `child` as the last child of `parent`. Updates depth recursively.
    pub fn append_child(&mut self, parent_id: usize, child_id: usize) -> bool {
        if parent_id >= self.nodes.len() || child_id >= self.nodes.len() {
            return false;
        }
        // Guard against self-append
        if parent_id == child_id {
            return false;
        }
        // Guard against cycles: child must not be an ancestor of parent
        if self.is_ancestor(child_id, parent_id) {
            return false;
        }
        // Remove from old parent if any
        if let Some(old_parent) = self.nodes[child_id].parent {
            if old_parent < self.nodes.len() {
                self.nodes[old_parent].children.retain(|&c| c != child_id);
            }
        }
        self.nodes[parent_id].children.push(child_id);
        self.nodes[child_id].parent = Some(parent_id);
        let new_depth = self.nodes[parent_id].depth + 1;
        self.update_depth(child_id, new_depth);
        true
    }

    /// Remove `child` from `parent`.
    pub fn remove_child(&mut self, parent_id: usize, child_id: usize) -> bool {
        if parent_id >= self.nodes.len() || child_id >= self.nodes.len() {
            return false;
        }
        let before = self.nodes[parent_id].children.len();
        self.nodes[parent_id].children.retain(|&c| c != child_id);
        if self.nodes[parent_id].children.len() < before {
            self.nodes[child_id].parent = None;
            true
        } else {
            false
        }
    }

    /// Set an attribute on a node.
    pub fn set_attribute(&mut self, node_id: usize, key: &str, value: &str) -> bool {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.attrs.insert(key.to_string(), value.to_string());
            true
        } else {
            false
        }
    }

    /// Get an attribute value from a node.
    pub fn get_attribute(&self, node_id: usize, key: &str) -> Option<&str> {
        self.nodes.get(node_id)?.attrs.get(key).map(|s| s.as_str())
    }

    /// Set the text content of a node (replaces existing text).
    pub fn set_text_content(&mut self, node_id: usize, text: &str) -> bool {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.text = text.to_string();
            true
        } else {
            false
        }
    }

    /// Set innerHTML on a node: re-parses HTML and attaches children.
    pub fn set_inner_html(&mut self, node_id: usize, html: &str) -> bool {
        if node_id >= self.nodes.len() {
            return false;
        }
        // Detach old children before clearing
        let old_children: Vec<usize> = self.nodes[node_id].children.clone();
        for &old_child in &old_children {
            if old_child < self.nodes.len() {
                self.nodes[old_child].parent = None;
            }
        }
        self.nodes[node_id].children.clear();
        self.nodes[node_id].text.clear();

        // Parse the HTML fragment
        let fragment = parse_html(html);
        let base_id = self.nodes.len();
        let parent_depth = self.nodes[node_id].depth;

        // Copy fragment nodes, remapping IDs
        for frag_node in &fragment.nodes {
            let new_id = base_id + frag_node.id;
            let new_parent = match frag_node.parent {
                Some(p) => Some(base_id + p),
                None => Some(node_id), // top-level fragment nodes become children of target
            };
            self.nodes.push(DomNode {
                id: new_id,
                tag: frag_node.tag.clone(),
                attrs: frag_node.attrs.clone(),
                text: frag_node.text.clone(),
                parent: new_parent,
                children: frag_node.children.iter().map(|c| base_id + c).collect(),
                depth: parent_depth + 1 + frag_node.depth,
            });
        }

        // Attach top-level fragment nodes as children
        for frag_node in &fragment.nodes {
            if frag_node.parent.is_none() {
                self.nodes[node_id].children.push(base_id + frag_node.id);
            }
        }

        true
    }

    /// Find a node by its `id` attribute.
    pub fn get_element_by_id(&self, id_value: &str) -> Option<usize> {
        self.nodes.iter().find_map(|n| {
            if n.attrs.get("id").map(|s| s.as_str()) == Some(id_value) {
                Some(n.id)
            } else {
                None
            }
        })
    }

    /// Simple query_selector: supports tag, #id, .class, tag.class, tag#id, [attr], [attr=val], *.
    pub fn query_selector(&self, selector: &str) -> Option<usize> {
        let selector = selector.trim();
        if selector.is_empty() { return None; }
        if selector == "*" { return self.nodes.first().map(|n| n.id); }
        self.nodes.iter().find_map(|n| {
            if matches_simple_selector(n, selector) { Some(n.id) } else { None }
        })
    }

    /// Query all matching elements (simple selectors only).
    pub fn query_selector_all(&self, selector: &str) -> Vec<usize> {
        let selector = selector.trim();
        if selector.is_empty() { return Vec::new(); }
        self.nodes.iter().filter_map(|n| {
            if matches_simple_selector(n, selector) { Some(n.id) } else { None }
        }).collect()
    }

    /// Recursively update depth of a node and its children.
    fn update_depth(&mut self, node_id: usize, new_depth: usize) {
        if node_id >= self.nodes.len() {
            return;
        }
        self.nodes[node_id].depth = new_depth;
        let children: Vec<usize> = self.nodes[node_id].children.clone();
        for child in children {
            self.update_depth(child, new_depth + 1);
        }
    }
}

/// Match a simple CSS selector against a node: tag, .class, #id, tag.class, tag#id, [attr], [attr=val], *
fn matches_simple_selector(node: &DomNode, selector: &str) -> bool {
    let sel = selector.trim();
    if sel.is_empty() { return false; }
    if sel == "*" { return true; }

    let mut tag_req: Option<String> = None;
    let mut class_reqs: Vec<String> = Vec::new();
    let mut id_req: Option<String> = None;

    let mut chars = sel.chars().peekable();
    let mut current = String::new();
    let mut mode = 'T'; // T=tag, C=class, I=id

    while let Some(&ch) = chars.peek() {
        match ch {
            '.' => {
                if mode == 'T' && !current.is_empty() { tag_req = Some(current.to_lowercase()); }
                else if mode == 'C' && !current.is_empty() { class_reqs.push(current.clone()); }
                current.clear();
                mode = 'C';
                chars.next();
            }
            '#' if mode != 'C' || current.is_empty() => {
                if mode == 'T' && !current.is_empty() { tag_req = Some(current.to_lowercase()); }
                else if mode == 'C' && !current.is_empty() { class_reqs.push(current.clone()); }
                current.clear();
                mode = 'I';
                chars.next();
            }
            '[' => {
                if mode == 'T' && !current.is_empty() { tag_req = Some(current.to_lowercase()); }
                else if mode == 'C' && !current.is_empty() { class_reqs.push(current.clone()); }
                current.clear();
                chars.next();
                // Read attribute selector
                let mut attr_content = String::new();
                while let Some(&ach) = chars.peek() {
                    if ach == ']' { chars.next(); break; }
                    attr_content.push(ach);
                    chars.next();
                }
                // Parse attr_content: "name" or "name=value" or "name=\"value\""
                if let Some(eq_pos) = attr_content.find('=') {
                    let attr_name = attr_content[..eq_pos].trim();
                    let attr_val = attr_content[eq_pos+1..].trim().trim_matches('"').trim_matches('\'');
                    if node.attrs.get(attr_name).map(|v| v.as_str()) != Some(attr_val) { return false; }
                } else {
                    if !node.attrs.contains_key(attr_content.trim()) { return false; }
                }
                mode = 'T';
                continue;
            }
            _ => {
                current.push(ch);
                chars.next();
            }
        }
    }
    match mode {
        'T' if !current.is_empty() => { tag_req = Some(current.to_lowercase()); }
        'C' if !current.is_empty() => { class_reqs.push(current); }
        'I' if !current.is_empty() => { id_req = Some(current); }
        _ => {}
    }

    if let Some(ref tag) = tag_req {
        if node.tag != *tag { return false; }
    }
    if let Some(ref id) = id_req {
        if node.attrs.get("id").map(|s| s.as_str()) != Some(id.as_str()) { return false; }
    }
    for cls in &class_reqs {
        if !node.attrs.get("class").is_some_and(|c| c.split_whitespace().any(|x| x == cls.as_str())) {
            return false;
        }
    }
    // Must have matched at least one requirement
    tag_req.is_some() || id_req.is_some() || !class_reqs.is_empty()
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
    let mut scripts = Vec::new();
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
                let closing_tag = tag_content[1..].split_whitespace()
                    .next().unwrap_or("").to_lowercase();
                // Find the matching tag in the stack
                if let Some(match_pos) = stack.iter().rposition(|&id| nodes[id].tag == closing_tag) {
                    // Pop from the top down to and including the match
                    stack.truncate(match_pos);
                }
                // If no match found, ignore the closing tag entirely
            } else {
                // ── Opening tag ──
                let self_closing = tag_content.ends_with('/') && {
                    // Only treat as self-closing if the trailing / is outside quotes
                    let mut in_q = false;
                    let mut last_outside_slash = false;
                    for c in tag_content.chars() {
                        if c == '"' || c == '\'' { in_q = !in_q; }
                        last_outside_slash = !in_q && c == '/';
                    }
                    last_outside_slash
                };
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

                // If this is a raw-text element (script/style), capture content
                // until the matching closing tag.
                if !is_void && RAW_TEXT_TAGS.contains(&tag_name.as_str()) {
                    let close_tag = format!("</{tag_name}>");
                    let script_content;
                    if let Some(found) = find_case_insensitive(&html[pos..], &close_tag) {
                        script_content = &html[pos..pos + found];
                        pos += found + close_tag.len();
                    } else {
                        // No closing tag found — consume rest as raw content
                        script_content = &html[pos..];
                        pos = len;
                    }

                    // Capture script info
                    if tag_name == "script" {
                        let src_attr = nodes.last()
                            .and_then(|n: &DomNode| n.attrs.get("src").cloned());
                        let type_attr = nodes.last()
                            .and_then(|n: &DomNode| n.attrs.get("type").cloned());
                        let defer = nodes.last()
                            .is_some_and(|n| n.attrs.contains_key("defer"));
                        let is_async = nodes.last()
                            .is_some_and(|n| n.attrs.contains_key("async"));

                        let source = if let Some(src) = src_attr {
                            ScriptSource::External(src)
                        } else {
                            let trimmed = script_content.trim();
                            if trimmed.is_empty() {
                                // Skip empty inline scripts entirely
                                continue;
                            }
                            ScriptSource::Inline(trimmed.to_string())
                        };

                        scripts.push(ScriptInfo {
                            source,
                            script_type: type_attr,
                            defer,
                            is_async,
                        });
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

    DomTree { nodes, scripts }
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
            // Collect up to 32 chars until ';' or limit
            for _ in 0..32 {
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
            if code == 0 { return Some('\u{FFFD}'); }
            return char::from_u32(code);
        } else {
            let code: u32 = rest.parse().ok()?;
            if code == 0 { return Some('\u{FFFD}'); }
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
        "ensp" => Some('\u{2002}'),
        "emsp" => Some('\u{2003}'),
        "thinsp" => Some('\u{2009}'),
        "zwnj" => Some('\u{200C}'),
        "zwj" => Some('\u{200D}'),
        "lrm" => Some('\u{200E}'),
        "rlm" => Some('\u{200F}'),
        "iexcl" => Some('\u{00A1}'),
        "iquest" => Some('\u{00BF}'),
        "frac14" => Some('\u{00BC}'),
        "frac12" => Some('\u{00BD}'),
        "frac34" => Some('\u{00BE}'),
        "sup1" => Some('\u{00B9}'),
        "sup2" => Some('\u{00B2}'),
        "sup3" => Some('\u{00B3}'),
        "plusmn" => Some('\u{00B1}'),
        "micro" => Some('\u{00B5}'),
        "para" => Some('\u{00B6}'),
        "sect" => Some('\u{00A7}'),
        "uml" => Some('\u{00A8}'),
        "macr" => Some('\u{00AF}'),
        "acute" => Some('\u{00B4}'),
        "cedil" => Some('\u{00B8}'),
        "ordf" => Some('\u{00AA}'),
        "ordm" => Some('\u{00BA}'),
        "not" => Some('\u{00AC}'),
        "shy" => Some('\u{00AD}'),
        "middot" => Some('\u{00B7}'),
        "larr" => Some('\u{2190}'),
        "uarr" => Some('\u{2191}'),
        "rarr" => Some('\u{2192}'),
        "darr" => Some('\u{2193}'),
        "harr" => Some('\u{2194}'),
        "hearts" => Some('\u{2665}'),
        "diams" => Some('\u{2666}'),
        "spades" => Some('\u{2660}'),
        "clubs" => Some('\u{2663}'),
        "permil" => Some('\u{2030}'),
        "infin" => Some('\u{221E}'),
        "ne" => Some('\u{2260}'),
        "le" => Some('\u{2264}'),
        "ge" => Some('\u{2265}'),
        "sum" => Some('\u{2211}'),
        "prod" => Some('\u{220F}'),
        "radic" => Some('\u{221A}'),
        "empty" => Some('\u{2205}'),
        "exist" => Some('\u{2203}'),
        "forall" => Some('\u{2200}'),
        "part" => Some('\u{2202}'),
        "nabla" => Some('\u{2207}'),
        "isin" => Some('\u{2208}'),
        "notin" => Some('\u{2209}'),
        "sub" => Some('\u{2282}'),
        "sup" => Some('\u{2283}'),
        "cap" => Some('\u{2229}'),
        "cup" => Some('\u{222A}'),
        "and" => Some('\u{2227}'),
        "or" => Some('\u{2228}'),
        "there4" => Some('\u{2234}'),
        "sim" => Some('\u{223C}'),
        "cong" => Some('\u{2245}'),
        "asymp" => Some('\u{2248}'),
        "equiv" => Some('\u{2261}'),
        "oplus" => Some('\u{2295}'),
        "otimes" => Some('\u{2297}'),
        "loz" => Some('\u{25CA}'),
        "circ" => Some('\u{02C6}'),
        "tilde" => Some('\u{02DC}'),
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

    // ── Script collection tests ──

    #[test]
    fn test_script_collection_inline() {
        let html = r#"<script>var x = 42;</script><p>Hello</p>"#;
        let dom = parse_html(html);
        assert_eq!(dom.scripts.len(), 1);
        match &dom.scripts[0].source {
            ScriptSource::Inline(code) => assert!(code.contains("var x = 42")),
            _ => panic!("Expected inline script"),
        }
    }

    #[test]
    fn test_script_collection_external() {
        let html = r#"<script src="app.js"></script><p>Hello</p>"#;
        let dom = parse_html(html);
        assert_eq!(dom.scripts.len(), 1);
        match &dom.scripts[0].source {
            ScriptSource::External(url) => assert_eq!(url, "app.js"),
            _ => panic!("Expected external script"),
        }
    }

    #[test]
    fn test_script_collection_multiple() {
        let html = r#"<script>var a=1;</script><script src="b.js" defer></script><script>var c=3;</script>"#;
        let dom = parse_html(html);
        assert_eq!(dom.scripts.len(), 3);
        assert!(dom.scripts[1].defer);
    }

    #[test]
    fn test_script_async_attr() {
        let html = r#"<script async src="analytics.js"></script>"#;
        let dom = parse_html(html);
        assert_eq!(dom.scripts.len(), 1);
        assert!(dom.scripts[0].is_async);
    }

    // ── Mutable DOM tests ──

    #[test]
    fn test_create_element() {
        let mut dom = parse_html("<div></div>");
        let id = dom.create_element("span");
        assert_eq!(dom.nodes[id].tag, "span");
        assert!(dom.nodes[id].parent.is_none());
    }

    #[test]
    fn test_append_child() {
        let mut dom = parse_html("<div></div>");
        let child = dom.create_element("p");
        dom.append_child(0, child);
        assert_eq!(dom.nodes[0].children.len(), 1);
        assert_eq!(dom.nodes[child].parent, Some(0));
        assert_eq!(dom.nodes[child].depth, 1);
    }

    #[test]
    fn test_remove_child() {
        let mut dom = parse_html("<div><p>Text</p></div>");
        let p_id = dom.query_selector("p").unwrap();
        let div_id = 0;
        assert!(dom.remove_child(div_id, p_id));
        assert!(!dom.nodes[div_id].children.contains(&p_id));
        assert!(dom.nodes[p_id].parent.is_none());
    }

    #[test]
    fn test_set_attribute() {
        let mut dom = parse_html("<div></div>");
        dom.set_attribute(0, "class", "container");
        assert_eq!(dom.nodes[0].attrs.get("class").unwrap(), "container");
    }

    #[test]
    fn test_set_text_content() {
        let mut dom = parse_html("<p>Old</p>");
        let p_id = dom.query_selector("p").unwrap();
        dom.set_text_content(p_id, "New");
        assert_eq!(dom.nodes[p_id].text, "New");
    }

    #[test]
    fn test_get_element_by_id() {
        let html = r#"<div id="main"><p id="intro">Hello</p></div>"#;
        let dom = parse_html(html);
        let id = dom.get_element_by_id("intro");
        assert!(id.is_some());
        assert_eq!(dom.nodes[id.unwrap()].tag, "p");
    }

    #[test]
    fn test_query_selector_tag() {
        let html = "<div><span>A</span><p>B</p></div>";
        let dom = parse_html(html);
        let p = dom.query_selector("p");
        assert!(p.is_some());
        assert_eq!(dom.nodes[p.unwrap()].text, "B");
    }

    #[test]
    fn test_query_selector_class() {
        let html = r#"<div class="a"><p class="target b">Hit</p></div>"#;
        let dom = parse_html(html);
        let hit = dom.query_selector(".target");
        assert!(hit.is_some());
        assert_eq!(dom.nodes[hit.unwrap()].text, "Hit");
    }

    #[test]
    fn test_query_selector_all() {
        let html = "<ul><li>A</li><li>B</li><li>C</li></ul>";
        let dom = parse_html(html);
        let lis = dom.query_selector_all("li");
        assert_eq!(lis.len(), 3);
    }

    #[test]
    fn test_set_inner_html() {
        let mut dom = parse_html("<div id='box'></div>");
        let box_id = dom.get_element_by_id("box").unwrap();
        dom.set_inner_html(box_id, "<p>New content</p><span>More</span>");
        assert!(dom.nodes[box_id].children.len() >= 2);
    }
}
