//! NPU: Content extraction — turns DOM into semantic blocks.
//!
//! This is the core "understanding" module. Instead of rendering
//! every HTML element, we extract MEANING:
//! - What is the main content?
//! - What are headings, paragraphs, images, code blocks?
//! - What is navigation/boilerplate/footer?
//!
//! Phase 1: Heuristic extraction (works without ONNX models)
//! Phase 2: ONNX model (MarkupLM/LayoutLM) for ML-based extraction

use crate::cpu::dom::{DomNode, DomTree};
use anyhow::Result;
use log::debug;

/// Kind of content block — the NPU's semantic classification.
#[derive(Debug, Clone)]
pub enum BlockKind {
    /// Page title
    Title,
    /// Heading (h1-h6)
    Heading { level: u8 },
    /// Text paragraph
    Paragraph,
    /// Image with optional alt text
    Image { src: String, alt: String },
    /// Code block (pre/code)
    Code { language: Option<String> },
    /// List (ul/ol)
    List { ordered: bool },
    /// List item
    ListItem,
    /// Blockquote
    Quote,
    /// Table
    Table,
    /// Table row (rendered as a single line of text)
    TableRow,
    /// Definition list
    DefinitionList,
    /// Definition term (dt)
    DefinitionTerm,
    /// Definition description (dd)
    DefinitionDesc,
    /// Figure with optional caption
    Figure,
    /// Figure caption
    FigCaption,
    /// Details/summary collapsible section
    Details { open: bool },
    /// Summary element (clickable header for details)
    Summary,
    /// Form element (rendered as text description)
    Form,
    /// Link
    Link { href: String },
    /// Separator / horizontal rule
    Separator,
    /// Navigation (identified by NPU as non-content)
    Navigation,
    /// Footer/boilerplate (identified by NPU as non-content)
    Boilerplate,
}

/// A semantic block of content — what the GPU renderer will display.
#[derive(Debug, Clone)]
pub struct ContentBlock {
    pub kind: BlockKind,
    pub text: String,
    pub depth: usize,
    /// Confidence that this is main content (0.0-1.0).
    /// NPU classifier output. GPU can use this for emphasis.
    pub relevance: f32,
    /// Children blocks (for nested structures like lists).
    pub children: Vec<ContentBlock>,
    /// Decoded image data: (width, height, RGBA bytes). Populated by NPU image fetcher.
    pub image_data: Option<(u32, u32, Vec<u8>)>,
    /// DOM node ID this block was extracted from (for CSS style mapping).
    pub node_id: Option<usize>,
    /// CSS computed style from the cascade engine (attached by NPU).
    /// GPU layout uses this instead of hardcoded values when present.
    pub computed_style: Option<Box<crate::css::cascade::ComputedStyle>>,
}

/// Content extractor — converts DOM to semantic blocks.
///
/// Phase 1: Rule-based (no ONNX model needed).
/// Phase 2: MarkupLM ONNX model for ML classification.
pub struct ContentExtractor {
    // Future: ort::Session for ONNX model
}

impl ContentExtractor {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }

    /// Extract semantic content blocks from DOM.
    pub fn extract(&self, dom: &DomTree) -> Result<Vec<ContentBlock>> {
        let mut blocks = Vec::new();

        // Handle empty DOM gracefully
        if dom.nodes.is_empty() {
            debug!("[NPU:CONTENT] Empty DOM — no blocks to extract");
            return Ok(blocks);
        }

        // Extract title
        if let Some(title) = dom.by_tag("title").first() {
            let title_text = title.text.trim().to_string();
            if !title_text.is_empty() {
                blocks.push(ContentBlock {
                    kind: BlockKind::Title,
                    text: title_text,
                    depth: 0,
                    relevance: 1.0,
                    children: Vec::new(),
                    image_data: None, node_id: Some(title.id), computed_style: None,
                });
            }
        }

        // Walk body content
        let body_nodes = self.find_body_content(dom);
        for node in &body_nodes {
            self.extract_node_recursive(node, dom, &mut blocks);
        }

        // Score relevance (text density heuristic)
        self.score_relevance(&mut blocks);

        debug!("[NPU:CONTENT] Extracted {} blocks", blocks.len());
        Ok(blocks)
    }

    /// Check if a node should be hidden (display:none, visibility:hidden, hidden attr, aria-hidden).
    fn is_hidden(node: &DomNode) -> bool {
        // HTML hidden attribute
        if node.attrs.contains_key("hidden") {
            return true;
        }
        // aria-hidden="true"
        if node.attrs.get("aria-hidden").map(|v| v == "true").unwrap_or(false) {
            return true;
        }
        // Inline style: display:none or visibility:hidden
        if let Some(style) = node.attrs.get("style") {
            let s = style.to_lowercase();
            if s.contains("display:none") || s.contains("display: none") {
                return true;
            }
            if s.contains("visibility:hidden") || s.contains("visibility: hidden") {
                return true;
            }
        }
        false
    }

    /// Recursively extract a node and its children into blocks.
    fn extract_node_recursive(
        &self,
        node: &DomNode,
        dom: &DomTree,
        blocks: &mut Vec<ContentBlock>,
    ) {
        // Skip hidden elements
        if Self::is_hidden(node) {
            return;
        }

        match node.tag.as_str() {
            // ── Nested list handling: ul/ol inside li ──
            "ul" | "ol" => {
                let ordered = node.tag == "ol";
                let mut list_children = Vec::new();
                for &child_id in &node.children {
                    if let Some(child) = dom.nodes.get(child_id) {
                        if child.tag == "li" {
                            let item = self.extract_list_item(child, dom, ordered);
                            list_children.push(item);
                        }
                    }
                }
                if !list_children.is_empty() {
                    blocks.push(ContentBlock {
                        kind: BlockKind::List { ordered },
                        text: String::new(),
                        depth: node.depth,
                        relevance: 0.5,
                        children: list_children,
                        image_data: None, node_id: Some(node.id), computed_style: None,
                    });
                }
            }

            // ── Table handling: extract structured text ──
            "table" => {
                let table_block = self.extract_table(node, dom);
                if let Some(b) = table_block {
                    blocks.push(b);
                }
            }

            // ── Definition list handling ──
            "dl" => {
                let dl_block = self.extract_definition_list(node, dom);
                if let Some(b) = dl_block {
                    blocks.push(b);
                }
            }

            // ── Figure / figcaption ──
            "figure" => {
                let fig_block = self.extract_figure(node, dom);
                if let Some(b) = fig_block {
                    blocks.push(b);
                }
            }

            // ── Details / summary ──
            "details" => {
                let details_block = self.extract_details(node, dom);
                if let Some(b) = details_block {
                    blocks.push(b);
                }
            }

            // ── Form elements ──
            "form" => {
                let form_block = self.extract_form(node, dom);
                if let Some(b) = form_block {
                    blocks.push(b);
                }
            }

            // ── Standard block-level elements ──
            _ => {
                if let Some(block) = self.node_to_block(node, dom) {
                    blocks.push(block);
                }
            }
        }
    }

    /// Extract a list item, handling nested lists within it.
    fn extract_list_item(&self, node: &DomNode, dom: &DomTree, _parent_ordered: bool) -> ContentBlock {
        let mut children = Vec::new();
        let mut text = node.text.clone();

        // Check for nested lists inside this li
        for &child_id in &node.children {
            if let Some(child) = dom.nodes.get(child_id) {
                match child.tag.as_str() {
                    "ul" | "ol" => {
                        let nested_ordered = child.tag == "ol";
                        let mut nested_items = Vec::new();
                        for &nested_child_id in &child.children {
                            if let Some(nested_child) = dom.nodes.get(nested_child_id) {
                                if nested_child.tag == "li" {
                                    nested_items.push(self.extract_list_item(
                                        nested_child, dom, nested_ordered,
                                    ));
                                }
                            }
                        }
                        if !nested_items.is_empty() {
                            children.push(ContentBlock {
                                kind: BlockKind::List { ordered: nested_ordered },
                                text: String::new(),
                                depth: child.depth,
                                relevance: 0.5,
                                children: nested_items,
                                image_data: None, node_id: Some(child.id), computed_style: None,
                            });
                        }
                    }
                    "a" => {
                        // Preserve link hrefs inside list items as child Link blocks
                        let href = child.attrs.get("href").cloned().unwrap_or_default();
                        let link_text = child.text.clone();
                        if !link_text.is_empty() {
                            // Also accumulate text for the parent ListItem
                            if !text.is_empty() {
                                text.push(' ');
                            }
                            text.push_str(&link_text);
                            children.push(ContentBlock {
                                kind: BlockKind::Link { href },
                                text: link_text,
                                depth: child.depth,
                                relevance: 0.6,
                                children: Vec::new(),
                                image_data: None, node_id: Some(child.id), computed_style: None,
                            });
                        }
                    }
                    _ => {
                        // Accumulate text from inline children
                        if !child.text.is_empty() && child.tag != "ul" && child.tag != "ol" {
                            if !text.is_empty() && !child.text.is_empty() {
                                text.push(' ');
                            }
                            text.push_str(&child.text);
                        }
                    }
                }
            }
        }

        ContentBlock {
            kind: BlockKind::ListItem,
            text,
            depth: node.depth,
            relevance: 0.5,
            children,
            image_data: None, node_id: Some(node.id), computed_style: None,
        }
    }

    /// Extract table as structured text with rows.
    fn extract_table(&self, table_node: &DomNode, dom: &DomTree) -> Option<ContentBlock> {
        let mut row_blocks = Vec::new();

        // Collect all tr elements (from thead, tbody, tfoot, or directly)
        let tr_ids = self.collect_descendant_tags(table_node, dom, "tr");

        for tr_id in &tr_ids {
            if let Some(tr_node) = dom.nodes.get(*tr_id) {
                let mut cells: Vec<String> = Vec::new();
                for &cell_id in &tr_node.children {
                    if let Some(cell) = dom.nodes.get(cell_id) {
                        if cell.tag == "td" || cell.tag == "th" {
                            let cell_text = self.collect_text_recursive(cell, dom);
                            cells.push(cell_text);
                        }
                    }
                }
                if !cells.is_empty() {
                    row_blocks.push(ContentBlock {
                        kind: BlockKind::TableRow,
                        text: cells.join(" | "),
                        depth: tr_node.depth,
                        relevance: 0.6,
                        children: Vec::new(),
                        image_data: None, node_id: Some(*tr_id), computed_style: None,
                    });
                }
            }
        }

        if row_blocks.is_empty() {
            return None;
        }

        // Build the full table text as a preview
        let table_text: String = row_blocks
            .iter()
            .map(|r| r.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        Some(ContentBlock {
            kind: BlockKind::Table,
            text: table_text,
            depth: table_node.depth,
            relevance: 0.7,
            children: row_blocks,
            image_data: None, node_id: Some(table_node.id), computed_style: None,
        })
    }

    /// Extract definition list (dl/dt/dd).
    fn extract_definition_list(&self, dl_node: &DomNode, dom: &DomTree) -> Option<ContentBlock> {
        let mut children = Vec::new();

        for &child_id in &dl_node.children {
            if let Some(child) = dom.nodes.get(child_id) {
                match child.tag.as_str() {
                    "dt" => {
                        let text = self.collect_text_recursive(child, dom);
                        if !text.is_empty() {
                            children.push(ContentBlock {
                                kind: BlockKind::DefinitionTerm,
                                text,
                                depth: child.depth,
                                relevance: 0.7,
                                children: Vec::new(),
                                image_data: None, node_id: Some(child_id), computed_style: None,
                            });
                        }
                    }
                    "dd" => {
                        let text = self.collect_text_recursive(child, dom);
                        if !text.is_empty() {
                            children.push(ContentBlock {
                                kind: BlockKind::DefinitionDesc,
                                text,
                                depth: child.depth,
                                relevance: 0.6,
                                children: Vec::new(),
                                image_data: None, node_id: Some(child_id), computed_style: None,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }

        if children.is_empty() {
            return None;
        }

        Some(ContentBlock {
            kind: BlockKind::DefinitionList,
            text: String::new(),
            depth: dl_node.depth,
            relevance: 0.6,
            children,
            image_data: None, node_id: Some(dl_node.id), computed_style: None,
        })
    }

    /// Extract figure with optional figcaption.
    fn extract_figure(&self, fig_node: &DomNode, dom: &DomTree) -> Option<ContentBlock> {
        let mut children = Vec::new();
        let mut caption = String::new();

        for &child_id in &fig_node.children {
            if let Some(child) = dom.nodes.get(child_id) {
                match child.tag.as_str() {
                    "img" => {
                        children.push(ContentBlock {
                            kind: BlockKind::Image {
                                src: child.attrs.get("src").cloned().unwrap_or_default(),
                                alt: child.attrs.get("alt").cloned().unwrap_or_default(),
                            },
                            text: child.attrs.get("alt").cloned().unwrap_or_default(),
                            depth: child.depth,
                            relevance: 0.7,
                            children: Vec::new(),
                            image_data: None, node_id: Some(child_id), computed_style: None,
                        });
                    }
                    "figcaption" => {
                        caption = self.collect_text_recursive(child, dom);
                        children.push(ContentBlock {
                            kind: BlockKind::FigCaption,
                            text: caption.clone(),
                            depth: child.depth,
                            relevance: 0.7,
                            children: Vec::new(),
                            image_data: None, node_id: Some(child_id), computed_style: None,
                        });
                    }
                    _ => {}
                }
            }
        }

        if children.is_empty() {
            return None;
        }

        Some(ContentBlock {
            kind: BlockKind::Figure,
            text: caption,
            depth: fig_node.depth,
            relevance: 0.7,
            children,
            image_data: None, node_id: Some(fig_node.id), computed_style: None,
        })
    }

    /// Extract details/summary collapsible section.
    fn extract_details(&self, details_node: &DomNode, dom: &DomTree) -> Option<ContentBlock> {
        let open = details_node.attrs.contains_key("open");
        let mut children = Vec::new();
        let mut summary_text = String::new();

        for &child_id in &details_node.children {
            if let Some(child) = dom.nodes.get(child_id) {
                if child.tag == "summary" {
                    summary_text = self.collect_text_recursive(child, dom);
                    children.push(ContentBlock {
                        kind: BlockKind::Summary,
                        text: summary_text.clone(),
                        depth: child.depth,
                        relevance: 0.7,
                        children: Vec::new(),
                        image_data: None, node_id: Some(child_id), computed_style: None,
                    });
                } else {
                    // Content inside details
                    self.extract_node_recursive(child, dom, &mut children);
                }
            }
        }

        if summary_text.is_empty() && children.is_empty() {
            return None;
        }

        Some(ContentBlock {
            kind: BlockKind::Details { open },
            text: summary_text,
            depth: details_node.depth,
            relevance: 0.5,
            children,
            image_data: None, node_id: Some(details_node.id), computed_style: None,
        })
    }

    /// Extract form as a text description.
    fn extract_form(&self, form_node: &DomNode, dom: &DomTree) -> Option<ContentBlock> {
        let mut parts: Vec<String> = Vec::new();
        let action = form_node.attrs.get("action").cloned().unwrap_or_default();

        self.collect_form_fields(form_node, dom, &mut parts);

        if parts.is_empty() {
            return None;
        }

        let text = if action.is_empty() {
            format!("[Form] {}", parts.join(", "))
        } else {
            format!("[Form -> {}] {}", action, parts.join(", "))
        };

        Some(ContentBlock {
            kind: BlockKind::Form,
            text,
            depth: form_node.depth,
            relevance: 0.4,
            children: Vec::new(),
            image_data: None, node_id: Some(form_node.id), computed_style: None,
        })
    }

    /// Recursively collect form field descriptions.
    fn collect_form_fields(&self, node: &DomNode, dom: &DomTree, parts: &mut Vec<String>) {
        match node.tag.as_str() {
            "input" => {
                let input_type = node.attrs.get("type").map(|s| s.as_str()).unwrap_or("text");
                let name = node.attrs.get("name").or(node.attrs.get("placeholder"));
                let label = name.map(|s| s.as_str()).unwrap_or("field");
                match input_type {
                    "submit" => {
                        let value = node.attrs.get("value").map(|s| s.as_str()).unwrap_or("Submit");
                        parts.push(format!("[Button: {}]", value));
                    }
                    "hidden" => {} // skip hidden fields
                    _ => {
                        parts.push(format!("[{}: {}]", input_type, label));
                    }
                }
            }
            "textarea" => {
                let name = node.attrs.get("name").map(|s| s.as_str()).unwrap_or("text area");
                parts.push(format!("[textarea: {}]", name));
            }
            "select" => {
                let name = node.attrs.get("name").map(|s| s.as_str()).unwrap_or("select");
                parts.push(format!("[select: {}]", name));
            }
            "button" => {
                let text = if node.text.is_empty() { "Button" } else { &node.text };
                parts.push(format!("[Button: {}]", text));
            }
            "label" => {
                if !node.text.is_empty() {
                    parts.push(node.text.clone());
                }
            }
            _ => {}
        }

        // Recurse into children
        for &child_id in &node.children {
            if let Some(child) = dom.nodes.get(child_id) {
                self.collect_form_fields(child, dom, parts);
            }
        }
    }

    /// Collect all descendant node IDs with a given tag.
    fn collect_descendant_tags(
        &self,
        node: &DomNode,
        dom: &DomTree,
        tag: &str,
    ) -> Vec<usize> {
        let mut result = Vec::new();
        let mut stack: Vec<usize> = node.children.clone();

        while let Some(id) = stack.pop() {
            if let Some(n) = dom.nodes.get(id) {
                if n.tag == tag {
                    result.push(id);
                }
                // Continue searching into children (e.g., tr inside tbody)
                for &child_id in n.children.iter().rev() {
                    stack.push(child_id);
                }
            }
        }
        result
    }

    /// Recursively collect all text from a node and its children.
    fn collect_text_recursive(&self, node: &DomNode, dom: &DomTree) -> String {
        let mut text = node.text.clone();
        for &child_id in &node.children {
            if let Some(child) = dom.nodes.get(child_id) {
                let child_text = self.collect_text_recursive(child, dom);
                if !child_text.is_empty() {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(&child_text);
                }
            }
        }
        // Only allocate new string if trimming actually removes something
        let trimmed = text.trim();
        if trimmed.len() == text.len() {
            text
        } else {
            trimmed.to_string()
        }
    }

    /// Find the main content area of the page.
    fn find_body_content<'a>(&self, dom: &'a DomTree) -> Vec<&'a DomNode> {
        // Priority: <main>, <article>, <div role="main">, then <body>
        let main_nodes = dom.by_tag("main");
        if !main_nodes.is_empty() {
            return self.collect_children(dom, main_nodes[0].id);
        }

        let article_nodes = dom.by_tag("article");
        if !article_nodes.is_empty() {
            return self.collect_children(dom, article_nodes[0].id);
        }

        // Check for div with role="main"
        for node in &dom.nodes {
            if node.tag == "div" {
                if let Some(role) = node.attrs.get("role") {
                    if role == "main" {
                        return self.collect_children(dom, node.id);
                    }
                }
            }
        }

        // Fallback: all nodes that look like content
        dom.nodes
            .iter()
            .filter(|n| self.is_content_tag(&n.tag))
            .collect()
    }

    /// Collect all descendant nodes of a given node.
    fn collect_children<'a>(&self, dom: &'a DomTree, parent_id: usize) -> Vec<&'a DomNode> {
        let mut result = Vec::new();
        let mut stack = vec![parent_id];

        while let Some(id) = stack.pop() {
            if let Some(node) = dom.nodes.get(id) {
                if self.is_content_tag(&node.tag) || !node.text.is_empty() {
                    result.push(node);
                }
                for &child_id in node.children.iter().rev() {
                    stack.push(child_id);
                }
            }
        }
        result
    }

    fn is_content_tag(&self, tag: &str) -> bool {
        matches!(
            tag,
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
                | "p" | "pre" | "code" | "blockquote"
                | "ul" | "ol" | "li"
                | "img" | "figure" | "figcaption"
                | "table" | "dl" | "dt" | "dd"
                | "details" | "summary"
                | "form" | "input" | "textarea" | "select" | "button"
                | "hr" | "a"
        )
    }

    /// Convert a single DOM node to a ContentBlock.
    fn node_to_block(&self, node: &DomNode, dom: &DomTree) -> Option<ContentBlock> {
        // Skip hidden elements and elements with hidden ancestors
        if Self::is_hidden(node) {
            return None;
        }
        // Check if any ancestor is hidden
        let mut parent_id = node.parent;
        while let Some(pid) = parent_id {
            if let Some(parent) = dom.nodes.get(pid) {
                if Self::is_hidden(parent) {
                    return None;
                }
                parent_id = parent.parent;
            } else {
                break;
            }
        }

        let kind = match node.tag.as_str() {
            "h1" => BlockKind::Heading { level: 1 },
            "h2" => BlockKind::Heading { level: 2 },
            "h3" => BlockKind::Heading { level: 3 },
            "h4" => BlockKind::Heading { level: 4 },
            "h5" => BlockKind::Heading { level: 5 },
            "h6" => BlockKind::Heading { level: 6 },
            "p" => BlockKind::Paragraph,
            "pre" | "code" => BlockKind::Code {
                language: node.attrs.get("class")
                    .and_then(|c| c.strip_prefix("language-"))
                    .map(|s| s.to_string()),
            },
            "blockquote" => BlockKind::Quote,
            // ul/ol/li are handled by extract_node_recursive, but keep fallback
            "li" => BlockKind::ListItem,
            "img" => BlockKind::Image {
                src: node.attrs.get("src").cloned().unwrap_or_default(),
                alt: node.attrs.get("alt").cloned().unwrap_or_default(),
            },
            "a" => {
                let href = node.attrs.get("href").cloned().unwrap_or_default();
                // If <a> has no text, check for child <img> and use alt text
                if node.text.is_empty() {
                    let mut img_alt = None;
                    for &child_id in &node.children {
                        if let Some(child) = dom.nodes.get(child_id) {
                            if child.tag == "img" {
                                img_alt = Some(child.attrs.get("alt")
                                    .cloned()
                                    .unwrap_or_else(|| "[image link]".to_string()));
                                break;
                            }
                        }
                    }
                    if let Some(alt) = img_alt {
                        return Some(ContentBlock {
                            kind: BlockKind::Link { href },
                            text: alt,
                            depth: node.depth,
                            relevance: 0.5,
                            children: Vec::new(),
                            image_data: None, node_id: Some(node.id), computed_style: None,
                        });
                    }
                }
                BlockKind::Link { href }
            }
            "hr" => BlockKind::Separator,
            "nav" => BlockKind::Navigation,
            "footer" => BlockKind::Boilerplate,
            _ => return None,
        };

        // Skip empty text blocks (except images, separators, navigation markers)
        if node.text.is_empty() {
            match &kind {
                BlockKind::Image { .. } | BlockKind::Separator
                | BlockKind::Navigation | BlockKind::Boilerplate => {}
                _ => return None,
            }
        }

        Some(ContentBlock {
            kind,
            text: node.text.clone(),
            depth: node.depth,
            relevance: 0.5, // default, will be scored
            children: Vec::new(),
            image_data: None, node_id: Some(node.id), computed_style: None,
        })
    }

    /// Score each block's relevance using text density heuristic.
    /// Boosts content near headings, penalizes repeated nav patterns.
    #[allow(unused_assignments)] // score is always written before read via match arms
    fn score_relevance(&self, blocks: &mut [ContentBlock]) {
        let total_text: usize = blocks.iter().map(|b| b.text.len()).sum();
        if total_text == 0 {
            return;
        }

        // Detect repeated nav-like patterns (short text blocks appearing many times)
        let mut text_frequency: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for block in blocks.iter() {
            if block.text.len() < 40 && !block.text.is_empty() {
                let key = block.text.to_lowercase();
                *text_frequency.entry(key).or_insert(0) += 1;
            }
        }

        // Track proximity to headings: blocks right after headings get a boost
        let mut after_heading = false;
        let mut heading_boost_remaining: u8 = 0;

        for block in blocks.iter_mut() {
            let mut score: f32 = 0.5;

            match &block.kind {
                BlockKind::Title => score = 1.0,
                BlockKind::Heading { level } => {
                    score = 1.0 - (*level as f32 * 0.1);
                    after_heading = true;
                    heading_boost_remaining = 3; // Boost next 3 blocks after heading
                }
                BlockKind::Paragraph => {
                    // Longer paragraphs are more likely to be content
                    let len = block.text.len() as f32;
                    score = (len / 200.0).min(1.0).max(0.3);

                    // Boost if near a heading
                    if after_heading && heading_boost_remaining > 0 {
                        score = (score + 0.15).min(1.0);
                    }
                }
                BlockKind::Code { .. } => score = 0.8,
                BlockKind::Quote => score = 0.7,
                BlockKind::Image { alt, .. } => {
                    score = if alt.is_empty() { 0.4 } else { 0.7 };
                }
                BlockKind::Table => score = 0.7,
                BlockKind::TableRow => score = 0.6,
                BlockKind::DefinitionList => score = 0.6,
                BlockKind::DefinitionTerm => score = 0.7,
                BlockKind::DefinitionDesc => score = 0.6,
                BlockKind::Figure => score = 0.7,
                BlockKind::FigCaption => score = 0.7,
                BlockKind::Details { .. } => score = 0.5,
                BlockKind::Summary => score = 0.6,
                BlockKind::Form => score = 0.4,
                BlockKind::Navigation | BlockKind::Boilerplate => score = 0.1,
                BlockKind::List { .. } => score = 0.6,
                BlockKind::ListItem => {
                    score = 0.6;
                    if after_heading && heading_boost_remaining > 0 {
                        score = (score + 0.1).min(1.0);
                    }
                }
                BlockKind::Link { .. } => score = 0.4,
                BlockKind::Separator => score = 0.3,
            }

            // Penalize repeated short text (likely nav items)
            if block.text.len() < 40 && !block.text.is_empty() {
                let key = block.text.to_lowercase();
                if let Some(&freq) = text_frequency.get(&key) {
                    if freq >= 3 {
                        score *= 0.5; // Heavily penalize text that appears 3+ times
                    }
                }
            }

            block.relevance = score;

            // Decrement heading boost counter
            if heading_boost_remaining > 0 && !matches!(block.kind, BlockKind::Heading { .. }) {
                heading_boost_remaining -= 1;
                if heading_boost_remaining == 0 {
                    after_heading = false;
                }
            }

            // Also score children recursively
            self.score_children(&mut block.children, after_heading, heading_boost_remaining);
        }
    }

    /// Score children blocks (for nested lists, tables, etc.)
    #[allow(unused_assignments)] // score is always written before read via match arms
    fn score_children(&self, children: &mut [ContentBlock], near_heading: bool, boost: u8) {
        for child in children.iter_mut() {
            let mut score: f32 = 0.5;
            match &child.kind {
                BlockKind::ListItem => {
                    let len = child.text.len() as f32;
                    score = (len / 150.0).min(0.8).max(0.4);
                    if near_heading && boost > 0 {
                        score = (score + 0.1).min(1.0);
                    }
                }
                BlockKind::TableRow => score = 0.6,
                BlockKind::DefinitionTerm => score = 0.7,
                BlockKind::DefinitionDesc => score = 0.6,
                BlockKind::FigCaption => score = 0.7,
                BlockKind::Summary => score = 0.6,
                _ => {}
            }
            child.relevance = score;
            self.score_children(&mut child.children, near_heading, boost);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::dom::parse_html;

    fn extract(html: &str) -> Vec<ContentBlock> {
        let dom = parse_html(html);
        let extractor = ContentExtractor::new().unwrap();
        extractor.extract(&dom).unwrap()
    }

    // ── Block extraction tests ──

    #[test]
    fn test_extract_title() {
        let blocks = extract("<html><head><title>My Page</title></head><body></body></html>");
        assert!(blocks.iter().any(|b| matches!(b.kind, BlockKind::Title) && b.text == "My Page"));
    }

    #[test]
    fn test_extract_heading() {
        let blocks = extract("<h1>Hello World</h1>");
        assert!(blocks.iter().any(|b| matches!(b.kind, BlockKind::Heading { level: 1 }) && b.text == "Hello World"));
    }

    #[test]
    fn test_extract_multiple_headings() {
        let blocks = extract("<h1>First</h1><h2>Second</h2><h3>Third</h3>");
        let headings: Vec<_> = blocks.iter().filter(|b| matches!(b.kind, BlockKind::Heading { .. })).collect();
        assert_eq!(headings.len(), 3);
    }

    #[test]
    fn test_extract_paragraph() {
        let blocks = extract("<p>This is a paragraph of text.</p>");
        assert!(blocks.iter().any(|b| matches!(b.kind, BlockKind::Paragraph)));
    }

    #[test]
    fn test_extract_image() {
        let blocks = extract(r#"<img src="photo.jpg" alt="A photo">"#);
        assert!(blocks.iter().any(|b| matches!(&b.kind, BlockKind::Image { src, alt } if src == "photo.jpg" && alt == "A photo")));
    }

    #[test]
    fn test_extract_link() {
        let blocks = extract(r#"<a href="https://example.com">Example</a>"#);
        assert!(blocks.iter().any(|b| matches!(&b.kind, BlockKind::Link { href } if href == "https://example.com")));
    }

    #[test]
    fn test_extract_code_block() {
        let blocks = extract(r#"<pre class="language-rust">fn main() {}</pre>"#);
        assert!(blocks.iter().any(|b| matches!(&b.kind, BlockKind::Code { language } if language.as_deref() == Some("rust"))));
    }

    #[test]
    fn test_extract_separator() {
        let blocks = extract("<p>Before</p><hr><p>After</p>");
        assert!(blocks.iter().any(|b| matches!(b.kind, BlockKind::Separator)));
    }

    #[test]
    fn test_extract_empty_dom() {
        let blocks = extract("");
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_empty_paragraph_skipped() {
        let blocks = extract("<p></p><p>Content</p>");
        let paras: Vec<_> = blocks.iter().filter(|b| matches!(b.kind, BlockKind::Paragraph)).collect();
        assert_eq!(paras.len(), 1);
        assert_eq!(paras[0].text, "Content");
    }

    // ── Main content detection tests ──

    #[test]
    fn test_main_tag_prioritized() {
        let blocks = extract(r#"
            <nav><a href="/">Home</a></nav>
            <main><h1>Main Content</h1><p>Important text here.</p></main>
            <footer>Footer text</footer>
        "#);
        // Should find the heading from main
        assert!(blocks.iter().any(|b| matches!(b.kind, BlockKind::Heading { level: 1 })));
    }

    #[test]
    fn test_article_tag_prioritized() {
        let blocks = extract(r#"
            <nav><a href="/">Home</a></nav>
            <article><h2>Article Title</h2><p>Article content.</p></article>
        "#);
        assert!(blocks.iter().any(|b| matches!(b.kind, BlockKind::Heading { level: 2 })));
    }

    // ── Relevance scoring tests ──

    #[test]
    fn test_title_high_relevance() {
        let blocks = extract("<title>Page Title</title>");
        let title = blocks.iter().find(|b| matches!(b.kind, BlockKind::Title));
        assert!(title.is_some());
        assert_eq!(title.unwrap().relevance, 1.0);
    }

    #[test]
    fn test_heading_higher_relevance_than_paragraph() {
        let blocks = extract("<h1>Heading</h1><p>Short text.</p>");
        let h1 = blocks.iter().find(|b| matches!(b.kind, BlockKind::Heading { level: 1 }));
        let p = blocks.iter().find(|b| matches!(b.kind, BlockKind::Paragraph));
        if let (Some(h), Some(para)) = (h1, p) {
            assert!(h.relevance > para.relevance);
        }
    }

    #[test]
    fn test_longer_paragraph_higher_relevance() {
        let short = "Short.";
        let long = "This is a much longer paragraph that contains a lot of text and should score higher in relevance because longer paragraphs are more likely to be actual content.";
        let html = format!("<p>{short}</p><p>{long}</p>");
        let blocks = extract(&html);
        let paras: Vec<_> = blocks.iter().filter(|b| matches!(b.kind, BlockKind::Paragraph)).collect();
        if paras.len() == 2 {
            assert!(paras[1].relevance > paras[0].relevance);
        }
    }

    #[test]
    fn test_blockquote_extraction() {
        let blocks = extract("<blockquote>A wise quote.</blockquote>");
        assert!(blocks.iter().any(|b| matches!(b.kind, BlockKind::Quote)));
    }

    #[test]
    fn test_list_extraction() {
        let blocks = extract("<ul><li>Item A</li><li>Item B</li></ul>");
        assert!(blocks.iter().any(|b| matches!(b.kind, BlockKind::List { ordered: false })));
    }

    #[test]
    fn test_ordered_list_extraction() {
        let blocks = extract("<ol><li>First</li><li>Second</li></ol>");
        assert!(blocks.iter().any(|b| matches!(b.kind, BlockKind::List { ordered: true })));
    }

    // ── Hidden element filtering tests ──

    #[test]
    fn test_hidden_attribute_filtered() {
        let blocks = extract(r#"<p>Visible</p><p hidden>Hidden</p>"#);
        let paras: Vec<_> = blocks.iter().filter(|b| matches!(b.kind, BlockKind::Paragraph)).collect();
        assert_eq!(paras.len(), 1);
        assert_eq!(paras[0].text, "Visible");
    }

    #[test]
    fn test_display_none_filtered() {
        let blocks = extract(r#"<p>Visible</p><p style="display:none">Hidden</p>"#);
        let paras: Vec<_> = blocks.iter().filter(|b| matches!(b.kind, BlockKind::Paragraph)).collect();
        assert_eq!(paras.len(), 1);
        assert_eq!(paras[0].text, "Visible");
    }

    #[test]
    fn test_visibility_hidden_filtered() {
        let blocks = extract(r#"<p>Visible</p><div style="visibility: hidden"><p>Hidden</p></div>"#);
        // The div with visibility:hidden should be skipped, so Hidden para won't appear
        let texts: Vec<_> = blocks.iter()
            .filter(|b| matches!(b.kind, BlockKind::Paragraph))
            .map(|b| b.text.as_str())
            .collect();
        assert!(texts.contains(&"Visible"));
        assert!(!texts.contains(&"Hidden"));
    }

    #[test]
    fn test_aria_hidden_filtered() {
        let blocks = extract(r#"<p>Visible</p><p aria-hidden="true">Screen reader only</p>"#);
        let paras: Vec<_> = blocks.iter().filter(|b| matches!(b.kind, BlockKind::Paragraph)).collect();
        assert_eq!(paras.len(), 1);
        assert_eq!(paras[0].text, "Visible");
    }

    // ── Link resolution bug fix tests ──

    #[test]
    fn test_link_inside_list_item_preserved() {
        let blocks = extract(r#"<ul><li><a href="https://example.com">Example Link</a></li></ul>"#);
        let list = blocks.iter().find(|b| matches!(b.kind, BlockKind::List { .. }));
        assert!(list.is_some(), "Should find a list block");
        let list = list.unwrap();
        assert!(!list.children.is_empty(), "List should have children");
        let li = &list.children[0];
        assert_eq!(li.text, "Example Link");
        // The li should have a child Link block with the href
        let has_link = li.children.iter().any(|c| {
            matches!(&c.kind, BlockKind::Link { href } if href == "https://example.com")
        });
        assert!(has_link, "List item should preserve child link with href");
    }

    #[test]
    fn test_anchor_wrapping_img_uses_alt() {
        let blocks = extract(r#"<a href="/home"><img src="logo.png" alt="Home Logo"></a>"#);
        let link = blocks.iter().find(|b| matches!(&b.kind, BlockKind::Link { .. }));
        assert!(link.is_some(), "Should create a Link block from <a> wrapping <img>");
        let link = link.unwrap();
        assert_eq!(link.text, "Home Logo");
        if let BlockKind::Link { href } = &link.kind {
            assert_eq!(href, "/home");
        }
    }

    #[test]
    fn test_anchor_wrapping_img_no_alt_fallback() {
        let blocks = extract(r#"<a href="/home"><img src="logo.png"></a>"#);
        let link = blocks.iter().find(|b| matches!(&b.kind, BlockKind::Link { .. }));
        assert!(link.is_some(), "Should create Link with fallback text for img without alt");
        let link = link.unwrap();
        assert_eq!(link.text, "[image link]");
    }
}
