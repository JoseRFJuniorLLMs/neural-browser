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
        // TODO: Load ONNX model for content classification
        // let session = ort::Session::builder()?
        //     .with_execution_providers([ort::DirectMLExecutionProvider::default().build()])?
        //     .commit_from_file("models/content_extractor.onnx")?;
        Ok(Self {})
    }

    /// Extract semantic content blocks from DOM.
    pub fn extract(&self, dom: &DomTree) -> Result<Vec<ContentBlock>> {
        let mut blocks = Vec::new();

        // Extract title
        if let Some(title) = dom.by_tag("title").first() {
            blocks.push(ContentBlock {
                kind: BlockKind::Title,
                text: title.text.clone(),
                depth: 0,
                relevance: 1.0,
                children: Vec::new(),
            });
        }

        // Walk body content
        let body_nodes = self.find_body_content(dom);
        for node in &body_nodes {
            if let Some(block) = self.node_to_block(node, dom) {
                blocks.push(block);
            }
        }

        // Score relevance (text density heuristic)
        self.score_relevance(&mut blocks);

        debug!("[NPU:CONTENT] Extracted {} blocks", blocks.len());
        Ok(blocks)
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
                | "table" | "hr" | "a"
        )
    }

    /// Convert a single DOM node to a ContentBlock.
    fn node_to_block(&self, node: &DomNode, _dom: &DomTree) -> Option<ContentBlock> {
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
            "ul" => BlockKind::List { ordered: false },
            "ol" => BlockKind::List { ordered: true },
            "li" => BlockKind::ListItem,
            "img" => BlockKind::Image {
                src: node.attrs.get("src").cloned().unwrap_or_default(),
                alt: node.attrs.get("alt").cloned().unwrap_or_default(),
            },
            "a" => BlockKind::Link {
                href: node.attrs.get("href").cloned().unwrap_or_default(),
            },
            "table" => BlockKind::Table,
            "hr" => BlockKind::Separator,
            "nav" => BlockKind::Navigation,
            "footer" => BlockKind::Boilerplate,
            _ => return None,
        };

        // Skip empty text blocks (except images and separators)
        if node.text.is_empty() {
            match &kind {
                BlockKind::Image { .. } | BlockKind::Separator => {}
                _ => return None,
            }
        }

        Some(ContentBlock {
            kind,
            text: node.text.clone(),
            depth: node.depth,
            relevance: 0.5, // default, will be scored
            children: Vec::new(),
        })
    }

    /// Score each block's relevance using text density heuristic.
    /// (Phase 2: Replace with ONNX model output)
    fn score_relevance(&self, blocks: &mut [ContentBlock]) {
        let total_text: usize = blocks.iter().map(|b| b.text.len()).sum();
        if total_text == 0 {
            return;
        }

        for block in blocks.iter_mut() {
            let mut score: f32 = 0.5;

            match &block.kind {
                BlockKind::Title => score = 1.0,
                BlockKind::Heading { level } => score = 1.0 - (*level as f32 * 0.1),
                BlockKind::Paragraph => {
                    // Longer paragraphs are more likely to be content
                    let len = block.text.len() as f32;
                    score = (len / 200.0).min(1.0).max(0.3);
                }
                BlockKind::Code { .. } => score = 0.8,
                BlockKind::Quote => score = 0.7,
                BlockKind::Image { alt, .. } => {
                    score = if alt.is_empty() { 0.4 } else { 0.7 };
                }
                BlockKind::Navigation | BlockKind::Boilerplate => score = 0.1,
                _ => {}
            }

            block.relevance = score;
        }
    }
}
