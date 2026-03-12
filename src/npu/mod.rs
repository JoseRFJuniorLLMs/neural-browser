//! NPU Engine — AI inference via ONNX Runtime + DirectML.
//!
//! All "understanding" happens here:
//! - Content extraction (what is the main content vs boilerplate?)
//! - Ad/tracker detection (ML-based, not filter lists)
//! - Text summarization
//! - Link relevance scoring (for smart prefetch)
//! - Language detection
//!
//! The NPU does ZERO rendering — it produces semantic ContentBlocks
//! that the GPU renderer turns into pixels.

mod content;
mod classifier;

pub use content::{ContentBlock, BlockKind};

use crate::cpu::dom::DomTree;
use anyhow::Result;
use log::info;

/// Result of NPU processing a page.
pub struct NpuResult {
    pub blocks: Vec<ContentBlock>,
    pub ads_blocked: usize,
    pub summary: Option<String>,
    pub language: Option<String>,
    pub prefetch_urls: Vec<String>,
}

/// NPU inference engine — wraps ONNX Runtime with DirectML.
pub struct NpuEngine {
    content_extractor: content::ContentExtractor,
    ad_classifier: classifier::AdClassifier,
}

impl NpuEngine {
    pub fn new() -> Result<Self> {
        info!("[NPU] Initializing ONNX Runtime with DirectML...");

        let content_extractor = content::ContentExtractor::new()?;
        let ad_classifier = classifier::AdClassifier::new()?;

        info!("[NPU] Engine ready — models loaded");
        Ok(Self {
            content_extractor,
            ad_classifier,
        })
    }

    /// Process a fetched page through the NPU pipeline.
    pub fn process_page(
        &mut self,
        url: &str,
        _html: &str,
        dom: &DomTree,
    ) -> Result<NpuResult> {
        // ── Step 1: Extract content blocks from DOM ──
        let mut blocks = self.content_extractor.extract(dom)?;

        // ── Step 2: Classify and filter ads/trackers ──
        let total_before = blocks.len();
        blocks.retain(|b| !self.ad_classifier.is_ad(b));
        let ads_blocked = total_before - blocks.len();

        if ads_blocked > 0 {
            info!("[NPU] Blocked {ads_blocked} ad/tracker elements");
        }

        // ── Step 3: Score links for smart prefetch ──
        let prefetch_urls = self.score_links_for_prefetch(dom, url);

        // ── Step 4: Detect language (from text blocks) ──
        let language = self.detect_language(&blocks);

        info!(
            "[NPU] Extracted {} content blocks, {} prefetch candidates",
            blocks.len(),
            prefetch_urls.len()
        );

        Ok(NpuResult {
            blocks,
            ads_blocked,
            summary: None, // TODO: ONNX summarization model
            language,
            prefetch_urls,
        })
    }

    /// Score links by likelihood of user clicking them.
    /// High-scoring links get prefetched by CPU.
    fn score_links_for_prefetch(&self, dom: &DomTree, current_url: &str) -> Vec<String> {
        let links = dom.links();

        // Heuristic scoring (replace with ONNX model later):
        // - Navigation links (nav, header) score high
        // - "Read more", "Continue" patterns score high
        // - External domains score low
        // - Already-visited score zero
        let mut scored: Vec<(f32, String)> = Vec::new();

        for (href, text) in &links {
            let mut score: f32 = 0.0;

            // Same domain bonus
            if href.starts_with('/') || href.starts_with(current_url) {
                score += 0.3;
            }

            // Navigation patterns
            let text_lower = text.to_lowercase();
            if text_lower.contains("next")
                || text_lower.contains("continue")
                || text_lower.contains("read more")
            {
                score += 0.5;
            }

            // Skip anchors and javascript
            if href.starts_with('#') || href.starts_with("javascript:") {
                continue;
            }

            if score > 0.2 {
                scored.push((score, href.to_string()));
            }
        }

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.into_iter().take(3).map(|(_, url)| url).collect()
    }

    /// Simple language detection from content blocks.
    fn detect_language(&self, blocks: &[ContentBlock]) -> Option<String> {
        let text: String = blocks
            .iter()
            .filter(|b| matches!(b.kind, BlockKind::Paragraph | BlockKind::Heading { .. }))
            .take(5)
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        if text.len() < 20 {
            return None;
        }

        // Very basic heuristic (replace with ONNX model)
        let has_accents = text.chars().any(|c| "àáâãéêíóôõúçñü".contains(c));
        let has_common_pt = text.contains(" de ") || text.contains(" com ") || text.contains(" para ");
        let has_common_es = text.contains(" el ") || text.contains(" los ") || text.contains(" por ");
        let has_common_en = text.contains(" the ") || text.contains(" and ") || text.contains(" for ");

        if has_accents && has_common_pt {
            Some("pt".into())
        } else if has_accents && has_common_es {
            Some("es".into())
        } else if has_common_en {
            Some("en".into())
        } else {
            None
        }
    }
}
