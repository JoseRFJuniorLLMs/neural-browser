//! NPU: Ad/tracker classifier.
//!
//! ML-based ad detection — no filter lists needed.
//! The NPU model classifies DOM elements as ad/tracker/content.
//!
//! Phase 1: Heuristic rules (class names, URLs, patterns)
//! Phase 2: ONNX model trained on EasyList + EasyPrivacy datasets

use crate::npu::ContentBlock;

/// Ad/tracker classifier.
pub struct AdClassifier {
    // Future: ort::Session for ONNX ad classification model
}

impl AdClassifier {
    pub fn new() -> anyhow::Result<Self> {
        // TODO: Load ONNX model
        // let session = ort::Session::builder()?
        //     .with_execution_providers([ort::DirectMLExecutionProvider::default().build()])?
        //     .commit_from_file("models/ad_classifier.onnx")?;
        Ok(Self {})
    }

    /// Classify a content block as ad/tracker.
    /// Returns true if the block should be blocked.
    pub fn is_ad(&self, block: &ContentBlock) -> bool {
        // Phase 1: Heuristic detection (replace with ONNX model)
        let text_lower = block.text.to_lowercase();

        // Check image sources for known ad patterns
        if let crate::npu::BlockKind::Image { src, .. } = &block.kind {
            let src_lower = src.to_lowercase();
            if AD_URL_PATTERNS.iter().any(|p| src_lower.contains(p)) {
                return true;
            }
        }

        // Check link hrefs for tracking
        if let crate::npu::BlockKind::Link { href } = &block.kind {
            let href_lower = href.to_lowercase();
            if TRACKER_PATTERNS.iter().any(|p| href_lower.contains(p)) {
                return true;
            }
        }

        // Check text for ad markers
        if AD_TEXT_PATTERNS.iter().any(|p| text_lower.contains(p)) {
            return true;
        }

        false
    }
}

const AD_URL_PATTERNS: &[&str] = &[
    "doubleclick",
    "googlesyndication",
    "googleadservices",
    "adsystem",
    "adservice",
    "adnxs",
    "advertising",
    "taboola",
    "outbrain",
    "criteo",
    "/ads/",
    "/ad/",
    "banner",
    "popunder",
];

const TRACKER_PATTERNS: &[&str] = &[
    "analytics",
    "tracking",
    "tracker",
    "pixel",
    "beacon",
    "telemetry",
    "collect?",
    "utm_",
    "fbclid",
    "gclid",
];

const AD_TEXT_PATTERNS: &[&str] = &[
    "advertisement",
    "sponsored content",
    "promoted",
    "ad choice",
    "adchoice",
];
