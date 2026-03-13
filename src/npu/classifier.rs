//! NPU: Ad/tracker classifier.
//!
//! ML-based ad detection — no filter lists needed.
//! The NPU model classifies DOM elements as ad/tracker/content.
//!
//! Phase 1: Heuristic rules (class names, URLs, patterns)
//! Phase 2: ONNX model trained on EasyList + EasyPrivacy datasets

use crate::npu::ContentBlock;

/// Classification result with confidence score.
#[derive(Debug, Clone)]
pub struct Classification {
    /// What kind of unwanted content this is.
    pub kind: ClassificationKind,
    /// Confidence that this classification is correct (0.0-1.0).
    pub confidence: f32,
}

/// Types of unwanted content the classifier can detect.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassificationKind {
    /// Not unwanted — regular content.
    Content,
    /// Ad or sponsored content.
    Ad,
    /// Tracking pixel/beacon.
    Tracker,
    /// Cookie consent / GDPR popup.
    CookieConsent,
    /// Social media widget (share/like/follow buttons).
    SocialWidget,
    /// Newsletter signup / popup overlay.
    NewsletterPopup,
}

/// Ad/tracker classifier.
pub struct AdClassifier {
    // Future: ort::Session for ONNX ad classification model
}

impl AdClassifier {
    pub fn new() -> anyhow::Result<Self> {
        // TODO: Load ONNX model
        Ok(Self {})
    }

    /// Classify a content block — returns true if the block should be blocked.
    pub fn is_ad(&self, block: &ContentBlock) -> bool {
        let result = self.classify(block);
        result.kind != ClassificationKind::Content && result.confidence >= 0.5
    }

    /// Full classification with confidence score.
    pub fn classify(&self, block: &ContentBlock) -> Classification {
        let text_lower = block.text.to_lowercase();

        // ── Check image sources for known ad patterns ──
        if let crate::npu::BlockKind::Image { src, .. } = &block.kind {
            let src_lower = src.to_lowercase();
            if let Some(conf) = match_patterns(&src_lower, AD_URL_PATTERNS) {
                return Classification {
                    kind: ClassificationKind::Ad,
                    confidence: conf,
                };
            }
        }

        // ── Check link hrefs for tracking ──
        if let crate::npu::BlockKind::Link { href } = &block.kind {
            let href_lower = href.to_lowercase();
            if let Some(conf) = match_patterns(&href_lower, TRACKER_PATTERNS) {
                return Classification {
                    kind: ClassificationKind::Tracker,
                    confidence: conf,
                };
            }
            // Also check for ad links
            if let Some(conf) = match_patterns(&href_lower, AD_URL_PATTERNS) {
                return Classification {
                    kind: ClassificationKind::Ad,
                    confidence: conf,
                };
            }
        }

        // ── Cookie consent / GDPR popup detection ──
        if let Some(conf) = match_patterns(&text_lower, COOKIE_CONSENT_PATTERNS) {
            return Classification {
                kind: ClassificationKind::CookieConsent,
                confidence: conf,
            };
        }

        // ── Social media widget detection ──
        if let Some(conf) = match_patterns(&text_lower, SOCIAL_WIDGET_PATTERNS) {
            return Classification {
                kind: ClassificationKind::SocialWidget,
                confidence: conf,
            };
        }

        // ── Newsletter / popup detection ──
        if let Some(conf) = match_patterns(&text_lower, NEWSLETTER_POPUP_PATTERNS) {
            return Classification {
                kind: ClassificationKind::NewsletterPopup,
                confidence: conf,
            };
        }

        // ── Check text for ad markers ──
        if let Some(conf) = match_patterns(&text_lower, AD_TEXT_PATTERNS) {
            return Classification {
                kind: ClassificationKind::Ad,
                confidence: conf,
            };
        }

        Classification {
            kind: ClassificationKind::Content,
            confidence: 1.0,
        }
    }
}

/// Pattern with associated confidence weight.
struct PatternEntry {
    pattern: &'static str,
    confidence: f32,
}

/// Check text against a list of patterns; return highest confidence match.
fn match_patterns(text: &str, patterns: &[PatternEntry]) -> Option<f32> {
    let mut best: Option<f32> = None;
    for p in patterns {
        if text.contains(p.pattern) {
            let current = best.unwrap_or(0.0);
            if p.confidence > current {
                best = Some(p.confidence);
            }
        }
    }
    best
}

// ── Ad URL patterns ──

const AD_URL_PATTERNS: &[PatternEntry] = &[
    // Google ads
    PatternEntry { pattern: "doubleclick", confidence: 0.95 },
    PatternEntry { pattern: "googlesyndication", confidence: 0.95 },
    PatternEntry { pattern: "googleadservices", confidence: 0.95 },
    PatternEntry { pattern: "pagead", confidence: 0.85 },
    // Generic ad systems
    PatternEntry { pattern: "adsystem", confidence: 0.9 },
    PatternEntry { pattern: "adservice", confidence: 0.9 },
    PatternEntry { pattern: "adnxs", confidence: 0.9 },
    PatternEntry { pattern: "advertising", confidence: 0.8 },
    // Major ad networks
    PatternEntry { pattern: "amazon-adsystem", confidence: 0.95 },
    PatternEntry { pattern: "taboola", confidence: 0.95 },
    PatternEntry { pattern: "outbrain", confidence: 0.95 },
    PatternEntry { pattern: "criteo", confidence: 0.95 },
    PatternEntry { pattern: "mediavine", confidence: 0.9 },
    PatternEntry { pattern: "revcontent", confidence: 0.9 },
    PatternEntry { pattern: "mgid.com", confidence: 0.9 },
    PatternEntry { pattern: "zergnet", confidence: 0.85 },
    PatternEntry { pattern: "adblade", confidence: 0.9 },
    PatternEntry { pattern: "adroll", confidence: 0.9 },
    PatternEntry { pattern: "pubmatic", confidence: 0.9 },
    PatternEntry { pattern: "openx", confidence: 0.85 },
    PatternEntry { pattern: "rubiconproject", confidence: 0.9 },
    PatternEntry { pattern: "indexexchange", confidence: 0.85 },
    PatternEntry { pattern: "appnexus", confidence: 0.9 },
    PatternEntry { pattern: "smartadserver", confidence: 0.9 },
    PatternEntry { pattern: "bidswitch", confidence: 0.85 },
    PatternEntry { pattern: "sharethrough", confidence: 0.85 },
    PatternEntry { pattern: "moat.com", confidence: 0.8 },
    // URL path patterns
    PatternEntry { pattern: "/ads/", confidence: 0.8 },
    PatternEntry { pattern: "/ad/", confidence: 0.7 },
    PatternEntry { pattern: "/adserv", confidence: 0.85 },
    PatternEntry { pattern: "/banner-ad", confidence: 0.7 },
    PatternEntry { pattern: "popunder", confidence: 0.9 },
    PatternEntry { pattern: "pop-up", confidence: 0.7 },
];

// ── Tracker patterns ──

const TRACKER_PATTERNS: &[PatternEntry] = &[
    PatternEntry { pattern: "analytics", confidence: 0.7 },
    PatternEntry { pattern: "tracking", confidence: 0.8 },
    PatternEntry { pattern: "tracker", confidence: 0.8 },
    PatternEntry { pattern: "pixel", confidence: 0.75 },
    PatternEntry { pattern: "beacon", confidence: 0.8 },
    PatternEntry { pattern: "telemetry", confidence: 0.85 },
    PatternEntry { pattern: "collect?", confidence: 0.75 },
    PatternEntry { pattern: "utm_", confidence: 0.35 },  // Low: UTM links are often legitimate
    PatternEntry { pattern: "fbclid", confidence: 0.7 },
    PatternEntry { pattern: "gclid", confidence: 0.7 },
    PatternEntry { pattern: "hotjar", confidence: 0.9 },
    PatternEntry { pattern: "fullstory", confidence: 0.9 },
    PatternEntry { pattern: "mouseflow", confidence: 0.9 },
    PatternEntry { pattern: "segment.io", confidence: 0.85 },
    PatternEntry { pattern: "mixpanel", confidence: 0.85 },
    PatternEntry { pattern: "amplitude", confidence: 0.8 },
    PatternEntry { pattern: "newrelic", confidence: 0.75 },
    PatternEntry { pattern: "sentry.io", confidence: 0.5 },
];

// ── Ad text patterns ──

const AD_TEXT_PATTERNS: &[PatternEntry] = &[
    PatternEntry { pattern: "advertisement", confidence: 0.9 },
    PatternEntry { pattern: "sponsored content", confidence: 0.95 },
    PatternEntry { pattern: "sponsored post", confidence: 0.9 },
    PatternEntry { pattern: "promoted content", confidence: 0.7 },
    PatternEntry { pattern: "ad choice", confidence: 0.9 },
    PatternEntry { pattern: "adchoice", confidence: 0.9 },
    PatternEntry { pattern: "paid partnership", confidence: 0.85 },
    PatternEntry { pattern: "affiliate link", confidence: 0.75 },
];

// ── Cookie consent / GDPR popup patterns ──

const COOKIE_CONSENT_PATTERNS: &[PatternEntry] = &[
    PatternEntry { pattern: "accept all cookies", confidence: 0.95 },
    PatternEntry { pattern: "accept cookies", confidence: 0.9 },
    PatternEntry { pattern: "cookie settings", confidence: 0.85 },
    PatternEntry { pattern: "cookie preferences", confidence: 0.85 },
    PatternEntry { pattern: "cookie policy", confidence: 0.8 },
    PatternEntry { pattern: "cookie consent", confidence: 0.9 },
    PatternEntry { pattern: "we use cookies", confidence: 0.9 },
    PatternEntry { pattern: "this site uses cookies", confidence: 0.9 },
    PatternEntry { pattern: "gdpr", confidence: 0.7 },
    PatternEntry { pattern: "manage consent", confidence: 0.85 },
    PatternEntry { pattern: "consent manager", confidence: 0.85 },
    PatternEntry { pattern: "privacy preferences", confidence: 0.75 },
    PatternEntry { pattern: "reject all", confidence: 0.7 },
    PatternEntry { pattern: "accept all", confidence: 0.6 },
    PatternEntry { pattern: "necessary cookies", confidence: 0.85 },
];

// ── Social media widget patterns ──

const SOCIAL_WIDGET_PATTERNS: &[PatternEntry] = &[
    PatternEntry { pattern: "share on facebook", confidence: 0.9 },
    PatternEntry { pattern: "share on twitter", confidence: 0.9 },
    PatternEntry { pattern: "share on linkedin", confidence: 0.9 },
    PatternEntry { pattern: "share this", confidence: 0.75 },
    PatternEntry { pattern: "follow us on", confidence: 0.8 },
    PatternEntry { pattern: "follow us", confidence: 0.65 },
    PatternEntry { pattern: "like us on", confidence: 0.8 },
    PatternEntry { pattern: "tweet this", confidence: 0.85 },
    PatternEntry { pattern: "pin it", confidence: 0.7 },
    PatternEntry { pattern: "share via", confidence: 0.75 },
    PatternEntry { pattern: "social share", confidence: 0.85 },
    PatternEntry { pattern: "sharing buttons", confidence: 0.85 },
];

// ── Newsletter / popup patterns ──

const NEWSLETTER_POPUP_PATTERNS: &[PatternEntry] = &[
    PatternEntry { pattern: "subscribe to our newsletter", confidence: 0.9 },
    PatternEntry { pattern: "sign up for our newsletter", confidence: 0.9 },
    PatternEntry { pattern: "join our mailing list", confidence: 0.85 },
    PatternEntry { pattern: "get updates in your inbox", confidence: 0.85 },
    PatternEntry { pattern: "enter your email", confidence: 0.6 },
    PatternEntry { pattern: "subscribe now", confidence: 0.7 },
    PatternEntry { pattern: "don't miss out", confidence: 0.6 },
    PatternEntry { pattern: "stay updated", confidence: 0.55 },
    PatternEntry { pattern: "no thanks", confidence: 0.5 },
    PatternEntry { pattern: "maybe later", confidence: 0.5 },
    PatternEntry { pattern: "close this popup", confidence: 0.85 },
    PatternEntry { pattern: "exit intent", confidence: 0.9 },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npu::{ContentBlock, BlockKind};

    fn make_block(kind: BlockKind, text: &str) -> ContentBlock {
        ContentBlock {
            kind,
            text: text.to_string(),
            depth: 0,
            relevance: 0.5,
            children: Vec::new(),
            image_data: None, node_id: None, computed_style: None,
        }
    }

    // ── Ad detection tests ──

    #[test]
    fn test_normal_content_not_blocked() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(BlockKind::Paragraph, "This is a normal paragraph about Rust.");
        assert!(!classifier.is_ad(&block));
        let c = classifier.classify(&block);
        assert_eq!(c.kind, ClassificationKind::Content);
    }

    #[test]
    fn test_ad_text_detected() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(BlockKind::Paragraph, "This is an Advertisement for products");
        assert!(classifier.is_ad(&block));
        let c = classifier.classify(&block);
        assert_eq!(c.kind, ClassificationKind::Ad);
    }

    #[test]
    fn test_sponsored_content_detected() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(BlockKind::Paragraph, "Sponsored Content by Brand");
        assert!(classifier.is_ad(&block));
    }

    #[test]
    fn test_promoted_content_detected() {
        let classifier = AdClassifier::new().unwrap();
        // "promoted" alone has too many false positives; "promoted content" is specific
        let block = make_block(BlockKind::Paragraph, "This is promoted content by a brand");
        assert!(classifier.is_ad(&block));
        // But "promoted" alone should NOT trigger
        let block2 = make_block(BlockKind::Paragraph, "Recently promoted to manager");
        assert!(!classifier.is_ad(&block2));
    }

    // ── Ad image URL tests ──

    #[test]
    fn test_googlesyndication_image_detected() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(
            BlockKind::Image {
                src: "https://pagead2.googlesyndication.com/banner.jpg".into(),
                alt: "".into(),
            },
            "",
        );
        assert!(classifier.is_ad(&block));
    }

    #[test]
    fn test_doubleclick_image_detected() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(
            BlockKind::Image {
                src: "https://ad.doubleclick.net/pixel.gif".into(),
                alt: "".into(),
            },
            "",
        );
        assert!(classifier.is_ad(&block));
    }

    #[test]
    fn test_taboola_image_detected() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(
            BlockKind::Image {
                src: "https://cdn.taboola.com/libtrc/thumbnails/thumb.jpg".into(),
                alt: "".into(),
            },
            "",
        );
        assert!(classifier.is_ad(&block));
    }

    #[test]
    fn test_normal_image_not_blocked() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(
            BlockKind::Image {
                src: "https://example.com/photo.jpg".into(),
                alt: "A photo".into(),
            },
            "",
        );
        assert!(!classifier.is_ad(&block));
    }

    // ── Tracker link tests ──

    #[test]
    fn test_analytics_tracker_detected() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(
            BlockKind::Link {
                href: "https://analytics.example.com/collect?id=123".into(),
            },
            "Track",
        );
        assert!(classifier.is_ad(&block));
        let c = classifier.classify(&block);
        assert_eq!(c.kind, ClassificationKind::Tracker);
    }

    #[test]
    fn test_utm_tracker_low_confidence() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(
            BlockKind::Link {
                href: "https://example.com/page?utm_source=newsletter".into(),
            },
            "Click here",
        );
        // UTM parameters have low confidence (0.35) so they should NOT be blocked
        // (threshold is 0.5). UTM links are often legitimate content links.
        assert!(!classifier.is_ad(&block));
        let c = classifier.classify(&block);
        assert_eq!(c.kind, ClassificationKind::Tracker);
        assert!(c.confidence < 0.5);
    }

    #[test]
    fn test_fbclid_tracker_detected() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(
            BlockKind::Link {
                href: "https://example.com/page?fbclid=abc123".into(),
            },
            "Facebook link",
        );
        assert!(classifier.is_ad(&block));
    }

    #[test]
    fn test_normal_link_not_blocked() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(
            BlockKind::Link {
                href: "https://example.com/about".into(),
            },
            "About us",
        );
        assert!(!classifier.is_ad(&block));
    }

    // ── Cookie consent tests ──

    #[test]
    fn test_cookie_consent_detected() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(BlockKind::Paragraph, "We use cookies to improve your experience");
        let c = classifier.classify(&block);
        assert_eq!(c.kind, ClassificationKind::CookieConsent);
        assert!(classifier.is_ad(&block));
    }

    #[test]
    fn test_accept_cookies_detected() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(BlockKind::Paragraph, "Accept all cookies to continue browsing");
        let c = classifier.classify(&block);
        assert_eq!(c.kind, ClassificationKind::CookieConsent);
    }

    // ── Social widget tests ──

    #[test]
    fn test_social_share_detected() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(BlockKind::Paragraph, "Share on Facebook or Tweet this article");
        let c = classifier.classify(&block);
        assert_eq!(c.kind, ClassificationKind::SocialWidget);
    }

    // ── Newsletter popup tests ──

    #[test]
    fn test_newsletter_popup_detected() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(BlockKind::Paragraph, "Subscribe to our newsletter for updates");
        let c = classifier.classify(&block);
        assert_eq!(c.kind, ClassificationKind::NewsletterPopup);
    }

    // ── Heading not blocked ──

    #[test]
    fn test_heading_not_blocked() {
        let classifier = AdClassifier::new().unwrap();
        let block = make_block(BlockKind::Heading { level: 1 }, "Welcome to our site");
        assert!(!classifier.is_ad(&block));
    }
}
