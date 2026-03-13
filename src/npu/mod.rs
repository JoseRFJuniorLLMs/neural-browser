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
#[allow(unused_imports)] // Exported for consumers of the NPU API
pub use classifier::{Classification, ClassificationKind};

use crate::cpu::dom::DomTree;
use anyhow::Result;
use log::info;

/// Result of NPU processing a page.
#[allow(dead_code)] // Fields used by future pipeline stages (prefetch, summarization)
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
        html: &str,
        dom: &DomTree,
    ) -> Result<NpuResult> {
        // ── Guard: handle empty or malformed pages ──
        if dom.nodes.is_empty() {
            info!("[NPU] Empty DOM for {url} — returning empty result");
            return Ok(NpuResult {
                blocks: Vec::new(),
                ads_blocked: 0,
                summary: None,
                language: None,
                prefetch_urls: Vec::new(),
            });
        }

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

        // ── Step 5: Generate page summary ──
        let summary = self.generate_summary(&blocks, html, dom);

        info!(
            "[NPU] Extracted {} content blocks, {} prefetch candidates, lang={:?}",
            blocks.len(),
            prefetch_urls.len(),
            language,
        );

        Ok(NpuResult {
            blocks,
            ads_blocked,
            summary,
            language,
            prefetch_urls,
        })
    }

    /// Score links by likelihood of user clicking them.
    /// Returns top 3 highest-scoring links for prefetch.
    fn score_links_for_prefetch(&self, dom: &DomTree, current_url: &str) -> Vec<String> {
        let links = dom.links();

        if links.is_empty() {
            return Vec::new();
        }

        // Extract base domain from current URL for same-origin checks
        let current_domain = extract_domain(current_url);

        let mut scored: Vec<(f32, String)> = Vec::new();

        for (href, text) in &links {
            // Skip anchors, dangerous schemes, and empty hrefs
            if href.is_empty()
                || href.starts_with('#')
                || href.starts_with("javascript:")
                || href.starts_with("data:")
                || href.starts_with("blob:")
                || href.starts_with("vbscript:")
                || href.starts_with("mailto:")
                || href.starts_with("tel:")
            {
                continue;
            }

            let mut score: f32 = 0.0;
            let text_lower = text.to_lowercase();
            let href_lower = href.to_lowercase();

            // ── Same domain bonus ──
            if href.starts_with('/') || href.starts_with("./") || href.starts_with("../") {
                score += 0.3; // Relative URL = same domain
            } else if let Some(ref link_domain) = extract_domain(href) {
                if current_domain.as_deref() == Some(link_domain.as_str()) {
                    score += 0.25; // Same domain absolute URL
                } else {
                    score -= 0.1; // External domain penalty
                }
            }

            // ── Navigation / pagination patterns (high priority) ──
            if text_lower.contains("next")
                || text_lower.contains("continue")
                || text_lower.contains("read more")
                || text_lower.contains("see more")
                || text_lower.contains("view more")
                || text_lower.contains("load more")
                || text_lower == ">"
                || text_lower == ">>"
                || text_lower.contains("next page")
                || text_lower.contains("older posts")
                || text_lower.contains("newer posts")
            {
                score += 0.5;
            }

            // ── Article / content link patterns ──
            if href_lower.contains("/article")
                || href_lower.contains("/post/")
                || href_lower.contains("/blog/")
                || href_lower.contains("/news/")
                || href_lower.contains("/story/")
            {
                score += 0.2;
            }

            // ── Deprioritize footer/sidebar-like links ──
            if text_lower.contains("privacy")
                || text_lower.contains("terms")
                || text_lower.contains("cookie")
                || text_lower.contains("contact us")
                || text_lower.contains("about us")
                || text_lower.contains("sitemap")
                || text_lower.contains("copyright")
                || text_lower.contains("legal")
                || text_lower.contains("advertise")
            {
                score -= 0.3;
            }

            // ── Deprioritize login/auth links ──
            if text_lower.contains("log in")
                || text_lower.contains("sign in")
                || text_lower.contains("sign up")
                || text_lower.contains("register")
            {
                score -= 0.2;
            }

            // ── Deprioritize social media links ──
            if href_lower.contains("facebook.com")
                || href_lower.contains("twitter.com")
                || href_lower.contains("instagram.com")
                || href_lower.contains("linkedin.com")
                || href_lower.contains("youtube.com")
            {
                score -= 0.3;
            }

            // ── Longer link text is often more meaningful than single words ──
            if text.len() > 10 && text.len() < 100 {
                score += 0.1;
            }

            // Only include links with a minimum positive score
            if score > 0.15 {
                scored.push((score, href.to_string()));
            }
        }

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(3).map(|(_, url)| url).collect()
    }

    /// Generate a page summary from the first meaningful paragraph or meta description.
    fn generate_summary(
        &self,
        blocks: &[ContentBlock],
        _html: &str,
        dom: &DomTree,
    ) -> Option<String> {
        // Strategy 1: Check for meta description
        for node in &dom.nodes {
            if node.tag == "meta" {
                let is_desc = node.attrs.get("name")
                    .map(|n| n.eq_ignore_ascii_case("description"))
                    .unwrap_or(false);
                let is_og_desc = node.attrs.get("property")
                    .map(|p| p == "og:description")
                    .unwrap_or(false);

                if is_desc || is_og_desc {
                    if let Some(content) = node.attrs.get("content") {
                        let trimmed = content.trim();
                        if !trimmed.is_empty() && trimmed.len() >= 20 {
                            return Some(truncate_summary(trimmed, 300));
                        }
                    }
                }
            }
        }

        // Strategy 2: First meaningful paragraph (>50 chars, after a heading if possible)
        let mut found_heading = false;
        for block in blocks {
            if matches!(block.kind, BlockKind::Heading { .. }) {
                found_heading = true;
                continue;
            }
            if matches!(block.kind, BlockKind::Paragraph) && block.text.len() >= 50 {
                if found_heading || block.relevance >= 0.5 {
                    return Some(truncate_summary(&block.text, 300));
                }
            }
        }

        // Strategy 3: Any paragraph with at least 50 characters
        for block in blocks {
            if matches!(block.kind, BlockKind::Paragraph) && block.text.len() >= 50 {
                return Some(truncate_summary(&block.text, 300));
            }
        }

        None
    }

    /// Detect language from content blocks using character and word patterns.
    fn detect_language(&self, blocks: &[ContentBlock]) -> Option<String> {
        let text: String = blocks
            .iter()
            .filter(|b| matches!(b.kind, BlockKind::Paragraph | BlockKind::Heading { .. }))
            .take(10) // Sample more blocks for better accuracy
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        if text.len() < 20 {
            return None;
        }

        let text_lower = text.to_lowercase();

        // ── CJK detection (check first — character-based, not word-based) ──
        let cjk_count = text.chars().filter(|c| is_cjk(*c)).count();
        let total_chars = text.chars().count();
        if total_chars > 0 {
            let cjk_ratio = cjk_count as f32 / total_chars as f32;

            if cjk_ratio > 0.3 {
                // Distinguish Japanese from Chinese
                let has_hiragana = text.chars().any(|c| ('\u{3040}'..='\u{309F}').contains(&c));
                let has_katakana = text.chars().any(|c| ('\u{30A0}'..='\u{30FF}').contains(&c));

                if has_hiragana || has_katakana {
                    return Some("ja".into());
                }
                // Default to Chinese if CJK without Japanese kana
                return Some("zh".into());
            }
        }

        // ── Korean detection ──
        let has_hangul = text.chars().any(|c| ('\u{AC00}'..='\u{D7AF}').contains(&c));
        if has_hangul {
            return Some("ko".into());
        }

        // ── Arabic detection ──
        let has_arabic = text.chars().any(|c| ('\u{0600}'..='\u{06FF}').contains(&c));
        if has_arabic {
            return Some("ar".into());
        }

        // ── Cyrillic detection (Russian, etc.) ──
        let has_cyrillic = text.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c));
        if has_cyrillic {
            return Some("ru".into());
        }

        // ── Latin-script language detection (word patterns) ──

        // Portuguese markers
        let pt_markers = [" de ", " com ", " para ", " uma ", " das ", " dos ",
            " que ", " nao ", " mais ", " seu ", " sua ", " como ",
            " esta ", " foi ", " pelo ", " pela "];
        let pt_accents = text.chars().any(|c| "àáâãçõ".contains(c));
        let pt_score: usize = pt_markers.iter()
            .filter(|m| text_lower.contains(*m))
            .count();

        // Spanish markers
        let es_markers = [" el ", " los ", " las ", " por ", " una ", " del ",
            " que ", " con ", " son ", " como ", " esta ",
            " pero ", " tiene ", " puede "];
        let es_accents = text.chars().any(|c| "áéíóúñ¿¡".contains(c));
        let es_score: usize = es_markers.iter()
            .filter(|m| text_lower.contains(*m))
            .count();

        // French markers
        let fr_markers = [" le ", " la ", " les ", " des ", " un ", " une ",
            " du ", " est ", " dans ", " pour ", " avec ",
            " sur ", " qui ", " que ", " sont ", " cette "];
        let fr_accents = text.chars().any(|c| "àâçéèêëïîôùûü".contains(c));
        let fr_score: usize = fr_markers.iter()
            .filter(|m| text_lower.contains(*m))
            .count();

        // German markers
        let de_markers = [" der ", " die ", " das ", " und ", " ist ", " ein ",
            " eine ", " nicht ", " mit ", " auf ", " den ",
            " dem ", " von ", " sich ", " haben ", " werden "];
        let de_accents = text.chars().any(|c| "äöüß".contains(c));
        let de_score: usize = de_markers.iter()
            .filter(|m| text_lower.contains(*m))
            .count();

        // Italian markers
        let it_markers = [" il ", " la ", " le ", " di ", " un ", " una ",
            " che ", " per ", " con ", " sono ", " del ",
            " della ", " nella ", " questo ", " questa "];
        let it_accents = text.chars().any(|c| "àèéìòù".contains(c));
        let it_score: usize = it_markers.iter()
            .filter(|m| text_lower.contains(*m))
            .count();

        // English markers
        let en_markers = [" the ", " and ", " for ", " that ", " with ",
            " this ", " have ", " from ", " they ", " been ",
            " which ", " their ", " would ", " about ", " there "];
        let en_score: usize = en_markers.iter()
            .filter(|m| text_lower.contains(*m))
            .count();

        // Dutch markers
        let nl_markers = [" de ", " het ", " een ", " van ", " en ", " dat ",
            " die ", " niet ", " zijn ", " voor ", " met "];
        let nl_score: usize = nl_markers.iter()
            .filter(|m| text_lower.contains(*m))
            .count();

        // Find the best match: require at least 2 marker matches and
        // give bonus for language-specific accents/characters
        let mut candidates: Vec<(&str, usize)> = Vec::new();

        if pt_score >= 2 { candidates.push(("pt", pt_score + if pt_accents { 3 } else { 0 })); }
        if es_score >= 2 { candidates.push(("es", es_score + if es_accents { 3 } else { 0 })); }
        if fr_score >= 2 { candidates.push(("fr", fr_score + if fr_accents { 3 } else { 0 })); }
        if de_score >= 2 { candidates.push(("de", de_score + if de_accents { 3 } else { 0 })); }
        if it_score >= 2 { candidates.push(("it", it_score + if it_accents { 3 } else { 0 })); }
        if en_score >= 2 { candidates.push(("en", en_score)); }
        if nl_score >= 2 { candidates.push(("nl", nl_score)); }

        candidates.sort_by(|a, b| b.1.cmp(&a.1));

        // Return best if it has a clear lead or meets minimum threshold
        if let Some((lang, score)) = candidates.first() {
            if *score >= 3 {
                return Some((*lang).into());
            }
        }

        None
    }
}

/// Check if a character is in a CJK unified ideographs range.
fn is_cjk(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c)       // CJK Unified Ideographs
        || ('\u{3400}'..='\u{4DBF}').contains(&c) // CJK Extension A
        || ('\u{F900}'..='\u{FAFF}').contains(&c) // CJK Compatibility
}

/// Extract domain from a URL string (best-effort, no external crate needed).
fn extract_domain(url: &str) -> Option<String> {
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);

    let domain = without_scheme.split('/').next()?;
    let domain = domain.split(':').next()?; // Remove port

    if domain.is_empty() || !domain.contains('.') {
        return None;
    }

    Some(domain.to_lowercase())
}

/// Truncate a summary to a max length, breaking at word boundaries (UTF-8 safe).
fn truncate_summary(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }

    // Find a valid UTF-8 boundary at or before max_len
    let mut end = max_len;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }

    // Find last space before the boundary for clean word break
    let truncated = &text[..end];
    if let Some(last_space) = truncated.rfind(' ') {
        format!("{}...", &text[..last_space])
    } else {
        format!("{}...", truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::dom::parse_html;

    fn make_engine() -> NpuEngine {
        NpuEngine::new().unwrap()
    }

    // ── Empty/malformed page handling ──

    #[test]
    fn test_empty_dom_returns_empty_result() {
        let mut engine = make_engine();
        let dom = DomTree { nodes: Vec::new() };
        let result = engine.process_page("https://example.com", "", &dom).unwrap();
        assert!(result.blocks.is_empty());
        assert_eq!(result.ads_blocked, 0);
        assert!(result.summary.is_none());
        assert!(result.language.is_none());
        assert!(result.prefetch_urls.is_empty());
    }

    #[test]
    fn test_minimal_page() {
        let mut engine = make_engine();
        let html = "<html><body><p>Hello</p></body></html>";
        let dom = parse_html(html);
        let result = engine.process_page("https://example.com", html, &dom).unwrap();
        assert!(!result.blocks.is_empty());
    }

    // ── Language detection ──

    #[test]
    fn test_detect_english() {
        let engine = make_engine();
        let blocks = vec![ContentBlock {
            kind: BlockKind::Paragraph,
            text: "The quick brown fox jumps over the lazy dog and they have been running for a while which would be about there".into(),
            depth: 0,
            relevance: 0.5,
            children: Vec::new(),
        }];
        assert_eq!(engine.detect_language(&blocks), Some("en".into()));
    }

    #[test]
    fn test_detect_portuguese() {
        let engine = make_engine();
        let blocks = vec![ContentBlock {
            kind: BlockKind::Paragraph,
            text: "Este texto foi escrito com palavras de uma linguagem para mostrar como esta funcionalidade das que foram criadas pelo programador".into(),
            depth: 0,
            relevance: 0.5,
            children: Vec::new(),
        }];
        assert_eq!(engine.detect_language(&blocks), Some("pt".into()));
    }

    #[test]
    fn test_detect_german() {
        let engine = make_engine();
        let blocks = vec![ContentBlock {
            kind: BlockKind::Paragraph,
            text: "Der Mann ist nicht auf dem Weg und die Frau hat ein Buch von dem Autor gelesen".into(),
            depth: 0,
            relevance: 0.5,
            children: Vec::new(),
        }];
        assert_eq!(engine.detect_language(&blocks), Some("de".into()));
    }

    #[test]
    fn test_detect_short_text_returns_none() {
        let engine = make_engine();
        let blocks = vec![ContentBlock {
            kind: BlockKind::Paragraph,
            text: "Hello".into(),
            depth: 0,
            relevance: 0.5,
            children: Vec::new(),
        }];
        assert_eq!(engine.detect_language(&blocks), None);
    }

    // ── Link scoring ──

    #[test]
    fn test_prefetch_skips_anchors_and_javascript() {
        let engine = make_engine();
        let html = r##"<a href="#">Top</a><a href="javascript:void(0)">Click</a><a href="/next">Next</a>"##;
        let dom = parse_html(html);
        let urls = engine.score_links_for_prefetch(&dom, "https://example.com");
        // Only /next should pass
        assert!(urls.iter().all(|u| !u.starts_with('#') && !u.starts_with("javascript:")));
    }

    #[test]
    fn test_prefetch_returns_max_3() {
        let engine = make_engine();
        let html = r#"
            <a href="/a">Read more about A</a>
            <a href="/b">Continue to B</a>
            <a href="/c">See more of C</a>
            <a href="/d">View more D</a>
            <a href="/e">Next page E</a>
        "#;
        let dom = parse_html(html);
        let urls = engine.score_links_for_prefetch(&dom, "https://example.com");
        assert!(urls.len() <= 3);
    }

    #[test]
    fn test_prefetch_deprioritizes_footer_links() {
        let engine = make_engine();
        let html = r#"
            <a href="/article/1">Read more about this topic</a>
            <a href="/privacy">Privacy Policy</a>
            <a href="/terms">Terms of Service</a>
        "#;
        let dom = parse_html(html);
        let urls = engine.score_links_for_prefetch(&dom, "https://example.com");
        // Privacy and Terms should not appear (or be ranked lower)
        assert!(!urls.contains(&"/privacy".to_string()));
        assert!(!urls.contains(&"/terms".to_string()));
    }

    // ── Summary generation ──

    #[test]
    fn test_summary_from_meta_description() {
        let engine = make_engine();
        let html = r#"<html><head><meta name="description" content="This is a detailed description of the page content for search engines and readers."></head><body><p>Body text.</p></body></html>"#;
        let dom = parse_html(html);
        let blocks = vec![];
        let summary = engine.generate_summary(&blocks, html, &dom);
        assert!(summary.is_some());
        assert!(summary.unwrap().contains("detailed description"));
    }

    #[test]
    fn test_summary_from_paragraph() {
        let engine = make_engine();
        let dom = DomTree { nodes: Vec::new() };
        let blocks = vec![
            ContentBlock {
                kind: BlockKind::Heading { level: 1 },
                text: "Title".into(),
                depth: 0,
                relevance: 0.9,
                children: Vec::new(),
            },
            ContentBlock {
                kind: BlockKind::Paragraph,
                text: "This is a meaningful paragraph that contains enough text to be considered a real summary of the page content.".into(),
                depth: 0,
                relevance: 0.7,
                children: Vec::new(),
            },
        ];
        let summary = engine.generate_summary(&blocks, "", &dom);
        assert!(summary.is_some());
    }

    // ── Helper function tests ──

    #[test]
    fn test_extract_domain() {
        assert_eq!(extract_domain("https://example.com/page"), Some("example.com".into()));
        assert_eq!(extract_domain("http://sub.example.com:8080/path"), Some("sub.example.com".into()));
        assert_eq!(extract_domain("/relative/path"), None);
        assert_eq!(extract_domain(""), None);
    }

    #[test]
    fn test_truncate_summary() {
        assert_eq!(truncate_summary("Short text", 100), "Short text");
        let long = "This is a very long text that needs to be truncated at a word boundary to avoid cutting words";
        let result = truncate_summary(long, 40);
        assert!(result.len() < 50);
        assert!(result.ends_with("..."));
    }
}
