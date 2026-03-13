//! GPU: Layout engine — positions content blocks for rendering.
//!
//! Vertical flow layout with CSS-aware styling.
//! The NPU classifies blocks semantically AND attaches ComputedStyle from the CSS cascade.
//! When a block has a computed_style, the GPU uses real CSS values (font-size, color,
//! margins, background-color, display). Falls back to theme-based heuristics otherwise.

use crate::css::values::CssDisplay;
use crate::npu::{ContentBlock, BlockKind};
use crate::ui::Theme;
use std::collections::HashMap;

/// A positioned element ready for GPU rendering.
#[derive(Debug, Clone)]
pub struct LayoutBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub kind: LayoutKind,
    /// Link destination URL (for hit testing on click).
    pub href: Option<String>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used by future GPU pipeline stages
pub enum LayoutKind {
    /// URL bar at top
    UrlBar { text: String },
    /// Text with font size and color
    Text {
        text: String,
        font_size: f32,
        color: [f32; 4],    // RGBA
        bold: bool,
        italic: bool,
    },
    /// Image placeholder (text fallback if no pixel data)
    Image { src: String, alt: String },
    /// Decoded image with RGBA pixel data ready for GPU texture upload
    DecodedImage {
        width: u32,
        height: u32,
        rgba: Vec<u8>,
        alt: String,
    },
    /// Horizontal separator line
    Separator,
    /// Code block with monospace font
    Code { text: String, language: Option<String>, font_size: f32 },
    /// Background rect (for code blocks, quotes, etc.)
    Background { color: [f32; 4] },
    /// Visual input box (rounded rectangle with placeholder text)
    InputBox { placeholder: String, font_size: f32 },
    /// Visual button (rounded filled rectangle with text)
    Button { text: String, font_size: f32, bg_color: [f32; 4], text_color: [f32; 4] },
}

/// Estimate the number of wrapped lines for a text given a width and font size.
/// Uses average character width heuristic (0.55 * font_size for sans-serif).
fn estimate_lines(text: &str, width: f32, font_size: f32) -> f32 {
    if text.is_empty() || width <= 0.0 {
        return 1.0;
    }
    let avg_char_width = font_size * 0.55;
    let chars_per_line = (width / avg_char_width).max(1.0);

    let mut total_lines: f32 = 0.0;
    for line in text.split('\n') {
        // Use char count, not byte length — CJK/emoji chars are multi-byte
        let line_len = line.chars().count() as f32;
        total_lines += (line_len / chars_per_line).ceil().max(1.0);
    }
    total_lines.max(1.0)
}

/// Check if a block should be hidden (CSS display:none or visibility:hidden).
fn is_css_hidden(block: &ContentBlock) -> bool {
    if let Some(ref cs) = block.computed_style {
        matches!(cs.display, CssDisplay::None) || !cs.visibility
    } else {
        false
    }
}

/// Extract font size from ComputedStyle (in px), applying zoom.
/// Falls back to the provided default if no style is attached.
fn css_font_size(block: &ContentBlock, default: f32, zoom: f32) -> f32 {
    if let Some(ref cs) = block.computed_style {
        cs.font_size_px() * zoom
    } else {
        default * zoom
    }
}

/// Extract text color from ComputedStyle as [f32;4] RGBA.
/// Falls back to the provided theme color.
fn css_color(block: &ContentBlock, default: [f32; 4]) -> [f32; 4] {
    if let Some(ref cs) = block.computed_style {
        cs.color_array()
    } else {
        default
    }
}

/// Extract background color from ComputedStyle.
/// Returns None if transparent (a < 0.01) or no style attached.
fn css_bg_color(block: &ContentBlock) -> Option<[f32; 4]> {
    if let Some(ref cs) = block.computed_style {
        let bg = cs.bg_color_array();
        if bg[3] > 0.01 { Some(bg) } else { None }
    } else {
        None
    }
}

/// Check if CSS says font should be bold.
fn css_bold(block: &ContentBlock, default: bool) -> bool {
    if let Some(ref cs) = block.computed_style {
        cs.font_weight.is_bold()
    } else {
        default
    }
}

/// Check if CSS says font should be italic.
fn css_italic(block: &ContentBlock, default: bool) -> bool {
    if let Some(ref cs) = block.computed_style {
        matches!(cs.font_style, crate::css::values::CssFontStyle::Italic | crate::css::values::CssFontStyle::Oblique)
    } else {
        default
    }
}

/// Extract CSS margin-top in px (zoomed).
fn css_margin_top(block: &ContentBlock, default: f32, zoom: f32) -> f32 {
    if let Some(ref cs) = block.computed_style {
        let fs = cs.font_size_px();
        cs.margin_top.to_px(fs, 0.0, 0.0) * zoom
    } else {
        default * zoom
    }
}

/// Extract CSS margin-bottom in px (zoomed).
fn css_margin_bottom(block: &ContentBlock, default: f32, zoom: f32) -> f32 {
    if let Some(ref cs) = block.computed_style {
        let fs = cs.font_size_px();
        cs.margin_bottom.to_px(fs, 0.0, 0.0) * zoom
    } else {
        default * zoom
    }
}

/// Extract CSS line-height in px.
fn css_line_height(block: &ContentBlock, font_size: f32, default_factor: f32) -> f32 {
    if let Some(ref cs) = block.computed_style {
        cs.line_height_px().max(font_size)
    } else {
        font_size * default_factor
    }
}

/// Compute layout for all content blocks.
/// Returns a list of positioned LayoutBoxes in DOCUMENT coordinates (scroll-independent).
/// The scroll offset is applied at render time, not here — so layout only needs
/// recomputation on content change or viewport resize, NOT on scroll.
/// `zoom` scales all font sizes, margins and spacings (1.0 = 100%).

/// Internal layout with zoom factor applied to all sizes.
/// `image_dimensions` maps image source URLs to their (width, height) in pixels,
/// used to properly size `Image` blocks when the GPU texture has been loaded.
pub fn compute_layout_zoom(
    blocks: &[ContentBlock],
    _scroll_y: f32,
    viewport_width: f32,
    theme: &Theme,
    zoom: f32,
    image_dimensions: &HashMap<String, (u32, u32)>,
) -> Vec<LayoutBox> {
    let mut layout = Vec::new();
    let z = zoom.clamp(0.25, 5.0); // safety clamp

    let margin_x: f32 = 40.0 * z;
    let content_width: f32 = (viewport_width - margin_x * 2.0).max(200.0).min(900.0 * z);

    // Content starts below toolbar — positions are in document space
    let toolbar_h = super::renderer::TOOLBAR_HEIGHT;
    let mut cursor_y: f32 = (toolbar_h + 10.0) * z;

    for block in blocks {
        // Skip low-relevance content
        if block.relevance < 0.15 {
            continue;
        }

        // ── CSS display:none / visibility:hidden → skip entirely ──
        if is_css_hidden(block) {
            continue;
        }

        match &block.kind {
            BlockKind::Title => {
                // URL bar shows the title
            }
            BlockKind::Heading { level } => {
                let default_size = match level {
                    1 => 38.0,
                    2 => 32.0,
                    3 => 26.0,
                    4 => 22.0,
                    _ => 20.0,
                };
                let font_size = css_font_size(block, default_size, z);
                let color = css_color(block, theme.heading);
                let bold = css_bold(block, true);
                let italic = css_italic(block, false);

                // Margin-top from CSS or heuristic
                let spacing_before = css_margin_top(block,
                    if *level <= 2 { default_size } else { default_size * 0.8 }, z);
                cursor_y += spacing_before;

                // Background from CSS (e.g. highlighted headings)
                if let Some(bg) = css_bg_color(block) {
                    let lines = estimate_lines(&block.text, content_width, font_size);
                    let bh = lines * font_size * 1.4;
                    layout.push(LayoutBox {
                        x: margin_x - 4.0, y: cursor_y - 2.0,
                        width: content_width + 8.0, height: bh + 4.0,
                        kind: LayoutKind::Background { color: bg },
                        href: None,
                    });
                }

                let line_height = css_line_height(block, font_size, 1.4);
                let lines = estimate_lines(&block.text, content_width, font_size);
                let block_height = lines * line_height;

                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: block_height,
                    kind: LayoutKind::Text {
                        text: block.text.clone(),
                        font_size,
                        color,
                        bold,
                        italic,
                    },
                    href: None,
                });

                let spacing_after = css_margin_bottom(block, default_size * 0.4, z);
                cursor_y += block_height + spacing_after;
            }
            BlockKind::Paragraph => {
                if block.text.is_empty() {
                    continue;
                }
                let font_size = css_font_size(block, 20.0, z);
                let line_height = css_line_height(block, font_size, 1.6);
                let color = css_color(block, theme.text);
                let bold = css_bold(block, false);
                let italic = css_italic(block, false);
                let lines = estimate_lines(&block.text, content_width, font_size);

                // CSS margin-top
                cursor_y += css_margin_top(block, 0.0, z);

                // Background from CSS
                if let Some(bg) = css_bg_color(block) {
                    layout.push(LayoutBox {
                        x: margin_x - 4.0, y: cursor_y - 2.0,
                        width: content_width + 8.0, height: lines * line_height + 4.0,
                        kind: LayoutKind::Background { color: bg },
                        href: None,
                    });
                }

                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: lines * line_height,
                    kind: LayoutKind::Text {
                        text: block.text.clone(),
                        font_size,
                        color,
                        bold,
                        italic,
                    },
                    href: None,
                });

                cursor_y += lines * line_height + css_margin_bottom(block, 14.0, z);
            }
            BlockKind::Code { language } => {
                let font_size = css_font_size(block, 16.0, z);
                let line_height = css_line_height(block, font_size, 1.5);
                let lines = block.text.lines().count().max(1) as f32;
                let block_height = lines * line_height + 24.0;

                // Background (CSS or theme)
                let bg = css_bg_color(block).unwrap_or(theme.code_bg);
                layout.push(LayoutBox {
                    x: margin_x - 12.0,
                    y: cursor_y - 4.0,
                    width: content_width + 24.0,
                    height: block_height,
                    kind: LayoutKind::Background { color: bg },
                    href: None,
                });

                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y + 8.0,
                    width: content_width,
                    height: block_height - 16.0,
                    kind: LayoutKind::Code {
                        text: block.text.clone(),
                        language: language.clone(),
                        font_size,
                    },
                    href: None,
                });

                cursor_y += block_height + 18.0;
            }
            BlockKind::Quote => {
                let font_size = css_font_size(block, 19.0, z);
                let line_height = css_line_height(block, font_size, 1.6);
                let lines = estimate_lines(&block.text, content_width - 40.0, font_size);
                let block_height = lines * line_height + 16.0;
                let color = css_color(block, [0.7, 0.7, 0.75, 1.0]);
                let italic = css_italic(block, true);

                // Left border + background (CSS or theme)
                let bg = css_bg_color(block).unwrap_or(theme.quote_bg);
                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: block_height,
                    kind: LayoutKind::Background { color: bg },
                    href: None,
                });

                layout.push(LayoutBox {
                    x: margin_x + 20.0,
                    y: cursor_y + 8.0,
                    width: content_width - 40.0,
                    height: block_height - 16.0,
                    kind: LayoutKind::Text {
                        text: block.text.clone(),
                        font_size,
                        color,
                        bold: false,
                        italic,
                    },
                    href: None,
                });

                cursor_y += block_height + 14.0;
            }
            BlockKind::Image { src, alt } => {
                if let Some((img_w, img_h, rgba)) = &block.image_data {
                    // NPU-decoded image: use DecodedImage layout kind
                    let scale = (content_width / *img_w as f32).min(1.0);
                    let display_w = (*img_w as f32 * scale).round();
                    let display_h = (*img_h as f32 * scale).round();

                    layout.push(LayoutBox {
                        x: margin_x,
                        y: cursor_y,
                        width: display_w,
                        height: display_h,
                        kind: LayoutKind::DecodedImage {
                            width: *img_w,
                            height: *img_h,
                            rgba: rgba.clone(),
                            alt: alt.clone(),
                        },
                        href: None,
                    });

                    if !alt.is_empty() {
                        layout.push(LayoutBox {
                            x: margin_x,
                            y: cursor_y + display_h + 4.0,
                            width: content_width,
                            height: 20.0,
                            kind: LayoutKind::Text {
                                text: alt.clone(),
                                font_size: 15.0,
                                color: theme.text_dim,
                                bold: false,
                                italic: true,
                            },
                            href: None,
                        });
                        cursor_y += display_h + 28.0;
                    } else {
                        cursor_y += display_h + 12.0;
                    }
                } else if let Some((tex_w, tex_h)) = image_dimensions.get(src) {
                    // GPU texture cache has dimensions: size properly with aspect ratio
                    let scale = (content_width / *tex_w as f32).min(1.0);
                    let display_w = (*tex_w as f32 * scale).round();
                    let display_h = (*tex_h as f32 * scale).round();

                    layout.push(LayoutBox {
                        x: margin_x,
                        y: cursor_y,
                        width: display_w,
                        height: display_h,
                        kind: LayoutKind::Image {
                            src: src.clone(),
                            alt: alt.clone(),
                        },
                        href: None,
                    });

                    if !alt.is_empty() {
                        layout.push(LayoutBox {
                            x: margin_x,
                            y: cursor_y + display_h + 4.0,
                            width: content_width,
                            height: 20.0,
                            kind: LayoutKind::Text {
                                text: alt.clone(),
                                font_size: 15.0,
                                color: theme.text_dim,
                                bold: false,
                                italic: true,
                            },
                            href: None,
                        });
                        cursor_y += display_h + 28.0;
                    } else {
                        cursor_y += display_h + 12.0;
                    }
                } else {
                    // Fallback: text placeholder (no decoded data, no cached texture)
                    let img_height = 30.0;
                    layout.push(LayoutBox {
                        x: margin_x,
                        y: cursor_y,
                        width: content_width,
                        height: img_height,
                        kind: LayoutKind::Image {
                            src: src.clone(),
                            alt: alt.clone(),
                        },
                        href: None,
                    });
                    cursor_y += img_height + 8.0;
                }
            }
            BlockKind::List { ordered } => {
                // Render children (ListItems) with proper numbering
                for (idx, child) in block.children.iter().enumerate() {
                    if child.relevance < 0.15 || is_css_hidden(child) {
                        continue;
                    }
                    let font_size = css_font_size(child, 19.0, z);
                    let color = css_color(child, theme.text);
                    let bold = css_bold(child, false);
                    let italic = css_italic(child, false);
                    let prefix = if *ordered {
                        format!("{}. {}", idx + 1, child.text)
                    } else {
                        format!("\u{2022}  {}", child.text)
                    };
                    let lines = estimate_lines(&prefix, content_width - 24.0, font_size);
                    let line_height = css_line_height(child, font_size, 1.5);

                    layout.push(LayoutBox {
                        x: margin_x + 24.0,
                        y: cursor_y,
                        width: content_width - 24.0,
                        height: lines * line_height,
                        kind: LayoutKind::Text {
                            text: prefix,
                            font_size,
                            color,
                            bold,
                            italic,
                        },
                        href: None,
                    });
                    cursor_y += lines * line_height + 4.0;

                    // Render nested lists
                    for nested in &child.children {
                        if let BlockKind::List { ordered: nested_ord } = &nested.kind {
                            for (ni, nc) in nested.children.iter().enumerate() {
                                let np = if *nested_ord {
                                    format!("  {}. {}", ni + 1, nc.text)
                                } else {
                                    format!("  \u{25E6}  {}", nc.text)
                                };
                                let nl = estimate_lines(&np, content_width - 48.0, font_size);
                                layout.push(LayoutBox {
                                    x: margin_x + 48.0,
                                    y: cursor_y,
                                    width: content_width - 48.0,
                                    height: nl * font_size * 1.5,
                                    kind: LayoutKind::Text {
                                        text: np,
                                        font_size,
                                        color: theme.text,
                                        bold: false,
                                        italic: false,
                                    },
                                    href: None,
                                });
                                cursor_y += nl * font_size * 1.5 + 3.0;
                            }
                        }
                    }
                }
                cursor_y += 6.0;
            }
            BlockKind::ListItem => {
                // Standalone list items (outside a List container)
                let font_size = 19.0 * z;
                let prefix = format!("\u{2022}  {}", block.text);
                let lines = estimate_lines(&prefix, content_width - 24.0, font_size);

                layout.push(LayoutBox {
                    x: margin_x + 24.0,
                    y: cursor_y,
                    width: content_width - 24.0,
                    height: lines * font_size * 1.5,
                    kind: LayoutKind::Text {
                        text: prefix,
                        font_size,
                        color: theme.text,
                        bold: false,
                        italic: false,
                    },
                    href: None,
                });
                cursor_y += lines * font_size * 1.5 + 4.0;
            }
            BlockKind::Link { href } => {
                if block.text.is_empty() {
                    continue;
                }
                let font_size = css_font_size(block, 19.0, z);
                let color = css_color(block, theme.link);
                let bold = css_bold(block, false);
                let italic = css_italic(block, false);
                let lines = estimate_lines(&block.text, content_width, font_size);
                let line_height = css_line_height(block, font_size, 1.5);

                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: lines * line_height,
                    kind: LayoutKind::Text {
                        text: block.text.clone(),
                        font_size,
                        color,
                        bold,
                        italic,
                    },
                    href: Some(href.clone()),
                });
                cursor_y += lines * line_height + 6.0;
            }
            BlockKind::Separator => {
                cursor_y += 10.0;
                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: 1.0,
                    kind: LayoutKind::Separator,
                    href: None,
                });
                cursor_y += 18.0;
            }
            BlockKind::Table => {
                // Render table rows as pipe-separated text (children are TableRow)
                if !block.children.is_empty() {
                    let font_size = 16.0 * z;
                    let line_h = font_size * 1.5;
                    for child in &block.children {
                        if let BlockKind::TableRow = &child.kind {
                            let lines = estimate_lines(&child.text, content_width, font_size);
                            layout.push(LayoutBox {
                                x: margin_x,
                                y: cursor_y,
                                width: content_width,
                                height: lines * line_h,
                                kind: LayoutKind::Code {
                                    text: child.text.clone(),
                                    language: None,
                                    font_size,
                                },
                                href: None,
                            });
                            cursor_y += lines * line_h + 2.0;
                        }
                    }
                    cursor_y += 8.0;
                } else if !block.text.is_empty() {
                    // Fallback: render table text as code block
                    let font_size = 16.0 * z;
                    let lines = block.text.lines().count().max(1) as f32;
                    let block_height = lines * font_size * 1.5 + 16.0;
                    layout.push(LayoutBox {
                        x: margin_x,
                        y: cursor_y,
                        width: content_width,
                        height: block_height,
                        kind: LayoutKind::Code {
                            text: block.text.clone(),
                            language: None,
                            font_size,
                        },
                        href: None,
                    });
                    cursor_y += block_height + 12.0;
                }
            }
            BlockKind::TableRow => {
                // Standalone table row (outside Table)
                if !block.text.is_empty() {
                    let font_size = 16.0 * z;
                    let lines = estimate_lines(&block.text, content_width, font_size);
                    layout.push(LayoutBox {
                        x: margin_x,
                        y: cursor_y,
                        width: content_width,
                        height: lines * font_size * 1.5,
                        kind: LayoutKind::Code {
                            text: block.text.clone(),
                            language: None,
                            font_size,
                        },
                        href: None,
                    });
                    cursor_y += lines * font_size * 1.5 + 4.0;
                }
            }
            BlockKind::DefinitionList => {
                // Render dt/dd children
                for child in &block.children {
                    match &child.kind {
                        BlockKind::DefinitionTerm => {
                            let font_size = 19.0 * z;
                            let lines = estimate_lines(&child.text, content_width, font_size);
                            layout.push(LayoutBox {
                                x: margin_x,
                                y: cursor_y,
                                width: content_width,
                                height: lines * font_size * 1.5,
                                kind: LayoutKind::Text {
                                    text: child.text.clone(),
                                    font_size,
                                    color: theme.heading,
                                    bold: true,
                                    italic: false,
                                },
                                href: None,
                            });
                            cursor_y += lines * font_size * 1.5 + 4.0;
                        }
                        BlockKind::DefinitionDesc => {
                            let font_size = 18.0 * z;
                            let lines = estimate_lines(&child.text, content_width - 24.0, font_size);
                            layout.push(LayoutBox {
                                x: margin_x + 24.0,
                                y: cursor_y,
                                width: content_width - 24.0,
                                height: lines * font_size * 1.5,
                                kind: LayoutKind::Text {
                                    text: child.text.clone(),
                                    font_size,
                                    color: theme.text,
                                    bold: false,
                                    italic: false,
                                },
                                href: None,
                            });
                            cursor_y += lines * font_size * 1.5 + 4.0;
                        }
                        _ => {}
                    }
                }
                cursor_y += 8.0;
            }
            BlockKind::Details { .. } => {
                // Render summary and content children
                for child in &block.children {
                    match &child.kind {
                        BlockKind::Summary => {
                            let font_size = 19.0 * z;
                            let text = format!("\u{25B6} {}", child.text);
                            let lines = estimate_lines(&text, content_width, font_size);
                            layout.push(LayoutBox {
                                x: margin_x,
                                y: cursor_y,
                                width: content_width,
                                height: lines * font_size * 1.5,
                                kind: LayoutKind::Text {
                                    text,
                                    font_size,
                                    color: theme.link,
                                    bold: true,
                                    italic: false,
                                },
                                href: None,
                            });
                            cursor_y += lines * font_size * 1.5 + 4.0;
                        }
                        _ => {
                            // Render other children as paragraphs
                            if !child.text.is_empty() {
                                let font_size = 19.0 * z;
                                let lines = estimate_lines(&child.text, content_width - 16.0, font_size);
                                layout.push(LayoutBox {
                                    x: margin_x + 16.0,
                                    y: cursor_y,
                                    width: content_width - 16.0,
                                    height: lines * font_size * 1.5,
                                    kind: LayoutKind::Text {
                                        text: child.text.clone(),
                                        font_size,
                                        color: theme.text,
                                        bold: false,
                                        italic: false,
                                    },
                                    href: None,
                                });
                                cursor_y += lines * font_size * 1.5 + 4.0;
                            }
                        }
                    }
                }
                cursor_y += 6.0;
            }
            BlockKind::Form => {
                // Render form children (InputFields, ButtonGroups, labels)
                cursor_y += 8.0 * z;
                for child in &block.children {
                    if child.relevance < 0.15 || is_css_hidden(child) {
                        continue;
                    }
                    match &child.kind {
                        BlockKind::InputField { placeholder, input_type } => {
                            let font_size = 18.0 * z;
                            // Search/text inputs get a wide visual box
                            let is_text_input = matches!(input_type.as_str(),
                                "search" | "text" | "email" | "url" | "tel" | "number" | "password" | "textarea");
                            if is_text_input {
                                let box_w = (content_width * 0.75).min(560.0 * z);
                                let box_h = 48.0 * z;
                                let box_x = margin_x + (content_width - box_w) / 2.0; // center
                                layout.push(LayoutBox {
                                    x: box_x,
                                    y: cursor_y,
                                    width: box_w,
                                    height: box_h,
                                    kind: LayoutKind::InputBox {
                                        placeholder: placeholder.clone(),
                                        font_size,
                                    },
                                    href: None,
                                });
                                cursor_y += box_h + 12.0 * z;
                            } else {
                                // Checkbox/radio/select: render as text
                                let lines = estimate_lines(placeholder, content_width, font_size);
                                layout.push(LayoutBox {
                                    x: margin_x + 24.0 * z,
                                    y: cursor_y,
                                    width: content_width - 24.0 * z,
                                    height: lines * font_size * 1.5,
                                    kind: LayoutKind::Text {
                                        text: placeholder.clone(),
                                        font_size,
                                        color: theme.text,
                                        bold: false,
                                        italic: false,
                                    },
                                    href: None,
                                });
                                cursor_y += lines * font_size * 1.5 + 6.0 * z;
                            }
                        }
                        BlockKind::ButtonGroup => {
                            // Render buttons side by side, centered
                            let btn_font_size = 16.0 * z;
                            let btn_h = 44.0 * z;
                            let btn_gap = 12.0 * z;
                            let btn_pad = 28.0 * z; // horizontal padding inside button

                            // Calculate total width of all buttons
                            let mut btn_widths: Vec<f32> = Vec::new();
                            for btn_child in &child.children {
                                let text_w = btn_child.text.len() as f32 * btn_font_size * 0.55;
                                btn_widths.push(text_w + btn_pad * 2.0);
                            }
                            let total_w: f32 = btn_widths.iter().sum::<f32>()
                                + btn_gap * (btn_widths.len().max(1) - 1) as f32;

                            let start_x = margin_x + (content_width - total_w).max(0.0) / 2.0;
                            let mut btn_x = start_x;

                            for (i, btn_child) in child.children.iter().enumerate() {
                                let bw = btn_widths.get(i).copied().unwrap_or(100.0 * z);
                                layout.push(LayoutBox {
                                    x: btn_x,
                                    y: cursor_y,
                                    width: bw,
                                    height: btn_h,
                                    kind: LayoutKind::Button {
                                        text: btn_child.text.clone(),
                                        font_size: btn_font_size,
                                        bg_color: [0.24, 0.24, 0.32, 1.0],
                                        text_color: theme.text,
                                    },
                                    href: None,
                                });
                                btn_x += bw + btn_gap;
                            }
                            cursor_y += btn_h + 14.0 * z;
                        }
                        BlockKind::Paragraph => {
                            // Label text
                            if !child.text.is_empty() {
                                let font_size = 16.0 * z;
                                let lines = estimate_lines(&child.text, content_width, font_size);
                                layout.push(LayoutBox {
                                    x: margin_x,
                                    y: cursor_y,
                                    width: content_width,
                                    height: lines * font_size * 1.5,
                                    kind: LayoutKind::Text {
                                        text: child.text.clone(),
                                        font_size,
                                        color: theme.text_dim,
                                        bold: false,
                                        italic: false,
                                    },
                                    href: None,
                                });
                                cursor_y += lines * font_size * 1.5 + 4.0 * z;
                            }
                        }
                        _ => {}
                    }
                }
                cursor_y += 6.0 * z;

                // Fallback: if no children but has text, render as before
                if block.children.is_empty() && !block.text.is_empty() {
                    let font_size = 18.0 * z;
                    let lines = estimate_lines(&block.text, content_width, font_size);
                    layout.push(LayoutBox {
                        x: margin_x,
                        y: cursor_y,
                        width: content_width,
                        height: lines * font_size * 1.5,
                        kind: LayoutKind::Text {
                            text: block.text.clone(),
                            font_size,
                            color: theme.text_dim,
                            bold: false,
                            italic: true,
                        },
                        href: None,
                    });
                    cursor_y += lines * font_size * 1.5 + 8.0;
                }
            }
            BlockKind::InputField { placeholder, input_type } => {
                // Standalone InputField (outside form)
                let font_size = 18.0 * z;
                let is_text_input = matches!(input_type.as_str(),
                    "search" | "text" | "email" | "url" | "tel" | "number" | "password" | "textarea");
                if is_text_input {
                    let box_w = (content_width * 0.75).min(560.0 * z);
                    let box_h = 48.0 * z;
                    let box_x = margin_x + (content_width - box_w) / 2.0;
                    layout.push(LayoutBox {
                        x: box_x,
                        y: cursor_y,
                        width: box_w,
                        height: box_h,
                        kind: LayoutKind::InputBox {
                            placeholder: placeholder.clone(),
                            font_size,
                        },
                        href: None,
                    });
                    cursor_y += box_h + 12.0 * z;
                } else {
                    let lines = estimate_lines(placeholder, content_width, font_size);
                    layout.push(LayoutBox {
                        x: margin_x,
                        y: cursor_y,
                        width: content_width,
                        height: lines * font_size * 1.5,
                        kind: LayoutKind::Text {
                            text: placeholder.clone(),
                            font_size,
                            color: theme.text,
                            bold: false,
                            italic: false,
                        },
                        href: None,
                    });
                    cursor_y += lines * font_size * 1.5 + 6.0 * z;
                }
            }
            BlockKind::ButtonGroup => {
                // Standalone ButtonGroup (outside form)
                let btn_font_size = 16.0 * z;
                let btn_h = 44.0 * z;
                let btn_gap = 12.0 * z;
                let btn_pad = 28.0 * z;

                let mut btn_widths: Vec<f32> = Vec::new();
                for btn_child in &block.children {
                    let text_w = btn_child.text.len() as f32 * btn_font_size * 0.55;
                    btn_widths.push(text_w + btn_pad * 2.0);
                }
                let total_w: f32 = btn_widths.iter().sum::<f32>()
                    + btn_gap * (btn_widths.len().max(1) - 1) as f32;
                let start_x = margin_x + (content_width - total_w).max(0.0) / 2.0;
                let mut btn_x = start_x;

                for (i, btn_child) in block.children.iter().enumerate() {
                    let bw = btn_widths.get(i).copied().unwrap_or(100.0 * z);
                    layout.push(LayoutBox {
                        x: btn_x,
                        y: cursor_y,
                        width: bw,
                        height: btn_h,
                        kind: LayoutKind::Button {
                            text: btn_child.text.clone(),
                            font_size: btn_font_size,
                            bg_color: [0.24, 0.24, 0.32, 1.0],
                            text_color: theme.text,
                        },
                        href: None,
                    });
                    btn_x += bw + btn_gap;
                }
                cursor_y += btn_h + 14.0 * z;
            }
            BlockKind::InlineGroup => {
                // Horizontal flow: render children side by side
                let font_size = 15.0 * z;
                let line_h = font_size * 1.5;
                let gap = 16.0 * z;
                let mut x = margin_x;
                let max_x = margin_x + content_width;

                for child in &block.children {
                    if child.text.is_empty() || child.relevance < 0.15 {
                        continue;
                    }
                    let text_w = child.text.len() as f32 * font_size * 0.55 + 4.0;

                    // Wrap to next line if exceeds width
                    if x + text_w > max_x && x > margin_x {
                        x = margin_x;
                        cursor_y += line_h + 4.0 * z;
                    }

                    let href = if let BlockKind::Link { href } = &child.kind {
                        Some(href.clone())
                    } else {
                        None
                    };

                    let color = if href.is_some() { theme.link } else { theme.text };

                    layout.push(LayoutBox {
                        x,
                        y: cursor_y,
                        width: text_w,
                        height: line_h,
                        kind: LayoutKind::Text {
                            text: child.text.clone(),
                            font_size,
                            color,
                            bold: false,
                            italic: false,
                        },
                        href,
                    });
                    x += text_w + gap;
                }
                cursor_y += line_h + 10.0 * z;
            }
            BlockKind::Figure => {
                // Render figure children (images + captions)
                for child in &block.children {
                    match &child.kind {
                        BlockKind::Image { src, alt } => {
                            if let Some((img_w, img_h, rgba)) = &child.image_data {
                                let scale = (content_width / *img_w as f32).min(1.0);
                                let display_w = (*img_w as f32 * scale).round();
                                let display_h = (*img_h as f32 * scale).round();
                                layout.push(LayoutBox {
                                    x: margin_x,
                                    y: cursor_y,
                                    width: display_w,
                                    height: display_h,
                                    kind: LayoutKind::DecodedImage {
                                        width: *img_w,
                                        height: *img_h,
                                        rgba: rgba.clone(),
                                        alt: alt.clone(),
                                    },
                                    href: None,
                                });
                                cursor_y += display_h + 8.0;
                            } else if let Some((tex_w, tex_h)) = image_dimensions.get(src) {
                                let scale = (content_width / *tex_w as f32).min(1.0);
                                let display_w = (*tex_w as f32 * scale).round();
                                let display_h = (*tex_h as f32 * scale).round();
                                layout.push(LayoutBox {
                                    x: margin_x,
                                    y: cursor_y,
                                    width: display_w,
                                    height: display_h,
                                    kind: LayoutKind::Image {
                                        src: src.clone(),
                                        alt: alt.clone(),
                                    },
                                    href: None,
                                });
                                cursor_y += display_h + 8.0;
                            } else {
                                layout.push(LayoutBox {
                                    x: margin_x,
                                    y: cursor_y,
                                    width: content_width,
                                    height: 30.0,
                                    kind: LayoutKind::Image {
                                        src: src.clone(),
                                        alt: alt.clone(),
                                    },
                                    href: None,
                                });
                                cursor_y += 38.0;
                            }
                        }
                        BlockKind::FigCaption => {
                            let font_size = 15.0 * z;
                            let lines = estimate_lines(&child.text, content_width, font_size);
                            layout.push(LayoutBox {
                                x: margin_x,
                                y: cursor_y,
                                width: content_width,
                                height: lines * font_size * 1.5,
                                kind: LayoutKind::Text {
                                    text: child.text.clone(),
                                    font_size,
                                    color: theme.text_dim,
                                    bold: false,
                                    italic: true,
                                },
                                href: None,
                            });
                            cursor_y += lines * font_size * 1.5 + 4.0;
                        }
                        _ => {}
                    }
                }
                cursor_y += 8.0;
            }
            _ => {}
        }
    }

    layout
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npu::{ContentBlock, BlockKind};

    fn make_paragraph(text: &str) -> ContentBlock {
        ContentBlock {
            kind: BlockKind::Paragraph,
            text: text.to_string(),
            depth: 0,
            relevance: 0.8,
            children: Vec::new(),
            image_data: None,
            node_id: None,
            computed_style: None,
        }
    }

    #[test]
    fn test_zoom_increases_font_size() {
        let blocks = vec![make_paragraph("Hello world")];
        let theme = Theme::default();

        let layout_1x = compute_layout_zoom(&blocks, 0.0, 800.0, &theme, 1.0, &HashMap::new());
        let layout_2x = compute_layout_zoom(&blocks, 0.0, 800.0, &theme, 2.0, &HashMap::new());

        // Find the text box in each layout
        let text_1x = layout_1x.iter().find(|b| matches!(&b.kind, LayoutKind::Text { .. }));
        let text_2x = layout_2x.iter().find(|b| matches!(&b.kind, LayoutKind::Text { .. }));
        assert!(text_1x.is_some());
        assert!(text_2x.is_some());

        if let (
            Some(LayoutBox { kind: LayoutKind::Text { font_size: fs1, .. }, .. }),
            Some(LayoutBox { kind: LayoutKind::Text { font_size: fs2, .. }, .. }),
        ) = (text_1x, text_2x) {
            assert!(*fs2 > *fs1, "2x zoom should have larger font: {} vs {}", fs2, fs1);
            assert!((fs2 / fs1 - 2.0).abs() < 0.01, "Font should scale 2x");
        }
    }

    #[test]
    fn test_zoom_reset_matches_default() {
        let blocks = vec![make_paragraph("Test text")];
        let theme = Theme::default();

        let layout_default = compute_layout_zoom(&blocks, 0.0, 800.0, &theme, 1.0, &HashMap::new());
        let layout_zoom1 = compute_layout_zoom(&blocks, 0.0, 800.0, &theme, 1.0, &HashMap::new());

        assert_eq!(layout_default.len(), layout_zoom1.len());
        for (a, b) in layout_default.iter().zip(layout_zoom1.iter()) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
            assert_eq!(a.height, b.height);
        }
    }

    #[test]
    fn test_zoom_clamp_min_max() {
        let blocks = vec![make_paragraph("Clamp test")];
        let theme = Theme::default();

        // Very small zoom should clamp to 0.25
        let layout_tiny = compute_layout_zoom(&blocks, 0.0, 800.0, &theme, 0.01, &HashMap::new());
        let text = layout_tiny.iter().find(|b| matches!(&b.kind, LayoutKind::Text { .. }));
        if let Some(LayoutBox { kind: LayoutKind::Text { font_size, .. }, .. }) = text {
            assert!(*font_size >= 16.0 * 0.25 - 0.01, "Min zoom should be 0.25x");
        }
    }
}
