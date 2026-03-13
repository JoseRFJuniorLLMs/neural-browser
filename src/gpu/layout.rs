//! GPU: Layout engine — positions content blocks for rendering.
//!
//! Simple vertical flow layout. Not CSS — we don't need it.
//! The NPU already classified what each block IS,
//! so layout is just: "put headings big, paragraphs normal, images centered."

use crate::npu::{ContentBlock, BlockKind};
use crate::ui::Theme;

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
    Code { text: String, language: Option<String> },
    /// Background rect (for code blocks, quotes, etc.)
    Background { color: [f32; 4] },
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

/// Compute layout for all content blocks.
/// Returns a list of positioned LayoutBoxes in DOCUMENT coordinates (scroll-independent).
/// The scroll offset is applied at render time, not here — so layout only needs
/// recomputation on content change or viewport resize, NOT on scroll.
pub fn compute_layout(
    blocks: &[ContentBlock],
    _scroll_y: f32, // DEPRECATED: scroll applied at render time now
    viewport_width: f32,
    theme: &Theme,
) -> Vec<LayoutBox> {
    let mut layout = Vec::new();

    let margin_x: f32 = 40.0;
    let content_width: f32 = (viewport_width - margin_x * 2.0).max(200.0).min(900.0);

    // ── URL bar background (fixed at top) ──
    layout.push(LayoutBox {
        x: 0.0,
        y: 0.0,
        width: viewport_width,
        height: 40.0,
        kind: LayoutKind::Background {
            color: theme.url_bar_bg,
        },
        href: None,
    });

    // Content starts below URL bar — positions are in document space
    let mut cursor_y: f32 = 50.0;

    for block in blocks {
        // Skip low-relevance content
        if block.relevance < 0.15 {
            continue;
        }

        match &block.kind {
            BlockKind::Title => {
                // URL bar shows the title
            }
            BlockKind::Heading { level } => {
                let font_size = match level {
                    1 => 32.0,
                    2 => 26.0,
                    3 => 22.0,
                    4 => 18.0,
                    _ => 16.0,
                };
                // More spacing before headings
                let spacing_before = if *level <= 2 { font_size * 1.0 } else { font_size * 0.8 };
                cursor_y += spacing_before;

                let lines = estimate_lines(&block.text, content_width, font_size);
                let block_height = lines * font_size * 1.4;

                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: block_height,
                    kind: LayoutKind::Text {
                        text: block.text.clone(),
                        font_size,
                        color: theme.heading,
                        bold: true,
                        italic: false,
                    },
                    href: None,
                });

                cursor_y += block_height + font_size * 0.4;
            }
            BlockKind::Paragraph => {
                if block.text.is_empty() {
                    continue;
                }
                let font_size = 16.0;
                let line_height = font_size * 1.6;
                let lines = estimate_lines(&block.text, content_width, font_size);

                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: lines * line_height,
                    kind: LayoutKind::Text {
                        text: block.text.clone(),
                        font_size,
                        color: theme.text,
                        bold: false,
                        italic: false,
                    },
                    href: None,
                });

                cursor_y += lines * line_height + 14.0;
            }
            BlockKind::Code { language } => {
                let font_size = 14.0;
                let line_height = font_size * 1.5;
                let lines = block.text.lines().count().max(1) as f32;
                let block_height = lines * line_height + 24.0;

                // Background
                layout.push(LayoutBox {
                    x: margin_x - 12.0,
                    y: cursor_y - 4.0,
                    width: content_width + 24.0,
                    height: block_height,
                    kind: LayoutKind::Background {
                        color: theme.code_bg,
                    },
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
                    },
                    href: None,
                });

                cursor_y += block_height + 18.0;
            }
            BlockKind::Quote => {
                let font_size = 15.0;
                let line_height = font_size * 1.6;
                let lines = estimate_lines(&block.text, content_width - 40.0, font_size);
                let block_height = lines * line_height + 16.0;

                // Left border + background
                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: block_height,
                    kind: LayoutKind::Background {
                        color: theme.quote_bg,
                    },
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
                        color: [0.7, 0.7, 0.75, 1.0],
                        bold: false,
                        italic: true,
                    },
                    href: None,
                });

                cursor_y += block_height + 14.0;
            }
            BlockKind::Image { src, alt } => {
                if let Some((img_w, img_h, rgba)) = &block.image_data {
                    // Calculate aspect-ratio-preserving dimensions
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
                                font_size: 12.0,
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
                    // Fallback: text placeholder
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
                    if child.relevance < 0.15 {
                        continue;
                    }
                    let font_size = 15.0;
                    let prefix = if *ordered {
                        format!("{}. {}", idx + 1, child.text)
                    } else {
                        format!("\u{2022}  {}", child.text)
                    };
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
                let font_size = 15.0;
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
                let font_size = 15.0;
                let lines = estimate_lines(&block.text, content_width, font_size);
                let line_height = font_size * 1.5;

                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: lines * line_height,
                    kind: LayoutKind::Text {
                        text: block.text.clone(),
                        font_size,
                        color: theme.link,
                        bold: false,
                        italic: false,
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
                    let font_size = 14.0;
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
                                },
                                href: None,
                            });
                            cursor_y += lines * line_h + 2.0;
                        }
                    }
                    cursor_y += 8.0;
                } else if !block.text.is_empty() {
                    // Fallback: render table text as code block
                    let font_size = 14.0;
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
                        },
                        href: None,
                    });
                    cursor_y += block_height + 12.0;
                }
            }
            BlockKind::TableRow => {
                // Standalone table row (outside Table)
                if !block.text.is_empty() {
                    let font_size = 14.0;
                    let lines = estimate_lines(&block.text, content_width, font_size);
                    layout.push(LayoutBox {
                        x: margin_x,
                        y: cursor_y,
                        width: content_width,
                        height: lines * font_size * 1.5,
                        kind: LayoutKind::Code {
                            text: block.text.clone(),
                            language: None,
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
                            let font_size = 15.0;
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
                            let font_size = 14.0;
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
                            let font_size = 15.0;
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
                                let font_size = 15.0;
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
                // Render form description
                if !block.text.is_empty() {
                    let font_size = 14.0;
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
                            let font_size = 12.0;
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
