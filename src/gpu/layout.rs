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
    /// Image placeholder (actual decode happens in renderer)
    Image { src: String, alt: String },
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
        let line_len = line.len() as f32;
        total_lines += (line_len / chars_per_line).ceil().max(1.0);
    }
    total_lines.max(1.0)
}

/// Compute layout for all content blocks.
/// Returns a list of positioned LayoutBoxes ready for GPU rendering.
pub fn compute_layout(
    blocks: &[ContentBlock],
    scroll_y: f32,
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

    // URL text is set by caller
    let mut cursor_y: f32 = 50.0;

    // Apply scroll offset
    cursor_y -= scroll_y;

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
                let img_height = 300.0; // Placeholder height

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

                if !alt.is_empty() {
                    layout.push(LayoutBox {
                        x: margin_x,
                        y: cursor_y + img_height + 4.0,
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
                    cursor_y += img_height + 28.0;
                } else {
                    cursor_y += img_height + 12.0;
                }
            }
            BlockKind::List { .. } => {
                // List container — children (ListItem) handle rendering
            }
            BlockKind::ListItem => {
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
            _ => {}
        }
    }

    layout
}
