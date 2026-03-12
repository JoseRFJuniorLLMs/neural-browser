//! GPU: Layout engine — positions content blocks for rendering.
//!
//! Simple vertical flow layout. Not CSS — we don't need it.
//! The NPU already classified what each block IS,
//! so layout is just: "put headings big, paragraphs normal, images centered."

use crate::npu::{ContentBlock, BlockKind};

/// A positioned element ready for GPU rendering.
#[derive(Debug, Clone)]
pub struct LayoutBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub kind: LayoutKind,
}

#[derive(Debug, Clone)]
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

/// Compute layout for all content blocks.
/// Returns a list of positioned LayoutBoxes ready for GPU rendering.
pub fn compute_layout(blocks: &[ContentBlock], scroll_y: f32) -> Vec<LayoutBox> {
    let mut layout = Vec::new();
    let mut cursor_y: f32 = 0.0;

    let margin_x: f32 = 40.0;
    let content_width: f32 = 800.0;

    // ── URL bar (fixed at top) ──
    layout.push(LayoutBox {
        x: 0.0,
        y: 0.0,
        width: content_width + margin_x * 2.0,
        height: 40.0,
        kind: LayoutKind::Background {
            color: [0.15, 0.15, 0.18, 1.0],
        },
    });

    // URL text will be set by caller
    cursor_y = 50.0;

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
                let spacing_before = font_size * 0.8;
                cursor_y += spacing_before;

                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: font_size * 1.4,
                    kind: LayoutKind::Text {
                        text: block.text.clone(),
                        font_size,
                        color: [0.95, 0.95, 0.97, 1.0],
                        bold: true,
                        italic: false,
                    },
                });

                cursor_y += font_size * 1.6;
            }
            BlockKind::Paragraph => {
                if block.text.is_empty() {
                    continue;
                }
                let font_size = 16.0;
                let line_height = font_size * 1.6;
                let estimated_lines = (block.text.len() as f32 / 80.0).ceil().max(1.0);

                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: estimated_lines * line_height,
                    kind: LayoutKind::Text {
                        text: block.text.clone(),
                        font_size,
                        color: [0.85, 0.85, 0.88, 1.0],
                        bold: false,
                        italic: false,
                    },
                });

                cursor_y += estimated_lines * line_height + 12.0;
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
                        color: [0.1, 0.1, 0.12, 1.0],
                    },
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
                });

                cursor_y += block_height + 16.0;
            }
            BlockKind::Quote => {
                let font_size = 15.0;
                let line_height = font_size * 1.6;
                let estimated_lines = (block.text.len() as f32 / 70.0).ceil().max(1.0);
                let block_height = estimated_lines * line_height + 16.0;

                // Left border + background
                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: block_height,
                    kind: LayoutKind::Background {
                        color: [0.12, 0.12, 0.15, 1.0],
                    },
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
                });

                cursor_y += block_height + 12.0;
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
                            color: [0.5, 0.5, 0.55, 1.0],
                            bold: false,
                            italic: true,
                        },
                    });
                    cursor_y += img_height + 28.0;
                } else {
                    cursor_y += img_height + 12.0;
                }
            }
            BlockKind::ListItem => {
                let font_size = 15.0;
                layout.push(LayoutBox {
                    x: margin_x + 24.0,
                    y: cursor_y,
                    width: content_width - 24.0,
                    height: font_size * 1.5,
                    kind: LayoutKind::Text {
                        text: format!("• {}", block.text),
                        font_size,
                        color: [0.85, 0.85, 0.88, 1.0],
                        bold: false,
                        italic: false,
                    },
                });
                cursor_y += font_size * 1.8;
            }
            BlockKind::Link { href: _ } => {
                if block.text.is_empty() {
                    continue;
                }
                let font_size = 15.0;
                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: font_size * 1.5,
                    kind: LayoutKind::Text {
                        text: block.text.clone(),
                        font_size,
                        color: [0.4, 0.6, 1.0, 1.0], // Blue for links
                        bold: false,
                        italic: false,
                    },
                });
                cursor_y += font_size * 1.8;
            }
            BlockKind::Separator => {
                cursor_y += 8.0;
                layout.push(LayoutBox {
                    x: margin_x,
                    y: cursor_y,
                    width: content_width,
                    height: 1.0,
                    kind: LayoutKind::Separator,
                });
                cursor_y += 16.0;
            }
            _ => {}
        }
    }

    layout
}
