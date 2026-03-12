//! GPU: wgpu + glyphon renderer — turns LayoutBoxes into pixels.
//!
//! Uses wgpu for hardware-accelerated rendering:
//! - Text via glyphon (GPU text rasterization using cosmic-text)
//! - Background clear via render pass
//! All rendering is GPU-accelerated.

use super::layout::{LayoutBox, LayoutKind};
use crate::ui::Theme;
use anyhow::Result;
use glyphon::{
    Attrs, Buffer as GlyphonBuffer, Cache, Color as GlyphonColor, Family, FontSystem, Metrics,
    Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
    Weight,
};
use std::collections::HashSet;
use std::sync::Arc;
use winit::window::Window;

/// Rendering context passed each frame — carries ephemeral state that
/// changes between frames (hover, loading, visited set).
pub struct RenderContext<'a> {
    pub url: &'a str,
    pub loading: bool,
    pub loading_ticks: u64,
    pub hovered_href: Option<&'a str>,
    pub visited_urls: &'a HashSet<String>,
    pub theme: &'a Theme,
}

pub struct WgpuRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: (u32, u32),
    // glyphon text rendering
    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    viewport: Viewport,
    text_renderer: TextRenderer,
}

impl WgpuRenderer {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12 | wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No suitable GPU adapter found"))?;

        log::info!(
            "[GPU] Adapter: {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend,
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("neural-browser-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ── glyphon setup ──
        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let mut atlas = TextAtlas::new(&device, &queue, &cache, surface_format);
        let viewport = Viewport::new(&device, &cache);
        let text_renderer = TextRenderer::new(
            &mut atlas,
            &device,
            wgpu::MultisampleState::default(),
            None,
        );

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size: (size.width, size.height),
            font_system,
            swash_cache,
            atlas,
            viewport,
            text_renderer,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.size = (width, height);
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Return current surface size.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Determine the color for a link box, considering hover and visited state.
    fn link_color(
        &self,
        href: &Option<String>,
        ctx: &RenderContext<'_>,
        base_color: [f32; 4],
    ) -> [f32; 4] {
        if let Some(h) = href {
            // Hovered link?
            if let Some(hovered) = ctx.hovered_href {
                if hovered == h {
                    return ctx.theme.link_hover;
                }
            }
            // Visited link?
            if ctx.visited_urls.contains(h) {
                return ctx.theme.link_visited;
            }
        }
        base_color
    }

    pub fn render(&mut self, layout: &[LayoutBox], ctx: &RenderContext<'_>) -> Result<()> {
        let (w, h) = self.size;
        if w == 0 || h == 0 {
            return Ok(());
        }

        // Update viewport resolution
        self.viewport.update(
            &self.queue,
            Resolution {
                width: w,
                height: h,
            },
        );

        // ── Build text buffers from layout ──
        let mut buffers: Vec<(GlyphonBuffer, f32, f32, TextBounds, GlyphonColor)> = Vec::new();

        // URL bar text
        let mut url_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));
        url_buf.set_size(&mut self.font_system, Some(w as f32 - 20.0), Some(30.0));
        url_buf.set_text(
            &mut self.font_system,
            &format!("  \u{1F310} {}", ctx.url),
            Attrs::new().family(Family::Monospace).weight(Weight::NORMAL),
            Shaping::Advanced,
        );
        url_buf.shape_until_scroll(&mut self.font_system, false);
        buffers.push((
            url_buf,
            0.0,
            10.0,
            TextBounds {
                left: 0,
                top: 0,
                right: w as i32,
                bottom: 40,
            },
            GlyphonColor::rgb(160, 190, 255),
        ));

        // URL bar border indicator (thin line at bottom of URL bar)
        {
            let border_color = ctx.theme.url_bar_border;
            let mut border_buf =
                GlyphonBuffer::new(&mut self.font_system, Metrics::new(2.0, 2.0));
            border_buf.set_size(&mut self.font_system, Some(w as f32), Some(2.0));
            let bar_char = "\u{2500}".repeat((w / 6).max(1) as usize); // horizontal line chars
            border_buf.set_text(
                &mut self.font_system,
                &bar_char,
                Attrs::new().family(Family::Monospace).weight(Weight::NORMAL),
                Shaping::Advanced,
            );
            border_buf.shape_until_scroll(&mut self.font_system, false);
            let r = (border_color[0] * 255.0) as u8;
            let g = (border_color[1] * 255.0) as u8;
            let b = (border_color[2] * 255.0) as u8;
            buffers.push((
                border_buf,
                0.0,
                38.0,
                TextBounds {
                    left: 0,
                    top: 36,
                    right: w as i32,
                    bottom: 42,
                },
                GlyphonColor::rgb(r, g, b),
            ));
        }

        // Loading indicator
        if ctx.loading {
            let mut load_buf =
                GlyphonBuffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));
            load_buf.set_size(&mut self.font_system, Some(w as f32 - 40.0), Some(30.0));
            let dots = match (ctx.loading_ticks / 15) % 4 {
                0 => "Loading",
                1 => "Loading.",
                2 => "Loading..",
                _ => "Loading...",
            };
            load_buf.set_text(
                &mut self.font_system,
                dots,
                Attrs::new()
                    .family(Family::SansSerif)
                    .weight(Weight::NORMAL),
                Shaping::Advanced,
            );
            load_buf.shape_until_scroll(&mut self.font_system, false);
            let lc = ctx.theme.loading;
            buffers.push((
                load_buf,
                (w as f32 / 2.0) - 40.0,
                (h as f32 / 2.0) - 10.0,
                TextBounds {
                    left: 0,
                    top: 40,
                    right: w as i32,
                    bottom: h as i32 - 25,
                },
                GlyphonColor::rgb(
                    (lc[0] * 255.0) as u8,
                    (lc[1] * 255.0) as u8,
                    (lc[2] * 255.0) as u8,
                ),
            ));
        }

        // Status bar
        let mut status_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(11.0, 14.0));
        status_buf.set_size(&mut self.font_system, Some(w as f32), Some(20.0));
        let n_blocks = layout.len();

        // Show hovered link URL in status bar, or default status
        let status_text = if let Some(href) = ctx.hovered_href {
            format!("  {href}")
        } else {
            format!(
                "  CPU: net+parse | NPU: content AI | GPU: render | {n_blocks} elements"
            )
        };
        status_buf.set_text(
            &mut self.font_system,
            &status_text,
            Attrs::new()
                .family(Family::Monospace)
                .weight(Weight::NORMAL),
            Shaping::Advanced,
        );
        status_buf.shape_until_scroll(&mut self.font_system, false);
        buffers.push((
            status_buf,
            0.0,
            h as f32 - 20.0,
            TextBounds {
                left: 0,
                top: h as i32 - 25,
                right: w as i32,
                bottom: h as i32,
            },
            GlyphonColor::rgb(90, 90, 110),
        ));

        // Content layout boxes -> text buffers
        for lbox in layout {
            match &lbox.kind {
                LayoutKind::Text {
                    text,
                    font_size,
                    color,
                    bold,
                    ..
                } => {
                    if text.is_empty() {
                        continue;
                    }
                    let top = lbox.y as i32;
                    let bottom = (lbox.y + lbox.height) as i32;
                    if bottom < 0 || top > h as i32 {
                        continue;
                    }

                    let line_h = font_size * 1.5;
                    let mut buf = GlyphonBuffer::new(
                        &mut self.font_system,
                        Metrics::new(*font_size, line_h),
                    );
                    buf.set_size(
                        &mut self.font_system,
                        Some(lbox.width),
                        Some(lbox.height.max(line_h * 2.0)),
                    );

                    // For links, add underline prefix character
                    let display_text;
                    let is_link = lbox.href.is_some();
                    if is_link {
                        // Show underlined text by using combining underline
                        display_text = text.clone();
                    } else {
                        display_text = text.clone();
                    }

                    let weight = if *bold { Weight::BOLD } else { Weight::NORMAL };
                    buf.set_text(
                        &mut self.font_system,
                        &display_text,
                        Attrs::new().family(Family::SansSerif).weight(weight),
                        Shaping::Advanced,
                    );
                    buf.shape_until_scroll(&mut self.font_system, false);

                    // Determine final color (hover/visited override for links)
                    let final_color = self.link_color(&lbox.href, ctx, *color);
                    let r = (final_color[0] * 255.0) as u8;
                    let g = (final_color[1] * 255.0) as u8;
                    let b = (final_color[2] * 255.0) as u8;

                    buffers.push((
                        buf,
                        lbox.x,
                        lbox.y,
                        TextBounds {
                            left: lbox.x as i32,
                            top: top.max(40),
                            right: (lbox.x + lbox.width) as i32,
                            bottom: bottom.min(h as i32 - 25),
                        },
                        GlyphonColor::rgb(r, g, b),
                    ));

                    // Link underline: render a thin line of underscores below the text
                    if is_link {
                        let underline_y = lbox.y + lbox.height - 2.0;
                        let mut ul_buf = GlyphonBuffer::new(
                            &mut self.font_system,
                            Metrics::new(2.0, 2.0),
                        );
                        ul_buf.set_size(
                            &mut self.font_system,
                            Some(lbox.width),
                            Some(4.0),
                        );
                        // Use thin horizontal line characters for underline
                        let ul_len = (lbox.width / 6.0).max(1.0) as usize;
                        let ul_str = "\u{2500}".repeat(ul_len);
                        ul_buf.set_text(
                            &mut self.font_system,
                            &ul_str,
                            Attrs::new()
                                .family(Family::Monospace)
                                .weight(Weight::NORMAL),
                            Shaping::Advanced,
                        );
                        ul_buf.shape_until_scroll(&mut self.font_system, false);

                        let ul_top = underline_y as i32;
                        if ul_top > 40 && ul_top < h as i32 - 25 {
                            buffers.push((
                                ul_buf,
                                lbox.x,
                                underline_y,
                                TextBounds {
                                    left: lbox.x as i32,
                                    top: ul_top,
                                    right: (lbox.x + lbox.width) as i32,
                                    bottom: (ul_top + 4).min(h as i32 - 25),
                                },
                                GlyphonColor::rgb(r, g, b),
                            ));
                        }
                    }
                }
                LayoutKind::Code { text, .. } => {
                    if text.is_empty() {
                        continue;
                    }
                    let top = lbox.y as i32;
                    let bottom = (lbox.y + lbox.height) as i32;
                    if bottom < 0 || top > h as i32 {
                        continue;
                    }

                    let mut buf = GlyphonBuffer::new(
                        &mut self.font_system,
                        Metrics::new(13.0, 19.0),
                    );
                    buf.set_size(
                        &mut self.font_system,
                        Some(lbox.width),
                        Some(lbox.height.max(40.0)),
                    );
                    buf.set_text(
                        &mut self.font_system,
                        text,
                        Attrs::new()
                            .family(Family::Monospace)
                            .weight(Weight::NORMAL),
                        Shaping::Advanced,
                    );
                    buf.shape_until_scroll(&mut self.font_system, false);

                    buffers.push((
                        buf,
                        lbox.x,
                        lbox.y,
                        TextBounds {
                            left: lbox.x as i32,
                            top: top.max(40),
                            right: (lbox.x + lbox.width) as i32,
                            bottom: bottom.min(h as i32 - 25),
                        },
                        GlyphonColor::rgb(200, 220, 170),
                    ));
                }
                LayoutKind::Image { alt, .. } => {
                    let label = if alt.is_empty() {
                        "[image]".to_string()
                    } else {
                        format!("[img: {alt}]")
                    };
                    let top = lbox.y as i32;
                    if top > h as i32 || (top + 30) < 0 {
                        continue;
                    }

                    let mut buf = GlyphonBuffer::new(
                        &mut self.font_system,
                        Metrics::new(13.0, 18.0),
                    );
                    buf.set_size(&mut self.font_system, Some(lbox.width), Some(30.0));
                    buf.set_text(
                        &mut self.font_system,
                        &label,
                        Attrs::new()
                            .family(Family::SansSerif)
                            .weight(Weight::NORMAL),
                        Shaping::Advanced,
                    );
                    buf.shape_until_scroll(&mut self.font_system, false);

                    buffers.push((
                        buf,
                        lbox.x,
                        lbox.y,
                        TextBounds {
                            left: lbox.x as i32,
                            top: top.max(40),
                            right: (lbox.x + lbox.width) as i32,
                            bottom: (top + 30).min(h as i32 - 25),
                        },
                        GlyphonColor::rgb(120, 120, 140),
                    ));
                }
                _ => {}
            }
        }

        // ── Build TextArea refs ──
        let text_areas: Vec<TextArea<'_>> = buffers
            .iter()
            .map(|(buf, x, y, bounds, color)| TextArea {
                buffer: buf,
                left: *x,
                top: *y,
                scale: 1.0,
                bounds: *bounds,
                default_color: *color,
                custom_glyphs: &[],
            })
            .collect();

        // ── Prepare text for GPU ──
        self.text_renderer
            .prepare(
                &self.device,
                &self.queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                text_areas,
                &mut self.swash_cache,
            )
            .map_err(|e| anyhow::anyhow!("glyphon prepare: {e:?}"))?;

        // ── Render pass ──
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        let bg = ctx.theme.bg;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg[0] as f64,
                            g: bg[1] as f64,
                            b: bg[2] as f64,
                            a: bg[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.text_renderer
                .render(&self.atlas, &self.viewport, &mut pass)
                .map_err(|e| anyhow::anyhow!("glyphon render: {e:?}"))?;
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        self.atlas.trim();

        Ok(())
    }
}
