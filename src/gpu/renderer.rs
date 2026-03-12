//! GPU: wgpu + glyphon renderer — turns LayoutBoxes into pixels.
//!
//! Uses wgpu for hardware-accelerated rendering:
//! - Text via glyphon (GPU text rasterization using cosmic-text)
//! - Background clear via render pass
//! All rendering is GPU-accelerated.

use super::layout::{LayoutBox, LayoutKind};
use anyhow::Result;
use glyphon::{
    Attrs, Buffer as GlyphonBuffer, Cache, Color as GlyphonColor, Family, FontSystem, Metrics,
    Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
    Weight,
};
use std::sync::Arc;
use winit::window::Window;

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

    pub fn render(&mut self, url: &str, layout: &[LayoutBox]) -> Result<()> {
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

        // URL bar
        let mut url_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));
        url_buf.set_size(&mut self.font_system, Some(w as f32 - 20.0), Some(30.0));
        url_buf.set_text(
            &mut self.font_system,
            &format!("  \u{1F310} {url}"),
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

        // Status bar
        let mut status_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(11.0, 14.0));
        status_buf.set_size(&mut self.font_system, Some(w as f32), Some(20.0));
        let n_blocks = layout.len();
        status_buf.set_text(
            &mut self.font_system,
            &format!("  CPU: net+parse | NPU: content AI | GPU: render | {n_blocks} elements"),
            Attrs::new().family(Family::Monospace).weight(Weight::NORMAL),
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

        // Content layout boxes → text buffers
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

                    let weight = if *bold { Weight::BOLD } else { Weight::NORMAL };
                    buf.set_text(
                        &mut self.font_system,
                        text,
                        Attrs::new().family(Family::SansSerif).weight(weight),
                        Shaping::Advanced,
                    );
                    buf.shape_until_scroll(&mut self.font_system, false);

                    let r = (color[0] * 255.0) as u8;
                    let g = (color[1] * 255.0) as u8;
                    let b = (color[2] * 255.0) as u8;

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

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.06,
                            g: 0.06,
                            b: 0.08,
                            a: 1.0,
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
