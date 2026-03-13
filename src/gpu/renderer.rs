//! GPU: wgpu + glyphon renderer — turns LayoutBoxes into pixels.
//!
//! Uses wgpu for hardware-accelerated rendering:
//! - Text via glyphon (GPU text rasterization using cosmic-text)
//! - Background clear via render pass
//! All rendering is GPU-accelerated.

use super::layout::{LayoutBox, LayoutKind};
#[allow(unused_imports)] // Will be used when EVA panel rendering is implemented
use crate::eva::panel::{EvaPanel, Role};
use crate::ui::Theme;
use anyhow::Result;
use wgpu::util::DeviceExt;
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
    pub url_editing: bool,
    pub url_input: &'a str,
}

/// Vertex for image quad rendering.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ImageVertex {
    position: [f32; 2],
    uv: [f32; 2],
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
    // Image rendering pipeline
    image_pipeline: wgpu::RenderPipeline,
    image_bind_group_layout: wgpu::BindGroupLayout,
    image_sampler: wgpu::Sampler,
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

        // ── Image rendering pipeline ──
        let image_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("image_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("image_shader.wgsl").into(),
            ),
        });

        let image_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("image_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let image_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("image_pipeline_layout"),
                bind_group_layouts: &[&image_bind_group_layout],
                push_constant_ranges: &[],
            });

        let image_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("image_pipeline"),
                layout: Some(&image_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &image_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ImageVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            wgpu::VertexAttribute {
                                offset: 8,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                        ],
                    }],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &image_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("image_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

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
            image_pipeline,
            image_bind_group_layout,
            image_sampler,
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

    /// Create a wgpu texture and bind group from decoded RGBA pixel data.
    fn create_image_texture(
        &self,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> (wgpu::Texture, wgpu::BindGroup) {
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image_texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            texture_size,
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("image_bind_group"),
            layout: &self.image_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.image_sampler),
                },
            ],
        });

        (texture, bind_group)
    }

    /// Convert pixel coordinates to NDC (normalized device coordinates).
    /// Screen (0,0) is top-left, NDC (-1,1) is top-left.
    fn pixel_to_ndc(&self, x: f32, y: f32) -> [f32; 2] {
        let (w, h) = self.size;
        [
            (x / w as f32) * 2.0 - 1.0,
            1.0 - (y / h as f32) * 2.0,
        ]
    }

    /// Build quad vertices for an image at the given pixel position and size.
    fn build_image_quad(&self, x: f32, y: f32, w: f32, h: f32) -> [ImageVertex; 6] {
        let tl = self.pixel_to_ndc(x, y);
        let tr = self.pixel_to_ndc(x + w, y);
        let bl = self.pixel_to_ndc(x, y + h);
        let br = self.pixel_to_ndc(x + w, y + h);

        [
            // Triangle 1: TL, BL, TR
            ImageVertex { position: tl, uv: [0.0, 0.0] },
            ImageVertex { position: bl, uv: [0.0, 1.0] },
            ImageVertex { position: tr, uv: [1.0, 0.0] },
            // Triangle 2: TR, BL, BR
            ImageVertex { position: tr, uv: [1.0, 0.0] },
            ImageVertex { position: bl, uv: [0.0, 1.0] },
            ImageVertex { position: br, uv: [1.0, 1.0] },
        ]
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

        // ── URL / Search bar (Chrome-style omnibox) ──
        // Show: "🔍 Search Google or type a URL" when empty editing
        //        "🌐 https://example.com" when displaying URL
        //        "🔍 typed query█" when editing with text
        let (url_display, url_color) = if ctx.url_editing {
            if ctx.url_input.is_empty() {
                // Placeholder text like Chrome
                (
                    "  \u{1F50D} Search Google or type a URL".to_string(),
                    GlyphonColor::rgb(120, 120, 140), // Dim gray placeholder
                )
            } else {
                // User is typing — show input with cursor
                (
                    format!("  \u{1F50D} {}\u{2588}", ctx.url_input),
                    GlyphonColor::rgb(230, 230, 240), // Bright white text
                )
            }
        } else if ctx.loading {
            (
                format!("  \u{23F3} {}", ctx.url),
                GlyphonColor::rgb(160, 190, 255),
            )
        } else {
            (
                format!("  \u{1F310} {}", ctx.url),
                GlyphonColor::rgb(160, 190, 255),
            )
        };

        let mut url_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));
        url_buf.set_size(&mut self.font_system, Some(w as f32 - 20.0), Some(30.0));
        url_buf.set_text(
            &mut self.font_system,
            &url_display,
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
            url_color,
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

                    let is_link = lbox.href.is_some();
                    let weight = if *bold { Weight::BOLD } else { Weight::NORMAL };
                    buf.set_text(
                        &mut self.font_system,
                        text, // pass reference directly — no clone needed
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
                LayoutKind::DecodedImage { .. } => {
                    // Handled in image render pass below
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

        // ── Pass 1: Clear + render text (glyphon) ──
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

        // ── Pass 2: Render decoded images as textured quads ──
        // Collect image data to render (textures + vertex buffers)
        let mut image_renders: Vec<(wgpu::Buffer, wgpu::BindGroup, wgpu::Texture)> = Vec::new();

        for lbox in layout {
            if let LayoutKind::DecodedImage { width: img_w, height: img_h, rgba, .. } = &lbox.kind {
                // Skip if completely off-screen
                let top = lbox.y;
                let bottom = lbox.y + lbox.height;
                if bottom < 40.0 || top > h as f32 {
                    continue;
                }

                // Create GPU texture from RGBA data
                let (texture, bind_group) = self.create_image_texture(*img_w, *img_h, rgba);

                // Build quad vertices in NDC
                let vertices = self.build_image_quad(lbox.x, lbox.y, lbox.width, lbox.height);
                let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("image_vertex_buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                image_renders.push((vertex_buffer, bind_group, texture));
            }
        }

        if !image_renders.is_empty() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("image_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Don't clear — render on top of text
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.image_pipeline);
            for (vertex_buffer, bind_group, _texture) in &image_renders {
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.draw(0..6, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        self.atlas.trim();

        Ok(())
    }

    /// Render with EVA panel overlay.
    /// Calls the normal render pipeline when panel is hidden,
    /// or renders page content clipped to the left + EVA panel on the right.
    pub fn render_with_eva(
        &mut self,
        layout: &[LayoutBox],
        ctx: &RenderContext<'_>,
        eva_panel: &EvaPanel,
    ) -> Result<()> {
        if !eva_panel.visible {
            return self.render(layout, ctx);
        }

        let (w, h) = self.size;
        if w == 0 || h == 0 {
            return Ok(());
        }

        self.viewport.update(
            &self.queue,
            Resolution { width: w, height: h },
        );

        let panel_width: f32 = 350.0;
        let panel_x = w as f32 - panel_width;

        let mut buffers: Vec<(GlyphonBuffer, f32, f32, TextBounds, GlyphonColor)> = Vec::new();

        // URL bar text (clipped to left of panel)
        let mut url_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));
        url_buf.set_size(&mut self.font_system, Some(panel_x - 20.0), Some(30.0));
        url_buf.set_text(
            &mut self.font_system,
            &format!("  \u{1F310} {}", ctx.url),
            Attrs::new().family(Family::Monospace).weight(Weight::NORMAL),
            Shaping::Advanced,
        );
        url_buf.shape_until_scroll(&mut self.font_system, false);
        buffers.push((
            url_buf, 0.0, 10.0,
            TextBounds { left: 0, top: 0, right: panel_x as i32, bottom: 40 },
            GlyphonColor::rgb(160, 190, 255),
        ));

        // Loading indicator
        if ctx.loading {
            let mut load_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));
            load_buf.set_size(&mut self.font_system, Some(panel_x - 40.0), Some(30.0));
            let dots = match (ctx.loading_ticks / 15) % 4 {
                0 => "Loading", 1 => "Loading.", 2 => "Loading..", _ => "Loading...",
            };
            load_buf.set_text(&mut self.font_system, dots,
                Attrs::new().family(Family::SansSerif).weight(Weight::NORMAL), Shaping::Advanced);
            load_buf.shape_until_scroll(&mut self.font_system, false);
            let lc = ctx.theme.loading;
            buffers.push((load_buf, (panel_x / 2.0) - 40.0, (h as f32 / 2.0) - 10.0,
                TextBounds { left: 0, top: 40, right: panel_x as i32, bottom: h as i32 - 25 },
                GlyphonColor::rgb((lc[0]*255.0) as u8, (lc[1]*255.0) as u8, (lc[2]*255.0) as u8),
            ));
        }

        // Content layout boxes (clipped to left of EVA panel)
        for lbox in layout {
            match &lbox.kind {
                LayoutKind::Text { text, font_size, color, bold, .. } => {
                    if text.is_empty() { continue; }
                    let top = lbox.y as i32;
                    let bottom = (lbox.y + lbox.height) as i32;
                    if bottom < 0 || top > h as i32 { continue; }
                    let ew = lbox.width.min(panel_x - lbox.x);
                    if ew <= 0.0 { continue; }

                    let line_h = font_size * 1.5;
                    let mut buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(*font_size, line_h));
                    buf.set_size(&mut self.font_system, Some(ew), Some(lbox.height.max(line_h * 2.0)));
                    let weight = if *bold { Weight::BOLD } else { Weight::NORMAL };
                    buf.set_text(&mut self.font_system, text,
                        Attrs::new().family(Family::SansSerif).weight(weight), Shaping::Advanced);
                    buf.shape_until_scroll(&mut self.font_system, false);

                    let fc = self.link_color(&lbox.href, ctx, *color);
                    buffers.push((buf, lbox.x, lbox.y,
                        TextBounds { left: lbox.x as i32, top: top.max(40), right: panel_x as i32, bottom: bottom.min(h as i32 - 25) },
                        GlyphonColor::rgb((fc[0]*255.0) as u8, (fc[1]*255.0) as u8, (fc[2]*255.0) as u8),
                    ));
                }
                LayoutKind::Code { text, .. } => {
                    if text.is_empty() { continue; }
                    let top = lbox.y as i32;
                    let bottom = (lbox.y + lbox.height) as i32;
                    if bottom < 0 || top > h as i32 { continue; }
                    let ew = lbox.width.min(panel_x - lbox.x);
                    if ew <= 0.0 { continue; }

                    let mut buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(13.0, 19.0));
                    buf.set_size(&mut self.font_system, Some(ew), Some(lbox.height.max(40.0)));
                    buf.set_text(&mut self.font_system, text,
                        Attrs::new().family(Family::Monospace).weight(Weight::NORMAL), Shaping::Advanced);
                    buf.shape_until_scroll(&mut self.font_system, false);
                    buffers.push((buf, lbox.x, lbox.y,
                        TextBounds { left: lbox.x as i32, top: top.max(40), right: panel_x as i32, bottom: bottom.min(h as i32 - 25) },
                        GlyphonColor::rgb(200, 220, 170),
                    ));
                }
                _ => {}
            }
        }

        // Status bar
        let mut status_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(11.0, 14.0));
        status_buf.set_size(&mut self.font_system, Some(w as f32), Some(20.0));
        let status_text = if let Some(href) = ctx.hovered_href {
            format!("  {href}")
        } else {
            format!("  CPU+NPU+GPU | EVA: Ctrl+E | {} elements", layout.len())
        };
        status_buf.set_text(&mut self.font_system, &status_text,
            Attrs::new().family(Family::Monospace).weight(Weight::NORMAL), Shaping::Advanced);
        status_buf.shape_until_scroll(&mut self.font_system, false);
        buffers.push((status_buf, 0.0, h as f32 - 20.0,
            TextBounds { left: 0, top: h as i32 - 25, right: w as i32, bottom: h as i32 },
            GlyphonColor::rgb(90, 90, 110),
        ));

        // ── AI panel header — shows current provider ──
        let provider_name = eva_panel.provider.name();
        let header_text = format!("  {}  [Tab: switch] [Esc: close]", provider_name);
        let mut header_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(16.0, 22.0));
        header_buf.set_size(&mut self.font_system, Some(panel_width - 20.0), Some(30.0));
        header_buf.set_text(&mut self.font_system, &header_text,
            Attrs::new().family(Family::SansSerif).weight(Weight::BOLD), Shaping::Advanced);
        header_buf.shape_until_scroll(&mut self.font_system, false);
        buffers.push((header_buf, panel_x + 10.0, 8.0,
            TextBounds { left: panel_x as i32, top: 0, right: w as i32, bottom: 40 },
            GlyphonColor::rgb(100, 200, 255),
        ));

        // Panel header separator
        {
            let mut sep_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(2.0, 2.0));
            sep_buf.set_size(&mut self.font_system, Some(panel_width), Some(2.0));
            let sep_chars = "\u{2500}".repeat((panel_width / 6.0).max(1.0) as usize);
            sep_buf.set_text(&mut self.font_system, &sep_chars,
                Attrs::new().family(Family::Monospace).weight(Weight::NORMAL), Shaping::Advanced);
            sep_buf.shape_until_scroll(&mut self.font_system, false);
            buffers.push((sep_buf, panel_x, 38.0,
                TextBounds { left: panel_x as i32, top: 36, right: w as i32, bottom: 42 },
                GlyphonColor::rgb(60, 60, 80),
            ));
        }

        // ── EVA messages ──
        let msg_top = 48.0;
        let msg_bottom = h as f32 - 60.0;
        let mut cursor_y = msg_top;

        for msg in &eva_panel.messages {
            if cursor_y > msg_bottom { break; }

            let (color, prefix) = match msg.role {
                Role::User => (GlyphonColor::rgb(180, 220, 255), "You".to_string()),
                Role::Ai   => {
                    let name = msg.provider.map(|p| p.name()).unwrap_or("AI");
                    (GlyphonColor::rgb(150, 255, 180), name.to_string())
                }
                Role::System => (GlyphonColor::rgb(130, 130, 150), String::new()),
            };
            let align_x = panel_x + 10.0;
            let display = if prefix.is_empty() {
                msg.text.clone()
            } else {
                format!("{prefix}: {}", msg.text)
            };
            let msg_width = panel_width - 30.0;
            let font_size = 13.0;
            let line_h = font_size * 1.5;
            let chars_per_line = (msg_width / (font_size * 0.55)).max(1.0);
            let lines = (display.len() as f32 / chars_per_line).ceil().max(1.0);
            let block_height = lines * line_h;

            let mut buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(font_size, line_h));
            buf.set_size(&mut self.font_system, Some(msg_width), Some(block_height.max(line_h * 2.0)));
            buf.set_text(&mut self.font_system, &display,
                Attrs::new().family(Family::SansSerif).weight(Weight::NORMAL), Shaping::Advanced);
            buf.shape_until_scroll(&mut self.font_system, false);

            let top = cursor_y as i32;
            let bottom = (cursor_y + block_height) as i32;
            if top < msg_bottom as i32 && bottom > msg_top as i32 {
                buffers.push((buf, align_x, cursor_y,
                    TextBounds {
                        left: panel_x as i32,
                        top: top.max(msg_top as i32),
                        right: w as i32,
                        bottom: bottom.min(msg_bottom as i32),
                    },
                    color,
                ));
            }
            cursor_y += block_height + 8.0;
        }

        // EVA loading indicator
        if eva_panel.is_loading {
            let mut load_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(12.0, 16.0));
            load_buf.set_size(&mut self.font_system, Some(panel_width - 20.0), Some(20.0));
            load_buf.set_text(&mut self.font_system, "EVA is thinking...",
                Attrs::new().family(Family::SansSerif).weight(Weight::NORMAL), Shaping::Advanced);
            load_buf.shape_until_scroll(&mut self.font_system, false);
            let y = cursor_y.min(msg_bottom - 20.0);
            buffers.push((load_buf, panel_x + 10.0, y,
                TextBounds { left: panel_x as i32, top: y as i32, right: w as i32, bottom: (y + 20.0) as i32 },
                GlyphonColor::rgb(100, 180, 255),
            ));
        }

        // ── EVA input field ──
        {
            let input_sep_y = h as f32 - 55.0;
            let mut sep2_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(2.0, 2.0));
            sep2_buf.set_size(&mut self.font_system, Some(panel_width), Some(2.0));
            let sep2_chars = "\u{2500}".repeat((panel_width / 6.0).max(1.0) as usize);
            sep2_buf.set_text(&mut self.font_system, &sep2_chars,
                Attrs::new().family(Family::Monospace).weight(Weight::NORMAL), Shaping::Advanced);
            sep2_buf.shape_until_scroll(&mut self.font_system, false);
            buffers.push((sep2_buf, panel_x, input_sep_y,
                TextBounds { left: panel_x as i32, top: input_sep_y as i32, right: w as i32, bottom: (input_sep_y + 4.0) as i32 },
                GlyphonColor::rgb(60, 60, 80),
            ));

            let input_display = format!("> {}\u{2588}", eva_panel.get_input());
            let mut input_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(13.0, 18.0));
            input_buf.set_size(&mut self.font_system, Some(panel_width - 20.0), Some(30.0));
            input_buf.set_text(&mut self.font_system, &input_display,
                Attrs::new().family(Family::Monospace).weight(Weight::NORMAL), Shaping::Advanced);
            input_buf.shape_until_scroll(&mut self.font_system, false);
            let input_y = h as f32 - 45.0;
            buffers.push((input_buf, panel_x + 10.0, input_y,
                TextBounds { left: panel_x as i32, top: input_y as i32, right: w as i32, bottom: (input_y + 30.0) as i32 },
                GlyphonColor::rgb(200, 210, 230),
            ));
        }

        // ── Build TextArea refs ──
        let text_areas: Vec<TextArea<'_>> = buffers
            .iter()
            .map(|(buf, x, y, bounds, color)| TextArea {
                buffer: buf, left: *x, top: *y, scale: 1.0, bounds: *bounds,
                default_color: *color, custom_glyphs: &[],
            })
            .collect();

        // ── Prepare + render ──
        self.text_renderer.prepare(
            &self.device, &self.queue, &mut self.font_system,
            &mut self.atlas, &self.viewport, text_areas, &mut self.swash_cache,
        ).map_err(|e| anyhow::anyhow!("glyphon prepare: {e:?}"))?;

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("render_encoder_eva") });

        let bg = ctx.theme.bg;

        // ── Pass 1: Clear + text ──
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass_eva"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg[0] as f64, g: bg[1] as f64, b: bg[2] as f64, a: bg[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.text_renderer.render(&self.atlas, &self.viewport, &mut pass)
                .map_err(|e| anyhow::anyhow!("glyphon render: {e:?}"))?;
        }

        // ── Pass 2: Images (clipped to left of EVA panel) ──
        let mut image_renders: Vec<(wgpu::Buffer, wgpu::BindGroup, wgpu::Texture)> = Vec::new();

        for lbox in layout {
            if let LayoutKind::DecodedImage { width: img_w, height: img_h, rgba, .. } = &lbox.kind {
                let top = lbox.y;
                let bottom = lbox.y + lbox.height;
                if bottom < 40.0 || top > h as f32 { continue; }
                // Clip width to panel boundary
                let clipped_w = lbox.width.min(panel_x - lbox.x);
                if clipped_w <= 0.0 { continue; }

                let (texture, bind_group) = self.create_image_texture(*img_w, *img_h, rgba);
                let vertices = self.build_image_quad(lbox.x, lbox.y, clipped_w, lbox.height);
                let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("image_vertex_buffer_eva"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                image_renders.push((vertex_buffer, bind_group, texture));
            }
        }

        if !image_renders.is_empty() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("image_pass_eva"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.image_pipeline);
            for (vertex_buffer, bind_group, _texture) in &image_renders {
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.draw(0..6, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        self.atlas.trim();

        Ok(())
    }
}
