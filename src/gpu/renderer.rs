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
use std::collections::{HashMap, HashSet};
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
    pub zoom_level: f32,
    pub eva_visible: bool,
}

/// Vertex for image quad rendering.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ImageVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

/// Vertex for solid-color rectangles (navbar, buttons, panels).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RectVertex {
    position: [f32; 2],
    color: [f32; 4],
}

/// A cached GPU texture with its bind group and dimensions.
struct CachedTexture {
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    #[allow(dead_code)] // must keep texture alive while bind_group references it
    texture: wgpu::Texture,
}

/// Toolbar height in pixels (navbar with buttons + URL bar).
pub const TOOLBAR_HEIGHT: f32 = 52.0;

/// Status bar height in pixels.
pub const STATUS_BAR_HEIGHT: f32 = 22.0;

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
    // Rectangle rendering pipeline (navbar, buttons, panels)
    rect_pipeline: wgpu::RenderPipeline,
    // Texture cache: URL -> CachedTexture (avoids re-uploading every frame)
    texture_cache: HashMap<String, CachedTexture>,
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

        // ── Rect rendering pipeline (solid color quads) ──
        let rect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rect_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("rect_shader.wgsl").into(),
            ),
        });

        let rect_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rect_pipeline_layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let rect_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("rect_pipeline"),
                layout: Some(&rect_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &rect_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<RectVertex>() as wgpu::BufferAddress,
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
                                format: wgpu::VertexFormat::Float32x4,
                            },
                        ],
                    }],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &rect_shader,
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
                    ..Default::default()
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
            rect_pipeline,
            texture_cache: HashMap::new(),
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

    /// Build 6 vertices for a solid-color rectangle at pixel coordinates.
    fn build_rect(&self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) -> [RectVertex; 6] {
        let tl = self.pixel_to_ndc(x, y);
        let tr = self.pixel_to_ndc(x + w, y);
        let bl = self.pixel_to_ndc(x, y + h);
        let br = self.pixel_to_ndc(x + w, y + h);
        [
            RectVertex { position: tl, color },
            RectVertex { position: bl, color },
            RectVertex { position: tr, color },
            RectVertex { position: tr, color },
            RectVertex { position: bl, color },
            RectVertex { position: br, color },
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

    /// Decode image bytes and upload as a GPU texture, caching by URL.
    /// Supports PNG, JPEG, WebP, GIF (via the `image` crate).
    pub fn upload_image(&mut self, url: &str, data: &[u8]) {
        if self.texture_cache.contains_key(url) {
            return; // already cached
        }

        let img = match image::load_from_memory(data) {
            Ok(img) => img.to_rgba8(),
            Err(e) => {
                log::warn!("[GPU] Failed to decode image {url}: {e}");
                return;
            }
        };

        let width = img.width();
        let height = img.height();
        if width == 0 || height == 0 {
            return;
        }

        let (texture, bind_group) = self.create_image_texture(width, height, &img);

        self.texture_cache.insert(url.to_string(), CachedTexture {
            bind_group,
            width,
            height,
            texture,
        });

        log::info!("[GPU] Uploaded texture: {url} ({width}x{height})");
    }

    /// Get the dimensions of a cached texture by URL.
    pub fn texture_dimensions(&self, url: &str) -> Option<(u32, u32)> {
        self.texture_cache.get(url).map(|t| (t.width, t.height))
    }

    /// Clear all cached textures (e.g. on page navigation).
    pub fn clear_texture_cache(&mut self) {
        self.texture_cache.clear();
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
        let mut rect_vertices: Vec<RectVertex> = Vec::new();

        let wf = w as f32;
        let hf = h as f32;
        let tb = TOOLBAR_HEIGHT;
        let sb = STATUS_BAR_HEIGHT;

        // ── Chrome-style toolbar ──
        rect_vertices.extend_from_slice(&self.build_rect(0.0, 0.0, wf, tb, ctx.theme.toolbar_bg));

        let btn_w: f32 = 36.0;
        let btn_h: f32 = 32.0;
        let btn_y: f32 = (tb - btn_h) / 2.0;
        let btn_gap: f32 = 4.0;
        let margin_left: f32 = 8.0;
        let back_x = margin_left;
        let fwd_x = back_x + btn_w + btn_gap;
        let ref_x = fwd_x + btn_w + btn_gap;

        // Nav button backgrounds
        rect_vertices.extend_from_slice(&self.build_rect(back_x, btn_y, btn_w, btn_h, ctx.theme.button_bg));
        rect_vertices.extend_from_slice(&self.build_rect(fwd_x, btn_y, btn_w, btn_h, ctx.theme.button_bg));
        rect_vertices.extend_from_slice(&self.build_rect(ref_x, btn_y, btn_w, btn_h, ctx.theme.button_bg));

        // Nav button labels
        let btn_text_y = btn_y + 6.0;
        for (label, bx) in [("\u{25C0}", back_x), ("\u{25B6}", fwd_x), ("\u{21BB}", ref_x)] {
            let mut btn_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(16.0, 20.0));
            btn_buf.set_size(&mut self.font_system, Some(btn_w), Some(btn_h));
            btn_buf.set_text(&mut self.font_system, label,
                Attrs::new().family(Family::SansSerif).weight(Weight::NORMAL), Shaping::Advanced);
            btn_buf.shape_until_scroll(&mut self.font_system, false);
            let tc = ctx.theme.button_text;
            buffers.push((btn_buf, bx + 10.0, btn_text_y,
                TextBounds { left: bx as i32, top: btn_y as i32, right: (bx + btn_w) as i32, bottom: (btn_y + btn_h) as i32 },
                GlyphonColor::rgb((tc[0]*255.0) as u8, (tc[1]*255.0) as u8, (tc[2]*255.0) as u8)));
        }

        // EVA button (right side of toolbar)
        let eva_btn_w: f32 = 52.0;
        let eva_btn_x = wf - eva_btn_w - 8.0;
        let eva_bg = if ctx.eva_visible {
            [0.25, 0.65, 0.55, 1.0] // green-teal when active
        } else {
            ctx.theme.button_bg
        };
        rect_vertices.extend_from_slice(&self.build_rect(eva_btn_x, btn_y, eva_btn_w, btn_h, eva_bg));

        let mut eva_btn_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(13.0, 18.0));
        eva_btn_buf.set_size(&mut self.font_system, Some(eva_btn_w), Some(btn_h));
        let eva_label = if ctx.eva_visible { "EVA \u{2715}" } else { "\u{2726} EVA" };
        eva_btn_buf.set_text(&mut self.font_system, eva_label,
            Attrs::new().family(Family::SansSerif).weight(Weight::BOLD), Shaping::Advanced);
        eva_btn_buf.shape_until_scroll(&mut self.font_system, false);
        let eva_text_color = if ctx.eva_visible {
            GlyphonColor::rgb(255, 255, 255)
        } else {
            let tc = ctx.theme.button_text;
            GlyphonColor::rgb((tc[0]*255.0) as u8, (tc[1]*255.0) as u8, (tc[2]*255.0) as u8)
        };
        buffers.push((eva_btn_buf, eva_btn_x + 6.0, btn_text_y,
            TextBounds { left: eva_btn_x as i32, top: btn_y as i32,
                right: (eva_btn_x + eva_btn_w) as i32, bottom: (btn_y + btn_h) as i32 },
            eva_text_color));

        // URL bar background (between nav buttons and EVA button)
        let url_x = ref_x + btn_w + btn_gap + 8.0;
        let url_w = eva_btn_x - url_x - 8.0;
        let url_bg = if ctx.url_editing {
            [ctx.theme.url_bar_bg[0]+0.04, ctx.theme.url_bar_bg[1]+0.04, ctx.theme.url_bar_bg[2]+0.04, 1.0]
        } else { ctx.theme.url_bar_bg };
        rect_vertices.extend_from_slice(&self.build_rect(url_x, btn_y, url_w, btn_h, url_bg));

        // Toolbar bottom border
        rect_vertices.extend_from_slice(&self.build_rect(0.0, tb - 1.0, wf, 1.0, ctx.theme.url_bar_border));

        let (url_display, url_color) = if ctx.url_editing {
            if ctx.url_input.is_empty() {
                ("\u{1F50D} Search or type a URL".to_string(), GlyphonColor::rgb(120, 120, 140))
            } else {
                (format!("{}\u{2588}", ctx.url_input), GlyphonColor::rgb(230, 230, 240))
            }
        } else if ctx.loading {
            (format!("\u{23F3} {}", ctx.url), GlyphonColor::rgb(160, 190, 255))
        } else {
            (format!("\u{1F512} {}", ctx.url), GlyphonColor::rgb(180, 195, 225))
        };

        let mut url_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));
        url_buf.set_size(&mut self.font_system, Some(url_w - 16.0), Some(btn_h));
        url_buf.set_text(
            &mut self.font_system, &url_display,
            Attrs::new().family(Family::SansSerif).weight(Weight::NORMAL),
            Shaping::Advanced,
        );
        url_buf.shape_until_scroll(&mut self.font_system, false);
        buffers.push((
            url_buf, url_x + 8.0, btn_y + 7.0,
            TextBounds {
                left: url_x as i32, top: btn_y as i32,
                right: (url_x + url_w) as i32, bottom: (btn_y + btn_h) as i32,
            },
            url_color,
        ));

        // Loading indicator
        if ctx.loading {
            let mut load_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));
            load_buf.set_size(&mut self.font_system, Some(wf - 40.0), Some(30.0));
            let dots = match (ctx.loading_ticks / 15) % 4 {
                0 => "Loading", 1 => "Loading.", 2 => "Loading..", _ => "Loading...",
            };
            load_buf.set_text(&mut self.font_system, dots,
                Attrs::new().family(Family::SansSerif).weight(Weight::NORMAL), Shaping::Advanced);
            load_buf.shape_until_scroll(&mut self.font_system, false);
            let lc = ctx.theme.loading;
            buffers.push((
                load_buf, (wf / 2.0) - 40.0, (hf / 2.0) - 10.0,
                TextBounds { left: 0, top: tb as i32, right: w as i32, bottom: h as i32 - sb as i32 },
                GlyphonColor::rgb((lc[0]*255.0) as u8, (lc[1]*255.0) as u8, (lc[2]*255.0) as u8),
            ));
        }

        // Status bar background
        rect_vertices.extend_from_slice(&self.build_rect(0.0, hf - sb, wf, sb, ctx.theme.toolbar_bg));

        // Status bar text
        let mut status_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(11.0, 14.0));
        status_buf.set_size(&mut self.font_system, Some(wf), Some(sb));
        let n_blocks = layout.len();

        // Show hovered link URL in status bar, or default status
        let zoom_info = if (ctx.zoom_level - 1.0).abs() > 0.01 {
            format!(" | Zoom: {:.0}%", ctx.zoom_level * 100.0)
        } else {
            String::new()
        };
        let status_text = if let Some(href) = ctx.hovered_href {
            format!("  {href}{zoom_info}")
        } else {
            format!(
                "  CPU: net+parse | NPU: content AI | GPU: render | {n_blocks} elements{zoom_info}"
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
            hf - STATUS_BAR_HEIGHT,
            TextBounds {
                left: 0,
                top: h as i32 - STATUS_BAR_HEIGHT as i32,
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
                    italic,
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
                    let style = if *italic {
                        glyphon::Style::Italic
                    } else {
                        glyphon::Style::Normal
                    };
                    buf.set_text(
                        &mut self.font_system,
                        text,
                        Attrs::new().family(Family::SansSerif).weight(weight).style(style),
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
                            top: top.max(TOOLBAR_HEIGHT as i32),
                            right: (lbox.x + lbox.width) as i32,
                            bottom: bottom.min(h as i32 - STATUS_BAR_HEIGHT as i32),
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
                        if ul_top > TOOLBAR_HEIGHT as i32 && ul_top < h as i32 - STATUS_BAR_HEIGHT as i32 {
                            buffers.push((
                                ul_buf,
                                lbox.x,
                                underline_y,
                                TextBounds {
                                    left: lbox.x as i32,
                                    top: ul_top,
                                    right: (lbox.x + lbox.width) as i32,
                                    bottom: (ul_top + 4).min(h as i32 - STATUS_BAR_HEIGHT as i32),
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
                            top: top.max(TOOLBAR_HEIGHT as i32),
                            right: (lbox.x + lbox.width) as i32,
                            bottom: bottom.min(h as i32 - STATUS_BAR_HEIGHT as i32),
                        },
                        GlyphonColor::rgb(200, 220, 170),
                    ));
                }
                LayoutKind::Image { src, alt } => {
                    // Check if we have a cached texture for this URL
                    if self.texture_cache.contains_key(src) {
                        // Will be rendered in the image pass below
                    } else {
                        // Show text placeholder
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
                                top: top.max(TOOLBAR_HEIGHT as i32),
                                right: (lbox.x + lbox.width) as i32,
                                bottom: (top + 30).min(h as i32 - STATUS_BAR_HEIGHT as i32),
                            },
                            GlyphonColor::rgb(120, 120, 140),
                        ));
                    }
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

        // ── Pass 1: Clear + render rectangles (toolbar, buttons, URL bar) ──
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_and_rects"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
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

            if !rect_vertices.is_empty() {
                let rect_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rect_vertex_buffer"),
                    contents: bytemuck::cast_slice(&rect_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                pass.set_pipeline(&self.rect_pipeline);
                pass.set_vertex_buffer(0, rect_buffer.slice(..));
                pass.draw(0..rect_vertices.len() as u32, 0..1);
            }
        }

        // ── Pass 2: Render text (glyphon) on top of rects ──
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("text_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
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

        // ── Pass 3: Render images as textured quads ──
        // Includes both DecodedImage blocks (NPU-decoded) and Image blocks with cached textures.

        // First, collect decoded images (creates new textures per-frame)
        let mut decoded_renders: Vec<(wgpu::Buffer, wgpu::BindGroup, wgpu::Texture)> = Vec::new();
        // Second, collect cached image URLs and their vertex buffers
        let mut cached_renders: Vec<(wgpu::Buffer, String)> = Vec::new();

        for lbox in layout {
            match &lbox.kind {
                LayoutKind::DecodedImage { width: img_w, height: img_h, rgba, .. } => {
                    let top = lbox.y;
                    let bottom = lbox.y + lbox.height;
                    if bottom < 40.0 || top > h as f32 {
                        continue;
                    }

                    let (texture, bind_group) = self.create_image_texture(*img_w, *img_h, rgba);
                    let vertices = self.build_image_quad(lbox.x, lbox.y, lbox.width, lbox.height);
                    let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("image_vertex_buffer"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    decoded_renders.push((vertex_buffer, bind_group, texture));
                }
                LayoutKind::Image { src, .. } => {
                    // Render from texture cache if available
                    if self.texture_cache.contains_key(src) {
                        let top = lbox.y;
                        let bottom = lbox.y + lbox.height;
                        if bottom < 40.0 || top > h as f32 {
                            continue;
                        }

                        let vertices = self.build_image_quad(lbox.x, lbox.y, lbox.width, lbox.height);
                        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("cached_image_vertex_buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                        cached_renders.push((vertex_buffer, src.clone()));
                    }
                }
                _ => {}
            }
        }

        let has_images = !decoded_renders.is_empty() || !cached_renders.is_empty();
        if has_images {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("image_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
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

            // Render decoded images (per-frame textures)
            for (vertex_buffer, bind_group, _texture) in &decoded_renders {
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.draw(0..6, 0..1);
            }

            // Render cached texture images
            for (vertex_buffer, src) in &cached_renders {
                if let Some(cached) = self.texture_cache.get(src) {
                    pass.set_bind_group(0, &cached.bind_group, &[]);
                    pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    pass.draw(0..6, 0..1);
                }
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

        let panel_width: f32 = (350.0_f32).min(w as f32 * 0.4);
        let panel_x = (w as f32 - panel_width).max(0.0);

        let mut buffers: Vec<(GlyphonBuffer, f32, f32, TextBounds, GlyphonColor)> = Vec::new();
        let mut rect_vertices: Vec<RectVertex> = Vec::new();

        let wf = w as f32;
        let hf = h as f32;
        let tb = TOOLBAR_HEIGHT;
        let sb = STATUS_BAR_HEIGHT;

        // ── Toolbar ──
        rect_vertices.extend_from_slice(&self.build_rect(0.0, 0.0, wf, tb, ctx.theme.toolbar_bg));

        let btn_w: f32 = 36.0;
        let btn_h: f32 = 32.0;
        let btn_y: f32 = (tb - btn_h) / 2.0;
        let btn_gap: f32 = 4.0;
        let margin_left: f32 = 8.0;
        let back_x = margin_left;
        let fwd_x = back_x + btn_w + btn_gap;
        let ref_x = fwd_x + btn_w + btn_gap;

        // Nav buttons
        rect_vertices.extend_from_slice(&self.build_rect(back_x, btn_y, btn_w, btn_h, ctx.theme.button_bg));
        rect_vertices.extend_from_slice(&self.build_rect(fwd_x, btn_y, btn_w, btn_h, ctx.theme.button_bg));
        rect_vertices.extend_from_slice(&self.build_rect(ref_x, btn_y, btn_w, btn_h, ctx.theme.button_bg));

        let btn_text_y = btn_y + 6.0;
        for (label, bx) in [("\u{25C0}", back_x), ("\u{25B6}", fwd_x), ("\u{21BB}", ref_x)] {
            let mut btn_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(16.0, 20.0));
            btn_buf.set_size(&mut self.font_system, Some(btn_w), Some(btn_h));
            btn_buf.set_text(&mut self.font_system, label,
                Attrs::new().family(Family::SansSerif).weight(Weight::NORMAL), Shaping::Advanced);
            btn_buf.shape_until_scroll(&mut self.font_system, false);
            let tc = ctx.theme.button_text;
            buffers.push((btn_buf, bx + 10.0, btn_text_y,
                TextBounds { left: bx as i32, top: btn_y as i32, right: (bx + btn_w) as i32, bottom: (btn_y + btn_h) as i32 },
                GlyphonColor::rgb((tc[0]*255.0) as u8, (tc[1]*255.0) as u8, (tc[2]*255.0) as u8)));
        }

        // EVA button (active state)
        let eva_btn_w: f32 = 52.0;
        let eva_btn_x = panel_x - eva_btn_w - 8.0;
        rect_vertices.extend_from_slice(&self.build_rect(eva_btn_x.max(0.0), btn_y, eva_btn_w, btn_h,
            [0.25, 0.65, 0.55, 1.0]));
        let mut eva_btn_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(13.0, 18.0));
        eva_btn_buf.set_size(&mut self.font_system, Some(eva_btn_w), Some(btn_h));
        eva_btn_buf.set_text(&mut self.font_system, "EVA \u{2715}",
            Attrs::new().family(Family::SansSerif).weight(Weight::BOLD), Shaping::Advanced);
        eva_btn_buf.shape_until_scroll(&mut self.font_system, false);
        buffers.push((eva_btn_buf, eva_btn_x.max(0.0) + 6.0, btn_text_y,
            TextBounds { left: eva_btn_x.max(0.0) as i32, top: btn_y as i32,
                right: (eva_btn_x + eva_btn_w) as i32, bottom: (btn_y + btn_h) as i32 },
            GlyphonColor::rgb(255, 255, 255)));

        // URL bar
        let url_x = ref_x + btn_w + btn_gap + 8.0;
        let url_w = (eva_btn_x - url_x - 8.0).max(50.0);
        let url_bg = if ctx.url_editing {
            [ctx.theme.url_bar_bg[0]+0.04, ctx.theme.url_bar_bg[1]+0.04, ctx.theme.url_bar_bg[2]+0.04, 1.0]
        } else { ctx.theme.url_bar_bg };
        rect_vertices.extend_from_slice(&self.build_rect(url_x, btn_y, url_w, btn_h, url_bg));

        // Toolbar border
        rect_vertices.extend_from_slice(&self.build_rect(0.0, tb - 1.0, wf, 1.0, ctx.theme.url_bar_border));

        let (url_display, url_color) = if ctx.url_editing {
            if ctx.url_input.is_empty() {
                ("\u{1F50D} Search or type a URL".to_string(), GlyphonColor::rgb(120, 120, 140))
            } else {
                (format!("{}\u{2588}", ctx.url_input), GlyphonColor::rgb(230, 230, 240))
            }
        } else if ctx.loading {
            (format!("\u{23F3} {}", ctx.url), GlyphonColor::rgb(160, 190, 255))
        } else {
            (format!("\u{1F512} {}", ctx.url), GlyphonColor::rgb(180, 195, 225))
        };

        let mut url_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));
        url_buf.set_size(&mut self.font_system, Some(url_w - 16.0), Some(btn_h));
        url_buf.set_text(&mut self.font_system, &url_display,
            Attrs::new().family(Family::SansSerif).weight(Weight::NORMAL), Shaping::Advanced);
        url_buf.shape_until_scroll(&mut self.font_system, false);
        buffers.push((
            url_buf, url_x + 8.0, btn_y + 7.0,
            TextBounds { left: url_x as i32, top: btn_y as i32, right: (url_x + url_w) as i32, bottom: (btn_y + btn_h) as i32 },
            url_color,
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
                TextBounds { left: 0, top: TOOLBAR_HEIGHT as i32, right: panel_x as i32, bottom: h as i32 - STATUS_BAR_HEIGHT as i32 },
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
                        TextBounds { left: lbox.x as i32, top: top.max(TOOLBAR_HEIGHT as i32), right: panel_x as i32, bottom: bottom.min(h as i32 - STATUS_BAR_HEIGHT as i32) },
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
                        TextBounds { left: lbox.x as i32, top: top.max(TOOLBAR_HEIGHT as i32), right: panel_x as i32, bottom: bottom.min(h as i32 - STATUS_BAR_HEIGHT as i32) },
                        GlyphonColor::rgb(200, 220, 170),
                    ));
                }
                _ => {}
            }
        }

        // Status bar
        let mut status_buf = GlyphonBuffer::new(&mut self.font_system, Metrics::new(11.0, 14.0));
        status_buf.set_size(&mut self.font_system, Some(w as f32), Some(20.0));
        let zoom_info2 = if (ctx.zoom_level - 1.0).abs() > 0.01 {
            format!(" | Zoom: {:.0}%", ctx.zoom_level * 100.0)
        } else {
            String::new()
        };
        let status_text = if let Some(href) = ctx.hovered_href {
            format!("  {href}{zoom_info2}")
        } else {
            format!("  CPU+NPU+GPU | EVA: Ctrl+E | {} elements{zoom_info2}", layout.len())
        };
        status_buf.set_text(&mut self.font_system, &status_text,
            Attrs::new().family(Family::Monospace).weight(Weight::NORMAL), Shaping::Advanced);
        status_buf.shape_until_scroll(&mut self.font_system, false);
        buffers.push((status_buf, 0.0, h as f32 - STATUS_BAR_HEIGHT,
            TextBounds { left: 0, top: h as i32 - STATUS_BAR_HEIGHT as i32, right: w as i32, bottom: h as i32 },
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
            TextBounds { left: panel_x as i32, top: 0, right: w as i32, bottom: TOOLBAR_HEIGHT as i32 },
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

        // ── Pass 0: Clear + rectangles (toolbar, buttons, status bar) ──
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rect_pass_eva"),
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

            // EVA panel background
            let panel_bg = [0.10, 0.10, 0.18, 1.0_f32];
            let panel_rect_verts = self.build_rect(panel_x, 0.0, panel_width, hf, panel_bg);
            let mut all_rects = rect_vertices.clone();
            all_rects.extend_from_slice(&panel_rect_verts);

            // Status bar background
            all_rects.extend_from_slice(&self.build_rect(0.0, hf - sb, wf, sb, ctx.theme.toolbar_bg));

            if !all_rects.is_empty() {
                let rect_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rect_vb_eva"),
                    contents: bytemuck::cast_slice(&all_rects),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                pass.set_pipeline(&self.rect_pipeline);
                pass.set_vertex_buffer(0, rect_buf.slice(..));
                pass.draw(0..all_rects.len() as u32, 0..1);
            }
        }

        // ── Pass 1: Text ──
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("text_pass_eva"),
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
            self.text_renderer.render(&self.atlas, &self.viewport, &mut pass)
                .map_err(|e| anyhow::anyhow!("glyphon render: {e:?}"))?;
        }

        // ── Pass 2: Images (clipped to left of EVA panel) ──
        let mut decoded_renders_eva: Vec<(wgpu::Buffer, wgpu::BindGroup, wgpu::Texture)> = Vec::new();
        let mut cached_renders_eva: Vec<(wgpu::Buffer, String)> = Vec::new();

        for lbox in layout {
            match &lbox.kind {
                LayoutKind::DecodedImage { width: img_w, height: img_h, rgba, .. } => {
                    let top = lbox.y;
                    let bottom = lbox.y + lbox.height;
                    if bottom < 40.0 || top > h as f32 { continue; }
                    let clipped_w = lbox.width.min(panel_x - lbox.x);
                    if clipped_w <= 0.0 { continue; }

                    let (texture, bind_group) = self.create_image_texture(*img_w, *img_h, rgba);
                    let vertices = self.build_image_quad(lbox.x, lbox.y, clipped_w, lbox.height);
                    let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("image_vertex_buffer_eva"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    decoded_renders_eva.push((vertex_buffer, bind_group, texture));
                }
                LayoutKind::Image { src, .. } => {
                    if self.texture_cache.contains_key(src) {
                        let top = lbox.y;
                        let bottom = lbox.y + lbox.height;
                        if bottom < 40.0 || top > h as f32 { continue; }
                        let clipped_w = lbox.width.min(panel_x - lbox.x);
                        if clipped_w <= 0.0 { continue; }

                        let vertices = self.build_image_quad(lbox.x, lbox.y, clipped_w, lbox.height);
                        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("cached_image_vertex_buffer_eva"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                        cached_renders_eva.push((vertex_buffer, src.clone()));
                    }
                }
                _ => {}
            }
        }

        let has_images_eva = !decoded_renders_eva.is_empty() || !cached_renders_eva.is_empty();
        if has_images_eva {
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
            for (vertex_buffer, bind_group, _texture) in &decoded_renders_eva {
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.draw(0..6, 0..1);
            }
            for (vertex_buffer, src) in &cached_renders_eva {
                if let Some(cached) = self.texture_cache.get(src) {
                    pass.set_bind_group(0, &cached.bind_group, &[]);
                    pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    pass.draw(0..6, 0..1);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        self.atlas.trim();

        Ok(())
    }
}
