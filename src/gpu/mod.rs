//! GPU Renderer — turns semantic ContentBlocks into pixels via wgpu.
//!
//! The GPU does ALL visual work:
//! - Text rasterization (glyphon)
//! - Image decoding + compositing
//! - Layout calculation (flexbox-lite)
//! - Scrolling + animation
//! - Final composite to framebuffer
//!
//! The GPU receives pre-understood content from NPU —
//! it never sees raw HTML or needs to parse anything.

mod renderer;
mod layout;
#[allow(dead_code)]
pub mod tabs;

use crate::eva::panel::EvaPanel;
use crate::npu::ContentBlock;
use crate::ui::Theme;
use crate::PipelineMsg;
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use log::{info, error};
use std::collections::{HashMap, HashSet};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{CursorIcon, Window, WindowId};
use std::sync::Arc;

/// GPU rendering state.
struct GpuApp {
    window: Option<Arc<Window>>,
    renderer: Option<renderer::WgpuRenderer>,
    content: Vec<ContentBlock>,
    scroll_y: f32,
    npu_rx: Receiver<PipelineMsg>,
    ui_tx: Sender<PipelineMsg>,
    url_bar: String,
    url_editing: bool,
    url_input: String,
    loading: bool,
    loading_ticks: u64,
    // Mouse state
    mouse_x: f32,
    mouse_y: f32,
    hovered_href: Option<String>,
    // Navigation history
    history: Vec<String>,
    history_idx: usize,
    // Visited URLs
    visited_urls: HashSet<String>,
    // Keyboard modifiers
    modifiers: ModifiersState,
    // Theme
    theme: Theme,
    // Cached layout (recomputed on content/scroll/resize change)
    cached_layout: Vec<layout::LayoutBox>,
    layout_dirty: bool,
    // EVA AI assistant
    eva_panel: EvaPanel,
    eva_tx: Sender<PipelineMsg>,
    eva_resp_rx: Receiver<PipelineMsg>,
    // Collected page text for EVA context
    page_text_cache: String,
    // Reading mode active (shows only high-relevance content)
    reading_mode: bool,
    // Page language detected by NPU (future: auto-translate, NietzscheDB metadata)
    #[allow(dead_code)]
    page_language: Option<String>,
    // Whether proactive insights have been sent for current page
    insights_sent_for_url: Option<String>,
    // DPI scale factor for HiDPI displays
    scale_factor: f64,
    // Page zoom level (1.0 = 100%, 0.5 = 50%, 2.0 = 200%)
    zoom_level: f32,
    // Image pipeline: CPU -> GPU (fetched image bytes)
    img_rx: Receiver<PipelineMsg>,
    // Image URLs that have been requested but not yet received
    pending_image_urls: HashSet<String>,
}

impl GpuApp {
    fn new(
        npu_rx: Receiver<PipelineMsg>,
        ui_tx: Sender<PipelineMsg>,
        eva_tx: Sender<PipelineMsg>,
        eva_resp_rx: Receiver<PipelineMsg>,
        img_rx: Receiver<PipelineMsg>,
    ) -> Self {
        let start_url = "https://example.com".to_string();
        let mut visited = HashSet::new();
        visited.insert(start_url.clone());
        Self {
            window: None,
            renderer: None,
            content: Vec::new(),
            scroll_y: 0.0,
            npu_rx,
            ui_tx,
            url_bar: start_url.clone(),
            url_editing: false,
            url_input: String::new(),
            loading: true,
            loading_ticks: 0,
            mouse_x: 0.0,
            mouse_y: 0.0,
            hovered_href: None,
            history: vec![start_url],
            history_idx: 0,
            visited_urls: visited,
            modifiers: ModifiersState::empty(),
            theme: Theme::default(),
            cached_layout: Vec::new(),
            layout_dirty: true,
            eva_panel: EvaPanel::new(),
            eva_tx,
            eva_resp_rx,
            page_text_cache: String::new(),
            reading_mode: false,
            page_language: None,
            insights_sent_for_url: None,
            scale_factor: 1.0,
            zoom_level: 1.0,
            img_rx,
            pending_image_urls: HashSet::new(),
        }
    }

    fn check_npu_messages(&mut self) -> bool {
        let mut got_content = false;
        while let Ok(msg) = self.npu_rx.try_recv() {
            if let PipelineMsg::ContentReady { url, blocks, ads_blocked } = msg {
                info!(
                    "[GPU] Received {} blocks from NPU ({} ads blocked)",
                    blocks.len(),
                    ads_blocked
                );
                self.url_bar = url;
                // Cache structured page text for EVA context
                let mut ctx_parts = Vec::new();
                let mut ctx_len = 0usize;
                for b in blocks.iter().filter(|b| !b.text.is_empty()) {
                    if ctx_len > 4000 { break; } // limit context size
                    let prefix = match &b.kind {
                        crate::npu::BlockKind::Title => "[Title] ",
                        crate::npu::BlockKind::Heading { .. } => "[Heading] ",
                        crate::npu::BlockKind::Code { .. } => "[Code] ",
                        crate::npu::BlockKind::Quote => "[Quote] ",
                        crate::npu::BlockKind::Link { .. } => "[Link] ",
                        _ => "",
                    };
                    let line = format!("{}{}", prefix, b.text);
                    ctx_len += line.len();
                    ctx_parts.push(line);
                }
                self.page_text_cache = ctx_parts.join("\n");
                // Update window title from page title
                let page_title = blocks.iter()
                    .find(|b| matches!(b.kind, crate::npu::BlockKind::Title))
                    .map(|b| b.text.as_str())
                    .unwrap_or(&self.url_bar);
                if let Some(window) = &self.window {
                    window.set_title(&format!("{} — Neural Browser", page_title));
                }
                // Clear texture cache for previous page
                if let Some(renderer) = &mut self.renderer {
                    renderer.clear_texture_cache();
                }
                self.pending_image_urls.clear();

                // Scan blocks for image URLs and request fetches
                self.request_image_fetches(&blocks);

                self.content = blocks;
                self.scroll_y = 0.0;
                self.loading = false;
                self.reading_mode = false;
                self.layout_dirty = true;
                got_content = true;

                // Auto-trigger proactive insights if panel is open
                if self.eva_panel.visible && !self.page_text_cache.is_empty() {
                    self.send_proactive_insights();
                }
            }
        }
        got_content
    }

    /// Check for AI responses from the CPU thread.
    fn check_eva_messages(&mut self) -> bool {
        // Check for loading timeout
        if self.eva_panel.check_loading_timeout() {
            self.request_redraw();
        }
        let mut got_response = false;
        while let Ok(msg) = self.eva_resp_rx.try_recv() {
            if let PipelineMsg::EvaResponse { text, provider } = msg {
                info!("[GPU] {} response received ({} chars)", provider.name(), text.len());
                self.eva_panel.add_ai_response(text, provider);
                got_response = true;
            }
        }
        got_response
    }

    /// Scan content blocks for image URLs that need fetching (no image_data from NPU).
    /// Sends FetchImage requests to CPU for each unfetched image.
    fn request_image_fetches(&mut self, blocks: &[ContentBlock]) {
        for block in blocks {
            if let crate::npu::BlockKind::Image { src, .. } = &block.kind {
                if block.image_data.is_none() && !self.pending_image_urls.contains(src)
                    && !src.is_empty() && (src.starts_with("http://") || src.starts_with("https://")) {
                        self.pending_image_urls.insert(src.clone());
                        let _ = self.ui_tx.send(PipelineMsg::FetchImage { url: src.clone() });
                        info!("[GPU] Requested image fetch: {src}");
                    }
            }
            // Also check children (e.g. Figure > Image)
            for child in &block.children {
                if let crate::npu::BlockKind::Image { src, .. } = &child.kind {
                    if child.image_data.is_none() && !self.pending_image_urls.contains(src)
                        && !src.is_empty() && (src.starts_with("http://") || src.starts_with("https://")) {
                            self.pending_image_urls.insert(src.clone());
                            let _ = self.ui_tx.send(PipelineMsg::FetchImage { url: src.clone() });
                            info!("[GPU] Requested image fetch: {src}");
                        }
                }
            }
        }
    }

    /// Check for incoming image data from the CPU thread.
    fn check_image_messages(&mut self) -> bool {
        let mut got_image = false;
        while let Ok(msg) = self.img_rx.try_recv() {
            if let PipelineMsg::ImageData { url, data } = msg {
                info!("[GPU] Received image data: {} ({} bytes)", url, data.len());
                self.pending_image_urls.remove(&url);
                if let Some(renderer) = &mut self.renderer {
                    renderer.upload_image(&url, &data);
                }
                self.layout_dirty = true;
                got_image = true;
            }
        }
        got_image
    }

    /// Send a message to the current AI provider via the CPU thread.
    fn send_eva_query(&mut self, message: String) {
        let provider = self.eva_panel.provider;
        self.eva_panel.add_user_message(message.clone());
        self.eva_panel.set_loading(true);
        let context = format!("URL: {}\n\n{}", self.url_bar, self.page_text_cache);
        let _ = self.eva_tx.send(PipelineMsg::EvaQuery {
            message,
            page_context: context,
            provider,
        });
    }

    /// Ask the current AI provider to summarize the current page.
    fn send_eva_summarize(&mut self) {
        if self.page_text_cache.is_empty() {
            return;
        }
        let provider = self.eva_panel.provider;
        self.eva_panel.visible = true;
        self.eva_panel.add_user_message(format!("Summarize this page [{}]", provider.name()));
        self.eva_panel.set_loading(true);
        let _ = self.eva_tx.send(PipelineMsg::EvaSummarize {
            content: self.page_text_cache.clone(),
            provider,
        });
    }

    /// Request EVA to speak the last AI response via voice.
    fn send_voice_request(&mut self) {
        // Find the last AI response
        let last_text = self.eva_panel.messages.iter().rev()
            .find(|m| m.role == crate::eva::panel::Role::Ai)
            .map(|m| m.text.clone());
        if let Some(text) = last_text {
            let _ = self.eva_tx.send(PipelineMsg::EvaVoice { text });
        }
    }

    /// Detect if input is a natural language question (smart search).
    fn is_natural_question(input: &str) -> bool {
        let lower = input.to_lowercase();
        let question_starters = [
            "what is", "what are", "who is", "who are", "how to", "how do",
            "why is", "why do", "when is", "when did", "where is", "where do",
            "can you", "could you", "explain", "tell me", "define",
            // Portuguese
            "o que é", "o que são", "quem é", "como fazer", "como funciona",
            "por que", "porque", "quando", "onde", "explica", "me diz",
            "qual é", "quais são",
            // Spanish
            "qué es", "cómo", "por qué", "quién es", "dónde",
        ];
        question_starters.iter().any(|q| lower.starts_with(q))
            || (lower.ends_with('?') && lower.contains(' '))
    }

    /// Toggle reading mode (filter to high-relevance content only).
    fn toggle_reading_mode(&mut self) {
        self.reading_mode = !self.reading_mode;
        if self.reading_mode {
            // Filter content to high-relevance blocks only
            let original = self.content.clone();
            self.content = original.into_iter()
                .filter(|b| b.relevance > 0.6 || matches!(b.kind, crate::npu::BlockKind::Heading { .. }))
                .collect();
            // Request AI summary at top of reading mode
            if !self.page_text_cache.is_empty() {
                self.send_eva_summarize();
            }
            info!("[UI] Reading mode ON — {} blocks shown", self.content.len());
        } else {
            // Reload page to restore full content
            let url = self.url_bar.clone();
            self.navigate(&url);
            info!("[UI] Reading mode OFF — reloading full page");
        }
        self.layout_dirty = true;
        self.request_redraw();
    }

    /// Request translation of current page content.
    fn send_translate_request(&mut self, target_lang: &str) {
        if self.page_text_cache.is_empty() {
            return;
        }
        let provider = self.eva_panel.provider;
        self.eva_panel.visible = true;
        self.eva_panel.add_user_message(format!("Translate page to {target_lang}"));
        self.eva_panel.set_loading(true);
        let _ = self.eva_tx.send(PipelineMsg::TranslatePage {
            content: self.page_text_cache.clone(),
            target_lang: target_lang.to_string(),
            provider,
        });
    }

    /// Send proactive insights request (auto-generated questions about the page).
    fn send_proactive_insights(&mut self) {
        if self.page_text_cache.is_empty() {
            return;
        }
        // Only send once per page
        if self.insights_sent_for_url.as_deref() == Some(&self.url_bar) {
            return;
        }
        self.insights_sent_for_url = Some(self.url_bar.clone());
        let provider = self.eva_panel.provider;
        let _ = self.eva_tx.send(PipelineMsg::ProactiveInsights {
            content: self.page_text_cache.clone(),
            provider,
        });
    }

    /// Detect if input is a search query or a URL.
    /// Like Chrome: if it has spaces, no dots, or looks like a question → Google search.
    fn normalize_input(&self, input: &str) -> String {
        normalize_url_input(input)
    }

    fn navigate(&mut self, input: &str) {
        // Smart Search: detect natural language questions → send to AI
        if Self::is_natural_question(input) {
            info!("[UI] Smart search detected: {input}");
            let provider = self.eva_panel.provider;
            self.eva_panel.visible = true;
            self.eva_panel.add_user_message(format!("🔍 {input}"));
            self.eva_panel.set_loading(true);
            let _ = self.eva_tx.send(PipelineMsg::SmartSearch {
                query: input.to_string(),
                provider,
            });
            self.url_editing = false;
            self.request_redraw();
            return;
        }

        let url = self.normalize_input(input);
        info!("[UI] Navigate to: {url}");
        self.loading = true;
        self.loading_ticks = 0;
        self.scroll_y = 0.0; // Reset scroll position for new page
        self.url_bar = url.clone();
        self.url_editing = false;
        self.visited_urls.insert(url.clone());

        // Update history: truncate forward entries and push new
        if self.history_idx + 1 < self.history.len() {
            self.history.truncate(self.history_idx + 1);
        }
        self.history.push(url.clone());
        self.history_idx = self.history.len() - 1;

        self.layout_dirty = true;
        let _ = self.ui_tx.send(PipelineMsg::Navigate(url));
        self.request_redraw();
    }

    fn go_back(&mut self) {
        info!("[UI] Requesting back navigation");
        self.loading = true;
        self.loading_ticks = 0;
        self.layout_dirty = true;
        let _ = self.ui_tx.send(PipelineMsg::Back);
        self.request_redraw();
    }

    fn go_forward(&mut self) {
        info!("[UI] Requesting forward navigation");
        self.loading = true;
        self.loading_ticks = 0;
        self.layout_dirty = true;
        let _ = self.ui_tx.send(PipelineMsg::Forward);
        self.request_redraw();
    }

    fn request_redraw(&self) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    /// Recompute layout from current content (scroll-independent).
    /// Layout positions are in document-space; scroll is applied at render time.
    fn recompute_layout(&mut self) {
        let sf = self.scale_factor as f32;
        let full_width = self.renderer.as_ref()
            .map(|r| r.size().0 as f32 / sf)
            .unwrap_or(1200.0);
        // Shrink content area when EVA panel is visible
        let viewport_width = if self.eva_panel.visible {
            (full_width - 350.0).max(200.0)
        } else {
            full_width
        };
        // Collect texture dimensions from renderer's cache
        let image_dimensions: HashMap<String, (u32, u32)> = self.renderer.as_ref()
            .map(|r| {
                let mut dims = HashMap::new();
                // We need to iterate the content to find image URLs,
                // then query the renderer for each
                for block in &self.content {
                    if let crate::npu::BlockKind::Image { src, .. } = &block.kind {
                        if let Some(d) = r.texture_dimensions(src) {
                            dims.insert(src.clone(), d);
                        }
                    }
                    for child in &block.children {
                        if let crate::npu::BlockKind::Image { src, .. } = &child.kind {
                            if let Some(d) = r.texture_dimensions(src) {
                                dims.insert(src.clone(), d);
                            }
                        }
                    }
                }
                dims
            })
            .unwrap_or_default();
        self.cached_layout = layout::compute_layout_zoom(
            &self.content,
            0.0, // scroll not used in layout anymore
            viewport_width,
            &self.theme,
            self.zoom_level,
            &image_dimensions,
        );
        // Clamp scroll to content bounds
        let viewport_h = self.renderer.as_ref()
            .map(|r| r.size().1 as f32 / sf)
            .unwrap_or(800.0);
        let content_bottom = self.cached_layout.iter()
            .map(|b| b.y + b.height)
            .fold(0.0_f32, f32::max);
        let max_scroll = (content_bottom - viewport_h + 70.0).max(0.0);
        self.scroll_y = self.scroll_y.min(max_scroll);
        self.layout_dirty = false;
    }

    /// Get layout boxes with scroll offset applied (for rendering).
    fn scrolled_layout(&self) -> Vec<layout::LayoutBox> {
        self.cached_layout.iter().map(|b| {
            let mut scrolled = b.clone();
            // Don't scroll the URL bar background (first element, y=0)
            if b.y > 0.0 || b.height > 40.0 {
                scrolled.y -= self.scroll_y;
            }
            scrolled
        }).collect()
    }

    /// Find which link (if any) is under the given screen coordinates.
    /// Layout is in document-space, so we add scroll_y to screen Y to get document Y.
    fn hit_test_link(&self, x: f32, y: f32) -> Option<String> {
        let doc_y = y + self.scroll_y; // convert screen Y to document Y
        for lbox in &self.cached_layout {
            if let Some(href) = &lbox.href {
                if x >= lbox.x
                    && x <= lbox.x + lbox.width
                    && doc_y >= lbox.y
                    && doc_y <= lbox.y + lbox.height
                    && y > renderer::TOOLBAR_HEIGHT // screen-space check for toolbar
                {
                    return Some(href.clone());
                }
            }
        }
        None
    }

    /// Update hover state based on current mouse position.
    fn update_hover(&mut self) {
        let new_href = self.hit_test_link(self.mouse_x, self.mouse_y);
        let changed = self.hovered_href != new_href;
        self.hovered_href = new_href;

        if changed || self.mouse_y < 40.0 {
            if let Some(window) = &self.window {
                if self.mouse_y < 40.0 {
                    window.set_cursor(CursorIcon::Text);
                } else if self.hovered_href.is_some() {
                    window.set_cursor(CursorIcon::Pointer);
                } else {
                    window.set_cursor(CursorIcon::Default);
                }
            }
            self.request_redraw();
        }
    }

    /// Resolve a potentially relative href against the current URL.
    /// Blocks dangerous URI schemes (javascript:, data:, vbscript:).
    fn resolve_href(&self, href: &str) -> String {
        // Block dangerous schemes that could execute code
        let lower = href.trim().to_lowercase();
        if lower.starts_with("javascript:")
            || lower.starts_with("data:")
            || lower.starts_with("vbscript:")
            || lower.starts_with("blob:")
        {
            return String::new(); // blocked
        }

        if href.starts_with("http://") || href.starts_with("https://") {
            return href.to_string();
        }
        if let Ok(base) = url::Url::parse(&self.url_bar) {
            if let Ok(resolved) = base.join(href) {
                // Verify resolved URL also has safe scheme
                let scheme = resolved.scheme();
                if scheme == "http" || scheme == "https" || scheme == "neural" {
                    return resolved.to_string();
                }
                return String::new(); // blocked
            }
        }
        href.to_string()
    }
}

impl ApplicationHandler for GpuApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("Neural Browser \u{2014} CPU+NPU+GPU")
            .with_inner_size(winit::dpi::LogicalSize::new(1200u32, 800u32))
            .with_min_inner_size(winit::dpi::LogicalSize::new(400u32, 200u32));

        match event_loop.create_window(attrs) {
            Ok(window) => {
                let window = Arc::new(window);
                info!("[GPU] Window created");

                match pollster::block_on(renderer::WgpuRenderer::new(window.clone())) {
                    Ok(r) => {
                        info!("[GPU] wgpu renderer initialized");
                        self.renderer = Some(r);
                    }
                    Err(e) => {
                        error!("[GPU] Failed to init wgpu: {e}");
                    }
                }

                self.scale_factor = window.scale_factor();
                self.window = Some(window);
                self.layout_dirty = true;
            }
            Err(e) => {
                error!("[GPU] Failed to create window: {e}");
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                info!("[GPU] Window close requested");
                event_loop.exit();
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
            }
            WindowEvent::RedrawRequested => {
                self.check_npu_messages();
                self.check_eva_messages();
                self.check_image_messages();

                if self.loading {
                    self.loading_ticks += 1;
                }

                if self.layout_dirty {
                    self.recompute_layout();
                }

                let hovered_href_ref = self.hovered_href.clone();
                let ctx = renderer::RenderContext {
                    url: &self.url_bar,
                    loading: self.loading,
                    loading_ticks: self.loading_ticks,
                    hovered_href: hovered_href_ref.as_deref(),
                    visited_urls: &self.visited_urls,
                    theme: &self.theme,
                    url_editing: self.url_editing,
                    url_input: &self.url_input,
                    zoom_level: self.zoom_level,
                    eva_visible: self.eva_panel.visible,
                };
                // Apply scroll offset to get viewport-space positions
                let scrolled = self.scrolled_layout();
                if let Some(renderer) = &mut self.renderer {
                    if let Err(e) = renderer.render_with_eva(
                        &scrolled,
                        &ctx,
                        &self.eva_panel,
                    ) {
                        error!("[GPU] Render error: {e}");
                    }
                }

                // Keep redrawing while loading or EVA is waiting for response
                // or images are pending (NOT when panel is just visible and idle — saves GPU)
                if self.loading || self.eva_panel.is_loading || !self.pending_image_urls.is_empty() {
                    self.request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
                self.layout_dirty = true;
                self.request_redraw();
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor = scale_factor;
                self.layout_dirty = true;
                self.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Convert physical pixels to logical pixels
                let sf = self.scale_factor as f32;
                self.mouse_x = position.x as f32 / sf;
                self.mouse_y = position.y as f32 / sf;
                self.update_hover();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if state == ElementState::Pressed && button == MouseButton::Left {
                    // Toolbar click handling (buttons + URL bar + EVA button)
                    if self.mouse_y < renderer::TOOLBAR_HEIGHT {
                        let sf = self.scale_factor as f32;
                        let wf = self.renderer.as_ref()
                            .map(|r| r.size().0 as f32 / sf).unwrap_or(1200.0);
                        let tb = renderer::TOOLBAR_HEIGHT;
                        let nav_btn_size: f32 = 36.0;
                        let btn_y = (tb - nav_btn_size) / 2.0;
                        let btn_h = nav_btn_size;
                        let btn_w = nav_btn_size;
                        let back_x: f32 = 14.0;
                        let fwd_x = back_x + nav_btn_size + 4.0;
                        let ref_x = fwd_x + nav_btn_size + 4.0;
                        let eva_btn_w: f32 = 68.0;
                        let eva_btn_x = wf - eva_btn_w - 14.0;
                        // Pill URL bar — centered between nav buttons and EVA button
                        let nav_zone_end = ref_x + nav_btn_size + 16.0;
                        let eva_zone_start = eva_btn_x - 16.0;
                        let available = eva_zone_start - nav_zone_end;
                        let pill_max_w: f32 = 680.0;
                        let pill_w = available.min(pill_max_w).max(200.0);
                        let pill_x = nav_zone_end + (available - pill_w) / 2.0;
                        let _url_x = pill_x;
                        let mx = self.mouse_x;
                        let my = self.mouse_y;

                        if my >= btn_y && my <= btn_y + btn_h {
                            // EVA button (rightmost)
                            if mx >= eva_btn_x && mx < eva_btn_x + eva_btn_w {
                                self.eva_panel.toggle();
                                self.layout_dirty = true;
                                self.request_redraw();
                                return;
                            }
                            if mx >= back_x && mx < back_x + btn_w {
                                let _ = self.ui_tx.send(PipelineMsg::Back);
                                self.loading = true;
                                self.request_redraw();
                                return;
                            }
                            if mx >= fwd_x && mx < fwd_x + btn_w {
                                let _ = self.ui_tx.send(PipelineMsg::Forward);
                                self.loading = true;
                                self.request_redraw();
                                return;
                            }
                            if mx >= ref_x && mx < ref_x + btn_w {
                                // Refresh: use current URL bar (not history.last())
                                let url = self.url_bar.clone();
                                if !url.is_empty() {
                                    let _ = self.ui_tx.send(PipelineMsg::Navigate(url));
                                    self.loading = true;
                                }
                                self.request_redraw();
                                return;
                            }
                        }
                        // Click in pill URL bar (centered)
                        if mx >= pill_x && mx < pill_x + pill_w && !self.url_editing {
                            self.url_editing = true;
                            // Pre-fill with current URL so user can edit in-place
                            self.url_input = self.url_bar.clone();
                        }
                        self.request_redraw();
                        return;
                    }

                    // Hit test links
                    if let Some(href) = self.hit_test_link(self.mouse_x, self.mouse_y) {
                        // Fragment-only links: scroll without reload
                        if href.starts_with('#') {
                            info!("[UI] Fragment link: {href} (scroll to top)");
                            self.scroll_y = 0.0;
                            self.layout_dirty = true;
                            self.request_redraw();
                        } else {
                            // Skip non-navigable schemes
                            let lower = href.trim().to_lowercase();
                            if lower.starts_with("mailto:")
                                || lower.starts_with("tel:")
                            {
                                info!("[UI] Non-navigable link ignored: {href}");
                            } else {
                                let resolved = self.resolve_href(&href);
                                if resolved.is_empty() {
                                    info!("[UI] Blocked dangerous link: {href}");
                                } else {
                                    info!("[UI] Link clicked: {resolved}");
                                    self.navigate(&resolved);
                                }
                            }
                        }
                    } else if self.url_editing {
                        // Click outside URL bar cancels editing
                        self.url_editing = false;
                        self.request_redraw();
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Ctrl+scroll = zoom
                if self.modifiers.control_key() {
                    let delta_y = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                    };
                    if delta_y > 0.0 {
                        self.zoom_level = (self.zoom_level + 0.1).min(3.0);
                    } else if delta_y < 0.0 {
                        self.zoom_level = (self.zoom_level - 0.1).max(0.3);
                    }
                    info!("[UI] Zoom: {:.0}%", self.zoom_level * 100.0);
                    self.layout_dirty = true;
                    self.request_redraw();
                    return;
                }
                // If EVA panel is visible and mouse is over it, scroll messages
                if self.eva_panel.visible {
                    let sf = self.scale_factor as f32;
                    let vw = self.renderer.as_ref()
                        .map(|r| r.size().0 as f32 / sf).unwrap_or(1200.0);
                    let panel_x = (vw - 350.0).max(0.0);
                    if self.mouse_x >= panel_x {
                        let delta_y = match delta {
                            winit::event::MouseScrollDelta::LineDelta(_, y) => y * 30.0,
                            winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                        };
                        self.eva_panel.scroll_offset = (self.eva_panel.scroll_offset - delta_y).max(0.0);
                        self.request_redraw();
                        return;
                    }
                }
                if !self.url_editing {
                    match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => {
                            self.scroll_y -= y * 40.0;
                        }
                        winit::event::MouseScrollDelta::PixelDelta(pos) => {
                            self.scroll_y -= pos.y as f32;
                        }
                    }
                    // Clamp scroll: min 0, max = content height - viewport + margin
                    self.scroll_y = self.scroll_y.max(0.0);
                    let sf = self.scale_factor as f32;
                    let max_y = self.cached_layout.iter()
                        .map(|b| b.y + b.height)
                        .fold(0.0f32, f32::max);
                    let viewport_h = self.renderer.as_ref()
                        .map(|r| r.size().1 as f32 / sf)
                        .unwrap_or(800.0);
                    let max_scroll = (max_y - viewport_h + 100.0).max(0.0);
                    self.scroll_y = self.scroll_y.min(max_scroll);
                    // No layout_dirty — scroll is applied at render time
                    self.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }

                // On macOS use Cmd (super_key), on Windows/Linux use Ctrl
                let ctrl = if cfg!(target_os = "macos") {
                    self.modifiers.super_key()
                } else {
                    self.modifiers.control_key()
                };
                let alt = self.modifiers.alt_key();

                // ── Zoom shortcuts (Ctrl+Plus, Ctrl+Minus, Ctrl+0) ──
                if ctrl {
                    let zoom_changed = match &event.logical_key {
                        Key::Character(ch) if ch.as_str() == "+" || ch.as_str() == "=" => {
                            self.zoom_level = (self.zoom_level + 0.1).min(3.0);
                            true
                        }
                        Key::Character(ch) if ch.as_str() == "-" => {
                            self.zoom_level = (self.zoom_level - 0.1).max(0.3);
                            true
                        }
                        Key::Character(ch) if ch.as_str() == "0" => {
                            self.zoom_level = 1.0;
                            true
                        }
                        _ => false,
                    };
                    if zoom_changed {
                        info!("[UI] Zoom: {:.0}%", self.zoom_level * 100.0);
                        self.layout_dirty = true;
                        self.request_redraw();
                        return;
                    }
                }

                // ── Global EVA shortcuts (always active) ──
                match &event.logical_key {
                    // Ctrl+E = toggle EVA panel
                    Key::Character(ch) if ctrl && ch.as_str() == "e" => {
                        self.eva_panel.toggle();
                        if self.eva_panel.visible {
                            self.url_editing = false; // unfocus URL bar
                        }
                        self.layout_dirty = true; // reflow content for panel
                        self.request_redraw();
                        return;
                    }
                    // F2 = toggle EVA panel
                    Key::Named(NamedKey::F2) => {
                        self.eva_panel.toggle();
                        if self.eva_panel.visible {
                            self.url_editing = false;
                        }
                        self.layout_dirty = true; // reflow content for panel
                        self.request_redraw();
                        return;
                    }
                    // Ctrl+S = ask AI to summarize page
                    Key::Character(ch) if ctrl && ch.as_str() == "s" => {
                        self.send_eva_summarize();
                        self.request_redraw();
                        return;
                    }
                    // Ctrl+D = toggle reading mode (high-relevance content only)
                    Key::Character(ch) if ctrl && ch.as_str() == "d" => {
                        self.toggle_reading_mode();
                        return;
                    }
                    // Ctrl+T = translate page to Portuguese
                    Key::Character(ch) if ctrl && ch.as_str() == "t" => {
                        self.send_translate_request("Portuguese");
                        self.request_redraw();
                        return;
                    }
                    _ => {}
                }

                // ── AI panel focused: route input to panel ──
                if self.eva_panel.is_focused() {
                    match &event.logical_key {
                        Key::Named(NamedKey::Escape) => {
                            self.eva_panel.toggle(); // close panel
                            self.request_redraw();
                        }
                        // Tab = cycle AI provider (EVA → Claude → Gemini → GPT-4)
                        Key::Named(NamedKey::Tab) => {
                            self.eva_panel.cycle_provider();
                            info!("[UI] AI provider switched to: {}", self.eva_panel.provider_name());
                            self.request_redraw();
                        }
                        Key::Named(NamedKey::Enter) => {
                            let input = self.eva_panel.take_input();
                            if !input.is_empty() {
                                self.send_eva_query(input);
                            }
                            self.request_redraw();
                        }
                        Key::Named(NamedKey::Backspace) => {
                            self.eva_panel.input_backspace();
                            self.request_redraw();
                        }
                        Key::Character(ch) => {
                            // Ctrl+Shift+V = request voice response
                            if ctrl && ch.as_str() == "V" {
                                self.send_voice_request();
                                self.request_redraw();
                            } else if ctrl && ch.as_str() == "v" {
                                // Ctrl+V = paste into EVA input
                                if let Ok(mut clip) = arboard::Clipboard::new() {
                                    if let Ok(text) = clip.get_text() {
                                        for c in text.chars() {
                                            self.eva_panel.input_char(c);
                                        }
                                        self.request_redraw();
                                    }
                                }
                            } else if ctrl && ch.as_str() == "a" {
                                // Ctrl+A = clear EVA input
                                let _ = self.eva_panel.take_input();
                                self.request_redraw();
                            } else if !ctrl {
                                // Normal character input
                                for c in ch.chars() {
                                    self.eva_panel.input_char(c);
                                }
                                self.request_redraw();
                            }
                        }
                        _ => {}
                    }
                    return; // AI panel consumes all input when focused
                }

                // ── Normal browser keyboard handling ──
                match &event.logical_key {
                    // F5 = refresh
                    Key::Named(NamedKey::F5) => {
                        let url = self.url_bar.clone();
                        self.navigate(&url);
                    }
                    // F6 = focus URL bar (clears input like Chrome)
                    Key::Named(NamedKey::F6) => {
                        self.url_editing = true;
                        self.url_input.clear();
                        self.request_redraw();
                    }
                    // Alt+Left = Back
                    Key::Named(NamedKey::ArrowLeft) if alt => {
                        self.go_back();
                    }
                    // Alt+Right = Forward
                    Key::Named(NamedKey::ArrowRight) if alt => {
                        self.go_forward();
                    }
                    // Escape = cancel URL editing
                    Key::Named(NamedKey::Escape) => {
                        if self.url_editing {
                            self.url_editing = false;
                            self.request_redraw();
                        }
                    }
                    // Enter = navigate to typed URL
                    Key::Named(NamedKey::Enter) => {
                        if self.url_editing {
                            let url = self.url_input.clone();
                            self.navigate(&url);
                        }
                    }
                    // Backspace in URL bar
                    Key::Named(NamedKey::Backspace) => {
                        if self.url_editing {
                            if ctrl {
                                // Ctrl+Backspace: delete last word
                                let trimmed = self.url_input.trim_end();
                                if let Some(pos) = trimmed.rfind([' ', '/', '.']) {
                                    self.url_input.truncate(pos);
                                } else {
                                    self.url_input.clear();
                                }
                            } else {
                                self.url_input.pop();
                            }
                            self.request_redraw();
                        }
                    }
                    // Character input
                    Key::Character(ch) => {
                        // Ctrl+L = focus URL bar (clears input like Chrome)
                        if ctrl && ch.as_str() == "l" {
                            self.url_editing = true;
                            self.url_input.clear();
                            self.request_redraw();
                            return;
                        }

                        // Ctrl+R = refresh
                        if ctrl && ch.as_str() == "r" {
                            let url = self.url_bar.clone();
                            self.navigate(&url);
                            return;
                        }

                        // Ctrl+A = select all (clear to prepare for replacement)
                        if ctrl && ch.as_str() == "a" {
                            if self.url_editing {
                                // Select-all: pre-fill current URL so next keystroke replaces
                                self.url_input = self.url_bar.clone();
                                self.request_redraw();
                            }
                            return;
                        }

                        // Ctrl+V = paste from clipboard
                        if ctrl && ch.as_str() == "v" {
                            if self.url_editing {
                                if let Ok(mut clip) = arboard::Clipboard::new() {
                                    if let Ok(text) = clip.get_text() {
                                        // Sanitize: strip newlines/tabs from pasted URLs
                                        let sanitized = text.replace(['\n', '\r', '\t'], "");
                                        self.url_input.push_str(&sanitized);
                                        self.request_redraw();
                                    }
                                }
                            }
                            return;
                        }

                        // Ctrl+C = copy URL to clipboard
                        if ctrl && ch.as_str() == "c" {
                            if let Ok(mut clip) = arboard::Clipboard::new() {
                                let text = if self.url_editing && !self.url_input.is_empty() {
                                    &self.url_input
                                } else {
                                    &self.url_bar
                                };
                                let _ = clip.set_text(text.clone());
                                info!("[UI] Copied to clipboard: {}", text);
                            }
                            return;
                        }

                        // Ctrl+W = close window
                        if ctrl && ch.as_str() == "w" {
                            info!("[UI] Ctrl+W: requesting window close");
                            if let Some(_window) = &self.window {
                                // Exit the event loop properly
                                event_loop.exit();
                                std::process::exit(0);
                            }
                            return;
                        }

                        if self.url_editing && !ctrl {
                            self.url_input.push_str(ch.as_str());
                            self.request_redraw();
                        }
                    }
                    // Page Up / Page Down — clamp to content bounds
                    Key::Named(NamedKey::PageDown) => {
                        self.scroll_y += 600.0;
                        let sf = self.scale_factor as f32;
                        let max_y = self.cached_layout.iter()
                            .map(|b| b.y + b.height).fold(0.0f32, f32::max);
                        let viewport_h = self.renderer.as_ref()
                            .map(|r| r.size().1 as f32 / sf).unwrap_or(800.0);
                        self.scroll_y = self.scroll_y.min((max_y - viewport_h + 70.0).max(0.0));
                        self.request_redraw();
                    }
                    Key::Named(NamedKey::PageUp) => {
                        self.scroll_y = (self.scroll_y - 600.0).max(0.0);
                        self.request_redraw();
                    }
                    Key::Named(NamedKey::Home) => {
                        self.scroll_y = 0.0;
                        self.request_redraw();
                    }
                    Key::Named(NamedKey::End) => {
                        // Use actual layout height instead of heuristic
                        let sf = self.scale_factor as f32;
                        let max_y = self.cached_layout.iter()
                            .map(|b| b.y + b.height)
                            .fold(0.0f32, f32::max);
                        let viewport_h = self.renderer.as_ref()
                            .map(|r| r.size().1 as f32 / sf)
                            .unwrap_or(800.0);
                        self.scroll_y = (max_y - viewport_h + 60.0).max(0.0);
                        self.request_redraw();
                    }
                    Key::Named(NamedKey::F11) => {
                        if let Some(window) = &self.window {
                            let is_fs = window.fullscreen().is_some();
                            window.set_fullscreen(if is_fs {
                                None
                            } else {
                                Some(winit::window::Fullscreen::Borderless(None))
                            });
                        }
                    }
                    // Arrow key scrolling (when not editing URL and not Alt)
                    Key::Named(NamedKey::ArrowDown) if !alt && !self.url_editing => {
                        self.scroll_y += 40.0;
                        // Clamp to content bounds
                        let sf = self.scale_factor as f32;
                        let max_y = self.cached_layout.iter()
                            .map(|b| b.y + b.height).fold(0.0f32, f32::max);
                        let viewport_h = self.renderer.as_ref()
                            .map(|r| r.size().1 as f32 / sf).unwrap_or(800.0);
                        self.scroll_y = self.scroll_y.min((max_y - viewport_h + 70.0).max(0.0));
                        self.request_redraw();
                    }
                    Key::Named(NamedKey::ArrowUp) if !alt && !self.url_editing => {
                        self.scroll_y = (self.scroll_y - 40.0).max(0.0);
                        self.request_redraw();
                    }
                    // Space = scroll down by viewport, Shift+Space = scroll up
                    Key::Named(NamedKey::Space) if !self.url_editing => {
                        let shift = self.modifiers.shift_key();
                        if shift {
                            self.scroll_y = (self.scroll_y - 500.0).max(0.0);
                        } else {
                            self.scroll_y += 500.0;
                            // Clamp
                            let sf = self.scale_factor as f32;
                            let max_y = self.cached_layout.iter()
                                .map(|b| b.y + b.height).fold(0.0f32, f32::max);
                            let vh = self.renderer.as_ref()
                                .map(|r| r.size().1 as f32 / sf).unwrap_or(800.0);
                            self.scroll_y = self.scroll_y.min((max_y - vh + 70.0).max(0.0));
                        }
                        self.request_redraw();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let mut needs_redraw = false;
        if self.check_npu_messages() {
            needs_redraw = true;
        }
        if self.check_eva_messages() {
            needs_redraw = true;
        }
        if self.check_image_messages() {
            needs_redraw = true;
        }
        // Keep animating loading indicator or EVA loading
        if self.loading || self.eva_panel.is_loading || !self.pending_image_urls.is_empty() {
            needs_redraw = true;
        }
        if needs_redraw {
            self.request_redraw();
        }
    }
}

/// Normalize user input: URL, search query, or special protocol.
/// Extracted as free function for testability.
fn normalize_url_input(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return "https://www.google.com".to_string();
    }

    // Already a full URL with known scheme
    if trimmed.starts_with("http://")
        || trimmed.starts_with("https://")
    {
        return trimmed.to_string();
    }

    // Block potentially dangerous schemes
    if trimmed.starts_with("data:") {
        log::warn!("[Security] Blocked data: URL");
        return "neural://error?reason=data_urls_blocked".to_string();
    }
    if trimmed.starts_with("ftp://") {
        log::warn!("[Security] Blocked ftp:// URL");
        return "neural://error?reason=ftp_urls_blocked".to_string();
    }

    // Block file:// — local filesystem access is a security risk
    if trimmed.starts_with("file://") {
        log::warn!("[Security] Blocked file:// URL: {trimmed}");
        return "neural://error?reason=file_urls_blocked".to_string();
    }

    // Internal pages
    if trimmed.starts_with("neural://") {
        return trimmed.to_string();
    }

    // Has spaces → definitely a search query
    if trimmed.contains(' ') {
        return format!(
            "https://www.google.com/search?q={}",
            urlencoding_simple(trimmed)
        );
    }

    // localhost or localhost:port → HTTP
    if trimmed.starts_with("localhost") || trimmed.starts_with("127.0.0.1") {
        return format!("http://{trimmed}");
    }

    // IPv6 address like [::1]:8080
    if trimmed.starts_with('[') {
        return format!("http://{trimmed}");
    }

    // IP address with port: 192.168.1.1:8080
    if trimmed.chars().all(|c| c.is_ascii_digit() || c == '.' || c == ':')
        && trimmed.contains('.')
    {
        return format!("http://{trimmed}");
    }

    // Looks like a domain (has dot): example.com, foo.bar.bz
    if trimmed.contains('.') {
        return format!("https://{trimmed}");
    }

    // Single word without dot → search (like Chrome)
    format!(
        "https://www.google.com/search?q={}",
        urlencoding_simple(trimmed)
    )
}

/// Simple URL percent-encoding for search queries.
/// Uses lookup table instead of format!() per byte for zero-allocation encoding.
fn urlencoding_simple(s: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push(HEX[(b >> 4) as usize] as char);
                out.push(HEX[(b & 0xF) as usize] as char);
            }
        }
    }
    out
}

/// Entry point for the GPU renderer. Must run on main thread (wgpu requirement).
pub fn run_gpu_renderer(
    npu_rx: Receiver<PipelineMsg>,
    ui_tx: Sender<PipelineMsg>,
    eva_tx: Sender<PipelineMsg>,
    eva_resp_rx: Receiver<PipelineMsg>,
    img_rx: Receiver<PipelineMsg>,
) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = GpuApp::new(npu_rx, ui_tx, eva_tx, eva_resp_rx, img_rx);
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── normalize_url_input tests ──

    #[test]
    fn test_empty_input_goes_to_google() {
        assert_eq!(normalize_url_input(""), "https://www.google.com");
        assert_eq!(normalize_url_input("   "), "https://www.google.com");
    }

    #[test]
    fn test_full_url_passthrough() {
        assert_eq!(normalize_url_input("https://example.com"), "https://example.com");
        assert_eq!(normalize_url_input("http://foo.bar"), "http://foo.bar");
    }

    #[test]
    fn test_dangerous_schemes_blocked() {
        assert!(normalize_url_input("ftp://files.example.com").starts_with("neural://error"));
        assert!(normalize_url_input("data:text/html,<script>alert(1)</script>").starts_with("neural://error"));
    }

    #[test]
    fn test_neural_protocol() {
        assert_eq!(normalize_url_input("neural://start"), "neural://start");
    }

    #[test]
    fn test_search_query_with_spaces() {
        let result = normalize_url_input("rust web browser");
        assert!(result.starts_with("https://www.google.com/search?q=rust"));
        assert!(result.contains("web"));
    }

    #[test]
    fn test_domain_gets_https() {
        assert_eq!(normalize_url_input("example.com"), "https://example.com");
        assert_eq!(normalize_url_input("foo.bar.bz"), "https://foo.bar.bz");
    }

    #[test]
    fn test_localhost_gets_http() {
        assert_eq!(normalize_url_input("localhost"), "http://localhost");
        assert_eq!(normalize_url_input("localhost:3000"), "http://localhost:3000");
        assert_eq!(normalize_url_input("127.0.0.1:8080"), "http://127.0.0.1:8080");
    }

    #[test]
    fn test_ipv6_gets_http() {
        assert_eq!(normalize_url_input("[::1]:8080"), "http://[::1]:8080");
    }

    #[test]
    fn test_ip_address_gets_http() {
        assert_eq!(normalize_url_input("192.168.1.1:8080"), "http://192.168.1.1:8080");
    }

    #[test]
    fn test_single_word_searches() {
        let result = normalize_url_input("rust");
        assert!(result.starts_with("https://www.google.com/search?q=rust"));
    }

    #[test]
    fn test_file_url_blocked() {
        let result = normalize_url_input("file:///etc/passwd");
        assert!(result.starts_with("neural://error"), "file:// should be blocked, got: {result}");
        let result2 = normalize_url_input("file://C:/Windows/System32");
        assert!(result2.starts_with("neural://error"), "file:// should be blocked, got: {result2}");
    }

    // ── urlencoding_simple tests ──

    #[test]
    fn test_encode_plain_text() {
        assert_eq!(urlencoding_simple("hello"), "hello");
    }

    #[test]
    fn test_encode_spaces() {
        assert_eq!(urlencoding_simple("hello world"), "hello+world");
    }

    #[test]
    fn test_encode_special_chars() {
        let encoded = urlencoding_simple("a&b=c");
        assert_eq!(encoded, "a%26b%3Dc");
    }

    #[test]
    fn test_encode_unicode() {
        let encoded = urlencoding_simple("café");
        assert!(encoded.contains("%"));
        assert!(encoded.starts_with("caf"));
    }

    #[test]
    fn test_encode_preserves_unreserved() {
        assert_eq!(urlencoding_simple("A-Z_a.z~0"), "A-Z_a.z~0");
    }
}
