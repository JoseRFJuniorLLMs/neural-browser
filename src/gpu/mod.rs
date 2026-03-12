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

use crate::eva::panel::EvaPanel;
use crate::npu::ContentBlock;
use crate::ui::Theme;
use crate::PipelineMsg;
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use log::{info, error};
use std::collections::HashSet;
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
}

impl GpuApp {
    fn new(
        npu_rx: Receiver<PipelineMsg>,
        ui_tx: Sender<PipelineMsg>,
        eva_tx: Sender<PipelineMsg>,
        eva_resp_rx: Receiver<PipelineMsg>,
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
        }
    }

    fn check_npu_messages(&mut self) -> bool {
        let mut got_content = false;
        while let Ok(msg) = self.npu_rx.try_recv() {
            match msg {
                PipelineMsg::ContentReady { url, blocks, ads_blocked } => {
                    info!(
                        "[GPU] Received {} blocks from NPU ({} ads blocked)",
                        blocks.len(),
                        ads_blocked
                    );
                    self.url_bar = url;
                    // Cache page text for EVA context
                    self.page_text_cache = blocks.iter()
                        .filter(|b| !b.text.is_empty())
                        .take(20)
                        .map(|b| b.text.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    self.content = blocks;
                    self.scroll_y = 0.0;
                    self.loading = false;
                    self.layout_dirty = true;
                    got_content = true;
                }
                _ => {}
            }
        }
        got_content
    }

    /// Check for EVA responses from the CPU thread.
    fn check_eva_messages(&mut self) -> bool {
        let mut got_response = false;
        while let Ok(msg) = self.eva_resp_rx.try_recv() {
            match msg {
                PipelineMsg::EvaResponse { text } => {
                    info!("[GPU] EVA response received ({} chars)", text.len());
                    self.eva_panel.add_eva_response(text);
                    got_response = true;
                }
                _ => {}
            }
        }
        got_response
    }

    /// Send a message to EVA via the CPU thread.
    fn send_eva_query(&mut self, message: String) {
        self.eva_panel.add_user_message(message.clone());
        self.eva_panel.set_loading(true);
        let _ = self.eva_tx.send(PipelineMsg::EvaQuery {
            message,
            page_context: self.page_text_cache.clone(),
        });
    }

    /// Ask EVA to summarize the current page.
    fn send_eva_summarize(&mut self) {
        if self.page_text_cache.is_empty() {
            return;
        }
        self.eva_panel.visible = true;
        self.eva_panel.add_user_message("Summarize this page".into());
        self.eva_panel.set_loading(true);
        let _ = self.eva_tx.send(PipelineMsg::EvaSummarize {
            content: self.page_text_cache.clone(),
        });
    }

    /// Detect if input is a search query or a URL.
    /// Like Chrome: if it has spaces, no dots, or looks like a question → Google search.
    fn normalize_input(&self, input: &str) -> String {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return "https://www.google.com".to_string();
        }

        // Already a full URL with known scheme
        if trimmed.starts_with("http://")
            || trimmed.starts_with("https://")
            || trimmed.starts_with("ftp://")
            || trimmed.starts_with("data:")
            || trimmed.starts_with("file://")
        {
            return trimmed.to_string();
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

    fn navigate(&mut self, input: &str) {
        let url = self.normalize_input(input);
        info!("[UI] Navigate to: {url}");
        self.loading = true;
        self.loading_ticks = 0;
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

    /// Recompute layout from current content + scroll position.
    fn recompute_layout(&mut self) {
        let viewport_width = self.renderer.as_ref()
            .map(|r| r.size().0 as f32)
            .unwrap_or(1200.0);
        self.cached_layout = layout::compute_layout(
            &self.content,
            self.scroll_y,
            viewport_width,
            &self.theme,
        );
        self.layout_dirty = false;
    }

    /// Find which link (if any) is under the given screen coordinates.
    fn hit_test_link(&self, x: f32, y: f32) -> Option<String> {
        for lbox in &self.cached_layout {
            if let Some(href) = &lbox.href {
                if x >= lbox.x
                    && x <= lbox.x + lbox.width
                    && y >= lbox.y
                    && y <= lbox.y + lbox.height
                    && y > 40.0
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

        if changed {
            if let Some(window) = &self.window {
                if self.hovered_href.is_some() {
                    window.set_cursor(CursorIcon::Pointer);
                } else {
                    window.set_cursor(CursorIcon::Default);
                }
            }
            self.request_redraw();
        }
    }

    /// Resolve a potentially relative href against the current URL.
    fn resolve_href(&self, href: &str) -> String {
        if href.starts_with("http://") || href.starts_with("https://") {
            return href.to_string();
        }
        if let Ok(base) = url::Url::parse(&self.url_bar) {
            if let Ok(resolved) = base.join(href) {
                return resolved.to_string();
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
            .with_inner_size(winit::dpi::LogicalSize::new(1200u32, 800u32));

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
                };
                if let Some(renderer) = &mut self.renderer {
                    if let Err(e) = renderer.render_with_eva(
                        &self.cached_layout,
                        &ctx,
                        &self.eva_panel,
                    ) {
                        error!("[GPU] Render error: {e}");
                    }
                }

                // Keep redrawing while loading (for animation) or EVA panel open
                if self.loading || self.eva_panel.visible {
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
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_x = position.x as f32;
                self.mouse_y = position.y as f32;
                self.update_hover();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if state == ElementState::Pressed && button == MouseButton::Left {
                    // Check if click is in URL bar area (like Chrome: click selects all)
                    if self.mouse_y < 40.0 {
                        if !self.url_editing {
                            // First click: enter edit mode, select all (cursor at end)
                            self.url_editing = true;
                            self.url_input.clear(); // Empty = ready to type new query
                        }
                        self.request_redraw();
                        return;
                    }

                    // Hit test links
                    if let Some(href) = self.hit_test_link(self.mouse_x, self.mouse_y) {
                        let resolved = self.resolve_href(&href);
                        info!("[UI] Link clicked: {resolved}");
                        self.navigate(&resolved);
                    } else if self.url_editing {
                        // Click outside URL bar cancels editing
                        self.url_editing = false;
                        self.request_redraw();
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
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
                    let max_y = self.cached_layout.iter()
                        .map(|b| b.y + b.height)
                        .fold(0.0f32, f32::max);
                    let viewport_h = self.renderer.as_ref()
                        .map(|r| r.size().1 as f32)
                        .unwrap_or(800.0);
                    let max_scroll = (max_y - viewport_h + 100.0).max(0.0);
                    self.scroll_y = self.scroll_y.min(max_scroll);
                    self.layout_dirty = true;
                    self.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }

                let ctrl = self.modifiers.control_key();
                let alt = self.modifiers.alt_key();

                // ── Global EVA shortcuts (always active) ──
                match &event.logical_key {
                    // Ctrl+E = toggle EVA panel
                    Key::Character(ch) if ctrl && ch.as_str() == "e" => {
                        self.eva_panel.toggle();
                        if self.eva_panel.visible {
                            self.url_editing = false; // unfocus URL bar
                        }
                        self.request_redraw();
                        return;
                    }
                    // F2 = toggle EVA panel
                    Key::Named(NamedKey::F2) => {
                        self.eva_panel.toggle();
                        if self.eva_panel.visible {
                            self.url_editing = false;
                        }
                        self.request_redraw();
                        return;
                    }
                    // Ctrl+S = ask EVA to summarize page
                    Key::Character(ch) if ctrl && ch.as_str() == "s" => {
                        self.send_eva_summarize();
                        self.request_redraw();
                        return;
                    }
                    _ => {}
                }

                // ── EVA panel focused: route input to EVA ──
                if self.eva_panel.is_focused() {
                    match &event.logical_key {
                        Key::Named(NamedKey::Escape) => {
                            self.eva_panel.toggle(); // close panel
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
                            // Don't capture Ctrl shortcuts in EVA input
                            if !ctrl {
                                for c in ch.chars() {
                                    self.eva_panel.input_char(c);
                                }
                                self.request_redraw();
                            }
                        }
                        _ => {}
                    }
                    return; // EVA panel consumes all input when focused
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
                                if let Some(pos) = trimmed.rfind(|c: char| c == ' ' || c == '/' || c == '.') {
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

                        // Ctrl+A = select all (clear to type new)
                        if ctrl && ch.as_str() == "a" {
                            if self.url_editing {
                                // Select-all behavior: next input replaces
                                self.url_input.clear();
                                self.request_redraw();
                            }
                            return;
                        }

                        // Ctrl+V = paste from clipboard (winit doesn't expose clipboard,
                        // but we can try via arboard if available; for now log warning)
                        if ctrl && ch.as_str() == "v" {
                            // Note: paste requires clipboard crate integration
                            // For now, winit may deliver pasted text as Key::Character events
                            return;
                        }

                        if self.url_editing && !ctrl {
                            self.url_input.push_str(ch.as_str());
                            self.request_redraw();
                        }
                    }
                    // Page Up / Page Down
                    Key::Named(NamedKey::PageDown) => {
                        self.scroll_y += 600.0;
                        self.layout_dirty = true;
                        self.request_redraw();
                    }
                    Key::Named(NamedKey::PageUp) => {
                        self.scroll_y = (self.scroll_y - 600.0).max(0.0);
                        self.layout_dirty = true;
                        self.request_redraw();
                    }
                    Key::Named(NamedKey::Home) => {
                        self.scroll_y = 0.0;
                        self.layout_dirty = true;
                        self.request_redraw();
                    }
                    Key::Named(NamedKey::End) => {
                        // Use actual layout height instead of heuristic
                        let max_y = self.cached_layout.iter()
                            .map(|b| b.y + b.height)
                            .fold(0.0f32, f32::max);
                        let viewport_h = self.renderer.as_ref()
                            .map(|r| r.size().1 as f32)
                            .unwrap_or(800.0);
                        self.scroll_y = (max_y - viewport_h + 60.0).max(0.0);
                        self.layout_dirty = true;
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
        // Keep animating loading indicator or EVA loading
        if self.loading || self.eva_panel.is_loading {
            needs_redraw = true;
        }
        if needs_redraw {
            self.request_redraw();
        }
    }
}

/// Simple URL percent-encoding for search queries.
fn urlencoding_simple(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push_str(&format!("{:02X}", b));
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
) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = GpuApp::new(npu_rx, ui_tx, eva_tx, eva_resp_rx);
    event_loop.run_app(&mut app)?;
    Ok(())
}
