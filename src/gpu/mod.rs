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
}

impl GpuApp {
    fn new(npu_rx: Receiver<PipelineMsg>, ui_tx: Sender<PipelineMsg>) -> Self {
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

    fn navigate(&mut self, url: &str) {
        let url = if !url.starts_with("http://") && !url.starts_with("https://") {
            format!("https://{url}")
        } else {
            url.to_string()
        };
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

    fn display_url(&self) -> String {
        if self.url_editing {
            format!("{}\u{2588}", self.url_input) // cursor block
        } else if self.loading {
            format!("{} \u{23F3}", self.url_bar) // hourglass
        } else {
            self.url_bar.clone()
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

                if self.loading {
                    self.loading_ticks += 1;
                }

                if self.layout_dirty {
                    self.recompute_layout();
                }

                let display_url = self.display_url();
                let hovered_href_ref = self.hovered_href.clone();
                let ctx = renderer::RenderContext {
                    url: &display_url,
                    loading: self.loading,
                    loading_ticks: self.loading_ticks,
                    hovered_href: hovered_href_ref.as_deref(),
                    visited_urls: &self.visited_urls,
                    theme: &self.theme,
                };
                if let Some(renderer) = &mut self.renderer {
                    if let Err(e) = renderer.render(&self.cached_layout, &ctx) {
                        error!("[GPU] Render error: {e}");
                    }
                }

                // Keep redrawing while loading (for animation)
                if self.loading {
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
                    // Check if click is in URL bar area
                    if self.mouse_y < 40.0 {
                        self.url_editing = true;
                        self.url_input = self.url_bar.clone();
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
                            self.scroll_y = self.scroll_y.max(0.0);
                        }
                        winit::event::MouseScrollDelta::PixelDelta(pos) => {
                            self.scroll_y -= pos.y as f32;
                            self.scroll_y = self.scroll_y.max(0.0);
                        }
                    }
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

                match &event.logical_key {
                    // F5 = refresh
                    Key::Named(NamedKey::F5) => {
                        let url = self.url_bar.clone();
                        self.navigate(&url);
                    }
                    // F6 = focus URL bar
                    Key::Named(NamedKey::F6) => {
                        self.url_editing = true;
                        self.url_input = self.url_bar.clone();
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
                            self.url_input.pop();
                            self.request_redraw();
                        }
                    }
                    // Character input
                    Key::Character(ch) => {
                        // Ctrl+L = focus URL bar
                        if ctrl && ch.as_str() == "l" {
                            self.url_editing = true;
                            self.url_input = self.url_bar.clone();
                            self.request_redraw();
                            return;
                        }

                        // Ctrl+R = refresh
                        if ctrl && ch.as_str() == "r" {
                            let url = self.url_bar.clone();
                            self.navigate(&url);
                            return;
                        }

                        if self.url_editing {
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
                        self.scroll_y = (self.content.len() as f32 * 50.0).max(0.0);
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
        if self.check_npu_messages() {
            self.request_redraw();
        }
        // Keep animating loading indicator
        if self.loading {
            self.request_redraw();
        }
    }
}

/// Entry point for the GPU renderer. Must run on main thread (wgpu requirement).
pub fn run_gpu_renderer(
    npu_rx: Receiver<PipelineMsg>,
    ui_tx: Sender<PipelineMsg>,
) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = GpuApp::new(npu_rx, ui_tx);
    event_loop.run_app(&mut app)?;
    Ok(())
}
