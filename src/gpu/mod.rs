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
use crate::PipelineMsg;
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use log::{info, error};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};
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
}

impl GpuApp {
    fn new(npu_rx: Receiver<PipelineMsg>, ui_tx: Sender<PipelineMsg>) -> Self {
        Self {
            window: None,
            renderer: None,
            content: Vec::new(),
            scroll_y: 0.0,
            npu_rx,
            ui_tx,
            url_bar: "https://example.com".into(),
            url_editing: false,
            url_input: String::new(),
            loading: true,
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
        self.url_bar = url.clone();
        self.url_editing = false;
        let _ = self.ui_tx.send(PipelineMsg::Navigate(url));
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
            WindowEvent::RedrawRequested => {
                self.check_npu_messages();

                let display_url = self.display_url();
                let layout = layout::compute_layout(&self.content, self.scroll_y);
                if let Some(renderer) = &mut self.renderer {
                    if let Err(e) = renderer.render(&display_url, &layout) {
                        error!("[GPU] Render error: {e}");
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
                self.request_redraw();
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
                    self.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != winit::event::ElementState::Pressed {
                    return;
                }

                match &event.logical_key {
                    // F5 = refresh
                    Key::Named(NamedKey::F5) => {
                        let url = self.url_bar.clone();
                        self.navigate(&url);
                    }
                    // Ctrl+L or F6 = focus URL bar
                    Key::Named(NamedKey::F6) => {
                        self.url_editing = true;
                        self.url_input = self.url_bar.clone();
                        self.request_redraw();
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
                    // Backspace
                    Key::Named(NamedKey::Backspace) => {
                        if self.url_editing {
                            self.url_input.pop();
                            self.request_redraw();
                        }
                    }
                    // Character input
                    Key::Character(ch) => {
                        // Ctrl+L = focus URL bar
                        if ch.as_str() == "l"
                            && event.state == winit::event::ElementState::Pressed
                        {
                            // Check for Ctrl modifier via physical key state
                            // For simplicity, just handle F6 for URL focus
                        }

                        if self.url_editing {
                            self.url_input.push_str(ch.as_str());
                            self.request_redraw();
                        }
                    }
                    // Page Up / Page Down
                    Key::Named(NamedKey::PageDown) => {
                        self.scroll_y += 600.0;
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
