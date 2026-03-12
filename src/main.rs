//! Neural Browser -- CPU + NPU + GPU working together
//!
//! Architecture:
//!   CPU  -> networking, HTML parsing, DOM tree, event loop, browser history
//!   NPU  -> content understanding, ad blocking, summarization (ONNX + DirectML)
//!   GPU  -> layout rendering, text rasterization, compositing (wgpu)
//!
//! Pipeline:
//!   URL -> [CPU] fetch+parse -> DOM
//!       -> [NPU] understand + extract + classify
//!       -> [GPU] layout + paint + composite -> display

mod cpu;
mod npu;
mod gpu;
mod ui;

use anyhow::Result;
use crossbeam_channel as channel;
use log::{info, error, warn};

/// Messages flowing between the three processors
#[derive(Debug, Clone)]
pub enum PipelineMsg {
    // CPU -> NPU: raw content ready for AI processing
    HtmlReady {
        url: String,
        html: String,
        dom: cpu::dom::DomTree,
    },

    // NPU -> GPU: AI-processed content ready for rendering
    ContentReady {
        url: String,
        blocks: Vec<npu::ContentBlock>,
        ads_blocked: usize,
    },

    // GPU -> UI: frame rendered
    FrameReady,

    // UI -> CPU: user navigation
    Navigate(String),

    // UI -> CPU: go back in history
    Back,

    // UI -> CPU: go forward in history
    Forward,

    // NPU -> CPU: prefetch suggestion
    Prefetch(String),
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    info!("Neural Browser v0.1.0");
    info!("  CPU: networking + parsing");
    info!("  NPU: AI inference (ONNX + DirectML)");
    info!("  GPU: rendering (wgpu)");

    // ── Inter-processor channels ──
    let (cpu_to_npu_tx, cpu_to_npu_rx) = channel::bounded::<PipelineMsg>(8);
    let (npu_to_gpu_tx, npu_to_gpu_rx) = channel::bounded::<PipelineMsg>(8);
    let (ui_tx, ui_rx) = channel::bounded::<PipelineMsg>(16);

    // ── NPU engine (spawn on dedicated thread) ──
    let _npu_ui_tx = ui_tx.clone();
    let npu_handle = std::thread::Builder::new()
        .name("npu-engine".into())
        .spawn(move || {
            info!("[NPU] Thread started -- DirectML inference engine");
            let mut engine = match npu::NpuEngine::new() {
                Ok(e) => e,
                Err(e) => {
                    error!("[NPU] Failed to init: {e}");
                    return;
                }
            };

            for msg in cpu_to_npu_rx {
                match msg {
                    PipelineMsg::HtmlReady { url, html, dom } => {
                        info!("[NPU] Processing: {url}");
                        match engine.process_page(&url, &html, &dom) {
                            Ok(result) => {
                                let _ = npu_to_gpu_tx.send(PipelineMsg::ContentReady {
                                    url,
                                    blocks: result.blocks,
                                    ads_blocked: result.ads_blocked,
                                });
                            }
                            Err(e) => error!("[NPU] Processing failed: {e}"),
                        }
                    }
                    _ => {}
                }
            }
        })?;

    // ── CPU networking (spawn on dedicated thread) ──
    let cpu_ui_rx = ui_rx;
    let cpu_handle = std::thread::Builder::new()
        .name("cpu-network".into())
        .spawn(move || {
            info!("[CPU] Thread started -- networking + parsing + history");
            let net = cpu::network::NetworkEngine::new();

            // ── Browser history stacks ──
            let mut back_stack: Vec<String> = Vec::new();
            let mut forward_stack: Vec<String> = Vec::new();
            // Navigate to start page or CLI-provided URL
            let mut current_url: Option<String> = match std::env::args().nth(1) {
                Some(url) => {
                    process_url(&net, &url, &cpu_to_npu_tx);
                    Some(url)
                }
                None => {
                    // Show built-in start page
                    let url = cpu::start_page::START_PAGE_URL;
                    let html = cpu::start_page::start_page_html().to_string();
                    let dom = cpu::dom::parse_html(&html);
                    info!("[CPU] Loaded start page ({} nodes)", dom.nodes.len());
                    let _ = cpu_to_npu_tx.send(PipelineMsg::HtmlReady {
                        url: url.to_string(),
                        html,
                        dom,
                    });
                    Some(url.to_string())
                }
            };

            // Listen for navigation events
            for msg in cpu_ui_rx {
                match msg {
                    PipelineMsg::Navigate(url) => {
                        info!("[CPU] Navigating to: {url}");
                        // Push current URL to back stack before navigating
                        if let Some(cur) = current_url.take() {
                            back_stack.push(cur);
                        }
                        // Clear forward stack on new navigation
                        forward_stack.clear();
                        current_url = Some(url.clone());
                        process_url(&net, &url, &cpu_to_npu_tx);
                    }
                    PipelineMsg::Back => {
                        if let Some(prev_url) = back_stack.pop() {
                            info!("[CPU] Going back to: {prev_url}");
                            if let Some(cur) = current_url.take() {
                                forward_stack.push(cur);
                            }
                            current_url = Some(prev_url.clone());
                            if prev_url == cpu::start_page::START_PAGE_URL {
                                // Reload start page from memory
                                let html = cpu::start_page::start_page_html().to_string();
                                let dom = cpu::dom::parse_html(&html);
                                let _ = cpu_to_npu_tx.send(PipelineMsg::HtmlReady {
                                    url: prev_url,
                                    html,
                                    dom,
                                });
                            } else {
                                process_url(&net, &prev_url, &cpu_to_npu_tx);
                            }
                        } else {
                            warn!("[CPU] No back history available");
                        }
                    }
                    PipelineMsg::Forward => {
                        if let Some(next_url) = forward_stack.pop() {
                            info!("[CPU] Going forward to: {next_url}");
                            if let Some(cur) = current_url.take() {
                                back_stack.push(cur);
                            }
                            current_url = Some(next_url.clone());
                            if next_url == cpu::start_page::START_PAGE_URL {
                                let html = cpu::start_page::start_page_html().to_string();
                                let dom = cpu::dom::parse_html(&html);
                                let _ = cpu_to_npu_tx.send(PipelineMsg::HtmlReady {
                                    url: next_url,
                                    html,
                                    dom,
                                });
                            } else {
                                process_url(&net, &next_url, &cpu_to_npu_tx);
                            }
                        } else {
                            warn!("[CPU] No forward history available");
                        }
                    }
                    PipelineMsg::Prefetch(url) => {
                        info!("[CPU] Prefetching: {url}");
                        let _ = net.prefetch(&url);
                    }
                    _ => {}
                }
            }
        })?;

    // ── GPU rendering + window (must be on main thread) ──
    info!("[GPU] Starting render engine on main thread");
    gpu::run_gpu_renderer(npu_to_gpu_rx, ui_tx)?;

    let _ = npu_handle.join();
    let _ = cpu_handle.join();

    Ok(())
}

/// Fetch and parse a URL, sending the result to the NPU.
/// On failure, generates an error page instead of propagating the error.
fn process_url(
    net: &cpu::network::NetworkEngine,
    url: &str,
    tx: &channel::Sender<PipelineMsg>,
) {
    let html = match net.fetch(url) {
        Ok(h) => h,
        Err(e) => {
            error!("[CPU] Fetch failed for {url}: {e}");
            cpu::network::generate_error_page(url, &format!("Failed to load page: {e}"))
        }
    };

    let dom = cpu::dom::parse_html(&html);
    info!("[CPU] Parsed {} nodes from {url}", dom.nodes.len());

    if let Err(e) = tx.send(PipelineMsg::HtmlReady {
        url: url.to_string(),
        html,
        dom,
    }) {
        error!("[CPU] Failed to send to NPU: {e}");
    }
}
