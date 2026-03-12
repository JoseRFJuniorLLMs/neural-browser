//! Neural Browser — CPU + NPU + GPU working together
//!
//! Architecture:
//!   CPU  → networking, HTML parsing, DOM tree, event loop
//!   NPU  → content understanding, ad blocking, summarization (ONNX + DirectML)
//!   GPU  → layout rendering, text rasterization, compositing (wgpu)
//!
//! Pipeline:
//!   URL → [CPU] fetch+parse → DOM
//!       → [NPU] understand + extract + classify
//!       → [GPU] layout + paint + composite → display

mod cpu;
mod npu;
mod gpu;
mod ui;

use anyhow::Result;
use crossbeam_channel as channel;
use log::{info, error};

/// Messages flowing between the three processors
#[derive(Debug, Clone)]
pub enum PipelineMsg {
    // CPU → NPU: raw content ready for AI processing
    HtmlReady {
        url: String,
        html: String,
        dom: cpu::dom::DomTree,
    },

    // NPU → GPU: AI-processed content ready for rendering
    ContentReady {
        url: String,
        blocks: Vec<npu::ContentBlock>,
        ads_blocked: usize,
    },

    // GPU → UI: frame rendered
    FrameReady,

    // UI → CPU: user navigation
    Navigate(String),

    // NPU → CPU: prefetch suggestion
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
            info!("[NPU] Thread started — DirectML inference engine");
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
            info!("[CPU] Thread started — networking + parsing");
            let net = cpu::network::NetworkEngine::new();

            // Navigate to start page
            let start_url = std::env::args()
                .nth(1)
                .unwrap_or_else(|| "https://example.com".into());

            if let Err(e) = process_url(&net, &start_url, &cpu_to_npu_tx) {
                error!("[CPU] Failed to load {start_url}: {e}");
            }

            // Listen for navigation events
            for msg in cpu_ui_rx {
                match msg {
                    PipelineMsg::Navigate(url) => {
                        info!("[CPU] Navigating to: {url}");
                        if let Err(e) = process_url(&net, &url, &cpu_to_npu_tx) {
                            error!("[CPU] Failed to load {url}: {e}");
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

fn process_url(
    net: &cpu::network::NetworkEngine,
    url: &str,
    tx: &channel::Sender<PipelineMsg>,
) -> Result<()> {
    let html = net.fetch(url)?;
    let dom = cpu::dom::parse_html(&html);

    info!("[CPU] Parsed {} nodes from {url}", dom.nodes.len());

    tx.send(PipelineMsg::HtmlReady {
        url: url.to_string(),
        html,
        dom,
    })?;

    Ok(())
}
