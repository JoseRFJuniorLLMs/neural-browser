# Neural Browser

**A browser that uses all three processors: CPU + NPU + GPU working together.**

Most browsers use CPU for everything and GPU only for compositing. Neural Browser distributes work across all three processing units in your machine:

```
CPU (orchestrator)        NPU (intelligence)         GPU (pixels)
┌─────────────┐          ┌──────────────────┐       ┌──────────────────┐
│ TCP/TLS     │─HTML──►  │ Page Understand   │       │                  │
│ HTML parse  │          │ Content Extract   │       │ Text rasterize   │
│ DOM tree    │─DOM───►  │ Ad/Tracker Block  │       │ Layout + Paint   │
│ Event loop  │          │ Smart Prefetch    │       │ Image composite  │
│             │◄─intent─ │ Language Detect   │──►────│ Scroll/Animate   │
└─────────────┘          └──────────────────┘       └────────┬─────────┘
                                                             │
                                                        ┌────▼────┐
                                                        │ Display │
                                                        └─────────┘
```

## How It Works

Three dedicated threads, one per processor, connected by lock-free channels:

| Processor | Thread | What It Does | Tech |
|-----------|--------|-------------|------|
| **CPU** | `cpu-network` | HTTP fetch, HTML parsing, DOM tree | ureq + custom parser |
| **NPU** | `npu-engine` | Content understanding, ad blocking, smart prefetch | ONNX Runtime + DirectML |
| **GPU** | main thread | Text rasterization, layout, compositing | wgpu (Vulkan/DX12) + glyphon |

### Pipeline

```
URL → [CPU] fetch HTML → parse → DomTree
    → channel →
    [NPU] extract content → classify ads → score links → ContentBlocks
    → channel →
    [GPU] layout → paint text (glyphon) → composite → display
```

The NPU classifies every DOM element semantically (heading, paragraph, code, image, ad, navigation, boilerplate) so the GPU never needs to interpret HTML — it just renders pre-understood content blocks.

## Build

```bash
# Requirements: Rust nightly, Windows 10/11 with DirectML-compatible NPU/GPU
cargo build --release
```

### Dependencies

| Layer | Crate | Purpose |
|-------|-------|---------|
| CPU | `ureq` | HTTP/HTTPS client (rustls + webpki-roots) |
| NPU | `ort` | ONNX Runtime with DirectML execution provider |
| GPU | `wgpu` | Cross-platform GPU API (Vulkan/DX12) |
| GPU | `glyphon` | GPU-accelerated text rendering (cosmic-text) |

## Run

```bash
# Load a website
./target/release/neural-browser https://www.rust-lang.org

# HTTP sites work too
./target/release/neural-browser http://info.cern.ch

# Default (no args)
./target/release/neural-browser
```

### Keyboard

| Key | Action |
|-----|--------|
| **F6** | Focus URL bar (type new URL) |
| **Enter** | Navigate to typed URL |
| **Escape** | Cancel URL editing |
| **F5** | Refresh current page |
| **Alt+Left** | Back in history |
| **Alt+Right** | Forward in history |
| **Scroll** | Mouse wheel or PageUp/PageDown |
| **Home** | Scroll to top |
| **Click** | Follow links |

## Architecture

```
neural-browser/
├── src/
│   ├── main.rs              # 3-thread orchestrator + channels
│   ├── cpu/
│   │   ├── network.rs       # HTTP fetch + prefetch cache
│   │   ├── dom.rs           # HTML → flat DomTree (with tests)
│   │   └── start_page.rs    # Built-in neural://start welcome page
│   ├── npu/
│   │   ├── mod.rs           # NPU pipeline: extract → classify → summarize → prefetch
│   │   ├── content.rs       # DOM → semantic ContentBlocks
│   │   └── classifier.rs    # ML-based ad/tracker detection + classification
│   ├── gpu/
│   │   ├── mod.rs           # winit event loop + window/input management
│   │   ├── renderer.rs      # wgpu + glyphon text rendering
│   │   └── layout.rs        # ContentBlocks → positioned LayoutBoxes + hit testing
│   └── ui/
│       └── mod.rs           # Theme + navigation types
└── models/                   # ONNX models (NPU inference)
```

### NPU Content Classification

The NPU classifies DOM elements into semantic blocks:

| BlockKind | Description |
|-----------|------------|
| `Title` | Page title |
| `Heading` | h1-h6 with level |
| `Paragraph` | Text content |
| `Code` | Pre/code blocks with language detection |
| `Image` | With src and alt text |
| `Quote` | Blockquotes |
| `Link` | Clickable links with href |
| `List/ListItem` | Ordered and unordered lists |
| `Navigation` | Detected as non-content (filtered) |
| `Boilerplate` | Footer/sidebar (filtered by relevance score) |

Each block gets a **relevance score** (0.0-1.0). Low-relevance blocks (ads, nav, boilerplate) are filtered before reaching the GPU.

### Ad Blocking

ML-based ad detection (currently heuristic, ONNX model planned):
- URL pattern matching (doubleclick, googlesyndication, taboola, etc.)
- Tracker detection (analytics, pixel, beacon, utm_ params)
- Text pattern matching (sponsored, promoted, adchoice)

### Smart Prefetch

The NPU scores all links on the page by click probability:
- Same-domain links score higher
- "Next", "Continue", "Read more" patterns score higher
- Top 3 links are prefetched by the CPU thread

## Performance

| Metric | Value |
|--------|-------|
| Binary size | 11 MB |
| Startup | ~2s (font system init) |
| Page load (HTTP) | <1s fetch + parse + NPU + render |
| GPU adapter | Auto-selects best (Vulkan/DX12) |

## Status

Working:
- [x] HTTP/HTTPS fetch with TLS
- [x] HTML parsing → DOM tree
- [x] NPU content extraction pipeline
- [x] Ad/tracker blocking (heuristic + ML classifier)
- [x] Smart link prefetch (top 3 by click probability)
- [x] GPU text rendering (glyphon + wgpu)
- [x] Dark theme layout engine
- [x] URL bar with keyboard navigation
- [x] Scroll (mouse wheel + keyboard)
- [x] 3-thread pipeline (CPU→NPU→GPU)
- [x] Click on links (hit testing + navigation)
- [x] Navigation history (back/forward)
- [x] Language detection
- [x] Page summarization
- [x] Built-in start page (`neural://start`)
- [x] Mouse hover link preview
- [x] Layout caching (recompute on change only)
- [x] Image support (PNG, JPEG, WebP)
- [x] Visited URL tracking

Planned:
- [ ] ONNX models for NPU (MarkupLM, ad classifier)
- [ ] Image decode → GPU textures
- [ ] Redirect handling (301/302)
- [ ] Tabs
- [ ] Rectangle shader (colored backgrounds)
- [ ] CSS color extraction

## Why?

Modern PCs ship with NPUs that sit completely idle. This browser is an experiment in using **all available silicon** — not just CPU and GPU, but the neural processor too.

The NPU handles "understanding" (what is this content? is it an ad? what will the user click next?) while the GPU handles "showing" (text, images, layout) and the CPU handles "connecting" (network, parsing, coordination).

## License

MIT
