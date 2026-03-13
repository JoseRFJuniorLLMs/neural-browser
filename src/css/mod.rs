//! CSS Engine — tokenizer, parser, selector matching, cascade.
//!
//! Zero dependencies. Rewritten from scratch for Neural Browser.
//! Inspired by the CSS3 specification and open-source browser engines
//! (Servo, Gosub), but adapted for our CPU+NPU+GPU pipeline.
//!
//! Pipeline integration (active):
//!   NPU: extract <style> tags + inline styles → parse → cascade → ComputedStyle per node
//!   NPU: attach ComputedStyle to ContentBlocks (via node_id mapping)
//!   GPU: layout uses ComputedStyle for font-size, color, margins, display:none, etc.

pub mod tokenizer;
pub mod values;
pub mod parser;
pub mod selector;
pub mod cascade;

// Re-exports used by NPU (css cascade) and GPU (layout styling)
#[allow(unused_imports)]
pub use cascade::{ComputedStyle, StyledNode};
#[allow(unused_imports)]
pub use values::CssDisplay;
