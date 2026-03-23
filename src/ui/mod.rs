//! UI module — shared types for user interaction.
//!
//! Handles the bridge between user input (keyboard/mouse)
//! and the three processors (CPU/NPU/GPU).

/// UI theme colors (dark mode default).
#[derive(Debug, Clone)]
#[allow(dead_code)] // Will be used when renderer reads theme from config
pub struct Theme {
    pub bg: [f32; 4],
    pub text: [f32; 4],
    pub text_dim: [f32; 4],
    pub heading: [f32; 4],
    pub link: [f32; 4],
    pub link_hover: [f32; 4],
    pub link_visited: [f32; 4],
    pub code_bg: [f32; 4],
    pub quote_bg: [f32; 4],
    pub url_bar_bg: [f32; 4],
    pub url_bar_border: [f32; 4],
    pub separator: [f32; 4],
    pub loading: [f32; 4],
    pub toolbar_bg: [f32; 4],
    pub button_bg: [f32; 4],
    pub button_hover: [f32; 4],
    pub button_text: [f32; 4],
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            bg: [0.98, 0.98, 0.98, 1.0],         // white page background
            text: [0.13, 0.13, 0.13, 1.0],        // near-black text
            text_dim: [0.45, 0.45, 0.47, 1.0],    // dimmed text (gray)
            heading: [0.08, 0.08, 0.10, 1.0],     // black headings
            link: [0.10, 0.33, 0.72, 1.0],        // Google-blue links
            link_hover: [0.15, 0.40, 0.85, 1.0],  // brighter blue on hover
            link_visited: [0.40, 0.15, 0.55, 1.0],// purple visited
            code_bg: [0.94, 0.94, 0.96, 1.0],     // light gray code bg
            quote_bg: [0.93, 0.93, 0.95, 1.0],    // light gray quote bg
            url_bar_bg: [0.15, 0.15, 0.18, 1.0],
            url_bar_border: [0.3, 0.3, 0.35, 1.0],
            separator: [0.25, 0.25, 0.28, 1.0],
            loading: [0.4, 0.7, 1.0, 1.0],
            toolbar_bg: [0.12, 0.12, 0.15, 1.0],
            button_bg: [0.18, 0.18, 0.22, 1.0],
            button_hover: [0.25, 0.25, 0.30, 1.0],
            button_text: [0.75, 0.75, 0.80, 1.0],
        }
    }
}

/// Navigation action triggered by user input.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Will be used when input handling is refactored
pub enum NavAction {
    /// Go to URL
    Navigate(String),
    /// Go back in history
    Back,
    /// Go forward in history
    Forward,
    /// Refresh current page
    Refresh,
    /// Scroll by delta pixels
    Scroll(f32),
    /// Click at (x, y) position
    Click(f32, f32),
}
