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
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            bg: [0.06, 0.06, 0.08, 1.0],
            text: [0.85, 0.85, 0.88, 1.0],
            text_dim: [0.5, 0.5, 0.55, 1.0],
            heading: [0.95, 0.95, 0.97, 1.0],
            link: [0.4, 0.6, 1.0, 1.0],
            link_hover: [0.55, 0.75, 1.0, 1.0],
            link_visited: [0.6, 0.4, 0.85, 1.0],
            code_bg: [0.1, 0.1, 0.12, 1.0],
            quote_bg: [0.12, 0.12, 0.15, 1.0],
            url_bar_bg: [0.15, 0.15, 0.18, 1.0],
            url_bar_border: [0.3, 0.3, 0.35, 1.0],
            separator: [0.25, 0.25, 0.28, 1.0],
            loading: [0.4, 0.7, 1.0, 1.0],
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
