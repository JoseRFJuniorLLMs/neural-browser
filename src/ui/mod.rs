//! UI module — shared types for user interaction.
//!
//! Handles the bridge between user input (keyboard/mouse)
//! and the three processors (CPU/NPU/GPU).

/// UI theme colors (dark mode default).
pub struct Theme {
    pub bg: [f32; 4],
    pub text: [f32; 4],
    pub text_dim: [f32; 4],
    pub heading: [f32; 4],
    pub link: [f32; 4],
    pub code_bg: [f32; 4],
    pub quote_bg: [f32; 4],
    pub url_bar_bg: [f32; 4],
    pub separator: [f32; 4],
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            bg: [0.08, 0.08, 0.10, 1.0],
            text: [0.85, 0.85, 0.88, 1.0],
            text_dim: [0.5, 0.5, 0.55, 1.0],
            heading: [0.95, 0.95, 0.97, 1.0],
            link: [0.4, 0.6, 1.0, 1.0],
            code_bg: [0.1, 0.1, 0.12, 1.0],
            quote_bg: [0.12, 0.12, 0.15, 1.0],
            url_bar_bg: [0.15, 0.15, 0.18, 1.0],
            separator: [0.25, 0.25, 0.28, 1.0],
        }
    }
}

/// Navigation action triggered by user input.
#[derive(Debug, Clone)]
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
