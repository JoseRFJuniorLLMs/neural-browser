//! EVA Panel — chat UI state for the EVA side panel.
//!
//! Manages conversation history, user input, and panel visibility.
//! The GPU renderer reads this state to draw the EVA panel.

use std::time::Instant;

/// Who sent a chat message.
#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    User,
    Eva,
    System,
}

/// A single message in the EVA conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub text: String,
    pub timestamp: Instant,
}

/// EVA chat panel state — drives the side panel UI.
pub struct EvaPanel {
    /// Whether the panel is currently visible.
    pub visible: bool,
    /// Conversation history.
    pub messages: Vec<ChatMessage>,
    /// Current user input text.
    input_text: String,
    /// Whether we are waiting for EVA to respond.
    pub is_loading: bool,
    /// Scroll offset for message history (in logical lines).
    pub scroll_offset: f32,
}

impl EvaPanel {
    /// Create a new (hidden) EVA panel.
    pub fn new() -> Self {
        Self {
            visible: false,
            messages: vec![ChatMessage {
                role: Role::System,
                text: "EVA is ready. Ask me anything about the page.".into(),
                timestamp: Instant::now(),
            }],
            input_text: String::new(),
            is_loading: false,
            scroll_offset: 0.0,
        }
    }

    /// Toggle panel visibility.
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Maximum number of messages to keep in history.
    const MAX_MESSAGES: usize = 100;

    /// Add a user message to the conversation.
    pub fn add_user_message(&mut self, text: String) {
        self.messages.push(ChatMessage {
            role: Role::User,
            text,
            timestamp: Instant::now(),
        });
        self.trim_messages();
    }

    /// Add an EVA response to the conversation.
    pub fn add_eva_response(&mut self, text: String) {
        self.messages.push(ChatMessage {
            role: Role::Eva,
            text,
            timestamp: Instant::now(),
        });
        self.is_loading = false;
        self.trim_messages();
    }

    /// Trim old messages if we exceed the limit.
    fn trim_messages(&mut self) {
        if self.messages.len() > Self::MAX_MESSAGES {
            let excess = self.messages.len() - Self::MAX_MESSAGES;
            self.messages.drain(..excess);
        }
    }

    /// Set loading state (waiting for EVA response).
    pub fn set_loading(&mut self, loading: bool) {
        self.is_loading = loading;
    }

    /// Clear the conversation history.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.messages.push(ChatMessage {
            role: Role::System,
            text: "Conversation cleared. Ask me anything.".into(),
            timestamp: Instant::now(),
        });
        self.input_text.clear();
        self.is_loading = false;
        self.scroll_offset = 0.0;
    }

    /// Handle a character typed into the EVA input field.
    pub fn input_char(&mut self, c: char) {
        if !c.is_control() {
            self.input_text.push(c);
        }
    }

    /// Handle backspace in the EVA input field.
    pub fn input_backspace(&mut self) {
        self.input_text.pop();
    }

    /// Get the current input text.
    pub fn get_input(&self) -> &str {
        &self.input_text
    }

    /// Take the current input text (clears it).
    pub fn take_input(&mut self) -> String {
        std::mem::take(&mut self.input_text)
    }

    /// Check if the panel is visible and should receive keyboard input.
    pub fn is_focused(&self) -> bool {
        self.visible
    }
}
