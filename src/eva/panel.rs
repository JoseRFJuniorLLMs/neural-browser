//! AI Panel — chat UI state for the AI side panel.
//!
//! Manages conversation history, user input, panel visibility,
//! and current AI provider selection.
//! The GPU renderer reads this state to draw the panel.

use super::AiProvider;
use std::time::Instant;

/// Who sent a chat message.
#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    User,
    Ai,
    System,
}

/// A single message in the conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub text: String,
    pub timestamp: Instant,
    /// Which provider generated this response (None for User/System).
    pub provider: Option<AiProvider>,
}

/// AI chat panel state — drives the side panel UI.
pub struct EvaPanel {
    /// Whether the panel is currently visible.
    pub visible: bool,
    /// Conversation history.
    pub messages: Vec<ChatMessage>,
    /// Current user input text.
    input_text: String,
    /// Whether we are waiting for an AI response.
    pub is_loading: bool,
    /// Scroll offset for message history (in logical lines).
    pub scroll_offset: f32,
    /// Currently selected AI provider.
    pub provider: AiProvider,
}

impl EvaPanel {
    /// Create a new (hidden) AI panel.
    pub fn new() -> Self {
        Self {
            visible: false,
            messages: vec![ChatMessage {
                role: Role::System,
                text: "AI assistant ready. Tab to switch providers. Ask anything about the page.".into(),
                timestamp: Instant::now(),
                provider: None,
            }],
            input_text: String::new(),
            is_loading: false,
            scroll_offset: 0.0,
            provider: AiProvider::Eva,
        }
    }

    /// Toggle panel visibility.
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Cycle to the next AI provider.
    pub fn cycle_provider(&mut self) {
        self.provider = self.provider.next();
    }

    /// Set a specific provider.
    pub fn set_provider(&mut self, provider: AiProvider) {
        self.provider = provider;
    }

    /// Get the current provider display name.
    pub fn provider_name(&self) -> &'static str {
        self.provider.name()
    }

    /// Maximum number of messages to keep in history.
    const MAX_MESSAGES: usize = 100;

    /// Add a user message to the conversation.
    pub fn add_user_message(&mut self, text: String) {
        self.messages.push(ChatMessage {
            role: Role::User,
            text,
            timestamp: Instant::now(),
            provider: None,
        });
        self.trim_messages();
    }

    /// Add an AI response to the conversation.
    pub fn add_ai_response(&mut self, text: String, provider: AiProvider) {
        self.messages.push(ChatMessage {
            role: Role::Ai,
            text,
            timestamp: Instant::now(),
            provider: Some(provider),
        });
        self.is_loading = false;
        self.trim_messages();
    }

    /// Legacy: add EVA response (for backward compat).
    pub fn add_eva_response(&mut self, text: String) {
        self.add_ai_response(text, self.provider);
    }

    /// Trim old messages if we exceed the limit.
    fn trim_messages(&mut self) {
        if self.messages.len() > Self::MAX_MESSAGES {
            let excess = self.messages.len() - Self::MAX_MESSAGES;
            self.messages.drain(..excess);
        }
    }

    /// Set loading state (waiting for AI response).
    pub fn set_loading(&mut self, loading: bool) {
        self.is_loading = loading;
    }

    /// Clear the conversation history.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.messages.push(ChatMessage {
            role: Role::System,
            text: format!("Conversation cleared. Using {}. Ask me anything.", self.provider.name()),
            timestamp: Instant::now(),
            provider: None,
        });
        self.input_text.clear();
        self.is_loading = false;
        self.scroll_offset = 0.0;
    }

    /// Handle a character typed into the input field.
    pub fn input_char(&mut self, c: char) {
        if !c.is_control() {
            self.input_text.push(c);
        }
    }

    /// Handle backspace in the input field.
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
