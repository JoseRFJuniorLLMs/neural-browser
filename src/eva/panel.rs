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
    /// When loading started (for timeout detection).
    pub loading_started: Option<Instant>,
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
            loading_started: None,
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
        self.loading_started = if loading { Some(Instant::now()) } else { None };
    }

    /// Check if loading has timed out (> 60 seconds).
    pub fn check_loading_timeout(&mut self) -> bool {
        if let Some(started) = self.loading_started {
            if started.elapsed() > std::time::Duration::from_secs(60) {
                self.is_loading = false;
                self.loading_started = None;
                self.messages.push(ChatMessage {
                    role: Role::System,
                    text: "⏱ Request timed out. Try again.".into(),
                    timestamp: Instant::now(),
                    provider: None,
                });
                return true;
            }
        }
        false
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

    /// Append a string to the input field (for paste support).
    pub fn input_append(&mut self, text: &str) {
        for c in text.chars() {
            if !c.is_control() {
                self.input_text.push(c);
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_panel_starts_hidden() {
        let panel = EvaPanel::new();
        assert!(!panel.visible);
        assert!(!panel.is_focused());
        assert_eq!(panel.provider, AiProvider::Eva);
        assert_eq!(panel.messages.len(), 1); // system message
    }

    #[test]
    fn test_toggle() {
        let mut panel = EvaPanel::new();
        assert!(!panel.visible);
        panel.toggle();
        assert!(panel.visible);
        assert!(panel.is_focused());
        panel.toggle();
        assert!(!panel.visible);
    }

    #[test]
    fn test_cycle_provider() {
        let mut panel = EvaPanel::new();
        assert_eq!(panel.provider, AiProvider::Eva);
        panel.cycle_provider();
        assert_eq!(panel.provider, AiProvider::Claude);
        panel.cycle_provider();
        assert_eq!(panel.provider, AiProvider::Gemini);
        panel.cycle_provider();
        assert_eq!(panel.provider, AiProvider::Gpt4);
        panel.cycle_provider();
        assert_eq!(panel.provider, AiProvider::Local);
        panel.cycle_provider();
        assert_eq!(panel.provider, AiProvider::Eva); // wraps
    }

    #[test]
    fn test_input_and_take() {
        let mut panel = EvaPanel::new();
        panel.input_char('h');
        panel.input_char('i');
        assert_eq!(panel.get_input(), "hi");
        let taken = panel.take_input();
        assert_eq!(taken, "hi");
        assert_eq!(panel.get_input(), ""); // cleared
    }

    #[test]
    fn test_input_backspace() {
        let mut panel = EvaPanel::new();
        panel.input_char('a');
        panel.input_char('b');
        panel.input_backspace();
        assert_eq!(panel.get_input(), "a");
        panel.input_backspace();
        assert_eq!(panel.get_input(), "");
        panel.input_backspace(); // no panic on empty
        assert_eq!(panel.get_input(), "");
    }

    #[test]
    fn test_control_chars_ignored() {
        let mut panel = EvaPanel::new();
        panel.input_char('\n');
        panel.input_char('\t');
        panel.input_char('\x08'); // backspace control char
        assert_eq!(panel.get_input(), "");
    }

    #[test]
    fn test_messages() {
        let mut panel = EvaPanel::new();
        panel.add_user_message("Hello".into());
        assert_eq!(panel.messages.len(), 2); // system + user
        assert_eq!(panel.messages[1].role, Role::User);

        panel.set_loading(true);
        assert!(panel.is_loading);

        panel.add_ai_response("Hi there!".into(), AiProvider::Eva);
        assert_eq!(panel.messages.len(), 3);
        assert_eq!(panel.messages[2].role, Role::Ai);
        assert!(!panel.is_loading); // auto-cleared
    }

    #[test]
    fn test_clear() {
        let mut panel = EvaPanel::new();
        panel.add_user_message("test".into());
        panel.add_ai_response("reply".into(), AiProvider::Claude);
        panel.input_char('x');
        panel.clear();
        assert_eq!(panel.messages.len(), 1); // only system msg
        assert_eq!(panel.get_input(), "");
        assert!(!panel.is_loading);
    }

    #[test]
    fn test_trim_messages_at_limit() {
        let mut panel = EvaPanel::new();
        for i in 0..150 {
            panel.add_user_message(format!("msg {i}"));
        }
        assert!(panel.messages.len() <= EvaPanel::MAX_MESSAGES);
    }
}
