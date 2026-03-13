//! Tab management for the Neural Browser.
//!
//! Each tab holds its own URL, content, scroll position and loading state.
//! The GPU renderer uses the active tab to decide what to display.

use crate::npu::ContentBlock;

/// Unique identifier for a browser tab.
pub type TabId = u32;

/// A single browser tab.
#[derive(Debug)]
pub struct Tab {
    pub id: TabId,
    pub title: String,
    pub url: String,
    pub content: Vec<ContentBlock>,
    pub scroll_y: f32,
    pub loading: bool,
}

impl Tab {
    fn new(id: TabId) -> Self {
        Self {
            id,
            title: "New Tab".into(),
            url: String::new(),
            content: Vec::new(),
            scroll_y: 0.0,
            loading: false,
        }
    }
}

/// Manages multiple browser tabs.
pub struct TabManager {
    tabs: Vec<Tab>,
    active_index: usize,
    next_id: TabId,
}

impl TabManager {
    /// Create a new TabManager with one empty tab.
    pub fn new() -> Self {
        Self {
            tabs: vec![Tab::new(1)],
            active_index: 0,
            next_id: 2,
        }
    }

    /// Get a reference to the active tab.
    pub fn active_tab(&self) -> &Tab {
        &self.tabs[self.active_index]
    }

    /// Get a mutable reference to the active tab.
    pub fn active_tab_mut(&mut self) -> &mut Tab {
        &mut self.tabs[self.active_index]
    }

    /// Get the active tab's ID.
    pub fn active_id(&self) -> TabId {
        self.tabs[self.active_index].id
    }

    /// Open a new tab and make it active. Returns the new tab's ID.
    pub fn new_tab(&mut self) -> TabId {
        let id = self.next_id;
        self.next_id += 1;
        let tab = Tab::new(id);
        self.tabs.push(tab);
        self.active_index = self.tabs.len() - 1;
        id
    }

    /// Close the tab with the given ID.
    /// If it's the last tab, it won't be closed.
    /// Returns true if the tab was closed.
    pub fn close_tab(&mut self, id: TabId) -> bool {
        if self.tabs.len() <= 1 {
            return false;
        }
        if let Some(idx) = self.tabs.iter().position(|t| t.id == id) {
            self.tabs.remove(idx);
            // Adjust active index
            if self.active_index >= self.tabs.len() {
                self.active_index = self.tabs.len() - 1;
            } else if idx < self.active_index {
                self.active_index -= 1;
            }
            true
        } else {
            false
        }
    }

    /// Close the currently active tab.
    pub fn close_active(&mut self) -> bool {
        let id = self.active_id();
        self.close_tab(id)
    }

    /// Switch to the tab with the given ID.
    pub fn switch_to(&mut self, id: TabId) -> bool {
        if let Some(idx) = self.tabs.iter().position(|t| t.id == id) {
            self.active_index = idx;
            true
        } else {
            false
        }
    }

    /// Switch to the next tab (wraps around).
    pub fn switch_next(&mut self) {
        self.active_index = (self.active_index + 1) % self.tabs.len();
    }

    /// Switch to the previous tab (wraps around).
    pub fn switch_prev(&mut self) {
        if self.active_index == 0 {
            self.active_index = self.tabs.len() - 1;
        } else {
            self.active_index -= 1;
        }
    }

    /// Number of open tabs.
    pub fn tab_count(&self) -> usize {
        self.tabs.len()
    }

    /// Iterator over all tabs.
    pub fn tabs(&self) -> &[Tab] {
        &self.tabs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_has_one_tab() {
        let tm = TabManager::new();
        assert_eq!(tm.tab_count(), 1);
        assert_eq!(tm.active_id(), 1);
    }

    #[test]
    fn test_new_tab_increments_id() {
        let mut tm = TabManager::new();
        let id2 = tm.new_tab();
        assert_eq!(id2, 2);
        let id3 = tm.new_tab();
        assert_eq!(id3, 3);
        assert_eq!(tm.tab_count(), 3);
    }

    #[test]
    fn test_new_tab_becomes_active() {
        let mut tm = TabManager::new();
        let id = tm.new_tab();
        assert_eq!(tm.active_id(), id);
    }

    #[test]
    fn test_close_last_tab_refused() {
        let mut tm = TabManager::new();
        assert!(!tm.close_active());
        assert_eq!(tm.tab_count(), 1);
    }

    #[test]
    fn test_close_active_tab() {
        let mut tm = TabManager::new();
        tm.new_tab();
        assert_eq!(tm.tab_count(), 2);
        assert!(tm.close_active());
        assert_eq!(tm.tab_count(), 1);
    }

    #[test]
    fn test_close_adjusts_active_index() {
        let mut tm = TabManager::new();
        let _id2 = tm.new_tab();
        let id3 = tm.new_tab();
        // Active is tab 3 (index 2)
        assert_eq!(tm.active_id(), id3);
        // Close tab 1 (index 0)
        tm.close_tab(1);
        // Active should still be tab 3
        assert_eq!(tm.active_id(), id3);
        assert_eq!(tm.tab_count(), 2);
    }

    #[test]
    fn test_switch_to() {
        let mut tm = TabManager::new();
        let id2 = tm.new_tab();
        tm.switch_to(1);
        assert_eq!(tm.active_id(), 1);
        tm.switch_to(id2);
        assert_eq!(tm.active_id(), id2);
    }

    #[test]
    fn test_switch_to_invalid() {
        let mut tm = TabManager::new();
        assert!(!tm.switch_to(999));
        assert_eq!(tm.active_id(), 1);
    }

    #[test]
    fn test_switch_next_wraps() {
        let mut tm = TabManager::new();
        tm.new_tab();
        tm.new_tab();
        tm.switch_to(1);
        assert_eq!(tm.active_id(), 1);
        tm.switch_next();
        assert_eq!(tm.active_id(), 2);
        tm.switch_next();
        assert_eq!(tm.active_id(), 3);
        tm.switch_next(); // wraps
        assert_eq!(tm.active_id(), 1);
    }

    #[test]
    fn test_switch_prev_wraps() {
        let mut tm = TabManager::new();
        tm.new_tab();
        tm.switch_to(1);
        tm.switch_prev(); // wraps to last
        assert_eq!(tm.active_id(), 2);
        tm.switch_prev();
        assert_eq!(tm.active_id(), 1);
    }

    #[test]
    fn test_active_tab_mut() {
        let mut tm = TabManager::new();
        tm.active_tab_mut().title = "Custom Title".into();
        assert_eq!(tm.active_tab().title, "Custom Title");
    }

    #[test]
    fn test_tab_url_and_scroll() {
        let mut tm = TabManager::new();
        tm.active_tab_mut().url = "https://example.com".into();
        tm.active_tab_mut().scroll_y = 150.0;
        assert_eq!(tm.active_tab().url, "https://example.com");
        assert_eq!(tm.active_tab().scroll_y, 150.0);
    }

    #[test]
    fn test_close_middle_tab() {
        let mut tm = TabManager::new();
        let id2 = tm.new_tab();
        let _id3 = tm.new_tab();
        tm.switch_to(id2);
        tm.close_active();
        // Should switch to neighbor
        assert_eq!(tm.tab_count(), 2);
    }

    #[test]
    fn test_tabs_iterator() {
        let mut tm = TabManager::new();
        tm.new_tab();
        tm.new_tab();
        let ids: Vec<TabId> = tm.tabs().iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }
}
