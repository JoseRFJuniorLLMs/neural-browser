//! CSS Selector Matching — matches selectors against DOM nodes.
//!
//! Evaluates whether a CSS selector matches a specific node in the DOM tree.
//! Handles compound selectors, combinators, pseudo-classes, attribute selectors.

use super::parser::{Selector, SelectorPart, Combinator, AttrOp};
use crate::cpu::dom::{DomNode, DomTree};

/// Check if a selector matches a DOM node.
pub fn matches_selector(selector: &Selector, node: &DomNode, tree: &DomTree) -> bool {
    // Walk the selector parts in reverse (rightmost = subject)
    let mut parts_iter = selector.parts.iter().rev().peekable();
    let mut current_node_id = Some(node.id);

    // First, match all simple selectors at the rightmost position
    let mut simple_parts = Vec::new();
    while let Some(part) = parts_iter.peek() {
        if matches!(part, SelectorPart::Combinator(_)) {
            break;
        }
        simple_parts.push(parts_iter.next().unwrap());
    }
    simple_parts.reverse();

    // Check the subject node matches all simple selectors
    if let Some(nid) = current_node_id {
        if let Some(n) = tree.nodes.get(nid) {
            if !matches_compound(&simple_parts, n) {
                return false;
            }
        } else {
            return false;
        }
    }

    // Walk up the selector chain with combinators
    while let Some(part) = parts_iter.next() {
        let combinator = match part {
            SelectorPart::Combinator(c) => *c,
            _ => continue,
        };

        // Collect the next compound selector
        let mut next_simple = Vec::new();
        while let Some(p) = parts_iter.peek() {
            if matches!(p, SelectorPart::Combinator(_)) {
                break;
            }
            next_simple.push(parts_iter.next().unwrap());
        }
        next_simple.reverse();

        match combinator {
            Combinator::Descendant => {
                // Walk up ancestors until we find a match
                let mut found = false;
                let mut ancestor_id = current_node_id.and_then(|id| tree.nodes.get(id)?.parent);
                while let Some(aid) = ancestor_id {
                    if let Some(ancestor) = tree.nodes.get(aid) {
                        if matches_compound(&next_simple, ancestor) {
                            current_node_id = Some(aid);
                            found = true;
                            break;
                        }
                        ancestor_id = ancestor.parent;
                    } else {
                        break;
                    }
                }
                if !found {
                    return false;
                }
            }
            Combinator::Child => {
                // Parent must match
                let parent_id = current_node_id.and_then(|id| tree.nodes.get(id)?.parent);
                if let Some(pid) = parent_id {
                    if let Some(parent) = tree.nodes.get(pid) {
                        if !matches_compound(&next_simple, parent) {
                            return false;
                        }
                        current_node_id = Some(pid);
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            Combinator::NextSibling => {
                // Previous sibling must match
                if let Some(prev) = get_previous_sibling(current_node_id.unwrap_or(0), tree) {
                    if !matches_compound(&next_simple, prev) {
                        return false;
                    }
                    current_node_id = Some(prev.id);
                } else {
                    return false;
                }
            }
            Combinator::Subsequent => {
                // Any preceding sibling must match
                let mut found = false;
                let mut sibling = get_previous_sibling(current_node_id.unwrap_or(0), tree);
                while let Some(s) = sibling {
                    if matches_compound(&next_simple, s) {
                        current_node_id = Some(s.id);
                        found = true;
                        break;
                    }
                    sibling = get_previous_sibling(s.id, tree);
                }
                if !found {
                    return false;
                }
            }
        }
    }

    true
}

/// Match a compound selector (list of simple selectors) against a node.
fn matches_compound(parts: &[&SelectorPart], node: &DomNode) -> bool {
    for part in parts {
        if !matches_simple(part, node) {
            return false;
        }
    }
    true
}

/// Match a single simple selector against a node.
fn matches_simple(part: &SelectorPart, node: &DomNode) -> bool {
    match part {
        SelectorPart::Universal => true,
        SelectorPart::Tag(tag) => node.tag == *tag,
        SelectorPart::Class(class) => {
            node.attrs.get("class")
                .map(|c| c.split_whitespace().any(|cls| cls == class.as_str()))
                .unwrap_or(false)
        }
        SelectorPart::Id(id) => {
            node.attrs.get("id")
                .map(|i| i == id)
                .unwrap_or(false)
        }
        SelectorPart::Attribute { name, op, value } => {
            match node.attrs.get(name) {
                None => false,
                Some(attr_val) => {
                    match (op, value) {
                        (None, _) => true, // [attr] — just existence
                        (Some(AttrOp::Equals), Some(v)) => attr_val == v,
                        (Some(AttrOp::Includes), Some(v)) => {
                            attr_val.split_whitespace().any(|w| w == v.as_str())
                        }
                        (Some(AttrOp::DashMatch), Some(v)) => {
                            attr_val == v || attr_val.starts_with(&format!("{v}-"))
                        }
                        (Some(AttrOp::Prefix), Some(v)) => attr_val.starts_with(v.as_str()),
                        (Some(AttrOp::Suffix), Some(v)) => attr_val.ends_with(v.as_str()),
                        (Some(AttrOp::Substring), Some(v)) => attr_val.contains(v.as_str()),
                        _ => false,
                    }
                }
            }
        }
        SelectorPart::PseudoClass(name, _args) => {
            // Basic pseudo-class support (structural ones need tree context)
            match name.as_str() {
                "hover" | "active" | "focus" | "visited" | "link" => {
                    // Dynamic pseudo-classes — always false during static matching.
                    // The renderer handles these dynamically.
                    match name.as_str() {
                        "link" => node.tag == "a" && node.attrs.contains_key("href"),
                        _ => false,
                    }
                }
                "first-child" => {
                    // True if node is the first child of its parent
                    // We'd need tree context; for now, approximate
                    node.depth > 0 // placeholder
                }
                "root" => node.depth == 0 && node.tag == "html",
                "empty" => node.text.is_empty() && node.children.is_empty(),
                "not" => {
                    // :not() inversion — would need to parse inner selector
                    // For now, always true (matches if not-condition can't be evaluated)
                    true
                }
                _ => true, // Unknown pseudo-classes match by default
            }
        }
        SelectorPart::PseudoElement(_) => {
            // Pseudo-elements always match the element (rendering handles them)
            true
        }
        SelectorPart::Combinator(_) => true, // Handled at higher level
    }
}

/// Get the previous sibling of a node in the DOM tree.
fn get_previous_sibling<'a>(node_id: usize, tree: &'a DomTree) -> Option<&'a DomNode> {
    let node = tree.nodes.get(node_id)?;
    let parent_id = node.parent?;
    let parent = tree.nodes.get(parent_id)?;

    let pos = parent.children.iter().position(|&id| id == node_id)?;
    if pos == 0 {
        return None;
    }
    tree.nodes.get(parent.children[pos - 1])
}

/// Get the index of a node among its siblings (0-based).
#[allow(dead_code)]
fn sibling_index(node_id: usize, tree: &DomTree) -> Option<usize> {
    let node = tree.nodes.get(node_id)?;
    let parent_id = node.parent?;
    let parent = tree.nodes.get(parent_id)?;
    parent.children.iter().position(|&id| id == node_id)
}

/// Count same-type siblings before this node.
#[allow(dead_code)]
fn typed_sibling_index(node_id: usize, tree: &DomTree) -> Option<usize> {
    let node = tree.nodes.get(node_id)?;
    let parent_id = node.parent?;
    let parent = tree.nodes.get(parent_id)?;
    let tag = &node.tag;
    let mut count = 0;
    for &child_id in &parent.children {
        if child_id == node_id {
            return Some(count);
        }
        if let Some(child) = tree.nodes.get(child_id) {
            if child.tag == *tag {
                count += 1;
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::dom::parse_html;
    use crate::css::parser::parse_stylesheet;

    fn first_selector(css: &str) -> Selector {
        let ss = parse_stylesheet(css);
        ss.rules[0].selectors[0].clone()
    }

    #[test]
    fn test_tag_selector() {
        let tree = parse_html("<html><body><p>Hello</p></body></html>");
        let sel = first_selector("p { }");
        let p_node = tree.nodes.iter().find(|n| n.tag == "p").unwrap();
        assert!(matches_selector(&sel, p_node, &tree));

        let body = tree.nodes.iter().find(|n| n.tag == "body").unwrap();
        assert!(!matches_selector(&sel, body, &tree));
    }

    #[test]
    fn test_class_selector() {
        let tree = parse_html(r#"<div class="foo bar">Hi</div><div class="baz">Bye</div>"#);
        let sel = first_selector(".foo { }");
        let foo = tree.nodes.iter().find(|n| {
            n.attrs.get("class").map(|c| c.contains("foo")).unwrap_or(false)
        }).unwrap();
        assert!(matches_selector(&sel, foo, &tree));

        let baz = tree.nodes.iter().find(|n| {
            n.attrs.get("class").map(|c| c.contains("baz")).unwrap_or(false)
        }).unwrap();
        assert!(!matches_selector(&sel, baz, &tree));
    }

    #[test]
    fn test_id_selector() {
        let tree = parse_html(r#"<div id="main">Content</div><div id="other">Other</div>"#);
        let sel = first_selector("#main { }");
        let main = tree.nodes.iter().find(|n| n.attrs.get("id") == Some(&"main".to_string())).unwrap();
        assert!(matches_selector(&sel, main, &tree));

        let other = tree.nodes.iter().find(|n| n.attrs.get("id") == Some(&"other".to_string())).unwrap();
        assert!(!matches_selector(&sel, other, &tree));
    }

    #[test]
    fn test_descendant_combinator() {
        let tree = parse_html("<div><p>Inside div</p></div><p>Outside div</p>");
        let sel = first_selector("div p { }");
        // The p inside div should match
        let ps: Vec<_> = tree.nodes.iter().filter(|n| n.tag == "p").collect();
        assert!(ps.len() >= 1);
        // First p is inside div
        assert!(matches_selector(&sel, ps[0], &tree));
    }

    #[test]
    fn test_child_combinator() {
        let tree = parse_html("<div><p>Direct child</p></div>");
        let sel = first_selector("div > p { }");
        let p = tree.nodes.iter().find(|n| n.tag == "p").unwrap();
        assert!(matches_selector(&sel, p, &tree));
    }

    #[test]
    fn test_universal_selector() {
        let tree = parse_html("<div>Hello</div>");
        let sel = first_selector("* { }");
        let div = tree.nodes.iter().find(|n| n.tag == "div").unwrap();
        assert!(matches_selector(&sel, div, &tree));
    }

    #[test]
    fn test_attribute_exists() {
        let tree = parse_html(r#"<input disabled type="text">"#);
        let sel = first_selector("[disabled] { }");
        let input = tree.nodes.iter().find(|n| n.tag == "input").unwrap();
        assert!(matches_selector(&sel, input, &tree));
    }

    #[test]
    fn test_attribute_equals() {
        let tree = parse_html(r#"<input type="text"><input type="password">"#);
        let sel = first_selector(r#"[type="text"] { }"#);
        let text_input = tree.nodes.iter().find(|n| {
            n.attrs.get("type") == Some(&"text".to_string())
        }).unwrap();
        assert!(matches_selector(&sel, text_input, &tree));

        let pw_input = tree.nodes.iter().find(|n| {
            n.attrs.get("type") == Some(&"password".to_string())
        }).unwrap();
        assert!(!matches_selector(&sel, pw_input, &tree));
    }

    #[test]
    fn test_compound_selector() {
        let tree = parse_html(r#"<p class="intro">Intro</p><p>Normal</p>"#);
        let sel = first_selector("p.intro { }");
        let intro = tree.nodes.iter().find(|n| {
            n.tag == "p" && n.attrs.get("class") == Some(&"intro".to_string())
        }).unwrap();
        assert!(matches_selector(&sel, intro, &tree));
    }

    #[test]
    fn test_link_pseudo_class() {
        let tree = parse_html(r#"<a href="/page">Link</a><span>Not a link</span>"#);
        let sel = first_selector("a:link { }");
        let a = tree.nodes.iter().find(|n| n.tag == "a").unwrap();
        assert!(matches_selector(&sel, a, &tree));
    }

    #[test]
    fn test_empty_pseudo_class() {
        let tree = parse_html("<div></div><div>Content</div>");
        let sel = first_selector("div:empty { }");
        let divs: Vec<_> = tree.nodes.iter().filter(|n| n.tag == "div").collect();
        // First div is empty
        assert!(matches_selector(&sel, divs[0], &tree));
        // Second div has content
        assert!(!matches_selector(&sel, divs[1], &tree));
    }
}
