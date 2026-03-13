//! CSS Cascade — compute final styles for each DOM node.
//!
//! Implements the CSS cascade algorithm:
//! 1. Collect all matching rules for each node
//! 2. Sort by specificity (and !important)
//! 3. Apply declarations to produce ComputedStyle
//! 4. Inherit inheritable properties from parent

use super::parser::{Stylesheet, Declaration, CssValue, Specificity};
use super::selector::matches_selector;
use super::values::*;
use crate::cpu::dom::DomTree;

/// Computed style for a single DOM node — all values resolved.
#[derive(Debug, Clone)]
pub struct ComputedStyle {
    // ── Box model ──
    pub display: CssDisplay,
    pub position: CssPosition,
    pub float: CssFloat,
    pub width: CssLength,
    pub height: CssLength,
    pub min_width: CssLength,
    pub min_height: CssLength,
    pub max_width: CssLength,
    pub max_height: CssLength,

    // ── Margin ──
    pub margin_top: CssLength,
    pub margin_right: CssLength,
    pub margin_bottom: CssLength,
    pub margin_left: CssLength,

    // ── Padding ──
    pub padding_top: CssLength,
    pub padding_right: CssLength,
    pub padding_bottom: CssLength,
    pub padding_left: CssLength,

    // ── Border ──
    pub border_width: CssLength,
    pub border_style: CssBorderStyle,
    pub border_color: CssColor,
    pub border_radius: CssLength,

    // ── Colors ──
    pub color: CssColor,
    pub background_color: CssColor,

    // ── Typography ──
    pub font_size: CssLength,
    pub font_weight: CssFontWeight,
    pub font_style: CssFontStyle,
    pub font_family: String,
    pub line_height: CssLength,
    pub text_align: CssTextAlign,
    pub text_decoration: CssTextDecoration,
    pub white_space: CssWhiteSpace,
    pub vertical_align: CssVerticalAlign,
    pub letter_spacing: CssLength,
    pub word_spacing: CssLength,
    pub text_indent: CssLength,
    pub text_transform: String,

    // ── Visual ──
    pub opacity: f32,
    pub overflow: CssOverflow,
    pub visibility: bool,
    pub cursor: String,
    pub z_index: Option<i32>,

    // ── Flex ──
    pub flex_direction: CssFlexDirection,
    pub flex_grow: f32,
    pub flex_shrink: f32,
    pub flex_basis: CssLength,
    pub justify_content: CssAlign,
    pub align_items: CssAlign,
    pub flex_wrap: String,
    pub gap: CssLength,

    // ── Background ──
    pub background_image: String,
    pub background_repeat: String,
    pub background_position: String,
    pub background_size: String,

    // ── List ──
    pub list_style_type: String,
}

impl Default for ComputedStyle {
    fn default() -> Self {
        Self {
            display: CssDisplay::Inline,
            position: CssPosition::Static,
            float: CssFloat::None,
            width: CssLength::Auto,
            height: CssLength::Auto,
            min_width: CssLength::None,
            min_height: CssLength::None,
            max_width: CssLength::None,
            max_height: CssLength::None,

            margin_top: CssLength::Zero,
            margin_right: CssLength::Zero,
            margin_bottom: CssLength::Zero,
            margin_left: CssLength::Zero,

            padding_top: CssLength::Zero,
            padding_right: CssLength::Zero,
            padding_bottom: CssLength::Zero,
            padding_left: CssLength::Zero,

            border_width: CssLength::Zero,
            border_style: CssBorderStyle::None,
            border_color: CssColor::BLACK,
            border_radius: CssLength::Zero,

            color: CssColor::BLACK,
            background_color: CssColor::TRANSPARENT,

            font_size: CssLength::Px(16.0),
            font_weight: CssFontWeight::NORMAL,
            font_style: CssFontStyle::Normal,
            font_family: String::new(),
            line_height: CssLength::Em(1.2),
            text_align: CssTextAlign::Left,
            text_decoration: CssTextDecoration::default(),
            white_space: CssWhiteSpace::Normal,
            vertical_align: CssVerticalAlign::Baseline,
            letter_spacing: CssLength::Zero,
            word_spacing: CssLength::Zero,
            text_indent: CssLength::Zero,
            text_transform: String::new(),

            opacity: 1.0,
            overflow: CssOverflow::Visible,
            visibility: true,
            cursor: "auto".into(),
            z_index: None,

            flex_direction: CssFlexDirection::Row,
            flex_grow: 0.0,
            flex_shrink: 1.0,
            flex_basis: CssLength::Auto,
            justify_content: CssAlign::Start,
            align_items: CssAlign::Stretch,
            flex_wrap: "nowrap".into(),
            gap: CssLength::Zero,

            background_image: String::new(),
            background_repeat: "repeat".into(),
            background_position: String::new(),
            background_size: "auto".into(),

            list_style_type: "disc".into(),
        }
    }
}

impl ComputedStyle {
    /// Get font size in pixels (for a default context).
    pub fn font_size_px(&self) -> f32 {
        self.font_size.to_px(16.0, 0.0, 0.0).max(1.0)
    }

    /// Get line height in pixels.
    pub fn line_height_px(&self) -> f32 {
        let fs = self.font_size_px();
        self.line_height.to_px(fs, 0.0, 0.0).max(fs)
    }

    /// Get the color as a [f32; 4] array for the renderer.
    pub fn color_array(&self) -> [f32; 4] {
        self.color.to_array()
    }

    /// Get background color as a [f32; 4] array.
    pub fn bg_color_array(&self) -> [f32; 4] {
        self.background_color.to_array()
    }
}

/// A DOM node paired with its computed style.
#[derive(Debug, Clone)]
pub struct StyledNode {
    pub node_id: usize,
    pub style: ComputedStyle,
    pub children: Vec<StyledNode>,
}

/// Build a style tree from a DOM tree and stylesheets.
pub fn style_tree(
    tree: &DomTree,
    stylesheets: &[Stylesheet],
    inline_styles: &std::collections::HashMap<usize, Vec<Declaration>>,
) -> Vec<StyledNode> {
    // Find root nodes (nodes with no parent)
    let roots: Vec<usize> = tree.nodes.iter()
        .filter(|n| n.parent.is_none())
        .map(|n| n.id)
        .collect();

    // Apply user-agent default styles first
    let ua_defaults = user_agent_defaults();
    let mut all_sheets: Vec<&Stylesheet> = vec![&ua_defaults];
    all_sheets.extend(stylesheets.iter());

    roots.into_iter()
        .map(|id| style_node(id, tree, &all_sheets, inline_styles, None))
        .collect()
}

/// Recursively compute style for a node and its children.
fn style_node(
    node_id: usize,
    tree: &DomTree,
    stylesheets: &[&Stylesheet],
    inline_styles: &std::collections::HashMap<usize, Vec<Declaration>>,
    parent_style: Option<&ComputedStyle>,
) -> StyledNode {
    let node = &tree.nodes[node_id];

    // Start with inherited properties from parent (or defaults)
    let mut style = if let Some(parent) = parent_style {
        inherit_from(parent)
    } else {
        ComputedStyle::default()
    };

    // Collect all matching declarations with their specificity
    let mut matched: Vec<(Specificity, &Declaration)> = Vec::new();

    for sheet in stylesheets {
        for rule in &sheet.rules {
            for selector in &rule.selectors {
                if matches_selector(selector, node, tree) {
                    for decl in &rule.declarations {
                        matched.push((selector.specificity, decl));
                    }
                }
            }
        }
    }

    // Sort by specificity (lower specificity first, so higher ones override)
    matched.sort_by(|a, b| a.0.cmp(&b.0));

    // Apply !important declarations after normal ones
    let (normal, important): (Vec<_>, Vec<_>) =
        matched.into_iter().partition(|(_, d)| !d.important);

    for (_, decl) in normal {
        apply_declaration(&mut style, decl);
    }

    // Inline styles (higher than any selector specificity, except !important)
    if let Some(inline) = inline_styles.get(&node_id) {
        let (inline_normal, inline_important): (Vec<_>, Vec<_>) =
            inline.iter().partition(|d| !d.important);
        for decl in inline_normal {
            apply_declaration(&mut style, decl);
        }
        for decl in inline_important {
            apply_declaration(&mut style, decl);
        }
    }

    // !important from stylesheets override everything except inline !important
    for (_, decl) in important {
        apply_declaration(&mut style, decl);
    }

    // Recurse into children
    let children = node.children.iter()
        .map(|&child_id| style_node(child_id, tree, stylesheets, inline_styles, Some(&style)))
        .collect();

    StyledNode {
        node_id,
        style,
        children,
    }
}

/// Create a new style inheriting inheritable properties from parent.
fn inherit_from(parent: &ComputedStyle) -> ComputedStyle {
    let mut style = ComputedStyle::default();

    // CSS inheritable properties
    style.color = parent.color;
    style.font_size = parent.font_size;
    style.font_weight = parent.font_weight;
    style.font_style = parent.font_style;
    style.font_family = parent.font_family.clone();
    style.line_height = parent.line_height;
    style.text_align = parent.text_align;
    style.text_decoration = parent.text_decoration;
    style.white_space = parent.white_space;
    style.letter_spacing = parent.letter_spacing;
    style.word_spacing = parent.word_spacing;
    style.text_indent = parent.text_indent;
    style.text_transform = parent.text_transform.clone();
    style.visibility = parent.visibility;
    style.cursor = parent.cursor.clone();
    style.list_style_type = parent.list_style_type.clone();

    style
}

/// Apply a single declaration to a computed style.
fn apply_declaration(style: &mut ComputedStyle, decl: &Declaration) {
    let prop = decl.property.as_str();
    let val = &decl.value;

    match prop {
        // ── Display / Position ──
        "display" => if let Some(k) = val.as_keyword() { style.display = CssDisplay::from_str(k); },
        "position" => if let Some(k) = val.as_keyword() { style.position = CssPosition::from_str(k); },
        "float" => if let Some(k) = val.as_keyword() { style.float = CssFloat::from_str(k); },

        // ── Dimensions ──
        "width" => if let Some(l) = val.as_length() { style.width = l; },
        "height" => if let Some(l) = val.as_length() { style.height = l; },
        "min-width" => if let Some(l) = val.as_length() { style.min_width = l; },
        "min-height" => if let Some(l) = val.as_length() { style.min_height = l; },
        "max-width" => if let Some(l) = val.as_length() { style.max_width = l; },
        "max-height" => if let Some(l) = val.as_length() { style.max_height = l; },

        // ── Margin ──
        "margin-top" => if let Some(l) = val.as_length() { style.margin_top = l; },
        "margin-right" => if let Some(l) = val.as_length() { style.margin_right = l; },
        "margin-bottom" => if let Some(l) = val.as_length() { style.margin_bottom = l; },
        "margin-left" => if let Some(l) = val.as_length() { style.margin_left = l; },

        // ── Padding ──
        "padding-top" => if let Some(l) = val.as_length() { style.padding_top = l; },
        "padding-right" => if let Some(l) = val.as_length() { style.padding_right = l; },
        "padding-bottom" => if let Some(l) = val.as_length() { style.padding_bottom = l; },
        "padding-left" => if let Some(l) = val.as_length() { style.padding_left = l; },

        // ── Border ──
        "border-width" => if let Some(l) = val.as_length() { style.border_width = l; },
        "border-style" => if let Some(k) = val.as_keyword() { style.border_style = CssBorderStyle::from_str(k); },
        "border-color" => if let Some(c) = val.as_color() { style.border_color = c; },
        "border-radius" | "border-top-left-radius" | "border-top-right-radius"
        | "border-bottom-right-radius" | "border-bottom-left-radius" => {
            if let Some(l) = val.as_length() { style.border_radius = l; }
        },

        // ── Colors ──
        "color" => if let Some(c) = val.as_color() { style.color = c; },
        "background-color" => if let Some(c) = val.as_color() { style.background_color = c; },

        // ── Typography ──
        "font-size" => {
            match val {
                CssValue::Length(l) => style.font_size = *l,
                CssValue::Percentage(p) => {
                    let parent_px = style.font_size.to_px(16.0, 0.0, 0.0);
                    style.font_size = CssLength::Px(parent_px * p / 100.0);
                }
                CssValue::Keyword(k) => {
                    style.font_size = match k.as_str() {
                        "xx-small" => CssLength::Px(9.0),
                        "x-small" => CssLength::Px(10.0),
                        "small" => CssLength::Px(13.0),
                        "medium" => CssLength::Px(16.0),
                        "large" => CssLength::Px(18.0),
                        "x-large" => CssLength::Px(24.0),
                        "xx-large" => CssLength::Px(32.0),
                        "smaller" => {
                            let cur = style.font_size.to_px(16.0, 0.0, 0.0);
                            CssLength::Px(cur * 0.833)
                        }
                        "larger" => {
                            let cur = style.font_size.to_px(16.0, 0.0, 0.0);
                            CssLength::Px(cur * 1.2)
                        }
                        _ => CssLength::Px(16.0),
                    };
                }
                _ => {}
            }
        }
        "font-weight" => {
            match val {
                CssValue::Keyword(k) => style.font_weight = CssFontWeight::from_str(k),
                CssValue::Number(n) => style.font_weight = CssFontWeight(*n as u16),
                _ => {}
            }
        }
        "font-style" => if let Some(k) = val.as_keyword() { style.font_style = CssFontStyle::from_str(k); },
        "font-family" => {
            match val {
                CssValue::Keyword(k) => style.font_family = k.clone(),
                CssValue::String(s) => style.font_family = s.clone(),
                CssValue::List(parts) => {
                    style.font_family = parts.iter().filter_map(|p| {
                        match p {
                            CssValue::Keyword(k) => Some(k.clone()),
                            CssValue::String(s) => Some(s.clone()),
                            _ => None,
                        }
                    }).next().unwrap_or_default();
                }
                _ => {}
            }
        }
        "line-height" => {
            match val {
                CssValue::Length(l) => style.line_height = *l,
                CssValue::Number(n) => style.line_height = CssLength::Em(*n),
                CssValue::Percentage(p) => style.line_height = CssLength::Percent(*p),
                CssValue::Keyword(k) if k == "normal" => style.line_height = CssLength::Em(1.2),
                _ => {}
            }
        }
        "text-align" => if let Some(k) = val.as_keyword() { style.text_align = CssTextAlign::from_str(k); },
        "text-decoration" => if let Some(k) = val.as_keyword() { style.text_decoration = CssTextDecoration::from_str(k); },
        "white-space" => if let Some(k) = val.as_keyword() { style.white_space = CssWhiteSpace::from_str(k); },
        "vertical-align" => if let Some(k) = val.as_keyword() { style.vertical_align = CssVerticalAlign::from_str(k); },
        "letter-spacing" => if let Some(l) = val.as_length() { style.letter_spacing = l; },
        "word-spacing" => if let Some(l) = val.as_length() { style.word_spacing = l; },
        "text-indent" => if let Some(l) = val.as_length() { style.text_indent = l; },
        "text-transform" => if let Some(k) = val.as_keyword() { style.text_transform = k.to_string(); },

        // ── Visual ──
        "opacity" => if let Some(n) = val.as_number() { style.opacity = n.clamp(0.0, 1.0); },
        "overflow" => if let Some(k) = val.as_keyword() { style.overflow = CssOverflow::from_str(k); },
        "visibility" => if let Some(k) = val.as_keyword() { style.visibility = k != "hidden"; },
        "cursor" => if let Some(k) = val.as_keyword() { style.cursor = k.to_string(); },
        "z-index" => if let Some(n) = val.as_number() { style.z_index = Some(n as i32); },

        // ── Flex ──
        "flex-direction" => if let Some(k) = val.as_keyword() { style.flex_direction = CssFlexDirection::from_str(k); },
        "flex-grow" => if let Some(n) = val.as_number() { style.flex_grow = n; },
        "flex-shrink" => if let Some(n) = val.as_number() { style.flex_shrink = n; },
        "flex-basis" => if let Some(l) = val.as_length() { style.flex_basis = l; },
        "justify-content" => if let Some(k) = val.as_keyword() { style.justify_content = CssAlign::from_str(k); },
        "align-items" => if let Some(k) = val.as_keyword() { style.align_items = CssAlign::from_str(k); },
        "flex-wrap" => if let Some(k) = val.as_keyword() { style.flex_wrap = k.to_string(); },
        "gap" => if let Some(l) = val.as_length() { style.gap = l; },

        // ── Background ──
        "background-image" => {
            match val {
                CssValue::Url(u) => style.background_image = u.clone(),
                CssValue::Keyword(k) if k == "none" => style.background_image.clear(),
                _ => {}
            }
        }
        "background-repeat" => if let Some(k) = val.as_keyword() { style.background_repeat = k.to_string(); },
        "background-position" => if let Some(k) = val.as_keyword() { style.background_position = k.to_string(); },
        "background-size" => if let Some(k) = val.as_keyword() { style.background_size = k.to_string(); },

        // ── List ──
        "list-style-type" | "list-style" => if let Some(k) = val.as_keyword() { style.list_style_type = k.to_string(); },

        // ── Ignored / unknown ──
        _ => {}
    }
}

/// User-agent default stylesheet (simplified).
/// Maps HTML elements to their default display and styling.
fn user_agent_defaults() -> Stylesheet {
    use super::parser::parse_stylesheet;

    parse_stylesheet(r#"
        html, body, div, article, section, main, nav, aside, header, footer,
        h1, h2, h3, h4, h5, h6, p, ul, ol, li, dl, dt, dd,
        pre, blockquote, figure, figcaption, form, fieldset, table,
        thead, tbody, tfoot, tr, address, details, summary { display: block; }

        span, a, strong, b, em, i, u, s, del, ins, mark, small, sub, sup,
        abbr, cite, code, kbd, samp, var, time, q, label, output { display: inline; }

        h1 { font-size: 2em; font-weight: bold; margin-top: 0.67em; margin-bottom: 0.67em; }
        h2 { font-size: 1.5em; font-weight: bold; margin-top: 0.83em; margin-bottom: 0.83em; }
        h3 { font-size: 1.17em; font-weight: bold; margin-top: 1em; margin-bottom: 1em; }
        h4 { font-size: 1em; font-weight: bold; margin-top: 1.33em; margin-bottom: 1.33em; }
        h5 { font-size: 0.83em; font-weight: bold; margin-top: 1.67em; margin-bottom: 1.67em; }
        h6 { font-size: 0.67em; font-weight: bold; margin-top: 2.33em; margin-bottom: 2.33em; }

        p { margin-top: 1em; margin-bottom: 1em; }

        strong, b { font-weight: bold; }
        em, i { font-style: italic; }
        u, ins { text-decoration: underline; }
        s, del { text-decoration: line-through; }
        small { font-size: 0.83em; }
        sub { font-size: 0.83em; vertical-align: sub; }
        sup { font-size: 0.83em; vertical-align: super; }

        a { color: blue; text-decoration: underline; }

        ul, ol { margin-top: 1em; margin-bottom: 1em; padding-left: 40px; }
        ul { list-style-type: disc; }
        ol { list-style-type: decimal; }
        li { display: list-item; }

        pre, code, kbd, samp { font-family: monospace; }
        pre { white-space: pre; margin-top: 1em; margin-bottom: 1em; }
        code { font-size: 0.9em; }

        blockquote { margin-top: 1em; margin-bottom: 1em; margin-left: 40px; margin-right: 40px; }

        table { display: table; border-collapse: separate; }
        tr { display: table-row; }
        td, th { display: table-cell; padding: 1px; }
        th { font-weight: bold; text-align: center; }
        thead { display: table-header-group; }
        tbody { display: table-row-group; }
        tfoot { display: table-footer-group; }

        img { display: inline; }
        br { display: block; }
        hr { display: block; margin-top: 0.5em; margin-bottom: 0.5em; border-style: solid; border-width: 1px; }

        mark { background-color: yellow; }

        input, button, select, textarea { display: inline-block; }
    "#)
}

/// Extract CSS text from <style> tags in the DOM.
pub fn extract_style_tags(tree: &DomTree) -> Vec<String> {
    let mut css_texts = Vec::new();
    for node in &tree.nodes {
        if node.tag == "style" && !node.text.is_empty() {
            css_texts.push(node.text.clone());
        }
    }
    css_texts
}

/// Extract inline styles from DOM nodes (style="..." attributes).
pub fn extract_inline_styles(tree: &DomTree) -> std::collections::HashMap<usize, Vec<Declaration>> {
    use super::parser::parse_inline_style;
    let mut map = std::collections::HashMap::new();
    for node in &tree.nodes {
        if let Some(style_attr) = node.attrs.get("style") {
            let decls = parse_inline_style(style_attr);
            if !decls.is_empty() {
                map.insert(node.id, decls);
            }
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::dom::parse_html;
    use crate::css::parser::parse_stylesheet;

    #[test]
    fn test_basic_style_tree() {
        let html = "<html><body><p>Hello</p></body></html>";
        let tree = parse_html(html);
        let ss = parse_stylesheet("p { color: red; font-size: 20px; }");
        let inline = extract_inline_styles(&tree);
        let styled = style_tree(&tree, &[ss], &inline);
        assert!(!styled.is_empty());
    }

    #[test]
    fn test_color_applied() {
        let html = "<html><body><p>Hello</p></body></html>";
        let tree = parse_html(html);
        let ss = parse_stylesheet("p { color: red; }");
        let inline = extract_inline_styles(&tree);
        let styled = style_tree(&tree, &[ss], &inline);

        // Find the styled p node
        fn find_styled(nodes: &[StyledNode], tag: &str, tree: &DomTree) -> Option<ComputedStyle> {
            for sn in nodes {
                if tree.nodes[sn.node_id].tag == tag {
                    return Some(sn.style.clone());
                }
                if let Some(found) = find_styled(&sn.children, tag, tree) {
                    return Some(found);
                }
            }
            None
        }

        let p_style = find_styled(&styled, "p", &tree).unwrap();
        assert_eq!(p_style.color, CssColor::rgb(255, 0, 0));
    }

    #[test]
    fn test_inheritance() {
        let html = "<html><body><div><p>Hello</p></div></body></html>";
        let tree = parse_html(html);
        let ss = parse_stylesheet("div { color: blue; font-size: 20px; }");
        let inline = extract_inline_styles(&tree);
        let styled = style_tree(&tree, &[ss], &inline);

        fn find_styled(nodes: &[StyledNode], tag: &str, tree: &DomTree) -> Option<ComputedStyle> {
            for sn in nodes {
                if tree.nodes[sn.node_id].tag == tag {
                    return Some(sn.style.clone());
                }
                if let Some(found) = find_styled(&sn.children, tag, tree) {
                    return Some(found);
                }
            }
            None
        }

        let p_style = find_styled(&styled, "p", &tree).unwrap();
        // p should inherit color from div
        assert_eq!(p_style.color, CssColor::rgb(0, 0, 255));
    }

    #[test]
    fn test_specificity_override() {
        let html = r#"<html><body><p class="intro" id="first">Hello</p></body></html>"#;
        let tree = parse_html(html);
        let ss = parse_stylesheet(r#"
            p { color: red; }
            .intro { color: green; }
            #first { color: blue; }
        "#);
        let inline = extract_inline_styles(&tree);
        let styled = style_tree(&tree, &[ss], &inline);

        fn find_styled(nodes: &[StyledNode], tag: &str, tree: &DomTree) -> Option<ComputedStyle> {
            for sn in nodes {
                if tree.nodes[sn.node_id].tag == tag {
                    return Some(sn.style.clone());
                }
                if let Some(found) = find_styled(&sn.children, tag, tree) {
                    return Some(found);
                }
            }
            None
        }

        let p_style = find_styled(&styled, "p", &tree).unwrap();
        // #first (ID) has highest specificity → blue
        assert_eq!(p_style.color, CssColor::rgb(0, 0, 255));
    }

    #[test]
    fn test_inline_style() {
        let html = r#"<html><body><p style="color: green;">Hello</p></body></html>"#;
        let tree = parse_html(html);
        let ss = parse_stylesheet("p { color: red; }");
        let inline = extract_inline_styles(&tree);
        let styled = style_tree(&tree, &[ss], &inline);

        fn find_styled(nodes: &[StyledNode], tag: &str, tree: &DomTree) -> Option<ComputedStyle> {
            for sn in nodes {
                if tree.nodes[sn.node_id].tag == tag {
                    return Some(sn.style.clone());
                }
                if let Some(found) = find_styled(&sn.children, tag, tree) {
                    return Some(found);
                }
            }
            None
        }

        let p_style = find_styled(&styled, "p", &tree).unwrap();
        // Inline style overrides stylesheet
        assert_eq!(p_style.color, CssColor::rgb(0, 128, 0));
    }

    #[test]
    fn test_important_override() {
        let html = r#"<html><body><p style="color: green;">Hello</p></body></html>"#;
        let tree = parse_html(html);
        let ss = parse_stylesheet("p { color: red !important; }");
        let inline = extract_inline_styles(&tree);
        let styled = style_tree(&tree, &[ss], &inline);

        fn find_styled(nodes: &[StyledNode], tag: &str, tree: &DomTree) -> Option<ComputedStyle> {
            for sn in nodes {
                if tree.nodes[sn.node_id].tag == tag {
                    return Some(sn.style.clone());
                }
                if let Some(found) = find_styled(&sn.children, tag, tree) {
                    return Some(found);
                }
            }
            None
        }

        let p_style = find_styled(&styled, "p", &tree).unwrap();
        // !important overrides inline style
        assert_eq!(p_style.color, CssColor::rgb(255, 0, 0));
    }

    #[test]
    fn test_user_agent_defaults() {
        let html = "<html><body><h1>Title</h1><p>Text</p></body></html>";
        let tree = parse_html(html);
        let inline = extract_inline_styles(&tree);
        let styled = style_tree(&tree, &[], &inline);

        fn find_styled(nodes: &[StyledNode], tag: &str, tree: &DomTree) -> Option<ComputedStyle> {
            for sn in nodes {
                if tree.nodes[sn.node_id].tag == tag {
                    return Some(sn.style.clone());
                }
                if let Some(found) = find_styled(&sn.children, tag, tree) {
                    return Some(found);
                }
            }
            None
        }

        let h1_style = find_styled(&styled, "h1", &tree).unwrap();
        assert_eq!(h1_style.display, CssDisplay::Block);
        assert!(h1_style.font_weight.is_bold());
    }

    #[test]
    fn test_display_none() {
        let html = "<html><body><p>Visible</p><div>Hidden</div></body></html>";
        let tree = parse_html(html);
        let ss = parse_stylesheet("div { display: none; }");
        let inline = extract_inline_styles(&tree);
        let styled = style_tree(&tree, &[ss], &inline);

        fn find_styled(nodes: &[StyledNode], tag: &str, tree: &DomTree) -> Option<ComputedStyle> {
            for sn in nodes {
                if tree.nodes[sn.node_id].tag == tag {
                    return Some(sn.style.clone());
                }
                if let Some(found) = find_styled(&sn.children, tag, tree) {
                    return Some(found);
                }
            }
            None
        }

        let div_style = find_styled(&styled, "div", &tree).unwrap();
        assert_eq!(div_style.display, CssDisplay::None);
    }
}
