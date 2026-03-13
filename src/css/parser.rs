//! CSS Parser — parses tokenized CSS into a Stylesheet (rules, selectors, declarations).
//!
//! Handles:
//! - Simple selectors: tag, .class, #id, *, [attr], :pseudo-class, ::pseudo-element
//! - Combinators: descendant (space), child (>), sibling (+), general sibling (~)
//! - Selector lists (comma-separated)
//! - Declarations with !important
//! - @media queries (basic)
//! - @import (recorded but not fetched)
//! - Shorthand properties (margin, padding, border, background, font)

use super::tokenizer::{Token, Tokenizer, HashType};
use super::values::*;

/// A complete CSS stylesheet.
#[derive(Debug, Clone)]
pub struct Stylesheet {
    pub rules: Vec<Rule>,
    pub imports: Vec<String>,
}

/// A CSS rule: selector list + declarations.
#[derive(Debug, Clone)]
pub struct Rule {
    pub selectors: Vec<Selector>,
    pub declarations: Vec<Declaration>,
    /// Media query condition (None = unconditional)
    pub media: Option<String>,
}

/// A CSS selector (chain of simple selectors with combinators).
#[derive(Debug, Clone)]
pub struct Selector {
    pub parts: Vec<SelectorPart>,
    pub specificity: Specificity,
}

/// A part of a compound selector.
#[derive(Debug, Clone)]
pub enum SelectorPart {
    /// Universal selector `*`
    Universal,
    /// Type/tag selector: `div`, `p`, `h1`
    Tag(String),
    /// Class selector: `.foo`
    Class(String),
    /// ID selector: `#bar`
    Id(String),
    /// Attribute selector: `[attr]`, `[attr=val]`, `[attr~=val]`, `[attr|=val]`,
    /// `[attr^=val]`, `[attr$=val]`, `[attr*=val]`
    Attribute {
        name: String,
        op: Option<AttrOp>,
        value: Option<String>,
    },
    /// Pseudo-class: `:hover`, `:first-child`, `:nth-child(n)`, `:not(sel)`
    PseudoClass(String, Option<String>),
    /// Pseudo-element: `::before`, `::after`, `::first-line`
    PseudoElement(String),
    /// Combinator between compound selectors
    Combinator(Combinator),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Combinator {
    Descendant,   // space
    Child,        // >
    NextSibling,  // +
    Subsequent,   // ~
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttrOp {
    Equals,        // =
    Includes,      // ~=
    DashMatch,     // |=
    Prefix,        // ^=
    Suffix,        // $=
    Substring,     // *=
}

/// CSS specificity (a, b, c) — IDs, classes+attrs+pseudo, elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Specificity(pub u32, pub u32, pub u32);

impl Specificity {
    pub const ZERO: Self = Self(0, 0, 0);
    pub const INLINE: Self = Self(1000, 0, 0);

    pub fn value(&self) -> u32 {
        self.0 * 10000 + self.1 * 100 + self.2
    }
}

/// A CSS property declaration.
#[derive(Debug, Clone)]
pub struct Declaration {
    pub property: String,
    pub value: CssValue,
    pub important: bool,
}

/// A generic CSS value (before resolution to typed values).
#[derive(Debug, Clone)]
pub enum CssValue {
    Keyword(String),
    Color(CssColor),
    Length(CssLength),
    Number(f32),
    Percentage(f32),
    Url(String),
    String(String),
    /// Multiple values (e.g., `margin: 10px 20px`)
    List(Vec<CssValue>),
    /// Function call result (e.g., `calc()`)
    Function(String, Vec<CssValue>),
    /// Initial/inherit/unset
    Initial,
    Inherit,
    Unset,
}

impl CssValue {
    /// Try to interpret as a color.
    pub fn as_color(&self) -> Option<CssColor> {
        match self {
            CssValue::Color(c) => Some(*c),
            CssValue::Keyword(k) => CssColor::from_name(k),
            _ => None,
        }
    }

    /// Try to interpret as a length.
    pub fn as_length(&self) -> Option<CssLength> {
        match self {
            CssValue::Length(l) => Some(*l),
            CssValue::Number(0.0) => Some(CssLength::Zero),
            CssValue::Percentage(p) => Some(CssLength::Percent(*p)),
            CssValue::Keyword(k) if k == "auto" => Some(CssLength::Auto),
            CssValue::Keyword(k) if k == "0" => Some(CssLength::Zero),
            _ => None,
        }
    }

    /// Try to interpret as a keyword string.
    pub fn as_keyword(&self) -> Option<&str> {
        match self {
            CssValue::Keyword(k) => Some(k),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<f32> {
        match self {
            CssValue::Number(n) => Some(*n),
            CssValue::Length(CssLength::Px(n)) => Some(*n),
            _ => None,
        }
    }
}

// ── Parser ──

pub struct CssParser {
    tokens: Vec<Token>,
    pos: usize,
}

impl CssParser {
    pub fn new(css: &str) -> Self {
        let tokens = Tokenizer::new(css).tokenize_all();
        Self { tokens, pos: 0 }
    }

    /// Parse a complete stylesheet.
    pub fn parse(mut self) -> Stylesheet {
        let mut rules = Vec::new();
        let mut imports = Vec::new();
        let mut current_media: Option<String> = None;

        while self.pos < self.tokens.len() {
            self.skip_whitespace();
            if self.pos >= self.tokens.len() {
                break;
            }

            match &self.tokens[self.pos] {
                Token::AtKeyword(kw) => {
                    let kw = kw.clone();
                    self.pos += 1;
                    match kw.as_str() {
                        "import" => {
                            if let Some(url) = self.parse_import() {
                                imports.push(url);
                            }
                        }
                        "media" => {
                            current_media = Some(self.parse_media_query());
                        }
                        "charset" | "namespace" | "font-face" | "keyframes" | "supports" | "page" => {
                            // Skip these at-rules by consuming until ; or { }
                            self.skip_at_rule();
                        }
                        _ => {
                            self.skip_at_rule();
                        }
                    }
                }
                Token::BraceClose => {
                    // End of @media block
                    self.pos += 1;
                    current_media = None;
                }
                _ => {
                    if let Some(rule) = self.parse_rule(current_media.clone()) {
                        rules.push(rule);
                    }
                }
            }
        }

        Stylesheet { rules, imports }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.tokens.len() && self.tokens[self.pos] == Token::Whitespace {
            self.pos += 1;
        }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn consume(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &Token) -> bool {
        self.skip_whitespace();
        if self.peek() == expected {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_import(&mut self) -> Option<String> {
        self.skip_whitespace();
        let url = match self.peek() {
            Token::StringTok(s) => {
                let s = s.clone();
                self.pos += 1;
                s
            }
            Token::Url(u) => {
                let u = u.clone();
                self.pos += 1;
                u
            }
            _ => {
                self.skip_until_semicolon();
                return None;
            }
        };
        self.skip_until_semicolon();
        Some(url)
    }

    fn parse_media_query(&mut self) -> String {
        let mut query = String::new();
        self.skip_whitespace();
        // Collect tokens until {
        while self.pos < self.tokens.len() {
            match &self.tokens[self.pos] {
                Token::BraceOpen => {
                    self.pos += 1;
                    break;
                }
                Token::Whitespace => {
                    query.push(' ');
                    self.pos += 1;
                }
                Token::Ident(s) => {
                    query.push_str(s);
                    self.pos += 1;
                }
                Token::ParenOpen => {
                    query.push('(');
                    self.pos += 1;
                }
                Token::ParenClose => {
                    query.push(')');
                    self.pos += 1;
                }
                Token::Colon => {
                    query.push(':');
                    self.pos += 1;
                }
                Token::Dimension(v, u) => {
                    query.push_str(&format!("{v}{u}"));
                    self.pos += 1;
                }
                _ => {
                    self.pos += 1;
                }
            }
        }
        query.trim().to_string()
    }

    fn skip_at_rule(&mut self) {
        let mut depth = 0;
        while self.pos < self.tokens.len() {
            match &self.tokens[self.pos] {
                Token::Semicolon if depth == 0 => {
                    self.pos += 1;
                    return;
                }
                Token::BraceOpen => {
                    depth += 1;
                    self.pos += 1;
                }
                Token::BraceClose => {
                    depth -= 1;
                    self.pos += 1;
                    if depth <= 0 {
                        return;
                    }
                }
                _ => {
                    self.pos += 1;
                }
            }
        }
    }

    fn skip_until_semicolon(&mut self) {
        while self.pos < self.tokens.len() {
            if self.tokens[self.pos] == Token::Semicolon {
                self.pos += 1;
                return;
            }
            self.pos += 1;
        }
    }

    fn parse_rule(&mut self, media: Option<String>) -> Option<Rule> {
        let selectors = self.parse_selector_list();
        if selectors.is_empty() {
            // Skip to end of block or next rule
            self.skip_at_rule();
            return None;
        }

        self.skip_whitespace();
        if !self.expect(&Token::BraceOpen) {
            return None;
        }

        let declarations = self.parse_declarations();

        self.skip_whitespace();
        let _ = self.expect(&Token::BraceClose);

        Some(Rule {
            selectors,
            declarations,
            media,
        })
    }

    /// Parse a comma-separated list of selectors.
    fn parse_selector_list(&mut self) -> Vec<Selector> {
        let mut selectors = Vec::new();
        loop {
            self.skip_whitespace();
            if let Some(sel) = self.parse_selector() {
                selectors.push(sel);
            }
            self.skip_whitespace();
            if self.peek() == &Token::Comma {
                self.pos += 1;
            } else {
                break;
            }
        }
        selectors
    }

    /// Parse a single selector (compound selectors with combinators).
    fn parse_selector(&mut self) -> Option<Selector> {
        let mut parts = Vec::new();
        let mut specificity = Specificity::ZERO;
        let mut last_was_combinator = true;

        loop {
            self.skip_whitespace();
            if self.pos >= self.tokens.len() {
                break;
            }

            match self.peek() {
                Token::BraceOpen | Token::Comma | Token::Eof => break,
                Token::Greater => {
                    self.pos += 1;
                    parts.push(SelectorPart::Combinator(Combinator::Child));
                    last_was_combinator = true;
                }
                Token::Plus => {
                    self.pos += 1;
                    parts.push(SelectorPart::Combinator(Combinator::NextSibling));
                    last_was_combinator = true;
                }
                Token::Tilde => {
                    self.pos += 1;
                    parts.push(SelectorPart::Combinator(Combinator::Subsequent));
                    last_was_combinator = true;
                }
                Token::Ident(_) | Token::Hash(_, _) | Token::Dot | Token::Asterisk
                | Token::Colon | Token::BracketOpen => {
                    // If we have a compound selector following another without
                    // an explicit combinator, insert a descendant combinator
                    if !last_was_combinator && !parts.is_empty() {
                        parts.push(SelectorPart::Combinator(Combinator::Descendant));
                    }
                    self.parse_compound_selector(&mut parts, &mut specificity);
                    last_was_combinator = false;
                }
                Token::Whitespace => {
                    self.pos += 1;
                    // Whitespace might be a descendant combinator or just spacing
                    // We'll handle it when we see the next token
                    continue;
                }
                _ => break,
            }
        }

        if parts.is_empty() {
            return None;
        }

        Some(Selector { parts, specificity })
    }

    /// Parse a compound selector (no combinators).
    fn parse_compound_selector(&mut self, parts: &mut Vec<SelectorPart>, spec: &mut Specificity) {
        loop {
            if self.pos >= self.tokens.len() {
                break;
            }
            match self.peek() {
                Token::Ident(tag) => {
                    let tag = tag.clone();
                    self.pos += 1;
                    parts.push(SelectorPart::Tag(tag));
                    spec.2 += 1;
                }
                Token::Hash(name, HashType::Id) => {
                    let name = name.clone();
                    self.pos += 1;
                    parts.push(SelectorPart::Id(name));
                    spec.0 += 1;
                }
                Token::Hash(name, HashType::Unrestricted) => {
                    // Treat unrestricted hash as class
                    let name = name.clone();
                    self.pos += 1;
                    parts.push(SelectorPart::Class(name));
                    spec.1 += 1;
                }
                Token::Dot => {
                    self.pos += 1;
                    if let Token::Ident(class) = self.peek() {
                        let class = class.clone();
                        self.pos += 1;
                        parts.push(SelectorPart::Class(class));
                        spec.1 += 1;
                    }
                }
                Token::Asterisk => {
                    self.pos += 1;
                    parts.push(SelectorPart::Universal);
                    // Universal doesn't add specificity
                }
                Token::Colon => {
                    self.pos += 1;
                    // :: pseudo-element
                    if self.peek() == &Token::Colon {
                        self.pos += 1;
                        if let Token::Ident(name) = self.peek() {
                            let name = name.clone();
                            self.pos += 1;
                            parts.push(SelectorPart::PseudoElement(name));
                            spec.2 += 1;
                        }
                    } else if let Token::Ident(name) = self.peek() {
                        let name = name.clone();
                        self.pos += 1;
                        // Check for functional pseudo-class: :nth-child(...)
                        let args = if self.peek() == &Token::ParenOpen
                            || matches!(self.peek(), Token::Function(_))
                        {
                            Some(self.consume_parens())
                        } else {
                            None
                        };
                        parts.push(SelectorPart::PseudoClass(name, args));
                        spec.1 += 1;
                    } else if let Token::Function(name) = self.peek() {
                        let name = name.clone();
                        self.pos += 1;
                        let args = self.consume_until_close_paren();
                        parts.push(SelectorPart::PseudoClass(name, Some(args)));
                        spec.1 += 1;
                    }
                }
                Token::BracketOpen => {
                    self.pos += 1;
                    self.parse_attribute_selector(parts, spec);
                }
                _ => break,
            }
        }
    }

    fn consume_parens(&mut self) -> String {
        if self.peek() == &Token::ParenOpen {
            self.pos += 1;
        }
        self.consume_until_close_paren()
    }

    fn consume_until_close_paren(&mut self) -> String {
        let mut result = String::new();
        let mut depth = 1;
        while self.pos < self.tokens.len() && depth > 0 {
            match &self.tokens[self.pos] {
                Token::ParenOpen => {
                    depth += 1;
                    result.push('(');
                }
                Token::ParenClose => {
                    depth -= 1;
                    if depth > 0 {
                        result.push(')');
                    }
                }
                Token::Ident(s) => result.push_str(s),
                Token::Number(n, _) => result.push_str(&n.to_string()),
                Token::Dimension(v, u) => result.push_str(&format!("{v}{u}")),
                Token::Whitespace => result.push(' '),
                Token::Comma => result.push(','),
                Token::Plus => result.push('+'),
                Token::Delim(c) => result.push(*c),
                _ => {}
            }
            self.pos += 1;
        }
        result.trim().to_string()
    }

    fn parse_attribute_selector(&mut self, parts: &mut Vec<SelectorPart>, spec: &mut Specificity) {
        self.skip_whitespace();
        let name = match self.peek() {
            Token::Ident(n) => {
                let n = n.clone();
                self.pos += 1;
                n
            }
            _ => {
                // Skip until ]
                while self.pos < self.tokens.len() && self.tokens[self.pos] != Token::BracketClose {
                    self.pos += 1;
                }
                if self.pos < self.tokens.len() { self.pos += 1; }
                return;
            }
        };

        self.skip_whitespace();

        // Check for operator
        let (op, value) = match self.peek() {
            Token::BracketClose => {
                self.pos += 1;
                (None, None)
            }
            Token::Equals => {
                self.pos += 1;
                let val = self.parse_attr_value();
                self.skip_whitespace();
                if self.peek() == &Token::BracketClose { self.pos += 1; }
                (Some(AttrOp::Equals), Some(val))
            }
            Token::Tilde => {
                self.pos += 1;
                if self.peek() == &Token::Equals { self.pos += 1; }
                let val = self.parse_attr_value();
                self.skip_whitespace();
                if self.peek() == &Token::BracketClose { self.pos += 1; }
                (Some(AttrOp::Includes), Some(val))
            }
            Token::Pipe => {
                self.pos += 1;
                if self.peek() == &Token::Equals { self.pos += 1; }
                let val = self.parse_attr_value();
                self.skip_whitespace();
                if self.peek() == &Token::BracketClose { self.pos += 1; }
                (Some(AttrOp::DashMatch), Some(val))
            }
            Token::Delim('^') => {
                self.pos += 1;
                if self.peek() == &Token::Equals { self.pos += 1; }
                let val = self.parse_attr_value();
                self.skip_whitespace();
                if self.peek() == &Token::BracketClose { self.pos += 1; }
                (Some(AttrOp::Prefix), Some(val))
            }
            Token::Delim('$') => {
                self.pos += 1;
                if self.peek() == &Token::Equals { self.pos += 1; }
                let val = self.parse_attr_value();
                self.skip_whitespace();
                if self.peek() == &Token::BracketClose { self.pos += 1; }
                (Some(AttrOp::Suffix), Some(val))
            }
            Token::Asterisk => {
                self.pos += 1;
                if self.peek() == &Token::Equals { self.pos += 1; }
                let val = self.parse_attr_value();
                self.skip_whitespace();
                if self.peek() == &Token::BracketClose { self.pos += 1; }
                (Some(AttrOp::Substring), Some(val))
            }
            _ => {
                // Skip until ]
                while self.pos < self.tokens.len() && self.tokens[self.pos] != Token::BracketClose {
                    self.pos += 1;
                }
                if self.pos < self.tokens.len() { self.pos += 1; }
                (None, None)
            }
        };

        parts.push(SelectorPart::Attribute { name, op, value });
        spec.1 += 1;
    }

    fn parse_attr_value(&mut self) -> String {
        self.skip_whitespace();
        match self.peek() {
            Token::StringTok(s) => {
                let s = s.clone();
                self.pos += 1;
                s
            }
            Token::Ident(s) => {
                let s = s.clone();
                self.pos += 1;
                s
            }
            _ => String::new(),
        }
    }

    /// Parse declarations inside a { } block.
    fn parse_declarations(&mut self) -> Vec<Declaration> {
        let mut decls = Vec::new();
        loop {
            self.skip_whitespace();
            if self.pos >= self.tokens.len() {
                break;
            }
            match self.peek() {
                Token::BraceClose | Token::Eof => break,
                Token::Semicolon => {
                    self.pos += 1;
                    continue;
                }
                _ => {}
            }

            if let Some(mut expanded) = self.parse_declaration() {
                decls.append(&mut expanded);
            }
        }
        decls
    }

    /// Parse a single declaration, returning expanded shorthands.
    fn parse_declaration(&mut self) -> Option<Vec<Declaration>> {
        self.skip_whitespace();
        let property = match self.peek() {
            Token::Ident(name) => {
                let name = name.clone();
                self.pos += 1;
                name
            }
            _ => {
                self.skip_until_semicolon_or_brace();
                return None;
            }
        };

        self.skip_whitespace();
        if !self.expect(&Token::Colon) {
            self.skip_until_semicolon_or_brace();
            return None;
        }

        let (values, important) = self.parse_value_list();

        // Expand shorthand properties
        let declarations = expand_shorthand(&property, values, important);
        Some(declarations)
    }

    /// Parse a value list until `;` or `}` or EOF.
    /// Also detects `!important`.
    fn parse_value_list(&mut self) -> (Vec<CssValue>, bool) {
        let mut values = Vec::new();
        let mut important = false;

        loop {
            self.skip_whitespace();
            if self.pos >= self.tokens.len() {
                break;
            }
            match self.peek() {
                Token::Semicolon => {
                    self.pos += 1;
                    break;
                }
                Token::BraceClose | Token::Eof => break,
                Token::Bang => {
                    self.pos += 1;
                    self.skip_whitespace();
                    if let Token::Ident(s) = self.peek() {
                        if s == "important" {
                            important = true;
                            self.pos += 1;
                        }
                    }
                }
                _ => {
                    if let Some(val) = self.parse_single_value() {
                        values.push(val);
                    } else {
                        self.pos += 1; // skip unrecognized token
                    }
                }
            }
        }

        (values, important)
    }

    /// Parse a single CSS value.
    fn parse_single_value(&mut self) -> Option<CssValue> {
        self.skip_whitespace();
        match self.peek() {
            Token::Ident(s) => {
                let s = s.clone();
                self.pos += 1;
                // Check for special keywords
                match s.as_str() {
                    "initial" => Some(CssValue::Initial),
                    "inherit" => Some(CssValue::Inherit),
                    "unset" => Some(CssValue::Unset),
                    _ => {
                        // Try as color name
                        if let Some(color) = CssColor::from_name(&s) {
                            Some(CssValue::Color(color))
                        } else {
                            Some(CssValue::Keyword(s))
                        }
                    }
                }
            }
            Token::Hash(hex, _) => {
                let hex = hex.clone();
                self.pos += 1;
                if let Some(color) = CssColor::from_hex(&hex) {
                    Some(CssValue::Color(color))
                } else {
                    Some(CssValue::Keyword(format!("#{hex}")))
                }
            }
            Token::Number(n, _) => {
                let n = *n;
                self.pos += 1;
                Some(CssValue::Number(n))
            }
            Token::Dimension(v, u) => {
                let v = *v;
                let u = u.clone();
                self.pos += 1;
                Some(CssValue::Length(CssLength::from_dimension(v, &u)))
            }
            Token::Percentage(p) => {
                let p = *p;
                self.pos += 1;
                Some(CssValue::Percentage(p))
            }
            Token::StringTok(s) => {
                let s = s.clone();
                self.pos += 1;
                Some(CssValue::String(s))
            }
            Token::Url(u) => {
                let u = u.clone();
                self.pos += 1;
                Some(CssValue::Url(u))
            }
            Token::Function(name) => {
                let name = name.clone();
                self.pos += 1;
                let args = self.parse_function_args();
                match name.as_str() {
                    "rgb" | "rgba" => {
                        let nums: Vec<f32> = args.iter().filter_map(|a| a.as_number()).collect();
                        if let Some(color) = CssColor::from_rgb_args(&nums) {
                            Some(CssValue::Color(color))
                        } else {
                            Some(CssValue::Function(name, args))
                        }
                    }
                    "hsl" | "hsla" => {
                        let nums: Vec<f32> = args.iter().filter_map(|a| {
                            match a {
                                CssValue::Number(n) => Some(*n),
                                CssValue::Percentage(p) => Some(*p),
                                CssValue::Length(CssLength::Px(n)) => Some(*n),
                                _ => None,
                            }
                        }).collect();
                        if let Some(color) = CssColor::from_hsl_args(&nums) {
                            Some(CssValue::Color(color))
                        } else {
                            Some(CssValue::Function(name, args))
                        }
                    }
                    _ => Some(CssValue::Function(name, args)),
                }
            }
            Token::Comma => {
                self.pos += 1;
                None // skip commas between values
            }
            Token::Delim('/') => {
                self.pos += 1;
                None // skip slash separators (e.g., font shorthand)
            }
            _ => None,
        }
    }

    fn parse_function_args(&mut self) -> Vec<CssValue> {
        let mut args = Vec::new();
        loop {
            self.skip_whitespace();
            match self.peek() {
                Token::ParenClose | Token::Eof => {
                    if self.peek() == &Token::ParenClose {
                        self.pos += 1;
                    }
                    break;
                }
                Token::Comma => {
                    self.pos += 1;
                    continue;
                }
                _ => {
                    if let Some(val) = self.parse_single_value() {
                        args.push(val);
                    } else {
                        self.pos += 1;
                    }
                }
            }
        }
        args
    }

    fn skip_until_semicolon_or_brace(&mut self) {
        while self.pos < self.tokens.len() {
            match &self.tokens[self.pos] {
                Token::Semicolon | Token::BraceClose => return,
                _ => self.pos += 1,
            }
        }
    }
}

/// Expand CSS shorthand properties into individual declarations.
fn expand_shorthand(property: &str, values: Vec<CssValue>, important: bool) -> Vec<Declaration> {
    match property {
        // margin: T R B L / T R B / T B / All
        "margin" => expand_trbl("margin", &values, important),
        "padding" => expand_trbl("padding", &values, important),

        // border: width style color
        "border" => expand_border(&values, important),

        // border-radius: single or multi value
        "border-radius" => {
            let val = if values.len() == 1 {
                values[0].clone()
            } else {
                CssValue::List(values)
            };
            vec![
                Declaration { property: "border-top-left-radius".into(), value: val.clone(), important },
                Declaration { property: "border-top-right-radius".into(), value: val.clone(), important },
                Declaration { property: "border-bottom-right-radius".into(), value: val.clone(), important },
                Declaration { property: "border-bottom-left-radius".into(), value: val, important },
            ]
        }

        // background: color url repeat position (simplified)
        "background" => expand_background(&values, important),

        // font: style weight size/line-height family
        "font" => expand_font(&values, important),

        // flex: grow shrink basis
        "flex" => expand_flex(&values, important),

        // text-decoration handled as compound
        "text-decoration" => {
            let combined: String = values.iter().filter_map(|v| v.as_keyword().map(String::from)).collect::<Vec<_>>().join(" ");
            vec![Declaration {
                property: "text-decoration".into(),
                value: CssValue::Keyword(combined),
                important,
            }]
        }

        // Not a shorthand — pass through
        _ => {
            let value = if values.len() == 1 {
                values.into_iter().next().unwrap()
            } else if values.is_empty() {
                CssValue::Keyword(String::new())
            } else {
                CssValue::List(values)
            };
            vec![Declaration { property: property.to_string(), value, important }]
        }
    }
}

fn expand_trbl(prefix: &str, values: &[CssValue], important: bool) -> Vec<Declaration> {
    let (top, right, bottom, left) = match values.len() {
        1 => (&values[0], &values[0], &values[0], &values[0]),
        2 => (&values[0], &values[1], &values[0], &values[1]),
        3 => (&values[0], &values[1], &values[2], &values[1]),
        4 => (&values[0], &values[1], &values[2], &values[3]),
        _ => return vec![],
    };
    vec![
        Declaration { property: format!("{prefix}-top"), value: top.clone(), important },
        Declaration { property: format!("{prefix}-right"), value: right.clone(), important },
        Declaration { property: format!("{prefix}-bottom"), value: bottom.clone(), important },
        Declaration { property: format!("{prefix}-left"), value: left.clone(), important },
    ]
}

fn expand_border(values: &[CssValue], important: bool) -> Vec<Declaration> {
    let mut width = CssValue::Length(CssLength::Px(1.0));
    let mut style = CssValue::Keyword("solid".into());
    let mut color = CssValue::Keyword("currentcolor".into());

    for val in values {
        match val {
            CssValue::Length(_) | CssValue::Number(_) => width = val.clone(),
            CssValue::Color(_) => color = val.clone(),
            CssValue::Keyword(k) => {
                match k.as_str() {
                    "none" | "solid" | "dashed" | "dotted" | "double"
                    | "groove" | "ridge" | "inset" | "outset" | "hidden" => {
                        style = val.clone();
                    }
                    "thin" => width = CssValue::Length(CssLength::Px(1.0)),
                    "medium" => width = CssValue::Length(CssLength::Px(3.0)),
                    "thick" => width = CssValue::Length(CssLength::Px(5.0)),
                    _ => {
                        if CssColor::from_name(k).is_some() {
                            color = val.clone();
                        }
                    }
                }
            }
            _ => {}
        }
    }

    vec![
        Declaration { property: "border-width".into(), value: width, important },
        Declaration { property: "border-style".into(), value: style, important },
        Declaration { property: "border-color".into(), value: color, important },
    ]
}

fn expand_background(values: &[CssValue], important: bool) -> Vec<Declaration> {
    let mut decls = Vec::new();
    for val in values {
        match val {
            CssValue::Color(_) => {
                decls.push(Declaration { property: "background-color".into(), value: val.clone(), important });
            }
            CssValue::Url(_) => {
                decls.push(Declaration { property: "background-image".into(), value: val.clone(), important });
            }
            CssValue::Keyword(k) => {
                match k.as_str() {
                    "repeat" | "no-repeat" | "repeat-x" | "repeat-y" => {
                        decls.push(Declaration { property: "background-repeat".into(), value: val.clone(), important });
                    }
                    "center" | "top" | "bottom" | "left" | "right" => {
                        decls.push(Declaration { property: "background-position".into(), value: val.clone(), important });
                    }
                    "cover" | "contain" => {
                        decls.push(Declaration { property: "background-size".into(), value: val.clone(), important });
                    }
                    "fixed" | "scroll" | "local" => {
                        decls.push(Declaration { property: "background-attachment".into(), value: val.clone(), important });
                    }
                    _ => {
                        if let Some(_color) = CssColor::from_name(k) {
                            decls.push(Declaration { property: "background-color".into(), value: val.clone(), important });
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if decls.is_empty() {
        decls.push(Declaration {
            property: "background-color".into(),
            value: CssValue::Color(CssColor::TRANSPARENT),
            important,
        });
    }
    decls
}

fn expand_font(values: &[CssValue], important: bool) -> Vec<Declaration> {
    let mut decls = Vec::new();
    let mut saw_size = false;

    for val in values {
        match val {
            CssValue::Length(_) | CssValue::Percentage(_) => {
                if !saw_size {
                    decls.push(Declaration { property: "font-size".into(), value: val.clone(), important });
                    saw_size = true;
                } else {
                    decls.push(Declaration { property: "line-height".into(), value: val.clone(), important });
                }
            }
            CssValue::Number(n) => {
                if saw_size {
                    decls.push(Declaration { property: "line-height".into(), value: CssValue::Number(*n), important });
                } else if (*n >= 100.0 && *n <= 900.0) && (*n % 100.0 == 0.0) {
                    decls.push(Declaration { property: "font-weight".into(), value: val.clone(), important });
                }
            }
            CssValue::Keyword(k) => {
                match k.as_str() {
                    "italic" | "oblique" => {
                        decls.push(Declaration { property: "font-style".into(), value: val.clone(), important });
                    }
                    "normal" => {} // ambiguous, skip
                    "bold" | "bolder" | "lighter" => {
                        decls.push(Declaration { property: "font-weight".into(), value: val.clone(), important });
                    }
                    "small-caps" => {
                        decls.push(Declaration { property: "font-variant".into(), value: val.clone(), important });
                    }
                    _ => {
                        // Assume it's a font-family
                        decls.push(Declaration { property: "font-family".into(), value: val.clone(), important });
                    }
                }
            }
            CssValue::String(s) => {
                decls.push(Declaration { property: "font-family".into(), value: CssValue::String(s.clone()), important });
            }
            _ => {}
        }
    }
    decls
}

fn expand_flex(values: &[CssValue], important: bool) -> Vec<Declaration> {
    match values.len() {
        1 => {
            vec![
                Declaration { property: "flex-grow".into(), value: values[0].clone(), important },
                Declaration { property: "flex-shrink".into(), value: CssValue::Number(1.0), important },
                Declaration { property: "flex-basis".into(), value: CssValue::Keyword("auto".into()), important },
            ]
        }
        2 => {
            vec![
                Declaration { property: "flex-grow".into(), value: values[0].clone(), important },
                Declaration { property: "flex-shrink".into(), value: values[1].clone(), important },
                Declaration { property: "flex-basis".into(), value: CssValue::Keyword("auto".into()), important },
            ]
        }
        3 => {
            vec![
                Declaration { property: "flex-grow".into(), value: values[0].clone(), important },
                Declaration { property: "flex-shrink".into(), value: values[1].clone(), important },
                Declaration { property: "flex-basis".into(), value: values[2].clone(), important },
            ]
        }
        _ => vec![],
    }
}

/// Parse inline CSS (style attribute): just declarations, no selector.
pub fn parse_inline_style(css: &str) -> Vec<Declaration> {
    let wrapped = format!("__inline__ {{ {css} }}");
    let stylesheet = CssParser::new(&wrapped).parse();
    stylesheet.rules.into_iter()
        .flat_map(|r| r.declarations)
        .collect()
}

/// Parse a `<style>` block or external stylesheet.
pub fn parse_stylesheet(css: &str) -> Stylesheet {
    CssParser::new(css).parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_rule() {
        let ss = parse_stylesheet("h1 { color: red; }");
        assert_eq!(ss.rules.len(), 1);
        assert_eq!(ss.rules[0].selectors.len(), 1);
        assert_eq!(ss.rules[0].declarations.len(), 1);
        assert_eq!(ss.rules[0].declarations[0].property, "color");
    }

    #[test]
    fn test_parse_class_selector() {
        let ss = parse_stylesheet(".foo { display: block; }");
        assert_eq!(ss.rules.len(), 1);
        let sel = &ss.rules[0].selectors[0];
        assert!(sel.parts.iter().any(|p| matches!(p, SelectorPart::Class(c) if c == "foo")));
        assert_eq!(sel.specificity, Specificity(0, 1, 0));
    }

    #[test]
    fn test_parse_id_selector() {
        let ss = parse_stylesheet("#bar { margin: 10px; }");
        assert_eq!(ss.rules.len(), 1);
        let sel = &ss.rules[0].selectors[0];
        assert!(sel.parts.iter().any(|p| matches!(p, SelectorPart::Id(id) if id == "bar")));
        assert_eq!(sel.specificity, Specificity(1, 0, 0));
    }

    #[test]
    fn test_parse_multiple_selectors() {
        let ss = parse_stylesheet("h1, h2, h3 { font-weight: bold; }");
        assert_eq!(ss.rules.len(), 1);
        assert_eq!(ss.rules[0].selectors.len(), 3);
    }

    #[test]
    fn test_parse_descendant_combinator() {
        let ss = parse_stylesheet("div p { color: blue; }");
        assert_eq!(ss.rules.len(), 1);
        let parts = &ss.rules[0].selectors[0].parts;
        assert!(parts.iter().any(|p| matches!(p, SelectorPart::Combinator(Combinator::Descendant))));
    }

    #[test]
    fn test_parse_child_combinator() {
        let ss = parse_stylesheet("div > p { color: blue; }");
        let parts = &ss.rules[0].selectors[0].parts;
        assert!(parts.iter().any(|p| matches!(p, SelectorPart::Combinator(Combinator::Child))));
    }

    #[test]
    fn test_parse_important() {
        let ss = parse_stylesheet("p { color: red !important; }");
        assert!(ss.rules[0].declarations[0].important);
    }

    #[test]
    fn test_parse_hex_color() {
        let ss = parse_stylesheet("p { color: #ff0000; }");
        let val = &ss.rules[0].declarations[0].value;
        assert!(matches!(val, CssValue::Color(c) if c.r == 1.0 && c.g == 0.0));
    }

    #[test]
    fn test_parse_rgb_function() {
        let ss = parse_stylesheet("p { color: rgb(255, 128, 0); }");
        let val = &ss.rules[0].declarations[0].value;
        assert!(matches!(val, CssValue::Color(_)));
    }

    #[test]
    fn test_parse_margin_shorthand() {
        let ss = parse_stylesheet("p { margin: 10px 20px; }");
        let decls = &ss.rules[0].declarations;
        assert_eq!(decls.len(), 4); // top, right, bottom, left
        assert_eq!(decls[0].property, "margin-top");
        assert_eq!(decls[1].property, "margin-right");
    }

    #[test]
    fn test_parse_border_shorthand() {
        let ss = parse_stylesheet("p { border: 1px solid red; }");
        let decls = &ss.rules[0].declarations;
        assert_eq!(decls.len(), 3); // width, style, color
    }

    #[test]
    fn test_parse_inline_style() {
        let decls = parse_inline_style("color: blue; font-size: 14px;");
        assert_eq!(decls.len(), 2);
        assert_eq!(decls[0].property, "color");
        assert_eq!(decls[1].property, "font-size");
    }

    #[test]
    fn test_parse_import() {
        let ss = parse_stylesheet("@import 'styles.css'; p { color: red; }");
        assert_eq!(ss.imports.len(), 1);
        assert_eq!(ss.imports[0], "styles.css");
        assert_eq!(ss.rules.len(), 1);
    }

    #[test]
    fn test_parse_media_query() {
        let ss = parse_stylesheet("@media screen { p { color: blue; } }");
        assert_eq!(ss.rules.len(), 1);
        assert_eq!(ss.rules[0].media, Some("screen".into()));
    }

    #[test]
    fn test_parse_attribute_selector() {
        let ss = parse_stylesheet("[data-active] { display: block; }");
        let parts = &ss.rules[0].selectors[0].parts;
        assert!(parts.iter().any(|p| matches!(p, SelectorPart::Attribute { name, .. } if name == "data-active")));
    }

    #[test]
    fn test_parse_attribute_equals() {
        let ss = parse_stylesheet(r#"[type="text"] { border: 1px solid gray; }"#);
        let parts = &ss.rules[0].selectors[0].parts;
        assert!(parts.iter().any(|p| matches!(
            p,
            SelectorPart::Attribute { name, op: Some(AttrOp::Equals), value: Some(v) }
            if name == "type" && v == "text"
        )));
    }

    #[test]
    fn test_parse_pseudo_class() {
        let ss = parse_stylesheet("a:hover { color: red; }");
        let parts = &ss.rules[0].selectors[0].parts;
        assert!(parts.iter().any(|p| matches!(p, SelectorPart::PseudoClass(name, _) if name == "hover")));
    }

    #[test]
    fn test_parse_pseudo_element() {
        let ss = parse_stylesheet("p::before { content: ''; }");
        let parts = &ss.rules[0].selectors[0].parts;
        assert!(parts.iter().any(|p| matches!(p, SelectorPart::PseudoElement(name) if name == "before")));
    }

    #[test]
    fn test_specificity_ordering() {
        // #id > .class > tag
        assert!(Specificity(1, 0, 0) > Specificity(0, 1, 0));
        assert!(Specificity(0, 1, 0) > Specificity(0, 0, 1));
        assert!(Specificity(0, 2, 0) > Specificity(0, 1, 3));
    }

    #[test]
    fn test_parse_multiple_rules() {
        let css = "h1 { font-size: 24px; } p { font-size: 16px; } a { color: blue; }";
        let ss = parse_stylesheet(css);
        assert_eq!(ss.rules.len(), 3);
    }

    #[test]
    fn test_parse_url_value() {
        let ss = parse_stylesheet("div { background: url(image.png); }");
        let decls = &ss.rules[0].declarations;
        assert!(decls.iter().any(|d| d.property == "background-image"));
    }

    #[test]
    fn test_parse_percentage() {
        let ss = parse_stylesheet("div { width: 50%; }");
        let val = &ss.rules[0].declarations[0].value;
        assert!(matches!(val, CssValue::Percentage(50.0)));
    }

    #[test]
    fn test_universal_selector() {
        let ss = parse_stylesheet("* { box-sizing: border-box; }");
        assert_eq!(ss.rules.len(), 1);
        let parts = &ss.rules[0].selectors[0].parts;
        assert!(parts.iter().any(|p| matches!(p, SelectorPart::Universal)));
        assert_eq!(ss.rules[0].selectors[0].specificity, Specificity(0, 0, 0));
    }
}
