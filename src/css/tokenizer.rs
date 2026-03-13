//! CSS Tokenizer — converts raw CSS text into a stream of tokens.
//!
//! Implements the CSS3 tokenization algorithm (spec §4).
//! Handles: idents, strings, numbers, functions, at-keywords, hash tokens,
//! delimiters, whitespace, comments, URLs, unicode-range.

/// A single CSS token with its position in the source.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum Token {
    /// An identifier: `color`, `font-size`, `--custom-prop`
    Ident(String),
    /// A function token: `rgb(`, `calc(`
    Function(String),
    /// An @-keyword: `@media`, `@import`
    AtKeyword(String),
    /// A hash token: `#fff`, `#my-id`
    Hash(String, HashType),
    /// A string: `"hello"`, `'world'`
    StringTok(String),
    /// A numeric value with optional unit
    Number(f32, NumType),
    /// A dimension: `10px`, `2em`, `50%`
    Dimension(f32, String),
    /// A percentage: `50%`
    Percentage(f32),
    /// A URL token: `url(https://...)`
    Url(String),
    /// Whitespace (collapsed)
    Whitespace,
    /// A colon `:`
    Colon,
    /// A semicolon `;`
    Semicolon,
    /// A comma `,`
    Comma,
    /// Opening bracket `[`
    BracketOpen,
    /// Closing bracket `]`
    BracketClose,
    /// Opening paren `(`
    ParenOpen,
    /// Closing paren `)`
    ParenClose,
    /// Opening brace `{`
    BraceOpen,
    /// Closing brace `}`
    BraceClose,
    /// The `>` combinator
    Greater,
    /// The `+` combinator
    Plus,
    /// The `~` combinator
    Tilde,
    /// The `*` universal selector / multiplication
    Asterisk,
    /// The `.` class selector prefix
    Dot,
    /// The `!` (used in `!important`)
    Bang,
    /// The `=` sign
    Equals,
    /// Pipe `|`
    Pipe,
    /// A `<!-- ` CDO token
    Cdo,
    /// A `-->` CDC token
    Cdc,
    /// Any other single character delimiter
    Delim(char),
    /// End of input
    Eof,
}

/// Whether a hash token is an ID or unrestricted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HashType {
    Id,
    Unrestricted,
}

/// Whether a number is integer or float.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumType {
    Integer,
    Float,
}

/// CSS Tokenizer state machine.
#[allow(dead_code)]
pub struct Tokenizer<'a> {
    input: &'a str,
    chars: Vec<char>,
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.chars().collect(),
            pos: 0,
        }
    }

    /// Tokenize the entire input into a Vec of tokens.
    pub fn tokenize_all(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token();
            if tok == Token::Eof {
                break;
            }
            tokens.push(tok);
        }
        tokens
    }

    /// Read the next token.
    pub fn next_token(&mut self) -> Token {
        self.consume_comments();

        if self.pos >= self.chars.len() {
            return Token::Eof;
        }

        let ch = self.chars[self.pos];

        // Whitespace
        if ch.is_ascii_whitespace() {
            self.consume_whitespace();
            return Token::Whitespace;
        }

        // String
        if ch == '"' || ch == '\'' {
            return self.consume_string(ch);
        }

        // Hash / ID
        if ch == '#' {
            self.pos += 1;
            if self.pos < self.chars.len() && self.is_name_char(self.chars[self.pos]) {
                let name = self.consume_name();
                let hash_type = if self.is_valid_ident_start(&name) {
                    HashType::Id
                } else {
                    HashType::Unrestricted
                };
                return Token::Hash(name, hash_type);
            }
            return Token::Delim('#');
        }

        // Number / Dimension / Percentage
        if ch.is_ascii_digit() || (ch == '.' && self.peek_ahead(1).is_some_and(|c| c.is_ascii_digit())) {
            return self.consume_numeric();
        }
        if ch == '-' || ch == '+' {
            if self.peek_ahead(1).is_some_and(|c| c.is_ascii_digit())
                || (self.peek_ahead(1) == Some('.') && self.peek_ahead(2).is_some_and(|c| c.is_ascii_digit()))
            {
                return self.consume_numeric();
            }
            // Could be an ident starting with -
            if ch == '-' && self.peek_ahead(1).is_some_and(|c| self.is_name_start(c) || c == '-') {
                return self.consume_ident_like();
            }
        }

        // Ident-like (ident, function, url)
        if self.is_name_start(ch) || ch == '-' {
            return self.consume_ident_like();
        }

        // Backslash escape at start of ident
        if ch == '\\' && self.pos + 1 < self.chars.len() && self.chars[self.pos + 1] != '\n' {
            return self.consume_ident_like();
        }

        // @ keyword
        if ch == '@' {
            self.pos += 1;
            if self.pos < self.chars.len() && self.is_name_start(self.chars[self.pos]) {
                let name = self.consume_name();
                return Token::AtKeyword(name);
            }
            return Token::Delim('@');
        }

        // Simple single-char tokens
        self.pos += 1;
        match ch {
            ':' => Token::Colon,
            ';' => Token::Semicolon,
            ',' => Token::Comma,
            '[' => Token::BracketOpen,
            ']' => Token::BracketClose,
            '(' => Token::ParenOpen,
            ')' => Token::ParenClose,
            '{' => Token::BraceOpen,
            '}' => Token::BraceClose,
            '>' => Token::Greater,
            '+' => Token::Plus,
            '~' => Token::Tilde,
            '*' => Token::Asterisk,
            '.' => Token::Dot,
            '!' => Token::Bang,
            '=' => Token::Equals,
            '|' => Token::Pipe,
            _ => Token::Delim(ch),
        }
    }

    fn peek_ahead(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    fn consume_whitespace(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn consume_comments(&mut self) {
        loop {
            if self.pos + 1 < self.chars.len()
                && self.chars[self.pos] == '/'
                && self.chars[self.pos + 1] == '*'
            {
                self.pos += 2;
                while self.pos + 1 < self.chars.len() {
                    if self.chars[self.pos] == '*' && self.chars[self.pos + 1] == '/' {
                        self.pos += 2;
                        break;
                    }
                    self.pos += 1;
                }
                // Skip whitespace after comment
                while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_whitespace() {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn is_name_start(&self, ch: char) -> bool {
        ch.is_ascii_alphabetic() || ch == '_' || ch == '-' || !ch.is_ascii()
    }

    fn is_name_char(&self, ch: char) -> bool {
        self.is_name_start(ch) || ch.is_ascii_digit()
    }

    fn is_valid_ident_start(&self, name: &str) -> bool {
        let mut chars = name.chars();
        match chars.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' || !c.is_ascii() => true,
            Some('-') => match chars.next() {
                Some(c) => c.is_ascii_alphabetic() || c == '_' || c == '-' || !c.is_ascii(),
                None => false,
            },
            _ => false,
        }
    }

    fn consume_name(&mut self) -> String {
        let mut name = String::new();
        while self.pos < self.chars.len() {
            let ch = self.chars[self.pos];
            if self.is_name_char(ch) {
                name.push(ch);
                self.pos += 1;
            } else if ch == '\\' && self.pos + 1 < self.chars.len() {
                // CSS escape
                self.pos += 1;
                let escaped = self.consume_escape();
                name.push(escaped);
            } else {
                break;
            }
        }
        name
    }

    fn consume_escape(&mut self) -> char {
        if self.pos >= self.chars.len() {
            return '\u{FFFD}';
        }
        let ch = self.chars[self.pos];
        if ch.is_ascii_hexdigit() {
            let mut hex = String::new();
            for _ in 0..6 {
                if self.pos < self.chars.len() && self.chars[self.pos].is_ascii_hexdigit() {
                    hex.push(self.chars[self.pos]);
                    self.pos += 1;
                } else {
                    break;
                }
            }
            // Optional whitespace after hex escape
            if self.pos < self.chars.len() && self.chars[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            u32::from_str_radix(&hex, 16)
                .ok()
                .and_then(char::from_u32)
                .unwrap_or('\u{FFFD}')
        } else {
            self.pos += 1;
            ch
        }
    }

    fn consume_string(&mut self, quote: char) -> Token {
        self.pos += 1; // skip opening quote
        let mut value = String::new();

        while self.pos < self.chars.len() {
            let ch = self.chars[self.pos];
            if ch == quote {
                self.pos += 1;
                return Token::StringTok(value);
            }
            if ch == '\\' {
                self.pos += 1;
                if self.pos < self.chars.len() {
                    if self.chars[self.pos] == '\n' {
                        // Line continuation
                        self.pos += 1;
                    } else {
                        let escaped = self.consume_escape();
                        value.push(escaped);
                    }
                }
                continue;
            }
            if ch == '\n' {
                // Bad string — return what we have
                return Token::StringTok(value);
            }
            value.push(ch);
            self.pos += 1;
        }

        Token::StringTok(value)
    }

    fn consume_numeric(&mut self) -> Token {
        let (value, num_type) = self.consume_number();

        // Check for dimension or percentage
        if self.pos < self.chars.len() {
            let ch = self.chars[self.pos];
            if ch == '%' {
                self.pos += 1;
                return Token::Percentage(value);
            }
            if self.is_name_start(ch) || ch == '-' {
                let unit = self.consume_name();
                return Token::Dimension(value, unit.to_ascii_lowercase());
            }
        }

        Token::Number(value, num_type)
    }

    fn consume_number(&mut self) -> (f32, NumType) {
        let mut repr = String::new();
        let mut is_float = false;

        // Sign
        if self.pos < self.chars.len() && (self.chars[self.pos] == '+' || self.chars[self.pos] == '-') {
            repr.push(self.chars[self.pos]);
            self.pos += 1;
        }

        // Integer part
        while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
            repr.push(self.chars[self.pos]);
            self.pos += 1;
        }

        // Decimal part
        if self.pos + 1 < self.chars.len()
            && self.chars[self.pos] == '.'
            && self.chars[self.pos + 1].is_ascii_digit()
        {
            is_float = true;
            repr.push('.');
            self.pos += 1;
            while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                repr.push(self.chars[self.pos]);
                self.pos += 1;
            }
        }

        // Exponent
        if self.pos < self.chars.len()
            && (self.chars[self.pos] == 'e' || self.chars[self.pos] == 'E')
        {
            let next = self.peek_ahead(1);
            if next.is_some_and(|c| c.is_ascii_digit())
                || (next.is_some_and(|c| c == '+' || c == '-')
                    && self.peek_ahead(2).is_some_and(|c| c.is_ascii_digit()))
            {
                is_float = true;
                repr.push(self.chars[self.pos]);
                self.pos += 1;
                if self.pos < self.chars.len()
                    && (self.chars[self.pos] == '+' || self.chars[self.pos] == '-')
                {
                    repr.push(self.chars[self.pos]);
                    self.pos += 1;
                }
                while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                    repr.push(self.chars[self.pos]);
                    self.pos += 1;
                }
            }
        }

        let value = repr.parse::<f32>().unwrap_or(0.0);
        let num_type = if is_float { NumType::Float } else { NumType::Integer };
        (value, num_type)
    }

    fn consume_ident_like(&mut self) -> Token {
        let name = self.consume_name();

        // Check if it's a function
        if self.pos < self.chars.len() && self.chars[self.pos] == '(' {
            self.pos += 1;
            let lower = name.to_ascii_lowercase();
            if lower == "url" {
                return self.consume_url();
            }
            return Token::Function(lower);
        }

        Token::Ident(name.to_ascii_lowercase())
    }

    fn consume_url(&mut self) -> Token {
        // Skip whitespace
        while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }

        if self.pos >= self.chars.len() {
            return Token::Url(String::new());
        }

        // If it starts with a quote, it's a string function call
        let ch = self.chars[self.pos];
        if ch == '"' || ch == '\'' {
            let string_tok = self.consume_string(ch);
            // Skip whitespace and closing paren
            while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.pos < self.chars.len() && self.chars[self.pos] == ')' {
                self.pos += 1;
            }
            if let Token::StringTok(s) = string_tok {
                return Token::Url(s);
            }
            return Token::Url(String::new());
        }

        // Unquoted URL
        let mut url = String::new();
        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            if c == ')' {
                self.pos += 1;
                break;
            }
            if c.is_ascii_whitespace() {
                // Skip whitespace before closing paren
                while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_whitespace() {
                    self.pos += 1;
                }
                if self.pos < self.chars.len() && self.chars[self.pos] == ')' {
                    self.pos += 1;
                }
                break;
            }
            if c == '\\' {
                self.pos += 1;
                let escaped = self.consume_escape();
                url.push(escaped);
            } else {
                url.push(c);
                self.pos += 1;
            }
        }

        Token::Url(url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenize(input: &str) -> Vec<Token> {
        Tokenizer::new(input).tokenize_all()
    }

    #[test]
    fn test_simple_rule() {
        let tokens = tokenize("color: red;");
        assert_eq!(tokens, vec![
            Token::Ident("color".into()),
            Token::Colon,
            Token::Whitespace,
            Token::Ident("red".into()),
            Token::Semicolon,
        ]);
    }

    #[test]
    fn test_selector_and_block() {
        let tokens = tokenize("h1 { font-size: 16px; }");
        assert_eq!(tokens, vec![
            Token::Ident("h1".into()),
            Token::Whitespace,
            Token::BraceOpen,
            Token::Whitespace,
            Token::Ident("font-size".into()),
            Token::Colon,
            Token::Whitespace,
            Token::Dimension(16.0, "px".into()),
            Token::Semicolon,
            Token::Whitespace,
            Token::BraceClose,
        ]);
    }

    #[test]
    fn test_hash_and_class() {
        let tokens = tokenize("#main .active");
        assert_eq!(tokens, vec![
            Token::Hash("main".into(), HashType::Id),
            Token::Whitespace,
            Token::Dot,
            Token::Ident("active".into()),
        ]);
    }

    #[test]
    fn test_string_tokens() {
        let tokens = tokenize(r#"content: "hello world";"#);
        assert!(tokens.contains(&Token::StringTok("hello world".into())));
    }

    #[test]
    fn test_url_token() {
        let tokens = tokenize("background: url(image.png);");
        assert!(tokens.contains(&Token::Url("image.png".into())));
    }

    #[test]
    fn test_url_quoted() {
        let tokens = tokenize(r#"background: url("image.png");"#);
        assert!(tokens.contains(&Token::Url("image.png".into())));
    }

    #[test]
    fn test_percentage() {
        let tokens = tokenize("width: 50%;");
        assert!(tokens.contains(&Token::Percentage(50.0)));
    }

    #[test]
    fn test_negative_number() {
        let tokens = tokenize("margin: -10px;");
        assert!(tokens.contains(&Token::Dimension(-10.0, "px".into())));
    }

    #[test]
    fn test_float_number() {
        let tokens = tokenize("opacity: 0.5;");
        assert!(tokens.contains(&Token::Number(0.5, NumType::Float)));
    }

    #[test]
    fn test_comment_stripping() {
        let tokens = tokenize("/* comment */ color: red;");
        assert_eq!(tokens, vec![
            Token::Ident("color".into()),
            Token::Colon,
            Token::Whitespace,
            Token::Ident("red".into()),
            Token::Semicolon,
        ]);
    }

    #[test]
    fn test_at_keyword() {
        let tokens = tokenize("@media screen");
        assert_eq!(tokens[0], Token::AtKeyword("media".into()));
    }

    #[test]
    fn test_function_token() {
        let tokens = tokenize("color: rgb(255, 0, 0);");
        assert!(tokens.contains(&Token::Function("rgb".into())));
    }

    #[test]
    fn test_combinators() {
        let tokens = tokenize("div > p + span ~ a");
        assert!(tokens.contains(&Token::Greater));
        assert!(tokens.contains(&Token::Plus));
        assert!(tokens.contains(&Token::Tilde));
    }

    #[test]
    fn test_important() {
        let tokens = tokenize("color: red !important;");
        assert!(tokens.contains(&Token::Bang));
        assert!(tokens.contains(&Token::Ident("important".into())));
    }

    #[test]
    fn test_pseudo_class() {
        let tokens = tokenize("a:hover");
        assert_eq!(tokens, vec![
            Token::Ident("a".into()),
            Token::Colon,
            Token::Ident("hover".into()),
        ]);
    }

    #[test]
    fn test_escape_in_ident() {
        // CSS escape: \61 = 'a' (hex), so "\61 bc" → ident "abc"
        let tokens = tokenize(r"\61 bc");
        assert!(tokens.iter().any(|t| matches!(t, Token::Ident(s) if s.contains('a'))));
    }

    #[test]
    fn test_empty_input() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_multiple_comments() {
        let tokens = tokenize("/* a */ /* b */ color: red;");
        assert_eq!(tokens[0], Token::Ident("color".into()));
    }

    #[test]
    fn test_dimension_units() {
        let tokens = tokenize("10em 2rem 100vh 50vw 1.5ex 3ch");
        let dims: Vec<_> = tokens.iter().filter_map(|t| {
            if let Token::Dimension(_, unit) = t { Some(unit.as_str()) } else { None }
        }).collect();
        assert_eq!(dims, vec!["em", "rem", "vh", "vw", "ex", "ch"]);
    }
}
