//! CSS Value types — colors, lengths, display, position, etc.
//!
//! These are the resolved value types used by the layout engine.
//! Parsing from CSS token streams happens in parser.rs.

/// A CSS color value (RGBA, 0.0–1.0 per channel).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CssColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl CssColor {
    pub const TRANSPARENT: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const INHERIT: Self = Self { r: -1.0, g: -1.0, b: -1.0, a: -1.0 };

    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: 1.0,
        }
    }

    pub fn rgba(r: u8, g: u8, b: u8, a: f32) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a,
        }
    }

    pub fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    pub fn is_inherit(&self) -> bool {
        self.r < 0.0
    }

    /// Parse a hex color: #RGB, #RRGGBB, #RGBA, #RRGGBBAA
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim_start_matches('#');
        match hex.len() {
            3 => {
                let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
                Some(Self::rgb(r, g, b))
            }
            4 => {
                let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
                let a = u8::from_str_radix(&hex[3..4], 16).ok()? * 17;
                Some(Self::rgba(r, g, b, a as f32 / 255.0))
            }
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                Some(Self::rgb(r, g, b))
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
                Some(Self::rgba(r, g, b, a as f32 / 255.0))
            }
            _ => None,
        }
    }

    /// Parse a named CSS color.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "black" => Some(Self::rgb(0, 0, 0)),
            "white" => Some(Self::rgb(255, 255, 255)),
            "red" => Some(Self::rgb(255, 0, 0)),
            "green" => Some(Self::rgb(0, 128, 0)),
            "blue" => Some(Self::rgb(0, 0, 255)),
            "yellow" => Some(Self::rgb(255, 255, 0)),
            "cyan" | "aqua" => Some(Self::rgb(0, 255, 255)),
            "magenta" | "fuchsia" => Some(Self::rgb(255, 0, 255)),
            "orange" => Some(Self::rgb(255, 165, 0)),
            "purple" => Some(Self::rgb(128, 0, 128)),
            "pink" => Some(Self::rgb(255, 192, 203)),
            "brown" => Some(Self::rgb(165, 42, 42)),
            "gray" | "grey" => Some(Self::rgb(128, 128, 128)),
            "silver" => Some(Self::rgb(192, 192, 192)),
            "maroon" => Some(Self::rgb(128, 0, 0)),
            "olive" => Some(Self::rgb(128, 128, 0)),
            "lime" => Some(Self::rgb(0, 255, 0)),
            "teal" => Some(Self::rgb(0, 128, 128)),
            "navy" => Some(Self::rgb(0, 0, 128)),
            "transparent" => Some(Self::TRANSPARENT),
            "inherit" => Some(Self::INHERIT),
            // Extended colors
            "coral" => Some(Self::rgb(255, 127, 80)),
            "crimson" => Some(Self::rgb(220, 20, 60)),
            "darkblue" => Some(Self::rgb(0, 0, 139)),
            "darkgray" | "darkgrey" => Some(Self::rgb(169, 169, 169)),
            "darkgreen" => Some(Self::rgb(0, 100, 0)),
            "darkred" => Some(Self::rgb(139, 0, 0)),
            "deeppink" => Some(Self::rgb(255, 20, 147)),
            "dodgerblue" => Some(Self::rgb(30, 144, 255)),
            "firebrick" => Some(Self::rgb(178, 34, 34)),
            "gold" => Some(Self::rgb(255, 215, 0)),
            "goldenrod" => Some(Self::rgb(218, 165, 32)),
            "hotpink" => Some(Self::rgb(255, 105, 180)),
            "indianred" => Some(Self::rgb(205, 92, 92)),
            "indigo" => Some(Self::rgb(75, 0, 130)),
            "ivory" => Some(Self::rgb(255, 255, 240)),
            "khaki" => Some(Self::rgb(240, 230, 140)),
            "lavender" => Some(Self::rgb(230, 230, 250)),
            "lightblue" => Some(Self::rgb(173, 216, 230)),
            "lightcoral" => Some(Self::rgb(240, 128, 128)),
            "lightgray" | "lightgrey" => Some(Self::rgb(211, 211, 211)),
            "lightgreen" => Some(Self::rgb(144, 238, 144)),
            "lightyellow" => Some(Self::rgb(255, 255, 224)),
            "limegreen" => Some(Self::rgb(50, 205, 50)),
            "mediumblue" => Some(Self::rgb(0, 0, 205)),
            "midnightblue" => Some(Self::rgb(25, 25, 112)),
            "mintcream" => Some(Self::rgb(245, 255, 250)),
            "mistyrose" => Some(Self::rgb(255, 228, 225)),
            "moccasin" => Some(Self::rgb(255, 228, 181)),
            "oldlace" => Some(Self::rgb(253, 245, 230)),
            "orangered" => Some(Self::rgb(255, 69, 0)),
            "orchid" => Some(Self::rgb(218, 112, 214)),
            "palegreen" => Some(Self::rgb(152, 251, 152)),
            "plum" => Some(Self::rgb(221, 160, 221)),
            "powderblue" => Some(Self::rgb(176, 224, 230)),
            "rosybrown" => Some(Self::rgb(188, 143, 143)),
            "royalblue" => Some(Self::rgb(65, 105, 225)),
            "salmon" => Some(Self::rgb(250, 128, 114)),
            "sandybrown" => Some(Self::rgb(244, 164, 96)),
            "seagreen" => Some(Self::rgb(46, 139, 87)),
            "sienna" => Some(Self::rgb(160, 82, 45)),
            "skyblue" => Some(Self::rgb(135, 206, 235)),
            "slateblue" => Some(Self::rgb(106, 90, 205)),
            "slategray" | "slategrey" => Some(Self::rgb(112, 128, 144)),
            "springgreen" => Some(Self::rgb(0, 255, 127)),
            "steelblue" => Some(Self::rgb(70, 130, 180)),
            "tan" => Some(Self::rgb(210, 180, 140)),
            "thistle" => Some(Self::rgb(216, 191, 216)),
            "tomato" => Some(Self::rgb(255, 99, 71)),
            "turquoise" => Some(Self::rgb(64, 224, 208)),
            "violet" => Some(Self::rgb(238, 130, 238)),
            "wheat" => Some(Self::rgb(245, 222, 179)),
            "whitesmoke" => Some(Self::rgb(245, 245, 245)),
            "yellowgreen" => Some(Self::rgb(154, 205, 50)),
            "rebeccapurple" => Some(Self::rgb(102, 51, 153)),
            "aliceblue" => Some(Self::rgb(240, 248, 255)),
            "antiquewhite" => Some(Self::rgb(250, 235, 215)),
            "aquamarine" => Some(Self::rgb(127, 255, 212)),
            "azure" => Some(Self::rgb(240, 255, 255)),
            "beige" => Some(Self::rgb(245, 245, 220)),
            "bisque" => Some(Self::rgb(255, 228, 196)),
            "blanchedalmond" => Some(Self::rgb(255, 235, 205)),
            "blueviolet" => Some(Self::rgb(138, 43, 226)),
            "burlywood" => Some(Self::rgb(222, 184, 135)),
            "cadetblue" => Some(Self::rgb(95, 158, 160)),
            "chartreuse" => Some(Self::rgb(127, 255, 0)),
            "chocolate" => Some(Self::rgb(210, 105, 30)),
            "cornflowerblue" => Some(Self::rgb(100, 149, 237)),
            "cornsilk" => Some(Self::rgb(255, 248, 220)),
            "darkcyan" => Some(Self::rgb(0, 139, 139)),
            "darkgoldenrod" => Some(Self::rgb(184, 134, 11)),
            "darkkhaki" => Some(Self::rgb(189, 183, 107)),
            "darkmagenta" => Some(Self::rgb(139, 0, 139)),
            "darkolivegreen" => Some(Self::rgb(85, 107, 47)),
            "darkorange" => Some(Self::rgb(255, 140, 0)),
            "darkorchid" => Some(Self::rgb(153, 50, 204)),
            "darksalmon" => Some(Self::rgb(233, 150, 122)),
            "darkseagreen" => Some(Self::rgb(143, 188, 143)),
            "darkslateblue" => Some(Self::rgb(72, 61, 139)),
            "darkslategray" | "darkslategrey" => Some(Self::rgb(47, 79, 79)),
            "darkturquoise" => Some(Self::rgb(0, 206, 209)),
            "darkviolet" => Some(Self::rgb(148, 0, 211)),
            "deepskyblue" => Some(Self::rgb(0, 191, 255)),
            "dimgray" | "dimgrey" => Some(Self::rgb(105, 105, 105)),
            "floralwhite" => Some(Self::rgb(255, 250, 240)),
            "forestgreen" => Some(Self::rgb(34, 139, 34)),
            "gainsboro" => Some(Self::rgb(220, 220, 220)),
            "ghostwhite" => Some(Self::rgb(248, 248, 255)),
            "greenyellow" => Some(Self::rgb(173, 255, 47)),
            "honeydew" => Some(Self::rgb(240, 255, 240)),
            "lawngreen" => Some(Self::rgb(124, 252, 0)),
            "lemonchiffon" => Some(Self::rgb(255, 250, 205)),
            "lightcyan" => Some(Self::rgb(224, 255, 255)),
            "lightpink" => Some(Self::rgb(255, 182, 193)),
            "lightsalmon" => Some(Self::rgb(255, 160, 122)),
            "lightseagreen" => Some(Self::rgb(32, 178, 170)),
            "lightskyblue" => Some(Self::rgb(135, 206, 250)),
            "lightslategray" | "lightslategrey" => Some(Self::rgb(119, 136, 153)),
            "lightsteelblue" => Some(Self::rgb(176, 196, 222)),
            "linen" => Some(Self::rgb(250, 240, 230)),
            "mediumaquamarine" => Some(Self::rgb(102, 205, 170)),
            "mediumorchid" => Some(Self::rgb(186, 85, 211)),
            "mediumpurple" => Some(Self::rgb(147, 112, 219)),
            "mediumseagreen" => Some(Self::rgb(60, 179, 113)),
            "mediumslateblue" => Some(Self::rgb(123, 104, 238)),
            "mediumspringgreen" => Some(Self::rgb(0, 250, 154)),
            "mediumturquoise" => Some(Self::rgb(72, 209, 204)),
            "mediumvioletred" => Some(Self::rgb(199, 21, 133)),
            "navajowhite" => Some(Self::rgb(255, 222, 173)),
            "olivedrab" => Some(Self::rgb(107, 142, 35)),
            "palegoldenrod" => Some(Self::rgb(238, 232, 170)),
            "paleturquoise" => Some(Self::rgb(175, 238, 238)),
            "palevioletred" => Some(Self::rgb(219, 112, 147)),
            "papayawhip" => Some(Self::rgb(255, 239, 213)),
            "peachpuff" => Some(Self::rgb(255, 218, 185)),
            "peru" => Some(Self::rgb(205, 133, 63)),
            "seashell" => Some(Self::rgb(255, 245, 238)),
            "snow" => Some(Self::rgb(255, 250, 250)),
            _ => None,
        }
    }

    /// Parse from rgb()/rgba() function arguments.
    pub fn from_rgb_args(args: &[f32]) -> Option<Self> {
        match args.len() {
            3 => Some(Self::rgb(
                args[0].clamp(0.0, 255.0) as u8,
                args[1].clamp(0.0, 255.0) as u8,
                args[2].clamp(0.0, 255.0) as u8,
            )),
            4 => Some(Self::rgba(
                args[0].clamp(0.0, 255.0) as u8,
                args[1].clamp(0.0, 255.0) as u8,
                args[2].clamp(0.0, 255.0) as u8,
                args[3].clamp(0.0, 1.0),
            )),
            _ => None,
        }
    }

    /// Parse from hsl()/hsla() function arguments.
    pub fn from_hsl_args(args: &[f32]) -> Option<Self> {
        if args.len() < 3 {
            return None;
        }
        let h = args[0] % 360.0 / 360.0;
        let s = (args[1] / 100.0).clamp(0.0, 1.0);
        let l = (args[2] / 100.0).clamp(0.0, 1.0);
        let a = if args.len() >= 4 { args[3].clamp(0.0, 1.0) } else { 1.0 };

        let (r, g, b) = hsl_to_rgb(h, s, l);
        Some(Self {
            r, g, b, a,
        })
    }
}

impl Default for CssColor {
    fn default() -> Self {
        Self::INHERIT
    }
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h);
    let b = hue_to_rgb(p, q, h - 1.0 / 3.0);
    (r, g, b)
}

fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 { t += 1.0; }
    if t > 1.0 { t -= 1.0; }
    if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
    if t < 1.0 / 2.0 { return q; }
    if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    p
}

/// A CSS length value with its unit.
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum CssLength {
    /// Pixels (absolute)
    Px(f32),
    /// Relative to font size of element
    Em(f32),
    /// Relative to font size of root element
    Rem(f32),
    /// Percentage of containing block
    Percent(f32),
    /// Viewport width percentage
    Vw(f32),
    /// Viewport height percentage
    Vh(f32),
    /// Points (1pt = 1/72 inch ≈ 1.333px)
    Pt(f32),
    /// Zero
    Zero,
    /// auto
    Auto,
    /// No value specified (use default/inherit)
    #[default]
    None,
}

impl CssLength {
    /// Resolve to pixels given context.
    pub fn to_px(&self, font_size: f32, viewport_w: f32, viewport_h: f32) -> f32 {
        match self {
            Self::Px(v) => *v,
            Self::Em(v) => v * font_size,
            Self::Rem(v) => v * 16.0, // root font size = 16px default
            Self::Percent(v) => v / 100.0 * viewport_w, // context-dependent
            Self::Vw(v) => v / 100.0 * viewport_w,
            Self::Vh(v) => v / 100.0 * viewport_h,
            Self::Pt(v) => v * 1.333,
            Self::Zero => 0.0,
            Self::Auto | Self::None => 0.0,
        }
    }

    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Parse a length from a numeric value and unit string.
    pub fn from_dimension(value: f32, unit: &str) -> Self {
        match unit {
            "px" => Self::Px(value),
            "em" => Self::Em(value),
            "rem" => Self::Rem(value),
            "vw" => Self::Vw(value),
            "vh" => Self::Vh(value),
            "pt" => Self::Pt(value),
            "%" => Self::Percent(value),
            "cm" => Self::Px(value * 37.795),
            "mm" => Self::Px(value * 3.7795),
            "in" => Self::Px(value * 96.0),
            "pc" => Self::Px(value * 16.0),
            "ex" => Self::Em(value * 0.5), // approximate
            "ch" => Self::Em(value * 0.5), // approximate
            _ => Self::Px(value), // fallback
        }
    }
}


/// CSS display property values.
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum CssDisplay {
    Block,
    #[default]
    Inline,
    InlineBlock,
    Flex,
    InlineFlex,
    Grid,
    None,
    ListItem,
    Table,
    TableRow,
    TableCell,
    TableHeaderGroup,
    TableRowGroup,
    TableFooterGroup,
    TableCaption,
    TableColumn,
    TableColumnGroup,
    Contents,
}

impl CssDisplay {
    pub fn from_str(s: &str) -> Self {
        match s {
            "block" => Self::Block,
            "inline" => Self::Inline,
            "inline-block" => Self::InlineBlock,
            "flex" => Self::Flex,
            "inline-flex" => Self::InlineFlex,
            "grid" => Self::Grid,
            "none" => Self::None,
            "list-item" => Self::ListItem,
            "table" => Self::Table,
            "table-row" => Self::TableRow,
            "table-cell" => Self::TableCell,
            "table-header-group" => Self::TableHeaderGroup,
            "table-row-group" => Self::TableRowGroup,
            "table-footer-group" => Self::TableFooterGroup,
            "table-caption" => Self::TableCaption,
            "table-column" => Self::TableColumn,
            "table-column-group" => Self::TableColumnGroup,
            "contents" => Self::Contents,
            _ => Self::Block,
        }
    }

    pub fn is_block_level(&self) -> bool {
        matches!(self, Self::Block | Self::Flex | Self::Grid | Self::Table | Self::ListItem)
    }
}


/// CSS position property.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CssPosition {
    #[default]
    Static,
    Relative,
    Absolute,
    Fixed,
    Sticky,
}

impl CssPosition {
    pub fn from_str(s: &str) -> Self {
        match s {
            "static" => Self::Static,
            "relative" => Self::Relative,
            "absolute" => Self::Absolute,
            "fixed" => Self::Fixed,
            "sticky" => Self::Sticky,
            _ => Self::Static,
        }
    }
}

/// CSS float property.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CssFloat {
    #[default]
    None,
    Left,
    Right,
}

impl CssFloat {
    pub fn from_str(s: &str) -> Self {
        match s {
            "left" => Self::Left,
            "right" => Self::Right,
            "none" => Self::None,
            _ => Self::None,
        }
    }
}

/// CSS overflow property.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CssOverflow {
    #[default]
    Visible,
    Hidden,
    Scroll,
    Auto,
}

impl CssOverflow {
    pub fn from_str(s: &str) -> Self {
        match s {
            "visible" => Self::Visible,
            "hidden" => Self::Hidden,
            "scroll" => Self::Scroll,
            "auto" => Self::Auto,
            _ => Self::Visible,
        }
    }
}

/// CSS text-align property.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CssTextAlign {
    #[default]
    Left,
    Right,
    Center,
    Justify,
}

impl CssTextAlign {
    pub fn from_str(s: &str) -> Self {
        match s {
            "left" => Self::Left,
            "right" => Self::Right,
            "center" => Self::Center,
            "justify" => Self::Justify,
            "start" => Self::Left,
            "end" => Self::Right,
            _ => Self::Left,
        }
    }
}

/// CSS font-weight (numeric 100-900 or named).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CssFontWeight(pub u16);

impl CssFontWeight {
    pub const NORMAL: Self = Self(400);
    pub const BOLD: Self = Self(700);
    pub const LIGHTER: Self = Self(100);
    pub const BOLDER: Self = Self(900);

    pub fn from_str(s: &str) -> Self {
        match s {
            "normal" => Self::NORMAL,
            "bold" => Self::BOLD,
            "lighter" => Self::LIGHTER,
            "bolder" => Self::BOLDER,
            _ => {
                if let Ok(n) = s.parse::<u16>() {
                    Self(n.clamp(1, 1000))
                } else {
                    Self::NORMAL
                }
            }
        }
    }

    pub fn is_bold(&self) -> bool {
        self.0 >= 700
    }
}

impl Default for CssFontWeight {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// CSS font-style.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CssFontStyle {
    #[default]
    Normal,
    Italic,
    Oblique,
}

impl CssFontStyle {
    pub fn from_str(s: &str) -> Self {
        match s {
            "italic" => Self::Italic,
            "oblique" => Self::Oblique,
            "normal" => Self::Normal,
            _ => Self::Normal,
        }
    }
}

/// CSS text-decoration.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CssTextDecoration {
    pub underline: bool,
    pub overline: bool,
    pub line_through: bool,
}

impl CssTextDecoration {
    pub fn from_str(s: &str) -> Self {
        let mut td = Self::default();
        for part in s.split_whitespace() {
            match part {
                "underline" => td.underline = true,
                "overline" => td.overline = true,
                "line-through" => td.line_through = true,
                "none" => return Self::default(),
                _ => {}
            }
        }
        td
    }
}

/// CSS white-space property.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CssWhiteSpace {
    #[default]
    Normal,
    Nowrap,
    Pre,
    PreWrap,
    PreLine,
    BreakSpaces,
}

impl CssWhiteSpace {
    pub fn from_str(s: &str) -> Self {
        match s {
            "normal" => Self::Normal,
            "nowrap" => Self::Nowrap,
            "pre" => Self::Pre,
            "pre-wrap" => Self::PreWrap,
            "pre-line" => Self::PreLine,
            "break-spaces" => Self::BreakSpaces,
            _ => Self::Normal,
        }
    }
}

/// CSS vertical-align property.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CssVerticalAlign {
    #[default]
    Baseline,
    Top,
    Middle,
    Bottom,
    TextTop,
    TextBottom,
    Sub,
    Super,
}

impl CssVerticalAlign {
    pub fn from_str(s: &str) -> Self {
        match s {
            "baseline" => Self::Baseline,
            "top" => Self::Top,
            "middle" => Self::Middle,
            "bottom" => Self::Bottom,
            "text-top" => Self::TextTop,
            "text-bottom" => Self::TextBottom,
            "sub" => Self::Sub,
            "super" => Self::Super,
            _ => Self::Baseline,
        }
    }
}

/// CSS border-style property.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CssBorderStyle {
    #[default]
    None,
    Solid,
    Dashed,
    Dotted,
    Double,
    Groove,
    Ridge,
    Inset,
    Outset,
    Hidden,
}

impl CssBorderStyle {
    pub fn from_str(s: &str) -> Self {
        match s {
            "none" => Self::None,
            "solid" => Self::Solid,
            "dashed" => Self::Dashed,
            "dotted" => Self::Dotted,
            "double" => Self::Double,
            "groove" => Self::Groove,
            "ridge" => Self::Ridge,
            "inset" => Self::Inset,
            "outset" => Self::Outset,
            "hidden" => Self::Hidden,
            _ => Self::None,
        }
    }
}

/// CSS flex-direction property.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CssFlexDirection {
    #[default]
    Row,
    RowReverse,
    Column,
    ColumnReverse,
}

impl CssFlexDirection {
    pub fn from_str(s: &str) -> Self {
        match s {
            "row" => Self::Row,
            "row-reverse" => Self::RowReverse,
            "column" => Self::Column,
            "column-reverse" => Self::ColumnReverse,
            _ => Self::Row,
        }
    }
}

/// CSS justify-content / align-items.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CssAlign {
    #[default]
    Start,
    End,
    Center,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
    Stretch,
    Baseline,
}

impl CssAlign {
    pub fn from_str(s: &str) -> Self {
        match s {
            "start" | "flex-start" => Self::Start,
            "end" | "flex-end" => Self::End,
            "center" => Self::Center,
            "space-between" => Self::SpaceBetween,
            "space-around" => Self::SpaceAround,
            "space-evenly" => Self::SpaceEvenly,
            "stretch" => Self::Stretch,
            "baseline" => Self::Baseline,
            _ => Self::Start,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_colors() {
        let c = CssColor::from_hex("#ff0000").unwrap();
        assert_eq!(c.r, 1.0);
        assert_eq!(c.g, 0.0);
        assert_eq!(c.b, 0.0);

        let c2 = CssColor::from_hex("#f00").unwrap();
        assert_eq!(c2.r, 1.0);
        assert_eq!(c2.g, 0.0);
    }

    #[test]
    fn test_named_colors() {
        assert_eq!(CssColor::from_name("red"), Some(CssColor::rgb(255, 0, 0)));
        assert_eq!(CssColor::from_name("blue"), Some(CssColor::rgb(0, 0, 255)));
        assert_eq!(CssColor::from_name("unknown"), None);
    }

    #[test]
    fn test_rgb_args() {
        let c = CssColor::from_rgb_args(&[255.0, 128.0, 0.0]).unwrap();
        assert_eq!(c.r, 1.0);
        assert!((c.g - 128.0/255.0).abs() < 0.01);
    }

    #[test]
    fn test_hsl_args() {
        // hsl(0, 100%, 50%) = red
        let c = CssColor::from_hsl_args(&[0.0, 100.0, 50.0]).unwrap();
        assert!((c.r - 1.0).abs() < 0.01);
        assert!(c.g < 0.01);
        assert!(c.b < 0.01);
    }

    #[test]
    fn test_length_to_px() {
        assert_eq!(CssLength::Px(10.0).to_px(16.0, 1920.0, 1080.0), 10.0);
        assert_eq!(CssLength::Em(2.0).to_px(16.0, 1920.0, 1080.0), 32.0);
        assert_eq!(CssLength::Rem(1.0).to_px(16.0, 1920.0, 1080.0), 16.0);
        assert_eq!(CssLength::Vw(50.0).to_px(16.0, 1920.0, 1080.0), 960.0);
        assert_eq!(CssLength::Vh(100.0).to_px(16.0, 1920.0, 1080.0), 1080.0);
    }

    #[test]
    fn test_display_values() {
        assert_eq!(CssDisplay::from_str("block"), CssDisplay::Block);
        assert_eq!(CssDisplay::from_str("none"), CssDisplay::None);
        assert_eq!(CssDisplay::from_str("flex"), CssDisplay::Flex);
        assert!(CssDisplay::Block.is_block_level());
        assert!(!CssDisplay::Inline.is_block_level());
    }

    #[test]
    fn test_font_weight() {
        assert!(CssFontWeight::BOLD.is_bold());
        assert!(!CssFontWeight::NORMAL.is_bold());
        assert_eq!(CssFontWeight::from_str("700").0, 700);
    }

    #[test]
    fn test_text_decoration() {
        let td = CssTextDecoration::from_str("underline line-through");
        assert!(td.underline);
        assert!(td.line_through);
        assert!(!td.overline);
    }

    #[test]
    fn test_hex_8digit_alpha() {
        let c = CssColor::from_hex("#ff000080").unwrap();
        assert_eq!(c.r, 1.0);
        assert!((c.a - 128.0/255.0).abs() < 0.01);
    }
}
