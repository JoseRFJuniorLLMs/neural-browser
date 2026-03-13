//! Local AI model inference via ONNX Runtime + DirectML (NPU/GPU).
//!
//! Provides lightweight on-device inference without cloud API calls.
//! Falls back gracefully when no ONNX model file is present.
//!
//! To use: place an ONNX model at `models/phi3-mini.onnx` (or set LOCAL_MODEL_PATH).

use std::path::PathBuf;

/// Simple word-level tokenizer using FNV-1a hashing.
/// Maps words to token IDs in range 0..VOCAB_SIZE.
pub struct SimpleTokenizer {
    vocab_size: u32,
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

impl SimpleTokenizer {
    /// Create a new tokenizer with the given vocabulary size.
    pub fn new(vocab_size: u32) -> Self {
        Self { vocab_size }
    }

    /// Encode text into a sequence of token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|word| self.hash_word(word))
            .collect()
    }

    /// Decode token IDs back to approximate text.
    /// Since this is a hash-based tokenizer, exact recovery is not possible.
    /// Returns token ID representations instead.
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .map(|t| format!("[{t}]"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Hash a word to a token ID using FNV-1a.
    fn hash_word(&self, word: &str) -> u32 {
        let mut hash = FNV_OFFSET;
        for byte in word.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        (hash % self.vocab_size as u64) as u32
    }
}

/// Local ONNX model for on-device inference.
pub struct LocalModel {
    model_path: PathBuf,
    max_tokens: usize,
    loaded: bool,
}

impl LocalModel {
    /// Create from explicit path.
    pub fn new(model_path: &str) -> Self {
        let max_tokens = std::env::var("LOCAL_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);

        Self {
            model_path: PathBuf::from(model_path),
            max_tokens,
            loaded: false,
        }
    }

    /// Create from environment variable LOCAL_MODEL_PATH.
    pub fn from_env() -> Self {
        let path = std::env::var("LOCAL_MODEL_PATH")
            .unwrap_or_else(|_| "models/phi3-mini.onnx".to_string());
        Self::new(&path)
    }

    /// Check if the model file exists on disk.
    pub fn model_available(&self) -> bool {
        self.model_path.exists()
    }

    /// Check if the model has been loaded into memory.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Attempt to load the model.
    /// Returns Ok if model loaded or Err with a helpful message.
    pub fn load(&mut self) -> Result<(), String> {
        if !self.model_available() {
            return Err(format!(
                "No ONNX model found at '{}'. Place a model file there or set LOCAL_MODEL_PATH.",
                self.model_path.display()
            ));
        }

        // TODO: Load ONNX model via ort crate with DirectML EP
        // let env = ort::Environment::builder()
        //     .with_execution_providers([ort::DirectMLExecutionProvider::default()])
        //     .build()?;
        // let session = env.new_session_builder()?
        //     .with_optimization_level(GraphOptimizationLevel::Level3)?
        //     .with_model_from_file(&self.model_path)?;

        self.loaded = true;
        Ok(())
    }

    /// Generate text from a prompt.
    /// Returns the model's response or an error message.
    pub fn generate(&mut self, prompt: &str, max_tokens: Option<usize>) -> Result<String, String> {
        let _max = max_tokens.unwrap_or(self.max_tokens);

        if !self.loaded {
            self.load()?;
        }

        if !self.model_available() {
            return Err(format!(
                "Local AI model not available. To use local inference:\n\
                 1. Download an ONNX model (e.g., Phi-3 Mini)\n\
                 2. Place it at: {}\n\
                 3. Restart Neural Browser",
                self.model_path.display()
            ));
        }

        // TODO: Run actual ONNX inference with DirectML
        // For now, return a stub response indicating the model would run here
        let _tokenizer = SimpleTokenizer::new(32000);
        let _tokens = _tokenizer.encode(prompt);

        Err(format!(
            "ONNX Runtime integration pending. Model at '{}' detected ({} bytes). \
             Install ort crate to enable local inference.",
            self.model_path.display(),
            std::fs::metadata(&self.model_path).map(|m| m.len()).unwrap_or(0),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_encode_empty() {
        let tok = SimpleTokenizer::new(32000);
        assert!(tok.encode("").is_empty());
    }

    #[test]
    fn test_tokenizer_encode_words() {
        let tok = SimpleTokenizer::new(32000);
        let tokens = tok.encode("hello world test");
        assert_eq!(tokens.len(), 3);
        assert!(tokens.iter().all(|&t| t < 32000));
    }

    #[test]
    fn test_tokenizer_deterministic() {
        let tok = SimpleTokenizer::new(32000);
        let t1 = tok.encode("neural browser");
        let t2 = tok.encode("neural browser");
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_tokenizer_different_words_different_tokens() {
        let tok = SimpleTokenizer::new(32000);
        let t1 = tok.encode("hello");
        let t2 = tok.encode("world");
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_tokenizer_decode() {
        let tok = SimpleTokenizer::new(32000);
        let decoded = tok.decode(&[100, 200, 300]);
        assert_eq!(decoded, "[100] [200] [300]");
    }

    #[test]
    fn test_tokenizer_vocab_range() {
        let tok = SimpleTokenizer::new(100);
        let tokens = tok.encode("the quick brown fox jumps over the lazy dog");
        assert!(tokens.iter().all(|&t| t < 100));
    }

    #[test]
    fn test_local_model_no_file() {
        let model = LocalModel::new("nonexistent_model.onnx");
        assert!(!model.model_available());
        assert!(!model.is_loaded());
    }

    #[test]
    fn test_local_model_load_missing() {
        let mut model = LocalModel::new("nonexistent_model.onnx");
        let result = model.load();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No ONNX model found"));
    }

    #[test]
    fn test_local_model_generate_missing() {
        let mut model = LocalModel::new("nonexistent_model.onnx");
        let result = model.generate("Hello", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_local_model_from_env() {
        // Without env var, should default to models/phi3-mini.onnx
        let model = LocalModel::from_env();
        assert!(!model.is_loaded());
    }

    #[test]
    fn test_tokenizer_unicode() {
        let tok = SimpleTokenizer::new(32000);
        let tokens = tok.encode("日本語 テスト 🦀 Rust");
        assert_eq!(tokens.len(), 4);
        assert!(tokens.iter().all(|&t| t < 32000));
    }

    #[test]
    fn test_fnv_hash_consistency() {
        let tok = SimpleTokenizer::new(32000);
        // Same word always gets same hash
        let h1 = tok.encode("consistency");
        let h2 = tok.encode("consistency");
        assert_eq!(h1, h2);
    }
}
