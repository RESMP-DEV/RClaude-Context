use std::collections::HashSet;
use std::env;

/// Supported file extensions for indexing (without leading dot).
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "py", "rs", "js", "ts", "tsx", "jsx", "go", "java", "cpp", "cc", "cxx", "c", "h", "hpp", "rb",
    "php", "swift", "kt", "scala", "cs", "fs", "ml", "mli", "hs", "lua", "sh", "bash", "zsh", "pl",
    "pm", "r", "jl", "ex", "exs", "erl", "hrl", "clj", "cljs", "lisp", "el", "vim", "sql",
    "graphql", "proto", "thrift", "yaml", "yml", "toml", "json", "xml", "html", "css", "scss",
    "sass", "less", "md", "rst", "tex",
];

/// Extensionless files to include.
pub const EXTENSIONLESS_FILES: &[&str] = &["dockerfile", "makefile", "justfile", "rakefile"];

/// Default patterns to ignore during directory traversal.
pub const DEFAULT_IGNORE_PATTERNS: &[&str] = &[
    "node_modules",
    ".git",
    ".hg",
    ".svn",
    "target",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    ".tox",
    ".eggs",
];

/// Application configuration loaded from environment variables with sensible defaults.
#[derive(Debug, Clone)]
pub struct Config {
    /// URL for the embedding service.
    pub embedding_url: String,
    /// Model name for embeddings.
    pub embedding_model: String,
    /// URL for Milvus vector database.
    pub milvus_url: String,
    /// Size of text chunks in characters.
    pub chunk_size: usize,
    /// Overlap between adjacent chunks in characters.
    pub chunk_overlap: usize,
    /// Number of items per batch for bulk operations.
    pub batch_size: usize,
    /// Maximum concurrent operations.
    pub concurrency: usize,
    /// Maximum file size in bytes to process (0 = unlimited).
    pub max_file_size: u64,
    /// Whether to follow symbolic links during traversal.
    pub follow_symlinks: bool,
    /// Number of threads for parallel operations (0 = auto-detect).
    pub parallelism: usize,
    /// Embedding vector dimension.
    pub embedding_dimension: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embedding_url: "http://localhost:8100".to_string(),
            embedding_model: "all-minilm".to_string(),
            milvus_url: "http://localhost:19530".to_string(),
            chunk_size: 512,
            chunk_overlap: 64,
            batch_size: 32,
            concurrency: 4,
            max_file_size: 1024 * 1024, // 1 MB
            follow_symlinks: false,
            parallelism: 0,
            embedding_dimension: 1024,
        }
    }
}

impl Config {
    /// Load configuration from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        let defaults = Self::default();

        Self {
            embedding_url: env::var("EMBEDDING_URL").unwrap_or(defaults.embedding_url),
            embedding_model: env::var("EMBEDDING_MODEL").unwrap_or(defaults.embedding_model),
            milvus_url: env::var("MILVUS_URL").unwrap_or(defaults.milvus_url),
            chunk_size: env::var("CHUNK_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.chunk_size),
            chunk_overlap: env::var("CHUNK_OVERLAP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.chunk_overlap),
            batch_size: env::var("BATCH_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.batch_size),
            concurrency: env::var("CONCURRENCY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.concurrency),
            max_file_size: env::var("MAX_FILE_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.max_file_size),
            follow_symlinks: env::var("FOLLOW_SYMLINKS")
                .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
                .unwrap_or(defaults.follow_symlinks),
            parallelism: env::var("PARALLELISM")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.parallelism),
            embedding_dimension: env::var("EMBEDDING_DIMENSION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.embedding_dimension),
        }
    }

    /// Get effective thread count for parallel operations.
    #[inline]
    pub fn thread_count(&self) -> usize {
        if self.parallelism == 0 {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        } else {
            self.parallelism
        }
    }

    /// Get supported extensions as a HashSet for O(1) lookup.
    pub fn extension_set(&self) -> HashSet<&'static str> {
        SUPPORTED_EXTENSIONS.iter().copied().collect()
    }

    /// Get extensionless files as a HashSet for O(1) lookup.
    pub fn extensionless_set(&self) -> HashSet<&'static str> {
        EXTENSIONLESS_FILES.iter().copied().collect()
    }

    /// Get skip directories as a HashSet for O(1) lookup.
    pub fn skip_dirs_set(&self) -> HashSet<&'static str> {
        DEFAULT_IGNORE_PATTERNS.iter().copied().collect()
    }

    /// Check if a file extension should be included.
    #[inline]
    pub fn should_include_extension(&self, ext: &str) -> bool {
        let ext_lower = ext.to_lowercase();
        SUPPORTED_EXTENSIONS.contains(&ext_lower.as_str())
    }

    /// Check if an extensionless filename should be included.
    #[inline]
    pub fn should_include_extensionless(&self, name: &str) -> bool {
        let name_lower = name.to_lowercase();
        EXTENSIONLESS_FILES.contains(&name_lower.as_str())
    }
}
