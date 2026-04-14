pub mod extractor;
pub mod node_types;
pub mod parsers;
pub mod refine;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;
use tracing::{debug, warn};

use crate::types::CodeChunk;

pub use extractor::extract_chunks;
pub use node_types::{get_splittable_nodes, is_splittable, supported_languages, SPLITTABLE_NODES};
pub use parsers::{extension_to_language, get_parser, get_parser_for_extension, LanguageParser};
pub use refine::refine_chunks;

/// Configuration for the code splitter.
#[derive(Clone, Debug)]
pub struct Config {
    /// Minimum chunk size in lines.
    pub min_chunk_lines: u32,
    /// Maximum chunk size in bytes for refinement.
    pub max_chunk_bytes: usize,
    /// Target chunk size in lines (for splitting large constructs).
    pub target_chunk_lines: u32,
    /// Number of lines to overlap between split chunks.
    pub overlap_lines: usize,
    /// Root path for computing relative paths.
    pub root_path: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            min_chunk_lines: 5,
            max_chunk_bytes: 4096,
            target_chunk_lines: 50,
            overlap_lines: 3,
            root_path: PathBuf::from("."),
        }
    }
}

/// Splits source code files into semantic chunks using tree-sitter parsing.
pub struct CodeSplitter {
    config: Config,
}

impl CodeSplitter {
    /// Create a new code splitter with the given configuration.
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Split a single file into code chunks.
    pub fn split_file(&self, path: &Path) -> Result<Vec<CodeChunk>> {
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let language = match extension_to_language(extension) {
            Some(lang) => lang,
            None => {
                debug!("Unsupported file extension: {:?}", path);
                return Ok(vec![]);
            }
        };

        let source = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        if source.is_empty() {
            return Ok(vec![]);
        }

        let mut parser = get_parser(language)
            .with_context(|| format!("No parser available for language: {}", language))?;

        let tree = parser
            .parse(&source, None)
            .with_context(|| format!("Failed to parse file: {}", path.display()))?;

        let node_types = get_splittable_nodes(language).unwrap_or(&[]);

        let relative_path = path
            .strip_prefix(&self.config.root_path)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        let raw_chunks = extract_chunks(&tree, &source, node_types, path, &relative_path, language);

        let refined_chunks = refine_chunks(
            raw_chunks,
            self.config.max_chunk_bytes,
            self.config.overlap_lines,
        );

        Ok(refined_chunks)
    }

    /// Split multiple files in parallel using rayon.
    pub fn split_files(&self, paths: &[PathBuf]) -> Result<Vec<CodeChunk>> {
        let results: Vec<Result<Vec<CodeChunk>>> =
            paths.par_iter().map(|path| self.split_file(path)).collect();

        let mut all_chunks = Vec::new();
        for result in results {
            match result {
                Ok(chunks) => all_chunks.extend(chunks),
                Err(e) => {
                    warn!("Failed to split file: {}", e);
                }
            }
        }

        Ok(all_chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.min_chunk_lines, 5);
        assert_eq!(config.max_chunk_bytes, 4096);
        assert_eq!(config.target_chunk_lines, 50);
        assert_eq!(config.overlap_lines, 3);
    }

    #[test]
    fn test_code_splitter_new() {
        let config = Config::default();
        let _splitter = CodeSplitter::new(config);
    }

    #[test]
    fn test_split_rust_file() {
        let source = r#"
fn hello() {
    println!("Hello, world!");
}

fn goodbye() {
    println!("Goodbye!");
}
"#;
        let mut temp = NamedTempFile::with_suffix(".rs").unwrap();
        temp.write_all(source.as_bytes()).unwrap();

        let config = Config::default();
        let splitter = CodeSplitter::new(config);
        let chunks = splitter.split_file(temp.path()).unwrap();

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("hello"));
        assert!(chunks[1].content.contains("goodbye"));
    }

    #[test]
    fn test_split_unsupported_extension() {
        let mut temp = NamedTempFile::with_suffix(".xyz").unwrap();
        temp.write_all(b"some content").unwrap();

        let config = Config::default();
        let splitter = CodeSplitter::new(config);
        let chunks = splitter.split_file(temp.path()).unwrap();

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_split_empty_file() {
        let temp = NamedTempFile::with_suffix(".rs").unwrap();

        let config = Config::default();
        let splitter = CodeSplitter::new(config);
        let chunks = splitter.split_file(temp.path()).unwrap();

        assert!(chunks.is_empty());
    }
}
