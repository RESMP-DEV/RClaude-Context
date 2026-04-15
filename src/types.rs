use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A chunk of source code extracted from a file for embedding and indexing.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CodeChunk {
    /// Unique identifier for this chunk.
    pub id: String,
    /// The source code content of this chunk.
    pub content: String,
    /// Absolute path to the source file.
    pub file_path: PathBuf,
    /// Path relative to the repository root.
    pub relative_path: String,
    /// Starting line number (1-indexed).
    pub start_line: u32,
    /// Ending line number (1-indexed, inclusive).
    pub end_line: u32,
    /// Programming language identifier (e.g., "rust", "python").
    pub language: String,
}

/// A vector embedding with its dimensionality.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct EmbeddingVector {
    /// The embedding values.
    pub vector: Vec<f32>,
    /// Dimensionality of the embedding.
    pub dimension: usize,
}

/// Current state of the indexing process.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum IndexState {
    /// No indexing in progress.
    #[default]
    Idle,
    /// Currently indexing files.
    Indexing,
    /// Indexing completed successfully.
    Completed,
    /// Indexing failed.
    Failed,
}

/// Status of the code index.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct IndexStatus {
    /// Total number of files to process.
    pub total_files: usize,
    /// Number of files processed so far.
    pub processed_files: usize,
    /// Total number of code chunks created.
    pub total_chunks: usize,
    /// Number of embeddings generated so far.
    #[serde(default)]
    pub embeddings_generated: usize,
    /// Number of vectors inserted into Milvus so far.
    #[serde(default)]
    pub vectors_inserted: usize,
    /// Current indexing state.
    pub status: IndexState,
}
