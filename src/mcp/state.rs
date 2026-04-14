//! MCP server shared state management.
//!
//! Provides the `ContextState` struct that holds all shared resources
//! for the MCP server including clients, configuration, and indexing status.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;

use crate::config::Config;
use crate::embedding::EmbeddingClient;
use crate::mcp::manifest::ManifestStore;
use crate::splitter::CodeSplitter;
use crate::types::{CodeChunk, EmbeddingVector, IndexState, IndexStatus};
use crate::vectordb::MilvusClient;

/// Search result returned from code search operations.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// The matched code chunk.
    pub chunk: CodeChunk,
    /// Similarity score (higher is better).
    pub score: f32,
}

/// Result of an indexing operation.
#[derive(Clone, Debug)]
pub struct IndexResult {
    /// Path that was indexed.
    pub path: PathBuf,
    /// Number of files processed.
    pub files_indexed: usize,
    /// Number of chunks created.
    pub chunks_created: usize,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
}

/// Result of clearing an index.
#[derive(Clone, Debug)]
pub struct ClearResult {
    /// Path that was cleared.
    pub path: PathBuf,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
}

/// Shared state for the MCP server.
///
/// This struct holds all resources needed by MCP tool operations.
/// It is designed to be wrapped in `Arc` for safe concurrent access.
pub struct ContextState {
    /// Server configuration.
    pub config: Config,
    /// Client for generating embeddings.
    pub embedding_client: EmbeddingClient,
    /// Client for Milvus vector database operations.
    pub milvus_client: MilvusClient,
    /// Machine-local manifest storage for completed full indexes.
    pub manifest_store: Arc<ManifestStore>,
    /// Current indexing status for each path being processed.
    pub indexing_status: DashMap<PathBuf, IndexStatus>,
    /// Code splitter for parsing and chunking source files.
    pub splitter: CodeSplitter,
}

impl ContextState {
    /// Create a new ContextState with the given configuration.
    ///
    /// Initializes all clients and resources needed for MCP operations.
    pub fn new(config: Config) -> Self {
        let embedding_config = crate::embedding::EmbeddingConfig {
            url: config.embedding_url.clone(),
            model: config.embedding_model.clone(),
            batch_size: config.batch_size,
        };
        let embedding_client = EmbeddingClient::new(embedding_config);
        let milvus_client = MilvusClient::new(&config.milvus_url);
        let splitter_config = crate::splitter::Config {
            max_chunk_bytes: config.chunk_size,
            overlap_lines: config.chunk_overlap / 80, // approximate lines from char overlap
            ..Default::default()
        };
        let splitter = CodeSplitter::new(splitter_config);

        Self {
            config,
            embedding_client,
            milvus_client,
            manifest_store: Arc::new(ManifestStore::default()),
            indexing_status: DashMap::new(),
            splitter,
        }
    }

    /// Create a new ContextState with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(Config::default())
    }

    /// Get the current indexing status for a path.
    ///
    /// Returns the default idle status if no indexing has been started for this path.
    pub fn get_status(&self, path: &Path) -> IndexStatus {
        self.indexing_status
            .get(&path.to_path_buf())
            .map(|r| r.clone())
            .unwrap_or_default()
    }

    /// Update the indexing status for a path.
    pub fn set_status(&self, path: PathBuf, status: IndexStatus) {
        self.indexing_status.insert(path, status);
    }

    /// Mark indexing as started for a path.
    pub fn start_indexing(&self, path: &Path, total_files: usize) {
        self.indexing_status.insert(
            path.to_path_buf(),
            IndexStatus {
                total_files,
                processed_files: 0,
                total_chunks: 0,
                status: IndexState::Indexing,
            },
        );
    }

    /// Update progress during indexing.
    pub fn update_progress(&self, path: &Path, processed_files: usize, total_chunks: usize) {
        if let Some(mut status) = self.indexing_status.get_mut(&path.to_path_buf()) {
            status.processed_files = processed_files;
            status.total_chunks = total_chunks;
        }
    }

    /// Mark indexing as completed for a path.
    pub fn complete_indexing(&self, path: &Path, total_chunks: usize) {
        if let Some(mut status) = self.indexing_status.get_mut(&path.to_path_buf()) {
            status.processed_files = status.total_files;
            status.total_chunks = total_chunks;
            status.status = IndexState::Completed;
        }
    }

    /// Mark indexing as failed for a path.
    pub fn fail_indexing(&self, path: &Path) {
        if let Some(mut status) = self.indexing_status.get_mut(&path.to_path_buf()) {
            status.status = IndexState::Failed;
        }
    }

    /// Check if indexing is currently in progress for a path.
    pub fn is_indexing(&self, path: &Path) -> bool {
        self.indexing_status
            .get(&path.to_path_buf())
            .map(|r| r.status == IndexState::Indexing)
            .unwrap_or(false)
    }

    /// Embed a single text string.
    pub async fn embed(&self, text: &str) -> Result<EmbeddingVector> {
        self.embedding_client.embed(text).await
    }

    /// Embed multiple texts in a batch.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        self.embedding_client.embed_batch(texts).await
    }

    /// Search for similar code chunks.
    ///
    /// Embeds the query and searches the vector database for matching chunks.
    pub async fn search(
        &self,
        collection: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embed(query).await?;
        let hits = self
            .milvus_client
            .search(collection, &query_embedding.vector, limit)
            .await?;

        Ok(hits
            .into_iter()
            .map(|hit| {
                // Extract metadata fields to reconstruct CodeChunk
                let file_path = hit
                    .metadata
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from)
                    .unwrap_or_default();
                let relative_path = hit
                    .metadata
                    .get("relative_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let start_line = hit
                    .metadata
                    .get("start_line")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                let end_line = hit
                    .metadata
                    .get("end_line")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                let language = hit
                    .metadata
                    .get("language")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                SearchResult {
                    chunk: CodeChunk {
                        id: hit.id,
                        content: hit.content,
                        file_path,
                        relative_path,
                        start_line,
                        end_line,
                        language,
                    },
                    score: hit.score,
                }
            })
            .collect())
    }

    /// Insert code chunks with their embeddings into the vector database.
    pub async fn insert_chunks(&self, collection: &str, chunks: &[CodeChunk]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let embeddings = self.embed_batch(&texts).await?;
        let vectors: Vec<Vec<f32>> = embeddings.into_iter().map(|e| e.vector).collect();

        self.milvus_client
            .insert_with_embeddings(collection, chunks.to_vec(), &vectors)
            .await
    }

    /// Ensure a collection exists for the given path.
    pub async fn ensure_collection(&self, path: &Path) -> Result<String> {
        let collection_name = crate::vectordb::collection_name_from_path(path);

        if !self.milvus_client.has_collection(&collection_name).await? {
            let dimension = self.config.embedding_dimension;
            self.milvus_client
                .create_collection(&collection_name, dimension)
                .await?;
        }

        Ok(collection_name)
    }

    /// Delete the collection for a path.
    pub async fn delete_collection(&self, path: &Path) -> Result<()> {
        let collection_name = crate::vectordb::collection_name_from_path(path);
        self.milvus_client.drop_collection(&collection_name).await
    }

    /// Split a file into code chunks.
    pub fn split_file(&self, path: &Path) -> Result<Vec<CodeChunk>> {
        self.splitter.split_file(path)
    }

    /// Split multiple files in parallel.
    pub fn split_files(&self, paths: &[PathBuf]) -> Result<Vec<CodeChunk>> {
        self.splitter.split_files(paths)
    }
}

/// Thread-safe shared state handle.
pub type SharedState = Arc<ContextState>;

/// Create a new shared state with the given configuration.
pub fn create_shared_state(config: Config) -> SharedState {
    Arc::new(ContextState::new(config))
}

/// Create a new shared state with default configuration.
pub fn create_default_shared_state() -> SharedState {
    Arc::new(ContextState::with_defaults())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_status_default() {
        let status = IndexStatus::default();
        assert_eq!(status.total_files, 0);
        assert_eq!(status.processed_files, 0);
        assert_eq!(status.total_chunks, 0);
        assert_eq!(status.status, IndexState::Idle);
    }
}
