//! MCP server shared state management.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;

use crate::config::Config;
use crate::embedding::{Embedder, EmbeddingClient, EmbeddingConfig};
use crate::mcp::manifest::ManifestStore;
use crate::splitter::CodeSplitter;
use crate::types::{CodeChunk, EmbeddingVector, IndexState, IndexStatus};
use crate::vectordb::{LocalStore, MilvusClient, VectorStore};

/// Search result returned from code search operations.
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub chunk: CodeChunk,
    pub score: f32,
}

/// Result of an indexing operation.
#[derive(Clone, Debug)]
pub struct IndexResult {
    pub path: PathBuf,
    pub files_indexed: usize,
    pub chunks_created: usize,
    pub success: bool,
    pub error: Option<String>,
}

/// Result of clearing an index.
#[derive(Clone, Debug)]
pub struct ClearResult {
    pub path: PathBuf,
    pub success: bool,
    pub error: Option<String>,
}

/// Shared state for the MCP server.
pub struct ContextState {
    pub config: Config,
    pub embedder: Embedder,
    pub vector_store: VectorStore,
    pub manifest_store: Arc<ManifestStore>,
    pub indexing_status: DashMap<PathBuf, IndexStatus>,
    pub splitter: CodeSplitter,
}

impl ContextState {
    pub fn new(config: Config) -> Self {
        let embedding_explicitly_set = std::env::var("EMBEDDING_URL")
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        let milvus_explicitly_set = std::env::var("MILVUS_URL")
            .map(|v| !v.is_empty())
            .unwrap_or(false);

        let embedder = if embedding_explicitly_set {
            let embedding_config = EmbeddingConfig::from_config(&config);
            Embedder::Http(EmbeddingClient::new(embedding_config))
        } else {
            Embedder::Disabled
        };

        let vector_store = if milvus_explicitly_set {
            VectorStore::Milvus(MilvusClient::new(&config.milvus_url, config.milvus_token.clone()))
        } else {
            VectorStore::Local(LocalStore::new())
        };

        let splitter_config = crate::splitter::Config {
            max_chunk_bytes: config.chunk_size,
            overlap_lines: config.chunk_overlap / 80,
            ..Default::default()
        };
        let splitter = CodeSplitter::new(splitter_config);

        Self {
            config,
            embedder,
            vector_store,
            manifest_store: Arc::new(ManifestStore),
            indexing_status: DashMap::new(),
            splitter,
        }
    }

    pub fn with_components(
        config: Config,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> Self {
        let splitter_config = crate::splitter::Config {
            max_chunk_bytes: config.chunk_size,
            overlap_lines: config.chunk_overlap / 80,
            ..Default::default()
        };
        let splitter = CodeSplitter::new(splitter_config);

        Self {
            config,
            embedder,
            vector_store,
            manifest_store: Arc::new(ManifestStore),
            indexing_status: DashMap::new(),
            splitter,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(Config::default())
    }

    pub fn get_status(&self, path: &Path) -> IndexStatus {
        self.indexing_status
            .get(&path.to_path_buf())
            .map(|r| r.clone())
            .or_else(|| self.manifest_store.load_status(path).ok().flatten())
            .unwrap_or_default()
    }

    pub fn set_status(&self, path: PathBuf, status: IndexStatus) {
        self.indexing_status.insert(path.clone(), status.clone());
        let _ = self.manifest_store.write_status(&path, &status);
    }

    pub fn start_indexing(&self, path: &Path, total_files: usize) {
        self.set_status(
            path.to_path_buf(),
            IndexStatus {
                total_files,
                processed_files: 0,
                total_chunks: 0,
                embeddings_generated: 0,
                vectors_inserted: 0,
                status: IndexState::Indexing,
            },
        );
    }

    pub fn update_progress(&self, path: &Path, processed_files: usize, total_chunks: usize) {
        if let Some(mut status) = self.indexing_status.get_mut(&path.to_path_buf()) {
            status.processed_files = processed_files;
            status.total_chunks = total_chunks;
            let snapshot = status.clone();
            drop(status);
            let _ = self.manifest_store.write_status(path, &snapshot);
        }
    }

    pub fn complete_indexing(&self, path: &Path, total_chunks: usize) {
        if let Some(mut status) = self.indexing_status.get_mut(&path.to_path_buf()) {
            status.processed_files = status.total_files;
            status.total_chunks = total_chunks;
            status.status = IndexState::Completed;
            let snapshot = status.clone();
            drop(status);
            let _ = self.manifest_store.write_status(path, &snapshot);
        }
    }

    pub fn fail_indexing(&self, path: &Path) {
        if let Some(mut status) = self.indexing_status.get_mut(&path.to_path_buf()) {
            status.status = IndexState::Failed;
            let snapshot = status.clone();
            drop(status);
            let _ = self.manifest_store.write_status(path, &snapshot);
        }
    }

    pub fn is_indexing(&self, path: &Path) -> bool {
        self.indexing_status
            .get(&path.to_path_buf())
            .map(|r| r.status == IndexState::Indexing)
            .unwrap_or(false)
    }

    pub async fn embed(&self, text: &str) -> Result<EmbeddingVector> {
        self.embedder.embed(text).await
    }

    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        self.embedder.embed_batch(texts).await
    }

    pub async fn search(
        &self,
        collection: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embed(query).await?;
        let hits = self
            .vector_store
            .search(collection, &query_embedding.vector, limit)
            .await?;

        Ok(hits
            .into_iter()
            .map(|hit| {
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

    pub async fn ensure_collection(&self, path: &Path) -> Result<String> {
        let collection_name = crate::vectordb::collection_name_from_path(path);

        if !self.vector_store.has_collection(&collection_name).await? {
            let dimension = self.config.embedding_dimension;
            self.vector_store
                .create_collection(&collection_name, dimension)
                .await?;
        }

        Ok(collection_name)
    }

    pub async fn delete_collection(&self, path: &Path) -> Result<()> {
        let collection_name = crate::vectordb::collection_name_from_path(path);
        self.vector_store.drop_collection(&collection_name).await
    }

    pub fn split_file(&self, path: &Path) -> Result<Vec<CodeChunk>> {
        self.splitter.split_file(path)
    }

    pub fn split_files(&self, paths: &[PathBuf]) -> Result<Vec<CodeChunk>> {
        self.splitter.split_files(paths)
    }
}

/// Thread-safe shared state handle.
pub type SharedState = Arc<ContextState>;

pub fn create_shared_state(config: Config) -> SharedState {
    Arc::new(ContextState::new(config))
}

pub fn create_shared_state_with_components(
    config: Config,
    embedder: Embedder,
    vector_store: VectorStore,
) -> SharedState {
    Arc::new(ContextState::with_components(config, embedder, vector_store))
}

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
