//! High-performance parallel codebase indexer.
//!
//! This module provides the critical path for indexing codebases:
//! 1. Walk files in parallel using ignore-based walker
//! 2. Split files into chunks using rayon for CPU parallelism
//! 3. Batch embed chunks (100 at a time) for efficient GPU/API utilization
//! 4. Insert into Milvus in batches for network efficiency

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use rayon::prelude::*;
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};

use crate::config::Config;
use crate::embedding::EmbeddingClient;
use crate::mcp::manifest::ManifestStore;
use crate::splitter::CodeSplitter;
use crate::types::{CodeChunk, IndexState, IndexStatus};
use crate::vectordb::{collection_name_from_path, InsertRow, MilvusClient};
use crate::walker::CodeWalker;

/// Batch size for embedding requests - balances throughput vs memory.
const EMBEDDING_BATCH_SIZE: usize = 100;

/// Batch size for Milvus insertions - larger batches are more efficient.
const MILVUS_BATCH_SIZE: usize = 500;

/// Number of files to process in parallel during splitting phase.
const FILE_PARALLEL_CHUNK_SIZE: usize = 64;

/// Result of a completed indexing operation.
#[derive(Debug, Clone)]
pub struct IndexResult {
    /// Total number of files processed.
    pub files_processed: usize,
    /// Total number of chunks created.
    pub chunks_created: usize,
    /// Total number of embeddings generated.
    pub embeddings_generated: usize,
    /// Total number of vectors inserted into Milvus.
    pub vectors_inserted: usize,
    /// Time taken for the entire operation.
    pub duration_ms: u64,
    /// Any warnings encountered during indexing.
    pub warnings: Vec<String>,
}

/// Shared state for the context server, including indexing status.
pub struct ContextState {
    /// Configuration used for this indexing run.
    pub config: Arc<Config>,
    /// Current indexing status, updated throughout the process.
    pub indexing_status: Arc<RwLock<IndexStatus>>,
    /// Code walker for discovering files.
    pub walker: Arc<CodeWalker>,
    /// Code splitter for chunking files.
    pub splitter: Arc<CodeSplitter>,
    /// Embedding client for generating vectors.
    pub embedding_client: Arc<EmbeddingClient>,
    /// Milvus client for vector storage.
    pub milvus_client: Arc<MilvusClient>,
    /// Machine-local manifest persistence for completed full indexes.
    pub manifest_store: Arc<ManifestStore>,
}

impl ContextState {
    /// Create a new context state with the given components.
    pub fn new(
        config: Config,
        walker: CodeWalker,
        splitter: CodeSplitter,
        embedding_client: EmbeddingClient,
        milvus_client: MilvusClient,
        manifest_store: Arc<ManifestStore>,
    ) -> Self {
        Self {
            config: Arc::new(config),
            indexing_status: Arc::new(RwLock::new(IndexStatus::default())),
            walker: Arc::new(walker),
            splitter: Arc::new(splitter),
            embedding_client: Arc::new(embedding_client),
            milvus_client: Arc::new(milvus_client),
            manifest_store,
        }
    }

    /// Get the current indexing status.
    pub async fn get_status(&self) -> IndexStatus {
        self.indexing_status.read().await.clone()
    }
}

/// Index a codebase at the given path.
///
/// This is the critical path - optimized for maximum parallelism:
/// - File discovery runs in parallel threads via ignore crate
/// - File splitting uses rayon for CPU-bound parallel processing
/// - Embedding happens in batches of 100 for efficient API/GPU usage
/// - Milvus insertion uses larger batches of 500 for network efficiency
///
/// # Arguments
/// * `state` - Shared context state with clients and status
/// * `path` - Root path of the codebase to index
/// * `force` - If true, re-index even if already indexed
///
/// # Returns
/// * `Ok(IndexResult)` - Summary of the indexing operation
/// * `Err` - If any critical error occurs
#[instrument(skip(state), fields(path = %path.display()))]
pub async fn index_codebase(state: &ContextState, path: &Path, force: bool) -> Result<IndexResult> {
    let start = Instant::now();
    let mut warnings = Vec::new();

    // Check if already indexing
    {
        let status = state.indexing_status.read().await;
        if status.status == IndexState::Indexing && !force {
            anyhow::bail!("Indexing already in progress");
        }
    }

    // Initialize status
    {
        let mut status = state.indexing_status.write().await;
        *status = IndexStatus {
            total_files: 0,
            processed_files: 0,
            total_chunks: 0,
            status: IndexState::Indexing,
        };
    }

    info!("Starting codebase indexing at {}", path.display());

    // Phase 1: Walk files (parallel via ignore crate)
    let files = match state.walker.walk(path).await {
        Ok(files) => files,
        Err(e) => {
            update_status_failed(state).await;
            return Err(e).context("Failed to walk codebase");
        }
    };

    let total_files = files.len();
    info!("Discovered {} files to index", total_files);
    let collection_name = collection_name_from_path(path);

    {
        let mut status = state.indexing_status.write().await;
        status.total_files = total_files;
    }

    if total_files == 0 {
        state
            .manifest_store
            .write_for_files(path, &collection_name, &state.config, &files)
            .with_context(|| format!("Failed to write manifest for {}", path.display()))?;
        update_status_completed(state, 0).await;
        return Ok(IndexResult {
            files_processed: 0,
            chunks_created: 0,
            embeddings_generated: 0,
            vectors_inserted: 0,
            duration_ms: start.elapsed().as_millis() as u64,
            warnings,
        });
    }

    // Phase 2: Split files into chunks using rayon (CPU-parallel)
    let processed_files = Arc::new(AtomicUsize::new(0));
    let splitter = state.splitter.clone();
    let status_ref = state.indexing_status.clone();
    let processed_ref = processed_files.clone();

    // Use rayon's parallel iterator for CPU-bound splitting
    let chunk_results: Vec<Result<Vec<CodeChunk>, String>> = files
        .par_chunks(FILE_PARALLEL_CHUNK_SIZE)
        .flat_map(|file_batch| {
            file_batch.par_iter().map(|file_path| {
                match splitter.split_file(file_path) {
                    Ok(chunks) => {
                        let count = processed_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        // Update status periodically (every 10 files to reduce lock contention)
                        if count % 10 == 0 {
                            // We can't await in rayon, so we use try_write
                            if let Ok(mut status) = status_ref.try_write() {
                                status.processed_files = count;
                            }
                        }
                        Ok(chunks)
                    }
                    Err(e) => {
                        debug!("Failed to split {}: {}", file_path.display(), e);
                        Err(format!("Failed to split {}: {}", file_path.display(), e))
                    }
                }
            })
        })
        .collect();

    // Collect chunks and warnings
    let mut all_chunks = Vec::new();
    for result in chunk_results {
        match result {
            Ok(chunks) => all_chunks.extend(chunks),
            Err(warning) => warnings.push(warning),
        }
    }

    let total_chunks = all_chunks.len();
    info!(
        "Split {} files into {} chunks ({} warnings)",
        total_files,
        total_chunks,
        warnings.len()
    );

    {
        let mut status = state.indexing_status.write().await;
        status.processed_files = total_files;
        status.total_chunks = total_chunks;
    }

    if total_chunks == 0 {
        state
            .manifest_store
            .write_for_files(path, &collection_name, &state.config, &files)
            .with_context(|| format!("Failed to write manifest for {}", path.display()))?;
        update_status_completed(state, 0).await;
        return Ok(IndexResult {
            files_processed: total_files,
            chunks_created: 0,
            embeddings_generated: 0,
            vectors_inserted: 0,
            duration_ms: start.elapsed().as_millis() as u64,
            warnings,
        });
    }

    // Phase 3: Batch embed chunks (100 at a time for API efficiency)
    let mut all_embeddings = Vec::with_capacity(total_chunks);
    let embedding_client = state.embedding_client.clone();

    // Process embeddings in batches using tokio for async parallelism
    let chunk_batches: Vec<_> = all_chunks.chunks(EMBEDDING_BATCH_SIZE).collect();
    let num_embedding_batches = chunk_batches.len();
    info!(
        "Generating embeddings in {} batches of {}",
        num_embedding_batches, EMBEDDING_BATCH_SIZE
    );

    // Process embedding batches with controlled concurrency
    // Using tokio::spawn for true async parallelism on embedding API calls
    let semaphore = Arc::new(tokio::sync::Semaphore::new(4)); // Limit concurrent API calls
    let mut embedding_handles = Vec::with_capacity(num_embedding_batches);

    for batch in chunk_batches {
        let batch_texts: Vec<String> = batch.iter().map(|c| c.content.clone()).collect();
        let client = embedding_client.clone();
        let permit = semaphore.clone().acquire_owned().await?;

        let handle = tokio::spawn(async move {
            let result = client.embed_batch(&batch_texts).await;
            drop(permit); // Release semaphore
            result
        });
        embedding_handles.push(handle);
    }

    // Collect embedding results
    let mut embeddings_generated = 0;
    for handle in embedding_handles {
        match handle.await {
            Ok(Ok(embeddings)) => {
                embeddings_generated += embeddings.len();
                all_embeddings.extend(embeddings);
            }
            Ok(Err(e)) => {
                warn!("Embedding batch failed: {}", e);
                warnings.push(format!("Embedding batch failed: {}", e));
            }
            Err(e) => {
                warn!("Embedding task panicked: {}", e);
                warnings.push(format!("Embedding task panicked: {}", e));
            }
        }
    }

    info!("Generated {} embeddings", embeddings_generated);

    if all_embeddings.is_empty() {
        update_status_failed(state).await;
        anyhow::bail!("Failed to generate any embeddings");
    }

    // Phase 4: Batch insert into Milvus
    let milvus_client = state.milvus_client.clone();
    let mut vectors_inserted = 0;

    info!("Using collection: {}", collection_name);

    // Convert chunks + embeddings to InsertRows
    let insert_rows: Vec<InsertRow> = all_chunks
        .into_iter()
        .zip(all_embeddings.into_iter())
        .map(|(chunk, embedding)| InsertRow {
            id: chunk.id,
            content: chunk.content.clone(),
            vector: embedding.vector,
            metadata: serde_json::json!({
                "file_path": chunk.file_path,
                "relative_path": chunk.relative_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "language": chunk.language,
            }),
        })
        .collect();

    // Insert in larger batches for network efficiency
    let insert_batches: Vec<&[InsertRow]> = insert_rows.chunks(MILVUS_BATCH_SIZE).collect();

    let num_insert_batches = insert_batches.len();
    info!(
        "Inserting into Milvus in {} batches of {}",
        num_insert_batches, MILVUS_BATCH_SIZE
    );

    // Use tokio::spawn for parallel Milvus insertions with controlled concurrency
    let insert_semaphore = Arc::new(tokio::sync::Semaphore::new(2)); // Limit concurrent DB writes
    let mut insert_handles = Vec::with_capacity(num_insert_batches);

    for batch in insert_batches {
        let batch_data: Vec<InsertRow> = batch.to_vec();
        let client = milvus_client.clone();
        let collection = collection_name.clone();
        let permit = insert_semaphore.clone().acquire_owned().await?;

        let handle = tokio::spawn(async move {
            let result = client.insert_batch(&collection, &batch_data).await;
            drop(permit);
            result.map(|_| batch_data.len())
        });
        insert_handles.push(handle);
    }

    // Collect insertion results
    for handle in insert_handles {
        match handle.await {
            Ok(Ok(count)) => {
                vectors_inserted += count;
            }
            Ok(Err(e)) => {
                error!("Milvus insertion failed: {}", e);
                warnings.push(format!("Milvus insertion failed: {}", e));
            }
            Err(e) => {
                error!("Milvus insertion task panicked: {}", e);
                warnings.push(format!("Milvus insertion task panicked: {}", e));
            }
        }
    }

    info!("Inserted {} vectors into Milvus", vectors_inserted);

    state
        .manifest_store
        .write_for_files(path, &collection_name, &state.config, &files)
        .with_context(|| format!("Failed to write manifest for {}", path.display()))?;

    // Update final status
    update_status_completed(state, total_chunks).await;

    let duration_ms = start.elapsed().as_millis() as u64;
    info!(
        "Indexing completed in {}ms: {} files, {} chunks, {} vectors",
        duration_ms, total_files, total_chunks, vectors_inserted
    );

    Ok(IndexResult {
        files_processed: total_files,
        chunks_created: total_chunks,
        embeddings_generated,
        vectors_inserted,
        duration_ms,
        warnings,
    })
}

/// Spawn background indexing task.
///
/// Returns immediately with a handle that can be used to await completion.
pub fn spawn_index_codebase(
    state: Arc<ContextState>,
    path: std::path::PathBuf,
    force: bool,
) -> tokio::task::JoinHandle<Result<IndexResult>> {
    tokio::spawn(async move { index_codebase(&state, &path, force).await })
}

async fn update_status_failed(state: &ContextState) {
    let mut status = state.indexing_status.write().await;
    status.status = IndexState::Failed;
}

async fn update_status_completed(state: &ContextState, chunks: usize) {
    let mut status = state.indexing_status.write().await;
    status.total_chunks = chunks;
    status.status = IndexState::Completed;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_index_result_default() {
        let result = IndexResult {
            files_processed: 0,
            chunks_created: 0,
            embeddings_generated: 0,
            vectors_inserted: 0,
            duration_ms: 0,
            warnings: vec![],
        };
        assert_eq!(result.files_processed, 0);
    }
}
