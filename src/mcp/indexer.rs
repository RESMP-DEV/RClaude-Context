//! High-performance parallel codebase indexer.
//!
//! This module provides the critical path for indexing codebases:
//! 1. Walk files in parallel using ignore-based walker
//! 2. Split files into chunks using rayon for CPU parallelism
//! 3. Batch embed chunks (100 at a time) for efficient GPU/API utilization
//! 4. Stream inserts into Milvus in small batches instead of buffering the
//!    whole embedding set in memory first

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use futures::stream::{FuturesUnordered, StreamExt};
use rayon::prelude::*;
use tokio::sync::RwLock;
use tokio::task;
use tracing::{debug, info, instrument, warn};

use super::manifest::{diff_manifest_against_files, IndexInputs, ManifestStore};
use crate::embedding::EmbeddingClient;
use crate::lexical::LexicalIndex;
use crate::splitter::CodeSplitter;
use crate::types::{CodeChunk, IndexState, IndexStatus};
use crate::vectordb::{collection_name_from_path, InsertRow, MilvusClient};
use crate::vectordb::client::milvus_id_for_chunk_id;
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

/// Shared state for the indexer, including indexing status.
pub struct IndexerState {
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
    /// Manifest store for incremental indexing.
    pub manifest_store: ManifestStore,
    /// Embedding vector dimension used when creating Milvus collections.
    pub embedding_dimension: usize,
}

impl IndexerState {
    /// Create a new indexer state with the given components.
    pub fn new(
        walker: CodeWalker,
        splitter: CodeSplitter,
        embedding_client: EmbeddingClient,
        milvus_client: MilvusClient,
        embedding_dimension: usize,
    ) -> Self {
        Self {
            indexing_status: Arc::new(RwLock::new(IndexStatus::default())),
            walker: Arc::new(walker),
            splitter: Arc::new(splitter),
            embedding_client: Arc::new(embedding_client),
            milvus_client: Arc::new(milvus_client),
            manifest_store: ManifestStore,
            embedding_dimension,
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
pub async fn index_codebase(
    state: &IndexerState,
    path: &Path,
    force: bool,
) -> Result<IndexResult> {
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
            embeddings_generated: 0,
            vectors_inserted: 0,
            status: IndexState::Indexing,
        };
        let _ = state.manifest_store.write_status(path, &status);
    }

    info!("Starting codebase indexing at {}", path.display());

    let collection_name = collection_name_from_path(path);
    let index_inputs = IndexInputs::from_splitter_and_walker(
        state.splitter.config(),
        &state.walker.extensions,
        &state.walker.ignore_patterns,
    );

    // Phase 1: Walk files (parallel via ignore crate)
    let files = match state.walker.walk(path).await {
        Ok(files) => files,
        Err(e) => {
            update_status_failed(state, path).await;
            return Err(e).context("Failed to walk codebase");
        }
    };

    let total_files = files.len();
    info!("Discovered {} files in codebase", total_files);

    let previous_manifest = match state.manifest_store.load(path) {
        Ok(previous) => previous,
        Err(e) => {
            update_status_failed(state, path).await;
            return Err(e).context("Failed to load index manifest");
        }
    };

    let mut full_reindex = force;
    let mut files_to_index = files.clone();
    let mut stale_relative_paths = Vec::new();

    if !force {
        match previous_manifest.as_ref() {
            Some(previous) if previous.matches_index_inputs(&collection_name, &index_inputs) => {
                let diff = match diff_manifest_against_files(
                    previous,
                    path,
                    &collection_name,
                    &index_inputs,
                    &files,
                ) {
                    Ok(diff) => diff,
                    Err(e) => {
                        update_status_failed(state, path).await;
                        return Err(e).context("Failed to diff index manifest");
                    }
                };

                if diff.is_empty() {
                    update_status_completed(state, path, 0).await;
                    return Ok(IndexResult {
                        files_processed: 0,
                        chunks_created: 0,
                        embeddings_generated: 0,
                        vectors_inserted: 0,
                        duration_ms: start.elapsed().as_millis() as u64,
                        warnings: vec!["already up to date".to_string()],
                    });
                }

                let changed_relative_paths = diff
                    .added
                    .iter()
                    .chain(diff.modified.iter())
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>();
                files_to_index = files
                    .iter()
                    .filter(|file_path| {
                        changed_relative_paths.contains(&relative_path(path, file_path))
                    })
                    .cloned()
                    .collect();

                stale_relative_paths = diff
                    .deleted
                    .iter()
                    .chain(diff.modified.iter())
                    .cloned()
                    .collect();
            }
            Some(_) | None => {
                full_reindex = true;
            }
        }
    }

    if let Err(e) = prepare_vector_index(state, &collection_name, full_reindex).await {
        update_status_failed(state, path).await;
        return Err(e).context("Failed to prepare Milvus collection");
    }

    if let Err(e) = prepare_lexical_index(path, &stale_relative_paths, full_reindex).await {
        update_status_failed(state, path).await;
        return Err(e).context("Failed to prepare lexical index");
    }

    if !stale_relative_paths.is_empty() {
        let filter = build_relative_path_filter(&stale_relative_paths);
        if let Err(e) = state.milvus_client.delete(&collection_name, &filter).await {
            update_status_failed(state, path).await;
            return Err(e).context("Failed to delete stale vectors");
        }
    }

    {
        let mut status = state.indexing_status.write().await;
        status.total_files = files_to_index.len();
    }

    if files_to_index.is_empty() {
        if let Err(e) =
            state
                .manifest_store
                .write_for_files(path, &collection_name, &index_inputs, &files)
        {
            update_status_failed(state, path).await;
            return Err(e).context("Failed to write index manifest");
        }

        update_status_completed(state, path, 0).await;
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
    let chunk_results: Vec<Result<Vec<CodeChunk>, String>> = files_to_index
        .par_chunks(FILE_PARALLEL_CHUNK_SIZE)
        .flat_map(|file_batch| {
            file_batch.par_iter().map(|file_path| {
                match splitter.split_file(file_path) {
                    Ok(chunks) => {
                        let count = processed_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        // Update status periodically (every 10 files to reduce lock contention)
                        if count.is_multiple_of(10) {
                            // We can't await in rayon, so we use try_write
                            if let Ok(mut status) = status_ref.try_write() {
                                status.processed_files = count;
                                let snapshot = status.clone();
                                drop(status);
                                let _ = state.manifest_store.write_status(path, &snapshot);
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
        files_to_index.len(),
        total_chunks,
        warnings.len()
    );

    {
        let mut status = state.indexing_status.write().await;
        status.processed_files = files_to_index.len();
        status.total_chunks = total_chunks;
        let _ = state.manifest_store.write_status(path, &status);
    }

    // Keep the lexical index in sync with the chunk set before embeddings/Milvus insertion.
    let lexical_chunks = all_chunks.clone();
    let lexical_path = path.to_path_buf();
    if let Err(e) = task::spawn_blocking(move || -> Result<()> {
        let lexical_index = LexicalIndex::create(&lexical_path)?;
        lexical_index.insert_chunks(&lexical_chunks)?;
        Ok(())
    })
    .await
    .context("Lexical index task panicked")
    .and_then(|result| result)
    {
        update_status_failed(state, path).await;
        return Err(e).context("Failed to update lexical index");
    }

    if total_chunks == 0 {
        if let Err(e) =
            state
                .manifest_store
                .write_for_files(path, &collection_name, &index_inputs, &files)
        {
            update_status_failed(state, path).await;
            return Err(e).context("Failed to write index manifest");
        }

        update_status_completed(state, path, 0).await;
        return Ok(IndexResult {
            files_processed: files_to_index.len(),
            chunks_created: 0,
            embeddings_generated: 0,
            vectors_inserted: 0,
            duration_ms: start.elapsed().as_millis() as u64,
            warnings,
        });
    }

    // Phase 3/4: Stream embedding + insertion in small batches so the first
    // successful Milvus writes land early and progress survives long runs.
    let embedding_client = state.embedding_client.clone();
    let milvus_client = state.milvus_client.clone();

    let chunk_batches: Vec<_> = all_chunks.chunks(EMBEDDING_BATCH_SIZE).collect();
    let num_embedding_batches = chunk_batches.len();
    info!(
        "Streaming embeddings + inserts in {} batches of {}",
        num_embedding_batches, EMBEDDING_BATCH_SIZE
    );

    let semaphore = Arc::new(tokio::sync::Semaphore::new(4));
    let mut batch_handles = FuturesUnordered::new();
    let mut embeddings_generated = 0usize;
    let mut vectors_inserted = 0usize;
    let mut batch_failures = Vec::new();

    for batch in chunk_batches {
        let batch_chunks: Vec<CodeChunk> = batch.to_vec();
        let client = embedding_client.clone();
        let milvus = milvus_client.clone();
        let collection = collection_name.clone();
        let permit = semaphore.clone().acquire_owned().await?;

        batch_handles.push(tokio::spawn(async move {
            let batch_texts: Vec<String> = batch_chunks.iter().map(|c| c.content.clone()).collect();
            let embeddings = client.embed_batch(&batch_texts).await?;
            let insert_rows: Vec<InsertRow> = batch_chunks
                .into_iter()
                .zip(embeddings.into_iter())
                .map(|(chunk, embedding)| InsertRow {
                    id: milvus_id_for_chunk_id(&chunk.id),
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

            let mut inserted = 0usize;
            for rows in insert_rows.chunks(MILVUS_BATCH_SIZE) {
                milvus.insert_batch(&collection, rows).await?;
                inserted += rows.len();
            }

            drop(permit);
            Ok::<(usize, usize), anyhow::Error>((batch_texts.len(), inserted))
        }));
    }

    while let Some(handle) = batch_handles.next().await {
        match handle {
            Ok(Ok((embedded, inserted))) => {
                embeddings_generated += embedded;
                vectors_inserted += inserted;
                info!(
                    "Streaming progress: embeddings_generated={} vectors_inserted={}",
                    embeddings_generated,
                    vectors_inserted
                );
                let mut status = state.indexing_status.write().await;
                status.embeddings_generated = embeddings_generated;
                status.vectors_inserted = vectors_inserted;
                let _ = state.manifest_store.write_status(path, &status);
            }
            Ok(Err(e)) => {
                warn!("Streaming batch failed: {}", e);
                batch_failures.push(format!("Streaming batch failed: {}", e));
            }
            Err(e) => {
                warn!("Streaming batch task panicked: {}", e);
                batch_failures.push(format!("Streaming batch task panicked: {}", e));
            }
        }
    }

    if !batch_failures.is_empty() {
        update_status_failed(state, path).await;
        anyhow::bail!(
            "Streaming index failed after inserting {} vectors: {}",
            vectors_inserted,
            batch_failures.join("; ")
        );
    }

    info!("Generated {} embeddings", embeddings_generated);

    if embeddings_generated == 0 {
        update_status_failed(state, path).await;
        anyhow::bail!("Failed to generate any embeddings");
    }

    info!("Using collection: {}", collection_name);

    info!("Inserted {} vectors into Milvus", vectors_inserted);

    if vectors_inserted == 0 {
        update_status_failed(state, path).await;
        anyhow::bail!(
            "Generated {} chunks and {} embeddings, but inserted 0 vectors into Milvus{}",
            total_chunks,
            embeddings_generated,
            if warnings.is_empty() {
                String::new()
            } else {
                format!(" ({})", warnings.join("; "))
            }
        );
    }

    if let Err(e) =
        state
            .manifest_store
            .write_for_files(path, &collection_name, &index_inputs, &files)
    {
        update_status_failed(state, path).await;
        return Err(e).context("Failed to write index manifest");
    }

    // Update final status
    update_status_completed(state, path, total_chunks).await;

    let duration_ms = start.elapsed().as_millis() as u64;
    info!(
        "Indexing completed in {}ms: {} files, {} chunks, {} vectors",
        duration_ms, total_files, total_chunks, vectors_inserted
    );

    Ok(IndexResult {
        files_processed: files_to_index.len(),
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
    state: Arc<IndexerState>,
    path: std::path::PathBuf,
    force: bool,
) -> tokio::task::JoinHandle<Result<IndexResult>> {
    tokio::spawn(async move { index_codebase(&state, &path, force).await })
}

async fn update_status_failed(state: &IndexerState, path: &Path) {
    let mut status = state.indexing_status.write().await;
    status.status = IndexState::Failed;
    let _ = state.manifest_store.write_status(path, &status);
}

async fn update_status_completed(state: &IndexerState, path: &Path, chunks: usize) {
    let mut status = state.indexing_status.write().await;
    status.total_chunks = chunks;
    status.processed_files = status.total_files;
    status.status = IndexState::Completed;
    let _ = state.manifest_store.write_status(path, &status);
}

async fn prepare_vector_index(
    state: &IndexerState,
    collection_name: &str,
    full_reindex: bool,
) -> Result<()> {
    if full_reindex && state.milvus_client.has_collection(collection_name).await? {
        state.milvus_client.drop_collection(collection_name).await?;
    }

    if !state.milvus_client.has_collection(collection_name).await? {
        state
            .milvus_client
            .create_collection(collection_name, state.embedding_dimension)
            .await?;
    }

    Ok(())
}

async fn prepare_lexical_index(
    path: &Path,
    stale_relative_paths: &[String],
    full_reindex: bool,
) -> Result<()> {
    let lexical_index = LexicalIndex::create(path)?;
    if full_reindex {
        lexical_index.clear()?;
    } else {
        lexical_index.delete_by_paths(stale_relative_paths)?;
    }
    Ok(())
}

fn relative_path(root: &Path, file_path: &Path) -> String {
    file_path
        .strip_prefix(root)
        .unwrap_or(file_path)
        .to_string_lossy()
        .to_string()
}

fn build_relative_path_filter(relative_paths: &[String]) -> String {
    let serialized = relative_paths
        .iter()
        .map(|path| serde_json::to_string(path).expect("relative path must serialize"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("metadata[\"relative_path\"] in [{serialized}]")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;
    use std::io;
    use std::sync::{Arc, Mutex};

    use serde_json::json;
    use tempfile::TempDir;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use crate::config::Config;
    use crate::embedding::{EmbeddingClient, EmbeddingConfig};
    use crate::lexical::test_support::set_test_cache_dir_async;
    use crate::splitter::{CodeSplitter, Config as SplitterConfig};
    use crate::vectordb::MilvusClient;
    use crate::walker::CodeWalker;

    struct MockHttpServer {
        base_url: String,
        handle: tokio::task::JoinHandle<()>,
    }

    impl MockHttpServer {
        async fn wait(self) {
            self.handle.await.unwrap();
        }
    }

    async fn spawn_mock_milvus_server() -> MockHttpServer {
        spawn_mock_json_server(HashMap::from([
            (
                "/v2/vectordb/collections/has",
                serde_json::json!({
                    "code": 0,
                    "data": {"has": false}
                }),
            ),
            (
                "/v2/vectordb/collections/create",
                serde_json::json!({
                    "code": 0
                }),
            ),
            (
                "/v2/vectordb/entities/insert",
                serde_json::json!({
                    "code": 0
                }),
            ),
            (
                "/v2/vectordb/entities/delete",
                serde_json::json!({
                    "code": 0
                }),
            ),
            (
                "/v2/vectordb/collections/drop",
                serde_json::json!({
                    "code": 0
                }),
            ),
        ]))
        .await
    }

    #[derive(Default)]
    struct MockMilvusState {
        has_collection: bool,
    }

    async fn spawn_stateful_mock_milvus_server(
        state: Arc<Mutex<MockMilvusState>>,
    ) -> MockHttpServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let handle = tokio::spawn(async move {
            loop {
                let accept =
                    tokio::time::timeout(std::time::Duration::from_millis(500), listener.accept())
                        .await;
                let Ok(Ok((mut stream, _))) = accept else {
                    break;
                };

                let request = read_http_request(&mut stream).await.unwrap();
                let request_line = request.lines().next().unwrap_or_default();

                let response_body = if request_line.contains("/v2/vectordb/collections/has") {
                    let has = state.lock().unwrap().has_collection;
                    json!({ "code": 0, "data": { "has": has } }).to_string()
                } else if request_line.contains("/v2/vectordb/collections/create") {
                    state.lock().unwrap().has_collection = true;
                    json!({ "code": 0 }).to_string()
                } else if request_line.contains("/v2/vectordb/collections/drop") {
                    state.lock().unwrap().has_collection = false;
                    json!({ "code": 0 }).to_string()
                } else if request_line.contains("/v2/vectordb/entities/insert")
                    || request_line.contains("/v2/vectordb/entities/delete")
                {
                    json!({ "code": 0 }).to_string()
                } else {
                    json!({ "code": 404, "message": "not found" }).to_string()
                };

                let status = if response_body.contains("\"code\":404") {
                    "404 Not Found"
                } else {
                    "200 OK"
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        MockHttpServer {
            base_url: format!("http://{addr}"),
            handle,
        }
    }

    async fn spawn_mock_embedding_server(response_body: serde_json::Value) -> MockHttpServer {
        spawn_mock_json_server(HashMap::from([("/v1/embeddings", response_body)])).await
    }

    async fn spawn_dynamic_mock_embedding_server() -> MockHttpServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let handle = tokio::spawn(async move {
            loop {
                let accept =
                    tokio::time::timeout(std::time::Duration::from_millis(500), listener.accept())
                        .await;
                let Ok(Ok((mut stream, _))) = accept else {
                    break;
                };

                let request = read_http_request(&mut stream).await.unwrap();
                let request_line = request.lines().next().unwrap_or_default();

                let (status, response_body) = if request_line.contains("/v1/embeddings") {
                    let body = request.split("\r\n\r\n").nth(1).unwrap_or_default();
                    let input_count = serde_json::from_str::<serde_json::Value>(body)
                        .ok()
                        .and_then(|payload| {
                            payload
                                .get("input")
                                .and_then(|v| v.as_array())
                                .map(|v| v.len())
                        })
                        .unwrap_or(0);
                    let response = json!({
                        "data": (0..input_count)
                            .map(|idx| json!({ "embedding": [idx as f32 + 0.1, 0.2, 0.3, 0.4] }))
                            .collect::<Vec<_>>()
                    });
                    ("200 OK", response.to_string())
                } else {
                    (
                        "404 Not Found",
                        json!({ "code": 404, "message": "not found" }).to_string(),
                    )
                };

                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        MockHttpServer {
            base_url: format!("http://{addr}"),
            handle,
        }
    }

    async fn spawn_mock_json_server(
        responses: HashMap<&'static str, serde_json::Value>,
    ) -> MockHttpServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let response_bodies: HashMap<&'static str, String> = responses
            .into_iter()
            .map(|(path, body)| (path, body.to_string()))
            .collect();

        let handle = tokio::spawn(async move {
            loop {
                let accept =
                    tokio::time::timeout(std::time::Duration::from_millis(500), listener.accept())
                        .await;
                let Ok(Ok((mut stream, _))) = accept else {
                    break;
                };

                let request = read_http_request(&mut stream).await.unwrap();
                let request_line = request.lines().next().unwrap_or_default();
                let matched = response_bodies.iter().find_map(|(path, body)| {
                    request_line
                        .contains(path)
                        .then_some((*path, body.as_str()))
                });

                let (status, response_body) = if let Some((_, body)) = matched {
                    ("200 OK", body.to_string())
                } else {
                    (
                        "404 Not Found",
                        r#"{"code":404,"message":"not found"}"#.to_string(),
                    )
                };

                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        MockHttpServer {
            base_url: format!("http://{addr}"),
            handle,
        }
    }

    async fn read_http_request(stream: &mut tokio::net::TcpStream) -> io::Result<String> {
        let mut buffer = Vec::new();
        let mut temp = [0_u8; 1024];
        let mut content_length = None;

        loop {
            let read = stream.read(&mut temp).await?;
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&temp[..read]);

            if let Some(header_end) = find_header_end(&buffer) {
                if content_length.is_none() {
                    let headers = String::from_utf8_lossy(&buffer[..header_end]);
                    content_length = headers.lines().find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    });
                }

                let body_len = buffer.len() - header_end - 4;
                if body_len >= content_length.unwrap_or(0) {
                    break;
                }
            }
        }

        Ok(String::from_utf8_lossy(&buffer).into_owned())
    }

    fn find_header_end(buffer: &[u8]) -> Option<usize> {
        buffer.windows(4).position(|window| window == b"\r\n\r\n")
    }

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

    #[test]
    fn test_build_relative_path_filter() {
        let filter = build_relative_path_filter(&[
            "src/lib.rs".to_string(),
            "src/nested/mod.rs".to_string(),
        ]);
        assert_eq!(
            filter,
            "metadata[\"relative_path\"] in [\"src/lib.rs\", \"src/nested/mod.rs\"]"
        );
    }

    #[tokio::test]
    async fn test_reindex_skips_unchanged_files() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        let src_dir = root.join("src");
        fs::create_dir_all(&src_dir).unwrap();
        fs::write(
            src_dir.join("lib.rs"),
            "pub fn alpha() -> i32 {\n    1\n}\n",
        )
        .unwrap();
        fs::write(
            src_dir.join("main.rs"),
            "fn main() {\n    println!(\"hi\");\n}\n",
        )
        .unwrap();

        let milvus = spawn_mock_milvus_server().await;
        let embedding = spawn_mock_embedding_server(serde_json::json!({
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ]
        }))
        .await;

        let state = IndexerState {
            indexing_status: Arc::new(RwLock::new(IndexStatus::default())),
            walker: Arc::new(CodeWalker::new()),
            splitter: Arc::new(CodeSplitter::new(SplitterConfig {
                root_path: root.to_path_buf(),
                max_chunk_bytes: Config::default().chunk_size,
                overlap_lines: Config::default().chunk_overlap / 80,
                ..SplitterConfig::default()
            })),
            embedding_client: Arc::new(EmbeddingClient::new(EmbeddingConfig {
                url: format!("{}/v1/embeddings", embedding.base_url),
                model: "test".to_string(),
                batch_size: 100,
                api_key: None,
            })),
            milvus_client: Arc::new(MilvusClient::new(&milvus.base_url, None)),
            manifest_store: ManifestStore,
            embedding_dimension: 3,
        };

        let first_result = index_codebase(&state, root, false).await.unwrap();
        assert!(first_result.files_processed >= 2);

        let manifest_path = root.join(".rust_sindexer").join("index-manifest.json");
        assert!(manifest_path.exists());

        let second_result = index_codebase(&state, root, false).await.unwrap();
        assert_eq!(second_result.chunks_created, 0);
        assert!(
            second_result.warnings.is_empty()
                || second_result
                    .warnings
                    .iter()
                    .any(|warning| warning.contains("already up to date"))
        );
        assert!(manifest_path.exists());

        embedding.wait().await;
        milvus.wait().await;
    }

    #[tokio::test]
    async fn test_force_reindex_ignores_manifest() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        let src_dir = root.join("src");
        fs::create_dir_all(&src_dir).unwrap();
        fs::write(
            src_dir.join("lib.rs"),
            "pub fn alpha() -> i32 {\n    let value = 1;\n    value\n}\n",
        )
        .unwrap();
        fs::write(
            src_dir.join("main.rs"),
            "fn main() {\n    println!(\"force reindex\");\n}\n",
        )
        .unwrap();

        let milvus_state = Arc::new(Mutex::new(MockMilvusState::default()));
        let milvus = spawn_stateful_mock_milvus_server(milvus_state).await;
        let embedding = spawn_dynamic_mock_embedding_server().await;

        let state = IndexerState {
            indexing_status: Arc::new(RwLock::new(IndexStatus::default())),
            walker: Arc::new(CodeWalker::new()),
            splitter: Arc::new(CodeSplitter::new(SplitterConfig {
                root_path: root.to_path_buf(),
                max_chunk_bytes: Config::default().chunk_size,
                overlap_lines: Config::default().chunk_overlap / 80,
                ..SplitterConfig::default()
            })),
            embedding_client: Arc::new(EmbeddingClient::new(EmbeddingConfig {
                url: format!("{}/v1/embeddings", embedding.base_url),
                model: "test".to_string(),
                batch_size: 100,
                api_key: None,
            })),
            milvus_client: Arc::new(MilvusClient::new(&milvus.base_url, None)),
            manifest_store: ManifestStore,
            embedding_dimension: 4,
        };

        let first_result = index_codebase(&state, root, false).await.unwrap();
        let forced_result = index_codebase(&state, root, true).await.unwrap();

        assert_eq!(first_result.files_processed, 2);
        assert!(first_result.chunks_created > 0);
        assert_eq!(forced_result.files_processed, 2);
        assert!(forced_result.chunks_created > 0);

        embedding.wait().await;
        milvus.wait().await;
    }

    #[tokio::test]
    async fn test_index_fails_when_no_vectors_are_inserted() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        let milvus = spawn_mock_json_server(HashMap::from([
            (
                "/v2/vectordb/collections/has",
                serde_json::json!({
                    "code": 0,
                    "data": {"has": false}
                }),
            ),
            (
                "/v2/vectordb/collections/create",
                serde_json::json!({
                    "code": 0
                }),
            ),
            (
                "/v2/vectordb/entities/insert",
                serde_json::json!({
                    "code": 1,
                    "message": "insert rejected"
                }),
            ),
            (
                "/v2/vectordb/entities/delete",
                serde_json::json!({
                    "code": 0
                }),
            ),
        ]))
        .await;
        let embedding = spawn_mock_embedding_server(serde_json::json!({
            "data": [
                {"embedding": [0.1, 0.2, 0.3]}
            ]
        }))
        .await;

        let state = IndexerState {
            indexing_status: Arc::new(RwLock::new(IndexStatus::default())),
            walker: Arc::new(CodeWalker::new()),
            splitter: Arc::new(CodeSplitter::new(SplitterConfig {
                root_path: root.to_path_buf(),
                max_chunk_bytes: Config::default().chunk_size,
                overlap_lines: Config::default().chunk_overlap / 80,
                ..SplitterConfig::default()
            })),
            embedding_client: Arc::new(EmbeddingClient::new(EmbeddingConfig {
                url: format!("{}/v1/embeddings", embedding.base_url),
                model: "test".to_string(),
                batch_size: 100,
                api_key: None,
            })),
            milvus_client: Arc::new(MilvusClient::new(&milvus.base_url, None)),
            manifest_store: ManifestStore,
            embedding_dimension: 3,
        };

        let err = index_codebase(&state, root, true).await.unwrap_err();
        assert!(err.to_string().contains("inserted 0 vectors into Milvus"));
        assert_eq!(state.get_status().await.status, IndexState::Failed);

        embedding.wait().await;
        milvus.wait().await;
    }
}
