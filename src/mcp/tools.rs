//! MCP tool definitions for codebase indexing and semantic search.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rmcp::{
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    tool, tool_router, ErrorData as McpError, Json, ServerHandler,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::config::Config;
use crate::embedding::{EmbeddingClient, EmbeddingConfig};
use crate::mcp::indexer::{spawn_index_codebase, ContextState as IndexerContextState};
use crate::mcp::manifest::ManifestStore;
use crate::mcp::state::{create_default_shared_state, SharedState};
use crate::splitter::{CodeSplitter, Config as SplitterConfig};
use crate::types::IndexStatus;
use crate::vectordb::{collection_name_from_path, MilvusClient};
use crate::walker::CodeWalker;

// ============================================================================
// Tool Input Schemas
// ============================================================================

/// Parameters for indexing a codebase.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IndexCodebaseParams {
    /// Absolute path to the codebase directory to index.
    pub path: String,
    /// Force re-indexing even if an index already exists.
    #[serde(default)]
    pub force: bool,
}

/// Parameters for searching indexed code.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchCodeParams {
    /// Absolute path to the indexed codebase.
    pub path: String,
    /// Natural language or code query to search for.
    pub query: String,
    /// Maximum number of results to return.
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    10
}

/// Parameters for checking indexing status.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GetIndexingStatusParams {
    /// Absolute path to the codebase to check.
    pub path: String,
}

/// Parameters for clearing an index.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ClearIndexParams {
    /// Absolute path to the codebase whose index should be cleared.
    pub path: String,
}

// ============================================================================
// Tool Output Schemas
// ============================================================================

/// Result of an indexing operation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IndexResult {
    /// Whether the indexing operation succeeded.
    pub success: bool,
    /// Human-readable message describing the result.
    pub message: String,
    /// Path that was indexed.
    pub path: PathBuf,
    /// Number of files indexed.
    pub files_indexed: usize,
    /// Number of code chunks created.
    pub chunks_created: usize,
}

/// A single search result from semantic code search.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchResultItem {
    /// Path to the file containing the match.
    pub file_path: PathBuf,
    /// Path relative to the repository root.
    pub relative_path: String,
    /// The matching code snippet.
    pub content: String,
    /// Starting line number (1-indexed).
    pub start_line: u32,
    /// Ending line number (1-indexed, inclusive).
    pub end_line: u32,
    /// Programming language of the code.
    pub language: String,
    /// Similarity score (0.0 to 1.0, higher is more relevant).
    pub score: f32,
}

/// Result of a search operation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchResults {
    /// The search results.
    pub results: Vec<SearchResultItem>,
    /// Number of results returned.
    pub count: usize,
}

/// Result of clearing an index.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ClearResult {
    /// Whether the clear operation succeeded.
    pub success: bool,
    /// Human-readable message describing the result.
    pub message: String,
    /// Path whose index was cleared.
    pub path: PathBuf,
}

// ============================================================================
// Tool Handler
// ============================================================================

/// MCP tool handler for codebase indexing and semantic search.
#[derive(Clone)]
pub struct CodebaseTools {
    state: SharedState,
    tool_router: ToolRouter<Self>,
}

impl CodebaseTools {
    /// Create a new tool handler instance.
    pub fn new(state: SharedState) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    /// Get the tool router for this handler.
    pub fn router(&self) -> &ToolRouter<Self> {
        &self.tool_router
    }
}

impl Default for CodebaseTools {
    fn default() -> Self {
        Self::new(create_default_shared_state())
    }
}

fn invalid_path(message: impl Into<String>) -> McpError {
    McpError::invalid_params(message.into(), None)
}

fn validate_directory_path(path: &str) -> Result<PathBuf, McpError> {
    let path = PathBuf::from(path);

    if !path.is_absolute() {
        return Err(invalid_path(format!(
            "Path must be absolute: {}",
            path.display()
        )));
    }

    if !path.exists() {
        return Err(invalid_path(format!(
            "Path does not exist: {}",
            path.display()
        )));
    }

    if !path.is_dir() {
        return Err(invalid_path(format!(
            "Path is not a directory: {}",
            path.display()
        )));
    }

    Ok(path)
}

fn validate_absolute_path(path: &str) -> Result<PathBuf, McpError> {
    let path = PathBuf::from(path);

    if !path.is_absolute() {
        return Err(invalid_path(format!(
            "Path must be absolute: {}",
            path.display()
        )));
    }

    Ok(path)
}

fn build_indexer_state(
    config: &Config,
    root_path: &Path,
    manifest_store: Arc<ManifestStore>,
) -> Arc<IndexerContextState> {
    let walker = CodeWalker::new();
    let splitter = CodeSplitter::new(SplitterConfig {
        max_chunk_bytes: config.chunk_size,
        overlap_lines: (config.chunk_overlap / 80).max(1),
        root_path: root_path.to_path_buf(),
        ..Default::default()
    });
    let embedding_client = EmbeddingClient::new(EmbeddingConfig::from_config(config));
    let milvus_client = MilvusClient::new(&config.milvus_url);

    Arc::new(IndexerContextState::new(
        config.clone(),
        walker,
        splitter,
        embedding_client,
        milvus_client,
        manifest_store,
    ))
}

async fn clear_existing_collection_state(state: &SharedState, path: &Path) -> anyhow::Result<bool> {
    let collection_name = collection_name_from_path(path);
    let had_collection = state.milvus_client.has_collection(&collection_name).await?;
    if had_collection {
        state
            .milvus_client
            .drop_collection(&collection_name)
            .await?;
    }
    if path.exists() {
        state.manifest_store.remove(path)?;
    }
    Ok(had_collection)
}

fn spawn_background_index(shared_state: SharedState, path: PathBuf, force: bool) {
    tokio::spawn(async move {
        if force {
            if let Err(err) = clear_existing_collection_state(&shared_state, &path).await {
                warn!(path = %path.display(), ?err, "failed to clear existing collection state");
                shared_state.fail_indexing(&path);
                return;
            }
        }

        let indexer_state = build_indexer_state(
            &shared_state.config,
            &path,
            shared_state.manifest_store.clone(),
        );
        let handle = spawn_index_codebase(indexer_state, path.clone(), false);

        match handle.await {
            Ok(Ok(result)) => {
                shared_state.complete_indexing(&path, result.chunks_created);
            }
            Ok(Err(err)) => {
                warn!(path = %path.display(), ?err, "background indexing failed");
                shared_state.fail_indexing(&path);
            }
            Err(err) => {
                warn!(path = %path.display(), ?err, "background indexing task panicked");
                shared_state.fail_indexing(&path);
            }
        }
    });
}

impl ServerHandler for CodebaseTools {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Semantic code indexing and search MCP server".into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[tool_router]
impl CodebaseTools {
    /// Index a codebase for semantic search.
    ///
    /// Walks the directory tree, extracts code chunks using tree-sitter,
    /// generates embeddings, and stores them in a vector database.
    #[tool(
        name = "index_codebase",
        description = "Index a codebase directory for semantic code search. Walks the directory, \
                       parses source files, extracts code chunks, and generates embeddings. \
                       Use force=true to re-index an already indexed codebase."
    )]
    async fn index_codebase(
        &self,
        params: Parameters<IndexCodebaseParams>,
    ) -> Result<Json<IndexResult>, McpError> {
        let params = params.0;
        let path = validate_directory_path(&params.path)?;

        if self.state.is_indexing(&path) && !params.force {
            return Err(invalid_path(format!(
                "Indexing is already running for {}. Use force=true to rebuild the index.",
                path.display()
            )));
        }

        self.state.start_indexing(&path, 0);
        spawn_background_index(self.state.clone(), path.clone(), params.force);

        Ok(Json(IndexResult {
            success: true,
            message: format!(
                "Started indexing {} in the background{}",
                path.display(),
                if params.force { " (force rebuild)" } else { "" }
            ),
            path,
            files_indexed: 0,
            chunks_created: 0,
        }))
    }

    /// Search indexed code using semantic similarity.
    ///
    /// Converts the query to an embedding and finds the most similar
    /// code chunks in the vector database.
    #[tool(
        name = "search_code",
        description = "Search indexed code using natural language or code queries. \
                       Returns the most semantically similar code chunks from the indexed codebase."
    )]
    async fn search_code(
        &self,
        params: Parameters<SearchCodeParams>,
    ) -> Result<Json<SearchResults>, McpError> {
        let params = params.0;
        let _path = validate_directory_path(&params.path)?;

        // Validate limit
        if params.limit == 0 {
            return Err(McpError::invalid_params(
                "Limit must be greater than 0".to_string(),
                None,
            ));
        }

        // TODO: Implement actual search logic
        // This is a placeholder that will be connected to the vector database
        Ok(Json(SearchResults {
            results: vec![],
            count: 0,
        }))
    }

    /// Get the current indexing status for a codebase.
    #[tool(
        name = "get_indexing_status",
        description = "Check the indexing status of a codebase. Returns information about \
                       whether indexing is in progress, completed, or not started."
    )]
    async fn get_indexing_status(
        &self,
        params: Parameters<GetIndexingStatusParams>,
    ) -> Result<Json<IndexStatus>, McpError> {
        let params = params.0;
        let path = validate_absolute_path(&params.path)?;

        Ok(Json(self.state.get_status(&path)))
    }

    /// Clear the index for a codebase.
    #[tool(
        name = "clear_index",
        description = "Remove the index for a codebase, freeing up storage. \
                       The codebase will need to be re-indexed before searching."
    )]
    async fn clear_index(
        &self,
        params: Parameters<ClearIndexParams>,
    ) -> Result<Json<ClearResult>, McpError> {
        let params = params.0;
        let path = validate_absolute_path(&params.path)?;

        let had_collection = clear_existing_collection_state(&self.state, &path)
            .await
            .map_err(|err| McpError::internal_error(err.to_string(), None))?;
        self.state.set_status(path.clone(), IndexStatus::default());

        Ok(Json(ClearResult {
            success: true,
            message: if had_collection {
                format!("Cleared index for {}", path.display())
            } else {
                format!(
                    "No index existed for {}; status reset to idle",
                    path.display()
                )
            },
            path,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    use crate::config::Config;
    use crate::mcp::state::create_shared_state;
    use crate::types::IndexState;

    #[test]
    fn test_default_limit() {
        assert_eq!(default_limit(), 10);
    }

    #[test]
    fn test_index_params_default_force() {
        let json = r#"{"path": "/tmp/test"}"#;
        let params: IndexCodebaseParams = serde_json::from_str(json).unwrap();
        assert!(!params.force);
    }

    #[test]
    fn test_search_params_default_limit() {
        let json = r#"{"path": "/tmp/test", "query": "find main function"}"#;
        let params: SearchCodeParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.limit, 10);
    }

    #[test]
    fn test_codebase_tools_creation() {
        let tools = CodebaseTools::default();
        let all_tools = tools.router().list_all();
        assert_eq!(all_tools.len(), 4); // index_codebase, search_code, get_indexing_status, clear_index
    }

    #[test]
    fn test_validate_directory_path_rejects_relative_paths() {
        let err = validate_directory_path("relative/path").unwrap_err();
        assert!(err.message.contains("Path must be absolute"));
    }

    #[test]
    fn test_validate_absolute_path_rejects_relative_paths() {
        let err = validate_absolute_path("relative/path").unwrap_err();
        assert!(err.message.contains("Path must be absolute"));
    }

    #[tokio::test]
    async fn test_index_codebase_rejects_duplicate_inflight_requests() {
        let tempdir = tempfile::tempdir().unwrap();
        let tools = CodebaseTools::default();
        tools.state.start_indexing(tempdir.path(), 0);

        let err = match tools
            .index_codebase(Parameters(IndexCodebaseParams {
                path: tempdir.path().display().to_string(),
                force: false,
            }))
            .await
        {
            Ok(_) => panic!("expected duplicate indexing request to fail"),
            Err(err) => err,
        };

        assert!(err.message.contains("already running"));
    }

    #[tokio::test]
    async fn test_index_codebase_starts_background_job() {
        let tempdir = tempfile::tempdir().unwrap();
        let tools = CodebaseTools::default();

        let response = tools
            .index_codebase(Parameters(IndexCodebaseParams {
                path: tempdir.path().display().to_string(),
                force: false,
            }))
            .await
            .unwrap();

        assert!(response.0.success);
        assert!(response.0.message.contains("Started indexing"));
        assert_eq!(response.0.path, tempdir.path());
        assert_eq!(
            tools.state.get_status(tempdir.path()).status,
            IndexState::Indexing
        );
    }

    #[tokio::test]
    async fn test_get_indexing_status_returns_default_for_never_indexed_path() {
        let tools = CodebaseTools::default();
        let unknown_path = std::env::temp_dir().join(format!(
            "rclaude-context-never-indexed-{}",
            std::process::id()
        ));

        let response = tools
            .get_indexing_status(Parameters(GetIndexingStatusParams {
                path: unknown_path.display().to_string(),
            }))
            .await
            .unwrap();

        assert_eq!(response.0.total_files, 0);
        assert_eq!(response.0.processed_files, 0);
        assert_eq!(response.0.total_chunks, 0);
        assert_eq!(response.0.status, IndexState::Idle);
    }

    #[tokio::test]
    async fn test_clear_index_drops_existing_collection_and_resets_status() {
        let requests = Arc::new(Mutex::new(Vec::<String>::new()));
        let server = spawn_mock_milvus_server(
            vec![
                (
                    "/v2/vectordb/collections/has",
                    r#"{"code":0,"data":{"has":true}}"#,
                ),
                ("/v2/vectordb/collections/drop", r#"{"code":0}"#),
            ],
            requests.clone(),
        );

        let tempdir = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.milvus_url = server.base_url.clone();
        let tools = CodebaseTools::new(create_shared_state(config));
        tools.state.set_status(
            tempdir.path().to_path_buf(),
            IndexStatus {
                total_files: 5,
                processed_files: 5,
                total_chunks: 9,
                status: IndexState::Completed,
            },
        );

        let response = tools
            .clear_index(Parameters(ClearIndexParams {
                path: tempdir.path().display().to_string(),
            }))
            .await
            .unwrap();

        assert!(response.0.success);
        assert!(response.0.message.contains("Cleared index"));
        assert_eq!(
            tools.state.get_status(tempdir.path()).status,
            IndexState::Idle
        );
        assert_eq!(
            requests.lock().unwrap().as_slice(),
            [
                "/v2/vectordb/collections/has",
                "/v2/vectordb/collections/drop",
            ]
        );

        server.join();
    }

    #[tokio::test]
    async fn test_clear_index_is_idempotent_when_collection_missing() {
        let requests = Arc::new(Mutex::new(Vec::<String>::new()));
        let server = spawn_mock_milvus_server(
            vec![(
                "/v2/vectordb/collections/has",
                r#"{"code":0,"data":{"has":false}}"#,
            )],
            requests.clone(),
        );

        let tempdir = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.milvus_url = server.base_url.clone();
        let tools = CodebaseTools::new(create_shared_state(config));
        tools.state.set_status(
            tempdir.path().to_path_buf(),
            IndexStatus {
                total_files: 2,
                processed_files: 1,
                total_chunks: 3,
                status: IndexState::Indexing,
            },
        );

        let response = tools
            .clear_index(Parameters(ClearIndexParams {
                path: tempdir.path().display().to_string(),
            }))
            .await
            .unwrap();

        assert!(response.0.success);
        assert!(response.0.message.contains("No index existed"));
        assert_eq!(
            tools.state.get_status(tempdir.path()).status,
            IndexState::Idle
        );
        assert_eq!(
            requests.lock().unwrap().as_slice(),
            ["/v2/vectordb/collections/has"]
        );

        server.join();
    }

    struct MockMilvusServer {
        base_url: String,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl MockMilvusServer {
        fn join(mut self) {
            if let Some(handle) = self.handle.take() {
                handle.join().unwrap();
            }
        }
    }

    fn spawn_mock_milvus_server(
        responses: Vec<(&'static str, &'static str)>,
        requests: Arc<Mutex<Vec<String>>>,
    ) -> MockMilvusServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            for (expected_path, body) in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0_u8; 4096];
                let bytes_read = stream.read(&mut buffer).unwrap();
                let request = String::from_utf8_lossy(&buffer[..bytes_read]);
                let request_line = request.lines().next().unwrap_or_default();
                let actual_path = request_line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or_default()
                    .to_string();
                requests.lock().unwrap().push(actual_path.clone());
                assert_eq!(actual_path, expected_path);

                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        MockMilvusServer {
            base_url: format!("http://{}", address),
            handle: Some(handle),
        }
    }
}
