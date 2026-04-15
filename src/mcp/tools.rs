//! MCP tool definitions for codebase indexing and semantic search.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rmcp::{
    handler::server::{
        tool::{ToolCallContext, ToolRouter},
        wrapper::Parameters,
    },
    model::{
        CallToolRequestParam, CallToolResult, ListToolsResult, PaginatedRequestParam,
        ServerCapabilities, ServerInfo,
    },
    service::{RequestContext, RoleServer},
    tool, tool_router, ErrorData as McpError, Json, ServerHandler,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::task;
use tokio::time::{sleep, Duration};

use super::indexer::{self, IndexerState};
use super::hybrid::{fuse_hybrid_hits, HybridFusionOptions, HybridHit};
use super::state::{create_default_shared_state, SharedState};
use crate::config::Config;
use crate::embedding::{EmbeddingClient, EmbeddingConfig};
use crate::lexical::LexicalIndex;
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
    /// Optional file extensions to include in the results (e.g. ["rs", "py"]).
    #[serde(default)]
    pub extensions: Vec<String>,
}

fn default_limit() -> u32 {
    10
}

fn create_indexer_state(config: &Config, root_path: &Path) -> Arc<IndexerState> {
    let splitter = CodeSplitter::new(SplitterConfig {
        root_path: root_path.to_path_buf(),
        max_chunk_bytes: config.chunk_size,
        overlap_lines: config.chunk_overlap / 80,
        ..SplitterConfig::default()
    });

    Arc::new(IndexerState::new(
        CodeWalker::new(),
        splitter,
        EmbeddingClient::new(EmbeddingConfig::from_config(config)),
        MilvusClient::new(&config.milvus_url, config.milvus_token.clone()),
        config.embedding_dimension,
    ))
}

fn mirror_index_status(
    shared_state: SharedState,
    indexer_state: Arc<IndexerState>,
    path: PathBuf,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let status = indexer_state.get_status().await;
            let done = !matches!(status.status, crate::types::IndexState::Indexing);
            shared_state.set_status(path.clone(), status);
            if done {
                break;
            }
            sleep(Duration::from_millis(250)).await;
        }
    })
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
    pub fn new() -> Self {
        Self::with_state(create_default_shared_state())
    }

    /// Create a new tool handler instance with shared state.
    pub fn with_state(state: SharedState) -> Self {
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
        Self::new()
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

impl ServerHandler for CodebaseTools {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Semantic code indexing and search MCP server".into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListToolsResult, McpError>> + Send + '_ {
        std::future::ready(Ok(ListToolsResult {
            tools: self.tool_router.list_all(),
            ..Default::default()
        }))
    }

    fn call_tool(
        &self,
        request: CallToolRequestParam,
        context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<CallToolResult, McpError>> + Send + '_ {
        async move {
            let tool_context = ToolCallContext::new(self, request, context);
            self.tool_router.call(tool_context).await
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

        let indexer_state = create_indexer_state(&self.state.config, &path);
        self.state.set_status(
            path.clone(),
            IndexStatus {
                total_files: 0,
                processed_files: 0,
                total_chunks: 0,
                embeddings_generated: 0,
                vectors_inserted: 0,
                status: crate::types::IndexState::Indexing,
            },
        );
        let status_mirror =
            mirror_index_status(self.state.clone(), indexer_state.clone(), path.clone());
        let result = indexer::index_codebase(&indexer_state, &path, params.force).await;
        let _ = status_mirror.await;
        let result = result.map_err(|err| McpError::internal_error(err.to_string(), None))?;

        Ok(Json(IndexResult {
            success: true,
            message: if result.warnings.is_empty() {
                format!("Indexed {}", params.path)
            } else {
                format!("Indexed {} ({})", params.path, result.warnings.join("; "))
            },
            path,
            files_indexed: result.files_processed,
            chunks_created: result.chunks_created,
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
        let path = validate_directory_path(&params.path)?;

        // Validate limit
        if params.limit == 0 {
            return Err(McpError::invalid_params(
                "Limit must be greater than 0".to_string(),
                None,
            ));
        }

        let collection = collection_name_from_path(&path);
        let vector_hits = self
            .state
            .search(&collection, &params.query, params.limit as usize)
            .await
            .map_err(|err| McpError::internal_error(err.to_string(), None))?
            .into_iter()
            .map(|result| HybridHit {
                chunk: result.chunk,
                score: result.score,
            })
            .collect::<Vec<_>>();
        let lexical_path = path.clone();
        let lexical_query = params.query.clone();
        let lexical_limit = params.limit as usize;
        let lexical_hits = task::spawn_blocking(move || -> anyhow::Result<Vec<HybridHit>> {
            if !LexicalIndex::exists(&lexical_path)? {
                return Ok(Vec::new());
            }

            let lexical_index = LexicalIndex::open(&lexical_path)?;
            let mut hits = lexical_index.search(&lexical_query, lexical_limit)?;
            for hit in &mut hits {
                if hit.chunk.file_path.as_os_str().is_empty() {
                    hit.chunk.file_path = lexical_path.join(&hit.chunk.relative_path);
                }
            }
            Ok(hits)
        })
        .await
        .map_err(|err| {
            McpError::internal_error(format!("Failed to join lexical search task: {err}"), None)
        })?
        .map_err(|err| {
            McpError::internal_error(format!("Failed to search lexical index: {err}"), None)
        })?;
        let options = HybridFusionOptions {
            limit: params.limit as usize,
            extension_filter: params.extensions.clone(),
        };
        let fused_hits = fuse_hybrid_hits(&params.query, vector_hits, lexical_hits, &options);

        let results = fused_hits
            .into_iter()
            .map(|result| SearchResultItem {
                file_path: result.chunk.file_path,
                relative_path: result.chunk.relative_path,
                content: result.chunk.content,
                start_line: result.chunk.start_line,
                end_line: result.chunk.end_line,
                language: result.chunk.language,
                score: result.score,
            })
            .collect::<Vec<_>>();

        Ok(Json(SearchResults {
            count: results.len(),
            results,
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

        // Validate path
        if !path.exists() {
            return Err(McpError::invalid_params(
                format!("Path does not exist: {}", params.path),
                None,
            ));
        }

        let status = self.state.get_status(&path);
        Ok(Json(status))
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

        // Drop the Milvus collection if it exists.
        let collection_name = collection_name_from_path(&path);
        let had_collection = self
            .state
            .milvus_client
            .has_collection(&collection_name)
            .await
            .map_err(|err| McpError::internal_error(err.to_string(), None))?;
        if had_collection {
            self.state
                .milvus_client
                .drop_collection(&collection_name)
                .await
                .map_err(|err| McpError::internal_error(err.to_string(), None))?;
        }
        self.state.set_status(path.clone(), IndexStatus::default());
        let _ = self.state.manifest_store.clear_status(&path);

        // Clearing is best-effort here: remove tracked status and report success
        // without failing if the backing collection has not been created yet.
        self.state.indexing_status.remove(&path);
        let lexical_path = path.clone();
        task::spawn_blocking(move || -> anyhow::Result<()> {
            let lexical_index = LexicalIndex::create(&lexical_path)?;
            lexical_index.clear()?;
            Ok(())
        })
        .await
        .map_err(|err| {
            McpError::internal_error(format!("Failed to join lexical clear task: {err}"), None)
        })?
        .map_err(|err| {
            McpError::internal_error(format!("Failed to clear lexical index: {err}"), None)
        })?;
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
    use std::collections::HashMap;
    use std::io;
    use std::path::PathBuf;

    use tempfile::tempdir;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use crate::config::Config;
    use crate::lexical::test_support::set_test_cache_dir_async;
    use crate::mcp::state::create_shared_state;

    struct MockHttpServer {
        base_url: String,
        handle: tokio::task::JoinHandle<()>,
    }

    impl MockHttpServer {
        async fn wait(self) {
            self.handle.await.unwrap();
        }
    }

    async fn spawn_mock_milvus_server(response_body: serde_json::Value) -> MockHttpServer {
        spawn_mock_json_server("/v2/vectordb/entities/search", response_body).await
    }

    async fn spawn_mock_embedding_server(response_body: serde_json::Value) -> MockHttpServer {
        spawn_mock_json_server("/v1/embeddings", response_body).await
    }

    async fn spawn_mock_json_server(
        expected_path: &'static str,
        response_body: serde_json::Value,
    ) -> MockHttpServer {
        spawn_mock_json_server_map_with_limit(HashMap::from([(expected_path, response_body)]), 1)
            .await
    }

    async fn spawn_mock_json_server_map_with_limit(
        responses: HashMap<&'static str, serde_json::Value>,
        max_requests: usize,
    ) -> MockHttpServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let mut served_requests = 0usize;
            loop {
                let accept =
                    tokio::time::timeout(std::time::Duration::from_secs(5), listener.accept())
                        .await;
                let Ok(Ok((mut stream, _))) = accept else {
                    break;
                };

                let request = read_http_request(&mut stream).await.unwrap();
                let matched = request
                    .lines()
                    .next()
                    .and_then(|line| responses.iter().find(|(path, _)| line.contains(*path)));
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
                served_requests += 1;
                if served_requests >= max_requests {
                    break;
                }
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
        assert!(params.extensions.is_empty());
    }

    #[test]
    fn test_codebase_tools_creation() {
        let tools = CodebaseTools::default();
        let all_tools = tools.router().list_all();
        assert_eq!(all_tools.len(), 4); // index_codebase, search_code, get_indexing_status, clear_index
    }

    #[tokio::test]
    async fn test_search_code_returns_results_from_milvus() {
        let milvus = spawn_mock_milvus_server(serde_json::json!({
            "code": 0,
            "data": [[
                {
                    "id": "chunk-1",
                    "distance": 0.95,
                    "content": "fn main() {}",
                    "metadata": {
                        "file_path": "/repo/src/main.rs",
                        "relative_path": "src/main.rs",
                        "start_line": 1,
                        "end_line": 3,
                        "language": "rust"
                    }
                },
                {
                    "id": "chunk-2",
                    "distance": 0.75,
                    "content": "fn helper() {}",
                    "metadata": {
                        "file_path": "/repo/src/lib.rs",
                        "relative_path": "src/lib.rs",
                        "start_line": 10,
                        "end_line": 12,
                        "language": "rust"
                    }
                }
            ]]
        }))
        .await;
        let embedding = spawn_mock_embedding_server(serde_json::json!({
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "model": "test",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }))
        .await;

        let repo_dir = tempdir().unwrap();
        let config = Config {
            embedding_url: embedding.base_url.clone(),
            embedding_model: "test".to_string(),
            milvus_url: milvus.base_url.clone(),
            ..Config::default()
        };
        let tools = CodebaseTools::with_state(create_shared_state(config.clone()));

        let Json(results) = tools
            .search_code(Parameters(SearchCodeParams {
                path: repo_dir.path().to_string_lossy().into_owned(),
                query: "main function".to_string(),
                limit: 2,
                extensions: Vec::new(),
            }))
            .await
            .unwrap();

        assert_eq!(results.count, 2);
        assert_eq!(
            results.results[0].file_path,
            PathBuf::from("/repo/src/main.rs")
        );
        assert_eq!(results.results[0].content, "fn main() {}");
        assert_eq!(results.results[0].score, 1.0 / 61.0);

        embedding.wait().await;
        milvus.wait().await;
    }

    #[tokio::test]
    async fn test_search_code_rejects_zero_limit() {
        let repo_dir = tempdir().unwrap();
        let tools = CodebaseTools::new();

        let err = match tools
            .search_code(Parameters(SearchCodeParams {
                path: repo_dir.path().to_string_lossy().into_owned(),
                query: "main function".to_string(),
                limit: 0,
                extensions: Vec::new(),
            }))
            .await
        {
            Ok(_) => panic!("expected search_code to reject zero limit"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("Limit must be greater than 0"));
    }

    #[tokio::test]
    async fn test_search_code_rejects_nonexistent_path() {
        let tools = CodebaseTools::new();

        let err = match tools
            .search_code(Parameters(SearchCodeParams {
                path: "/nonexistent/path".to_string(),
                query: "main function".to_string(),
                limit: 10,
                extensions: Vec::new(),
            }))
            .await
        {
            Ok(_) => panic!("expected search_code to reject nonexistent path"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("does not exist"));
    }

    #[tokio::test]
    async fn test_search_code_returns_lexical_hits() {
        let milvus = spawn_mock_milvus_server(serde_json::json!({
            "code": 0,
            "data": [[]]
        }))
        .await;
        let embedding = spawn_mock_embedding_server(serde_json::json!({
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "model": "test",
            "usage": {"prompt_tokens": 3, "total_tokens": 3}
        }))
        .await;

        let repo_dir = tempdir().unwrap();
        let cache_dir = tempdir().unwrap();
        let _cache_guard = set_test_cache_dir_async(cache_dir.path()).await;
        let lexical_index = LexicalIndex::create(repo_dir.path()).unwrap();
        lexical_index
            .insert_chunks(&[crate::types::CodeChunk {
                id: "chunk-lexical".to_string(),
                content: "fn calculate_score(value: i32) -> i32 { value + 1 }".to_string(),
                file_path: repo_dir.path().join("src/lib.rs"),
                relative_path: "src/lib.rs".to_string(),
                start_line: 1,
                end_line: 1,
                language: "rust".to_string(),
            }])
            .unwrap();

        let config = Config {
            embedding_url: embedding.base_url.clone(),
            embedding_model: "test".to_string(),
            milvus_url: milvus.base_url.clone(),
            ..Config::default()
        };
        let tools = CodebaseTools::with_state(create_shared_state(config.clone()));

        let Json(results) = tools
            .search_code(Parameters(SearchCodeParams {
                path: repo_dir.path().to_string_lossy().into_owned(),
                query: "calculate_score".to_string(),
                limit: 5,
                extensions: Vec::new(),
            }))
            .await
            .unwrap();

        assert_eq!(results.count, 1);
        assert_eq!(results.results[0].relative_path, "src/lib.rs");
        assert_eq!(
            results.results[0].file_path,
            repo_dir.path().join("src/lib.rs")
        );
        assert!(results.results[0].score > 0.0);

        embedding.wait().await;
        milvus.wait().await;
    }

    #[tokio::test]
    async fn test_index_codebase_updates_shared_status() {
        let milvus = spawn_mock_json_server_map_with_limit(HashMap::from([
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
        ]), 8)
        .await;
        let embedding = spawn_mock_embedding_server(serde_json::json!({
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "model": "test",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }))
        .await;

        let repo_dir = tempdir().unwrap();
        let cache_dir = tempdir().unwrap();
        let _cache_guard = set_test_cache_dir_async(cache_dir.path()).await;
        std::fs::write(repo_dir.path().join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        let config = Config {
            embedding_url: embedding.base_url.clone(),
            embedding_model: "test".to_string(),
            milvus_url: milvus.base_url.clone(),
            embedding_dimension: 3,
            ..Config::default()
        };
        let tools = CodebaseTools::with_state(create_shared_state(config.clone()));

        let Json(result) = tools
            .index_codebase(Parameters(IndexCodebaseParams {
                path: repo_dir.path().to_string_lossy().into_owned(),
                force: true,
            }))
            .await
            .unwrap();

        assert!(result.success);

        let fresh_tools = CodebaseTools::with_state(create_shared_state(config));
        let Json(status) = fresh_tools
            .get_indexing_status(Parameters(GetIndexingStatusParams {
                path: repo_dir.path().to_string_lossy().into_owned(),
            }))
            .await
            .unwrap();

        assert_eq!(status.status, crate::types::IndexState::Completed);
        assert_eq!(status.processed_files, 1);
        assert_eq!(status.total_chunks, 1);

        embedding.wait().await;
        milvus.wait().await;
    }
}
