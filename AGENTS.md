# sindexer (Semantic Indexer)

High-performance Rust MCP server for semantic code indexing. Single native binary, no Node.js overhead. Works out of the box with zero external services.

## Quick Start

```bash
cargo build --release
./target/release/sindexer
```

No configuration needed. By default, uses BM25 lexical search with a local vector store. Set `EMBEDDING_URL` and optionally `MILVUS_URL` to enable semantic search.

Global MCP client configuration (Claude Code `~/.claude.json`, Claude Desktop, etc.):
```json
{
  "mcpServers": {
    "sindexer": {
      "command": "/path/to/sindexer"
    }
  }
}
```

The AlphaHENG wrapper script (`scripts/run_claude_context_mcp.sh`) auto-discovers the binary and loads credentials from `~/.context/.env`.

## Operating Modes

- **Lexical only (default)** — No env vars needed. BM25 keyword/symbol search with local vector store. Good for exact matches and code navigation.
- **Semantic + lexical** — Set `EMBEDDING_URL` to an OpenAI-compatible endpoint. Hybrid RRF fusion of semantic similarity + BM25. Local vector store handles project-scale indexing (<50K chunks).
- **Full scale** — Set both `EMBEDDING_URL` and `MILVUS_URL` for large-scale deployments with Milvus/Zilliz Cloud as the vector backend.

## Production Configuration

Current deployment uses Jina Embeddings + Zilliz Cloud, configured via `~/.context/.env`:

- **Embedding**: Jina API (`jina-code-embeddings-1.5b`, 1536-dim) via OpenAI-compatible endpoint at `https://api.jina.ai/v1`
- **Vector DB**: Zilliz Cloud (managed Milvus) with token auth
- **Transport**: rmcp native stdio (newline-delimited JSON per MCP spec)

## Architecture

```
Walker (files) → Splitter (chunks) → Embedder (vectors) → Vector Store
                                          │
                                     Lexical (BM25)
                                          │
                                     Hybrid Fusion (RRF)
```

When embeddings are disabled, the pipeline stops after splitting and only populates the lexical index.

## Components

**Walker** (`src/walker/mod.rs`) — Parallel file discovery using the `ignore` crate with native .gitignore support. Filters by extension and extensionless filenames (Dockerfile, Makefile, etc.) during traversal. Uses `config::SUPPORTED_EXTENSIONS` as the single source of truth for 60+ file types.

**Splitter** (`src/splitter/`) — Tree-sitter AST parsing for semantic code chunking. Extracts functions, classes, structs, traits, impl blocks per language. Splits oversized chunks at line boundaries with configurable overlap. Falls back to markdown heading or line-based splitting for unsupported languages.

Supported AST languages: Python, JavaScript, TypeScript, TSX, Rust, Go, Java, C++, C, Ruby, PHP, Swift, Scala, C#

**Embedder** (`src/embedding/mod.rs`) — `Embedder` enum: `Http(EmbeddingClient)` for OpenAI-compatible APIs, or `Disabled` for lexical-only mode. Auto-detected from `EMBEDDING_URL` env var. Batches 100 texts per request. Empty env vars treated as unset.

**Vector Store** (`src/vectordb/`) — `VectorStore` enum: `Local(LocalStore)` for brute-force in-memory cosine similarity with JSON disk persistence (~75MB for 50K chunks at 384-dim), or `Milvus(MilvusClient)` for remote Milvus/Zilliz Cloud. Auto-detected from `MILVUS_URL` env var.

**Lexical Search** (`src/lexical/mod.rs`) — Tantivy-based BM25 index for keyword/symbol search.

**Hybrid Fusion** (`src/mcp/hybrid.rs`) — Reciprocal Rank Fusion (RRF) combining semantic and lexical results. Works correctly when either source is empty.

**Incremental Indexing** (`src/mcp/manifest.rs`) — File-hash manifest stored at `.sindexer/index-manifest.json`. Tracks SHA-256 per file to skip unchanged files on reindex. Pass `force: true` to bypass.

## Key Files

- `src/main.rs` — MCP server entry point, rmcp stdio transport
- `src/types.rs` — CodeChunk, EmbeddingVector, IndexStatus
- `src/config.rs` — walker/splitter configuration
- `src/mcp/state.rs` — shared async state with Embedder/VectorStore enums
- `src/mcp/indexer.rs` — indexing pipeline (lexical-only or full)
- `src/mcp/hybrid.rs` — hybrid search (semantic + lexical fusion)
- `src/mcp/manifest.rs` — index manifest for incremental reindexing
- `src/mcp/tools.rs` — MCP tool definitions and JSON schemas
- `src/vectordb/local.rs` — brute-force local vector store with disk persistence
- `src/vectordb/client.rs` — Milvus/Zilliz REST API client

## MCP Tools

- **index_codebase** — walk, split, embed, and store a directory (incremental by default)
- **search_code** — hybrid search (semantic + lexical) over indexed codebase
- **get_indexing_status** — check indexing progress
- **clear_index** — remove indexed data (vector + lexical)

## Environment Variables

All optional. The server works with zero configuration.

- `EMBEDDING_URL` — Embedding API base URL. Setting this enables semantic search. Any OpenAI-compatible endpoint (Jina, OpenAI, local).
- `EMBEDDING_API_KEY` — API key for the embedding endpoint. Not needed for local servers.
- `EMBEDDING_MODEL` — Model name (default: `all-minilm`). Production uses `jina-code-embeddings-1.5b`.
- `EMBEDDING_DIMENSION` — Vector dimension (default: `384`). Must match your model's output. Jina code embeddings use `1536`.
- `MILVUS_URL` — Milvus/Zilliz Cloud endpoint. Setting this uses Milvus instead of the local vector store.
- `MILVUS_TOKEN` — Authentication token for Milvus/Zilliz.
- `MAX_FILE_SIZE` — Maximum file size in bytes (default: `1048576`, 1MB).
- `LOG_LEVEL` — Logging verbosity: trace/debug/info/warn/error (default: `info`).

## Tests

```bash
cargo test              # all tests
cargo test walker       # file discovery
cargo test splitter     # AST parsing
cargo test embedding    # embedding client
cargo test local        # local vector store
cargo test lexical      # BM25 search
```

## Dependencies

- **Core:** rmcp 1.5.0, tokio, rayon, ignore
- **Parsing:** tree-sitter + language grammars
- **HTTP:** reqwest with connection pooling and rustls-tls
- **Concurrency:** dashmap, parking_lot
- **Lexical:** tantivy

## rmcp Usage Patterns

This project uses rmcp 1.5.0 with its built-in stdio transport. Key pattern:

```rust
use rmcp::transport::io::stdio;
use rmcp::ServiceExt;

let tools = MyHandler::new();
let service = tools.serve(stdio()).await?;
service.waiting().await?;
```

Tools are defined with `#[tool_router]` and `#[tool]` macros on a `#[derive(Clone)]` struct holding a `ToolRouter<Self>`.

## Code Quality

**Less code is better.** This codebase should be minimal and focused:

- Prefer single-file modules over directory structures
- Delete unused code rather than commenting it out
- No placeholder/stub implementations — working code only
- Every line must serve a purpose
- If something can be done in 10 lines instead of 100, do it in 10
- Avoid abstractions until needed 3+ times
- No backwards compatibility shims — just change the code

**The goal is a fast, minimal MCP server — not a framework.**
