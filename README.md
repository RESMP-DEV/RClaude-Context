# (Rust Semantic Indexer) rust_sindexer

A high-performance Rust MCP server for semantic code indexing and search. Drop-in replacement for [`@zilliz/claude-context-mcp`](https://www.npmjs.com/package/@zilliz/claude-context-mcp) — single native binary, no Node.js required.

## Why rust_sindexer?

The official JS-based Claude Context MCP has several pain points:

- **Node.js overhead** — each `npx` invocation spawns a full Node.js process
- **No real incremental indexing** — re-indexes everything on each run
- **Shallow .gitignore support** — only root-level, no nested ignore files
- **Startup latency** — npm download/cache step on cold starts
- **gRPC timeout issues** — `DEADLINE_EXCEEDED` errors during indexing

rust_sindexer fixes all of these:

- **Single binary** — no runtime dependencies, ~37MB native executable
- **Incremental indexing** — SHA-256 manifest tracks file changes, skips unchanged files
- **Full .gitignore support** — via the `ignore` crate with nested directory support
- **Instant startup** — native binary, no package manager involved
- **REST API for Milvus** — no gRPC, no timeout issues
- **Parallel everything** — Rayon-based parallel file walking, AST parsing, and chunk extraction
- **Hybrid search** — BM25 lexical + semantic vector search with Reciprocal Rank Fusion
- **Any embedding provider** — works with any OpenAI-compatible API (cloud or local)

## Supported Languages

Tree-sitter parsers for AST-aware code chunking:

Python, JavaScript, TypeScript, TSX, Rust, Go, Java, C, C++, Ruby, PHP, Swift, Scala, C#

## Installation

### From source

```bash
git clone https://github.com/RESMP-DEV/rust_sindexer
cd rust_sindexer
cargo build --release
# Binary at target/release/rust_sindexer
```

### Via cargo install

```bash
cargo install rust_sindexer
```

## Prerequisites

### Embedding Provider

Any service that speaks the OpenAI `/v1/embeddings` format:

**Cloud providers:**

```bash
# OpenAI
EMBEDDING_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Jina AI (free tier available)
EMBEDDING_URL=https://api.jina.ai/v1
EMBEDDING_API_KEY=jina_xxx
EMBEDDING_MODEL=jina-code-embeddings-1.5b
EMBEDDING_DIMENSION=1536

# Voyage AI
EMBEDDING_URL=https://api.voyageai.com/v1
EMBEDDING_API_KEY=pa-xxx
EMBEDDING_MODEL=voyage-code-3
EMBEDDING_DIMENSION=1024

# Cohere (via OpenAI-compatible proxy)
# Google Gemini (via OpenAI-compatible proxy)
# Any other provider with an OpenAI-compatible endpoint
```

**Local / self-hosted:**

```bash
# Ollama
ollama pull nomic-embed-text
EMBEDDING_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# vLLM
EMBEDDING_URL=http://localhost:8000/v1
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768

# Text Embeddings Inference (TEI)
EMBEDDING_URL=http://localhost:8080/v1
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768

# sentence-transformers
EMBEDDING_URL=http://localhost:8080/v1
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

No API key is needed for local servers — just omit `EMBEDDING_API_KEY`.

### Vector Database

Milvus for vector storage. Either managed or self-hosted:

```bash
# Zilliz Cloud (managed Milvus, free tier available)
MILVUS_URL=https://your-cluster.zillizcloud.com:443
MILVUS_TOKEN=your-api-key

# Self-hosted Milvus via Docker
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
MILVUS_URL=http://localhost:19530
# No MILVUS_TOKEN needed for local unauthenticated instances
```

## Configuration

All configuration via environment variables:

- `EMBEDDING_URL` — embedding API base URL (default: `http://localhost:8080/v1`)
- `EMBEDDING_API_KEY` — API key for the embedding endpoint (omit for local servers)
- `EMBEDDING_MODEL` — model name to request
- `EMBEDDING_DIMENSION` — vector dimension (default: `384`). Must match your model's output.
- `MILVUS_URL` — Milvus endpoint (default: `http://localhost:19530`)
- `MILVUS_TOKEN` — Milvus auth token (omit for local unauthenticated instances)
- `MAX_FILE_SIZE` — max file size to process in bytes (default: `1048576` / 1MB)
- `LOG_LEVEL` — trace/debug/info/warn/error (default: `info`)

## Usage

### Global MCP (all tools — Claude Code, Codex, Copilot)

Add to `~/.mcp.json`:

```json
{
  "mcpServers": {
    "claude-context": {
      "type": "stdio",
      "command": "/path/to/rust_sindexer",
      "args": [],
      "env": {
        "EMBEDDING_URL": "https://api.openai.com/v1",
        "EMBEDDING_API_KEY": "your-key",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIMENSION": "1536",
        "MILVUS_URL": "https://your-cluster.zillizcloud.com:443",
        "MILVUS_TOKEN": "your-token"
      }
    }
  }
}
```

### Claude Code only

Add to `~/.claude/mcp.json` instead.

### Project-level

Add to `.mcp.json` in the project root.

### MCP Tools

Once configured, the server exposes four tools:

- **`index_codebase`** — Starts indexing a directory in the background. Incremental by default; pass `force: true` to rebuild from scratch. Poll `get_indexing_status` until the status becomes `completed` or `failed`.
- **`search_code`** — Hybrid search (semantic + BM25 lexical) with Reciprocal Rank Fusion. Returns code chunks with file paths, line numbers, and relevance scores. If semantic search is unavailable but the lexical index exists, the server falls back to lexical-only results.
- **`get_indexing_status`** — Check whether indexing is in progress, completed, or not started.
- **`clear_index`** — Remove all indexed data (vectors + lexical index) for a codebase.

### MCP-specific behavior

`index_codebase` intentionally returns quickly instead of holding the MCP stdio request open until the whole repository is indexed. Long-running MCP tool calls are a common source of client-side failures and timeouts even when the underlying indexing pipeline works correctly.

Use this pattern:

1. Call `index_codebase` with an absolute repository path.
2. Poll `get_indexing_status` for the same absolute path.
3. Start calling `search_code` once the status is `completed`.

All tool paths must be absolute filesystem paths. Relative paths are rejected.

## Migrating from @zilliz/claude-context-mcp

### 1. Build the binary

```bash
git clone https://github.com/RESMP-DEV/rust_sindexer
cd rust_sindexer
cargo build --release
```

### 2. Replace your MCP configuration

The server name stays `claude-context` so existing tool references keep working.

**Before** (JS version):

```json
{
  "mcpServers": {
    "claude-context": {
      "command": "npx",
      "args": ["@zilliz/claude-context-mcp@latest"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "MILVUS_ADDRESS": "https://your-cluster.zillizcloud.com",
        "MILVUS_TOKEN": "your-token"
      }
    }
  }
}
```

**After** (rust_sindexer):

```json
{
  "mcpServers": {
    "claude-context": {
      "type": "stdio",
      "command": "/path/to/rust_sindexer",
      "args": [],
      "env": {
        "EMBEDDING_URL": "https://api.openai.com/v1",
        "EMBEDDING_API_KEY": "sk-...",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIMENSION": "1536",
        "MILVUS_URL": "https://your-cluster.zillizcloud.com:443",
        "MILVUS_TOKEN": "your-token"
      }
    }
  }
}
```

### 3. Clear and re-index

Since the embedding model and chunk boundaries differ, clear your existing index and re-index:

```
> clear_index for /path/to/project
> index_codebase for /path/to/project
> get_indexing_status for /path/to/project
```

Subsequent runs will be incremental — only changed files get re-indexed.

### Environment variable mapping

The JS version uses different variable names. Here's the mapping:

- `OPENAI_API_KEY` → `EMBEDDING_API_KEY` (any OpenAI-compatible endpoint works, not just OpenAI)
- `OPENAI_BASE_URL` → `EMBEDDING_URL` (full base URL to the embedding API)
- `EMBEDDING_MODEL` → `EMBEDDING_MODEL` (same name)
- (new) `EMBEDDING_DIMENSION` — required, must match your model's output dimension
- `MILVUS_ADDRESS` → `MILVUS_URL` (full URL including port)
- `MILVUS_TOKEN` → `MILVUS_TOKEN` (same name)

### What stays the same

- Tool names: `index_codebase`, `search_code`, `get_indexing_status`, `clear_index`
- Core parameters: `path`, `query`, `limit`, `force`
- Config file locations: `~/.mcp.json`, `~/.claude/mcp.json`, `.mcp.json`

### What changes

- `extensionFilter: [".ts", ".py"]` becomes `extensions: ["ts", "py"]` (no dot prefix)
- No `EMBEDDING_PROVIDER` selection — uses any OpenAI-compatible HTTP endpoint
- No `~/.context/.env` file — all config via environment variables

## Architecture

```
                     ┌─────────────────┐
                     │   MCP Client    │
                     │ (Claude, Codex, │
                     │  Copilot, etc.) │
                     └────────┬────────┘
                              │ stdio
                     ┌────────▼────────┐
                     │  rust_sindexer  │
                     │   (MCP Server)  │
                     └────────┬────────┘
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼────────┐ ┌────▼────┐ ┌────────▼────────┐
     │  File Walker    │ │ Splitter│ │   Embedding     │
     │ (ignore + rayon)│ │ (tree-  │ │   Client        │
     │                 │ │ sitter) │ │   (reqwest)     │
     └─────────────────┘ └─────────┘ └────────┬────────┘
                                              │
                              ┌───────────────┼──────────────┐
                              │                              │
                     ┌────────▼────────┐           ┌────────▼────────┐
                     │     Milvus      │           │    Tantivy      │
                     │  (Vector Store) │           │   (BM25 Index)  │
                     └─────────────────┘           └─────────────────┘
                              │                              │
                              └──────────────┬───────────────┘
                                    ┌────────▼────────┐
                                    │  Hybrid Fusion  │
                                    │   (RRF Merge)   │
                                    └─────────────────┘
```

## Tests

```bash
cargo test              # All tests
cargo test walker       # File discovery
cargo test splitter     # AST parsing
cargo test embedding    # Embedding client
```

## License

MIT
