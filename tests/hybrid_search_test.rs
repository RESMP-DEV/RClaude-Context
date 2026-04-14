use std::path::PathBuf;

use rclaude_context::mcp::{fuse_hybrid_hits, HybridFusionOptions, HybridHit};
use rclaude_context::types::CodeChunk;

fn chunk(id: &str, relative_path: &str, content: &str, start_line: u32) -> CodeChunk {
    CodeChunk {
        id: id.to_string(),
        content: content.to_string(),
        file_path: PathBuf::from(format!("/repo/{relative_path}")),
        relative_path: relative_path.to_string(),
        start_line,
        end_line: start_line + 4,
        language: PathBuf::from(relative_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default()
            .to_string(),
    }
}

fn hit(id: &str, relative_path: &str, content: &str, score: f32, start_line: u32) -> HybridHit {
    HybridHit {
        chunk: chunk(id, relative_path, content, start_line),
        score,
    }
}

#[test]
fn test_exact_symbol_match_outranks_loose_semantic_match() {
    let vector_hits = vec![
        hit(
            "loose-semantic",
            "src/config.rs",
            "Search configuration drives the query ranking heuristics and scoring.",
            0.97,
            10,
        ),
        hit(
            "exact-symbol",
            "src/searcher.rs",
            "pub struct SearchParams {\n    pub limit: usize,\n}",
            0.62,
            40,
        ),
    ];
    let lexical_hits = vec![
        hit(
            "exact-symbol",
            "src/searcher.rs",
            "impl SearchParams {\n    pub fn default_limit() -> usize { 10 }\n}",
            0.91,
            60,
        ),
        hit(
            "loose-semantic",
            "src/config.rs",
            "These settings control how search parameters are interpreted.",
            0.35,
            80,
        ),
    ];

    let results = fuse_hybrid_hits(
        "SearchParams",
        vector_hits,
        lexical_hits,
        &HybridFusionOptions::default(),
    );

    assert_eq!(results[0].chunk.id, "exact-symbol");
    assert!(results[0].exact_symbol_match);
    assert_eq!(results[1].chunk.id, "loose-semantic");
}

#[test]
fn test_filename_match_boosts_relevant_chunk() {
    let vector_hits = vec![
        hit(
            "state",
            "src/mcp/state.rs",
            "Search requests share cached context and indexing status.",
            0.93,
            12,
        ),
        hit(
            "searcher",
            "src/mcp/searcher.rs",
            "This module ranks query results and shapes the MCP response.",
            0.89,
            12,
        ),
    ];
    let lexical_hits = vec![
        hit(
            "state",
            "src/mcp/state.rs",
            "Search state snapshots are emitted for dashboard consumers.",
            0.82,
            25,
        ),
        hit(
            "searcher",
            "src/mcp/searcher.rs",
            "Result ordering happens after recall and before serialization.",
            0.80,
            25,
        ),
    ];

    let results = fuse_hybrid_hits(
        "searcher",
        vector_hits,
        lexical_hits,
        &HybridFusionOptions::default(),
    );

    assert_eq!(results[0].chunk.id, "searcher");
    assert!(results[0].filename_match);
    assert_eq!(results[1].chunk.id, "state");
}

#[test]
fn test_duplicate_chunk_ids_are_deduped_before_ranking() {
    let vector_hits = vec![
        hit(
            "dup",
            "src/mcp/searcher.rs",
            "pub fn fuse_ranked_results() {}",
            0.94,
            10,
        ),
        hit(
            "dup",
            "src/mcp/searcher.rs",
            "pub fn fuse_ranked_results() {}",
            0.41,
            10,
        ),
        hit(
            "other",
            "src/mcp/state.rs",
            "pub struct SearchState;",
            0.88,
            22,
        ),
    ];
    let lexical_hits = vec![
        hit(
            "dup",
            "src/mcp/searcher.rs",
            "fuse ranked results merges dense and lexical recall",
            0.79,
            18,
        ),
        hit(
            "other",
            "src/mcp/state.rs",
            "search state caches the active index manifest",
            0.74,
            18,
        ),
    ];

    let results = fuse_hybrid_hits(
        "fuse_ranked_results",
        vector_hits,
        lexical_hits,
        &HybridFusionOptions::default(),
    );

    assert_eq!(results.len(), 2);
    assert_eq!(
        results
            .iter()
            .filter(|result| result.chunk.id == "dup")
            .count(),
        1
    );

    let dup = results
        .iter()
        .find(|result| result.chunk.id == "dup")
        .expect("deduped result should remain");
    let other = results
        .iter()
        .find(|result| result.chunk.id == "other")
        .expect("other result should remain");

    assert_eq!(dup.vector_rank, Some(1));
    assert_eq!(other.vector_rank, Some(2));
}

#[test]
fn test_extension_filter_is_preserved_after_fusion() {
    let vector_hits = vec![
        hit(
            "rust-hit",
            "src/mcp/searcher.rs",
            "pub fn hybrid_search() -> usize { 1 }",
            0.78,
            10,
        ),
        hit(
            "python-hit",
            "scripts/search.py",
            "def hybrid_search():\n    return 1",
            0.77,
            10,
        ),
    ];
    let lexical_hits = vec![
        hit(
            "guide-hit",
            "docs/hybrid-search.md",
            "hybrid_search is documented here with the exact phrase.",
            0.99,
            5,
        ),
        hit(
            "rust-hit",
            "src/mcp/searcher.rs",
            "hybrid_search merges lexical and dense recall.",
            0.65,
            20,
        ),
    ];

    let options = HybridFusionOptions {
        extension_filter: vec!["rs".to_string()],
        ..HybridFusionOptions::default()
    };

    let results = fuse_hybrid_hits("hybrid_search", vector_hits, lexical_hits, &options);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.id, "rust-hit");
    assert!(results[0].chunk.relative_path.ends_with(".rs"));
}
