use std::path::PathBuf;

use sindexer::mcp::hybrid::{fuse_hybrid_hits, HybridFusionOptions, HybridHit};
use sindexer::types::CodeChunk;

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
fn test_fuse_merges_duplicate_ids() {
    let semantic_hits = vec![hit("chunk-1", "src/lib.rs", "fn main() {}", 0.9, 1)];
    let lexical_hits = vec![hit("chunk-1", "src/lib.rs", "fn main() {}", 12.0, 1)];

    let fused = fuse_hybrid_hits(
        "main",
        semantic_hits,
        lexical_hits,
        &HybridFusionOptions {
            limit: 10,
            ..Default::default()
        },
    );

    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].chunk.id, "chunk-1");
}

#[test]
fn test_fuse_respects_limit() {
    let semantic_hits = vec![
        hit("chunk-1", "src/a.rs", "fn a() {}", 0.9, 1),
        hit("chunk-2", "src/b.rs", "fn b() {}", 0.8, 2),
    ];
    let lexical_hits = vec![hit("chunk-3", "src/c.rs", "fn c() {}", 10.0, 3)];

    let fused = fuse_hybrid_hits(
        "function",
        semantic_hits,
        lexical_hits,
        &HybridFusionOptions {
            limit: 2,
            ..Default::default()
        },
    );

    assert_eq!(fused.len(), 2);
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
        limit: 10,
    };

    let results = fuse_hybrid_hits("hybrid_search", vector_hits, lexical_hits, &options);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.id, "rust-hit");
    assert!(results[0].chunk.relative_path.ends_with(".rs"));
}
