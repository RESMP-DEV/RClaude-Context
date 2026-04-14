use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use crate::types::CodeChunk;

/// Dense or lexical hit participating in hybrid fusion.
#[derive(Clone, Debug)]
pub struct HybridHit {
    /// The recalled code chunk.
    pub chunk: CodeChunk,
    /// Modality-local score. Higher is better.
    pub score: f32,
}

/// Deterministic knobs for hybrid rank fusion.
#[derive(Clone, Debug)]
pub struct HybridFusionOptions {
    /// Maximum number of fused results to return.
    pub limit: usize,
    /// Allowed file extensions without the leading dot. Empty means no filter.
    pub extension_filter: Vec<String>,
    /// Reciprocal-rank-fusion constant.
    pub rrf_k: usize,
    /// Additional score for an exact identifier match in the chunk content.
    pub exact_symbol_boost: f32,
    /// Additional score when the filename or path matches the query.
    pub filename_match_boost: f32,
}

impl Default for HybridFusionOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            extension_filter: Vec::new(),
            rrf_k: 60,
            exact_symbol_boost: 0.35,
            filename_match_boost: 0.08,
        }
    }
}

/// Final fused result for a code chunk.
#[derive(Clone, Debug)]
pub struct HybridSearchResult {
    /// The code chunk returned to the caller.
    pub chunk: CodeChunk,
    /// Final fused score after RRF and lexical boosts.
    pub score: f32,
    /// 1-based rank in the dense/vector result list after dedupe.
    pub vector_rank: Option<usize>,
    /// 1-based rank in the lexical result list after dedupe.
    pub lexical_rank: Option<usize>,
    /// Whether the query matched an identifier exactly in the chunk content.
    pub exact_symbol_match: bool,
    /// Whether the query matched the filename or relative path.
    pub filename_match: bool,
}

#[derive(Clone, Debug)]
struct Accumulator {
    chunk: CodeChunk,
    score: f32,
    vector_rank: Option<usize>,
    lexical_rank: Option<usize>,
    exact_symbol_match: bool,
    filename_match: bool,
}

/// Fuse vector recall with lexical recall using deterministic reciprocal rank fusion.
///
/// The helper is intentionally pure so tests can validate the public `search_code`
/// behavior without depending on Milvus, embeddings, or network availability.
pub fn fuse_hybrid_hits(
    query: &str,
    vector_hits: Vec<HybridHit>,
    lexical_hits: Vec<HybridHit>,
    options: &HybridFusionOptions,
) -> Vec<HybridSearchResult> {
    let extension_filter = normalize_extensions(&options.extension_filter);
    let vector_hits = dedupe_and_sort_hits(vector_hits, &extension_filter);
    let lexical_hits = dedupe_and_sort_hits(lexical_hits, &extension_filter);

    let mut fused: BTreeMap<String, Accumulator> = BTreeMap::new();

    for (index, hit) in vector_hits.iter().enumerate() {
        let entry = fused
            .entry(hit.chunk.id.clone())
            .or_insert_with(|| Accumulator {
                chunk: hit.chunk.clone(),
                score: 0.0,
                vector_rank: None,
                lexical_rank: None,
                exact_symbol_match: false,
                filename_match: false,
            });
        entry.vector_rank.get_or_insert(index + 1);
        entry.score += reciprocal_rank_score(options.rrf_k, index);
    }

    for (index, hit) in lexical_hits.iter().enumerate() {
        let entry = fused
            .entry(hit.chunk.id.clone())
            .or_insert_with(|| Accumulator {
                chunk: hit.chunk.clone(),
                score: 0.0,
                vector_rank: None,
                lexical_rank: None,
                exact_symbol_match: false,
                filename_match: false,
            });
        entry.lexical_rank.get_or_insert(index + 1);
        entry.score += reciprocal_rank_score(options.rrf_k, index);
    }

    for entry in fused.values_mut() {
        entry.exact_symbol_match = contains_exact_symbol(&entry.chunk.content, query);
        entry.filename_match = matches_filename(&entry.chunk, query);
        if entry.exact_symbol_match {
            entry.score += options.exact_symbol_boost;
        }
        if entry.filename_match {
            entry.score += options.filename_match_boost;
        }
    }

    let mut ranked: Vec<HybridSearchResult> = fused
        .into_values()
        .map(|entry| HybridSearchResult {
            chunk: entry.chunk,
            score: entry.score,
            vector_rank: entry.vector_rank,
            lexical_rank: entry.lexical_rank,
            exact_symbol_match: entry.exact_symbol_match,
            filename_match: entry.filename_match,
        })
        .collect();

    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| right.exact_symbol_match.cmp(&left.exact_symbol_match))
            .then_with(|| right.filename_match.cmp(&left.filename_match))
            .then_with(|| {
                left.vector_rank
                    .unwrap_or(usize::MAX)
                    .cmp(&right.vector_rank.unwrap_or(usize::MAX))
            })
            .then_with(|| {
                left.lexical_rank
                    .unwrap_or(usize::MAX)
                    .cmp(&right.lexical_rank.unwrap_or(usize::MAX))
            })
            .then_with(|| left.chunk.relative_path.cmp(&right.chunk.relative_path))
            .then_with(|| left.chunk.start_line.cmp(&right.chunk.start_line))
            .then_with(|| left.chunk.id.cmp(&right.chunk.id))
    });

    ranked.truncate(options.limit);
    ranked
}

fn dedupe_and_sort_hits(
    hits: Vec<HybridHit>,
    extension_filter: &HashSet<String>,
) -> Vec<HybridHit> {
    let mut deduped: BTreeMap<String, HybridHit> = BTreeMap::new();

    for hit in hits {
        if !extension_allowed(&hit.chunk, extension_filter) {
            continue;
        }

        match deduped.get_mut(&hit.chunk.id) {
            Some(existing) => {
                if hit.score > existing.score
                    || (hit.score == existing.score
                        && chunk_sort_key(&hit.chunk) < chunk_sort_key(&existing.chunk))
                {
                    *existing = hit;
                }
            }
            None => {
                deduped.insert(hit.chunk.id.clone(), hit);
            }
        }
    }

    let mut hits: Vec<HybridHit> = deduped.into_values().collect();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| chunk_sort_key(&left.chunk).cmp(&chunk_sort_key(&right.chunk)))
    });
    hits
}

fn chunk_sort_key(chunk: &CodeChunk) -> (&str, u32, &str) {
    (&chunk.relative_path, chunk.start_line, &chunk.id)
}

fn reciprocal_rank_score(k: usize, index: usize) -> f32 {
    1.0 / (k as f32 + index as f32 + 1.0)
}

fn normalize_extensions(extensions: &[String]) -> HashSet<String> {
    extensions
        .iter()
        .map(|ext| ext.trim_start_matches('.').to_ascii_lowercase())
        .filter(|ext| !ext.is_empty())
        .collect()
}

fn extension_allowed(chunk: &CodeChunk, extension_filter: &HashSet<String>) -> bool {
    if extension_filter.is_empty() {
        return true;
    }

    chunk_extension(chunk)
        .map(|ext| extension_filter.contains(&ext))
        .unwrap_or(false)
}

fn chunk_extension(chunk: &CodeChunk) -> Option<String> {
    Path::new(&chunk.relative_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .or_else(|| chunk.file_path.extension().and_then(|ext| ext.to_str()))
        .map(|ext| ext.to_ascii_lowercase())
}

fn contains_exact_symbol(content: &str, query: &str) -> bool {
    let needle = query.trim().to_ascii_lowercase();
    if needle.is_empty() {
        return false;
    }

    let haystack = content.to_ascii_lowercase();
    haystack.match_indices(&needle).any(|(index, _)| {
        let before = haystack[..index].chars().next_back();
        let after = haystack[index + needle.len()..].chars().next();
        !before.map(is_identifier_char).unwrap_or(false)
            && !after.map(is_identifier_char).unwrap_or(false)
    })
}

fn matches_filename(chunk: &CodeChunk, query: &str) -> bool {
    let query = query.trim().to_ascii_lowercase();
    if query.is_empty() {
        return false;
    }

    let relative_path = chunk.relative_path.to_ascii_lowercase();
    let file_stem = Path::new(&chunk.relative_path)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if file_stem == query || file_stem.contains(&query) || relative_path.contains(&query) {
        return true;
    }

    query
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|token| token.len() >= 2)
        .any(|token| file_stem.contains(token) || relative_path.contains(token))
}

fn is_identifier_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}
