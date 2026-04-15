use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::splitter;
use crate::types::IndexStatus;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexInputs {
    pub chunk_size: usize,
    pub overlap_lines: usize,
    pub min_chunk_lines: u32,
    pub target_chunk_lines: u32,
    pub extensions: Vec<String>,
    pub ignore_patterns: Vec<String>,
}

impl IndexInputs {
    pub fn from_splitter_and_walker(
        splitter: &splitter::Config,
        extensions: &[String],
        ignore_patterns: &[String],
    ) -> Self {
        Self {
            chunk_size: splitter.max_chunk_bytes,
            overlap_lines: splitter.overlap_lines,
            min_chunk_lines: splitter.min_chunk_lines,
            target_chunk_lines: splitter.target_chunk_lines,
            extensions: extensions.to_vec(),
            ignore_patterns: ignore_patterns.to_vec(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileFingerprint {
    pub relative_path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexManifest {
    pub collection_name: String,
    pub inputs: IndexInputs,
    pub files: Vec<FileFingerprint>,
}

impl IndexManifest {
    pub fn matches_index_inputs(&self, collection_name: &str, inputs: &IndexInputs) -> bool {
        self.collection_name == collection_name && self.inputs == *inputs
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestDiff {
    pub added: Vec<String>,
    pub modified: Vec<String>,
    pub deleted: Vec<String>,
}

impl ManifestDiff {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.modified.is_empty() && self.deleted.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ManifestStore;

impl ManifestStore {
    pub fn load(&self, path: &Path) -> Result<Option<IndexManifest>> {
        let manifest_path = manifest_path(path);
        if !manifest_path.exists() {
            return Ok(None);
        }

        let contents = fs::read_to_string(&manifest_path)
            .with_context(|| format!("failed to read manifest {}", manifest_path.display()))?;
        let manifest = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse manifest {}", manifest_path.display()))?;
        Ok(Some(manifest))
    }

    pub fn write_for_files(
        &self,
        path: &Path,
        collection_name: &str,
        inputs: &IndexInputs,
        files: &[PathBuf],
    ) -> Result<()> {
        let manifest_path = manifest_path(path);
        if let Some(parent) = manifest_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create manifest directory {}", parent.display())
            })?;
        }

        let manifest = IndexManifest {
            collection_name: collection_name.to_string(),
            inputs: inputs.clone(),
            files: fingerprint_files(path, files)?,
        };

        let json =
            serde_json::to_string_pretty(&manifest).context("failed to serialize manifest")?;
        fs::write(&manifest_path, json)
            .with_context(|| format!("failed to write manifest {}", manifest_path.display()))?;
        Ok(())
    }

    pub fn load_status(&self, path: &Path) -> Result<Option<IndexStatus>> {
        let status_path = status_path(path);
        if !status_path.exists() {
            return Ok(None);
        }

        let contents = fs::read_to_string(&status_path)
            .with_context(|| format!("failed to read status {}", status_path.display()))?;
        let status = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse status {}", status_path.display()))?;
        Ok(Some(status))
    }

    pub fn write_status(&self, path: &Path, status: &IndexStatus) -> Result<()> {
        let status_path = status_path(path);
        if let Some(parent) = status_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create status directory {}", parent.display()))?;
        }

        let json =
            serde_json::to_string_pretty(status).context("failed to serialize index status")?;
        fs::write(&status_path, json)
            .with_context(|| format!("failed to write status {}", status_path.display()))?;
        Ok(())
    }

    pub fn clear_status(&self, path: &Path) -> Result<()> {
        let status_path = status_path(path);
        if !status_path.exists() {
            return Ok(());
        }

        fs::remove_file(&status_path)
            .with_context(|| format!("failed to remove status {}", status_path.display()))?;
        Ok(())
    }
}

pub fn diff_manifest_against_files(
    previous: &IndexManifest,
    path: &Path,
    collection_name: &str,
    inputs: &IndexInputs,
    files: &[PathBuf],
) -> Result<ManifestDiff> {
    let current = IndexManifest {
        collection_name: collection_name.to_string(),
        inputs: inputs.clone(),
        files: fingerprint_files(path, files)?,
    };

    let previous_map: BTreeMap<_, _> = previous
        .files
        .iter()
        .map(|file| (file.relative_path.clone(), file.sha256.clone()))
        .collect();
    let current_map: BTreeMap<_, _> = current
        .files
        .iter()
        .map(|file| (file.relative_path.clone(), file.sha256.clone()))
        .collect();

    let previous_paths: BTreeSet<_> = previous_map.keys().cloned().collect();
    let current_paths: BTreeSet<_> = current_map.keys().cloned().collect();

    let added = current_paths
        .difference(&previous_paths)
        .cloned()
        .collect::<Vec<_>>();
    let deleted = previous_paths
        .difference(&current_paths)
        .cloned()
        .collect::<Vec<_>>();
    let modified = current_paths
        .intersection(&previous_paths)
        .filter(|path| previous_map.get(*path) != current_map.get(*path))
        .cloned()
        .collect::<Vec<_>>();

    Ok(ManifestDiff {
        added,
        modified,
        deleted,
    })
}

fn fingerprint_files(root: &Path, files: &[PathBuf]) -> Result<Vec<FileFingerprint>> {
    let mut fingerprints = files
        .iter()
        .map(|path| -> Result<FileFingerprint> {
            let contents = fs::read(path)
                .with_context(|| format!("failed to read file for manifest {}", path.display()))?;
            let relative_path = path
                .strip_prefix(root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            let mut hasher = Sha256::new();
            hasher.update(&contents);
            Ok(FileFingerprint {
                relative_path,
                sha256: hex::encode(hasher.finalize()),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    fingerprints.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    Ok(fingerprints)
}

fn manifest_path(root: &Path) -> PathBuf {
    root.join(".rust_sindexer").join("index-manifest.json")
}

fn status_path(root: &Path) -> PathBuf {
    root.join(".rust_sindexer").join("index-status.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn diff_reports_added_modified_and_deleted_files() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        let src = root.join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("keep.rs"), "fn keep() {}\n").unwrap();
        fs::write(src.join("new.rs"), "fn new_file() {}\n").unwrap();

        let inputs = IndexInputs {
            chunk_size: 512,
            overlap_lines: 3,
            min_chunk_lines: 5,
            target_chunk_lines: 50,
            extensions: vec!["rs".into()],
            ignore_patterns: vec!["target".into()],
        };

        let previous = IndexManifest {
            collection_name: "collection".into(),
            inputs: inputs.clone(),
            files: vec![
                FileFingerprint {
                    relative_path: "src/keep.rs".into(),
                    sha256: "old".into(),
                },
                FileFingerprint {
                    relative_path: "src/deleted.rs".into(),
                    sha256: "gone".into(),
                },
            ],
        };

        let files = vec![src.join("keep.rs"), src.join("new.rs")];
        let diff =
            diff_manifest_against_files(&previous, root, "collection", &inputs, &files).unwrap();

        assert_eq!(diff.added, vec!["src/new.rs"]);
        assert_eq!(diff.deleted, vec!["src/deleted.rs"]);
        assert_eq!(diff.modified, vec!["src/keep.rs"]);
    }

    #[test]
    fn manifest_store_round_trips_and_matches_inputs() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        let src = root.join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("lib.rs"), "fn main() {}\n").unwrap();

        let inputs = IndexInputs {
            chunk_size: 1024,
            overlap_lines: 2,
            min_chunk_lines: 5,
            target_chunk_lines: 40,
            extensions: vec!["rs".into()],
            ignore_patterns: vec!["target".into()],
        };
        let store = ManifestStore;
        let files = vec![src.join("lib.rs")];

        store
            .write_for_files(root, "collection", &inputs, &files)
            .unwrap();
        let manifest = store.load(root).unwrap().unwrap();

        assert!(manifest.matches_index_inputs("collection", &inputs));
        assert!(!manifest.matches_index_inputs("other", &inputs));
        assert_eq!(manifest.files.len(), 1);
        assert_eq!(manifest.files[0].relative_path, "src/lib.rs");
    }

    #[test]
    fn status_store_round_trips() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        let store = ManifestStore;
        let status = IndexStatus {
            total_files: 10,
            processed_files: 4,
            total_chunks: 12,
            embeddings_generated: 6,
            vectors_inserted: 5,
            status: crate::types::IndexState::Indexing,
        };

        store.write_status(root, &status).unwrap();
        let loaded = store.load_status(root).unwrap().unwrap();

        assert_eq!(loaded.total_files, 10);
        assert_eq!(loaded.processed_files, 4);
        assert_eq!(loaded.total_chunks, 12);
        assert_eq!(loaded.embeddings_generated, 6);
        assert_eq!(loaded.vectors_inserted, 5);
        assert_eq!(loaded.status, crate::types::IndexState::Indexing);

        store.clear_status(root).unwrap();
        assert!(store.load_status(root).unwrap().is_none());
    }
}
