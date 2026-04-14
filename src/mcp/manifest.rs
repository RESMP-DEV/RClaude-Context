//! Persistent local manifests for indexed codebases.
//!
//! Manifests are keyed by canonical absolute codebase path and stored in a
//! machine-local cache directory so they survive process restarts without
//! polluting the repository.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{Config, DEFAULT_IGNORE_PATTERNS, EXTENSIONLESS_FILES, SUPPORTED_EXTENSIONS};

/// Current manifest schema version.
pub const MANIFEST_FORMAT_VERSION: u32 = 1;

/// Fingerprint for a single indexed file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileFingerprint {
    /// Hex-encoded SHA-256 of the file contents.
    pub content_sha256: String,
    /// File size at indexing time.
    pub size_bytes: u64,
}

/// Persistent manifest for a fully indexed codebase.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexManifest {
    /// Canonical absolute codebase path this manifest belongs to.
    pub codebase_path: PathBuf,
    /// Collection name used in the vector store.
    pub collection_name: String,
    /// Manifest schema version.
    pub manifest_format_version: u32,
    /// Stable hash of indexing-relevant configuration.
    pub config_hash: String,
    /// Completion time of the last successful full index.
    pub last_indexed_unix_ms: u64,
    /// Relative-path keyed file fingerprints for indexed files.
    pub file_fingerprints: BTreeMap<String, FileFingerprint>,
}

/// Diff between a previous manifest and the current walked file set.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ManifestDiff {
    pub added: BTreeSet<String>,
    pub modified: BTreeSet<String>,
    pub deleted: BTreeSet<String>,
}

/// Machine-local manifest storage.
#[derive(Debug, Clone)]
pub struct ManifestStore {
    root: PathBuf,
}

impl ManifestStore {
    /// Create a store rooted at an explicit local directory.
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Create a store rooted in the default local cache location.
    pub fn default() -> Self {
        Self::new(default_manifest_root())
    }

    /// Return the local manifest root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Return the manifest file path for a codebase.
    pub fn path_for_codebase(&self, codebase_path: &Path) -> Result<PathBuf> {
        let absolute_path = canonical_codebase_path(codebase_path)?;
        let mut hasher = Sha256::new();
        hasher.update(absolute_path.to_string_lossy().as_bytes());
        let digest = hex::encode(hasher.finalize());
        Ok(self.root.join(format!("{digest}.json")))
    }

    /// Load the manifest for a codebase if it exists.
    pub fn load(&self, codebase_path: &Path) -> Result<Option<IndexManifest>> {
        let manifest_path = self.path_for_codebase(codebase_path)?;
        if !manifest_path.exists() {
            return Ok(None);
        }

        let bytes = fs::read(&manifest_path)
            .with_context(|| format!("failed to read manifest {}", manifest_path.display()))?;
        let manifest = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse manifest {}", manifest_path.display()))?;
        Ok(Some(manifest))
    }

    /// Persist a manifest atomically.
    pub fn write(&self, manifest: &IndexManifest) -> Result<()> {
        fs::create_dir_all(&self.root)
            .with_context(|| format!("failed to create manifest dir {}", self.root.display()))?;

        let manifest_path = self.path_for_codebase(&manifest.codebase_path)?;
        let tmp_path = manifest_path.with_extension(format!(
            "tmp-{}-{}",
            std::process::id(),
            current_unix_ms()
        ));

        let bytes = serde_json::to_vec_pretty(manifest).context("failed to serialize manifest")?;
        fs::write(&tmp_path, bytes)
            .with_context(|| format!("failed to write temp manifest {}", tmp_path.display()))?;
        if manifest_path.exists() {
            fs::remove_file(&manifest_path).with_context(|| {
                format!(
                    "failed to replace existing manifest {}",
                    manifest_path.display()
                )
            })?;
        }
        fs::rename(&tmp_path, &manifest_path).with_context(|| {
            format!(
                "failed to move manifest {} -> {}",
                tmp_path.display(),
                manifest_path.display()
            )
        })?;

        Ok(())
    }

    /// Build and persist a manifest for a walked file set.
    pub fn write_for_files(
        &self,
        codebase_path: &Path,
        collection_name: &str,
        config: &Config,
        files: &[PathBuf],
    ) -> Result<IndexManifest> {
        let manifest = IndexManifest::from_files(codebase_path, collection_name, config, files)?;
        self.write(&manifest)?;
        Ok(manifest)
    }

    /// Remove the manifest for a codebase if it exists.
    pub fn remove(&self, codebase_path: &Path) -> Result<()> {
        let manifest_path = self.path_for_codebase(codebase_path)?;
        match fs::remove_file(&manifest_path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err)
                .with_context(|| format!("failed to remove manifest {}", manifest_path.display())),
        }
    }
}

impl IndexManifest {
    /// Build a manifest from a current file walk.
    pub fn from_files(
        codebase_path: &Path,
        collection_name: &str,
        config: &Config,
        files: &[PathBuf],
    ) -> Result<Self> {
        let absolute_path = canonical_codebase_path(codebase_path)?;
        let mut file_fingerprints = BTreeMap::new();

        for file_path in files {
            let relative_path = normalize_relative_path(codebase_path, &absolute_path, file_path)?;
            file_fingerprints.insert(relative_path, fingerprint_file(file_path)?);
        }

        Ok(Self {
            codebase_path: absolute_path,
            collection_name: collection_name.to_string(),
            manifest_format_version: MANIFEST_FORMAT_VERSION,
            config_hash: config_hash(config),
            last_indexed_unix_ms: current_unix_ms(),
            file_fingerprints,
        })
    }

    /// Return whether this manifest still matches the current indexing inputs.
    pub fn matches_index_inputs(
        &self,
        codebase_path: &Path,
        collection_name: &str,
        config: &Config,
    ) -> Result<bool> {
        Ok(self.manifest_format_version == MANIFEST_FORMAT_VERSION
            && self.codebase_path == canonical_codebase_path(codebase_path)?
            && self.collection_name == collection_name
            && self.config_hash == config_hash(config))
    }
}

/// Compute the diff between two manifests.
pub fn diff_manifests(previous: &IndexManifest, current: &IndexManifest) -> ManifestDiff {
    let previous_paths: BTreeSet<_> = previous.file_fingerprints.keys().cloned().collect();
    let current_paths: BTreeSet<_> = current.file_fingerprints.keys().cloned().collect();

    let added = current_paths
        .difference(&previous_paths)
        .cloned()
        .collect::<BTreeSet<_>>();
    let deleted = previous_paths
        .difference(&current_paths)
        .cloned()
        .collect::<BTreeSet<_>>();
    let modified = current_paths
        .intersection(&previous_paths)
        .filter(|path| {
            previous.file_fingerprints.get(*path) != current.file_fingerprints.get(*path)
        })
        .cloned()
        .collect::<BTreeSet<_>>();

    ManifestDiff {
        added,
        modified,
        deleted,
    }
}

/// Compute the file diff between a previous manifest and the current walked file set.
pub fn diff_manifest_against_files(
    previous: &IndexManifest,
    codebase_path: &Path,
    collection_name: &str,
    config: &Config,
    files: &[PathBuf],
) -> Result<ManifestDiff> {
    let current = IndexManifest::from_files(codebase_path, collection_name, config, files)?;
    Ok(diff_manifests(previous, &current))
}

fn fingerprint_file(path: &Path) -> Result<FileFingerprint> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read file for fingerprint {}", path.display()))?;
    Ok(FileFingerprint {
        content_sha256: hash_bytes(&bytes),
        size_bytes: bytes.len() as u64,
    })
}

fn normalize_relative_path(root_hint: &Path, canonical_root: &Path, path: &Path) -> Result<String> {
    if let Ok(relative) = path.strip_prefix(root_hint) {
        return Ok(path_to_manifest_key(relative));
    }

    let canonical_path = path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", path.display()))?;
    let relative = canonical_path
        .strip_prefix(canonical_root)
        .with_context(|| {
            format!(
                "file {} is not contained within codebase {}",
                canonical_path.display(),
                canonical_root.display()
            )
        })?;
    Ok(path_to_manifest_key(relative))
}

fn path_to_manifest_key(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn canonical_codebase_path(path: &Path) -> Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("failed to canonicalize codebase path {}", path.display()))
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn config_hash(config: &Config) -> String {
    let serialized = serde_json::json!({
        "embedding_url": config.embedding_url,
        "embedding_model": config.embedding_model,
        "milvus_url": config.milvus_url,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "batch_size": config.batch_size,
        "concurrency": config.concurrency,
        "max_file_size": config.max_file_size,
        "follow_symlinks": config.follow_symlinks,
        "parallelism": config.parallelism,
        "embedding_dimension": config.embedding_dimension,
        "supported_extensions": SUPPORTED_EXTENSIONS,
        "extensionless_files": EXTENSIONLESS_FILES,
        "default_ignore_patterns": DEFAULT_IGNORE_PATTERNS,
    });
    hash_bytes(serialized.to_string().as_bytes())
}

fn default_manifest_root() -> PathBuf {
    platform_cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("rclaude-context")
        .join("index-manifests")
}

fn platform_cache_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    }

    #[cfg(target_os = "macos")]
    {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join("Library").join("Caches"))
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
    }
}
