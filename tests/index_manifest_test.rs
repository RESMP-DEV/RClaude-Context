use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use rclaude_context::config::Config;
use rclaude_context::mcp::{diff_manifest_against_files, IndexManifest, ManifestStore};
use rclaude_context::vectordb::collection_name_from_path;
use tempfile::TempDir;

fn create_file(root: &Path, relative_path: &str, content: &str) {
    let full_path = root.join(relative_path);
    if let Some(parent) = full_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(full_path, content).unwrap();
}

fn collect_files(root: &Path) -> Vec<PathBuf> {
    fn visit(path: &Path, out: &mut Vec<PathBuf>) {
        let mut entries: Vec<_> = fs::read_dir(path)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        entries.sort();

        for entry in entries {
            if entry.is_dir() {
                visit(&entry, out);
            } else {
                out.push(entry);
            }
        }
    }

    let mut files = Vec::new();
    visit(root, &mut files);
    files
}

fn temp_codebase() -> TempDir {
    let temp = TempDir::new().unwrap();
    fs::create_dir_all(temp.path().join("repo")).unwrap();
    temp
}

fn build_manifest(repo: &Path, config: &Config) -> IndexManifest {
    let files = collect_files(repo);
    let collection_name = collection_name_from_path(repo);
    IndexManifest::from_files(repo, &collection_name, config, &files).unwrap()
}

#[test]
fn test_manifest_path_is_derived_from_absolute_codebase_path() {
    let temp = temp_codebase();
    let repo = temp.path().join("repo");
    let same_repo_with_dot = repo.join(".");
    let store = ManifestStore::new(temp.path().join("cache"));

    let manifest_path = store.path_for_codebase(&repo).unwrap();
    let normalized_manifest_path = store.path_for_codebase(&same_repo_with_dot).unwrap();

    let sibling_temp = TempDir::new().unwrap();
    let sibling_repo = sibling_temp.path().join("repo");
    fs::create_dir_all(&sibling_repo).unwrap();
    let sibling_manifest_path = store.path_for_codebase(&sibling_repo).unwrap();

    assert!(manifest_path.is_absolute());
    assert_eq!(manifest_path, normalized_manifest_path);
    assert_ne!(manifest_path, sibling_manifest_path);
    assert!(!manifest_path.starts_with(&repo));
}

#[test]
fn test_default_store_is_local_and_not_in_repo() {
    let temp = temp_codebase();
    let repo = temp.path().join("repo");
    let store = ManifestStore::default();

    let manifest_path = store.path_for_codebase(&repo).unwrap();

    assert!(manifest_path.starts_with(store.root()));
    assert!(!manifest_path.starts_with(&repo));
}

#[test]
fn test_manifest_round_trip_preserves_file_fingerprints() {
    let temp = temp_codebase();
    let repo = temp.path().join("repo");
    let store = ManifestStore::new(temp.path().join("cache"));
    create_file(&repo, "src/lib.rs", "pub fn answer() -> u32 { 42 }\n");
    create_file(&repo, "README.md", "# RClaude Context\n");

    let manifest = build_manifest(&repo, &Config::default());
    store.write(&manifest).unwrap();
    let round_tripped = store.load(&repo).unwrap().unwrap();

    assert_eq!(round_tripped.codebase_path, manifest.codebase_path);
    assert_eq!(round_tripped.collection_name, manifest.collection_name);
    assert_eq!(
        round_tripped.manifest_format_version,
        manifest.manifest_format_version
    );
    assert_eq!(round_tripped.config_hash, manifest.config_hash);
    assert_eq!(round_tripped.file_fingerprints, manifest.file_fingerprints);
    assert!(round_tripped.last_indexed_unix_ms > 0);
}

#[test]
fn test_added_modified_deleted_files_are_detected() {
    let temp = temp_codebase();
    let repo = temp.path().join("repo");
    create_file(&repo, "src/kept.rs", "pub fn kept() {}\n");
    create_file(&repo, "src/modified.rs", "pub fn version() -> u8 { 1 }\n");
    create_file(&repo, "src/deleted.rs", "pub fn delete_me() {}\n");

    let config = Config::default();
    let manifest = build_manifest(&repo, &config);

    create_file(&repo, "src/modified.rs", "pub fn version() -> u8 { 2 }\n");
    fs::remove_file(repo.join("src/deleted.rs")).unwrap();
    create_file(&repo, "src/added.rs", "pub fn added() {}\n");

    let files = collect_files(&repo);
    let collection_name = collection_name_from_path(&repo);
    let delta =
        diff_manifest_against_files(&manifest, &repo, &collection_name, &config, &files).unwrap();

    assert_eq!(delta.added, BTreeSet::from(["src/added.rs".to_string()]));
    assert_eq!(
        delta.modified,
        BTreeSet::from(["src/modified.rs".to_string()])
    );
    assert_eq!(
        delta.deleted,
        BTreeSet::from(["src/deleted.rs".to_string()])
    );
}

#[test]
fn test_config_hash_change_invalidates_manifest_reuse() {
    let temp = temp_codebase();
    let repo = temp.path().join("repo");
    create_file(&repo, "src/lib.rs", "pub fn fingerprinted() {}\n");

    let base_config = Config::default();
    let manifest = build_manifest(&repo, &base_config);
    let collection_name = collection_name_from_path(&repo);

    let mut changed_config = base_config.clone();
    changed_config.chunk_size += 128;

    assert!(manifest
        .matches_index_inputs(&repo, &collection_name, &base_config)
        .unwrap());
    assert!(!manifest
        .matches_index_inputs(&repo, &collection_name, &changed_config)
        .unwrap());
}

#[test]
fn test_store_can_write_and_remove_manifest_for_files() {
    let temp = temp_codebase();
    let repo = temp.path().join("repo");
    let store = ManifestStore::new(temp.path().join("cache"));
    create_file(&repo, "src/lib.rs", "pub fn persisted() {}\n");

    let files = collect_files(&repo);
    let manifest = store
        .write_for_files(
            &repo,
            &collection_name_from_path(&repo),
            &Config::default(),
            &files,
        )
        .unwrap();

    assert_eq!(store.load(&repo).unwrap(), Some(manifest));

    store.remove(&repo).unwrap();
    assert_eq!(store.load(&repo).unwrap(), None);
}
