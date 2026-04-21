use std::fs;
use std::path::{Path, PathBuf};

use sindexer::mcp::manifest::{diff_manifest_against_files, IndexInputs, ManifestStore};
use tempfile::TempDir;

fn create_file(root: &Path, relative_path: &str, content: &str) {
    let full_path = root.join(relative_path);
    if let Some(parent) = full_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(full_path, content).unwrap();
}

fn collect_files(root: &Path) -> Vec<PathBuf> {
    fn visit(path: &Path, root: &Path, out: &mut Vec<PathBuf>) {
        let mut entries: Vec<_> = fs::read_dir(path)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        entries.sort();

        for entry in entries {
            // Skip the manifest directory
            if entry.is_dir() {
                if entry.file_name().map_or(false, |name| name == ".sindexer") {
                    continue;
                }
                visit(&entry, root, out);
            } else {
                out.push(entry);
            }
        }
    }

    let mut files = Vec::new();
    visit(root, root, &mut files);
    files
}

fn default_inputs() -> IndexInputs {
    IndexInputs {
        chunk_size: 512,
        overlap_lines: 3,
        min_chunk_lines: 5,
        target_chunk_lines: 50,
        extensions: vec!["rs".into()],
        ignore_patterns: vec!["target".into()],
    }
}

#[test]
fn test_manifest_round_trip_and_matches_inputs() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    let src = root.join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("lib.rs"), "fn main() {}\n").unwrap();

    let inputs = default_inputs();
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
fn test_diff_reports_added_modified_and_deleted_files() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    let src = root.join("src");
    fs::create_dir_all(&src).unwrap();

    create_file(root, "src/kept.rs", "pub fn kept() {}\n");
    create_file(root, "src/modified.rs", "pub fn version() -> u8 { 1 }\n");
    create_file(root, "src/deleted.rs", "pub fn delete_me() {}\n");

    let inputs = default_inputs();
    let store = ManifestStore;
    let files = collect_files(root);
    store
        .write_for_files(root, "collection", &inputs, &files)
        .unwrap();
    let previous = store.load(root).unwrap().unwrap();

    // Modify, delete, and add
    create_file(root, "src/modified.rs", "pub fn version() -> u8 { 2 }\n");
    fs::remove_file(root.join("src/deleted.rs")).unwrap();
    create_file(root, "src/added.rs", "pub fn added() {}\n");

    let new_files = collect_files(root);
    let diff =
        diff_manifest_against_files(&previous, root, "collection", &inputs, &new_files).unwrap();

    assert_eq!(diff.added, vec!["src/added.rs"]);
    assert_eq!(diff.deleted, vec!["src/deleted.rs"]);
    assert_eq!(diff.modified, vec!["src/modified.rs"]);
}

#[test]
fn test_config_change_invalidates_manifest() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    create_file(root, "src/lib.rs", "pub fn fingerprinted() {}\n");

    let inputs = default_inputs();
    let store = ManifestStore;
    let files = collect_files(root);
    store
        .write_for_files(root, "collection", &inputs, &files)
        .unwrap();
    let manifest = store.load(root).unwrap().unwrap();

    assert!(manifest.matches_index_inputs("collection", &inputs));

    let mut changed_inputs = inputs.clone();
    changed_inputs.chunk_size += 128;
    assert!(!manifest.matches_index_inputs("collection", &changed_inputs));
}
