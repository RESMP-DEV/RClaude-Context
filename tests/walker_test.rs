use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use ignore::WalkBuilder;
use tempfile::TempDir;

/// Create a test file with given content
fn create_file(dir: &TempDir, path: &str, content: &str) -> PathBuf {
    let full_path = dir.path().join(path);
    if let Some(parent) = full_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let mut file = File::create(&full_path).unwrap();
    file.write_all(content.as_bytes()).unwrap();
    full_path
}

/// Initialize a fake git repository (creates .git directory)
fn init_git_repo(dir: &TempDir) {
    fs::create_dir_all(dir.path().join(".git")).unwrap();
}

/// Collect all discovered file paths from a walker
fn collect_paths(builder: WalkBuilder) -> Vec<PathBuf> {
    builder
        .build()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .map(|entry| entry.into_path())
        .collect()
}

// =============================================================================
// File Discovery Tests
// =============================================================================

#[test]
fn test_discovers_rust_files() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "main.rs", "fn main() {}");
    create_file(&temp, "lib.rs", "pub mod foo;");
    create_file(&temp, "src/foo.rs", "pub fn bar() {}");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    let rust_files: Vec<_> = paths
        .iter()
        .filter(|p| p.extension().map(|e| e == "rs").unwrap_or(false))
        .collect();

    assert_eq!(rust_files.len(), 3);
}

#[test]
fn test_discovers_python_files() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "main.py", "print('hello')");
    create_file(&temp, "lib/utils.py", "def foo(): pass");
    create_file(&temp, "tests/test_main.py", "def test_foo(): pass");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    let py_files: Vec<_> = paths
        .iter()
        .filter(|p| p.extension().map(|e| e == "py").unwrap_or(false))
        .collect();

    assert_eq!(py_files.len(), 3);
}

#[test]
fn test_discovers_javascript_files() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "index.js", "console.log('hi');");
    create_file(&temp, "src/app.js", "export default {};");
    create_file(&temp, "src/utils.mjs", "export const foo = 1;");
    create_file(&temp, "config.cjs", "module.exports = {};");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    let js_files: Vec<_> = paths
        .iter()
        .filter(|p| {
            p.extension()
                .map(|e| e == "js" || e == "mjs" || e == "cjs")
                .unwrap_or(false)
        })
        .collect();

    assert_eq!(js_files.len(), 4);
}

#[test]
fn test_discovers_typescript_files() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "index.ts", "const x: number = 1;");
    create_file(&temp, "component.tsx", "export const App = () => <div/>;");
    create_file(&temp, "types.d.ts", "declare module 'foo';");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    let ts_files: Vec<_> = paths
        .iter()
        .filter(|p| {
            p.extension()
                .map(|e| e == "ts" || e == "tsx")
                .unwrap_or(false)
        })
        .collect();

    assert_eq!(ts_files.len(), 3);
}

#[test]
fn test_discovers_mixed_extensions() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "main.rs", "fn main() {}");
    create_file(&temp, "script.py", "print('hi')");
    create_file(&temp, "app.js", "console.log('hi');");
    create_file(&temp, "lib.go", "package main");
    create_file(&temp, "Main.java", "class Main {}");
    create_file(&temp, "helper.cpp", "int main() {}");
    create_file(&temp, "readme.md", "# Readme");
    create_file(&temp, "config.json", "{}");
    create_file(&temp, "style.css", "body {}");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert_eq!(paths.len(), 9);
}

#[test]
fn test_discovers_files_in_nested_directories() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "src/main.rs", "");
    create_file(&temp, "src/lib/utils.rs", "");
    create_file(&temp, "src/lib/helpers/string.rs", "");
    create_file(&temp, "src/lib/helpers/math/ops.rs", "");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert_eq!(paths.len(), 4);
}

// =============================================================================
// .gitignore Respect Tests
// =============================================================================

#[test]
fn test_respects_gitignore_file_patterns() {
    let temp = TempDir::new().unwrap();
    init_git_repo(&temp);

    create_file(&temp, ".gitignore", "*.log\n*.tmp\n");
    create_file(&temp, "main.rs", "fn main() {}");
    create_file(&temp, "debug.log", "log content");
    create_file(&temp, "cache.tmp", "temp data");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(!paths.iter().any(|p| p.ends_with("debug.log")));
    assert!(!paths.iter().any(|p| p.ends_with("cache.tmp")));
}

#[test]
fn test_respects_gitignore_directory_patterns() {
    let temp = TempDir::new().unwrap();
    init_git_repo(&temp);

    create_file(
        &temp,
        ".gitignore",
        "target/\nnode_modules/\n__pycache__/\n",
    );
    create_file(&temp, "main.rs", "fn main() {}");
    create_file(&temp, "target/debug/main", "binary");
    create_file(&temp, "node_modules/lodash/index.js", "");
    create_file(&temp, "__pycache__/main.cpython-39.pyc", "");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(!paths.iter().any(|p| p.to_string_lossy().contains("target")));
    assert!(!paths
        .iter()
        .any(|p| p.to_string_lossy().contains("node_modules")));
    assert!(!paths
        .iter()
        .any(|p| p.to_string_lossy().contains("__pycache__")));
}

#[test]
fn test_respects_gitignore_negation() {
    let temp = TempDir::new().unwrap();
    init_git_repo(&temp);

    create_file(&temp, ".gitignore", "*.log\n!important.log\n");
    create_file(&temp, "debug.log", "ignore me");
    create_file(&temp, "important.log", "keep me");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert!(!paths.iter().any(|p| p.ends_with("debug.log")));
    assert!(paths.iter().any(|p| p.ends_with("important.log")));
}

#[test]
fn test_respects_nested_gitignore() {
    let temp = TempDir::new().unwrap();
    init_git_repo(&temp);

    create_file(&temp, ".gitignore", "*.log\n");
    create_file(&temp, "src/.gitignore", "*.bak\n");
    create_file(&temp, "main.rs", "");
    create_file(&temp, "debug.log", "");
    create_file(&temp, "src/lib.rs", "");
    create_file(&temp, "src/old.bak", "");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(paths.iter().any(|p| p.ends_with("lib.rs")));
    assert!(!paths.iter().any(|p| p.ends_with("debug.log")));
    assert!(!paths.iter().any(|p| p.ends_with("old.bak")));
}

#[test]
fn test_respects_gitignore_wildcards() {
    let temp = TempDir::new().unwrap();
    init_git_repo(&temp);

    create_file(&temp, ".gitignore", "*.min.js\n*.min.css\nbuild/**\n");
    create_file(&temp, "app.js", "");
    create_file(&temp, "app.min.js", "");
    create_file(&temp, "style.css", "");
    create_file(&temp, "style.min.css", "");
    create_file(&temp, "build/output/bundle.js", "");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("app.js")));
    assert!(paths.iter().any(|p| p.ends_with("style.css")));
    assert!(!paths.iter().any(|p| p.ends_with("app.min.js")));
    assert!(!paths.iter().any(|p| p.ends_with("style.min.css")));
    assert!(!paths.iter().any(|p| p.to_string_lossy().contains("build")));
}

#[test]
fn test_can_disable_gitignore() {
    let temp = TempDir::new().unwrap();
    init_git_repo(&temp);

    create_file(&temp, ".gitignore", "*.log\n");
    create_file(&temp, "main.rs", "");
    create_file(&temp, "debug.log", "");

    let mut walker = WalkBuilder::new(temp.path());
    walker.git_ignore(false);
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(paths.iter().any(|p| p.ends_with("debug.log")));
}

#[test]
fn test_gitignore_without_git_repo_requires_flag() {
    let temp = TempDir::new().unwrap();
    // Note: NOT initializing .git directory

    create_file(&temp, ".gitignore", "*.log\n");
    create_file(&temp, "main.rs", "");
    create_file(&temp, "debug.log", "");

    // Without require_git(false), gitignore is not respected outside git repos
    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    // Without .git, gitignore is not respected by default
    assert!(paths.iter().any(|p| p.ends_with("debug.log")));

    // With require_git(false), gitignore IS respected even without .git
    let mut walker = WalkBuilder::new(temp.path());
    walker.require_git(false);
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(!paths.iter().any(|p| p.ends_with("debug.log")));
}

// =============================================================================
// Custom Ignore Pattern Tests
// =============================================================================

#[test]
fn test_custom_ignore_patterns_via_types() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "main.rs", "");
    create_file(&temp, "lib.rs", "");
    create_file(&temp, "test.py", "");
    create_file(&temp, "script.js", "");

    let mut types = ignore::types::TypesBuilder::new();
    types.add_defaults();
    types.select("rust");
    let types = types.build().unwrap();

    let mut walker = WalkBuilder::new(temp.path());
    walker.types(types);
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(paths.iter().any(|p| p.ends_with("lib.rs")));
    assert!(!paths.iter().any(|p| p.ends_with("test.py")));
    assert!(!paths.iter().any(|p| p.ends_with("script.js")));
}

#[test]
fn test_ignore_hidden_files() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "main.rs", "");
    create_file(&temp, ".hidden.rs", "");
    create_file(&temp, ".config/settings.json", "");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    // By default, hidden files are ignored
    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(!paths.iter().any(|p| p.ends_with(".hidden.rs")));
    assert!(!paths
        .iter()
        .any(|p| p.to_string_lossy().contains(".config")));
}

#[test]
fn test_include_hidden_files() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "main.rs", "");
    create_file(&temp, ".hidden.rs", "");

    let mut walker = WalkBuilder::new(temp.path());
    walker.hidden(false);
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(paths.iter().any(|p| p.ends_with(".hidden.rs")));
}

#[test]
fn test_custom_ignore_file() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, ".customignore", "*.generated.rs\nvendor/\n");
    create_file(&temp, "main.rs", "");
    create_file(&temp, "types.generated.rs", "");
    create_file(&temp, "vendor/dep.rs", "");

    let mut walker = WalkBuilder::new(temp.path());
    walker.add_custom_ignore_filename(".customignore");
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(!paths.iter().any(|p| p.ends_with("types.generated.rs")));
    assert!(!paths.iter().any(|p| p.to_string_lossy().contains("vendor")));
}

#[test]
fn test_max_depth_limit() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "level0.rs", "");
    create_file(&temp, "a/level1.rs", "");
    create_file(&temp, "a/b/level2.rs", "");
    create_file(&temp, "a/b/c/level3.rs", "");

    let mut walker = WalkBuilder::new(temp.path());
    walker.max_depth(Some(2));
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("level0.rs")));
    assert!(paths.iter().any(|p| p.ends_with("level1.rs")));
    assert!(!paths.iter().any(|p| p.ends_with("level2.rs")));
    assert!(!paths.iter().any(|p| p.ends_with("level3.rs")));
}

#[test]
fn test_follow_symlinks() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "src/main.rs", "");
    create_file(&temp, "external/lib.rs", "");

    // Create symlink
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let link_path = temp.path().join("src/external_link");
        let target = temp.path().join("external");
        let _ = symlink(&target, &link_path);
    }

    let mut walker = WalkBuilder::new(temp.path());
    walker.follow_links(true);
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(paths.iter().any(|p| p.ends_with("lib.rs")));
}

#[test]
fn test_parallel_walk() {
    let temp = TempDir::new().unwrap();

    // Create many files
    for i in 0..100 {
        create_file(&temp, &format!("file_{}.rs", i), "");
    }

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert_eq!(paths.len(), 100);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_empty_directory() {
    let temp = TempDir::new().unwrap();

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert!(paths.is_empty());
}

#[test]
fn test_files_with_no_extension() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "Makefile", "all:");
    create_file(&temp, "Dockerfile", "FROM alpine");
    create_file(&temp, "LICENSE", "MIT");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert_eq!(paths.len(), 3);
}

#[test]
fn test_unicode_filenames() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "日本語.rs", "");
    create_file(&temp, "émoji_🦀.rs", "");
    create_file(&temp, "中文/模块.rs", "");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert_eq!(paths.len(), 3);
}

#[test]
fn test_files_with_spaces_in_name() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "my file.rs", "");
    create_file(&temp, "path with spaces/another file.rs", "");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert_eq!(paths.len(), 2);
}

#[test]
fn test_very_long_path() {
    let temp = TempDir::new().unwrap();

    // Create a deeply nested path
    let mut path = String::from("a");
    for _ in 0..20 {
        path.push_str("/abcdefghij");
    }
    path.push_str("/file.rs");

    create_file(&temp, &path, "");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert_eq!(paths.len(), 1);
}

#[test]
fn test_gitignore_comments_and_empty_lines() {
    let temp = TempDir::new().unwrap();
    init_git_repo(&temp);

    create_file(
        &temp,
        ".gitignore",
        "# This is a comment\n\n*.log\n\n# Another comment\n*.tmp\n",
    );
    create_file(&temp, "main.rs", "");
    create_file(&temp, "debug.log", "");
    create_file(&temp, "cache.tmp", "");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(!paths.iter().any(|p| p.ends_with("debug.log")));
    assert!(!paths.iter().any(|p| p.ends_with("cache.tmp")));
}

#[test]
fn test_global_gitignore_patterns() {
    let temp = TempDir::new().unwrap();
    init_git_repo(&temp);

    // Test patterns commonly found in global gitignore
    create_file(&temp, ".gitignore", ".DS_Store\nThumbs.db\n*.swp\n*~\n");
    create_file(&temp, "main.rs", "");
    create_file(&temp, ".DS_Store", "");
    create_file(&temp, "Thumbs.db", "");
    create_file(&temp, "file.swp", "");
    create_file(&temp, "backup~", "");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(!paths.iter().any(|p| p.ends_with(".DS_Store")));
    assert!(!paths.iter().any(|p| p.ends_with("Thumbs.db")));
    assert!(!paths.iter().any(|p| p.ends_with("file.swp")));
    assert!(!paths.iter().any(|p| p.ends_with("backup~")));
}

#[test]
fn test_gitignore_slash_patterns() {
    let temp = TempDir::new().unwrap();
    init_git_repo(&temp);

    // Leading slash anchors to root
    create_file(&temp, ".gitignore", "/root_only.txt\neverywhere.txt\n");
    create_file(&temp, "root_only.txt", "");
    create_file(&temp, "sub/root_only.txt", "");
    create_file(&temp, "everywhere.txt", "");
    create_file(&temp, "sub/everywhere.txt", "");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    // /root_only.txt only ignores at root
    assert!(!paths
        .iter()
        .any(|p| p.ends_with("root_only.txt") && !p.to_string_lossy().contains("sub")));
    assert!(paths
        .iter()
        .any(|p| p.to_string_lossy().contains("sub/root_only.txt")));

    // everywhere.txt ignores everywhere
    assert!(!paths.iter().any(|p| p.ends_with("everywhere.txt")));
}

#[test]
fn test_gitignore_double_star() {
    let temp = TempDir::new().unwrap();
    init_git_repo(&temp);

    create_file(&temp, ".gitignore", "**/logs/**\n");
    create_file(&temp, "main.rs", "");
    create_file(&temp, "logs/app.log", "");
    create_file(&temp, "src/logs/debug.log", "");
    create_file(&temp, "a/b/logs/c/d.log", "");

    let walker = WalkBuilder::new(temp.path());
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(!paths.iter().any(|p| p.to_string_lossy().contains("logs")));
}

#[test]
fn test_add_ignore_overrides() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "main.rs", "");
    create_file(&temp, "test.rs", "");
    create_file(&temp, "lib.rs", "");

    let mut builder = ignore::overrides::OverrideBuilder::new(temp.path());
    builder.add("!test.rs").unwrap();
    let overrides = builder.build().unwrap();

    let mut walker = WalkBuilder::new(temp.path());
    walker.overrides(overrides);
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("main.rs")));
    assert!(paths.iter().any(|p| p.ends_with("lib.rs")));
    assert!(!paths.iter().any(|p| p.ends_with("test.rs")));
}

#[test]
fn test_filter_entry_callback() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "keep.rs", "");
    create_file(&temp, "skip_this.rs", "");
    create_file(&temp, "another.rs", "");

    let mut walker = WalkBuilder::new(temp.path());
    walker.filter_entry(|entry| !entry.file_name().to_string_lossy().starts_with("skip"));
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("keep.rs")));
    assert!(paths.iter().any(|p| p.ends_with("another.rs")));
    assert!(!paths.iter().any(|p| p.ends_with("skip_this.rs")));
}

#[test]
fn test_multiple_root_paths() {
    let temp1 = TempDir::new().unwrap();
    let temp2 = TempDir::new().unwrap();

    create_file(&temp1, "file1.rs", "");
    create_file(&temp2, "file2.rs", "");

    let mut walker = WalkBuilder::new(temp1.path());
    walker.add(temp2.path());
    let paths = collect_paths(walker);

    assert!(paths.iter().any(|p| p.ends_with("file1.rs")));
    assert!(paths.iter().any(|p| p.ends_with("file2.rs")));
}

#[test]
fn test_sort_by_file_name() {
    let temp = TempDir::new().unwrap();

    create_file(&temp, "zebra.rs", "");
    create_file(&temp, "alpha.rs", "");
    create_file(&temp, "middle.rs", "");

    let mut walker = WalkBuilder::new(temp.path());
    walker.sort_by_file_name(|a, b| a.cmp(b));
    let paths = collect_paths(walker);

    let names: Vec<_> = paths
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();

    // Check that files are sorted
    let mut sorted_names = names.clone();
    sorted_names.sort();
    assert_eq!(names, sorted_names);
}
