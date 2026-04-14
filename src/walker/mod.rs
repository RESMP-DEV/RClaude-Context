//! File walker for discovering code files in a codebase.

use std::path::{Path, PathBuf};

use anyhow::Result;

/// Walks a codebase to discover indexable files.
///
/// Uses the `ignore` crate internally for parallel, gitignore-aware traversal.
pub struct CodeWalker {
    /// File extensions to include (e.g., ["rs", "py", "ts"]).
    pub extensions: Vec<String>,
    /// Additional patterns to ignore beyond .gitignore.
    pub ignore_patterns: Vec<String>,
}

impl CodeWalker {
    /// Create a new walker with default settings.
    pub fn new() -> Self {
        Self {
            extensions: vec![
                "rs".into(),
                "py".into(),
                "ts".into(),
                "tsx".into(),
                "js".into(),
                "jsx".into(),
                "go".into(),
                "c".into(),
                "cpp".into(),
                "h".into(),
                "hpp".into(),
                "java".into(),
                "kt".into(),
                "swift".into(),
                "rb".into(),
                "php".into(),
                "cs".into(),
                "scala".into(),
                "zig".into(),
            ],
            ignore_patterns: vec![
                "node_modules".into(),
                "target".into(),
                ".git".into(),
                "dist".into(),
                "build".into(),
                "__pycache__".into(),
                ".venv".into(),
                "venv".into(),
            ],
        }
    }

    /// Walk the codebase at the given path and return all indexable files.
    ///
    /// This runs in parallel using the ignore crate's parallel walker.
    pub async fn walk(&self, path: &Path) -> Result<Vec<PathBuf>> {
        use ignore::WalkBuilder;
        use std::sync::Mutex;

        let files = std::sync::Arc::new(Mutex::new(Vec::new()));
        let extensions = self.extensions.clone();

        let walker = WalkBuilder::new(path)
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .build_parallel();

        let files_ref = files.clone();
        walker.run(|| {
            let files = files_ref.clone();
            let extensions = extensions.clone();
            Box::new(move |entry| {
                if let Ok(entry) = entry {
                    if entry.file_type().map_or(false, |ft| ft.is_file()) {
                        if let Some(ext) = entry.path().extension() {
                            if extensions
                                .iter()
                                .any(|e| e == ext.to_string_lossy().as_ref())
                            {
                                files.lock().unwrap().push(entry.into_path());
                            }
                        }
                    }
                }
                ignore::WalkState::Continue
            })
        });

        let result = std::sync::Arc::try_unwrap(files)
            .map_err(|_| anyhow::anyhow!("Failed to unwrap Arc"))?
            .into_inner()
            .map_err(|e| anyhow::anyhow!("Failed to get mutex: {}", e))?;

        Ok(result)
    }
}

impl Default for CodeWalker {
    fn default() -> Self {
        Self::new()
    }
}
