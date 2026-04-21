use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::client::SearchHit;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalDoc {
    pub id: String,
    pub content: String,
    pub vector: Vec<f32>,
    pub metadata: serde_json::Value,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Collection {
    dimension: usize,
    docs: Vec<LocalDoc>,
}

pub struct LocalStore {
    collections: parking_lot::Mutex<HashMap<String, Collection>>,
}

impl LocalStore {
    pub fn new() -> Self {
        Self {
            collections: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    fn storage_dir() -> PathBuf {
        dirs_path().join("vector-indexes")
    }

    fn collection_path(name: &str) -> PathBuf {
        Self::storage_dir().join(format!("{name}.json"))
    }

    fn load_collection(&self, name: &str) -> Option<Collection> {
        let path = Self::collection_path(name);
        let data = std::fs::read(&path).ok()?;
        serde_json::from_slice(&data).ok()
    }

    fn persist_collection(&self, name: &str, collection: &Collection) -> Result<()> {
        let dir = Self::storage_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create vector index dir: {}", dir.display()))?;
        let path = Self::collection_path(name);
        let data = serde_json::to_vec(collection)?;
        std::fs::write(&path, data)?;
        Ok(())
    }

    pub fn create_collection(&self, name: &str, dimension: usize) -> Result<()> {
        let mut collections = self.collections.lock();
        let collection = Collection {
            dimension,
            docs: Vec::new(),
        };
        collections.insert(name.to_string(), collection.clone());
        self.persist_collection(name, &collection)?;
        Ok(())
    }

    pub fn has_collection(&self, name: &str) -> Result<bool> {
        let collections = self.collections.lock();
        if collections.contains_key(name) {
            return Ok(true);
        }
        Ok(Self::collection_path(name).exists())
    }

    pub fn drop_collection(&self, name: &str) -> Result<()> {
        self.collections.lock().remove(name);
        let path = Self::collection_path(name);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }

    pub fn insert_docs(
        &self,
        collection_name: &str,
        docs: Vec<LocalDoc>,
    ) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let mut collections = self.collections.lock();
        let collection = collections
            .entry(collection_name.to_string())
            .or_insert_with(|| {
                self.load_collection(collection_name).unwrap_or_default()
            });
        collection.docs.extend(docs);
        self.persist_collection(collection_name, collection)?;
        Ok(())
    }

    pub fn insert_rows(
        &self,
        collection_name: &str,
        ids: &[String],
        contents: &[String],
        vectors: &[Vec<f32>],
        metadatas: &[serde_json::Value],
    ) -> Result<()> {
        let docs: Vec<LocalDoc> = ids
            .iter()
            .zip(contents.iter())
            .zip(vectors.iter())
            .zip(metadatas.iter())
            .map(|(((id, content), vector), metadata)| LocalDoc {
                id: id.clone(),
                content: content.clone(),
                vector: vector.clone(),
                metadata: metadata.clone(),
            })
            .collect();
        self.insert_docs(collection_name, docs)
    }

    pub fn search(
        &self,
        collection_name: &str,
        query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchHit>> {
        let mut collections = self.collections.lock();
        let collection = match collections.get(collection_name) {
            Some(c) => c,
            None => {
                if let Some(loaded) = self.load_collection(collection_name) {
                    collections.insert(collection_name.to_string(), loaded);
                    collections.get(collection_name).unwrap()
                } else {
                    return Ok(Vec::new());
                }
            }
        };

        let mut scored: Vec<(f32, &LocalDoc)> = collection
            .docs
            .iter()
            .map(|doc| (cosine_similarity(query_vector, &doc.vector), doc))
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        Ok(scored
            .into_iter()
            .map(|(score, doc)| SearchHit {
                id: doc.id.clone(),
                score,
                content: doc.content.clone(),
                metadata: doc.metadata.clone(),
            })
            .collect())
    }

    pub fn delete_by_filter(
        &self,
        collection_name: &str,
        relative_paths: &[String],
    ) -> Result<()> {
        let mut collections = self.collections.lock();
        let collection = match collections.get_mut(collection_name) {
            Some(c) => c,
            None => return Ok(()),
        };

        collection.docs.retain(|doc| {
            let doc_path = doc
                .metadata
                .get("relative_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            !relative_paths.iter().any(|p| p == doc_path)
        });

        self.persist_collection(collection_name, collection)?;
        Ok(())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

fn dirs_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(dir).join("sindexer")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home)
            .join(".cache")
            .join("sindexer")
    } else {
        PathBuf::from("/tmp/sindexer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_local_store_crud() {
        let store = LocalStore::new();
        let name = format!("test_{}", uuid::Uuid::new_v4());

        assert!(!store.has_collection(&name).unwrap());
        store.create_collection(&name, 3).unwrap();
        assert!(store.has_collection(&name).unwrap());

        store
            .insert_rows(
                &name,
                &["id1".to_string()],
                &["hello world".to_string()],
                &[vec![1.0, 0.0, 0.0]],
                &[serde_json::json!({"relative_path": "src/lib.rs"})],
            )
            .unwrap();

        let results = store.search(&name, &[1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "hello world");

        store.drop_collection(&name).unwrap();
        assert!(!store.has_collection(&name).unwrap());
    }
}
