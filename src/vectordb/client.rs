use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::types::CodeChunk;

/// A document to be inserted into Milvus.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier for the document.
    pub id: String,
    /// Text content of the document.
    pub content: String,
    /// Embedding vector.
    pub vector: Vec<f32>,
    /// Additional metadata as JSON.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A search result from Milvus.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchHit {
    /// Document ID.
    pub id: String,
    /// Similarity score (distance).
    pub score: f32,
    /// Document content.
    pub content: String,
    /// Metadata.
    pub metadata: serde_json::Value,
}

/// HTTP client for Milvus vector database.
#[derive(Clone)]
pub struct MilvusClient {
    client: Client,
    base_url: String,
}

#[derive(Serialize)]
struct CreateCollectionRequest {
    #[serde(rename = "dbName")]
    db_name: String,
    #[serde(rename = "collectionName")]
    collection_name: String,
    dimension: usize,
    #[serde(rename = "metricType")]
    metric_type: String,
    #[serde(rename = "primaryFieldName")]
    primary_field_name: String,
    #[serde(rename = "vectorFieldName")]
    vector_field_name: String,
}

#[derive(Serialize)]
struct HasCollectionRequest {
    #[serde(rename = "dbName")]
    db_name: String,
    #[serde(rename = "collectionName")]
    collection_name: String,
}

#[derive(Deserialize)]
struct HasCollectionResponse {
    code: i32,
    data: Option<HasCollectionData>,
    message: Option<String>,
}

#[derive(Deserialize)]
struct HasCollectionData {
    has: bool,
}

#[derive(Serialize)]
struct InsertRequest {
    #[serde(rename = "dbName")]
    db_name: String,
    #[serde(rename = "collectionName")]
    collection_name: String,
    data: Vec<InsertRow>,
}

/// A row to be inserted into Milvus.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InsertRow {
    /// Unique identifier.
    pub id: String,
    /// Text content.
    pub content: String,
    /// Embedding vector.
    pub vector: Vec<f32>,
    /// Additional metadata as JSON.
    pub metadata: serde_json::Value,
}

#[derive(Serialize)]
struct SearchRequest {
    #[serde(rename = "dbName")]
    db_name: String,
    #[serde(rename = "collectionName")]
    collection_name: String,
    data: Vec<Vec<f32>>,
    #[serde(rename = "annsField")]
    anns_field: String,
    limit: usize,
    #[serde(rename = "outputFields")]
    output_fields: Vec<String>,
}

#[derive(Deserialize)]
struct SearchResponse {
    code: i32,
    data: Option<Vec<Vec<SearchResultItem>>>,
    message: Option<String>,
}

#[derive(Deserialize)]
struct SearchResultItem {
    id: String,
    distance: f32,
    content: Option<String>,
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct DeleteRequest {
    #[serde(rename = "dbName")]
    db_name: String,
    #[serde(rename = "collectionName")]
    collection_name: String,
    filter: String,
}

#[derive(Deserialize)]
struct MilvusResponse {
    code: i32,
    message: Option<String>,
}

#[derive(Serialize)]
struct DropCollectionRequest {
    #[serde(rename = "dbName")]
    db_name: String,
    #[serde(rename = "collectionName")]
    collection_name: String,
}

impl MilvusClient {
    /// Create a new Milvus client.
    ///
    /// # Arguments
    /// * `base_url` - Base URL for Milvus REST API (e.g., "http://localhost:19530")
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
        }
    }

    /// Create a collection with the specified name and vector dimension.
    ///
    /// Uses cosine similarity as the metric type. The collection will have:
    /// - `id` as the primary field (VARCHAR)
    /// - `content` for document text (VARCHAR)
    /// - `vector` for embeddings (FLOAT_VECTOR)
    /// - `metadata` for additional data (JSON)
    pub async fn create_collection(&self, name: &str, dimension: usize) -> Result<()> {
        let url = format!("{}/v2/vectordb/collections/create", self.base_url);

        let request = CreateCollectionRequest {
            db_name: "default".to_string(),
            collection_name: name.to_string(),
            dimension,
            metric_type: "COSINE".to_string(),
            primary_field_name: "id".to_string(),
            vector_field_name: "vector".to_string(),
        };

        let response: MilvusResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send create collection request")?
            .json()
            .await
            .context("failed to parse create collection response")?;

        if response.code != 0 {
            anyhow::bail!(
                "create collection failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(())
    }

    /// Check if a collection exists.
    pub async fn has_collection(&self, name: &str) -> Result<bool> {
        let url = format!("{}/v2/vectordb/collections/has", self.base_url);

        let request = HasCollectionRequest {
            db_name: "default".to_string(),
            collection_name: name.to_string(),
        };

        let response: HasCollectionResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send has collection request")?
            .json()
            .await
            .context("failed to parse has collection response")?;

        if response.code != 0 {
            anyhow::bail!(
                "has collection failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(response.data.map(|d| d.has).unwrap_or(false))
    }

    /// Insert documents into a collection.
    pub async fn insert(&self, collection: &str, docs: Vec<Document>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }

        let url = format!("{}/v2/vectordb/entities/insert", self.base_url);

        let data: Vec<InsertRow> = docs
            .into_iter()
            .map(|doc| InsertRow {
                id: doc.id,
                content: doc.content,
                vector: doc.vector,
                metadata: doc.metadata,
            })
            .collect();

        let request = InsertRequest {
            db_name: "default".to_string(),
            collection_name: collection.to_string(),
            data,
        };

        let response: MilvusResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send insert request")?
            .json()
            .await
            .context("failed to parse insert response")?;

        if response.code != 0 {
            anyhow::bail!(
                "insert failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(())
    }

    /// Search for similar vectors in a collection.
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `vector` - Query vector
    /// * `top_k` - Number of results to return
    pub async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchHit>> {
        let url = format!("{}/v2/vectordb/entities/search", self.base_url);

        let request = SearchRequest {
            db_name: "default".to_string(),
            collection_name: collection.to_string(),
            data: vec![vector.to_vec()],
            anns_field: "vector".to_string(),
            limit: top_k,
            output_fields: vec![
                "id".to_string(),
                "content".to_string(),
                "metadata".to_string(),
            ],
        };

        let response: SearchResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send search request")?
            .json()
            .await
            .context("failed to parse search response")?;

        if response.code != 0 {
            anyhow::bail!(
                "search failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        let hits = response
            .data
            .and_then(|results| results.into_iter().next())
            .unwrap_or_default()
            .into_iter()
            .map(|item| SearchHit {
                id: item.id,
                score: item.distance,
                content: item.content.unwrap_or_default(),
                metadata: item.metadata.unwrap_or(serde_json::Value::Null),
            })
            .collect();

        Ok(hits)
    }

    /// Delete entities from a collection using a filter expression.
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `filter` - Filter expression (e.g., `id in ["id1", "id2"]`)
    pub async fn delete(&self, collection: &str, filter: &str) -> Result<()> {
        let url = format!("{}/v2/vectordb/entities/delete", self.base_url);

        let request = DeleteRequest {
            db_name: "default".to_string(),
            collection_name: collection.to_string(),
            filter: filter.to_string(),
        };

        let response: MilvusResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send delete request")?
            .json()
            .await
            .context("failed to parse delete response")?;

        if response.code != 0 {
            anyhow::bail!(
                "delete failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(())
    }

    /// Batch insert rows into a collection.
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `data` - Rows to insert
    pub async fn insert_batch(&self, collection: &str, data: &[InsertRow]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let url = format!("{}/v2/vectordb/entities/insert", self.base_url);

        let request = InsertRequest {
            db_name: "default".to_string(),
            collection_name: collection.to_string(),
            data: data.to_vec(),
        };

        let response: MilvusResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send insert batch request")?
            .json()
            .await
            .context("failed to parse insert batch response")?;

        if response.code != 0 {
            anyhow::bail!(
                "insert batch failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(())
    }

    /// Insert code chunks with their embeddings into a collection.
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `chunks` - Code chunks to insert
    /// * `embeddings` - Corresponding embedding vectors (must match chunks length)
    pub async fn insert_with_embeddings(
        &self,
        collection: &str,
        chunks: Vec<CodeChunk>,
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        if chunks.len() != embeddings.len() {
            anyhow::bail!(
                "chunks and embeddings length mismatch: {} vs {}",
                chunks.len(),
                embeddings.len()
            );
        }

        let rows: Vec<InsertRow> = chunks
            .into_iter()
            .zip(embeddings.iter())
            .map(|(chunk, embedding)| InsertRow {
                id: chunk.id,
                content: chunk.content,
                vector: embedding.clone(),
                metadata: serde_json::json!({
                    "file_path": chunk.file_path,
                    "relative_path": chunk.relative_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                }),
            })
            .collect();

        self.insert_batch(collection, &rows).await
    }

    /// Drop a collection.
    ///
    /// # Arguments
    /// * `name` - Collection name to drop
    pub async fn drop_collection(&self, name: &str) -> Result<()> {
        let url = format!("{}/v2/vectordb/collections/drop", self.base_url);

        let request = DropCollectionRequest {
            db_name: "default".to_string(),
            collection_name: name.to_string(),
        };

        let response: MilvusResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send drop collection request")?
            .json()
            .await
            .context("failed to parse drop collection response")?;

        if response.code != 0 {
            anyhow::bail!(
                "drop collection failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(())
    }
}
