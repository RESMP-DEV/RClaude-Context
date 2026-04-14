use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::types::EmbeddingVector;

/// Configuration for the embedding client.
#[derive(Clone, Debug)]
pub struct EmbeddingConfig {
    /// Base URL for the embedding API.
    pub url: String,
    /// Model name to use for embeddings.
    pub model: String,
    /// Maximum number of texts per batch request.
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8100/v1/embeddings".to_string(),
            model: "jina-v3".to_string(),
            batch_size: 100,
        }
    }
}

impl EmbeddingConfig {
    /// Creates an EmbeddingConfig from the application Config.
    pub fn from_config(config: &Config) -> Self {
        Self {
            url: format!(
                "{}/v1/embeddings",
                config.embedding_url.trim_end_matches('/')
            ),
            model: config.embedding_model.clone(),
            batch_size: config.batch_size,
        }
    }
}

/// Request payload for the embeddings API.
#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    input: &'a [String],
    model: &'a str,
}

/// Single embedding in the API response.
#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Response from the embeddings API.
#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

/// Client for generating text embeddings via HTTP API.
pub struct EmbeddingClient {
    client: Client,
    config: EmbeddingConfig,
}

impl EmbeddingClient {
    /// Creates a new embedding client with the given configuration.
    pub fn new(config: EmbeddingConfig) -> Self {
        let client = Client::builder()
            .pool_max_idle_per_host(32)
            .build()
            .expect("failed to build HTTP client");

        Self { client, config }
    }

    /// Creates a new embedding client with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(EmbeddingConfig::default())
    }

    /// Generates an embedding for a single text.
    pub async fn embed(&self, text: &str) -> Result<EmbeddingVector> {
        let texts = vec![text.to_string()];
        let mut results = self.embed_batch(&texts).await?;

        results
            .pop()
            .context("embedding API returned empty response")
    }

    /// Generates embeddings for multiple texts in batches.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.config.batch_size) {
            let embeddings = self.embed_chunk(chunk).await?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    /// Embeds a single chunk of texts (up to batch_size).
    async fn embed_chunk(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        let request = EmbeddingRequest {
            input: texts,
            model: &self.config.model,
        };

        let response = self
            .client
            .post(&self.config.url)
            .json(&request)
            .send()
            .await
            .context("failed to send embedding request")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("embedding API returned {status}: {body}");
        }

        let response: EmbeddingResponse = response
            .json()
            .await
            .context("failed to parse embedding response")?;

        let embeddings = response
            .data
            .into_iter()
            .map(|d| {
                let dimension = d.embedding.len();
                EmbeddingVector {
                    vector: d.embedding,
                    dimension,
                }
            })
            .collect();

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.url, "http://localhost:8100/v1/embeddings");
        assert_eq!(config.model, "jina-v3");
        assert_eq!(config.batch_size, 100);
    }
}
