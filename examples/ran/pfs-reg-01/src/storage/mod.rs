//! Model artifact storage implementations

use anyhow::Result;
use async_trait::async_trait;

pub mod filesystem_storage;
pub mod compressed_storage;

pub use filesystem_storage::FilesystemStorage;
pub use compressed_storage::CompressedFilesystemStorage;

/// Trait for model artifact storage backends
#[async_trait]
pub trait ModelArtifactStorage: Send + Sync {
    /// Store a model artifact and return the storage path
    async fn store_artifact(
        &self,
        model_id: &str,
        version_id: &str,
        artifact_data: &[u8],
        checksum: &str,
    ) -> Result<String>;

    /// Load a model artifact from storage
    async fn load_artifact(&self, artifact_path: &str) -> Result<Vec<u8>>;

    /// Delete a model artifact from storage
    async fn delete_artifact(&self, artifact_path: &str) -> Result<()>;

    /// Check if an artifact exists in storage
    async fn artifact_exists(&self, artifact_path: &str) -> Result<bool>;

    /// Get artifact metadata (size, checksum, etc.)
    async fn get_artifact_metadata(&self, artifact_path: &str) -> Result<ArtifactMetadata>;

    /// List all artifacts for a given model
    async fn list_model_artifacts(&self, model_id: &str) -> Result<Vec<String>>;

    /// Cleanup orphaned artifacts
    async fn cleanup_orphaned_artifacts(&self, active_paths: Vec<String>) -> Result<u64>;
}

#[derive(Debug, Clone)]
pub struct ArtifactMetadata {
    pub path: String,
    pub size_bytes: u64,
    pub checksum: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub modified_at: chrono::DateTime<chrono::Utc>,
}