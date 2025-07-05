//! Filesystem-based model artifact storage

use std::path::{Path, PathBuf};
use std::io::Write;

use anyhow::{Context, Result};
use async_trait::async_trait;
use blake3::Hasher;
use tokio::fs;
use chrono::{DateTime, Utc};
use tracing::{info, warn, instrument};

use super::{ModelArtifactStorage, ArtifactMetadata};

/// Filesystem-based storage for model artifacts
pub struct FilesystemStorage {
    base_path: PathBuf,
}

impl FilesystemStorage {
    /// Create a new filesystem storage instance
    pub fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        
        // Create base directory if it doesn't exist
        std::fs::create_dir_all(&base_path)
            .with_context(|| format!("Failed to create storage directory: {:?}", base_path))?;
        
        Ok(Self { base_path })
    }

    /// Generate the storage path for a model artifact
    fn get_artifact_path(&self, model_id: &str, version_id: &str) -> PathBuf {
        self.base_path
            .join("models")
            .join(model_id)
            .join("versions")
            .join(format!("{}.bin", version_id))
    }

    /// Generate the model directory path
    fn get_model_directory(&self, model_id: &str) -> PathBuf {
        self.base_path.join("models").join(model_id)
    }

    /// Verify artifact integrity
    async fn verify_artifact(&self, path: &Path, expected_checksum: &str) -> Result<bool> {
        let data = fs::read(path).await?;
        let mut hasher = Hasher::new();
        hasher.update(&data);
        let actual_checksum = hasher.finalize().to_hex().to_string();
        Ok(actual_checksum == expected_checksum)
    }
}

#[async_trait]
impl ModelArtifactStorage for FilesystemStorage {
    #[instrument(skip(self, artifact_data))]
    async fn store_artifact(
        &self,
        model_id: &str,
        version_id: &str,
        artifact_data: &[u8],
        checksum: &str,
    ) -> Result<String> {
        let artifact_path = self.get_artifact_path(model_id, version_id);
        
        // Create directory structure
        if let Some(parent) = artifact_path.parent() {
            fs::create_dir_all(parent).await
                .with_context(|| format!("Failed to create directory: {:?}", parent))?;
        }

        // Write artifact data
        fs::write(&artifact_path, artifact_data).await
            .with_context(|| format!("Failed to write artifact: {:?}", artifact_path))?;

        // Verify the written data
        if !self.verify_artifact(&artifact_path, checksum).await? {
            fs::remove_file(&artifact_path).await.ok(); // Clean up on failure
            return Err(anyhow::anyhow!("Artifact checksum verification failed after storage"));
        }

        let path_str = artifact_path.to_string_lossy().to_string();
        
        info!(
            model_id = %model_id,
            version_id = %version_id,
            path = %path_str,
            size_bytes = artifact_data.len(),
            "Artifact stored successfully"
        );

        Ok(path_str)
    }

    #[instrument(skip(self))]
    async fn load_artifact(&self, artifact_path: &str) -> Result<Vec<u8>> {
        let path = Path::new(artifact_path);
        
        let data = fs::read(path).await
            .with_context(|| format!("Failed to read artifact: {}", artifact_path))?;

        info!(
            path = %artifact_path,
            size_bytes = data.len(),
            "Artifact loaded successfully"
        );

        Ok(data)
    }

    #[instrument(skip(self))]
    async fn delete_artifact(&self, artifact_path: &str) -> Result<()> {
        let path = Path::new(artifact_path);
        
        if path.exists() {
            fs::remove_file(path).await
                .with_context(|| format!("Failed to delete artifact: {}", artifact_path))?;
            
            info!(path = %artifact_path, "Artifact deleted successfully");
        } else {
            warn!(path = %artifact_path, "Artifact not found for deletion");
        }

        Ok(())
    }

    #[instrument(skip(self))]
    async fn artifact_exists(&self, artifact_path: &str) -> Result<bool> {
        let path = Path::new(artifact_path);
        Ok(path.exists())
    }

    #[instrument(skip(self))]
    async fn get_artifact_metadata(&self, artifact_path: &str) -> Result<ArtifactMetadata> {
        let path = Path::new(artifact_path);
        let metadata = fs::metadata(path).await
            .with_context(|| format!("Failed to get metadata for: {}", artifact_path))?;

        let data = fs::read(path).await?;
        let mut hasher = Hasher::new();
        hasher.update(&data);
        let checksum = hasher.finalize().to_hex().to_string();

        let created_at = metadata.created()
            .map(|t| DateTime::<Utc>::from(t))
            .unwrap_or_else(|_| Utc::now());

        let modified_at = metadata.modified()
            .map(|t| DateTime::<Utc>::from(t))
            .unwrap_or_else(|_| Utc::now());

        Ok(ArtifactMetadata {
            path: artifact_path.to_string(),
            size_bytes: metadata.len(),
            checksum,
            created_at,
            modified_at,
        })
    }

    #[instrument(skip(self))]
    async fn list_model_artifacts(&self, model_id: &str) -> Result<Vec<String>> {
        let model_dir = self.get_model_directory(model_id);
        let versions_dir = model_dir.join("versions");

        if !versions_dir.exists() {
            return Ok(vec![]);
        }

        let mut artifacts = Vec::new();
        let mut entries = fs::read_dir(&versions_dir).await
            .with_context(|| format!("Failed to read versions directory: {:?}", versions_dir))?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "bin") {
                artifacts.push(path.to_string_lossy().to_string());
            }
        }

        artifacts.sort();
        Ok(artifacts)
    }

    #[instrument(skip(self, active_paths))]
    async fn cleanup_orphaned_artifacts(&self, active_paths: Vec<String>) -> Result<u64> {
        let models_dir = self.base_path.join("models");
        
        if !models_dir.exists() {
            return Ok(0);
        }

        let active_paths_set: std::collections::HashSet<String> = active_paths.into_iter().collect();
        let mut cleaned_count = 0;

        fn visit_directory(
            dir: &Path,
            active_paths: &std::collections::HashSet<String>,
            cleaned_count: &mut u64,
        ) -> Result<()> {
            if !dir.exists() {
                return Ok(());
            }

            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_file() {
                    let path_str = path.to_string_lossy().to_string();
                    if !active_paths.contains(&path_str) {
                        if let Err(e) = std::fs::remove_file(&path) {
                            warn!(path = %path_str, error = %e, "Failed to cleanup orphaned artifact");
                        } else {
                            *cleaned_count += 1;
                            info!(path = %path_str, "Cleaned up orphaned artifact");
                        }
                    }
                } else if path.is_dir() {
                    visit_directory(&path, active_paths, cleaned_count)?;
                    
                    // Remove empty directories
                    if std::fs::read_dir(&path)?.next().is_none() {
                        if let Err(e) = std::fs::remove_dir(&path) {
                            warn!(path = ?path, error = %e, "Failed to remove empty directory");
                        }
                    }
                }
            }
            
            Ok(())
        }

        visit_directory(&models_dir, &active_paths_set, &mut cleaned_count)?;

        info!(cleaned_count, "Cleanup completed");
        Ok(cleaned_count)
    }
}