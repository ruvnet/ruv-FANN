//! Core Model Registry Service Implementation

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use blake3::Hasher;
use chrono::{DateTime, Utc};
use sea_orm::{
    ActiveModelTrait, ColumnTrait, Database, DatabaseConnection, EntityTrait, ModelTrait,
    PaginatorTrait, QueryFilter, QueryOrder, Set, TransactionTrait,
};
use serde_json::Value;
use tokio::fs;
use tracing::{error, info, warn, instrument};
use uuid::Uuid;

use crate::database::entities::{
    deployments, metrics, model_versions, models, 
    deployments::Entity as Deployments,
    metrics::Entity as Metrics,
    model_versions::Entity as ModelVersions,
    models::Entity as Models,
};
use crate::config::RegistryConfig;
use crate::storage::ModelArtifactStorage;
use crate::error::{RegistryError, RegistryResult};

/// Core Model Registry Service
#[derive(Clone)]
pub struct ModelRegistryService {
    db: Arc<DatabaseConnection>,
    storage: Arc<dyn ModelArtifactStorage>,
    config: Arc<RegistryConfig>,
}

impl ModelRegistryService {
    /// Create a new Model Registry Service
    pub async fn new(
        database_url: &str,
        storage: Arc<dyn ModelArtifactStorage>,
        config: RegistryConfig,
    ) -> Result<Self> {
        let db = Database::connect(database_url)
            .await
            .context("Failed to connect to database")?;
        
        Ok(Self {
            db: Arc::new(db),
            storage,
            config: Arc::new(config),
        })
    }

    /// Register a new model with the registry
    #[instrument(skip(self, model_artifact))]
    pub async fn register_model(
        &self,
        model_info: ModelRegistrationInfo,
        model_artifact: Vec<u8>,
        created_by: String,
    ) -> RegistryResult<ModelRegistrationResult> {
        let txn = self.db.begin().await.map_err(RegistryError::DatabaseError)?;
        
        // Generate model ID
        let model_id = Uuid::new_v4().to_string();
        let version_id = Uuid::new_v4().to_string();
        
        // Calculate artifact checksum
        let mut hasher = Hasher::new();
        hasher.update(&model_artifact);
        let checksum = hasher.finalize().to_hex().to_string();
        
        // Store artifact
        let artifact_path = self.storage
            .store_artifact(&model_id, &version_id, &model_artifact, &checksum)
            .await
            .map_err(RegistryError::StorageError)?;

        // Create model record
        let model = models::ActiveModel {
            id: Set(model_id.clone()),
            name: Set(model_info.name.clone()),
            description: Set(model_info.description.clone()),
            category: Set(model_info.category.clone()),
            model_type: Set(model_info.model_type.clone()),
            status: Set("registered".to_string()),
            tags: Set(model_info.tags.clone().map(|t| serde_json::to_value(t).unwrap_or(Value::Null))),
            capabilities: Set(model_info.capabilities.clone().map(|c| serde_json::to_value(c).unwrap_or(Value::Null))),
            configuration: Set(model_info.configuration.clone().map(|c| serde_json::to_value(c).unwrap_or(Value::Null))),
            metadata: Set(model_info.metadata.clone().map(|m| serde_json::to_value(m).unwrap_or(Value::Null))),
            created_by: Set(created_by.clone()),
            storage_path: Set(Some(artifact_path.clone())),
            checksum: Set(Some(checksum.clone())),
            size_bytes: Set(Some(model_artifact.len() as i64)),
            ..Default::default()
        };

        let model_result = model.insert(&txn).await.map_err(RegistryError::DatabaseError)?;

        // Create initial version record
        let version = model_versions::ActiveModel {
            id: Set(version_id.clone()),
            model_id: Set(model_id.clone()),
            version_number: Set(model_info.initial_version.unwrap_or_else(|| "1.0.0".to_string())),
            description: Set(Some("Initial version".to_string())),
            is_active: Set(true),
            status: Set("active".to_string()),
            artifact_path: Set(artifact_path),
            artifact_format: Set(model_info.artifact_format.unwrap_or_else(|| "ruvrann_binary".to_string())),
            artifact_checksum: Set(checksum),
            artifact_size_bytes: Set(model_artifact.len() as i64),
            performance_metrics: Set(model_info.performance_metrics.map(|p| serde_json::to_value(p).unwrap_or(Value::Null))),
            training_data_info: Set(model_info.training_data_info.map(|t| serde_json::to_value(t).unwrap_or(Value::Null))),
            created_by: Set(created_by),
            ..Default::default()
        };

        let version_result = version.insert(&txn).await.map_err(RegistryError::DatabaseError)?;

        txn.commit().await.map_err(RegistryError::DatabaseError)?;

        info!(
            model_id = %model_id,
            version_id = %version_id,
            name = %model_info.name,
            "Model registered successfully"
        );

        Ok(ModelRegistrationResult {
            model_id,
            version_id,
            artifact_path: version_result.artifact_path,
            checksum: version_result.artifact_checksum,
        })
    }

    /// Get model information by ID
    #[instrument(skip(self))]
    pub async fn get_model(
        &self,
        model_id: &str,
        version_id: Option<&str>,
        include_artifact: bool,
    ) -> RegistryResult<Option<ModelInfo>> {
        let model = Models::find_by_id(model_id)
            .one(self.db.as_ref())
            .await
            .map_err(RegistryError::DatabaseError)?;

        let Some(model) = model else {
            return Ok(None);
        };

        let version = if let Some(version_id) = version_id {
            ModelVersions::find_by_id(version_id)
                .one(self.db.as_ref())
                .await
                .map_err(RegistryError::DatabaseError)?
        } else {
            // Get active version
            ModelVersions::find()
                .filter(model_versions::Column::ModelId.eq(model_id))
                .filter(model_versions::Column::IsActive.eq(true))
                .one(self.db.as_ref())
                .await
                .map_err(RegistryError::DatabaseError)?
        };

        let Some(version) = version else {
            return Err(RegistryError::VersionNotFound(version_id.unwrap_or("active").to_string()));
        };

        let artifact_data = if include_artifact {
            Some(self.storage
                .load_artifact(&version.artifact_path)
                .await
                .map_err(RegistryError::StorageError)?)
        } else {
            None
        };

        Ok(Some(ModelInfo {
            model_id: model.id,
            name: model.name,
            description: model.description,
            category: model.category,
            model_type: model.model_type,
            status: model.status,
            tags: model.tags.and_then(|t| serde_json::from_value(t).ok()),
            capabilities: model.capabilities.and_then(|c| serde_json::from_value(c).ok()),
            configuration: model.configuration.and_then(|c| serde_json::from_value(c).ok()),
            metadata: model.metadata.and_then(|m| serde_json::from_value(m).ok()),
            created_at: model.created_at,
            updated_at: model.updated_at,
            created_by: model.created_by,
            version_info: Some(VersionInfo {
                version_id: version.id,
                version_number: version.version_number,
                description: version.description,
                is_active: version.is_active,
                status: version.status,
                artifact_path: version.artifact_path,
                artifact_format: version.artifact_format,
                artifact_checksum: version.artifact_checksum,
                artifact_size_bytes: version.artifact_size_bytes,
                performance_metrics: version.performance_metrics.and_then(|p| serde_json::from_value(p).ok()),
                training_data_info: version.training_data_info.and_then(|t| serde_json::from_value(t).ok()),
                created_at: version.created_at,
                created_by: version.created_by,
            }),
            artifact_data,
        }))
    }

    /// List models with filtering and pagination
    #[instrument(skip(self))]
    pub async fn list_models(
        &self,
        filters: ModelListFilters,
        page_size: u64,
        page: u64,
    ) -> RegistryResult<ModelListResult> {
        let mut query = Models::find();

        // Apply filters
        if let Some(category) = &filters.category {
            query = query.filter(models::Column::Category.eq(category));
        }
        if let Some(status) = &filters.status {
            query = query.filter(models::Column::Status.eq(status));
        }
        if let Some(model_type) = &filters.model_type {
            query = query.filter(models::Column::ModelType.eq(model_type));
        }

        // Apply pagination
        let paginator = query
            .order_by_desc(models::Column::CreatedAt)
            .paginate(self.db.as_ref(), page_size);

        let total_count = paginator.num_items().await.map_err(RegistryError::DatabaseError)?;
        let models = paginator
            .fetch_page(page)
            .await
            .map_err(RegistryError::DatabaseError)?;

        let model_infos = models
            .into_iter()
            .map(|model| ModelSummary {
                model_id: model.id,
                name: model.name,
                description: model.description,
                category: model.category,
                model_type: model.model_type,
                status: model.status,
                created_at: model.created_at,
                updated_at: model.updated_at,
                created_by: model.created_by,
            })
            .collect();

        Ok(ModelListResult {
            models: model_infos,
            total_count,
            page,
            page_size,
            has_next_page: (page + 1) * page_size < total_count,
        })
    }

    /// Search models by query and tags
    #[instrument(skip(self))]
    pub async fn search_models(
        &self,
        query: &str,
        tags: Option<Vec<String>>,
        filters: ModelListFilters,
        page_size: u64,
        page: u64,
    ) -> RegistryResult<ModelSearchResult> {
        let mut db_query = Models::find();

        // Apply text search (simplified - in production you'd use full-text search)
        if !query.is_empty() {
            db_query = db_query.filter(
                models::Column::Name.contains(query)
                    .or(models::Column::Description.contains(query))
            );
        }

        // Apply other filters
        if let Some(category) = &filters.category {
            db_query = db_query.filter(models::Column::Category.eq(category));
        }
        if let Some(status) = &filters.status {
            db_query = db_query.filter(models::Column::Status.eq(status));
        }
        if let Some(model_type) = &filters.model_type {
            db_query = db_query.filter(models::Column::ModelType.eq(model_type));
        }

        // Apply pagination
        let paginator = db_query
            .order_by_desc(models::Column::CreatedAt)
            .paginate(self.db.as_ref(), page_size);

        let total_count = paginator.num_items().await.map_err(RegistryError::DatabaseError)?;
        let models = paginator
            .fetch_page(page)
            .await
            .map_err(RegistryError::DatabaseError)?;

        let model_summaries = models
            .into_iter()
            .map(|model| ModelSummary {
                model_id: model.id,
                name: model.name,
                description: model.description,
                category: model.category,
                model_type: model.model_type,
                status: model.status,
                created_at: model.created_at,
                updated_at: model.updated_at,
                created_by: model.created_by,
            })
            .collect();

        Ok(ModelSearchResult {
            models: model_summaries,
            total_count,
            page,
            page_size,
            has_next_page: (page + 1) * page_size < total_count,
            suggestions: vec![], // TODO: Implement search suggestions
        })
    }

    /// Create a new version of an existing model
    #[instrument(skip(self, model_artifact))]
    pub async fn create_model_version(
        &self,
        model_id: &str,
        version_info: CreateVersionInfo,
        model_artifact: Vec<u8>,
        created_by: String,
        activate: bool,
    ) -> RegistryResult<String> {
        let txn = self.db.begin().await.map_err(RegistryError::DatabaseError)?;

        // Verify model exists
        let model = Models::find_by_id(model_id)
            .one(&txn)
            .await
            .map_err(RegistryError::DatabaseError)?
            .ok_or_else(|| RegistryError::ModelNotFound(model_id.to_string()))?;

        // Generate version ID
        let version_id = Uuid::new_v4().to_string();

        // Calculate artifact checksum
        let mut hasher = Hasher::new();
        hasher.update(&model_artifact);
        let checksum = hasher.finalize().to_hex().to_string();

        // Store artifact
        let artifact_path = self.storage
            .store_artifact(model_id, &version_id, &model_artifact, &checksum)
            .await
            .map_err(RegistryError::StorageError)?;

        // Deactivate other versions if this should be active
        if activate {
            let update_result = model_versions::Entity::update_many()
                .col_expr(model_versions::Column::IsActive, sea_orm::sea_query::Expr::value(false))
                .filter(model_versions::Column::ModelId.eq(model_id))
                .exec(&txn)
                .await
                .map_err(RegistryError::DatabaseError)?;

            info!(
                model_id = %model_id,
                deactivated_versions = update_result.rows_affected,
                "Deactivated existing model versions"
            );
        }

        // Create version record
        let version = model_versions::ActiveModel {
            id: Set(version_id.clone()),
            model_id: Set(model_id.to_string()),
            version_number: Set(version_info.version_number),
            description: Set(version_info.description),
            is_active: Set(activate),
            status: Set("active".to_string()),
            artifact_path: Set(artifact_path),
            artifact_format: Set(version_info.artifact_format.unwrap_or_else(|| "ruvrann_binary".to_string())),
            artifact_checksum: Set(checksum),
            artifact_size_bytes: Set(model_artifact.len() as i64),
            performance_metrics: Set(version_info.performance_metrics.map(|p| serde_json::to_value(p).unwrap_or(Value::Null))),
            training_data_info: Set(version_info.training_data_info.map(|t| serde_json::to_value(t).unwrap_or(Value::Null))),
            created_by: Set(created_by),
            ..Default::default()
        };

        version.insert(&txn).await.map_err(RegistryError::DatabaseError)?;

        txn.commit().await.map_err(RegistryError::DatabaseError)?;

        info!(
            model_id = %model_id,
            version_id = %version_id,
            version_number = %version_info.version_number,
            is_active = activate,
            "Model version created successfully"
        );

        Ok(version_id)
    }

    /// Get registry statistics
    #[instrument(skip(self))]
    pub async fn get_registry_stats(&self) -> RegistryResult<RegistryStats> {
        let total_models = Models::find()
            .count(self.db.as_ref())
            .await
            .map_err(RegistryError::DatabaseError)?;

        let total_versions = ModelVersions::find()
            .count(self.db.as_ref())
            .await
            .map_err(RegistryError::DatabaseError)?;

        let active_deployments = Deployments::find()
            .filter(deployments::Column::Status.eq("active"))
            .count(self.db.as_ref())
            .await
            .map_err(RegistryError::DatabaseError)?;

        // Get models by category (simplified aggregation)
        let models_by_category = Models::find()
            .all(self.db.as_ref())
            .await
            .map_err(RegistryError::DatabaseError)?
            .into_iter()
            .fold(HashMap::new(), |mut acc, model| {
                *acc.entry(model.category).or_insert(0) += 1;
                acc
            });

        // Get models by status
        let models_by_status = Models::find()
            .all(self.db.as_ref())
            .await
            .map_err(RegistryError::DatabaseError)?
            .into_iter()
            .fold(HashMap::new(), |mut acc, model| {
                *acc.entry(model.status).or_insert(0) += 1;
                acc
            });

        Ok(RegistryStats {
            total_models,
            total_versions,
            active_deployments,
            models_by_category,
            models_by_status,
            total_predictions_served: 0, // TODO: Implement prediction tracking
            average_model_latency_ms: 0.0, // TODO: Implement latency tracking
            last_updated: Utc::now(),
        })
    }
}

// Supporting structs and types

#[derive(Debug, Clone)]
pub struct ModelRegistrationInfo {
    pub name: String,
    pub description: Option<String>,
    pub category: String,
    pub model_type: String,
    pub tags: Option<Vec<String>>,
    pub capabilities: Option<serde_json::Value>,
    pub configuration: Option<serde_json::Value>,
    pub metadata: Option<serde_json::Value>,
    pub initial_version: Option<String>,
    pub artifact_format: Option<String>,
    pub performance_metrics: Option<serde_json::Value>,
    pub training_data_info: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct ModelRegistrationResult {
    pub model_id: String,
    pub version_id: String,
    pub artifact_path: String,
    pub checksum: String,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_id: String,
    pub name: String,
    pub description: Option<String>,
    pub category: String,
    pub model_type: String,
    pub status: String,
    pub tags: Option<Vec<String>>,
    pub capabilities: Option<serde_json::Value>,
    pub configuration: Option<serde_json::Value>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub created_by: String,
    pub version_info: Option<VersionInfo>,
    pub artifact_data: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct VersionInfo {
    pub version_id: String,
    pub version_number: String,
    pub description: Option<String>,
    pub is_active: bool,
    pub status: String,
    pub artifact_path: String,
    pub artifact_format: String,
    pub artifact_checksum: String,
    pub artifact_size_bytes: i64,
    pub performance_metrics: Option<serde_json::Value>,
    pub training_data_info: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
}

#[derive(Debug, Clone)]
pub struct ModelSummary {
    pub model_id: String,
    pub name: String,
    pub description: Option<String>,
    pub category: String,
    pub model_type: String,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub created_by: String,
}

#[derive(Debug, Clone, Default)]
pub struct ModelListFilters {
    pub category: Option<String>,
    pub status: Option<String>,
    pub model_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelListResult {
    pub models: Vec<ModelSummary>,
    pub total_count: u64,
    pub page: u64,
    pub page_size: u64,
    pub has_next_page: bool,
}

#[derive(Debug, Clone)]
pub struct ModelSearchResult {
    pub models: Vec<ModelSummary>,
    pub total_count: u64,
    pub page: u64,
    pub page_size: u64,
    pub has_next_page: bool,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CreateVersionInfo {
    pub version_number: String,
    pub description: Option<String>,
    pub artifact_format: Option<String>,
    pub performance_metrics: Option<serde_json::Value>,
    pub training_data_info: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total_models: u64,
    pub total_versions: u64,
    pub active_deployments: u64,
    pub models_by_category: HashMap<String, u32>,
    pub models_by_status: HashMap<String, u32>,
    pub total_predictions_served: u64,
    pub average_model_latency_ms: f64,
    pub last_updated: DateTime<Utc>,
}