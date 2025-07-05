use crate::error::{ServiceError, ServiceResult};
use crate::neural_service::*;
use ruv_fann::{Network, NetworkBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub config: ModelConfig,
    pub metadata: ModelMetadata,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub file_path: PathBuf,
    pub input_size: usize,
    pub output_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub layers: Vec<u32>,
    pub activation: i32,
    pub learning_rate: f64,
    pub max_epochs: u32,
    pub desired_error: f64,
    pub training_algorithm: i32,
    pub metadata: ModelMetadata,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub file_path: PathBuf,
    pub input_size: usize,
    pub output_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub size_bytes: u64,
    pub total_parameters: u32,
    pub version: String,
    pub training_completed: bool,
    pub last_training_error: Option<f64>,
    pub training_epochs: Option<u32>,
}

pub struct ModelManager {
    models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    storage_path: PathBuf,
    max_models: usize,
}

impl ModelManager {
    pub fn new(storage_path: PathBuf, max_models: usize) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            storage_path,
            max_models,
        }
    }

    pub async fn initialize(&self) -> ServiceResult<()> {
        // Create storage directory if it doesn't exist
        if !self.storage_path.exists() {
            tokio::fs::create_dir_all(&self.storage_path).await?;
        }

        // Load existing models
        self.load_existing_models().await?;
        Ok(())
    }

    async fn load_existing_models(&self) -> ServiceResult<()> {
        let mut models = self.models.write().await;
        
        let mut entries = tokio::fs::read_dir(&self.storage_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            if let Some(extension) = entry.path().extension() {
                if extension == "json" {
                    if let Ok(model_info) = self.load_model_info(&entry.path()).await {
                        models.insert(model_info.id.clone(), model_info);
                    }
                }
            }
        }
        
        Ok(())
    }

    async fn load_model_info(&self, info_path: &Path) -> ServiceResult<ModelInfo> {
        let content = tokio::fs::read_to_string(info_path).await?;
        let serializable: SerializableModelInfo = serde_json::from_str(&content)?;
        
        let config = ModelConfig {
            name: serializable.name.clone(),
            description: serializable.description.clone(),
            layers: serializable.layers,
            activation: serializable.activation,
            learning_rate: serializable.learning_rate,
            max_epochs: serializable.max_epochs,
            desired_error: serializable.desired_error,
            training_algorithm: serializable.training_algorithm,
        };
        
        let model_info = ModelInfo {
            id: serializable.id,
            name: serializable.name,
            description: serializable.description,
            config,
            metadata: serializable.metadata,
            created_at: serializable.created_at,
            updated_at: serializable.updated_at,
            file_path: serializable.file_path,
            input_size: serializable.input_size,
            output_size: serializable.output_size,
        };
        
        Ok(model_info)
    }

    pub async fn create_model(&self, config: ModelConfig) -> ServiceResult<String> {
        let model_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        
        // Build the neural network
        let layers: Vec<usize> = config.layers.iter().map(|&x| x as usize).collect();
        let activation = crate::conversion::convert_activation_function(config.activation)?;
        
        let mut builder = NetworkBuilder::new();
        for &layer_size in &layers {
            builder = builder.layer(layer_size);
        }
        
        let network = builder
            .activation(activation)
            .learning_rate(config.learning_rate)
            .build()
            .map_err(|e| ServiceError::Configuration(format!("Failed to build network: {}", e)))?;

        // Calculate model size and parameters
        let total_parameters = self.calculate_network_parameters(&layers);
        let model_file_path = self.storage_path.join(format!("{}.bin", model_id));
        let info_file_path = self.storage_path.join(format!("{}.json", model_id));

        // Save the network
        self.save_network(&network, &model_file_path).await?;

        // Create model info
        let model_info = ModelInfo {
            id: model_id.clone(),
            name: config.name.clone(),
            description: config.description.clone(),
            config: config.clone(),
            metadata: ModelMetadata {
                size_bytes: self.get_file_size(&model_file_path).await?,
                total_parameters,
                version: "1.0.0".to_string(),
                training_completed: false,
                last_training_error: None,
                training_epochs: None,
            },
            created_at: now,
            updated_at: now,
            file_path: model_file_path,
            input_size: layers[0],
            output_size: layers[layers.len() - 1],
        };

        // Save model info
        self.save_model_info(&model_info, &info_file_path).await?;

        // Add to in-memory cache
        let mut models = self.models.write().await;
        
        // Check if we need to cleanup old models
        if models.len() >= self.max_models {
            self.cleanup_old_models(&mut models).await?;
        }
        
        models.insert(model_id.clone(), model_info);
        
        Ok(model_id)
    }

    async fn save_network(&self, network: &Network<f64>, path: &Path) -> ServiceResult<()> {
        let serialized = bincode::serialize(network)?;
        tokio::fs::write(path, serialized).await?;
        Ok(())
    }

    async fn save_model_info(&self, model_info: &ModelInfo, path: &Path) -> ServiceResult<()> {
        let serializable = SerializableModelInfo {
            id: model_info.id.clone(),
            name: model_info.name.clone(),
            description: model_info.description.clone(),
            layers: model_info.config.layers.clone(),
            activation: model_info.config.activation,
            learning_rate: model_info.config.learning_rate,
            max_epochs: model_info.config.max_epochs,
            desired_error: model_info.config.desired_error,
            training_algorithm: model_info.config.training_algorithm,
            metadata: model_info.metadata.clone(),
            created_at: model_info.created_at,
            updated_at: model_info.updated_at,
            file_path: model_info.file_path.clone(),
            input_size: model_info.input_size,
            output_size: model_info.output_size,
        };
        
        let serialized = serde_json::to_string_pretty(&serializable)?;
        tokio::fs::write(path, serialized).await?;
        Ok(())
    }

    pub async fn load_network(&self, model_id: &str) -> ServiceResult<Network<f64>> {
        let models = self.models.read().await;
        let model_info = models.get(model_id)
            .ok_or_else(|| ServiceError::ModelNotFound(model_id.to_string()))?;
        
        let serialized = tokio::fs::read(&model_info.file_path).await?;
        let network: Network<f64> = bincode::deserialize(&serialized)?;
        Ok(network)
    }

    pub async fn get_model_info(&self, model_id: &str) -> ServiceResult<ModelInfo> {
        let models = self.models.read().await;
        models.get(model_id)
            .cloned()
            .ok_or_else(|| ServiceError::ModelNotFound(model_id.to_string()))
    }

    pub async fn update_model_after_training(
        &self,
        model_id: &str,
        network: &Network<f64>,
        training_error: f64,
        epochs: u32,
    ) -> ServiceResult<()> {
        let mut models = self.models.write().await;
        let model_info = models.get_mut(model_id)
            .ok_or_else(|| ServiceError::ModelNotFound(model_id.to_string()))?;

        // Save updated network
        self.save_network(network, &model_info.file_path).await?;

        // Update metadata
        model_info.metadata.training_completed = true;
        model_info.metadata.last_training_error = Some(training_error);
        model_info.metadata.training_epochs = Some(epochs);
        model_info.metadata.size_bytes = self.get_file_size(&model_info.file_path).await?;
        model_info.updated_at = Utc::now();

        // Save updated model info
        let info_path = model_info.file_path.with_extension("json");
        self.save_model_info(model_info, &info_path).await?;

        Ok(())
    }

    pub async fn list_models(&self, page: u32, page_size: u32, filter: Option<String>) -> ServiceResult<Vec<ModelInfo>> {
        let models = self.models.read().await;
        let mut model_list: Vec<ModelInfo> = models.values().cloned().collect();

        // Apply filter if provided
        if let Some(filter_str) = filter {
            let filter_lower = filter_str.to_lowercase();
            model_list.retain(|model| {
                model.name.to_lowercase().contains(&filter_lower) ||
                model.description.to_lowercase().contains(&filter_lower)
            });
        }

        // Sort by creation date (newest first)
        model_list.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        // Apply pagination
        let start = (page * page_size) as usize;
        let end = start + page_size as usize;
        
        if start < model_list.len() {
            Ok(model_list[start..end.min(model_list.len())].to_vec())
        } else {
            Ok(Vec::new())
        }
    }

    pub async fn delete_model(&self, model_id: &str) -> ServiceResult<()> {
        let mut models = self.models.write().await;
        let model_info = models.remove(model_id)
            .ok_or_else(|| ServiceError::ModelNotFound(model_id.to_string()))?;

        // Delete files
        if model_info.file_path.exists() {
            tokio::fs::remove_file(&model_info.file_path).await?;
        }
        
        let info_path = model_info.file_path.with_extension("json");
        if info_path.exists() {
            tokio::fs::remove_file(&info_path).await?;
        }

        Ok(())
    }

    pub async fn get_model_count(&self) -> usize {
        let models = self.models.read().await;
        models.len()
    }

    fn calculate_network_parameters(&self, layers: &[usize]) -> u32 {
        let mut total = 0;
        for i in 1..layers.len() {
            // Weights: previous_layer_size * current_layer_size
            total += layers[i - 1] * layers[i];
            // Biases: current_layer_size
            total += layers[i];
        }
        total as u32
    }

    async fn get_file_size(&self, path: &Path) -> ServiceResult<u64> {
        let metadata = tokio::fs::metadata(path).await?;
        Ok(metadata.len())
    }

    async fn cleanup_old_models(&self, models: &mut HashMap<String, ModelInfo>) -> ServiceResult<()> {
        if models.len() < self.max_models {
            return Ok(());
        }

        // Get oldest model
        let oldest_model = models.values()
            .min_by(|a, b| a.created_at.cmp(&b.created_at))
            .map(|m| m.id.clone());

        if let Some(model_id) = oldest_model {
            let model_info = models.remove(&model_id).unwrap();
            
            // Delete files
            if model_info.file_path.exists() {
                tokio::fs::remove_file(&model_info.file_path).await?;
            }
            
            let info_path = model_info.file_path.with_extension("json");
            if info_path.exists() {
                tokio::fs::remove_file(&info_path).await?;
            }
        }

        Ok(())
    }
}