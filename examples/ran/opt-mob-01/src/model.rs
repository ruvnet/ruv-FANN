//! Neural network model for handover prediction
//!
//! This module implements the ruv-FANN-based neural network for handover prediction,
//! including model configuration, training, and inference capabilities.

use crate::features::{FeatureVector, NormalizationParams};
use crate::{OptMobError, Result};
use ruv_fann::{Network, NetworkBuilder, TrainingAlgorithm as FannTrainingAlgorithm, TrainingData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Configuration for the handover prediction model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Neural network architecture
    pub input_size: usize,
    pub hidden_layers: Vec<usize>,
    pub output_size: usize,
    
    /// Training parameters
    pub learning_rate: f64,
    pub momentum: f64,
    pub training_algorithm: TrainingAlgorithm,
    pub max_epochs: u32,
    pub desired_error: f64,
    pub epochs_between_reports: u32,
    
    /// Regularization
    pub dropout_rate: Option<f64>,
    pub weight_decay: Option<f64>,
    
    /// Data handling
    pub validation_split: f64,
    pub batch_size: Option<usize>,
    pub shuffle_data: bool,
    
    /// Feature engineering
    pub feature_normalization: bool,
    pub feature_selection: bool,
    pub feature_importance_threshold: Option<f64>,
    
    /// Model metadata
    pub model_version: String,
    pub description: String,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model_config: ModelConfig,
    pub early_stopping: bool,
    pub early_stopping_patience: u32,
    pub early_stopping_min_delta: f64,
    pub save_best_model: bool,
    pub checkpoint_frequency: u32,
    pub enable_logging: bool,
    pub cross_validation_folds: Option<u32>,
}

/// Supported training algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingAlgorithm {
    BackPropagation,
    ResilientPropagation,
    QuickPropagation,
    Batch,
}

/// Handover prediction model
pub struct HandoverModel {
    network: Network,
    config: ModelConfig,
    normalization_params: Option<NormalizationParams>,
    feature_importance: HashMap<String, f64>,
    training_history: TrainingHistory,
    model_id: String,
}

/// Training history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    pub training_errors: Vec<f64>,
    pub validation_errors: Vec<f64>,
    pub training_accuracies: Vec<f64>,
    pub validation_accuracies: Vec<f64>,
    pub epochs_completed: u32,
    pub best_epoch: u32,
    pub best_validation_accuracy: f64,
    pub total_training_time_ms: u64,
}

/// Training metrics for a single epoch
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub epoch: u32,
    pub training_error: f64,
    pub validation_error: f64,
    pub training_accuracy: f64,
    pub validation_accuracy: f64,
    pub learning_rate: f64,
    pub epoch_time_ms: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_size: 57, // From feature extraction
            hidden_layers: vec![128, 64, 32],
            output_size: 1, // Binary classification (handover probability)
            
            learning_rate: 0.001,
            momentum: 0.9,
            training_algorithm: TrainingAlgorithm::ResilientPropagation,
            max_epochs: 1000,
            desired_error: 0.001,
            epochs_between_reports: 10,
            
            dropout_rate: Some(0.2),
            weight_decay: Some(0.0001),
            
            validation_split: 0.2,
            batch_size: Some(32),
            shuffle_data: true,
            
            feature_normalization: true,
            feature_selection: false,
            feature_importance_threshold: Some(0.01),
            
            model_version: "1.0.0".to_string(),
            description: "Handover prediction neural network".to_string(),
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model_config: ModelConfig::default(),
            early_stopping: true,
            early_stopping_patience: 50,
            early_stopping_min_delta: 0.001,
            save_best_model: true,
            checkpoint_frequency: 100,
            enable_logging: true,
            cross_validation_folds: Some(5),
        }
    }
}

impl HandoverModel {
    /// Create a new model with the specified configuration
    pub fn new(config: ModelConfig) -> Result<Self> {
        let mut builder = NetworkBuilder::new();
        
        // Build network architecture
        builder.add_layer(config.input_size);
        for &hidden_size in &config.hidden_layers {
            builder.add_layer(hidden_size);
        }
        builder.add_layer(config.output_size);
        
        // Set activation functions
        builder.set_activation_function_hidden(ruv_fann::ActivationFunction::SigmoidSymmetric);
        builder.set_activation_function_output(ruv_fann::ActivationFunction::Sigmoid);
        
        let network = builder.build()?;
        
        Ok(Self {
            network,
            config,
            normalization_params: None,
            feature_importance: HashMap::new(),
            training_history: TrainingHistory::new(),
            model_id: uuid::Uuid::new_v4().to_string(),
        })
    }
    
    /// Train the model with the provided data
    pub fn train(
        &mut self,
        training_features: &[FeatureVector],
        training_labels: &[f64],
        validation_features: Option<&[FeatureVector]>,
        validation_labels: Option<&[f64]>,
        config: &TrainingConfig,
    ) -> Result<TrainingHistory> {
        let start_time = std::time::Instant::now();
        
        // Prepare normalization parameters
        if config.model_config.feature_normalization {
            self.normalization_params = Some(NormalizationParams::from_features(training_features)?);
        }
        
        // Normalize features
        let mut norm_training_features = training_features.to_vec();
        if let Some(ref norm_params) = self.normalization_params {
            for features in &mut norm_training_features {
                norm_params.normalize(features)?;
            }
        }
        
        let mut norm_validation_features = validation_features.map(|f| f.to_vec());
        if let (Some(ref norm_params), Some(ref mut val_features)) = 
            (&self.normalization_params, &mut norm_validation_features) {
            for features in val_features {
                norm_params.normalize(features)?;
            }
        }
        
        // Prepare training data
        let training_data = self.prepare_training_data(&norm_training_features, training_labels)?;
        
        // Configure training algorithm
        self.configure_training_algorithm(&config.model_config)?;
        
        // Train the network
        let mut best_validation_accuracy = 0.0;
        let mut patience_counter = 0;
        let mut best_network = self.network.clone();
        
        for epoch in 0..config.model_config.max_epochs {
            // Train one epoch
            let training_error = self.network.train_epoch(&training_data)?;
            
            // Calculate training accuracy
            let training_accuracy = self.calculate_accuracy(&norm_training_features, training_labels)?;
            
            // Validate if validation data is provided
            let (validation_error, validation_accuracy) = if let (Some(val_features), Some(val_labels)) = 
                (&norm_validation_features, validation_labels) {
                let val_error = self.calculate_validation_error(val_features, val_labels)?;
                let val_accuracy = self.calculate_accuracy(val_features, val_labels)?;
                (val_error, val_accuracy)
            } else {
                (training_error, training_accuracy)
            };
            
            // Update training history
            let epoch_metrics = EpochMetrics {
                epoch,
                training_error,
                validation_error,
                training_accuracy,
                validation_accuracy,
                learning_rate: config.model_config.learning_rate,
                epoch_time_ms: 0, // Will be updated
            };
            
            self.training_history.add_epoch(epoch_metrics);
            
            // Early stopping check
            if config.early_stopping {
                if validation_accuracy > best_validation_accuracy + config.early_stopping_min_delta {
                    best_validation_accuracy = validation_accuracy;
                    best_network = self.network.clone();
                    patience_counter = 0;
                    self.training_history.best_epoch = epoch;
                    self.training_history.best_validation_accuracy = best_validation_accuracy;
                } else {
                    patience_counter += 1;
                }
                
                if patience_counter >= config.early_stopping_patience {
                    if config.enable_logging {
                        tracing::info!("Early stopping at epoch {} with validation accuracy: {:.4}", 
                                     epoch, best_validation_accuracy);
                    }
                    break;
                }
            }
            
            // Progress reporting
            if epoch % config.model_config.epochs_between_reports == 0 && config.enable_logging {
                tracing::info!("Epoch {}: train_acc={:.4}, val_acc={:.4}, train_err={:.6}, val_err={:.6}",
                             epoch, training_accuracy, validation_accuracy, training_error, validation_error);
            }
            
            // Check for convergence
            if training_error <= config.model_config.desired_error {
                if config.enable_logging {
                    tracing::info!("Converged at epoch {} with error: {:.6}", epoch, training_error);
                }
                break;
            }
        }
        
        // Restore best model if using early stopping
        if config.early_stopping && config.save_best_model {
            self.network = best_network;
        }
        
        self.training_history.total_training_time_ms = start_time.elapsed().as_millis() as u64;
        self.training_history.epochs_completed = std::cmp::min(
            config.model_config.max_epochs,
            self.training_history.training_errors.len() as u32
        );
        
        // Calculate feature importance (simplified)
        if config.model_config.feature_selection {
            self.calculate_feature_importance(&norm_training_features, training_labels)?;
        }
        
        Ok(self.training_history.clone())
    }
    
    /// Predict handover probability for given features
    pub fn predict(&self, features: &FeatureVector) -> Result<f64> {
        let mut norm_features = features.clone();
        
        // Apply normalization if available
        if let Some(ref norm_params) = self.normalization_params {
            norm_params.normalize(&mut norm_features)?;
        }
        
        // Validate features
        norm_features.validate()?;
        
        // Convert to network input format
        let input = norm_features.to_array();
        let input_slice: Vec<f64> = input.to_vec();
        
        // Run prediction
        let output = self.network.run(&input_slice)?;
        
        // Return probability (output should be between 0 and 1)
        Ok(output[0].max(0.0).min(1.0))
    }
    
    /// Batch prediction for multiple feature vectors
    pub fn predict_batch(&self, features: &[FeatureVector]) -> Result<Vec<f64>> {
        let mut predictions = Vec::with_capacity(features.len());
        
        for feature_vec in features {
            predictions.push(self.predict(feature_vec)?);
        }
        
        Ok(predictions)
    }
    
    /// Save model to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let model_data = ModelData {
            network_state: self.serialize_network()?,
            config: self.config.clone(),
            normalization_params: self.normalization_params.clone(),
            feature_importance: self.feature_importance.clone(),
            training_history: self.training_history.clone(),
            model_id: self.model_id.clone(),
        };
        
        let serialized = serde_json::to_string_pretty(&model_data)?;
        std::fs::write(path, serialized)?;
        
        Ok(())
    }
    
    /// Load model from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let model_data: ModelData = serde_json::from_str(&content)?;
        
        let network = Self::deserialize_network(&model_data.network_state)?;
        
        Ok(Self {
            network,
            config: model_data.config,
            normalization_params: model_data.normalization_params,
            feature_importance: model_data.feature_importance,
            training_history: model_data.training_history,
            model_id: model_data.model_id,
        })
    }
    
    /// Get model information
    pub fn get_model_info(&self) -> ModelInfo {
        ModelInfo {
            model_id: self.model_id.clone(),
            model_version: self.config.model_version.clone(),
            training_timestamp: chrono::Utc::now().to_rfc3339(),
            training_accuracy: self.training_history.best_validation_accuracy,
            validation_accuracy: self.training_history.best_validation_accuracy,
            total_features: self.config.input_size as i32,
            feature_names: self.get_feature_names(),
            description: self.config.description.clone(),
        }
    }
    
    /// Configure training algorithm
    fn configure_training_algorithm(&mut self, config: &ModelConfig) -> Result<()> {
        match config.training_algorithm {
            TrainingAlgorithm::BackPropagation => {
                self.network.set_learning_rate(config.learning_rate as f32)?;
                self.network.set_learning_momentum(config.momentum as f32)?;
            },
            TrainingAlgorithm::ResilientPropagation => {
                // RProp doesn't use learning rate and momentum in the same way
                // The ruv-FANN library will handle RProp-specific parameters
            },
            TrainingAlgorithm::QuickPropagation => {
                self.network.set_learning_rate(config.learning_rate as f32)?;
            },
            TrainingAlgorithm::Batch => {
                self.network.set_learning_rate(config.learning_rate as f32)?;
            },
        }
        
        Ok(())
    }
    
    /// Prepare training data for ruv-FANN
    fn prepare_training_data(&self, features: &[FeatureVector], labels: &[f64]) -> Result<TrainingData> {
        if features.len() != labels.len() {
            return Err(OptMobError::Data(
                "Feature count doesn't match label count".to_string()
            ));
        }
        
        let mut training_data = TrainingData::new();
        
        for (feature_vec, &label) in features.iter().zip(labels.iter()) {
            let input: Vec<f32> = feature_vec.features.iter().map(|&x| x as f32).collect();
            let output = vec![label as f32];
            
            training_data.add_sample(input, output);
        }
        
        Ok(training_data)
    }
    
    /// Calculate accuracy on a dataset
    fn calculate_accuracy(&self, features: &[FeatureVector], labels: &[f64]) -> Result<f64> {
        let mut correct = 0;
        let threshold = 0.5;
        
        for (feature_vec, &actual_label) in features.iter().zip(labels.iter()) {
            let prediction = self.predict(feature_vec)?;
            let predicted_label = if prediction >= threshold { 1.0 } else { 0.0 };
            
            if (predicted_label - actual_label).abs() < 0.1 {
                correct += 1;
            }
        }
        
        Ok(correct as f64 / features.len() as f64)
    }
    
    /// Calculate validation error
    fn calculate_validation_error(&self, features: &[FeatureVector], labels: &[f64]) -> Result<f64> {
        let mut total_error = 0.0;
        
        for (feature_vec, &actual_label) in features.iter().zip(labels.iter()) {
            let prediction = self.predict(feature_vec)?;
            let error = (prediction - actual_label).powi(2);
            total_error += error;
        }
        
        Ok(total_error / features.len() as f64)
    }
    
    /// Calculate feature importance (simplified)
    fn calculate_feature_importance(&mut self, features: &[FeatureVector], labels: &[f64]) -> Result<()> {
        // Simplified feature importance calculation
        // In a full implementation, this could use permutation importance or other methods
        
        let baseline_accuracy = self.calculate_accuracy(features, labels)?;
        
        if !features.is_empty() {
            let feature_names = &features[0].feature_names;
            
            for (i, feature_name) in feature_names.iter().enumerate() {
                // Create modified features with this feature set to mean
                let feature_mean = features.iter()
                    .map(|f| f.features[i])
                    .sum::<f64>() / features.len() as f64;
                
                let mut modified_features = features.to_vec();
                for feature_vec in &mut modified_features {
                    feature_vec.features[i] = feature_mean;
                }
                
                let modified_accuracy = self.calculate_accuracy(&modified_features, labels)?;
                let importance = baseline_accuracy - modified_accuracy;
                
                self.feature_importance.insert(feature_name.clone(), importance);
            }
        }
        
        Ok(())
    }
    
    /// Serialize network (simplified)
    fn serialize_network(&self) -> Result<Vec<u8>> {
        // In a full implementation, this would serialize the network weights and structure
        // For now, we'll use a placeholder
        Ok(vec![0u8; 1024]) // Placeholder
    }
    
    /// Deserialize network (simplified)
    fn deserialize_network(_data: &[u8]) -> Result<Network> {
        // In a full implementation, this would reconstruct the network from serialized data
        // For now, we'll create a default network
        let mut builder = NetworkBuilder::new();
        builder.add_layer(57);
        builder.add_layer(128);
        builder.add_layer(64);
        builder.add_layer(32);
        builder.add_layer(1);
        Ok(builder.build()?)
    }
    
    /// Get feature names
    fn get_feature_names(&self) -> Vec<String> {
        // Return the standard feature names
        crate::features::FeatureExtractor::new(10).feature_names().to_vec()
    }
}

/// Serializable model data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelData {
    network_state: Vec<u8>,
    config: ModelConfig,
    normalization_params: Option<NormalizationParams>,
    feature_importance: HashMap<String, f64>,
    training_history: TrainingHistory,
    model_id: String,
}

/// Model information for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub model_version: String,
    pub training_timestamp: String,
    pub training_accuracy: f64,
    pub validation_accuracy: f64,
    pub total_features: i32,
    pub feature_names: Vec<String>,
    pub description: String,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            training_errors: Vec::new(),
            validation_errors: Vec::new(),
            training_accuracies: Vec::new(),
            validation_accuracies: Vec::new(),
            epochs_completed: 0,
            best_epoch: 0,
            best_validation_accuracy: 0.0,
            total_training_time_ms: 0,
        }
    }
    
    pub fn add_epoch(&mut self, metrics: EpochMetrics) {
        self.training_errors.push(metrics.training_error);
        self.validation_errors.push(metrics.validation_error);
        self.training_accuracies.push(metrics.training_accuracy);
        self.validation_accuracies.push(metrics.validation_accuracy);
    }
}

impl From<TrainingAlgorithm> for FannTrainingAlgorithm {
    fn from(algo: TrainingAlgorithm) -> Self {
        match algo {
            TrainingAlgorithm::BackPropagation => FannTrainingAlgorithm::IncrementalBackprop,
            TrainingAlgorithm::ResilientPropagation => FannTrainingAlgorithm::RProp,
            TrainingAlgorithm::QuickPropagation => FannTrainingAlgorithm::QuickProp,
            TrainingAlgorithm::Batch => FannTrainingAlgorithm::Batch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FeatureExtractor;
    use crate::data::UeMetrics;
    
    #[test]
    fn test_model_creation() {
        let config = ModelConfig::default();
        let model = HandoverModel::new(config);
        assert!(model.is_ok());
    }
    
    #[test]
    fn test_model_training() {
        // Create simple test data
        let mut extractor = FeatureExtractor::new(5);
        let mut features = Vec::new();
        let mut labels = Vec::new();
        
        for i in 0..10 {
            let metrics = UeMetrics::new(&format!("UE_{}", i), "Cell_1")
                .with_rsrp(-90.0 - i as f64)
                .with_sinr(10.0)
                .with_speed(30.0);
            
            extractor.add_metrics(metrics);
            if extractor.is_ready() {
                features.push(extractor.extract_features().unwrap());
                labels.push(if i % 2 == 0 { 1.0 } else { 0.0 });
            }
        }
        
        if !features.is_empty() {
            let mut config = ModelConfig::default();
            config.max_epochs = 10; // Quick test
            
            let mut model = HandoverModel::new(config.clone()).unwrap();
            let training_config = TrainingConfig {
                model_config: config,
                ..Default::default()
            };
            
            let result = model.train(&features, &labels, None, None, &training_config);
            assert!(result.is_ok());
        }
    }
}