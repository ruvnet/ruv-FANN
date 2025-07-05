//! ML model implementation for SCell prediction

use crate::config::ModelConfig;
use crate::types::*;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use log::{debug, info, warn};
use ndarray::{Array1, Array2};
use ruv_fann::{ActivationFunction, Network, TrainingAlgorithm};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

/// SCell prediction model using ruv-FANN
#[derive(Debug)]
pub struct SCellPredictionModel {
    network: Arc<RwLock<Network>>,
    config: ModelConfig,
    model_id: String,
    created_at: DateTime<Utc>,
    last_trained_at: Option<DateTime<Utc>>,
    training_metrics: Arc<RwLock<ModelMetrics>>,
    feature_scaler: Arc<RwLock<Option<FeatureScaler>>>,
}

impl SCellPredictionModel {
    /// Create a new SCell prediction model
    pub async fn new(config: ModelConfig, model_id: String) -> Result<Self> {
        let network_config = &config.neural_network;
        
        // Create neural network with specified architecture
        let network = Network::new(&[
            network_config.input_neurons,
            network_config.hidden_layers.clone(),
            vec![network_config.output_neurons],
        ].concat())?;
        
        // Set activation function
        let activation_fn = match network_config.activation_function.as_str() {
            "sigmoid" => ActivationFunction::Sigmoid,
            "tanh" => ActivationFunction::Tanh,
            "relu" => ActivationFunction::Linear, // ruv-FANN doesn't have ReLU, use Linear
            _ => ActivationFunction::Sigmoid,
        };
        
        network.set_activation_function_hidden(activation_fn)?;
        network.set_activation_function_output(ActivationFunction::Sigmoid)?;
        
        // Set training algorithm
        let training_algo = match network_config.training_algorithm.as_str() {
            "rprop" => TrainingAlgorithm::Rprop,
            "quickprop" => TrainingAlgorithm::Quickprop,
            "batch" => TrainingAlgorithm::Batch,
            _ => TrainingAlgorithm::Rprop,
        };
        
        network.set_training_algorithm(training_algo)?;
        network.set_learning_rate(network_config.learning_rate)?;
        network.set_learning_momentum(network_config.momentum)?;
        
        Ok(Self {
            network: Arc::new(RwLock::new(network)),
            config,
            model_id,
            created_at: Utc::now(),
            last_trained_at: None,
            training_metrics: Arc::new(RwLock::new(ModelMetrics::new())),
            feature_scaler: Arc::new(RwLock::new(None)),
        })
    }
    
    /// Load model from file
    pub async fn load_from_file(model_path: &Path, config: ModelConfig) -> Result<Self> {
        let model_data = fs::read_to_string(model_path)?;
        let model_info: SavedModelInfo = serde_json::from_str(&model_data)?;
        
        // Load the neural network
        let network = Network::load_from_file(&model_info.network_path)?;
        
        // Load feature scaler if available
        let feature_scaler = if let Some(ref scaler_path) = model_info.scaler_path {
            let scaler_data = fs::read_to_string(scaler_path)?;
            Some(serde_json::from_str(&scaler_data)?)
        } else {
            None
        };
        
        Ok(Self {
            network: Arc::new(RwLock::new(network)),
            config,
            model_id: model_info.model_id,
            created_at: model_info.created_at,
            last_trained_at: model_info.last_trained_at,
            training_metrics: Arc::new(RwLock::new(model_info.training_metrics)),
            feature_scaler: Arc::new(RwLock::new(feature_scaler)),
        })
    }
    
    /// Save model to file
    pub async fn save_to_file(&self, model_path: &Path) -> Result<()> {
        let model_dir = model_path.parent().unwrap();
        fs::create_dir_all(model_dir)?;
        
        let network_path = model_dir.join(format!("{}_network.fann", self.model_id));
        let scaler_path = model_dir.join(format!("{}_scaler.json", self.model_id));
        
        // Save neural network
        {
            let network = self.network.read().await;
            network.save_to_file(&network_path)?;
        }
        
        // Save feature scaler if available
        let scaler_path_opt = {
            let scaler_guard = self.feature_scaler.read().await;
            if let Some(ref scaler) = *scaler_guard {
                let scaler_data = serde_json::to_string_pretty(scaler)?;
                fs::write(&scaler_path, scaler_data)?;
                Some(scaler_path)
            } else {
                None
            }
        };
        
        // Save model metadata
        let model_info = SavedModelInfo {
            model_id: self.model_id.clone(),
            created_at: self.created_at,
            last_trained_at: self.last_trained_at,
            training_metrics: self.training_metrics.read().await.clone(),
            network_path,
            scaler_path: scaler_path_opt,
        };
        
        let model_data = serde_json::to_string_pretty(&model_info)?;
        fs::write(model_path, model_data)?;
        
        Ok(())
    }
    
    /// Train the model with training data
    pub async fn train(&mut self, training_data: &[TrainingExample]) -> Result<ModelMetrics> {
        info!("Starting training with {} examples", training_data.len());
        
        if training_data.is_empty() {
            return Err(anyhow!("Training data cannot be empty"));
        }
        
        // Prepare training data
        let (inputs, targets) = self.prepare_training_data(training_data).await?;
        
        // Fit feature scaler
        let scaler = FeatureScaler::fit(&inputs)?;
        let scaled_inputs = scaler.transform(&inputs)?;
        
        {
            let mut scaler_guard = self.feature_scaler.write().await;
            *scaler_guard = Some(scaler);
        }
        
        // Split data for validation
        let split_idx = (inputs.nrows() as f32 * (1.0 - self.config.training.validation_split)) as usize;
        let train_inputs = scaled_inputs.slice(s![..split_idx, ..]).to_owned();
        let train_targets = targets.slice(s![..split_idx, ..]).to_owned();
        let val_inputs = scaled_inputs.slice(s![split_idx.., ..]).to_owned();
        let val_targets = targets.slice(s![split_idx.., ..]).to_owned();
        
        // Train the network
        let mut best_error = f32::INFINITY;
        let mut patience_counter = 0;
        let mut metrics = ModelMetrics::new();
        
        {
            let mut network = self.network.write().await;
            
            for epoch in 0..self.config.training.max_epochs {
                // Train one epoch
                let epoch_error = self.train_epoch(&mut network, &train_inputs, &train_targets).await?;
                
                // Validate
                if epoch % 10 == 0 {
                    let val_metrics = self.validate(&network, &val_inputs, &val_targets).await?;
                    
                    debug!("Epoch {}: train_error={:.4}, val_accuracy={:.4}", 
                           epoch, epoch_error, val_metrics.accuracy);
                    
                    // Early stopping
                    if epoch_error < best_error {
                        best_error = epoch_error;
                        patience_counter = 0;
                        metrics = val_metrics;
                    } else {
                        patience_counter += 1;
                        if patience_counter >= self.config.training.early_stopping_patience {
                            info!("Early stopping at epoch {} with best error: {:.4}", epoch, best_error);
                            break;
                        }
                    }
                }
                
                // Check minimum error threshold
                if epoch_error < self.config.training.min_error {
                    info!("Reached minimum error threshold at epoch {}", epoch);
                    break;
                }
            }
        }
        
        self.last_trained_at = Some(Utc::now());
        
        // Update training metrics
        {
            let mut metrics_guard = self.training_metrics.write().await;
            *metrics_guard = metrics.clone();
        }
        
        info!("Training completed. Final metrics: accuracy={:.4}, precision={:.4}, recall={:.4}", 
              metrics.accuracy, metrics.precision, metrics.recall);
        
        Ok(metrics)
    }
    
    /// Make prediction for a single UE
    pub async fn predict(&self, request: &PredictionRequest) -> Result<SCellPrediction> {
        let features = self.extract_features(request).await?;
        
        // Scale features
        let scaled_features = {
            let scaler_guard = self.feature_scaler.read().await;
            if let Some(ref scaler) = *scaler_guard {
                scaler.transform_single(&features)?
            } else {
                features
            }
        };
        
        // Run inference
        let outputs = {
            let network = self.network.read().await;
            network.run(&scaled_features)?
        };
        
        // Parse outputs
        let activation_prob = outputs[0];
        let throughput_demand = if outputs.len() > 1 { outputs[1] } else { 0.0 };
        
        let scell_activation_recommended = activation_prob > self.config.confidence_threshold;
        
        let reasoning = self.generate_reasoning(activation_prob, throughput_demand, request).await;
        
        Ok(SCellPrediction {
            ue_id: request.ue_id.clone(),
            scell_activation_recommended,
            confidence_score: activation_prob,
            predicted_throughput_demand: throughput_demand * self.config.throughput_threshold_mbps,
            reasoning,
            timestamp_utc: Utc::now(),
        })
    }
    
    /// Get model metrics
    pub async fn get_metrics(&self) -> ModelMetrics {
        self.training_metrics.read().await.clone()
    }
    
    /// Get model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
    
    /// Get creation timestamp
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
    
    /// Get last training timestamp
    pub fn last_trained_at(&self) -> Option<DateTime<Utc>> {
        self.last_trained_at
    }
    
    // Private helper methods
    
    async fn prepare_training_data(&self, examples: &[TrainingExample]) -> Result<(Array2<f32>, Array2<f32>)> {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        
        for example in examples {
            let features = self.extract_features_from_example(example).await;
            inputs.push(features);
            
            let target = vec![
                if example.actual_scell_needed { 1.0 } else { 0.0 },
                example.actual_throughput_demand / self.config.throughput_threshold_mbps,
            ];
            targets.push(target);
        }
        
        let input_array = Array2::from_shape_vec(
            (inputs.len(), inputs[0].len()),
            inputs.into_iter().flatten().collect()
        )?;
        
        let target_array = Array2::from_shape_vec(
            (targets.len(), targets[0].len()),
            targets.into_iter().flatten().collect()
        )?;
        
        Ok((input_array, target_array))
    }
    
    async fn extract_features(&self, request: &PredictionRequest) -> Result<Vec<f32>> {
        let current_features = request.current_metrics.to_feature_vector();
        
        // Add historical features (rolling statistics)
        let historical_features = if !request.historical_metrics.is_empty() {
            self.compute_historical_features(&request.historical_metrics)
        } else {
            vec![0.0; 5] // Default historical features
        };
        
        let mut features = current_features;
        features.extend(historical_features);
        
        Ok(features)
    }
    
    async fn extract_features_from_example(&self, example: &TrainingExample) -> Vec<f32> {
        let current_features = example.input_metrics.to_feature_vector();
        
        let historical_features = if !example.historical_sequence.is_empty() {
            self.compute_historical_features(&example.historical_sequence)
        } else {
            vec![0.0; 5] // Default historical features
        };
        
        let mut features = current_features;
        features.extend(historical_features);
        
        features
    }
    
    fn compute_historical_features(&self, historical_metrics: &[UEMetrics]) -> Vec<f32> {
        if historical_metrics.is_empty() {
            return vec![0.0; 5];
        }
        
        let throughputs: Vec<f32> = historical_metrics.iter()
            .map(|m| m.pcell_throughput_mbps)
            .collect();
        
        let cqis: Vec<f32> = historical_metrics.iter()
            .map(|m| m.pcell_cqi)
            .collect();
        
        vec![
            throughputs.iter().sum::<f32>() / throughputs.len() as f32, // mean throughput
            self.compute_std(&throughputs), // std throughput
            cqis.iter().sum::<f32>() / cqis.len() as f32, // mean CQI
            self.compute_std(&cqis), // std CQI
            throughputs.iter().copied().fold(0.0, f32::max), // max throughput
        ]
    }
    
    fn compute_std(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        
        variance.sqrt()
    }
    
    async fn train_epoch(&self, network: &mut Network, inputs: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        let mut total_error = 0.0;
        let batch_size = self.config.training.batch_size;
        
        for batch_start in (0..inputs.nrows()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(inputs.nrows());
            
            for i in batch_start..batch_end {
                let input = inputs.row(i).to_vec();
                let target = targets.row(i).to_vec();
                
                network.train(&input, &target)?;
                
                let output = network.run(&input)?;
                let error = self.compute_error(&output, &target);
                total_error += error;
            }
        }
        
        Ok(total_error / inputs.nrows() as f32)
    }
    
    async fn validate(&self, network: &Network, inputs: &Array2<f32>, targets: &Array2<f32>) -> Result<ModelMetrics> {
        let mut metrics = ModelMetrics::new();
        
        for i in 0..inputs.nrows() {
            let input = inputs.row(i).to_vec();
            let target = targets.row(i).to_vec();
            
            let output = network.run(&input)?;
            let predicted = output[0] > 0.5;
            let actual = target[0] > 0.5;
            
            match (predicted, actual) {
                (true, true) => metrics.true_positives += 1,
                (true, false) => metrics.false_positives += 1,
                (false, true) => metrics.false_negatives += 1,
                (false, false) => metrics.true_negatives += 1,
            }
            
            let error = (output[0] - target[0]).abs();
            metrics.mean_absolute_error += error;
        }
        
        metrics.total_predictions = inputs.nrows() as i32;
        metrics.mean_absolute_error /= inputs.nrows() as f32;
        metrics.calculate_derived_metrics();
        
        Ok(metrics)
    }
    
    fn compute_error(&self, output: &[f32], target: &[f32]) -> f32 {
        output.iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f32>()
            / output.len() as f32
    }
    
    async fn generate_reasoning(&self, activation_prob: f32, throughput_demand: f32, request: &PredictionRequest) -> String {
        let mut reasoning = String::new();
        
        if activation_prob > self.config.confidence_threshold {
            reasoning.push_str("SCell activation recommended based on: ");
            
            if request.current_metrics.pcell_throughput_mbps > 80.0 {
                reasoning.push_str("high current throughput, ");
            }
            
            if request.current_metrics.buffer_status_report_bytes > 100000 {
                reasoning.push_str("large buffer size, ");
            }
            
            if request.current_metrics.pcell_cqi > 10.0 {
                reasoning.push_str("good channel quality, ");
            }
            
            reasoning.push_str(&format!("predicted demand: {:.1} Mbps", throughput_demand));
        } else {
            reasoning.push_str("SCell activation not needed: ");
            reasoning.push_str(&format!("low predicted demand ({:.1} Mbps), ", throughput_demand));
            reasoning.push_str(&format!("confidence: {:.3}", activation_prob));
        }
        
        reasoning
    }
}

/// Feature scaler for normalizing input features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureScaler {
    means: Vec<f32>,
    stds: Vec<f32>,
}

impl FeatureScaler {
    pub fn fit(data: &Array2<f32>) -> Result<Self> {
        let mut means = Vec::new();
        let mut stds = Vec::new();
        
        for col in 0..data.ncols() {
            let column = data.column(col);
            let mean = column.sum() / column.len() as f32;
            let variance = column.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / column.len() as f32;
            let std = variance.sqrt();
            
            means.push(mean);
            stds.push(if std > 0.0 { std } else { 1.0 });
        }
        
        Ok(Self { means, stds })
    }
    
    pub fn transform(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let mut result = data.clone();
        
        for ((i, j), value) in result.indexed_iter_mut() {
            *value = (*value - self.means[j]) / self.stds[j];
        }
        
        Ok(result)
    }
    
    pub fn transform_single(&self, data: &[f32]) -> Result<Vec<f32>> {
        if data.len() != self.means.len() {
            return Err(anyhow!("Data length mismatch"));
        }
        
        let mut result = Vec::new();
        for (i, &value) in data.iter().enumerate() {
            result.push((value - self.means[i]) / self.stds[i]);
        }
        
        Ok(result)
    }
}

/// Saved model information
#[derive(Debug, Serialize, Deserialize)]
struct SavedModelInfo {
    model_id: String,
    created_at: DateTime<Utc>,
    last_trained_at: Option<DateTime<Utc>>,
    training_metrics: ModelMetrics,
    network_path: PathBuf,
    scaler_path: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    
    #[tokio::test]
    async fn test_model_creation() {
        let config = ModelConfig::default();
        let model = SCellPredictionModel::new(config, "test_model".to_string()).await;
        assert!(model.is_ok());
    }
    
    #[tokio::test]
    async fn test_feature_scaler() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let scaler = FeatureScaler::fit(&data).unwrap();
        let scaled = scaler.transform(&data).unwrap();
        
        // Check that scaled data has approximately zero mean
        let col0_mean = scaled.column(0).sum() / scaled.nrows() as f32;
        let col1_mean = scaled.column(1).sum() / scaled.nrows() as f32;
        
        assert!((col0_mean).abs() < 1e-6);
        assert!((col1_mean).abs() < 1e-6);
    }
}