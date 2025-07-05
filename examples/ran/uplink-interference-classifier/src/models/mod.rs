//! Neural network models for interference classification
//! 
//! This module contains the implementation of neural network models using ruv-FANN
//! for classifying different types of uplink interference.

use crate::{
    InterferenceClass, InterferenceClassifierError, Result, ModelConfig, ModelMetrics,
    TrainingExample, ClassificationResult, FEATURE_VECTOR_SIZE, TARGET_ACCURACY_THRESHOLD,
};
use ruv_fann::{Network, ActivationFunction, TrainingAlgorithm, TrainingData, ErrorFunction};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Neural network classifier for interference detection
pub struct InterferenceClassifierModel {
    network: Network,
    model_id: String,
    model_version: String,
    training_config: ModelConfig,
    class_weights: HashMap<InterferenceClass, f64>,
    training_metrics: Option<ModelMetrics>,
    created_at: DateTime<Utc>,
    last_updated: DateTime<Utc>,
}

impl InterferenceClassifierModel {
    /// Create a new interference classifier model
    pub fn new(config: ModelConfig) -> Result<Self> {
        let mut network = Network::new();
        
        // Build network architecture
        let mut layers = vec![FEATURE_VECTOR_SIZE as u32];
        layers.extend(config.hidden_layers.iter().cloned());
        layers.push(InterferenceClass::num_classes() as u32);
        
        network.create_standard_array(&layers)
            .map_err(|e| InterferenceClassifierError::ModelLoadingError(
                format!("Failed to create network: {}", e)
            ))?;
        
        // Configure activation functions
        let activation_func = match config.activation_function.as_str() {
            "relu" => ActivationFunction::RectifiedLinear,
            "sigmoid" => ActivationFunction::Sigmoid,
            "tanh" => ActivationFunction::Tanh,
            "leaky_relu" => ActivationFunction::LeakyRectifiedLinear,
            _ => ActivationFunction::RectifiedLinear,
        };
        
        network.set_activation_function_hidden(activation_func);
        network.set_activation_function_output(ActivationFunction::Softmax);
        
        // Configure training parameters
        network.set_learning_rate(config.learning_rate);
        network.set_training_algorithm(TrainingAlgorithm::Rprop);
        network.set_train_error_function(ErrorFunction::CrossEntropy);
        
        // Set up class weights for balanced training
        let mut class_weights = HashMap::new();
        for i in 0..InterferenceClass::num_classes() {
            let class = InterferenceClass::from_index(i);
            class_weights.insert(class, 1.0); // Initially uniform weights
        }
        
        let now = Utc::now();
        
        Ok(Self {
            network,
            model_id: Uuid::new_v4().to_string(),
            model_version: "1.0.0".to_string(),
            training_config: config,
            class_weights,
            training_metrics: None,
            created_at: now,
            last_updated: now,
        })
    }
    
    /// Train the model with provided training examples
    pub fn train(&mut self, training_examples: &[TrainingExample]) -> Result<ModelMetrics> {
        if training_examples.is_empty() {
            return Err(InterferenceClassifierError::TrainingError(
                "No training examples provided".to_string()
            ));
        }
        
        // Convert training examples to training data format
        let (inputs, outputs) = self.prepare_training_data(training_examples)?;
        
        // Create training data
        let training_data = TrainingData::new(inputs, outputs)
            .map_err(|e| InterferenceClassifierError::TrainingError(
                format!("Failed to create training data: {}", e)
            ))?;
        
        // Calculate class weights for balanced training
        self.calculate_class_weights(training_examples);
        
        // Train the network
        let max_epochs = self.training_config.max_epochs;
        let target_error = 1.0 - self.training_config.target_accuracy;
        
        log::info!("Starting training with {} examples, target accuracy: {:.2}%", 
                  training_examples.len(), self.training_config.target_accuracy * 100.0);
        
        let mut best_metrics = ModelMetrics {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            class_metrics: HashMap::new(),
            confusion_matrix: vec![vec![0; InterferenceClass::num_classes()]; InterferenceClass::num_classes()],
        };
        
        let mut epochs_without_improvement = 0;
        let patience = 50; // Early stopping patience
        
        for epoch in 0..max_epochs {
            // Train for one epoch
            let epoch_error = self.network.train_epoch(&training_data)
                .map_err(|e| InterferenceClassifierError::TrainingError(
                    format!("Training epoch failed: {}", e)
                ))?;
            
            // Evaluate every 10 epochs
            if epoch % 10 == 0 {
                let metrics = self.evaluate_model(training_examples)?;
                
                if metrics.accuracy > best_metrics.accuracy {
                    best_metrics = metrics;
                    epochs_without_improvement = 0;
                } else {
                    epochs_without_improvement += 10;
                }
                
                log::info!("Epoch {}: Error = {:.6}, Accuracy = {:.4}%", 
                          epoch, epoch_error, best_metrics.accuracy * 100.0);
                
                // Check if target accuracy is reached
                if best_metrics.accuracy >= self.training_config.target_accuracy {
                    log::info!("Target accuracy reached at epoch {}", epoch);
                    break;
                }
                
                // Early stopping
                if epochs_without_improvement >= patience {
                    log::info!("Early stopping at epoch {} due to no improvement", epoch);
                    break;
                }
            }
            
            // Check for convergence
            if epoch_error < target_error {
                log::info!("Training converged at epoch {}", epoch);
                break;
            }
        }
        
        self.training_metrics = Some(best_metrics.clone());
        self.last_updated = Utc::now();
        
        log::info!("Training completed. Final accuracy: {:.2}%", best_metrics.accuracy * 100.0);
        
        Ok(best_metrics)
    }
    
    /// Classify interference from feature vector
    pub fn classify(&self, feature_vector: &[f64]) -> Result<ClassificationResult> {
        if feature_vector.len() != FEATURE_VECTOR_SIZE {
            return Err(InterferenceClassifierError::ClassificationError(
                format!("Invalid feature vector size: {} != {}", 
                       feature_vector.len(), FEATURE_VECTOR_SIZE)
            ));
        }
        
        // Run inference
        let output = self.network.run(feature_vector)
            .map_err(|e| InterferenceClassifierError::ClassificationError(
                format!("Network inference failed: {}", e)
            ))?;
        
        // Find the class with highest probability
        let max_index = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0);
        
        let confidence = output[max_index];
        let interference_class = InterferenceClass::from_index(max_index);
        
        Ok(ClassificationResult {
            interference_class,
            confidence,
            timestamp: Utc::now(),
            feature_vector: feature_vector.to_vec(),
            model_version: self.model_version.clone(),
        })
    }
    
    /// Get classification probabilities for all classes
    pub fn get_class_probabilities(&self, feature_vector: &[f64]) -> Result<HashMap<InterferenceClass, f64>> {
        if feature_vector.len() != FEATURE_VECTOR_SIZE {
            return Err(InterferenceClassifierError::ClassificationError(
                format!("Invalid feature vector size: {} != {}", 
                       feature_vector.len(), FEATURE_VECTOR_SIZE)
            ));
        }
        
        let output = self.network.run(feature_vector)
            .map_err(|e| InterferenceClassifierError::ClassificationError(
                format!("Network inference failed: {}", e)
            ))?;
        
        let mut probabilities = HashMap::new();
        for (i, &prob) in output.iter().enumerate() {
            let class = InterferenceClass::from_index(i);
            probabilities.insert(class, prob);
        }
        
        Ok(probabilities)
    }
    
    /// Evaluate model performance on test data
    pub fn evaluate_model(&self, test_examples: &[TrainingExample]) -> Result<ModelMetrics> {
        if test_examples.is_empty() {
            return Err(InterferenceClassifierError::InvalidInputError(
                "No test examples provided".to_string()
            ));
        }
        
        let num_classes = InterferenceClass::num_classes();
        let mut confusion_matrix = vec![vec![0u32; num_classes]; num_classes];
        let mut class_counts = HashMap::new();
        let mut class_correct = HashMap::new();
        
        // Initialize counters
        for i in 0..num_classes {
            let class = InterferenceClass::from_index(i);
            class_counts.insert(class.clone(), 0);
            class_correct.insert(class, 0);
        }
        
        // Process each test example
        for example in test_examples {
            // Extract features (simplified - in practice would use FeatureExtractor)
            let feature_vector = self.extract_features_from_example(example)?;
            
            // Classify
            let result = self.classify(&feature_vector)?;
            
            // Update confusion matrix
            let true_idx = example.true_interference_class.to_index();
            let pred_idx = result.interference_class.to_index();
            confusion_matrix[true_idx][pred_idx] += 1;
            
            // Update counters
            *class_counts.get_mut(&example.true_interference_class).unwrap() += 1;
            if result.interference_class == example.true_interference_class {
                *class_correct.get_mut(&example.true_interference_class).unwrap() += 1;
            }
        }
        
        // Calculate metrics
        let total_correct: u32 = class_correct.values().sum();
        let total_samples = test_examples.len() as f64;
        let accuracy = total_correct as f64 / total_samples;
        
        // Calculate per-class metrics
        let mut class_metrics = HashMap::new();
        let mut total_precision = 0.0;
        let mut total_recall = 0.0;
        let mut total_f1 = 0.0;
        
        for i in 0..num_classes {
            let class = InterferenceClass::from_index(i);
            let true_positives = confusion_matrix[i][i] as f64;
            let false_positives: f64 = (0..num_classes).map(|j| confusion_matrix[j][i] as f64).sum::<f64>() - true_positives;
            let false_negatives: f64 = (0..num_classes).map(|j| confusion_matrix[i][j] as f64).sum::<f64>() - true_positives;
            
            let precision = if true_positives + false_positives > 0.0 {
                true_positives / (true_positives + false_positives)
            } else {
                0.0
            };
            
            let recall = if true_positives + false_negatives > 0.0 {
                true_positives / (true_positives + false_negatives)
            } else {
                0.0
            };
            
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            
            class_metrics.insert(format!("{}_precision", class.as_str()), precision);
            class_metrics.insert(format!("{}_recall", class.as_str()), recall);
            class_metrics.insert(format!("{}_f1", class.as_str()), f1);
            
            total_precision += precision;
            total_recall += recall;
            total_f1 += f1;
        }
        
        // Calculate macro-averaged metrics
        let macro_precision = total_precision / num_classes as f64;
        let macro_recall = total_recall / num_classes as f64;
        let macro_f1 = total_f1 / num_classes as f64;
        
        Ok(ModelMetrics {
            accuracy,
            precision: macro_precision,
            recall: macro_recall,
            f1_score: macro_f1,
            class_metrics,
            confusion_matrix,
        })
    }
    
    /// Save model to file
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.network.save(path.as_ref())
            .map_err(|e| InterferenceClassifierError::ModelLoadingError(
                format!("Failed to save model: {}", e)
            ))?;
        
        // Save metadata
        let metadata = ModelMetadata {
            model_id: self.model_id.clone(),
            model_version: self.model_version.clone(),
            training_config: self.training_config.clone(),
            class_weights: self.class_weights.clone(),
            training_metrics: self.training_metrics.clone(),
            created_at: self.created_at,
            last_updated: self.last_updated,
        };
        
        let metadata_path = path.as_ref().with_extension("json");
        let metadata_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| InterferenceClassifierError::ModelLoadingError(
                format!("Failed to serialize metadata: {}", e)
            ))?;
        
        std::fs::write(metadata_path, metadata_json)
            .map_err(|e| InterferenceClassifierError::ModelLoadingError(
                format!("Failed to write metadata: {}", e)
            ))?;
        
        Ok(())
    }
    
    /// Load model from file
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<Self> {
        let network = Network::load(path.as_ref())
            .map_err(|e| InterferenceClassifierError::ModelLoadingError(
                format!("Failed to load model: {}", e)
            ))?;
        
        // Load metadata
        let metadata_path = path.as_ref().with_extension("json");
        let metadata_json = std::fs::read_to_string(metadata_path)
            .map_err(|e| InterferenceClassifierError::ModelLoadingError(
                format!("Failed to read metadata: {}", e)
            ))?;
        
        let metadata: ModelMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| InterferenceClassifierError::ModelLoadingError(
                format!("Failed to deserialize metadata: {}", e)
            ))?;
        
        Ok(Self {
            network,
            model_id: metadata.model_id,
            model_version: metadata.model_version,
            training_config: metadata.training_config,
            class_weights: metadata.class_weights,
            training_metrics: metadata.training_metrics,
            created_at: metadata.created_at,
            last_updated: metadata.last_updated,
        })
    }
    
    /// Get model information
    pub fn get_model_info(&self) -> ModelInfo {
        ModelInfo {
            model_id: self.model_id.clone(),
            model_version: self.model_version.clone(),
            created_at: self.created_at,
            last_updated: self.last_updated,
            training_accuracy: self.training_metrics.as_ref().map(|m| m.accuracy),
            feature_vector_size: FEATURE_VECTOR_SIZE,
            num_classes: InterferenceClass::num_classes(),
        }
    }
    
    /// Private helper methods
    fn prepare_training_data(&self, examples: &[TrainingExample]) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        
        for example in examples {
            // Extract features (simplified - in practice would use FeatureExtractor)
            let feature_vector = self.extract_features_from_example(example)?;
            inputs.push(feature_vector);
            
            // Create one-hot encoded output
            let mut output = vec![0.0; InterferenceClass::num_classes()];
            output[example.true_interference_class.to_index()] = 1.0;
            outputs.push(output);
        }
        
        Ok((inputs, outputs))
    }
    
    fn extract_features_from_example(&self, example: &TrainingExample) -> Result<Vec<f64>> {
        // This is a simplified feature extraction
        // In practice, this would use the FeatureExtractor
        let mut features = Vec::with_capacity(FEATURE_VECTOR_SIZE);
        
        if example.measurements.is_empty() {
            return Err(InterferenceClassifierError::FeatureExtractionError(
                "No measurements in example".to_string()
            ));
        }
        
        // Extract basic statistics from measurements
        let noise_pusch: Vec<f64> = example.measurements.iter()
            .map(|m| m.noise_floor_pusch)
            .collect();
        let noise_pucch: Vec<f64> = example.measurements.iter()
            .map(|m| m.noise_floor_pucch)
            .collect();
        
        // Add mean values
        features.push(noise_pusch.iter().sum::<f64>() / noise_pusch.len() as f64);
        features.push(noise_pucch.iter().sum::<f64>() / noise_pucch.len() as f64);
        
        // Add other features to reach FEATURE_VECTOR_SIZE
        while features.len() < FEATURE_VECTOR_SIZE {
            features.push(0.0);
        }
        
        Ok(features)
    }
    
    fn calculate_class_weights(&mut self, examples: &[TrainingExample]) {
        let mut class_counts = HashMap::new();
        
        // Count examples per class
        for example in examples {
            *class_counts.entry(example.true_interference_class.clone()).or_insert(0) += 1;
        }
        
        // Calculate inverse frequency weights
        let total_samples = examples.len() as f64;
        for (class, count) in class_counts {
            let weight = total_samples / (InterferenceClass::num_classes() as f64 * count as f64);
            self.class_weights.insert(class, weight);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelMetadata {
    model_id: String,
    model_version: String,
    training_config: ModelConfig,
    class_weights: HashMap<InterferenceClass, f64>,
    training_metrics: Option<ModelMetrics>,
    created_at: DateTime<Utc>,
    last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub model_version: String,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub training_accuracy: Option<f64>,
    pub feature_vector_size: usize,
    pub num_classes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NoiseFloorMeasurement, CellParameters};
    use chrono::Utc;
    
    fn create_test_example(class: InterferenceClass) -> TrainingExample {
        let measurement = NoiseFloorMeasurement {
            timestamp: Utc::now(),
            noise_floor_pusch: -100.0,
            noise_floor_pucch: -102.0,
            cell_ret: 0.05,
            rsrp: -80.0,
            sinr: 15.0,
            active_users: 50,
            prb_utilization: 0.6,
        };
        
        let cell_params = CellParameters {
            cell_id: "test_cell".to_string(),
            frequency_band: "B1".to_string(),
            tx_power: 43.0,
            antenna_count: 4,
            bandwidth_mhz: 20.0,
            technology: "LTE".to_string(),
        };
        
        TrainingExample {
            measurements: vec![measurement; 20],
            cell_params,
            true_interference_class: class,
        }
    }
    
    #[test]
    fn test_model_creation() {
        let config = ModelConfig::default();
        let model = InterferenceClassifierModel::new(config);
        assert!(model.is_ok());
    }
    
    #[test]
    fn test_feature_extraction() {
        let config = ModelConfig::default();
        let model = InterferenceClassifierModel::new(config).unwrap();
        let example = create_test_example(InterferenceClass::ThermalNoise);
        
        let features = model.extract_features_from_example(&example);
        assert!(features.is_ok());
        assert_eq!(features.unwrap().len(), FEATURE_VECTOR_SIZE);
    }
    
    #[test]
    fn test_classification() {
        let config = ModelConfig::default();
        let model = InterferenceClassifierModel::new(config).unwrap();
        let feature_vector = vec![0.0; FEATURE_VECTOR_SIZE];
        
        let result = model.classify(&feature_vector);
        assert!(result.is_ok());
        
        let classification = result.unwrap();
        assert!(classification.confidence >= 0.0 && classification.confidence <= 1.0);
    }
}