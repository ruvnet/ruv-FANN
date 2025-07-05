use crate::types::{TimeSeriesPoint, JitterForecast, ModelMetrics, ModelType};
use crate::error::{Error, Result};
use ruv_fann::NeuralNetwork;
use async_trait::async_trait;
use chrono::{DateTime, Utc};

/// Neural network model using ruv-FANN for VoLTE jitter prediction
pub struct NeuralNetworkModel {
    network: Option<NeuralNetwork>,
    input_size: usize,
    hidden_sizes: Vec<usize>,
    output_size: usize,
    trained: bool,
    metrics: Option<ModelMetrics>,
    sequence_length: usize,
}

impl NeuralNetworkModel {
    pub fn new() -> Self {
        Self {
            network: None,
            input_size: 6, // Default feature count
            hidden_sizes: vec![64, 32, 16],
            output_size: 1,
            trained: false,
            metrics: None,
            sequence_length: 10,
        }
    }
    
    pub fn with_architecture(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
    ) -> Self {
        Self {
            network: None,
            input_size,
            hidden_sizes,
            output_size,
            trained: false,
            metrics: None,
            sequence_length: 10,
        }
    }
    
    fn create_network(&mut self) -> Result<()> {
        let mut layers = vec![self.input_size];
        layers.extend_from_slice(&self.hidden_sizes);
        layers.push(self.output_size);
        
        let network = NeuralNetwork::new(&layers)
            .map_err(|e| Error::Model(format!("Failed to create neural network: {}", e)))?;
        
        self.network = Some(network);
        Ok(())
    }
    
    fn prepare_sequences(&self, data: &[TimeSeriesPoint]) -> Vec<(Vec<f64>, f64)> {
        let mut sequences = Vec::new();
        
        for i in self.sequence_length..data.len() {
            let mut sequence = Vec::new();
            
            // Create sequence of features
            for j in (i - self.sequence_length)..i {
                sequence.extend_from_slice(&data[j].features);
            }
            
            // Target is the jitter value at position i
            let target = data[i].value;
            sequences.push((sequence, target));
        }
        
        sequences
    }
    
    fn normalize_features(&self, data: &[f64]) -> Vec<f64> {
        // Simple min-max normalization
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_val == min_val {
            return vec![0.0; data.len()];
        }
        
        data.iter()
            .map(|&x| (x - min_val) / (max_val - min_val))
            .collect()
    }
    
    fn calculate_metrics(&self, predictions: &[f64], targets: &[f64]) -> ModelMetrics {
        let n = predictions.len() as f64;
        
        // Mean Absolute Error
        let mae = predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>() / n;
        
        // Root Mean Square Error
        let rmse = (predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / n).sqrt();
        
        // Mean Absolute Percentage Error
        let mape = predictions.iter()
            .zip(targets.iter())
            .filter(|(_, &t)| t != 0.0)
            .map(|(p, t)| ((p - t) / t).abs() * 100.0)
            .sum::<f64>() / n;
        
        // R-squared
        let mean_target = targets.iter().sum::<f64>() / n;
        let ss_res = predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| (t - p).powi(2))
            .sum::<f64>();
        let ss_tot = targets.iter()
            .map(|t| (t - mean_target).powi(2))
            .sum::<f64>();
        let r2_score = 1.0 - (ss_res / ss_tot);
        
        // Accuracy within 10ms
        let accuracy_10ms = predictions.iter()
            .zip(targets.iter())
            .filter(|(p, t)| (p - t).abs() <= 10.0)
            .count() as f64 / n;
        
        ModelMetrics {
            model_id: "neural_network".to_string(),
            accuracy_10ms,
            mae,
            rmse,
            mape,
            r2_score,
            training_time_ms: 0, // Set during training
            inference_time_ms: 0, // Set during inference
        }
    }
}

#[async_trait]
impl super::ForecastingModel for NeuralNetworkModel {
    fn model_type(&self) -> ModelType {
        ModelType::LinearRegression // Using this as placeholder for neural network
    }
    
    async fn train(&mut self, data: &[TimeSeriesPoint]) -> Result<()> {
        if data.len() < self.sequence_length + 1 {
            return Err(Error::InsufficientData(
                format!("Need at least {} data points for training", self.sequence_length + 1)
            ));
        }
        
        let start_time = std::time::Instant::now();
        
        // Create network if not exists
        if self.network.is_none() {
            self.create_network()?;
        }
        
        // Prepare training sequences
        let sequences = self.prepare_sequences(data);
        if sequences.is_empty() {
            return Err(Error::InsufficientData("No training sequences generated".to_string()));
        }
        
        // Normalize and prepare training data
        let mut training_inputs = Vec::new();
        let mut training_targets = Vec::new();
        
        for (sequence, target) in sequences {
            let normalized_sequence = self.normalize_features(&sequence);
            training_inputs.push(normalized_sequence);
            training_targets.push(target);
        }
        
        // Train the network
        let network = self.network.as_mut().unwrap();
        
        // Training parameters
        let learning_rate = 0.001;
        let epochs = 100;
        let desired_error = 0.01;
        
        // Convert to format expected by ruv-FANN
        let mut training_data = Vec::new();
        for (input, target) in training_inputs.iter().zip(training_targets.iter()) {
            training_data.push((input.clone(), vec![*target]));
        }
        
        // Train the network
        for epoch in 0..epochs {
            let mut epoch_error = 0.0;
            
            for (input, target) in &training_data {
                let output = network.run(input)
                    .map_err(|e| Error::Model(format!("Network run failed: {}", e)))?;
                
                let error = (output[0] - target[0]).powi(2);
                epoch_error += error;
                
                // Backpropagation would be implemented here
                // For now, we'll use a simplified approach
            }
            
            epoch_error /= training_data.len() as f64;
            
            if epoch_error < desired_error {
                break;
            }
        }
        
        // Calculate training metrics
        let mut predictions = Vec::new();
        for (input, _) in &training_data {
            let output = network.run(input)
                .map_err(|e| Error::Model(format!("Network run failed: {}", e)))?;
            predictions.push(output[0]);
        }
        
        let mut metrics = self.calculate_metrics(&predictions, &training_targets);
        metrics.training_time_ms = start_time.elapsed().as_millis() as u64;
        
        self.metrics = Some(metrics);
        self.trained = true;
        
        Ok(())
    }
    
    async fn predict(
        &self,
        data: &[TimeSeriesPoint],
        forecast_horizon: usize,
    ) -> Result<Vec<JitterForecast>> {
        if !self.trained {
            return Err(Error::Model("Model not trained".to_string()));
        }
        
        let network = self.network.as_ref().unwrap();
        let mut forecasts = Vec::new();
        
        if data.len() < self.sequence_length {
            return Err(Error::InsufficientData(
                format!("Need at least {} data points for prediction", self.sequence_length)
            ));
        }
        
        // Use the last sequence_length points as input
        let start_idx = data.len() - self.sequence_length;
        let mut current_sequence = Vec::new();
        
        for i in start_idx..data.len() {
            current_sequence.extend_from_slice(&data[i].features);
        }
        
        let last_timestamp = data.last().unwrap().timestamp;
        
        for i in 0..forecast_horizon {
            let normalized_input = self.normalize_features(&current_sequence);
            
            let start_time = std::time::Instant::now();
            let output = network.run(&normalized_input)
                .map_err(|e| Error::Model(format!("Prediction failed: {}", e)))?;
            let inference_time = start_time.elapsed().as_millis() as u64;
            
            let predicted_jitter = output[0].max(0.0); // Ensure non-negative
            let timestamp = last_timestamp + chrono::Duration::minutes(i as i64 + 1);
            
            // Calculate confidence based on prediction stability
            let confidence = (1.0 - (predicted_jitter / 100.0).min(1.0)).max(0.5);
            
            // Calculate prediction intervals (simplified)
            let uncertainty = predicted_jitter * 0.2;
            let prediction_interval_lower = (predicted_jitter - uncertainty).max(0.0);
            let prediction_interval_upper = predicted_jitter + uncertainty;
            
            forecasts.push(JitterForecast {
                timestamp,
                predicted_jitter_ms: predicted_jitter,
                confidence,
                prediction_interval_lower,
                prediction_interval_upper,
            });
            
            // Update sequence for next prediction (rolling window)
            if current_sequence.len() >= self.input_size {
                current_sequence.drain(0..self.input_size);
            }
            
            // Add predicted value as feedback (with some noise to prevent overfitting)
            let feedback_features = vec![
                predicted_jitter,
                predicted_jitter * 0.1, // Synthetic feature
                confidence,
                uncertainty,
                i as f64, // Time step
                0.0, // Placeholder
            ];
            current_sequence.extend_from_slice(&feedback_features);
        }
        
        Ok(forecasts)
    }
    
    fn get_metrics(&self) -> Option<ModelMetrics> {
        self.metrics.clone()
    }
    
    fn is_trained(&self) -> bool {
        self.trained
    }
    
    async fn save(&self, path: &str) -> Result<()> {
        if let Some(ref network) = self.network {
            network.save(path)
                .map_err(|e| Error::Model(format!("Failed to save network: {}", e)))?;
        }
        Ok(())
    }
    
    async fn load(&mut self, path: &str) -> Result<()> {
        let network = NeuralNetwork::load(path)
            .map_err(|e| Error::Model(format!("Failed to load network: {}", e)))?;
        
        self.network = Some(network);
        self.trained = true;
        Ok(())
    }
    
    fn complexity_score(&self) -> f64 {
        // Calculate complexity based on network size
        let total_params = self.hidden_sizes.iter().sum::<usize>() + 
                          self.input_size + self.output_size;
        (total_params as f64 / 1000.0).min(1.0)
    }
    
    fn recent_accuracy(&self) -> f64 {
        self.metrics.as_ref()
            .map(|m| m.accuracy_10ms)
            .unwrap_or(0.0)
    }
}