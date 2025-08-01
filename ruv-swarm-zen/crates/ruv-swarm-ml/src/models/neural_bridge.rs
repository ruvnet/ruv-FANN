//! Bridge between ruv-fann neural networks and ruv-swarm-ml forecast models
//!
//! This module provides a simple integration layer to use ruv-fann's
//! neural network implementation as forecast models.

use alloc::{boxed::Box, string::{String, ToString}, vec::Vec};
use ruv_fann::{Network, NetworkBuilder, TrainingData};
use ruv_fann::training::{Adam, TrainingAlgorithm};

use super::{ForecastModel, ModelType, ModelParameters, TimeSeriesData};

/// A forecast model backed by a ruv-fann neural network
pub struct NeuralForecastModel {
    network: Network<f32>,
    model_type: ModelType,
    input_size: usize,
    output_size: usize,
}

impl NeuralForecastModel {
    /// Create a simple MLP forecast model
    pub fn new_mlp(input_size: usize, hidden_size: usize, output_size: usize) -> Result<Self, String> {
        let network = NetworkBuilder::new()
            .input_layer(input_size)
            .hidden_layer(hidden_size)
            .output_layer(output_size)
            .build();
        
        Ok(Self {
            network,
            model_type: ModelType::MLP,
            input_size,
            output_size,
        })
    }
    
    /// Create a deeper MLP with multiple hidden layers
    pub fn new_deep_mlp(input_size: usize, hidden_sizes: &[usize], output_size: usize) -> Result<Self, String> {
        let mut builder = NetworkBuilder::new().input_layer(input_size);
        
        for &hidden_size in hidden_sizes {
            builder = builder.hidden_layer(hidden_size);
        }
        
        let network = builder.output_layer(output_size).build();
        
        Ok(Self {
            network,
            model_type: ModelType::MLP,
            input_size,
            output_size,
        })
    }
    
    /// Train the neural network with time series data
    pub fn train(&mut self, inputs: &[Vec<f32>], outputs: &[Vec<f32>]) -> Result<f32, String> {
        let training_data = TrainingData { 
            inputs: inputs.to_vec(), 
            outputs: outputs.to_vec() 
        };
        
        // Use Adam optimizer for training
        let mut trainer = Adam::new(0.001)
            .with_beta1(0.9)
            .with_beta2(0.999)
            .with_epsilon(1e-8);
        
        // Train for a few epochs
        let mut error = 0.0;
        for _ in 0..10 {
            error = trainer.train_epoch(&mut self.network, &training_data)
                .map_err(|e| format!("Training error: {:?}", e))?;
        }
        
        Ok(error)
    }
    
    /// Generate predictions
    pub fn predict(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        Ok(self.network.run(input))
    }
}

impl ForecastModel for NeuralForecastModel {
    fn model_type(&self) -> ModelType {
        self.model_type
    }
    
    fn complexity_score(&self) -> f32 {
        // Estimate based on network size
        let total_weights = self.network.get_total_connections() as f32;
        total_weights / 1000.0 // Normalize to reasonable range
    }
    
    fn fit(&mut self, data: &TimeSeriesData) -> Result<(), String> {
        // Create sliding windows for time series prediction
        let window_size = self.input_size;
        let horizon = self.output_size;
        
        if data.values.len() < window_size + horizon {
            return Err("Insufficient data for training".to_string());
        }
        
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        
        for i in 0..data.values.len() - window_size - horizon + 1 {
            let input: Vec<f32> = data.values[i..i + window_size].to_vec();
            let output: Vec<f32> = data.values[i + window_size..i + window_size + horizon].to_vec();
            inputs.push(input);
            outputs.push(output);
        }
        
        // Train the network
        self.train(&inputs, &outputs)?;
        Ok(())
    }
    
    fn predict(&mut self, horizon: usize) -> Result<Vec<f32>, String> {
        // For now, just return zeros - in a real implementation,
        // this would use the last window of data
        Ok(vec![0.0; horizon])
    }
    
    fn get_parameters(&self) -> ModelParameters {
        ModelParameters {
            model_type: self.model_type,
            hyperparameters: vec![
                ("input_size".to_string(), self.input_size as f32),
                ("output_size".to_string(), self.output_size as f32),
            ],
            weights: None, // Could extract from network if needed
            metadata: vec![("implementation".to_string(), "ruv-fann".to_string())],
        }
    }
    
    fn load_parameters(&mut self, _params: ModelParameters) -> Result<(), String> {
        // For now, just return Ok - in a real implementation,
        // this would restore network weights
        Ok(())
    }
}

/// Factory function to create neural models based on ModelType
pub fn create_neural_model(
    model_type: ModelType,
    input_size: usize,
    output_size: usize,
) -> Result<Box<dyn ForecastModel>, String> {
    match model_type {
        ModelType::MLP => {
            // Simple MLP with one hidden layer
            let hidden_size = (input_size + output_size) / 2 + 5;
            Ok(Box::new(NeuralForecastModel::new_mlp(input_size, hidden_size, output_size)?))
        }
        ModelType::DLinear => {
            // DLinear can be approximated with a linear network (no hidden layer)
            let network = NetworkBuilder::new()
                .input_layer(input_size)
                .output_layer(output_size)
                .build();
            
            Ok(Box::new(NeuralForecastModel {
                network,
                model_type: ModelType::DLinear,
                input_size,
                output_size,
            }))
        }
        ModelType::MLPMultivariate => {
            // Wider MLP for multivariate
            let hidden_size = input_size + 10;
            let mut model = NeuralForecastModel::new_mlp(input_size, hidden_size, output_size)?;
            model.model_type = ModelType::MLPMultivariate;
            Ok(Box::new(model))
        }
        _ => {
            // For now, return a basic MLP for unsupported types
            // This allows gradual migration
            let hidden_size = (input_size + output_size) / 2 + 5;
            Ok(Box::new(NeuralForecastModel::new_mlp(input_size, hidden_size, output_size)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neural_forecast_model_creation() {
        let model = NeuralForecastModel::new_mlp(3, 5, 2).unwrap();
        assert_eq!(model.input_size, 3);
        assert_eq!(model.output_size, 2);
        assert_eq!(model.model_type(), ModelType::MLP);
    }
    
    #[test]
    fn test_neural_model_prediction() {
        let mut model = NeuralForecastModel::new_mlp(3, 5, 2).unwrap();
        let input = vec![1.0, 2.0, 3.0];
        let prediction = model.predict(&input).unwrap();
        assert_eq!(prediction.len(), 2);
    }
    
    #[test]
    fn test_neural_model_training() {
        let mut model = NeuralForecastModel::new_mlp(2, 3, 1).unwrap();
        
        // Simple XOR-like problem
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let outputs = vec![
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![0.0],
        ];
        
        let error = model.train(&inputs, &outputs).unwrap();
        assert!(error >= 0.0); // Error should be non-negative
    }
    
    #[test]
    fn test_model_factory() {
        let model = create_neural_model(ModelType::MLP, 4, 2).unwrap();
        assert_eq!(model.model_type(), ModelType::MLP);
        assert!(model.complexity_score() > 0.0);
    }
}