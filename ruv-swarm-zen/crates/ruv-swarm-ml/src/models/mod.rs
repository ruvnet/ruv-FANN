//! Forecasting model definitions and factory
//!
//! This module provides definitions for all 27+ forecasting models supported
//! by ruv-fann neural networks and the factory for creating them.

use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt;

pub mod neural_models;
pub mod neural_bridge;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Enumeration of all supported forecasting model types
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ModelType {
    // Basic Models
    MLP,
    DLinear,
    NLinear,
    MLPMultivariate,

    // Recurrent Models
    RNN,
    LSTM,
    GRU,

    // Advanced Models
    NBEATS,
    NBEATSx,
    NHITS,
    TiDE,

    // Transformer Models
    TFT,
    Informer,
    AutoFormer,
    FedFormer,
    PatchTST,
    ITransformer,

    // Specialized Models
    DeepAR,
    DeepNPTS,
    TCN,
    BiTCN,
    TimesNet,
    StemGNN,
    TSMixer,
    TSMixerx,
    PatchMixer,
    SegRNN,
    DishTS,
}

/// Model metadata and requirements
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub name: String,
    pub model_type: ModelType,
    pub category: ModelCategory,
    pub min_samples: usize,
    pub supports_multivariate: bool,
    pub supports_static_features: bool,
    pub supports_probabilistic: bool,
    pub typical_memory_mb: f32,
    pub typical_training_time: TrainingTime,
    pub interpretability_score: f32, // 0.0 to 1.0
}

/// Model category for grouping similar models
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelCategory {
    Basic,
    Recurrent,
    Advanced,
    Transformer,
    Specialized,
}

/// Expected training time category
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrainingTime {
    Fast,     // < 1 minute
    Medium,   // 1-10 minutes
    Slow,     // 10-60 minutes
    VerySlow, // > 60 minutes
}

/// Trait for forecast models
pub trait ForecastModel {
    /// Get the model type
    fn model_type(&self) -> ModelType;

    /// Get model complexity score (number of parameters)
    fn complexity_score(&self) -> f32;

    /// Fit the model to training data
    fn fit(&mut self, data: &TimeSeriesData) -> Result<(), String>;

    /// Generate predictions
    fn predict(&mut self, horizon: usize) -> Result<Vec<f32>, String>;

    /// Get model parameters for serialization
    fn get_parameters(&self) -> ModelParameters;

    /// Load model parameters
    fn load_parameters(&mut self, params: ModelParameters) -> Result<(), String>;
}

/// Model parameters for serialization
#[derive(Clone, Debug)]
pub struct ModelParameters {
    pub model_type: ModelType,
    pub hyperparameters: Vec<(String, f32)>,
    pub weights: Option<Vec<f32>>,
    pub metadata: Vec<(String, String)>,
}

/// Time series data structure
#[derive(Clone, Debug)]
pub struct TimeSeriesData {
    pub values: Vec<f32>,
    pub timestamps: Vec<f64>,
    pub frequency: String,
    pub static_features: Option<Vec<f32>>,
    pub dynamic_features: Option<Vec<Vec<f32>>>,
}

/// Model factory for creating forecasting models
pub struct ModelFactory;

impl ModelFactory {
    /// Create a concrete neural network model instance
    pub fn create_model(model_type: ModelType, input_size: usize, output_size: usize) -> Result<Box<dyn ForecastModel>, String> {
        neural_models::create_model(model_type, input_size, output_size)
    }
    
    /// Create a model with default parameters for time series forecasting
    pub fn create_default_model(model_type: ModelType, sequence_length: usize, horizon: usize) -> Result<Box<dyn ForecastModel>, String> {
        Self::create_model(model_type, sequence_length, horizon)
    }
    
    /// Get information about all available models
    pub fn get_available_models() -> Vec<ModelInfo> {
        vec![
            // Basic Models
            ModelInfo {
                name: "Multi-Layer Perceptron".to_string(),
                model_type: ModelType::MLP,
                category: ModelCategory::Basic,
                min_samples: 50,
                supports_multivariate: true,
                supports_static_features: true,
                supports_probabilistic: false,
                typical_memory_mb: 1.0,
                typical_training_time: TrainingTime::Fast,
                interpretability_score: 0.3,
            },
            ModelInfo {
                name: "Decomposition Linear".to_string(),
                model_type: ModelType::DLinear,
                category: ModelCategory::Basic,
                min_samples: 30,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 0.5,
                typical_training_time: TrainingTime::Fast,
                interpretability_score: 0.8,
            },
            ModelInfo {
                name: "Normalization Linear".to_string(),
                model_type: ModelType::NLinear,
                category: ModelCategory::Basic,
                min_samples: 30,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 0.5,
                typical_training_time: TrainingTime::Fast,
                interpretability_score: 0.8,
            },
            ModelInfo {
                name: "MLP Multivariate".to_string(),
                model_type: ModelType::MLPMultivariate,
                category: ModelCategory::Basic,
                min_samples: 50,
                supports_multivariate: true,
                supports_static_features: true,
                supports_probabilistic: false,
                typical_memory_mb: 1.5,
                typical_training_time: TrainingTime::Fast,
                interpretability_score: 0.3,
            },
            // Recurrent Models
            ModelInfo {
                name: "Recurrent Neural Network".to_string(),
                model_type: ModelType::RNN,
                category: ModelCategory::Recurrent,
                min_samples: 100,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 4.0,
                typical_training_time: TrainingTime::Medium,
                interpretability_score: 0.2,
            },
            ModelInfo {
                name: "Long Short-Term Memory".to_string(),
                model_type: ModelType::LSTM,
                category: ModelCategory::Recurrent,
                min_samples: 100,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 5.0,
                typical_training_time: TrainingTime::Medium,
                interpretability_score: 0.2,
            },
            ModelInfo {
                name: "Gated Recurrent Unit".to_string(),
                model_type: ModelType::GRU,
                category: ModelCategory::Recurrent,
                min_samples: 100,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 4.0,
                typical_training_time: TrainingTime::Medium,
                interpretability_score: 0.2,
            },
            // Advanced Models
            ModelInfo {
                name: "Neural Basis Expansion Analysis Extended".to_string(),
                model_type: ModelType::NBEATSx,
                category: ModelCategory::Advanced,
                min_samples: 200,
                supports_multivariate: false,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 12.0,
                typical_training_time: TrainingTime::Slow,
                interpretability_score: 0.6,
            },
            ModelInfo {
                name: "Time Series Dense Encoder".to_string(),
                model_type: ModelType::TiDE,
                category: ModelCategory::Advanced,
                min_samples: 200,
                supports_multivariate: true,
                supports_static_features: true,
                supports_probabilistic: false,
                typical_memory_mb: 10.0,
                typical_training_time: TrainingTime::Medium,
                interpretability_score: 0.5,
            },
            ModelInfo {
                name: "Neural Basis Expansion Analysis".to_string(),
                model_type: ModelType::NBEATS,
                category: ModelCategory::Advanced,
                min_samples: 200,
                supports_multivariate: false,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 10.0,
                typical_training_time: TrainingTime::Slow,
                interpretability_score: 0.6,
            },
            ModelInfo {
                name: "Neural Hierarchical Interpolation".to_string(),
                model_type: ModelType::NHITS,
                category: ModelCategory::Advanced,
                min_samples: 200,
                supports_multivariate: true,
                supports_static_features: true,
                supports_probabilistic: false,
                typical_memory_mb: 8.0,
                typical_training_time: TrainingTime::Medium,
                interpretability_score: 0.7,
            },
            // Transformer Models
            ModelInfo {
                name: "Temporal Fusion Transformer".to_string(),
                model_type: ModelType::TFT,
                category: ModelCategory::Transformer,
                min_samples: 500,
                supports_multivariate: true,
                supports_static_features: true,
                supports_probabilistic: true,
                typical_memory_mb: 20.0,
                typical_training_time: TrainingTime::VerySlow,
                interpretability_score: 0.8,
            },
            ModelInfo {
                name: "Informer".to_string(),
                model_type: ModelType::Informer,
                category: ModelCategory::Transformer,
                min_samples: 500,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 15.0,
                typical_training_time: TrainingTime::Slow,
                interpretability_score: 0.4,
            },
            ModelInfo {
                name: "AutoFormer".to_string(),
                model_type: ModelType::AutoFormer,
                category: ModelCategory::Transformer,
                min_samples: 500,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 18.0,
                typical_training_time: TrainingTime::Slow,
                interpretability_score: 0.4,
            },
            ModelInfo {
                name: "FedFormer".to_string(),
                model_type: ModelType::FedFormer,
                category: ModelCategory::Transformer,
                min_samples: 500,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 16.0,
                typical_training_time: TrainingTime::Slow,
                interpretability_score: 0.4,
            },
            ModelInfo {
                name: "PatchTST".to_string(),
                model_type: ModelType::PatchTST,
                category: ModelCategory::Transformer,
                min_samples: 400,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 14.0,
                typical_training_time: TrainingTime::Medium,
                interpretability_score: 0.3,
            },
            ModelInfo {
                name: "iTransformer".to_string(),
                model_type: ModelType::ITransformer,
                category: ModelCategory::Transformer,
                min_samples: 400,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 15.0,
                typical_training_time: TrainingTime::Medium,
                interpretability_score: 0.3,
            },
            // Specialized Models
            ModelInfo {
                name: "Deep Neural Point Temporal Smoothing".to_string(),
                model_type: ModelType::DeepNPTS,
                category: ModelCategory::Specialized,
                min_samples: 300,
                supports_multivariate: true,
                supports_static_features: true,
                supports_probabilistic: true,
                typical_memory_mb: 9.0,
                typical_training_time: TrainingTime::Medium,
                interpretability_score: 0.4,
            },
            ModelInfo {
                name: "Bidirectional Temporal Convolutional Network".to_string(),
                model_type: ModelType::BiTCN,
                category: ModelCategory::Specialized,
                min_samples: 200,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 7.0,
                typical_training_time: TrainingTime::Medium,
                interpretability_score: 0.3,
            },
            ModelInfo {
                name: "TimesNet".to_string(),
                model_type: ModelType::TimesNet,
                category: ModelCategory::Specialized,
                min_samples: 300,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 12.0,
                typical_training_time: TrainingTime::Slow,
                interpretability_score: 0.4,
            },
            ModelInfo {
                name: "Spectral Temporal Graph Neural Network".to_string(),
                model_type: ModelType::StemGNN,
                category: ModelCategory::Specialized,
                min_samples: 400,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 15.0,
                typical_training_time: TrainingTime::Slow,
                interpretability_score: 0.3,
            },
            ModelInfo {
                name: "Time Series Mixer".to_string(),
                model_type: ModelType::TSMixer,
                category: ModelCategory::Specialized,
                min_samples: 200,
                supports_multivariate: true,
                supports_static_features: true,
                supports_probabilistic: false,
                typical_memory_mb: 8.0,
                typical_training_time: TrainingTime::Medium,
                interpretability_score: 0.5,
            },
            ModelInfo {
                name: "Time Series Mixer Extended".to_string(),
                model_type: ModelType::TSMixerx,
                category: ModelCategory::Specialized,
                min_samples: 200,
                supports_multivariate: true,
                supports_static_features: true,
                supports_probabilistic: false,
                typical_memory_mb: 10.0,
                typical_training_time: TrainingTime::Medium,
                interpretability_score: 0.5,
            },
            ModelInfo {
                name: "Patch Mixer".to_string(),
                model_type: ModelType::PatchMixer,
                category: ModelCategory::Specialized,
                min_samples: 200,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 6.0,
                typical_training_time: TrainingTime::Fast,
                interpretability_score: 0.4,
            },
            ModelInfo {
                name: "Segment Recurrent Neural Network".to_string(),
                model_type: ModelType::SegRNN,
                category: ModelCategory::Specialized,
                min_samples: 150,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 5.0,
                typical_training_time: TrainingTime::Fast,
                interpretability_score: 0.6,
            },
            ModelInfo {
                name: "Dish Time Series".to_string(),
                model_type: ModelType::DishTS,
                category: ModelCategory::Specialized,
                min_samples: 100,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 4.0,
                typical_training_time: TrainingTime::Fast,
                interpretability_score: 0.7,
            },
            ModelInfo {
                name: "Deep AutoRegressive".to_string(),
                model_type: ModelType::DeepAR,
                category: ModelCategory::Specialized,
                min_samples: 300,
                supports_multivariate: true,
                supports_static_features: true,
                supports_probabilistic: true,
                typical_memory_mb: 8.0,
                typical_training_time: TrainingTime::Medium,
                interpretability_score: 0.5,
            },
            ModelInfo {
                name: "Temporal Convolutional Network".to_string(),
                model_type: ModelType::TCN,
                category: ModelCategory::Specialized,
                min_samples: 200,
                supports_multivariate: true,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb: 6.0,
                typical_training_time: TrainingTime::Medium,
                interpretability_score: 0.3,
            },
        ]
    }

    /// Get model information by type
    pub fn get_model_info(model_type: ModelType) -> Option<ModelInfo> {
        Self::get_available_models()
            .into_iter()
            .find(|info| info.model_type == model_type)
    }

    /// Get model requirements
    pub fn get_model_requirements(model_type: ModelType) -> ModelRequirements {
        match model_type {
            ModelType::LSTM => ModelRequirements {
                required_params: vec![
                    "hidden_size".to_string(),
                    "num_layers".to_string(),
                    "horizon".to_string(),
                ],
                optional_params: vec![
                    "dropout".to_string(),
                    "bidirectional".to_string(),
                    "input_size".to_string(),
                ],
                min_samples: 100,
                max_horizon: Some(100),
                supports_missing_values: false,
            },
            ModelType::NBEATS => ModelRequirements {
                required_params: vec!["horizon".to_string(), "input_size".to_string()],
                optional_params: vec![
                    "stacks".to_string(),
                    "layers".to_string(),
                    "layer_widths".to_string(),
                ],
                min_samples: 200,
                max_horizon: None,
                supports_missing_values: false,
            },
            ModelType::TFT => ModelRequirements {
                required_params: vec!["horizon".to_string(), "hidden_size".to_string()],
                optional_params: vec![
                    "num_attention_heads".to_string(),
                    "dropout".to_string(),
                    "input_size".to_string(),
                ],
                min_samples: 500,
                max_horizon: Some(200),
                supports_missing_values: true,
            },
            ModelType::DeepAR => ModelRequirements {
                required_params: vec!["horizon".to_string()],
                optional_params: vec![
                    "cell_type".to_string(),
                    "hidden_size".to_string(),
                    "num_layers".to_string(),
                    "dropout".to_string(),
                ],
                min_samples: 300,
                max_horizon: None,
                supports_missing_values: true,
            },
            _ => ModelRequirements {
                required_params: vec!["horizon".to_string()],
                optional_params: vec![],
                min_samples: 50,
                max_horizon: None,
                supports_missing_values: false,
            },
        }
    }
}

/// Model requirements specification
#[derive(Clone, Debug)]
pub struct ModelRequirements {
    pub required_params: Vec<String>,
    pub optional_params: Vec<String>,
    pub min_samples: usize,
    pub max_horizon: Option<usize>,
    pub supports_missing_values: bool,
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ModelType::MLP => "MLP",
            ModelType::DLinear => "DLinear",
            ModelType::NLinear => "NLinear",
            ModelType::MLPMultivariate => "MLPMultivariate",
            ModelType::RNN => "RNN",
            ModelType::LSTM => "LSTM",
            ModelType::GRU => "GRU",
            ModelType::NBEATS => "NBEATS",
            ModelType::NBEATSx => "NBEATSx",
            ModelType::NHITS => "NHITS",
            ModelType::TiDE => "TiDE",
            ModelType::TFT => "TFT",
            ModelType::Informer => "Informer",
            ModelType::AutoFormer => "AutoFormer",
            ModelType::FedFormer => "FedFormer",
            ModelType::PatchTST => "PatchTST",
            ModelType::ITransformer => "iTransformer",
            ModelType::DeepAR => "DeepAR",
            ModelType::DeepNPTS => "DeepNPTS",
            ModelType::TCN => "TCN",
            ModelType::BiTCN => "BiTCN",
            ModelType::TimesNet => "TimesNet",
            ModelType::StemGNN => "StemGNN",
            ModelType::TSMixer => "TSMixer",
            ModelType::TSMixerx => "TSMixerx",
            ModelType::PatchMixer => "PatchMixer",
            ModelType::SegRNN => "SegRNN",
            ModelType::DishTS => "DishTS",
        };
        write!(f, "{}", name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_factory() {
        let models = ModelFactory::get_available_models();
        assert!(!models.is_empty());

        // Check that we have models from each category
        let categories: Vec<ModelCategory> = models.iter().map(|m| m.category).collect();

        assert!(categories.contains(&ModelCategory::Basic));
        assert!(categories.contains(&ModelCategory::Recurrent));
        assert!(categories.contains(&ModelCategory::Advanced));
        assert!(categories.contains(&ModelCategory::Transformer));
        assert!(categories.contains(&ModelCategory::Specialized));
    }

    #[test]
    fn test_model_requirements() {
        let lstm_reqs = ModelFactory::get_model_requirements(ModelType::LSTM);
        assert!(lstm_reqs
            .required_params
            .contains(&"hidden_size".to_string()));
        assert!(lstm_reqs.required_params.contains(&"horizon".to_string()));
        assert_eq!(lstm_reqs.min_samples, 100);
    }
}
