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

pub mod neural_bridge;
pub mod neural_models;

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
    pub fn create_model(
        model_type: ModelType,
        input_size: usize,
        output_size: usize,
    ) -> Result<Box<dyn ForecastModel>, String> {
        neural_models::create_model(model_type, input_size, output_size)
    }

    /// Create a model with default parameters for time series forecasting
    pub fn create_default_model(
        model_type: ModelType,
        sequence_length: usize,
        horizon: usize,
    ) -> Result<Box<dyn ForecastModel>, String> {
        Self::create_model(model_type, sequence_length, horizon)
    }

    /// Models implemented by the ruv-fann bridge.
    ///
    /// Memory figures are planning hints, not measured allocation limits. The
    /// legacy DLinear/MLPMultivariate names identify a linear baseline and a
    /// wider MLP; decomposition and multivariate feature handling are not implemented.
    pub fn get_available_models() -> Vec<ModelInfo> {
        [
            ("Multi-Layer Perceptron", ModelType::MLP, 1.0, 0.3),
            (
                "Linear baseline (legacy DLinear)",
                ModelType::DLinear,
                0.5,
                0.8,
            ),
            (
                "Wide MLP (legacy MLPMultivariate)",
                ModelType::MLPMultivariate,
                1.5,
                0.3,
            ),
        ]
        .into_iter()
        .map(
            |(name, model_type, typical_memory_mb, interpretability_score)| ModelInfo {
                name: name.into(),
                model_type,
                category: ModelCategory::Basic,
                min_samples: 1,
                supports_multivariate: false,
                supports_static_features: false,
                supports_probabilistic: false,
                typical_memory_mb,
                typical_training_time: TrainingTime::Fast,
                interpretability_score,
            },
        )
        .collect()
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

        assert_eq!(models.len(), 3);
        for info in models {
            let model = ModelFactory::create_model(info.model_type, 3, 1).unwrap();
            assert_eq!(model.model_type(), info.model_type);
            assert!(!info.supports_static_features);
            assert!(!info.supports_probabilistic);
        }
        assert!(ModelFactory::create_model(ModelType::LSTM, 3, 1).is_err());
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
