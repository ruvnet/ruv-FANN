#![doc = include_str!("../README.md")]

// Core error handling
pub use errors::{NeuroDivergentError, NeuroDivergentResult};

// Core traits and foundations
pub use foundation::{BaseModel, NetworkAdapter, ModelConfig};
pub use foundation::{TimeSeriesInput, ForecastOutput, ValidationConfig};

// Data structures
pub use data::{TimeSeriesDataFrame, ForecastDataFrame, TimeSeriesSchema};

// Main forecasting interface
pub use forecasting::NeuralForecast;

// Model configurations
pub use config::{LSTMConfig, RNNConfig, GRUConfig};
pub use config::{TrainingConfig, PredictionConfig, CrossValidationConfig};

// Re-export commonly used types from ruv-FANN
pub use ruv_fann::{ActivationFunction, Network, Layer, Neuron};
pub use num_traits::Float;

// Modules
pub mod errors;
pub mod foundation;
pub mod data;
pub mod forecasting;
pub mod config;
mod rnn_config_map;

// Model implementations
pub mod models {
    //! Neural forecasting model implementations
    pub use crate::recurrent::{RNN, LSTM, GRU};
}

// Model categories
pub mod recurrent;

// Core components
pub mod activations;
pub mod layers;

// Utilities
pub mod utils;

// The testing feature reserves optional assertion helpers; no test_utils module is shipped.
