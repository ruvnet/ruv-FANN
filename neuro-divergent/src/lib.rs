#![doc = include_str!("../README.md")]

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![allow(clippy::new_without_default)]

// Re-export essential types at the crate root
pub use neural_forecast::{BaseModel, NeuralForecast, NeuralForecastBuilder, PredictionConfig, ValidationConfig};
pub use baseline::NaiveForecaster;
pub use conformal::{CalibrationGroup, ConformalCalibrator, ConformalConfig, ConformalForecastDataFrame, ConformalLimits};
#[cfg(feature = "models")]
pub use model_adapter::ModelAdapter;
pub use config::{Frequency, ScalerType, PredictionIntervals, CrossValidationConfig};
pub use results::{AccuracyMetrics, ForecastDataFrame, CrossValidationDataFrame, TimeSeriesDataFrame, TimeSeriesSchema};
pub use errors::{NeuroDivergentError, NeuroDivergentResult};

// Core modules
pub mod neural_forecast;
pub mod builders;
pub mod config;
pub mod results;
pub mod utils;
pub mod errors;
pub mod baseline;
pub mod conformal;
#[cfg(feature = "models")]
pub mod model_adapter;
mod timeline;
mod forecast;
mod backtest;
mod scaling;
mod persistence;
mod pipeline_config;
mod evaluation;

// Core abstractions
// Re-export from external crates
#[cfg(feature = "models")]
pub use neuro_divergent_models as models;

#[cfg(feature = "registry")]
pub use neuro_divergent_registry as registry;

pub use neuro_divergent_core as core;

// TODO: Re-enable when data and training crates are published
// #[cfg(feature = "data")]
// pub use neuro_divergent_data as data;
//
// #[cfg(feature = "training")]
// pub use neuro_divergent_training as training;

// Prelude for common imports
pub mod prelude {
    //! Common imports for everyday use
    
    pub use crate::{
        NeuralForecast, NeuralForecastBuilder,
        ForecastDataFrame, CrossValidationDataFrame, TimeSeriesDataFrame,
        Frequency, ScalerType, PredictionIntervals,
        NeuroDivergentError, NeuroDivergentResult,
    };
    
    // TODO: Re-enable when all crates are published
    // pub use crate::core::{BaseModel, ModelConfig};
    // pub use crate::models::{LSTM, NBEATS, DeepAR, RNN, Transformer};
    // pub use crate::data::{TimeSeriesDataset, TimeSeriesSchema};
    // pub use crate::training::{TrainingConfig, AccuracyMetrics};
}

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const PACKAGE_NAME: &str = env!("CARGO_PKG_NAME");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Library information
pub mod info {
    //! Library metadata and version information
    
    /// Get the library version
    pub fn version() -> &'static str {
        super::VERSION
    }
    
    /// Get the package name
    pub fn name() -> &'static str {
        super::PACKAGE_NAME
    }
    
    /// Get the library description
    pub fn description() -> &'static str {
        super::DESCRIPTION
    }
    
    /// Whether a GPU forecasting implementation is available (currently false).
    /// The `gpu` Cargo feature is reserved and does not enable GPU execution.
    pub fn has_gpu_support() -> bool {
        false
    }

    /// Whether optional async runtime dependencies were enabled.
    /// Forecasting methods themselves are synchronous.
    #[cfg(feature = "async")]
    pub fn has_async_support() -> bool {
        true
    }
    
    #[cfg(not(feature = "async"))]
    pub fn has_async_support() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version_info() {
        assert!(!info::version().is_empty());
        assert!(!info::name().is_empty());
        assert!(!info::description().is_empty());
    }
    
    #[test]
    fn test_feature_detection() {
        // These should not panic
        assert!(!info::has_gpu_support());
        let _ = info::has_async_support();
    }
}
#[cfg(test)]
mod forecast_contract_tests;
