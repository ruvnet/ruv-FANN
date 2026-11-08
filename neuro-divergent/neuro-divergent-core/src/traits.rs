//! Core traits that define the forecasting model interface.
//!
//! This module contains the fundamental traits that all forecasting models must implement,
//! providing a consistent and type-safe interface for neural forecasting operations.

use std::collections::HashMap;
use std::marker::PhantomData;

use chrono::{DateTime, Utc};
use num_traits::Float;
use serde::{Deserialize, Serialize};

use crate::{
    data::{TimeSeriesDataFrame, TimeSeriesDataset},
    error::{NeuroDivergentError, NeuroDivergentResult},
};

/// Core trait that all forecasting models must implement.
///
/// This trait provides a standardized interface for all neural forecasting models,
/// ensuring consistent behavior across different model architectures.
///
/// # Type Parameters
///
/// * `T` - The floating-point type used for calculations (f32 or f64)
///
/// # Examples
///
/// ```rust
/// use neuro_divergent_core::traits::{BaseModel, ForecastResult};
/// use neuro_divergent_core::data::TimeSeriesDataset;
/// use neuro_divergent_core::error::NeuroDivergentResult;
///
/// fn checked_forecast<M: BaseModel<f64>>(
///     model: &M,
///     data: &TimeSeriesDataset<f64>,
/// ) -> NeuroDivergentResult<ForecastResult<f64>> {
///     model.validate_data(data)?;
///     model.predict(data)
/// }
/// ```
pub trait BaseModel<T: Float + Send + Sync + 'static>: Send + Sync {
    /// Configuration type for this model
    type Config: ModelConfig<T>;
    
    /// State type for model serialization and restoration
    type State: ModelState<T>;

    /// Create a new model instance with the given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - The model configuration
    ///
    /// # Returns
    ///
    /// A new model instance or an error if the configuration is invalid
    fn new(config: Self::Config) -> NeuroDivergentResult<Self>
    where
        Self: Sized;

    /// Fit the model to training data
    ///
    /// # Arguments
    ///
    /// * `data` - The training dataset
    ///
    /// # Returns
    ///
    /// Ok(()) if training succeeds, or an error describing the failure
    fn fit(&mut self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<()>;

    /// Generate forecasts for the given dataset
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to generate forecasts for
    ///
    /// # Returns
    ///
    /// Forecast results or an error if prediction fails
    fn predict(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<ForecastResult<T>>;

    /// Perform cross-validation on the model
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset for cross-validation
    /// * `config` - Cross-validation configuration
    ///
    /// # Returns
    ///
    /// Cross-validation results or an error if validation fails
    fn cross_validation(
        &mut self,
        data: &TimeSeriesDataset<T>,
        config: CrossValidationConfig,
    ) -> NeuroDivergentResult<CrossValidationResult<T>>;

    /// Get the model configuration
    fn config(&self) -> &Self::Config;

    /// Get the internal model state for serialization
    fn state(&self) -> &Self::State;

    /// Restore model from saved state
    ///
    /// # Arguments
    ///
    /// * `state` - The state to restore from
    ///
    /// # Returns
    ///
    /// Ok(()) if restoration succeeds, or an error if the state is invalid
    fn restore_state(&mut self, state: Self::State) -> NeuroDivergentResult<()>;

    /// Reset model to initial state
    ///
    /// This clears all learned parameters and resets the model to its
    /// initial untrained state.
    fn reset(&mut self) -> NeuroDivergentResult<()>;

    /// Get model metadata and information
    fn metadata(&self) -> ModelMetadata;

    /// Validate input data compatibility with this model
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to validate
    ///
    /// # Returns
    ///
    /// Ok(()) if data is compatible, or an error describing the incompatibility
    fn validate_data(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<()>;

    /// Check if the model is trained and ready for prediction
    fn is_trained(&self) -> bool;

    /// Get the number of parameters in the model
    fn parameter_count(&self) -> usize;

    /// Get model training statistics (if available)
    fn training_stats(&self) -> Option<TrainingStatistics<T>>;
}

/// Configuration trait for model parameters
///
/// This trait defines the interface for model configuration objects,
/// ensuring consistent validation and parameter access across different models.
pub trait ModelConfig<T: Float + Send + Sync + 'static>: Clone + Send + Sync + 'static {
    /// Validate configuration parameters
    ///
    /// # Returns
    ///
    /// Ok(()) if configuration is valid, or an error describing the issue
    fn validate(&self) -> NeuroDivergentResult<()>;

    /// Get the forecast horizon (number of steps to predict)
    fn horizon(&self) -> usize;

    /// Get the input window size (number of historical steps to use)
    fn input_size(&self) -> usize;

    /// Get the output size (usually equal to horizon for single-target forecasting)
    fn output_size(&self) -> usize;

    /// Get exogenous variable configuration
    fn exogenous_config(&self) -> &ExogenousConfig;

    /// Get the model name/type identifier
    fn model_type(&self) -> &str;

    /// Convert configuration to a generic parameter map
    fn to_parameters(&self) -> HashMap<String, ConfigParameter<T>>;

    /// Create configuration from a generic parameter map
    fn from_parameters(params: HashMap<String, ConfigParameter<T>>) -> NeuroDivergentResult<Self>
    where
        Self: Sized;

    /// Create a builder for this configuration type
    fn builder() -> impl ConfigBuilder<Self, T>
    where
        Self: Sized;
}

/// Model state trait for serialization and restoration
///
/// This trait defines the interface for model state objects that can be
/// serialized and restored for model persistence.
pub trait ModelState<T: Float + Send + Sync + 'static>: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> {
    /// Get the model type this state belongs to
    fn model_type(&self) -> &str;

    /// Get the state version for backward compatibility
    fn version(&self) -> u32;

    /// Check if this state is compatible with the given model configuration
    fn is_compatible<C: ModelConfig<T>>(&self, config: &C) -> bool;

    /// Get the training completion timestamp
    fn trained_at(&self) -> Option<DateTime<Utc>>;

    /// Get training metrics associated with this state
    fn training_metrics(&self) -> Option<&TrainingStatistics<T>>;
}

/// High-level forecasting engine trait for batch operations
///
/// This trait provides advanced forecasting capabilities including
/// batch prediction, probabilistic forecasting, and interval estimation.
pub trait ForecastingEngine<T: Float + Send + Sync + 'static>: Send + Sync {
    /// Batch prediction for multiple time series
    ///
    /// # Arguments
    ///
    /// * `datasets` - Vector of datasets to predict
    ///
    /// # Returns
    ///
    /// Vector of forecast results, one for each input dataset
    fn batch_predict(
        &self,
        datasets: &[TimeSeriesDataset<T>],
    ) -> NeuroDivergentResult<Vec<ForecastResult<T>>>;

    /// Probabilistic forecasting with prediction intervals
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to predict
    /// * `confidence_levels` - Confidence levels for intervals (e.g., [0.8, 0.9, 0.95])
    ///
    /// # Returns
    ///
    /// Interval forecasts with specified confidence levels
    fn predict_intervals(
        &self,
        data: &TimeSeriesDataset<T>,
        confidence_levels: &[f64],
    ) -> NeuroDivergentResult<IntervalForecast<T>>;

    /// Quantile forecasting
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to predict
    /// * `quantiles` - Quantile levels (e.g., [0.1, 0.5, 0.9])
    ///
    /// # Returns
    ///
    /// Quantile forecasts for the specified levels
    fn predict_quantiles(
        &self,
        data: &TimeSeriesDataset<T>,
        quantiles: &[f64],
    ) -> NeuroDivergentResult<QuantileForecast<T>>;

    /// Multi-horizon forecasting with different horizons
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to predict
    /// * `horizons` - Different forecast horizons to predict
    ///
    /// # Returns
    ///
    /// Forecasts for each specified horizon
    fn predict_multi_horizon(
        &self,
        data: &TimeSeriesDataset<T>,
        horizons: &[usize],
    ) -> NeuroDivergentResult<MultiHorizonForecast<T>>;
}

/// Forecast result containing predictions and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult<T: Float> {
    /// The forecast values
    pub forecasts: Vec<T>,
    /// Future timestamps corresponding to forecasts
    pub timestamps: Vec<DateTime<Utc>>,
    /// Series identifier
    pub series_id: String,
    /// Model name that generated these forecasts
    pub model_name: String,
    /// Forecast generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Optional metadata
    pub metadata: Option<HashMap<String, String>>,
}

/// Interval forecast result with confidence bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalForecast<T: Float> {
    /// Point forecasts
    pub forecasts: Vec<T>,
    /// Lower bounds for each confidence level
    pub lower_bounds: Vec<Vec<T>>,
    /// Upper bounds for each confidence level
    pub upper_bounds: Vec<Vec<T>>,
    /// Confidence levels used
    pub confidence_levels: Vec<f64>,
    /// Future timestamps
    pub timestamps: Vec<DateTime<Utc>>,
    /// Series identifier
    pub series_id: String,
    /// Model name
    pub model_name: String,
    /// Generation timestamp
    pub generated_at: DateTime<Utc>,
}

/// Quantile forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileForecast<T: Float> {
    /// Quantile forecasts for each quantile level
    pub quantile_forecasts: HashMap<String, Vec<T>>, // quantile level as string key
    /// Quantile levels used
    pub quantile_levels: Vec<f64>,
    /// Future timestamps
    pub timestamps: Vec<DateTime<Utc>>,
    /// Series identifier
    pub series_id: String,
    /// Model name
    pub model_name: String,
    /// Generation timestamp
    pub generated_at: DateTime<Utc>,
}

/// Multi-horizon forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHorizonForecast<T: Float> {
    /// Forecasts for each horizon
    pub horizon_forecasts: HashMap<usize, Vec<T>>, // horizon as key
    /// Horizons used
    pub horizons: Vec<usize>,
    /// Future timestamps for each horizon
    pub timestamps: HashMap<usize, Vec<DateTime<Utc>>>,
    /// Series identifier
    pub series_id: String,
    /// Model name
    pub model_name: String,
    /// Generation timestamp
    pub generated_at: DateTime<Utc>,
}

/// Cross-validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    /// Number of cross-validation windows
    pub n_windows: usize,
    /// Forecast horizon for each window
    pub horizon: usize,
    /// Step size between windows (None for non-overlapping)
    pub step_size: Option<usize>,
    /// Size of test set for each fold
    pub test_size: Option<usize>,
    /// Seasonal length for time series-aware splitting
    pub season_length: Option<usize>,
    /// Whether to refit models for each window
    pub refit: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

/// Cross-validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResult<T: Float> {
    /// Cutoff dates for each fold
    pub cutoffs: Vec<DateTime<Utc>>,
    /// Forecasts for each fold
    pub forecasts: Vec<ForecastResult<T>>,
    /// Actual values for each fold
    pub actuals: Vec<Vec<T>>,
    /// Evaluation metrics for each fold
    pub fold_metrics: Vec<HashMap<String, T>>,
    /// Overall aggregated metrics
    pub overall_metrics: HashMap<String, T>,
    /// Model name
    pub model_name: String,
    /// Cross-validation configuration used
    pub config: CrossValidationConfig,
}

/// Model metadata and information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name/type
    pub model_type: String,
    /// Model version
    pub version: String,
    /// Model description
    pub description: String,
    /// Supported features
    pub capabilities: ModelCapabilities,
    /// Parameter count
    pub parameter_count: usize,
    /// Memory requirements (in bytes)
    pub memory_requirements: Option<usize>,
    /// Training time (in seconds)
    pub training_time: Option<f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Model capabilities flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Supports future exogenous variables
    pub supports_future_exogenous: bool,
    /// Supports historical exogenous variables
    pub supports_historical_exogenous: bool,
    /// Supports static exogenous variables
    pub supports_static_exogenous: bool,
    /// Supports multivariate forecasting
    pub supports_multivariate: bool,
    /// Supports probabilistic forecasting
    pub supports_probabilistic: bool,
    /// Supports quantile forecasting
    pub supports_quantile: bool,
    /// Supports recursive prediction
    pub supports_recursive: bool,
    /// Supports parallel training
    pub supports_parallel_training: bool,
    /// Supports online learning
    pub supports_online_learning: bool,
}

/// Training statistics and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatistics<T: Float> {
    /// Number of training epochs completed
    pub epochs_completed: usize,
    /// Training loss history
    pub training_loss: Vec<T>,
    /// Validation loss history (if validation was performed)
    pub validation_loss: Option<Vec<T>>,
    /// Learning rate history
    pub learning_rate: Vec<T>,
    /// Training time per epoch (in seconds)
    pub epoch_times: Vec<f64>,
    /// Total training time (in seconds)
    pub total_training_time: f64,
    /// Best epoch (lowest validation loss)
    pub best_epoch: Option<usize>,
    /// Early stopping information
    pub early_stopped: bool,
    /// Convergence information
    pub converged: bool,
    /// Additional training metrics
    pub metrics: HashMap<String, Vec<T>>,
}

/// Exogenous variable configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExogenousConfig {
    /// Static feature column names
    pub static_features: Vec<String>,
    /// Historical exogenous feature column names
    pub historical_features: Vec<String>,
    /// Future exogenous feature column names
    pub future_features: Vec<String>,
    /// Whether to automatically encode categorical features
    pub auto_encode_categorical: bool,
    /// Maximum cardinality for categorical encoding
    pub max_categorical_cardinality: Option<usize>,
}

/// Configuration parameter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigParameter<T: Float + Send + Sync + 'static> {
    /// Float parameter
    Float(T),
    /// Integer parameter
    Integer(i64),
    /// String parameter
    String(String),
    /// Boolean parameter
    Boolean(bool),
    /// Float vector parameter
    FloatVector(Vec<T>),
    /// Integer vector parameter
    IntegerVector(Vec<i64>),
    /// String vector parameter
    StringVector(Vec<String>),
}

/// Configuration builder trait
pub trait ConfigBuilder<C, T>
where
    C: ModelConfig<T>,
    T: Float + Send + Sync + 'static,
{
    /// Build the configuration
    fn build(self) -> NeuroDivergentResult<C>;
    
    /// Set the forecast horizon
    fn with_horizon(self, horizon: usize) -> Self;
    
    /// Set the input size
    fn with_input_size(self, input_size: usize) -> Self;
    
    /// Set exogenous configuration
    fn with_exogenous_config(self, config: ExogenousConfig) -> Self;
}

impl Default for ExogenousConfig {
    fn default() -> Self {
        Self {
            static_features: Vec::new(),
            historical_features: Vec::new(),
            future_features: Vec::new(),
            auto_encode_categorical: true,
            max_categorical_cardinality: Some(100),
        }
    }
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            supports_future_exogenous: false,
            supports_historical_exogenous: false,
            supports_static_exogenous: false,
            supports_multivariate: false,
            supports_probabilistic: false,
            supports_quantile: false,
            supports_recursive: true,
            supports_parallel_training: false,
            supports_online_learning: false,
        }
    }
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            n_windows: 3,
            horizon: 1,
            step_size: None,
            test_size: None,
            season_length: None,
            refit: true,
            random_seed: None,
        }
    }
}

impl<T: Float> Default for TrainingStatistics<T> {
    fn default() -> Self {
        Self {
            epochs_completed: 0,
            training_loss: Vec::new(),
            validation_loss: None,
            learning_rate: Vec::new(),
            epoch_times: Vec::new(),
            total_training_time: 0.0,
            best_epoch: None,
            early_stopped: false,
            converged: false,
            metrics: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configurations() {
        let exog_config = ExogenousConfig::default();
        assert!(exog_config.static_features.is_empty());
        assert!(exog_config.auto_encode_categorical);

        let capabilities = ModelCapabilities::default();
        assert!(!capabilities.supports_multivariate);
        assert!(capabilities.supports_recursive);

        let cv_config = CrossValidationConfig::default();
        assert_eq!(cv_config.n_windows, 3);
        assert_eq!(cv_config.horizon, 1);
        assert!(cv_config.refit);
    }

    #[test]
    fn test_config_parameter_serialization() {
        let param = ConfigParameter::<f64>::Float(3.14);
        let serialized = serde_json::to_string(&param).unwrap();
        let deserialized: ConfigParameter<f64> = serde_json::from_str(&serialized).unwrap();
        
        match deserialized {
            ConfigParameter::Float(val) => assert!((val - 3.14).abs() < f64::EPSILON),
            _ => panic!("Expected Float parameter"),
        }
    }

    #[test]
    fn test_forecast_result_creation() {
        let result = ForecastResult {
            forecasts: vec![1.0, 2.0, 3.0],
            timestamps: vec![Utc::now(), Utc::now(), Utc::now()],
            series_id: "test_series".to_string(),
            model_name: "test_model".to_string(),
            generated_at: Utc::now(),
            metadata: None,
        };

        assert_eq!(result.forecasts.len(), 3);
        assert_eq!(result.series_id, "test_series");
        assert_eq!(result.model_name, "test_model");
    }
}