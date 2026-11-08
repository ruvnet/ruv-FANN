//! Integration layer with ruv-FANN neural networks.
//!
//! This module provides the bridge between neuro-divergent's time series forecasting
//! capabilities and ruv-FANN's neural network infrastructure, enabling seamless
//! integration of existing neural network functionality with forecasting-specific features.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use num_traits::Float;
use serde::{Deserialize, Serialize};

use ruv_fann::{
    ActivationFunction, Network, NetworkBuilder, TrainingData,
};

use ruv_fann::training::TrainingAlgorithm;

use crate::{
    data::{SeriesData, TimeSeriesDataset, TimeSeriesDataFrame},
    error::{ErrorBuilder, NetworkIntegrationError, NeuroDivergentError, NeuroDivergentResult},
    traits::{ForecastResult, TrainingStatistics},
};

/// Adapter for integrating ruv-FANN networks with forecasting models
///
/// This adapter wraps a ruv-FANN Network and provides time series-specific
/// preprocessing and postprocessing capabilities.
pub struct NetworkAdapter<T: Float + Send + Sync + 'static> {
    /// The underlying ruv-FANN network
    network: Network<T>,
    /// Input preprocessing pipeline
    input_preprocessor: Box<dyn InputPreprocessor<T>>,
    /// Output postprocessing pipeline
    output_postprocessor: Box<dyn OutputPostprocessor<T>>,
    /// Activation function mapper
    activation_mapper: ActivationMapper,
    /// Network configuration metadata
    config: NetworkAdapterConfig<T>,
    /// Training state
    training_state: Option<NetworkTrainingState<T>>,
}

/// Configuration for NetworkAdapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAdapterConfig<T: Float + Send + Sync + 'static> {
    /// Input dimension
    pub input_size: usize,
    /// Output dimension
    pub output_size: usize,
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Activation functions per layer
    pub activations: Vec<ActivationFunction>,
    /// Whether to use bias terms
    pub use_bias: bool,
    /// Dropout rates per layer (if supported)
    pub dropout_rates: Option<Vec<T>>,
    /// Network type identifier
    pub network_type: String,
}

/// Training state for NetworkAdapter
#[derive(Debug, Clone)]
pub struct NetworkTrainingState<T: Float + Send + Sync + 'static> {
    /// Current epoch
    pub current_epoch: usize,
    /// Training loss history
    pub loss_history: Vec<T>,
    /// Validation loss history
    pub validation_loss: Option<Vec<T>>,
    /// Learning rate history
    pub learning_rate_history: Vec<T>,
    /// Best validation loss
    pub best_validation_loss: Option<T>,
    /// Best epoch
    pub best_epoch: Option<usize>,
    /// Training start time
    pub training_start: chrono::DateTime<chrono::Utc>,
    /// Last update time
    pub last_update: chrono::DateTime<chrono::Utc>,
    /// Training completed flag
    pub training_completed: bool,
}

/// Input preprocessing trait for time series data
pub trait InputPreprocessor<T: Float + Send + Sync + 'static>: Send + Sync {
    /// Process input time series data for network consumption
    fn process(&self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<Vec<T>>;
    
    /// Get the expected output size after preprocessing
    fn output_size(&self) -> usize;
    
    /// Get preprocessing configuration
    fn config(&self) -> &PreprocessorConfig<T>;
    
    /// Reset internal state (if any)
    fn reset(&mut self);
}

/// Output postprocessing trait for network outputs
pub trait OutputPostprocessor<T: Float + Send + Sync + 'static>: Send + Sync {
    /// Process network output to generate forecasts
    fn process(&self, output: &[T], context: &PostprocessorContext<T>) -> NeuroDivergentResult<ForecastResult<T>>;
    
    /// Get postprocessing configuration
    fn config(&self) -> &PostprocessorConfig;
    
    /// Reset internal state (if any)
    fn reset(&mut self);
}

/// Input data structure for time series processing
#[derive(Debug, Clone)]
pub struct TimeSeriesInput<T: Float + Send + Sync + 'static> {
    /// Historical target values
    pub target_history: Vec<T>,
    /// Historical exogenous variables (if any)
    pub exogenous_history: Option<Array2<T>>,
    /// Future exogenous variables (if any)
    pub exogenous_future: Option<Array2<T>>,
    /// Static features
    pub static_features: Option<Vec<T>>,
    /// Timestamps
    pub timestamps: Vec<chrono::DateTime<chrono::Utc>>,
    /// Series identifier
    pub series_id: String,
    /// Input context metadata
    pub metadata: HashMap<String, String>,
}

/// Context for output postprocessing
#[derive(Debug, Clone)]
pub struct PostprocessorContext<T: Float + Send + Sync + 'static> {
    /// Original input data
    pub input: TimeSeriesInput<T>,
    /// Forecast horizon
    pub horizon: usize,
    /// Future timestamps
    pub future_timestamps: Vec<chrono::DateTime<chrono::Utc>>,
    /// Model name
    pub model_name: String,
    /// Additional context
    pub metadata: HashMap<String, String>,
}

/// Configuration for input preprocessors
#[derive(Debug, Clone)]
pub struct PreprocessorConfig<T: Float + Send + Sync + 'static> {
    /// Scaling method
    pub scaling_method: ScalingMethod,
    /// Window size for input
    pub window_size: usize,
    /// Whether to include trend features
    pub include_trend: bool,
    /// Whether to include seasonal features
    pub include_seasonal: bool,
    /// Lag features to include
    pub lag_features: Vec<usize>,
    /// Rolling window features
    pub rolling_features: Option<RollingFeatureConfig>,
    /// Normalization parameters
    pub normalization: Option<NormalizationParams<T>>,
}

/// Configuration for output postprocessors
#[derive(Debug, Clone)]
pub struct PostprocessorConfig {
    /// Forecast horizon
    pub horizon: usize,
    /// Whether to denormalize outputs
    pub denormalize: bool,
    /// Confidence interval settings
    pub confidence_intervals: Option<Vec<f64>>,
    /// Whether to apply trend adjustment
    pub trend_adjustment: bool,
    /// Seasonal adjustment settings
    pub seasonal_adjustment: Option<SeasonalAdjustmentConfig>,
}

/// Scaling methods for preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingMethod {
    /// No scaling
    None,
    /// Standard scaling (z-score)
    Standard,
    /// Min-max scaling
    MinMax,
    /// Robust scaling
    Robust,
    /// Custom scaling with parameters
    Custom(CustomScalingParams),
}

/// Custom scaling parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomScalingParams {
    /// Scaling factor
    pub scale: f64,
    /// Offset value
    pub offset: f64,
}

/// Rolling feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingFeatureConfig {
    /// Window sizes for rolling features
    pub window_sizes: Vec<usize>,
    /// Statistics to compute
    pub statistics: Vec<RollingStatistic>,
}

/// Rolling statistics to compute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollingStatistic {
    /// Mean
    Mean,
    /// Standard deviation
    Std,
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// Median
    Median,
}

/// Normalization parameters
#[derive(Debug, Clone)]
pub struct NormalizationParams<T: Float + Send + Sync + 'static> {
    /// Mean values per feature
    pub means: Vec<T>,
    /// Standard deviation values per feature
    pub stds: Vec<T>,
    /// Minimum values per feature (for MinMax scaling)
    pub mins: Option<Vec<T>>,
    /// Maximum values per feature (for MinMax scaling)
    pub maxs: Option<Vec<T>>,
}

/// Seasonal adjustment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalAdjustmentConfig {
    /// Seasonal period
    pub period: usize,
    /// Adjustment method
    pub method: SeasonalAdjustmentMethod,
}

/// Seasonal adjustment methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeasonalAdjustmentMethod {
    /// Additive seasonal adjustment
    Additive,
    /// Multiplicative seasonal adjustment
    Multiplicative,
    /// No adjustment
    None,
}

/// Activation function mapper for compatibility
pub struct ActivationMapper {
    /// Mapping from string names to ruv-FANN activation functions
    function_map: HashMap<String, ActivationFunction>,
}

/// Bridge between neuro-divergent training and ruv-FANN training algorithms
pub struct TrainingBridge<T: Float + Send + Sync + 'static> {
    /// ruv-FANN training algorithm
    algorithm: Box<dyn TrainingAlgorithm<T>>,
    /// Training configuration
    config: TrainingBridgeConfig<T>,
    /// Training state tracking
    state: Arc<Mutex<TrainingBridgeState<T>>>,
}

/// Configuration for training bridge
#[derive(Debug, Clone)]
pub struct TrainingBridgeConfig<T: Float + Send + Sync + 'static> {
    /// Maximum number of epochs
    pub max_epochs: usize,
    /// Learning rate
    pub learning_rate: T,
    /// Target error threshold
    pub target_error: T,
    /// Validation split ratio
    pub validation_split: Option<T>,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig<T>>,
    /// Batch size for training
    pub batch_size: Option<usize>,
    /// Shuffle data between epochs
    pub shuffle: bool,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig<T: Float + Send + Sync + 'static> {
    /// Patience (epochs to wait)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_delta: T,
    /// Metric to monitor ('loss' or 'validation_loss')
    pub monitor: String,
    /// Mode ('min' or 'max')
    pub mode: String,
}

/// Training bridge state
#[derive(Debug, Clone)]
pub struct TrainingBridgeState<T: Float + Send + Sync + 'static> {
    /// Current epoch
    pub epoch: usize,
    /// Training metrics
    pub metrics: TrainingStatistics<T>,
    /// Early stopping state
    pub early_stopping_state: Option<EarlyStoppingState>,
    /// Training active flag
    pub training_active: bool,
}

/// Early stopping state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingState {
    /// Current patience counter
    pub patience_counter: usize,
    /// Best metric value seen
    pub best_metric: f64,
    /// Should stop training
    pub should_stop: bool,
}

/// Network output wrapper
#[derive(Debug, Clone)]
pub struct NetworkOutput<T: Float + Send + Sync + 'static> {
    /// Raw network outputs
    pub raw_outputs: Vec<T>,
    /// Processed outputs
    pub processed_outputs: Option<Vec<T>>,
    /// Confidence scores (if available)
    pub confidence_scores: Option<Vec<T>>,
    /// Output metadata
    pub metadata: HashMap<String, String>,
}

impl<T: Float + Send + Sync + 'static + 'static> NetworkAdapter<T> {
    /// Create adapter from ruv-FANN network
    pub fn from_network(network: Network<T>) -> Self {
        let config = NetworkAdapterConfig {
            input_size: network.num_inputs(),
            output_size: network.num_outputs(),
            hidden_layers: network.layers.iter().skip(1).take(network.layers.len().saturating_sub(2)).map(|layer| layer.num_regular_neurons()).collect(),
            activations: network.layers.iter().skip(1).filter_map(|layer| layer.neurons.iter().find(|neuron| !neuron.is_bias).map(|neuron| neuron.activation_function)).collect(),
            use_bias: network.layers.iter().any(|layer| layer.neurons.iter().any(|neuron| neuron.is_bias)),
            dropout_rates: None,
            network_type: "feedforward".to_string(),
        };

        Self {
            network,
            input_preprocessor: Box::new(DefaultInputPreprocessor::<T>::new()),
            output_postprocessor: Box::new(DefaultOutputPostprocessor::<T>::new()),
            activation_mapper: ActivationMapper::new(),
            config,
            training_state: None,
        }
    }

    /// Configure input preprocessing
    pub fn with_input_processor(
        mut self,
        processor: Box<dyn InputPreprocessor<T>>,
    ) -> Self {
        self.input_preprocessor = processor;
        self
    }

    /// Configure output postprocessing
    pub fn with_output_processor(
        mut self,
        processor: Box<dyn OutputPostprocessor<T>>,
    ) -> Self {
        self.output_postprocessor = processor;
        self
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<NetworkOutput<T>> {
        // Preprocess input
        let processed_input = self.input_preprocessor.process(input)
            .map_err(|e| NetworkIntegrationError::ValidationError {
                message: format!("Input preprocessing failed: {}", e),
                layer: None,
            })?;

        // Ensure input size matches network expectations
        if processed_input.len() != self.config.input_size {
            return Err(NetworkIntegrationError::ArchitectureMismatch {
                expected: format!("input size {}", self.config.input_size),
                found: format!("input size {}", processed_input.len()),
            }.into());
        }

        // Run through network
        let raw_outputs = self.network.run(&processed_input)
            .map_err(|error| NetworkIntegrationError::ValidationError {
                message: format!("Network inference failed: {error}"),
                layer: None,
            })?;

        Ok(NetworkOutput {
            raw_outputs,
            processed_outputs: None,
            confidence_scores: None,
            metadata: HashMap::new(),
        })
    }

    /// Get underlying network for direct access
    pub fn network(&self) -> &Network<T> {
        &self.network
    }

    /// Get mutable reference to underlying network
    pub fn network_mut(&mut self) -> &mut Network<T> {
        &mut self.network
    }

    /// Get adapter configuration
    pub fn config(&self) -> &NetworkAdapterConfig<T> {
        &self.config
    }

    /// Get training state
    pub fn training_state(&self) -> Option<&NetworkTrainingState<T>> {
        self.training_state.as_ref()
    }

    /// Reset adapter state
    pub fn reset(&mut self) {
        self.input_preprocessor.reset();
        self.output_postprocessor.reset();
        self.training_state = None;
    }

}


impl<T: Float + Send + Sync + 'static> TrainingBridge<T> {
    /// Create new training bridge with algorithm
    pub fn new(algorithm: Box<dyn TrainingAlgorithm<T>>) -> Self {
        let config = TrainingBridgeConfig {
            max_epochs: 1000,
            learning_rate: T::from(0.001).unwrap_or_else(|| T::one() / T::from(1000).unwrap()),
            target_error: T::from(0.001).unwrap_or_else(|| T::one() / T::from(1000).unwrap()),
            validation_split: None,
            early_stopping: None,
            batch_size: None,
            shuffle: true,
        };

        let state = Arc::new(Mutex::new(TrainingBridgeState {
            epoch: 0,
            metrics: TrainingStatistics::default(),
            early_stopping_state: None,
            training_active: false,
        }));

        Self {
            algorithm,
            config,
            state,
        }
    }

    /// Train network using ruv-FANN algorithms
    pub fn train_network(
        &mut self,
        network: &mut Network<T>,
        data: &TimeSeriesDataset<T>,
        config: &TrainingBridgeConfig<T>,
    ) -> NeuroDivergentResult<T> {
        if config.max_epochs == 0 || !config.learning_rate.is_finite() || config.learning_rate <= T::zero()
            || !config.target_error.is_finite() || config.target_error < T::zero() {
            return Err(ErrorBuilder::config("Epoch count/rate must be positive and target error finite and nonnegative").build());
        }
        if config.validation_split.is_some() || config.early_stopping.is_some() || config.batch_size.is_some() {
            return Err(ErrorBuilder::config("Use the explicit chronological forecasting backtester for validation; bridge mini-batching/early-stopping is unsupported").build());
        }
        let mut algorithm_state = self.algorithm.save_state();
        let rate = algorithm_state.algorithm_specific.get_mut("learning_rate")
            .ok_or_else(|| ErrorBuilder::config("This bridge requires an algorithm exposing a configurable learning_rate").build())?;
        *rate = vec![config.learning_rate];
        self.algorithm.restore_state(algorithm_state);
        let mut training_data = self.prepare_training_data(data, network.num_inputs(), network.num_outputs())?;
        self.config = config.clone();
        {
            let mut state = self.state.lock().unwrap_or_else(|poison| poison.into_inner());
            state.training_active = true; state.epoch = 0; state.metrics = TrainingStatistics::default();
        }
        let started = std::time::Instant::now();
        let result = (|| {
            let mut final_error = T::infinity();
            for epoch in 0..self.config.max_epochs {
                if self.config.shuffle {
                    use rand::Rng;
                    let mut rng = rand::thread_rng();
                    for index in (1..training_data.inputs.len()).rev() {
                        let other = rng.gen_range(0..=index);
                        training_data.inputs.swap(index, other); training_data.outputs.swap(index, other);
                    }
                }
                let epoch_started = std::time::Instant::now();
                final_error = self.algorithm.train_epoch(network, &training_data)
                    .map_err(|error| NetworkIntegrationError::TrainingAlgorithmError { message: format!("Training epoch {epoch} failed: {error}"), algorithm: Some("ruv-fann".into()) })?;
                if !final_error.is_finite() { return Err(ErrorBuilder::training("Non-finite training loss").build()); }
                let mut state = self.state.lock().unwrap_or_else(|poison| poison.into_inner());
                state.epoch = epoch + 1;
                state.metrics.epochs_completed = epoch + 1;
                state.metrics.training_loss.push(final_error);
                state.metrics.learning_rate.push(self.config.learning_rate);
                state.metrics.epoch_times.push(epoch_started.elapsed().as_secs_f64());
                if final_error <= self.config.target_error { state.metrics.converged = true; break; }
            }
            Ok(final_error)
        })();
        let mut state = self.state.lock().unwrap_or_else(|poison| poison.into_inner());
        state.training_active = false;
        state.metrics.total_training_time = started.elapsed().as_secs_f64();
        result
    }

    /// Prepare time series data for ruv-FANN training
    fn prepare_training_data(&self, dataset: &TimeSeriesDataset<T>, window_size: usize, horizon: usize) -> NeuroDivergentResult<TrainingData<T>> {
        let needed = window_size.checked_add(horizon).filter(|_| window_size > 0 && horizon > 0)
            .ok_or_else(|| ErrorBuilder::config("Network input/output dimensions must be positive and bounded").build())?;
        if !dataset.schema.static_features.is_empty() || !dataset.schema.historical_exogenous.is_empty() || !dataset.schema.future_exogenous.is_empty() {
            return Err(ErrorBuilder::config("The bridge maps univariate target windows; exogenous/static mapping is unsupported").build());
        }
        let mut inputs = Vec::new(); let mut outputs = Vec::new();
        for (id, series) in &dataset.series_data {
            if series.target_values.len() < needed || series.target_values.iter().any(|value| !value.is_finite()) {
                return Err(ErrorBuilder::data(format!("Series '{id}' requires at least {needed} finite observations")).build());
            }
            for index in 0..=series.target_values.len() - needed {
                inputs.push(series.target_values[index..index + window_size].to_vec());
                outputs.push(series.target_values[index + window_size..index + needed].to_vec());
            }
        }
        if inputs.is_empty() { return Err(ErrorBuilder::data("No training windows").build()); }
        Ok(TrainingData { inputs, outputs })
    }

    /// Setup cascade correlation training
    pub fn setup_cascade_training(
        &mut self,
        _config: CascadeConfig<T>,
    ) -> NeuroDivergentResult<()> {
        Err(ErrorBuilder::config("Cascade-correlation training is not implemented by this bridge").build())
    }

    /// Get training state
    pub fn state(&self) -> Arc<Mutex<TrainingBridgeState<T>>> {
        Arc::clone(&self.state)
    }

    /// Check if training should continue
    pub fn should_continue_training(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.training_active && state.epoch < self.config.max_epochs
    }
}

/// Cascade correlation training configuration
#[derive(Debug, Clone)]
pub struct CascadeConfig<T: Float + Send + Sync + 'static> {
    /// Maximum number of candidate neurons
    pub max_candidates: usize,
    /// Correlation threshold
    pub correlation_threshold: T,
    /// Training epochs per candidate
    pub epochs_per_candidate: usize,
    /// Minimum improvement threshold
    pub min_improvement: T,
}

impl ActivationMapper {
    /// Create new activation mapper
    pub fn new() -> Self {
        let mut function_map = HashMap::new();
        
        // Map common activation function names
        function_map.insert("linear".to_string(), ActivationFunction::Linear);
        function_map.insert("sigmoid".to_string(), ActivationFunction::Sigmoid);
        function_map.insert("sigmoid_symmetric".to_string(), ActivationFunction::SigmoidSymmetric);
        function_map.insert("gaussian".to_string(), ActivationFunction::Gaussian);
        function_map.insert("gaussian_symmetric".to_string(), ActivationFunction::GaussianSymmetric);
        function_map.insert("elliot".to_string(), ActivationFunction::Elliot);
        function_map.insert("elliot_symmetric".to_string(), ActivationFunction::ElliotSymmetric);
        function_map.insert("linear_piece".to_string(), ActivationFunction::LinearPiece);
        function_map.insert("linear_piece_symmetric".to_string(), ActivationFunction::LinearPieceSymmetric);
        function_map.insert("sin_symmetric".to_string(), ActivationFunction::SinSymmetric);
        function_map.insert("cos_symmetric".to_string(), ActivationFunction::CosSymmetric);

        Self { function_map }
    }

    /// Map string name to activation function
    pub fn map_function(&self, name: &str) -> Option<ActivationFunction> {
        self.function_map.get(name).copied()
    }

    /// Get all available function names
    pub fn available_functions(&self) -> Vec<&String> {
        self.function_map.keys().collect()
    }
}

/// Default input preprocessor implementation
pub struct DefaultInputPreprocessor<T: Float + Send + Sync + 'static> {
    config: PreprocessorConfig<T>,
}

impl<T: Float + Send + Sync + 'static> DefaultInputPreprocessor<T> {
    /// Create a new default input preprocessor
    pub fn new() -> Self {
        Self {
            config: PreprocessorConfig {
                scaling_method: ScalingMethod::Standard,
                window_size: 10,
                include_trend: false,
                include_seasonal: false,
                lag_features: Vec::new(),
                rolling_features: None,
                normalization: None,
            },
        }
    }
}

impl<T: Float + Send + Sync + 'static> InputPreprocessor<T> for DefaultInputPreprocessor<T> {
    fn process(&self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<Vec<T>> {
        // Simple default preprocessing: just return the target history
        let window_size = self.config.window_size.min(input.target_history.len());
        Ok(input.target_history[input.target_history.len() - window_size..].to_vec())
    }

    fn output_size(&self) -> usize {
        self.config.window_size
    }

    fn config(&self) -> &PreprocessorConfig<T> {
        &self.config
    }

    fn reset(&mut self) {
        // Nothing to reset in default implementation
    }
}

/// Default output postprocessor implementation
pub struct DefaultOutputPostprocessor<T: Float + Send + Sync + 'static> {
    config: PostprocessorConfig,
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync + 'static> DefaultOutputPostprocessor<T> {
    /// Create a new default output postprocessor
    pub fn new() -> Self {
        Self {
            config: PostprocessorConfig {
                horizon: 1,
                denormalize: false,
                confidence_intervals: None,
                trend_adjustment: false,
                seasonal_adjustment: None,
            },
            _phantom: PhantomData,
        }
    }
}

impl<T: Float + Send + Sync + 'static> OutputPostprocessor<T> for DefaultOutputPostprocessor<T> {
    fn process(&self, output: &[T], context: &PostprocessorContext<T>) -> NeuroDivergentResult<ForecastResult<T>> {
        Ok(ForecastResult {
            forecasts: output.to_vec(),
            timestamps: context.future_timestamps.clone(),
            series_id: context.input.series_id.clone(),
            model_name: context.model_name.clone(),
            generated_at: chrono::Utc::now(),
            metadata: None,
        })
    }

    fn config(&self) -> &PostprocessorConfig {
        &self.config
    }

    fn reset(&mut self) {
        // Nothing to reset in default implementation
    }
}

impl Default for ActivationMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruv_fann::NetworkBuilder;

    #[test]
    fn test_activation_mapper() {
        let mapper = ActivationMapper::new();
        
        assert_eq!(mapper.map_function("sigmoid"), Some(ActivationFunction::Sigmoid));
        assert_eq!(mapper.map_function("linear"), Some(ActivationFunction::Linear));
        assert_eq!(mapper.map_function("nonexistent"), None);
        
        let functions = mapper.available_functions();
        assert!(!functions.is_empty());
        assert!(functions.contains(&&"sigmoid".to_string()));
    }

    #[test]
    fn test_network_adapter_creation() {
        let network = NetworkBuilder::<f64>::new()
            .input_layer(3)
            .hidden_layer(5)
            .output_layer(1)
            .build();

        let adapter = NetworkAdapter::from_network(network);
        assert_eq!(adapter.config.input_size, 3);
        assert_eq!(adapter.config.output_size, 1);
    }

    #[test]
    fn network_adapter_propagates_checked_core_inference_errors() {
        let network = NetworkBuilder::<f64>::new().input_layer(2)
            .output_layer_with_activation(1, ActivationFunction::Linear, 1.0).build();
        let mut adapter = NetworkAdapter::from_network(network);
        adapter.network_mut().set_weights(&[0.5, -0.25, 0.1]).unwrap();
        let input = TimeSeriesInput {
            target_history: vec![1.0, 2.0],
            exogenous_history: None,
            exogenous_future: None,
            static_features: None,
            timestamps: Vec::new(),
            series_id: "test".to_string(),
            metadata: HashMap::new(),
        };
        let output = adapter.forward(&input).unwrap();
        assert!((output.raw_outputs[0] - 0.1).abs() < 1e-12);
        adapter.network_mut().layers.clear();
        let error = adapter.forward(&input).unwrap_err();
        assert!(error.to_string().contains("Network has no layers"));
    }

    #[test]
    fn test_training_bridge_creation() {
        let algorithm = Box::new(ruv_fann::training::IncrementalBackprop::new(0.01));
        let bridge = TrainingBridge::<f64>::new(algorithm);
        
        assert_eq!(bridge.config.max_epochs, 1000);
        assert!(bridge.config.shuffle);
    }

    #[test]
    fn test_default_preprocessor() {
        let mut preprocessor = DefaultInputPreprocessor::<f64>::new();
        
        let input = TimeSeriesInput {
            target_history: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            exogenous_history: None,
            exogenous_future: None,
            static_features: None,
            timestamps: Vec::new(),
            series_id: "test".to_string(),
            metadata: HashMap::new(),
        };

        let processed = preprocessor.process(&input).unwrap();
        assert_eq!(processed.len(), 5); // Should return all values since window_size is 10
        assert_eq!(processed, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_configuration_serialization() {
        let config = NetworkAdapterConfig::<f64> {
            input_size: 10,
            output_size: 1,
            hidden_layers: vec![20, 15],
            activations: vec![ActivationFunction::Sigmoid, ActivationFunction::Linear],
            use_bias: true,
            dropout_rates: Some(vec![0.1, 0.2]),
            network_type: "feedforward".to_string(),
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: NetworkAdapterConfig<f64> = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(config.input_size, deserialized.input_size);
        assert_eq!(config.output_size, deserialized.output_size);
        assert_eq!(config.hidden_layers, deserialized.hidden_layers);
    }
}
#[cfg(test)]
#[path = "integration_regression_tests.rs"]
mod integration_regression_tests;
