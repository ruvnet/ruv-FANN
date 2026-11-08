//! Fluent API builders for model construction
//! 
//! This module provides builder patterns for creating neural forecasting models
//! with a fluent, user-friendly API that matches the Python NeuralForecast experience.

use std::collections::HashMap;
use std::marker::PhantomData;
use num_traits::{Float, NumCast};
use serde::{Serialize, Deserialize};

use crate::config::{
    LossFunction, OptimizerType, ScalerType, EarlyStoppingConfig, 
    SchedulerConfig, Device, PredictionIntervals
};
use crate::errors::{NeuroDivergentError, NeuroDivergentResult};

/// Base trait for model configuration builders
pub trait ModelBuilder<T: Float, M> {
    /// Build the final model configuration
    fn build(self) -> NeuroDivergentResult<M>;
    
    /// Validate the builder configuration
    fn validate(&self) -> NeuroDivergentResult<()>;
}

/// Generic model configuration builder
#[derive(Debug, Clone)]
pub struct GenericModelBuilder<T: Float> {
    model_type: String,
    horizon: Option<usize>,
    input_size: Option<usize>,
    parameters: HashMap<String, ConfigValue<T>>,
    phantom: PhantomData<T>,
}

impl<T: Float> GenericModelBuilder<T> {
    /// Create new generic model builder
    pub fn new(model_type: impl Into<String>) -> Self {
        Self {
            model_type: model_type.into(),
            horizon: None,
            input_size: None,
            parameters: HashMap::new(),
            phantom: PhantomData,
        }
    }
    
    /// Set forecast horizon
    pub fn horizon(mut self, horizon: usize) -> Self {
        self.horizon = Some(horizon);
        self
    }
    
    /// Set input window size
    pub fn input_size(mut self, input_size: usize) -> Self {
        self.input_size = Some(input_size);
        self
    }
    
    /// Set a parameter value
    pub fn parameter<K: Into<String>>(mut self, key: K, value: ConfigValue<T>) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }
    
    /// Set float parameter
    pub fn float_param<K: Into<String>>(self, key: K, value: T) -> Self {
        self.parameter(key, ConfigValue::Float(value))
    }
    
    /// Set integer parameter
    pub fn int_param<K: Into<String>>(self, key: K, value: i64) -> Self {
        self.parameter(key, ConfigValue::Int(value))
    }
    
    /// Set string parameter
    pub fn string_param<K: Into<String>>(self, key: K, value: String) -> Self {
        self.parameter(key, ConfigValue::String(value))
    }
    
    /// Set boolean parameter
    pub fn bool_param<K: Into<String>>(self, key: K, value: bool) -> Self {
        self.parameter(key, ConfigValue::Bool(value))
    }
}

/// Configuration value for generic builders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue<T: Float> {
    Float(T),
    Int(i64),
    String(String),
    Bool(bool),
    FloatVec(Vec<T>),
    IntVec(Vec<i64>),
    StringVec(Vec<String>),
}

/// Build an LSTM-shaped configuration. This does not implement an LSTM model.
/// 
/// # Example
/// ```rust
/// use neuro_divergent::builders::{LSTMBuilder, ModelBuilder};
/// 
/// let lstm_config = LSTMBuilder::<f64>::new()
///     .horizon(12)
///     .input_size(24)
///     .hidden_size(128)
///     .num_layers(2)
///     .dropout(0.1)
///     .bidirectional(true)
///     .learning_rate(0.001)
///     .max_steps(1000)
///     .early_stopping_patience(10)
///     .build()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct LSTMBuilder<T: Float> {
    // Required parameters
    horizon: Option<usize>,
    input_size: Option<usize>,
    
    // Architecture parameters
    hidden_size: usize,
    num_layers: usize,
    dropout: T,
    bidirectional: bool,
    
    // Encoder-decoder parameters
    encoder_hidden_size: Option<usize>,
    decoder_hidden_size: Option<usize>,
    
    // Training parameters
    max_steps: usize,
    learning_rate: T,
    weight_decay: T,
    gradient_clip_val: Option<T>,
    
    // Data parameters
    scaler_type: ScalerType,
    static_features: Option<Vec<String>>,
    hist_exog_features: Option<Vec<String>>,
    futr_exog_features: Option<Vec<String>>,
    
    // Advanced parameters
    loss_function: LossFunction,
    optimizer: OptimizerType,
    scheduler: Option<SchedulerConfig<T>>,
    early_stopping: Option<EarlyStoppingConfig<T>>,
    
    // Inference parameters
    prediction_intervals: Option<PredictionIntervals>,
    num_samples: usize,
    
    // Device
    device: Device,
    
    phantom: PhantomData<T>,
}

impl<T: Float + From<f32>> LSTMBuilder<T> {
    /// Create new LSTM builder with default values
    pub fn new() -> Self {
        Self {
            horizon: None,
            input_size: None,
            hidden_size: 128,
            num_layers: 2,
            dropout: NumCast::from(0.1f64).unwrap(),
            bidirectional: false,
            encoder_hidden_size: None,
            decoder_hidden_size: None,
            max_steps: 1000,
            learning_rate: NumCast::from(0.001f64).unwrap(),
            weight_decay: NumCast::from(1e-3f64).unwrap(),
            gradient_clip_val: None,
            scaler_type: ScalerType::StandardScaler,
            static_features: None,
            hist_exog_features: None,
            futr_exog_features: None,
            loss_function: LossFunction::MAE,
            optimizer: OptimizerType::Adam,
            scheduler: None,
            early_stopping: None,
            prediction_intervals: None,
            num_samples: 100,
            device: Device::CPU,
            phantom: PhantomData,
        }
    }
    
    /// Set forecast horizon (required)
    pub fn horizon(mut self, horizon: usize) -> Self {
        self.horizon = Some(horizon);
        self
    }
    
    /// Set input window size (required)
    pub fn input_size(mut self, input_size: usize) -> Self {
        self.input_size = Some(input_size);
        self
    }
    
    /// Set hidden layer size
    pub fn hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }
    
    /// Set number of LSTM layers
    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }
    
    /// Set dropout rate
    pub fn dropout(mut self, dropout: T) -> Self {
        self.dropout = dropout;
        self
    }
    
    /// Enable bidirectional LSTM
    pub fn bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }
    
    /// Set encoder hidden size (for encoder-decoder architecture)
    pub fn encoder_hidden_size(mut self, size: usize) -> Self {
        self.encoder_hidden_size = Some(size);
        self
    }
    
    /// Set decoder hidden size (for encoder-decoder architecture)
    pub fn decoder_hidden_size(mut self, size: usize) -> Self {
        self.decoder_hidden_size = Some(size);
        self
    }
    
    /// Set maximum training steps
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
    
    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: T) -> Self {
        self.learning_rate = learning_rate;
        self
    }
    
    /// Set weight decay for regularization
    pub fn weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    
    /// Set gradient clipping value
    pub fn gradient_clip_val(mut self, clip_val: T) -> Self {
        self.gradient_clip_val = Some(clip_val);
        self
    }
    
    /// Set data scaler type
    pub fn scaler_type(mut self, scaler_type: ScalerType) -> Self {
        self.scaler_type = scaler_type;
        self
    }
    
    /// Set static features
    pub fn static_features(mut self, features: Vec<String>) -> Self {
        self.static_features = Some(features);
        self
    }
    
    /// Set historical exogenous features
    pub fn hist_exog_features(mut self, features: Vec<String>) -> Self {
        self.hist_exog_features = Some(features);
        self
    }
    
    /// Set future exogenous features
    pub fn futr_exog_features(mut self, features: Vec<String>) -> Self {
        self.futr_exog_features = Some(features);
        self
    }
    
    /// Set loss function
    pub fn loss_function(mut self, loss_function: LossFunction) -> Self {
        self.loss_function = loss_function;
        self
    }
    
    /// Set optimizer type
    pub fn optimizer(mut self, optimizer: OptimizerType) -> Self {
        self.optimizer = optimizer;
        self
    }
    
    /// Set learning rate scheduler
    pub fn scheduler(mut self, scheduler: SchedulerConfig<T>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }
    
    /// Set early stopping configuration
    pub fn early_stopping(mut self, config: EarlyStoppingConfig<T>) -> Self {
        self.early_stopping = Some(config);
        self
    }
    
    /// Enable early stopping with patience
    pub fn early_stopping_patience(mut self, patience: usize) -> Self 
    where
        T: From<f32>,
    {
        self.early_stopping = Some(EarlyStoppingConfig::new(
            "val_loss".to_string(),
            patience,
            NumCast::from(0.001f64).unwrap(),
            crate::config::EarlyStoppingMode::Min,
        ));
        self
    }
    
    /// Set prediction intervals
    pub fn prediction_intervals(mut self, intervals: PredictionIntervals) -> Self {
        self.prediction_intervals = Some(intervals);
        self
    }
    
    /// Set number of samples for probabilistic prediction
    pub fn num_samples(mut self, num_samples: usize) -> Self {
        self.num_samples = num_samples;
        self
    }
    
    /// Set computation device
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
    
    /// Convenience method to set architecture parameters
    pub fn with_architecture(mut self, hidden_size: usize, num_layers: usize, dropout: T) -> Self {
        self.hidden_size = hidden_size;
        self.num_layers = num_layers;
        self.dropout = dropout;
        self
    }
    
    /// Convenience method to set training parameters
    pub fn with_training(mut self, max_steps: usize, learning_rate: T) -> Self {
        self.max_steps = max_steps;
        self.learning_rate = learning_rate;
        self
    }
    
    /// Convenience method to set exogenous features
    pub fn with_exog_features(
        mut self,
        static_features: Option<Vec<String>>,
        hist_exog_features: Option<Vec<String>>,
        futr_exog_features: Option<Vec<String>>,
    ) -> Self {
        self.static_features = static_features;
        self.hist_exog_features = hist_exog_features;
        self.futr_exog_features = futr_exog_features;
        self
    }
}

impl<T: Float + From<f32>> ModelBuilder<T, LSTMConfig<T>> for LSTMBuilder<T> {
    fn validate(&self) -> NeuroDivergentResult<()> {
        if self.horizon.is_none() {
            return Err(NeuroDivergentError::config("horizon is required"));
        }
        if self.input_size.is_none() {
            return Err(NeuroDivergentError::config("input_size is required"));
        }
        if self.hidden_size == 0 {
            return Err(NeuroDivergentError::config("hidden_size must be greater than 0"));
        }
        if self.num_layers == 0 {
            return Err(NeuroDivergentError::config("num_layers must be greater than 0"));
        }
        if self.dropout < NumCast::from(0.0f64).unwrap() || self.dropout >= NumCast::from(1.0f64).unwrap() {
            return Err(NeuroDivergentError::config("dropout must be in range [0, 1)"));
        }
        if self.max_steps == 0 {
            return Err(NeuroDivergentError::config("max_steps must be greater than 0"));
        }
        if self.learning_rate <= NumCast::from(0.0f64).unwrap() {
            return Err(NeuroDivergentError::config("learning_rate must be positive"));
        }
        Ok(())
    }
    
    fn build(self) -> NeuroDivergentResult<LSTMConfig<T>> {
        self.validate()?;
        
        Ok(LSTMConfig {
            horizon: self.horizon.unwrap(),
            input_size: self.input_size.unwrap(),
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            dropout: self.dropout,
            bidirectional: self.bidirectional,
            encoder_hidden_size: self.encoder_hidden_size.unwrap_or(self.hidden_size),
            decoder_hidden_size: self.decoder_hidden_size.unwrap_or(self.hidden_size),
            max_steps: self.max_steps,
            learning_rate: self.learning_rate,
            weight_decay: self.weight_decay,
            gradient_clip_val: self.gradient_clip_val,
            scaler_type: self.scaler_type,
            static_features: self.static_features,
            hist_exog_features: self.hist_exog_features,
            futr_exog_features: self.futr_exog_features,
            loss_function: self.loss_function,
            optimizer: self.optimizer,
            scheduler: self.scheduler,
            early_stopping: self.early_stopping,
            prediction_intervals: self.prediction_intervals,
            num_samples: self.num_samples,
            device: self.device,
        })
    }
}

impl<T: Float + From<f32>> Default for LSTMBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// LSTM configuration struct (output of LSTMBuilder)
#[derive(Debug, Clone)]
pub struct LSTMConfig<T: Float> {
    pub horizon: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: T,
    pub bidirectional: bool,
    pub encoder_hidden_size: usize,
    pub decoder_hidden_size: usize,
    pub max_steps: usize,
    pub learning_rate: T,
    pub weight_decay: T,
    pub gradient_clip_val: Option<T>,
    pub scaler_type: ScalerType,
    pub static_features: Option<Vec<String>>,
    pub hist_exog_features: Option<Vec<String>>,
    pub futr_exog_features: Option<Vec<String>>,
    pub loss_function: LossFunction,
    pub optimizer: OptimizerType,
    pub scheduler: Option<SchedulerConfig<T>>,
    pub early_stopping: Option<EarlyStoppingConfig<T>>,
    pub prediction_intervals: Option<PredictionIntervals>,
    pub num_samples: usize,
    pub device: Device,
}

/// Build an N-BEATS-shaped configuration. The facade does not implement N-BEATS.
/// 
/// # Example
/// ```rust
/// use neuro_divergent::builders::{NBEATSBuilder, ModelBuilder};
/// 
/// let nbeats_config = NBEATSBuilder::<f64>::interpretable()
///     .horizon(12)
///     .input_size(24)
///     .n_blocks(vec![3, 3])
///     .learning_rate(0.001)
///     .build()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct NBEATSBuilder<T: Float> {
    // Required parameters
    horizon: Option<usize>,
    input_size: Option<usize>,
    
    // Architecture parameters
    stack_types: Vec<StackType>,
    n_blocks: Vec<usize>,
    mlp_units: Vec<Vec<usize>>,
    shared_weights: bool,
    activation: ActivationFunction,
    
    // Training parameters
    max_steps: usize,
    learning_rate: T,
    weight_decay: T,
    loss_function: LossFunction,
    
    // Data parameters
    scaler_type: ScalerType,
    static_features: Option<Vec<String>>,
    
    // Interpretability parameters
    include_trend: bool,
    include_seasonality: bool,
    seasonality_period: Option<usize>,
    
    // Advanced parameters
    optimizer: OptimizerType,
    scheduler: Option<SchedulerConfig<T>>,
    early_stopping: Option<EarlyStoppingConfig<T>>,
    device: Device,
    
    phantom: PhantomData<T>,
}

/// Stack types for NBEATS architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackType {
    Generic,
    Trend,
    Seasonality,
}

/// Activation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    ReLU,
    Tanh,
    Sigmoid,
    Swish,
    GELU,
}

impl<T: Float + From<f32>> NBEATSBuilder<T> {
    /// Create new NBEATS builder with generic configuration
    pub fn new() -> Self {
        Self {
            horizon: None,
            input_size: None,
            stack_types: vec![StackType::Generic; 4],
            n_blocks: vec![3; 4],
            mlp_units: vec![vec![512, 512]; 4],
            shared_weights: true,
            activation: ActivationFunction::ReLU,
            max_steps: 1000,
            learning_rate: NumCast::from(0.001f64).unwrap(),
            weight_decay: NumCast::from(1e-3f64).unwrap(),
            loss_function: LossFunction::MAE,
            scaler_type: ScalerType::StandardScaler,
            static_features: None,
            include_trend: false,
            include_seasonality: false,
            seasonality_period: None,
            optimizer: OptimizerType::Adam,
            scheduler: None,
            early_stopping: None,
            device: Device::CPU,
            phantom: PhantomData,
        }
    }
    
    /// Create interpretable NBEATS configuration
    pub fn interpretable() -> Self {
        Self {
            stack_types: vec![StackType::Trend, StackType::Seasonality],
            n_blocks: vec![3, 3],
            mlp_units: vec![vec![512, 512], vec![512, 512]],
            include_trend: true,
            include_seasonality: true,
            ..Self::new()
        }
    }
    
    /// Set forecast horizon (required)
    pub fn horizon(mut self, horizon: usize) -> Self {
        self.horizon = Some(horizon);
        self
    }
    
    /// Set input window size (required)
    pub fn input_size(mut self, input_size: usize) -> Self {
        self.input_size = Some(input_size);
        self
    }
    
    /// Set stack types
    pub fn stack_types(mut self, stack_types: Vec<StackType>) -> Self {
        self.stack_types = stack_types;
        self
    }
    
    /// Set number of blocks per stack
    pub fn n_blocks(mut self, n_blocks: Vec<usize>) -> Self {
        self.n_blocks = n_blocks;
        self
    }
    
    /// Set MLP units for each stack
    pub fn mlp_units(mut self, mlp_units: Vec<Vec<usize>>) -> Self {
        self.mlp_units = mlp_units;
        self
    }
    
    /// Set whether weights are shared across blocks
    pub fn shared_weights(mut self, shared: bool) -> Self {
        self.shared_weights = shared;
        self
    }
    
    /// Set activation function
    pub fn activation(mut self, activation: ActivationFunction) -> Self {
        self.activation = activation;
        self
    }
    
    /// Set maximum training steps
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
    
    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: T) -> Self {
        self.learning_rate = learning_rate;
        self
    }
    
    /// Set seasonality period
    pub fn seasonality_period(mut self, period: usize) -> Self {
        self.seasonality_period = Some(period);
        self
    }
    
    /// Enable trend component
    pub fn with_trend(mut self) -> Self {
        self.include_trend = true;
        self
    }
    
    /// Enable seasonality component
    pub fn with_seasonality(mut self) -> Self {
        self.include_seasonality = true;
        self
    }
}

impl<T: Float + From<f32>> ModelBuilder<T, NBEATSConfig<T>> for NBEATSBuilder<T> {
    fn validate(&self) -> NeuroDivergentResult<()> {
        if self.horizon.is_none() {
            return Err(NeuroDivergentError::config("horizon is required"));
        }
        if self.input_size.is_none() {
            return Err(NeuroDivergentError::config("input_size is required"));
        }
        if self.stack_types.len() != self.n_blocks.len() {
            return Err(NeuroDivergentError::config(
                "stack_types and n_blocks must have the same length"
            ));
        }
        if self.stack_types.len() != self.mlp_units.len() {
            return Err(NeuroDivergentError::config(
                "stack_types and mlp_units must have the same length"
            ));
        }
        Ok(())
    }
    
    fn build(self) -> NeuroDivergentResult<NBEATSConfig<T>> {
        self.validate()?;
        
        Ok(NBEATSConfig {
            horizon: self.horizon.unwrap(),
            input_size: self.input_size.unwrap(),
            stack_types: self.stack_types,
            n_blocks: self.n_blocks,
            mlp_units: self.mlp_units,
            shared_weights: self.shared_weights,
            activation: self.activation,
            max_steps: self.max_steps,
            learning_rate: self.learning_rate,
            weight_decay: self.weight_decay,
            loss_function: self.loss_function,
            scaler_type: self.scaler_type,
            static_features: self.static_features,
            include_trend: self.include_trend,
            include_seasonality: self.include_seasonality,
            seasonality_period: self.seasonality_period,
            optimizer: self.optimizer,
            scheduler: self.scheduler,
            early_stopping: self.early_stopping,
            device: self.device,
        })
    }
}

/// NBEATS configuration struct
#[derive(Debug, Clone)]
pub struct NBEATSConfig<T: Float> {
    pub horizon: usize,
    pub input_size: usize,
    pub stack_types: Vec<StackType>,
    pub n_blocks: Vec<usize>,
    pub mlp_units: Vec<Vec<usize>>,
    pub shared_weights: bool,
    pub activation: ActivationFunction,
    pub max_steps: usize,
    pub learning_rate: T,
    pub weight_decay: T,
    pub loss_function: LossFunction,
    pub scaler_type: ScalerType,
    pub static_features: Option<Vec<String>>,
    pub include_trend: bool,
    pub include_seasonality: bool,
    pub seasonality_period: Option<usize>,
    pub optimizer: OptimizerType,
    pub scheduler: Option<SchedulerConfig<T>>,
    pub early_stopping: Option<EarlyStoppingConfig<T>>,
    pub device: Device,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lstm_builder() {
        let config = LSTMBuilder::<f32>::new()
            .horizon(12)
            .input_size(24)
            .hidden_size(64)
            .num_layers(1)
            .dropout(0.2)
            .learning_rate(0.01)
            .build()
            .unwrap();
            
        assert_eq!(config.horizon, 12);
        assert_eq!(config.input_size, 24);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_layers, 1);
        assert_eq!(config.dropout, 0.2);
        assert_eq!(config.learning_rate, 0.01);
    }
    
    #[test]
    fn test_lstm_builder_validation() {
        let result = LSTMBuilder::<f32>::new()
            .hidden_size(64)
            .build();
            
        assert!(result.is_err());
        assert!(result.unwrap_err().is_config_error());
    }
    
    #[test]
    fn test_nbeats_builder() {
        let config = NBEATSBuilder::<f32>::interpretable()
            .horizon(12)
            .input_size(24)
            .learning_rate(0.001)
            .build()
            .unwrap();
            
        assert_eq!(config.horizon, 12);
        assert_eq!(config.input_size, 24);
        assert!(config.include_trend);
        assert!(config.include_seasonality);
        assert_eq!(config.stack_types, vec![StackType::Trend, StackType::Seasonality]);
    }
    
    #[test]
    fn test_nbeats_generic() {
        let config = NBEATSBuilder::<f32>::new()
            .horizon(12)
            .input_size(24)
            .build()
            .unwrap();
            
        assert!(!config.include_trend);
        assert!(!config.include_seasonality);
        assert_eq!(config.stack_types, vec![StackType::Generic; 4]);
    }
}