//! Configuration structures for neural forecasting models
//!
//! This module provides configuration structures for all supported models,
//! including training parameters, model architecture, and validation settings.

use crate::errors::{NeuroDivergentError, NeuroDivergentResult};
use crate::foundation::{ConfigValue, ModelConfig};
use num_traits::Float;
use ruv_fann::ActivationFunction;
use std::collections::HashMap;

/// Configuration for RNN models
#[derive(Debug, Clone)]
pub struct RNNConfig<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    // Required parameters
    pub horizon: usize,
    pub input_size: usize,

    // Architecture parameters
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: T,
    pub activation: ActivationFunction,

    // Training parameters
    pub max_steps: usize,
    pub learning_rate: T,
    pub weight_decay: T,
    pub gradient_clip_val: Option<T>,

    // Data parameters
    pub static_features: Option<Vec<String>>,
    pub hist_exog_features: Option<Vec<String>>,
    pub futr_exog_features: Option<Vec<String>>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> RNNConfig<T> {
    pub fn default_with_horizon(horizon: usize) -> Self {
        Self {
            horizon,
            input_size: 1,
            hidden_size: 64,
            num_layers: 1,
            dropout: T::from(0.0).unwrap(),
            activation: ActivationFunction::Tanh,
            max_steps: 1000,
            learning_rate: T::from(0.001).unwrap(),
            weight_decay: T::from(0.0).unwrap(),
            gradient_clip_val: None,
            static_features: None,
            hist_exog_features: None,
            futr_exog_features: None,
        }
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> ModelConfig<T>
    for RNNConfig<T>
{
    fn model_type(&self) -> &'static str {
        "RNN"
    }

    fn horizon(&self) -> usize {
        self.horizon
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn validate(&self) -> NeuroDivergentResult<()> {
        if self.horizon == 0 {
            return Err(NeuroDivergentError::config(
                "Horizon must be greater than 0",
            ));
        }
        if self.hidden_size == 0 {
            return Err(NeuroDivergentError::config(
                "Hidden size must be greater than 0",
            ));
        }
        if self.num_layers == 0 || self.num_layers > 1024 {
            return Err(NeuroDivergentError::config(
                "Number of RNN layers must be in 1..=1024",
            ));
        }
        if self.input_size == 0 || self.max_steps == 0 {
            return Err(NeuroDivergentError::config(
                "Input size and max_steps must be positive",
            ));
        }
        if !self.learning_rate.is_finite()
            || self.learning_rate <= T::zero()
            || !self.weight_decay.is_finite()
            || self.weight_decay < T::zero()
            || !self.dropout.is_finite()
            || self.dropout < T::zero()
            || self.dropout >= T::one()
            || self
                .gradient_clip_val
                .is_some_and(|v| !v.is_finite() || v <= T::zero())
        {
            return Err(NeuroDivergentError::config("Learning rate and clipping must be positive and finite; weight decay must be nonnegative; dropout must be in [0, 1)"));
        }
        if !matches!(
            self.activation,
            ActivationFunction::Tanh
                | ActivationFunction::SigmoidSymmetric
                | ActivationFunction::Sigmoid
                | ActivationFunction::ReLU
                | ActivationFunction::Linear
                | ActivationFunction::ReLULeaky
                | ActivationFunction::LeakyRelu
        ) {
            return Err(NeuroDivergentError::config(
                "RNN training supports tanh, sigmoid, ReLU, leaky ReLU, and linear activations",
            ));
        }
        let count = self
            .input_size
            .checked_add(self.hidden_size)
            .and_then(|v| v.checked_add(1))
            .and_then(|v| v.checked_mul(self.hidden_size))
            .and_then(|first| {
                self.hidden_size
                    .checked_mul(2)
                    .and_then(|v| v.checked_add(1))
                    .and_then(|v| v.checked_mul(self.hidden_size))
                    .and_then(|v| v.checked_mul(self.num_layers - 1))
                    .and_then(|v| v.checked_add(first))
            })
            .and_then(|recurrent| {
                self.hidden_size
                    .checked_add(1)
                    .and_then(|v| v.checked_mul(self.horizon))
                    .and_then(|v| v.checked_add(recurrent))
            });
        if count.map_or(true, |v| v > 16 * 1024 * 1024) {
            return Err(NeuroDivergentError::config(
                "RNN exceeds the 16 million parameter allocation budget",
            ));
        }
        if self
            .futr_exog_features
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        {
            return Err(NeuroDivergentError::config(
                "RNN does not support future exogenous features",
            ));
        }
        Ok(())
    }

    fn to_map(&self) -> HashMap<String, ConfigValue<T>> {
        crate::rnn_config_map::to_map(self)
    }

    fn from_map(map: HashMap<String, ConfigValue<T>>) -> NeuroDivergentResult<Self> {
        crate::rnn_config_map::from_map(map)
    }
}

/// Configuration for LSTM models
#[derive(Debug, Clone)]
pub struct LSTMConfig<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    // Required parameters
    pub horizon: usize,
    pub input_size: usize,

    // Architecture parameters
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: T,
    pub bidirectional: bool,

    // Encoder-decoder parameters
    pub encoder_hidden_size: usize,
    pub decoder_hidden_size: usize,

    // Training parameters
    pub max_steps: usize,
    pub learning_rate: T,
    pub weight_decay: T,
    pub gradient_clip_val: Option<T>,

    // Data parameters
    pub static_features: Option<Vec<String>>,
    pub hist_exog_features: Option<Vec<String>>,
    pub futr_exog_features: Option<Vec<String>>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> LSTMConfig<T> {
    pub fn default_with_horizon(horizon: usize) -> Self {
        Self {
            horizon,
            input_size: 1,
            hidden_size: 128,
            num_layers: 2,
            dropout: T::from(0.1).unwrap(),
            bidirectional: false,
            encoder_hidden_size: 128,
            decoder_hidden_size: 128,
            max_steps: 1000,
            learning_rate: T::from(0.001).unwrap(),
            weight_decay: T::from(1e-4).unwrap(),
            gradient_clip_val: Some(T::from(1.0).unwrap()),
            static_features: None,
            hist_exog_features: None,
            futr_exog_features: None,
        }
    }

    pub fn with_architecture(mut self, hidden_size: usize, num_layers: usize, dropout: T) -> Self {
        self.hidden_size = hidden_size;
        self.num_layers = num_layers;
        self.dropout = dropout;
        self
    }

    pub fn with_training(mut self, max_steps: usize, learning_rate: T) -> Self {
        self.max_steps = max_steps;
        self.learning_rate = learning_rate;
        self
    }

    pub fn bidirectional(mut self) -> Self {
        self.bidirectional = true;
        self
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> ModelConfig<T>
    for LSTMConfig<T>
{
    fn model_type(&self) -> &'static str {
        "LSTM"
    }

    fn horizon(&self) -> usize {
        self.horizon
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn validate(&self) -> NeuroDivergentResult<()> {
        if self.horizon == 0 {
            return Err(NeuroDivergentError::config(
                "Horizon must be greater than 0",
            ));
        }
        if self.hidden_size == 0 {
            return Err(NeuroDivergentError::config(
                "Hidden size must be greater than 0",
            ));
        }
        if self.dropout < T::zero() || self.dropout >= T::one() {
            return Err(NeuroDivergentError::config(
                "Dropout must be in range [0, 1)",
            ));
        }
        Ok(())
    }

    fn to_map(&self) -> HashMap<String, ConfigValue<T>> {
        let mut map = HashMap::new();
        map.insert("horizon".to_string(), ConfigValue::UInt(self.horizon));
        map.insert("input_size".to_string(), ConfigValue::UInt(self.input_size));
        map.insert(
            "hidden_size".to_string(),
            ConfigValue::UInt(self.hidden_size),
        );
        map.insert("num_layers".to_string(), ConfigValue::UInt(self.num_layers));
        map.insert("dropout".to_string(), ConfigValue::Float(self.dropout));
        map.insert(
            "bidirectional".to_string(),
            ConfigValue::Bool(self.bidirectional),
        );
        // Add other parameters as needed
        map
    }

    fn from_map(_map: HashMap<String, ConfigValue<T>>) -> NeuroDivergentResult<Self> {
        // Placeholder implementation
        Ok(Self::default_with_horizon(1))
    }
}

/// Configuration for GRU models
#[derive(Debug, Clone)]
pub struct GRUConfig<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    // Required parameters
    pub horizon: usize,
    pub input_size: usize,

    // Architecture parameters
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: T,

    // Training parameters
    pub max_steps: usize,
    pub learning_rate: T,
    pub weight_decay: T,
    pub gradient_clip_val: Option<T>,

    // Data parameters
    pub static_features: Option<Vec<String>>,
    pub hist_exog_features: Option<Vec<String>>,
    pub futr_exog_features: Option<Vec<String>>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> GRUConfig<T> {
    pub fn default_with_horizon(horizon: usize) -> Self {
        Self {
            horizon,
            input_size: 1,
            hidden_size: 64,
            num_layers: 1,
            dropout: T::from(0.0).unwrap(),
            max_steps: 1000,
            learning_rate: T::from(0.001).unwrap(),
            weight_decay: T::from(0.0).unwrap(),
            gradient_clip_val: None,
            static_features: None,
            hist_exog_features: None,
            futr_exog_features: None,
        }
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> ModelConfig<T>
    for GRUConfig<T>
{
    fn model_type(&self) -> &'static str {
        "GRU"
    }

    fn horizon(&self) -> usize {
        self.horizon
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn validate(&self) -> NeuroDivergentResult<()> {
        if self.horizon == 0 {
            return Err(NeuroDivergentError::config(
                "Horizon must be greater than 0",
            ));
        }
        if self.hidden_size == 0 {
            return Err(NeuroDivergentError::config(
                "Hidden size must be greater than 0",
            ));
        }
        Ok(())
    }

    fn to_map(&self) -> HashMap<String, ConfigValue<T>> {
        let mut map = HashMap::new();
        map.insert("horizon".to_string(), ConfigValue::UInt(self.horizon));
        map.insert("input_size".to_string(), ConfigValue::UInt(self.input_size));
        map.insert(
            "hidden_size".to_string(),
            ConfigValue::UInt(self.hidden_size),
        );
        map.insert("num_layers".to_string(), ConfigValue::UInt(self.num_layers));
        map.insert("dropout".to_string(), ConfigValue::Float(self.dropout));
        // Add other parameters as needed
        map
    }

    fn from_map(_map: HashMap<String, ConfigValue<T>>) -> NeuroDivergentResult<Self> {
        // Placeholder implementation
        Ok(Self::default_with_horizon(1))
    }
}

/// General training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    pub max_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: T,
    pub weight_decay: T,
    pub gradient_clip_val: Option<T>,
    pub validation_split: Option<T>,
    pub shuffle: bool,
    pub early_stopping_patience: Option<usize>,
    pub early_stopping_min_delta: Option<T>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> Default
    for TrainingConfig<T>
{
    fn default() -> Self {
        Self {
            max_epochs: 1000,
            batch_size: 32,
            learning_rate: T::from(0.001).unwrap(),
            weight_decay: T::from(1e-4).unwrap(),
            gradient_clip_val: Some(T::from(1.0).unwrap()),
            validation_split: Some(T::from(0.2).unwrap()),
            shuffle: true,
            early_stopping_patience: Some(10),
            early_stopping_min_delta: Some(T::from(1e-6).unwrap()),
        }
    }
}

/// Prediction configuration
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    pub return_intervals: bool,
    pub confidence_levels: Vec<f64>,
    pub num_samples: usize,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            return_intervals: false,
            confidence_levels: vec![0.8, 0.9, 0.95],
            num_samples: 100,
        }
    }
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    pub n_windows: usize,
    pub h: usize,
    pub step_size: Option<usize>,
    pub test_size: Option<usize>,
    pub season_length: Option<usize>,
    pub refit: bool,
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            n_windows: 3,
            h: 1,
            step_size: None,
            test_size: None,
            season_length: None,
            refit: true,
        }
    }
}
