//! Pipeline construction and explicit runtime options.
use crate::config::{Device, Frequency, PredictionIntervals, ScalerType};
use crate::neural_forecast::{BaseModel, NeuralForecast};
use crate::{NeuroDivergentError, NeuroDivergentResult};
use num_traits::Float;

/// Builder for constructing NeuralForecast instances with fluent API
#[derive(Debug)]
pub struct NeuralForecastBuilder<T: Float + Send + Sync> {
    models: Vec<Box<dyn BaseModel<T>>>,
    frequency: Option<Frequency>,
    local_scaler_type: Option<ScalerType>,
    num_threads: Option<usize>,
    prediction_intervals: Option<PredictionIntervals>,
    device: Device,
}

impl<T: Float + Send + Sync> NeuralForecastBuilder<T> {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            frequency: None,
            local_scaler_type: None,
            num_threads: None,
            prediction_intervals: None,
            device: Device::default(),
        }
    }

    /// Add models to the forecasting ensemble
    pub fn with_models(mut self, models: Vec<Box<dyn BaseModel<T>>>) -> Self {
        self.models = models;
        self
    }

    /// Add a single model
    pub fn with_model(mut self, model: Box<dyn BaseModel<T>>) -> Self {
        self.models.push(model);
        self
    }

    /// Set the frequency of the time series
    pub fn with_frequency(mut self, frequency: Frequency) -> Self {
        self.frequency = Some(frequency);
        self
    }

    /// Set local scaler type for data preprocessing
    pub fn with_local_scaler(mut self, scaler_type: ScalerType) -> Self {
        self.local_scaler_type = Some(scaler_type);
        self
    }

    /// Set number of threads for parallel processing
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Configure prediction intervals
    pub fn with_prediction_intervals(mut self, intervals: PredictionIntervals) -> Self {
        self.prediction_intervals = Some(intervals);
        self
    }

    /// Set computation device
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Build the NeuralForecast instance
    pub fn build(self) -> NeuroDivergentResult<NeuralForecast<T>> {
        let frequency = self
            .frequency
            .ok_or_else(|| NeuroDivergentError::config("Frequency is required"))?;

        if self.models.is_empty() {
            return Err(NeuroDivergentError::config(
                "At least one model is required",
            ));
        }

        if self
            .num_threads
            .is_some_and(|count| count == 0 || count > 256)
        {
            return Err(NeuroDivergentError::config(
                "Thread count must be in 1..=256",
            ));
        }
        if !matches!(self.device, Device::CPU) || self.prediction_intervals.is_some() {
            return Err(NeuroDivergentError::config(
                "GPU execution and prediction intervals are not implemented",
            ));
        }
        let mut nf = NeuralForecast::new(self.models, frequency)?;
        nf.local_scaler_type = self.local_scaler_type;
        nf.num_threads = self.num_threads;
        nf.prediction_intervals = self.prediction_intervals;
        nf.device = self.device;

        Ok(nf)
    }
}

impl<T: Float + Send + Sync> Default for NeuralForecastBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for model validation during training
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Validation split ratio
    pub validation_split: Option<f64>,
    /// Whether to shuffle data
    pub shuffle: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl ValidationConfig {
    /// Create new validation configuration
    pub fn new() -> Self {
        Self {
            validation_split: None,
            shuffle: false,
            random_seed: None,
        }
    }

    /// Set validation split ratio
    pub fn with_validation_split(mut self, split: f64) -> Self {
        self.validation_split = Some(split);
        self
    }

    /// Set whether to shuffle data
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random seed
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for prediction generation
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    /// Whether to include prediction intervals
    pub include_intervals: bool,
    /// Number of samples for probabilistic models
    pub num_samples: Option<usize>,
    /// Temperature for sampling
    pub temperature: Option<f64>,
}

impl PredictionConfig {
    /// Create new prediction configuration
    pub fn new() -> Self {
        Self {
            include_intervals: false,
            num_samples: None,
            temperature: None,
        }
    }

    /// Enable prediction intervals
    pub fn with_intervals(mut self) -> Self {
        self.include_intervals = true;
        self
    }

    /// Set number of samples for probabilistic prediction
    pub fn with_num_samples(mut self, num_samples: usize) -> Self {
        self.num_samples = Some(num_samples);
        self
    }

    /// Set sampling temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self::new()
    }
}
