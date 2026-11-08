#![doc = include_str!("../README.md")]

use num_traits::Float;
use std::collections::HashMap;
use thiserror::Error;

// Re-export ruv-FANN types for convenience
pub use ruv_fann::{Network, NetworkError};

// Core training modules
pub mod loss;
pub mod metrics;
pub mod optimizer;
pub mod scheduler;

// Re-export main types for convenience
pub use loss::*;
pub use metrics::{
    CalibrationError, CoverageProbability, EnergyScore, IntervalScore, LogScore,
    MedianAbsoluteError, Metric, MetricCalculator, PearsonCorrelation,
    PinballLoss as PinballMetric, QuantileLoss as QuantileMetric, Sharpness, SpearmanCorrelation,
    WinklerScore, CRPS, MAE, MAPE, MASE, MSE, R2, RMSE, SMAPE,
};
pub use optimizer::*;
pub use scheduler::*;

/// Error types for training operations
#[derive(Error, Debug)]
pub enum TrainingError {
    #[error("Invalid training configuration: {0}")]
    InvalidConfig(String),

    #[error("Training data error: {0}")]
    DataError(String),

    #[error("Optimizer error: {0}")]
    OptimizerError(String),

    #[error("Loss function error: {0}")]
    LossError(String),

    #[error("Scheduler error: {0}")]
    SchedulerError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Callback error: {0}")]
    CallbackError(String),

    #[error("Network integration error: {0}")]
    NetworkError(#[from] NetworkError),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Training failed: {0}")]
    TrainingFailed(String),
}

pub type TrainingResult<T> = Result<T, TrainingError>;

/// Core training data structure for time series
#[derive(Debug, Clone)]
pub struct TrainingData<T: Float + Send + Sync> {
    /// Input sequences [batch_size, sequence_length, input_features]
    pub inputs: Vec<Vec<Vec<T>>>,
    /// Target outputs [batch_size, horizon, output_features]
    pub targets: Vec<Vec<Vec<T>>>,
    /// Optional exogenous features [batch_size, total_length, exog_features]
    pub exogenous: Option<Vec<Vec<Vec<T>>>>,
    /// Static features per time series [batch_size, static_features]
    pub static_features: Option<Vec<Vec<T>>>,
    /// Metadata for each time series
    pub metadata: Vec<TimeSeriesMetadata>,
}

/// Metadata for individual time series
#[derive(Debug, Clone)]
pub struct TimeSeriesMetadata {
    pub id: String,
    pub frequency: String,
    pub seasonal_periods: Vec<usize>,
    pub scale: Option<f64>,
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig<T: Float + Send + Sync> {
    /// Maximum number of training epochs
    pub max_epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Validation frequency (every N epochs)
    pub validation_frequency: usize,
    /// Early stopping patience
    pub patience: Option<usize>,
    /// Gradient clipping threshold
    pub gradient_clip: Option<T>,
    /// Enable mixed precision training
    pub mixed_precision: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Device configuration
    pub device: DeviceConfig,
    /// Checkpoint configuration
    pub checkpoint: CheckpointConfig,
}

/// Device configuration for training
#[derive(Debug, Clone)]
pub enum DeviceConfig {
    Cpu { num_threads: Option<usize> },
    Gpu { device_id: usize },
    Multi { devices: Vec<usize> },
}

/// Checkpoint configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    pub enabled: bool,
    pub save_frequency: usize,
    pub keep_best_only: bool,
    pub monitor_metric: String,
    pub mode: CheckpointMode,
}

#[derive(Debug, Clone)]
pub enum CheckpointMode {
    Min,
    Max,
}

/// Training results and metrics
#[derive(Debug, Clone)]
pub struct TrainingResults<T: Float + Send + Sync> {
    pub final_loss: T,
    pub best_loss: T,
    pub epochs_trained: usize,
    pub training_history: Vec<EpochMetrics<T>>,
    pub validation_history: Vec<EpochMetrics<T>>,
    pub early_stopped: bool,
    pub training_time: std::time::Duration,
}

/// Metrics for a single epoch
#[derive(Debug, Clone)]
pub struct EpochMetrics<T: Float + Send + Sync> {
    pub epoch: usize,
    pub loss: T,
    pub learning_rate: T,
    pub gradient_norm: Option<T>,
    pub additional_metrics: HashMap<String, T>,
}

/// Bridge for integrating with ruv-FANN training algorithms
pub struct TrainingBridge<T: Float + Send + Sync> {
    pub(crate) ruv_fann_trainer: Option<Box<dyn ruv_fann::training::TrainingAlgorithm<T>>>,
    pub(crate) loss_adapter: Option<LossAdapter<T>>,
    pub(crate) config: Option<TrainingConfig<T>>,
}

impl<T: Float + Send + Sync> TrainingBridge<T> {
    /// Create a new training bridge
    pub fn new() -> Self {
        Self {
            ruv_fann_trainer: None,
            loss_adapter: None,
            config: None,
        }
    }

    /// Set the ruv-FANN training algorithm
    pub fn with_ruv_fann_trainer(
        mut self,
        trainer: Box<dyn ruv_fann::training::TrainingAlgorithm<T>>,
    ) -> Self {
        self.ruv_fann_trainer = Some(trainer);
        self
    }

    /// Set the loss adapter
    pub fn with_loss_adapter(mut self, adapter: LossAdapter<T>) -> Self {
        self.loss_adapter = Some(adapter);
        self
    }

    /// Set the training configuration
    pub fn with_config(mut self, config: TrainingConfig<T>) -> Self {
        self.config = Some(config);
        self
    }
}

impl<T: Float + Send + Sync> Default for TrainingBridge<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Adapter for integrating different loss functions with ruv-FANN
pub struct LossAdapter<T: Float + Send + Sync> {
    loss_function: Box<dyn LossFunction<T>>,
}

impl<T: Float + Send + Sync> LossAdapter<T> {
    pub fn new(loss_function: Box<dyn LossFunction<T>>) -> Self {
        Self { loss_function }
    }

    pub fn calculate_loss(&self, predictions: &[T], targets: &[T]) -> TrainingResult<T> {
        self.loss_function.forward(predictions, targets)
    }

    pub fn calculate_gradient(&self, predictions: &[T], targets: &[T]) -> TrainingResult<Vec<T>> {
        self.loss_function.backward(predictions, targets)
    }
}

impl<T: Float + Send + Sync> ruv_fann::training::ErrorFunction<T> for LossAdapter<T> {
    fn calculate(&self, actual: &[T], desired: &[T]) -> T {
        self.calculate_loss(actual, desired)
            .unwrap_or_else(|_| T::nan())
    }

    fn derivative(&self, actual: T, desired: T) -> T {
        let actual_slice = [actual];
        let desired_slice = [desired];
        self.calculate_gradient(&actual_slice, &desired_slice)
            .unwrap_or_else(|_| vec![T::nan()])
            .first()
            .copied()
            .unwrap_or_else(T::nan)
    }
}

/// Utility functions
pub mod utils {
    use super::*;

    /// Construct a reproducible RNG. Pass it explicitly to operations that draw
    /// randomness; rand's thread-local RNG cannot be globally reseeded.
    pub fn seeded_rng(seed: u64) -> rand::rngs::StdRng {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    #[deprecated(
        note = "Use seeded_rng and pass the returned RNG explicitly; there is no global seed"
    )]
    #[must_use]
    pub fn set_seed(seed: u64) -> rand::rngs::StdRng {
        seeded_rng(seed)
    }

    /// Calculate gradient norm for monitoring
    pub fn gradient_norm<T: Float + Send + Sync>(gradients: &[Vec<T>]) -> T {
        let sum_squares = gradients
            .iter()
            .flat_map(|layer| layer.iter())
            .fold(T::zero(), |norm, &gradient| norm.hypot(gradient));
        sum_squares
    }

    /// Clip gradients by norm
    pub fn clip_gradients_by_norm<T: Float + Send + Sync>(
        gradients: &mut [Vec<T>],
        max_norm: T,
    ) -> T {
        let norm = gradient_norm(gradients);
        if norm > max_norm && max_norm.is_finite() && max_norm >= T::zero() {
            let largest = gradients
                .iter()
                .flatten()
                .fold(T::zero(), |largest, &gradient| largest.max(gradient.abs()));
            let scaled_norm = gradients
                .iter()
                .flatten()
                .fold(T::zero(), |norm, &gradient| norm.hypot(gradient / largest));
            let scale = (max_norm / largest) / scaled_norm;
            for layer in gradients.iter_mut() {
                for gradient in layer.iter_mut() {
                    *gradient = *gradient * scale;
                }
            }
        }
        norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_bridge_creation() {
        let bridge = TrainingBridge::<f32>::new();
        assert!(bridge.ruv_fann_trainer.is_none());
        assert!(bridge.loss_adapter.is_none());
        assert!(bridge.config.is_none());
    }

    #[test]
    fn test_gradient_norm_calculation() {
        let gradients = vec![vec![3.0, 4.0], vec![0.0, 0.0]];
        let norm = utils::gradient_norm(&gradients);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_clipping() {
        let mut gradients = vec![vec![3.0, 4.0], vec![6.0, 8.0]];
        let norm = utils::clip_gradients_by_norm(&mut gradients, 5.0);
        assert!((norm - 125.0_f64.sqrt()).abs() < 1e-10); // Original norm

        // Check clipped values
        let new_norm = utils::gradient_norm(&gradients);
        assert!((new_norm - 5.0).abs() < 1e-6);
    }
}

#[cfg(test)]
#[path = "numerical_tests.rs"]
mod numerical_tests;
