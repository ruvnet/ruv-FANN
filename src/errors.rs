//! Comprehensive error handling system for ruv-FANN
//!
//! This module provides a unified error handling framework with detailed error categories,
//! context information, and recovery mechanisms for robust neural network operations.

use std::error::Error;
use thiserror::Error;
use crate::{NetworkError, TrainingError};

/// Main error type for all ruv-FANN operations
#[derive(Error, Debug)]
pub enum RuvFannError {
    /// Network configuration and topology errors
    #[error("Network error: {category:?} - {message}")]
    Network {
        category: NetworkErrorCategory,
        message: String,
        context: Option<String>,
    },

    /// Training and learning algorithm errors
    #[error("Training error: {category:?} - {message}")]
    Training {
        category: TrainingErrorCategory,
        message: String,
        context: Option<String>,
    },

    /// Cascade correlation specific errors
    #[error("Cascade error: {category:?} - {message}")]
    Cascade {
        category: CascadeErrorCategory,
        message: String,
        context: Option<String>,
    },

    /// Data validation and format errors
    #[error("Validation error: {category:?} - {message}")]
    Validation {
        category: ValidationErrorCategory,
        message: String,
        details: Vec<String>,
    },

    /// I/O and serialization errors
    #[error("I/O error: {category:?} - {message}")]
    Io {
        category: IoErrorCategory,
        message: String,
        source: Option<Box<dyn Error + Send + Sync>>,
    },

    /// Parallel processing and concurrency errors
    #[error("Parallel processing error: {message}")]
    Parallel {
        message: String,
        thread_count: usize,
        context: Option<String>,
    },

    /// Memory allocation and management errors
    #[error("Memory error: {message}")]
    Memory {
        message: String,
        requested_bytes: Option<usize>,
        available_bytes: Option<usize>,
    },

    /// Performance and optimization errors
    #[error("Performance error: {message}")]
    Performance {
        message: String,
        metric: String,
        threshold: f64,
        actual: f64,
    },

    /// FANN compatibility errors
    #[error("FANN compatibility error: {message}")]
    Compatibility {
        message: String,
        fann_version: Option<String>,
        operation: String,
    },
}

/// Network error categories for detailed classification
#[derive(Debug, Clone, PartialEq)]
pub enum NetworkErrorCategory {
    /// Invalid network topology or structure
    Topology,
    /// Weight and bias configuration issues
    Weights,
    /// Layer configuration problems
    Layers,
    /// Neuron connection issues
    Connections,
    /// Activation function problems
    Activation,
    /// Forward propagation errors
    Propagation,
}

/// Training error categories
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingErrorCategory {
    /// Learning algorithm failures
    Algorithm,
    /// Convergence problems
    Convergence,
    /// Gradient calculation issues
    Gradients,
    /// Learning rate problems
    LearningRate,
    /// Epoch and iteration errors
    Iteration,
    /// Stop criteria issues
    StopCriteria,
}

/// Cascade correlation error categories
#[derive(Debug, Clone, PartialEq)]
pub enum CascadeErrorCategory {
    /// Candidate neuron generation issues
    CandidateGeneration,
    /// Candidate training failures
    CandidateTraining,
    /// Candidate selection problems
    CandidateSelection,
    /// Network topology modification errors
    TopologyModification,
    /// Correlation calculation issues
    CorrelationCalculation,
    /// Output training problems
    OutputTraining,
}

/// Validation error categories
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationErrorCategory {
    /// Input data validation
    InputData,
    /// Output data validation
    OutputData,
    /// Network configuration validation
    NetworkConfig,
    /// Training parameter validation
    TrainingParams,
    /// Cascade parameter validation
    CascadeParams,
}

/// I/O error categories
#[derive(Debug, Clone, PartialEq)]
pub enum IoErrorCategory {
    /// File reading/writing issues
    FileAccess,
    /// Serialization/deserialization problems
    Serialization,
    /// Format compatibility issues
    Format,
    /// Network export/import errors
    NetworkIo,
    /// Training data I/O problems
    DataIo,
}

/// Comprehensive error category enum for uniform handling
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCategory {
    Network(NetworkErrorCategory),
    Training(TrainingErrorCategory),
    Cascade(CascadeErrorCategory),
    Validation(ValidationErrorCategory),
    Io(IoErrorCategory),
    Parallel,
    Memory,
    Performance,
    Compatibility,
}

/// Validation error for detailed parameter checking
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Parameter out of range: {parameter} = {value}, expected {min} <= value <= {max}")]
    OutOfRange {
        parameter: String,
        value: f64,
        min: f64,
        max: f64,
    },

    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },

    #[error("Missing required parameter: {parameter}")]
    MissingParameter { parameter: String },

    #[error("Incompatible parameters: {message}")]
    IncompatibleParams { message: String },

    #[error("Data format error: {message}")]
    DataFormat { message: String },
}

/// Error context for providing additional debugging information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub network_id: Option<String>,
    pub layer_index: Option<usize>,
    pub neuron_index: Option<usize>,
    pub epoch: Option<usize>,
    pub timestamp: std::time::SystemTime,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            network_id: None,
            layer_index: None,
            neuron_index: None,
            epoch: None,
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        }
    }

    pub fn with_network_id(mut self, id: impl Into<String>) -> Self {
        self.network_id = Some(id.into());
        self
    }

    pub fn with_layer(mut self, index: usize) -> Self {
        self.layer_index = Some(index);
        self
    }

    pub fn with_neuron(mut self, index: usize) -> Self {
        self.neuron_index = Some(index);
        self
    }

    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.epoch = Some(epoch);
        self
    }

    pub fn with_info(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.additional_info.insert(key.into(), value.into());
        self
    }
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry the operation with the same parameters
    Retry,
    /// Retry with modified parameters
    RetryWithModification(std::collections::HashMap<String, String>),
    /// Reset to a known good state
    Reset,
    /// Skip the problematic operation
    Skip,
    /// Abort the entire process
    Abort,
    /// Use fallback implementation
    Fallback(String),
}

/// Error recovery context
#[derive(Debug)]
pub struct RecoveryContext {
    pub strategy: RecoveryStrategy,
    pub max_retries: usize,
    pub current_retry: usize,
    pub fallback_available: bool,
    pub checkpoints: Vec<String>,
}

impl RecoveryContext {
    pub fn new(strategy: RecoveryStrategy) -> Self {
        Self {
            strategy,
            max_retries: 3,
            current_retry: 0,
            fallback_available: false,
            checkpoints: Vec::new(),
        }
    }

    pub fn should_retry(&self) -> bool {
        self.current_retry < self.max_retries
    }

    pub fn increment_retry(&mut self) {
        self.current_retry += 1;
    }

    pub fn reset_retry_count(&mut self) {
        self.current_retry = 0;
    }
}

/// Professional error logging and debugging facilities
pub struct ErrorLogger {
    log_level: log::Level,
    structured_logging: bool,
    performance_tracking: bool,
}

impl ErrorLogger {
    pub fn new() -> Self {
        Self {
            log_level: log::Level::Warn,
            structured_logging: true,
            performance_tracking: false,
        }
    }

    pub fn with_level(mut self, level: log::Level) -> Self {
        self.log_level = level;
        self
    }

    pub fn with_structured_logging(mut self, enabled: bool) -> Self {
        self.structured_logging = enabled;
        self
    }

    pub fn with_performance_tracking(mut self, enabled: bool) -> Self {
        self.performance_tracking = enabled;
        self
    }

    pub fn log_error(&self, error: &RuvFannError, context: Option<&ErrorContext>) {
        if self.structured_logging {
            self.log_structured_error(error, context);
        } else {
            self.log_simple_error(error, context);
        }
    }

    fn log_structured_error(&self, error: &RuvFannError, context: Option<&ErrorContext>) {
        let mut fields = serde_json::Map::new();
        fields.insert("error_type".to_string(), serde_json::Value::String(format!("{:?}", error)));
        fields.insert("message".to_string(), serde_json::Value::String(error.to_string()));

        if let Some(ctx) = context {
            fields.insert("operation".to_string(), serde_json::Value::String(ctx.operation.clone()));
            if let Some(ref network_id) = ctx.network_id {
                fields.insert("network_id".to_string(), serde_json::Value::String(network_id.clone()));
            }
            if let Some(layer_idx) = ctx.layer_index {
                fields.insert("layer_index".to_string(), serde_json::Value::Number(serde_json::Number::from(layer_idx)));
            }
            if let Some(neuron_idx) = ctx.neuron_index {
                fields.insert("neuron_index".to_string(), serde_json::Value::Number(serde_json::Number::from(neuron_idx)));
            }
            if let Some(epoch) = ctx.epoch {
                fields.insert("epoch".to_string(), serde_json::Value::Number(serde_json::Number::from(epoch)));
            }
        }

        #[cfg(feature = "logging")]
        log::log!(self.log_level, "{}", serde_json::Value::Object(fields));
    }

    fn log_simple_error(&self, error: &RuvFannError, context: Option<&ErrorContext>) {
        let context_str = context
            .map(|c| format!(" [{}]", c.operation))
            .unwrap_or_default();

        #[cfg(feature = "logging")]
        log::log!(self.log_level, "Error{}: {}", context_str, error);
    }
}

impl Default for ErrorLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert from legacy error types for backward compatibility
impl From<NetworkError> for RuvFannError {
    fn from(error: NetworkError) -> Self {
        match error {
            NetworkError::InputSizeMismatch { expected, actual } => {
                RuvFannError::Network {
                    category: NetworkErrorCategory::Topology,
                    message: format!("Input size mismatch: expected {}, got {}", expected, actual),
                    context: None,
                }
            }
            NetworkError::WeightCountMismatch { expected, actual } => {
                RuvFannError::Network {
                    category: NetworkErrorCategory::Weights,
                    message: format!("Weight count mismatch: expected {}, got {}", expected, actual),
                    context: None,
                }
            }
            NetworkError::InvalidLayerConfiguration => {
                RuvFannError::Network {
                    category: NetworkErrorCategory::Layers,
                    message: "Invalid layer configuration".to_string(),
                    context: None,
                }
            }
            NetworkError::NoLayers => {
                RuvFannError::Network {
                    category: NetworkErrorCategory::Topology,
                    message: "Network has no layers".to_string(),
                    context: None,
                }
            }
        }
    }
}

impl From<TrainingError> for RuvFannError {
    fn from(error: TrainingError) -> Self {
        match error {
            TrainingError::InvalidData(msg) => {
                RuvFannError::Validation {
                    category: ValidationErrorCategory::InputData,
                    message: msg,
                    details: vec![],
                }
            }
            TrainingError::NetworkError(msg) => {
                RuvFannError::Network {
                    category: NetworkErrorCategory::Topology,
                    message: msg,
                    context: None,
                }
            }
            TrainingError::TrainingFailed(msg) => {
                RuvFannError::Training {
                    category: TrainingErrorCategory::Algorithm,
                    message: msg,
                    context: None,
                }
            }
        }
    }
}

/// Helper macros for error creation with context
#[macro_export]
macro_rules! network_error {
    ($category:expr, $msg:expr) => {
        RuvFannError::Network {
            category: $category,
            message: $msg.to_string(),
            context: None,
        }
    };
    ($category:expr, $msg:expr, $context:expr) => {
        RuvFannError::Network {
            category: $category,
            message: $msg.to_string(),
            context: Some($context.to_string()),
        }
    };
}

#[macro_export]
macro_rules! training_error {
    ($category:expr, $msg:expr) => {
        RuvFannError::Training {
            category: $category,
            message: $msg.to_string(),
            context: None,
        }
    };
    ($category:expr, $msg:expr, $context:expr) => {
        RuvFannError::Training {
            category: $category,
            message: $msg.to_string(),
            context: Some($context.to_string()),
        }
    };
}

#[macro_export]
macro_rules! cascade_error {
    ($category:expr, $msg:expr) => {
        RuvFannError::Cascade {
            category: $category,
            message: $msg.to_string(),
            context: None,
        }
    };
    ($category:expr, $msg:expr, $context:expr) => {
        RuvFannError::Cascade {
            category: $category,
            message: $msg.to_string(),
            context: Some($context.to_string()),
        }
    };
}

/// Comprehensive result type for all ruv-FANN operations
pub type RuvFannResult<T> = Result<T, RuvFannError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = RuvFannError::Network {
            category: NetworkErrorCategory::Topology,
            message: "Test error".to_string(),
            context: None,
        };

        assert!(matches!(error, RuvFannError::Network { .. }));
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_operation")
            .with_network_id("network_1")
            .with_layer(2)
            .with_epoch(100);

        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.network_id, Some("network_1".to_string()));
        assert_eq!(context.layer_index, Some(2));
        assert_eq!(context.epoch, Some(100));
    }

    #[test]
    fn test_recovery_context() {
        let mut recovery = RecoveryContext::new(RecoveryStrategy::Retry);
        assert!(recovery.should_retry());

        recovery.max_retries = 2;
        recovery.current_retry = 2;
        assert!(!recovery.should_retry());
    }

    #[test]
    fn test_error_conversion() {
        let network_error = NetworkError::NoLayers;
        let ruv_error: RuvFannError = network_error.into();

        match ruv_error {
            RuvFannError::Network { category, .. } => {
                assert_eq!(category, NetworkErrorCategory::Topology);
            }
            _ => panic!("Expected Network error"),
        }
    }
}