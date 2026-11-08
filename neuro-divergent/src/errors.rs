//! Error types and handling for the neuro-divergent library

use thiserror::Error;
use polars::error::PolarsError;
use ruv_fann::errors::RuvFannError;

/// Result type alias for neuro-divergent operations
pub type NeuroDivergentResult<T> = Result<T, NeuroDivergentError>;

/// Main error type for the neuro-divergent library
#[derive(Error, Debug)]
pub enum NeuroDivergentError {
    /// Configuration errors
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    /// Data validation errors
    #[error("Data validation error: {0}")]
    DataError(String),
    
    /// Model training errors
    #[error("Training error: {0}")]
    TrainingError(String),
    
    /// Model prediction errors
    #[error("Prediction error: {0}")]
    PredictionError(String),
    
    /// I/O errors (file operations, serialization, etc.)
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// Network architecture errors
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// Mathematical computation errors
    #[error("Math error: {0}")]
    MathError(String),
    
    /// Time series specific errors
    #[error("Time series error: {0}")]
    TimeSeriesError(String),
    
    /// Integration errors with ruv-FANN
    #[error("FANN integration error: {0}")]
    FannError(String),
    
    /// GPU/hardware acceleration errors
    #[cfg(feature = "gpu")]
    #[error("GPU error: {0}")]
    GpuError(String),
    
    /// Async operation errors
    #[cfg(feature = "async")]
    #[error("Async error: {0}")]
    AsyncError(String),
    
    /// Generic errors with context
    #[error("Error: {message}")]
    Generic { message: String },
    
    /// Multiple errors combined
    #[error("Multiple errors occurred: {errors:?}")]
    Multiple { errors: Vec<NeuroDivergentError> },
}

impl NeuroDivergentError {
    /// Create a configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::ConfigError(message.into())
    }
    
    /// Create a data validation error
    pub fn data<S: Into<String>>(message: S) -> Self {
        Self::DataError(message.into())
    }
    
    /// Create a training error
    pub fn training<S: Into<String>>(message: S) -> Self {
        Self::TrainingError(message.into())
    }
    
    /// Create a prediction error
    pub fn prediction<S: Into<String>>(message: S) -> Self {
        Self::PredictionError(message.into())
    }
    
    /// Create a network error
    pub fn network<S: Into<String>>(message: S) -> Self {
        Self::NetworkError(message.into())
    }
    
    /// Create a time series error
    pub fn time_series<S: Into<String>>(message: S) -> Self {
        Self::TimeSeriesError(message.into())
    }
    
    /// Create a FANN integration error
    pub fn fann<S: Into<String>>(message: S) -> Self {
        Self::FannError(message.into())
    }
    
    /// Create a mathematical computation error
    pub fn math<S: Into<String>>(message: S) -> Self {
        Self::MathError(message.into())
    }
    
    /// Create a serialization or checkpoint error.
    pub fn serialization<S: Into<String>>(message: S) -> Self {
        Self::SerializationError(message.into())
    }

    /// Create a generic error
    pub fn generic<S: Into<String>>(message: S) -> Self {
        Self::Generic { 
            message: message.into() 
        }
    }
    
    /// Combine multiple errors
    pub fn multiple(errors: Vec<NeuroDivergentError>) -> Self {
        Self::Multiple { errors }
    }
    
    /// Check if this is a configuration error
    pub fn is_config_error(&self) -> bool {
        matches!(self, Self::ConfigError(_))
    }
    
    /// Check if this is a data error
    pub fn is_data_error(&self) -> bool {
        matches!(self, Self::DataError(_))
    }
    
    /// Check if this is a training error
    pub fn is_training_error(&self) -> bool {
        matches!(self, Self::TrainingError(_))
    }
    
    /// Check if this is a prediction error
    pub fn is_prediction_error(&self) -> bool {
        matches!(self, Self::PredictionError(_))
    }
    
    /// Get the error category as a string
    pub fn category(&self) -> &'static str {
        match self {
            Self::ConfigError(_) => "Configuration",
            Self::DataError(_) => "Data",
            Self::TrainingError(_) => "Training",
            Self::PredictionError(_) => "Prediction",
            Self::IoError(_) => "I/O",
            Self::SerializationError(_) => "Serialization",
            Self::NetworkError(_) => "Network",
            Self::MathError(_) => "Math",
            Self::TimeSeriesError(_) => "TimeSeries",
            Self::FannError(_) => "FANN",
            #[cfg(feature = "gpu")]
            Self::GpuError(_) => "GPU",
            #[cfg(feature = "async")]
            Self::AsyncError(_) => "Async",
            Self::Generic { .. } => "Generic",
            Self::Multiple { .. } => "Multiple",
        }
    }
}

// Integration with polars errors
impl From<PolarsError> for NeuroDivergentError {
    fn from(err: PolarsError) -> Self {
        Self::DataError(format!("Polars error: {}", err))
    }
}

// Integration with serde errors
impl From<serde_json::Error> for NeuroDivergentError {
    fn from(err: serde_json::Error) -> Self {
        Self::SerializationError(format!("JSON error: {}", err))
    }
}

// Integration with chrono errors
impl From<chrono::ParseError> for NeuroDivergentError {
    fn from(err: chrono::ParseError) -> Self {
        Self::TimeSeriesError(format!("Date parsing error: {}", err))
    }
}

// Integration with ruv-FANN errors
impl From<RuvFannError> for NeuroDivergentError {
    fn from(err: RuvFannError) -> Self {
        Self::FannError(format!("ruv-FANN error: {}", err))
    }
}

// Integration with ndarray errors
impl From<ndarray::ShapeError> for NeuroDivergentError {
    fn from(err: ndarray::ShapeError) -> Self {
        Self::MathError(format!("Array shape error: {}", err))
    }
}

/// Error context trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add context to an error
    fn with_context<F>(self, f: F) -> NeuroDivergentResult<T>
    where
        F: FnOnce() -> String;
    
    /// Add context with a static message
    fn context(self, message: &'static str) -> NeuroDivergentResult<T>;
}

impl<T> ErrorContext<T> for NeuroDivergentResult<T> {
    fn with_context<F>(self, f: F) -> NeuroDivergentResult<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|err| {
            NeuroDivergentError::Generic {
                message: format!("{}: {}", f(), err),
            }
        })
    }
    
    fn context(self, message: &'static str) -> NeuroDivergentResult<T> {
        self.map_err(|err| {
            NeuroDivergentError::Generic {
                message: format!("{}: {}", message, err),
            }
        })
    }
}

/// Helper macro for creating configuration errors
#[macro_export]
macro_rules! config_error {
    ($($arg:tt)*) => {
        $crate::errors::NeuroDivergentError::config(format!($($arg)*))
    };
}

/// Helper macro for creating data errors
#[macro_export]
macro_rules! data_error {
    ($($arg:tt)*) => {
        $crate::errors::NeuroDivergentError::data(format!($($arg)*))
    };
}

/// Helper macro for creating training errors
#[macro_export]
macro_rules! training_error {
    ($($arg:tt)*) => {
        $crate::errors::NeuroDivergentError::training(format!($($arg)*))
    };
}

/// Helper macro for creating prediction errors
#[macro_export]
macro_rules! prediction_error {
    ($($arg:tt)*) => {
        $crate::errors::NeuroDivergentError::prediction(format!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let err = NeuroDivergentError::config("test config error");
        assert!(err.is_config_error());
        assert_eq!(err.category(), "Configuration");
    }
    
    #[test]
    fn test_error_context() {
        let result: NeuroDivergentResult<()> = Err(NeuroDivergentError::config("original error"));
        let with_context = result.context("additional context");
        
        assert!(with_context.is_err());
        let error_msg = with_context.unwrap_err().to_string();
        assert!(error_msg.contains("additional context"));
        assert!(error_msg.contains("original error"));
    }
    
    #[test]
    fn test_error_macros() {
        let err = config_error!("test {} error", "config");
        assert!(err.is_config_error());
        
        let err = data_error!("test data error");
        assert!(err.is_data_error());
    }
    
    #[test]
    fn test_multiple_errors() {
        let errors = vec![
            NeuroDivergentError::config("config error"),
            NeuroDivergentError::data("data error"),
        ];
        let combined = NeuroDivergentError::multiple(errors);
        assert_eq!(combined.category(), "Multiple");
    }
}