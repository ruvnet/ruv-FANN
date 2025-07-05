//! Error handling for the DNI-CAP-01 Capacity Cliff Forecaster

use thiserror::Error;

/// Main error type for the capacity planning system
#[derive(Error, Debug)]
pub enum CapacityPlanningError {
    /// Data-related errors
    #[error("Data error: {0}")]
    Data(#[from] DataError),

    /// Model-related errors
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    /// Forecasting errors
    #[error("Forecasting error: {0}")]
    Forecasting(#[from] ForecastingError),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(#[from] ConfigurationError),

    /// Service errors
    #[error("Service error: {0}")]
    Service(#[from] ServiceError),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Database errors
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    /// General errors
    #[error("General error: {0}")]
    General(#[from] anyhow::Error),
}

/// Data-related errors
#[derive(Error, Debug)]
pub enum DataError {
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    #[error("Invalid data format: {message}")]
    InvalidFormat { message: String },

    #[error("Data validation failed: {field} is {issue}")]
    ValidationFailed { field: String, issue: String },

    #[error("Missing required field: {field}")]
    MissingField { field: String },

    #[error("Data range error: {message}")]
    RangeError { message: String },

    #[error("Duplicate data point: timestamp {timestamp}")]
    DuplicateDataPoint { timestamp: String },

    #[error("Data gaps detected: {gaps} missing intervals")]
    DataGaps { gaps: usize },

    #[error("Data quality issue: {metric} = {value}, threshold = {threshold}")]
    QualityIssue {
        metric: String,
        value: f64,
        threshold: f64,
    },
}

/// Model-related errors
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Model not found: {name}")]
    NotFound { name: String },

    #[error("Model not trained: {name}")]
    NotTrained { name: String },

    #[error("Model training failed: {reason}")]
    TrainingFailed { reason: String },

    #[error("Model prediction failed: {reason}")]
    PredictionFailed { reason: String },

    #[error("Model validation failed: {metric} = {value}, threshold = {threshold}")]
    ValidationFailed {
        metric: String,
        value: f64,
        threshold: f64,
    },

    #[error("Model parameters invalid: {parameter} = {value}")]
    InvalidParameters { parameter: String, value: String },

    #[error("Model convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },

    #[error("Model overfitting detected: validation score {validation_score} < training score {training_score}")]
    OverfittingDetected {
        validation_score: f64,
        training_score: f64,
    },

    #[error("Model underfitting detected: score {score} < minimum threshold {threshold}")]
    UnderfittingDetected { score: f64, threshold: f64 },
}

/// Forecasting-related errors
#[derive(Error, Debug)]
pub enum ForecastingError {
    #[error("Forecast horizon too large: {requested} months, maximum is {maximum}")]
    HorizonTooLarge { requested: usize, maximum: usize },

    #[error("Forecast accuracy below threshold: {actual:.2}% < {threshold:.2}%")]
    AccuracyBelowThreshold { actual: f64, threshold: f64 },

    #[error("Forecast confidence too low: {confidence:.2} < {minimum:.2}")]
    ConfidenceTooLow { confidence: f64, minimum: f64 },

    #[error("Seasonal decomposition failed: {reason}")]
    SeasonalDecompositionFailed { reason: String },

    #[error("Trend analysis failed: {reason}")]
    TrendAnalysisFailed { reason: String },

    #[error("Ensemble prediction failed: {reason}")]
    EnsemblePredictionFailed { reason: String },

    #[error("Forecast validation failed: {metric} = {value}")]
    ValidationFailed { metric: String, value: f64 },

    #[error("Capacity breach prediction failed: {reason}")]
    BreachPredictionFailed { reason: String },
}

/// Configuration-related errors
#[derive(Error, Debug)]
pub enum ConfigurationError {
    #[error("Configuration file not found: {path}")]
    FileNotFound { path: String },

    #[error("Configuration parse error: {message}")]
    ParseError { message: String },

    #[error("Invalid configuration value: {key} = {value}")]
    InvalidValue { key: String, value: String },

    #[error("Missing required configuration: {key}")]
    MissingRequired { key: String },

    #[error("Configuration validation failed: {message}")]
    ValidationFailed { message: String },
}

/// Service-related errors
#[derive(Error, Debug)]
pub enum ServiceError {
    #[error("Service unavailable: {service}")]
    Unavailable { service: String },

    #[error("Service initialization failed: {reason}")]
    InitializationFailed { reason: String },

    #[error("Service request failed: {reason}")]
    RequestFailed { reason: String },

    #[error("Service timeout: {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("Service authentication failed")]
    AuthenticationFailed,

    #[error("Service rate limit exceeded")]
    RateLimitExceeded,

    #[error("Service internal error: {message}")]
    InternalError { message: String },
}

/// Result type alias for capacity planning operations
pub type CapacityResult<T> = Result<T, CapacityPlanningError>;

/// Result type alias for data operations
pub type DataResult<T> = Result<T, DataError>;

/// Result type alias for model operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Result type alias for forecasting operations
pub type ForecastingResult<T> = Result<T, ForecastingError>;

/// Result type alias for configuration operations
pub type ConfigurationResult<T> = Result<T, ConfigurationError>;

/// Result type alias for service operations
pub type ServiceResult<T> = Result<T, ServiceError>;

impl DataError {
    /// Create an insufficient data error
    pub fn insufficient_data(required: usize, actual: usize) -> Self {
        Self::InsufficientData { required, actual }
    }

    /// Create an invalid format error
    pub fn invalid_format(message: impl Into<String>) -> Self {
        Self::InvalidFormat {
            message: message.into(),
        }
    }

    /// Create a validation failed error
    pub fn validation_failed(field: impl Into<String>, issue: impl Into<String>) -> Self {
        Self::ValidationFailed {
            field: field.into(),
            issue: issue.into(),
        }
    }

    /// Create a missing field error
    pub fn missing_field(field: impl Into<String>) -> Self {
        Self::MissingField {
            field: field.into(),
        }
    }

    /// Create a quality issue error
    pub fn quality_issue(metric: impl Into<String>, value: f64, threshold: f64) -> Self {
        Self::QualityIssue {
            metric: metric.into(),
            value,
            threshold,
        }
    }
}

impl ModelError {
    /// Create a not found error
    pub fn not_found(name: impl Into<String>) -> Self {
        Self::NotFound { name: name.into() }
    }

    /// Create a not trained error
    pub fn not_trained(name: impl Into<String>) -> Self {
        Self::NotTrained { name: name.into() }
    }

    /// Create a training failed error
    pub fn training_failed(reason: impl Into<String>) -> Self {
        Self::TrainingFailed {
            reason: reason.into(),
        }
    }

    /// Create a prediction failed error
    pub fn prediction_failed(reason: impl Into<String>) -> Self {
        Self::PredictionFailed {
            reason: reason.into(),
        }
    }

    /// Create a validation failed error
    pub fn validation_failed(metric: impl Into<String>, value: f64, threshold: f64) -> Self {
        Self::ValidationFailed {
            metric: metric.into(),
            value,
            threshold,
        }
    }
}

impl ForecastingError {
    /// Create a horizon too large error
    pub fn horizon_too_large(requested: usize, maximum: usize) -> Self {
        Self::HorizonTooLarge { requested, maximum }
    }

    /// Create an accuracy below threshold error
    pub fn accuracy_below_threshold(actual: f64, threshold: f64) -> Self {
        Self::AccuracyBelowThreshold { actual, threshold }
    }

    /// Create a confidence too low error
    pub fn confidence_too_low(confidence: f64, minimum: f64) -> Self {
        Self::ConfidenceTooLow { confidence, minimum }
    }

    /// Create a breach prediction failed error
    pub fn breach_prediction_failed(reason: impl Into<String>) -> Self {
        Self::BreachPredictionFailed {
            reason: reason.into(),
        }
    }
}

impl ConfigurationError {
    /// Create a file not found error
    pub fn file_not_found(path: impl Into<String>) -> Self {
        Self::FileNotFound { path: path.into() }
    }

    /// Create a parse error
    pub fn parse_error(message: impl Into<String>) -> Self {
        Self::ParseError {
            message: message.into(),
        }
    }

    /// Create an invalid value error
    pub fn invalid_value(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self::InvalidValue {
            key: key.into(),
            value: value.into(),
        }
    }

    /// Create a missing required error
    pub fn missing_required(key: impl Into<String>) -> Self {
        Self::MissingRequired { key: key.into() }
    }
}

impl ServiceError {
    /// Create a service unavailable error
    pub fn unavailable(service: impl Into<String>) -> Self {
        Self::Unavailable {
            service: service.into(),
        }
    }

    /// Create an initialization failed error
    pub fn initialization_failed(reason: impl Into<String>) -> Self {
        Self::InitializationFailed {
            reason: reason.into(),
        }
    }

    /// Create a request failed error
    pub fn request_failed(reason: impl Into<String>) -> Self {
        Self::RequestFailed {
            reason: reason.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout { timeout_ms }
    }

    /// Create an internal error
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::InternalError {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_error_creation() {
        let error = DataError::insufficient_data(100, 50);
        assert!(error.to_string().contains("50"));
        assert!(error.to_string().contains("100"));
    }

    #[test]
    fn test_model_error_creation() {
        let error = ModelError::not_found("LSTM");
        assert!(error.to_string().contains("LSTM"));
    }

    #[test]
    fn test_forecasting_error_creation() {
        let error = ForecastingError::horizon_too_large(36, 24);
        assert!(error.to_string().contains("36"));
        assert!(error.to_string().contains("24"));
    }

    #[test]
    fn test_configuration_error_creation() {
        let error = ConfigurationError::file_not_found("/path/to/config.toml");
        assert!(error.to_string().contains("config.toml"));
    }

    #[test]
    fn test_service_error_creation() {
        let error = ServiceError::timeout(5000);
        assert!(error.to_string().contains("5000"));
    }

    #[test]
    fn test_error_conversion() {
        let data_error = DataError::insufficient_data(100, 50);
        let capacity_error: CapacityPlanningError = data_error.into();
        assert!(capacity_error.to_string().contains("Data error"));
    }
}