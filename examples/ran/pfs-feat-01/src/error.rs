use thiserror::Error;

/// Feature engineering error types
#[derive(Error, Debug)]
pub enum FeatureEngineError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Polars error: {0}")]
    Polars(#[from] polars::error::PolarsError),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Data processing error: {0}")]
    DataProcessing(String),
    
    #[error("Feature generation error: {0}")]
    FeatureGeneration(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Time parsing error: {0}")]
    TimeParsing(#[from] chrono::ParseError),
    
    #[error("Missing column: {0}")]
    MissingColumn(String),
    
    #[error("Invalid feature configuration: {0}")]
    InvalidFeatureConfig(String),
    
    #[error("Memory limit exceeded: {current_mb}MB > {limit_mb}MB")]
    MemoryLimitExceeded { current_mb: u64, limit_mb: u64 },
    
    #[error("Timeout error: operation took longer than {timeout_seconds}s")]
    Timeout { timeout_seconds: u64 },
    
    #[error("Concurrent processing error: {0}")]
    ConcurrentProcessing(String),
    
    #[error("Schema validation error: {0}")]
    SchemaValidation(String),
    
    #[error("File format error: {0}")]
    FileFormat(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl FeatureEngineError {
    /// Create a configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
    
    /// Create a validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }
    
    /// Create a data processing error
    pub fn data_processing(msg: impl Into<String>) -> Self {
        Self::DataProcessing(msg.into())
    }
    
    /// Create a feature generation error
    pub fn feature_generation(msg: impl Into<String>) -> Self {
        Self::FeatureGeneration(msg.into())
    }
    
    /// Create a missing column error
    pub fn missing_column(column: impl Into<String>) -> Self {
        Self::MissingColumn(column.into())
    }
    
    /// Create an invalid feature configuration error
    pub fn invalid_feature_config(msg: impl Into<String>) -> Self {
        Self::InvalidFeatureConfig(msg.into())
    }
    
    /// Create a memory limit exceeded error
    pub fn memory_limit_exceeded(current_mb: u64, limit_mb: u64) -> Self {
        Self::MemoryLimitExceeded { current_mb, limit_mb }
    }
    
    /// Create a timeout error
    pub fn timeout(timeout_seconds: u64) -> Self {
        Self::Timeout { timeout_seconds }
    }
    
    /// Create a concurrent processing error
    pub fn concurrent_processing(msg: impl Into<String>) -> Self {
        Self::ConcurrentProcessing(msg.into())
    }
    
    /// Create a schema validation error
    pub fn schema_validation(msg: impl Into<String>) -> Self {
        Self::SchemaValidation(msg.into())
    }
    
    /// Create a file format error
    pub fn file_format(msg: impl Into<String>) -> Self {
        Self::FileFormat(msg.into())
    }
    
    /// Create an internal error
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}

/// Result type for feature engineering operations
pub type FeatureEngineResult<T> = Result<T, FeatureEngineError>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let err = FeatureEngineError::config("test config error");
        assert!(matches!(err, FeatureEngineError::Config(_)));
        assert_eq!(err.to_string(), "Configuration error: test config error");
    }
    
    #[test]
    fn test_memory_limit_error() {
        let err = FeatureEngineError::memory_limit_exceeded(1024, 512);
        assert!(matches!(err, FeatureEngineError::MemoryLimitExceeded { .. }));
        assert_eq!(err.to_string(), "Memory limit exceeded: 1024MB > 512MB");
    }
    
    #[test]
    fn test_timeout_error() {
        let err = FeatureEngineError::timeout(300);
        assert!(matches!(err, FeatureEngineError::Timeout { .. }));
        assert_eq!(err.to_string(), "Timeout error: operation took longer than 300s");
    }
}