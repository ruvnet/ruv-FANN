//! Error handling for the data ingestion service

use std::fmt;
use thiserror::Error;

pub type IngestionResult<T> = Result<T, IngestionError>;

#[derive(Error, Debug)]
pub enum IngestionError {
    #[error("File I/O error: {0}")]
    FileIo(#[from] std::io::Error),
    
    #[error("CSV parsing error: {0}")]
    CsvParse(#[from] csv::Error),
    
    #[error("JSON parsing error: {0}")]
    JsonParse(#[from] serde_json::Error),
    
    #[error("Parquet writing error: {0}")]
    ParquetWrite(#[from] parquet::errors::ParquetError),
    
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    
    #[error("Configuration error: {message}")]
    Config { message: String },
    
    #[error("Schema validation error: {message}")]
    SchemaValidation { message: String },
    
    #[error("Data validation error: {message}")]
    DataValidation { message: String },
    
    #[error("Error rate exceeded: {current_rate:.4} > {max_rate:.4}")]
    ErrorRateExceeded { current_rate: f64, max_rate: f64 },
    
    #[error("Job not found: {job_id}")]
    JobNotFound { job_id: String },
    
    #[error("Invalid file format: {path}")]
    InvalidFileFormat { path: String },
    
    #[error("Watch error: {0}")]
    Watch(#[from] notify::Error),
    
    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),
    
    #[error("Service error: {message}")]
    Service { message: String },
    
    #[error("Processing timeout for file: {path}")]
    ProcessingTimeout { path: String },
    
    #[error("Resource limit exceeded: {resource}")]
    ResourceLimitExceeded { resource: String },
}

impl IngestionError {
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config { message: message.into() }
    }
    
    pub fn schema_validation(message: impl Into<String>) -> Self {
        Self::SchemaValidation { message: message.into() }
    }
    
    pub fn data_validation(message: impl Into<String>) -> Self {
        Self::DataValidation { message: message.into() }
    }
    
    pub fn service(message: impl Into<String>) -> Self {
        Self::Service { message: message.into() }
    }
    
    pub fn job_not_found(job_id: impl Into<String>) -> Self {
        Self::JobNotFound { job_id: job_id.into() }
    }
    
    pub fn invalid_file_format(path: impl Into<String>) -> Self {
        Self::InvalidFileFormat { path: path.into() }
    }
    
    pub fn processing_timeout(path: impl Into<String>) -> Self {
        Self::ProcessingTimeout { path: path.into() }
    }
    
    pub fn resource_limit_exceeded(resource: impl Into<String>) -> Self {
        Self::ResourceLimitExceeded { resource: resource.into() }
    }
    
    /// Check if the error is recoverable (can retry)
    pub fn is_recoverable(&self) -> bool {
        match self {
            IngestionError::FileIo(_) => true,
            IngestionError::ProcessingTimeout { .. } => true,
            IngestionError::ResourceLimitExceeded { .. } => true,
            IngestionError::Grpc(_) => true,
            _ => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            IngestionError::ErrorRateExceeded { .. } => ErrorSeverity::Critical,
            IngestionError::ResourceLimitExceeded { .. } => ErrorSeverity::High,
            IngestionError::Config { .. } => ErrorSeverity::High,
            IngestionError::SchemaValidation { .. } => ErrorSeverity::Medium,
            IngestionError::DataValidation { .. } => ErrorSeverity::Low,
            IngestionError::CsvParse(_) => ErrorSeverity::Low,
            IngestionError::JsonParse(_) => ErrorSeverity::Low,
            _ => ErrorSeverity::Medium,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "LOW"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::High => write!(f, "HIGH"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Error statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct ErrorStats {
    pub total_errors: u64,
    pub parsing_errors: u64,
    pub validation_errors: u64,
    pub io_errors: u64,
    pub timeout_errors: u64,
    pub recoverable_errors: u64,
    pub by_severity: [u64; 4], // Low, Medium, High, Critical
}

impl ErrorStats {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn record_error(&mut self, error: &IngestionError) {
        self.total_errors += 1;
        
        match error {
            IngestionError::CsvParse(_) | IngestionError::JsonParse(_) => {
                self.parsing_errors += 1;
            }
            IngestionError::DataValidation { .. } | IngestionError::SchemaValidation { .. } => {
                self.validation_errors += 1;
            }
            IngestionError::FileIo(_) => {
                self.io_errors += 1;
            }
            IngestionError::ProcessingTimeout { .. } => {
                self.timeout_errors += 1;
            }
            _ => {}
        }
        
        if error.is_recoverable() {
            self.recoverable_errors += 1;
        }
        
        let severity_index = match error.severity() {
            ErrorSeverity::Low => 0,
            ErrorSeverity::Medium => 1,
            ErrorSeverity::High => 2,
            ErrorSeverity::Critical => 3,
        };
        self.by_severity[severity_index] += 1;
    }
    
    pub fn error_rate(&self, total_processed: u64) -> f64 {
        if total_processed == 0 {
            0.0
        } else {
            self.total_errors as f64 / total_processed as f64
        }
    }
    
    pub fn critical_error_rate(&self, total_processed: u64) -> f64 {
        if total_processed == 0 {
            0.0
        } else {
            self.by_severity[3] as f64 / total_processed as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_stats() {
        let mut stats = ErrorStats::new();
        
        let csv_error = IngestionError::CsvParse(csv::Error::from("test"));
        stats.record_error(&csv_error);
        
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.parsing_errors, 1);
        assert_eq!(stats.by_severity[0], 1); // Low severity
    }
    
    #[test]
    fn test_error_rate_calculation() {
        let mut stats = ErrorStats::new();
        let error = IngestionError::DataValidation { message: "test".to_string() };
        stats.record_error(&error);
        
        assert_eq!(stats.error_rate(100), 0.01);
        assert_eq!(stats.error_rate(0), 0.0);
    }
    
    #[test]
    fn test_error_severity() {
        let config_error = IngestionError::Config { message: "test".to_string() };
        assert_eq!(config_error.severity(), ErrorSeverity::High);
        
        let data_error = IngestionError::DataValidation { message: "test".to_string() };
        assert_eq!(data_error.severity(), ErrorSeverity::Low);
    }
    
    #[test]
    fn test_error_recoverability() {
        let io_error = IngestionError::FileIo(std::io::Error::new(std::io::ErrorKind::Other, "test"));
        assert!(io_error.is_recoverable());
        
        let config_error = IngestionError::Config { message: "test".to_string() };
        assert!(!config_error.is_recoverable());
    }
}