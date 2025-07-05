// Error handling for RAN Intelligence Platform
use thiserror::Error;

pub type RanResult<T> = Result<T, RanError>;

#[derive(Error, Debug)]
pub enum RanError {
    // Service errors
    #[error("Service unavailable: {service}")]
    ServiceUnavailable { service: String },
    
    #[error("Service configuration error: {message}")]
    Configuration { message: String },
    
    #[error("Service initialization failed: {message}")]
    Initialization { message: String },

    // Data processing errors
    #[error("Data ingestion failed: {message}")]
    DataIngestion { message: String },
    
    #[error("Data validation failed: {message}")]
    DataValidation { message: String },
    
    #[error("Feature engineering failed: {message}")]
    FeatureEngineering { message: String },
    
    #[error("Data format error: {message}")]
    DataFormat { message: String },

    // ML model errors
    #[error("Model training failed: {message}")]
    ModelTraining { message: String },
    
    #[error("Model prediction failed: {message}")]
    ModelPrediction { message: String },
    
    #[error("Model not found: {model_id}")]
    ModelNotFound { model_id: String },
    
    #[error("Model registry error: {message}")]
    ModelRegistry { message: String },
    
    #[error("Invalid model configuration: {message}")]
    InvalidModelConfig { message: String },

    // Network/gRPC errors
    #[error("gRPC transport error: {message}")]
    Transport { message: String },
    
    #[error("Request timeout: {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
    
    #[error("Authentication failed: {message}")]
    Authentication { message: String },
    
    #[error("Authorization failed: {message}")]
    Authorization { message: String },

    // Database errors
    #[error("Database connection failed: {message}")]
    DatabaseConnection { message: String },
    
    #[error("Database query failed: {message}")]
    DatabaseQuery { message: String },
    
    #[error("Database transaction failed: {message}")]
    DatabaseTransaction { message: String },

    // File system errors
    #[error("File not found: {path}")]
    FileNotFound { path: String },
    
    #[error("File access denied: {path}")]
    FileAccessDenied { path: String },
    
    #[error("File I/O error: {message}")]
    FileIo { message: String },

    // Business logic errors
    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },
    
    #[error("Resource conflict: {message}")]
    ResourceConflict { message: String },
    
    #[error("Resource not found: {resource}")]
    ResourceNotFound { resource: String },
    
    #[error("Capacity limit exceeded: {limit}")]
    CapacityExceeded { limit: String },

    // External service errors
    #[error("External service error: {service}: {message}")]
    ExternalService { service: String, message: String },

    // Internal errors
    #[error("Internal error: {message}")]
    Internal { message: String },
    
    #[error("Not implemented: {feature}")]
    NotImplemented { feature: String },
}

impl RanError {
    pub fn error_code(&self) -> &str {
        match self {
            // Service errors
            RanError::ServiceUnavailable { .. } => "SERVICE_UNAVAILABLE",
            RanError::Configuration { .. } => "CONFIGURATION_ERROR",
            RanError::Initialization { .. } => "INITIALIZATION_ERROR",

            // Data processing errors
            RanError::DataIngestion { .. } => "DATA_INGESTION_ERROR",
            RanError::DataValidation { .. } => "DATA_VALIDATION_ERROR",
            RanError::FeatureEngineering { .. } => "FEATURE_ENGINEERING_ERROR",
            RanError::DataFormat { .. } => "DATA_FORMAT_ERROR",

            // ML model errors
            RanError::ModelTraining { .. } => "MODEL_TRAINING_ERROR",
            RanError::ModelPrediction { .. } => "MODEL_PREDICTION_ERROR",
            RanError::ModelNotFound { .. } => "MODEL_NOT_FOUND",
            RanError::ModelRegistry { .. } => "MODEL_REGISTRY_ERROR",
            RanError::InvalidModelConfig { .. } => "INVALID_MODEL_CONFIG",

            // Network/gRPC errors
            RanError::Transport { .. } => "TRANSPORT_ERROR",
            RanError::Timeout { .. } => "TIMEOUT_ERROR",
            RanError::Authentication { .. } => "AUTHENTICATION_ERROR",
            RanError::Authorization { .. } => "AUTHORIZATION_ERROR",

            // Database errors
            RanError::DatabaseConnection { .. } => "DATABASE_CONNECTION_ERROR",
            RanError::DatabaseQuery { .. } => "DATABASE_QUERY_ERROR",
            RanError::DatabaseTransaction { .. } => "DATABASE_TRANSACTION_ERROR",

            // File system errors
            RanError::FileNotFound { .. } => "FILE_NOT_FOUND",
            RanError::FileAccessDenied { .. } => "FILE_ACCESS_DENIED",
            RanError::FileIo { .. } => "FILE_IO_ERROR",

            // Business logic errors
            RanError::InvalidRequest { .. } => "INVALID_REQUEST",
            RanError::ResourceConflict { .. } => "RESOURCE_CONFLICT",
            RanError::ResourceNotFound { .. } => "RESOURCE_NOT_FOUND",
            RanError::CapacityExceeded { .. } => "CAPACITY_EXCEEDED",

            // External service errors
            RanError::ExternalService { .. } => "EXTERNAL_SERVICE_ERROR",

            // Internal errors
            RanError::Internal { .. } => "INTERNAL_ERROR",
            RanError::NotImplemented { .. } => "NOT_IMPLEMENTED",
        }
    }

    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            RanError::ServiceUnavailable { .. }
                | RanError::Transport { .. }
                | RanError::Timeout { .. }
                | RanError::DatabaseConnection { .. }
                | RanError::ExternalService { .. }
        )
    }

    pub fn to_grpc_status(&self) -> tonic::Status {
        let code = match self {
            RanError::ServiceUnavailable { .. } => tonic::Code::Unavailable,
            RanError::Configuration { .. } => tonic::Code::FailedPrecondition,
            RanError::Initialization { .. } => tonic::Code::FailedPrecondition,
            RanError::DataValidation { .. } => tonic::Code::InvalidArgument,
            RanError::InvalidRequest { .. } => tonic::Code::InvalidArgument,
            RanError::InvalidModelConfig { .. } => tonic::Code::InvalidArgument,
            RanError::ModelNotFound { .. } => tonic::Code::NotFound,
            RanError::ResourceNotFound { .. } => tonic::Code::NotFound,
            RanError::FileNotFound { .. } => tonic::Code::NotFound,
            RanError::Authentication { .. } => tonic::Code::Unauthenticated,
            RanError::Authorization { .. } => tonic::Code::PermissionDenied,
            RanError::FileAccessDenied { .. } => tonic::Code::PermissionDenied,
            RanError::ResourceConflict { .. } => tonic::Code::AlreadyExists,
            RanError::CapacityExceeded { .. } => tonic::Code::ResourceExhausted,
            RanError::Timeout { .. } => tonic::Code::DeadlineExceeded,
            RanError::NotImplemented { .. } => tonic::Code::Unimplemented,
            _ => tonic::Code::Internal,
        };

        tonic::Status::new(code, self.to_string())
    }
}

// Conversion from common error types
impl From<std::io::Error> for RanError {
    fn from(err: std::io::Error) -> Self {
        RanError::FileIo {
            message: err.to_string(),
        }
    }
}

impl From<serde_json::Error> for RanError {
    fn from(err: serde_json::Error) -> Self {
        RanError::DataFormat {
            message: err.to_string(),
        }
    }
}

impl From<tonic::Status> for RanError {
    fn from(status: tonic::Status) -> Self {
        RanError::Transport {
            message: status.message().to_string(),
        }
    }
}

impl From<tonic::transport::Error> for RanError {
    fn from(err: tonic::transport::Error) -> Self {
        RanError::Transport {
            message: err.to_string(),
        }
    }
}

impl From<config::ConfigError> for RanError {
    fn from(err: config::ConfigError) -> Self {
        RanError::Configuration {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        let error = RanError::ModelNotFound {
            model_id: "test-model".to_string(),
        };
        assert_eq!(error.error_code(), "MODEL_NOT_FOUND");
    }

    #[test]
    fn test_retryable_errors() {
        let retryable = RanError::ServiceUnavailable {
            service: "test".to_string(),
        };
        assert!(retryable.is_retryable());

        let non_retryable = RanError::InvalidRequest {
            message: "bad request".to_string(),
        };
        assert!(!non_retryable.is_retryable());
    }

    #[test]
    fn test_grpc_status_conversion() {
        let error = RanError::ModelNotFound {
            model_id: "test-model".to_string(),
        };
        let status = error.to_grpc_status();
        assert_eq!(status.code(), tonic::Code::NotFound);
    }
}