//! Error types for the Model Registry service

use thiserror::Error;

/// Registry-specific error types
#[derive(Error, Debug)]
pub enum RegistryError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model version not found: {0}")]
    VersionNotFound(String),

    #[error("Model already exists: {0}")]
    ModelAlreadyExists(String),

    #[error("Deployment not found: {0}")]
    DeploymentNotFound(String),

    #[error("Invalid model configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Database error: {0}")]
    DatabaseError(#[from] sea_orm::DbErr),

    #[error("Storage error: {0}")]
    StorageError(#[from] anyhow::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Model artifact verification failed")]
    ArtifactVerificationFailed,

    #[error("Model deployment failed: {0}")]
    DeploymentFailed(String),

    #[error("Model retirement failed: {0}")]
    RetirementFailed(String),

    #[error("Insufficient permissions: {0}")]
    PermissionDenied(String),

    #[error("Resource not available: {0}")]
    ResourceUnavailable(String),

    #[error("Operation not supported: {0}")]
    UnsupportedOperation(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Conflict: {0}")]
    Conflict(String),
}

impl RegistryError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            RegistryError::ServiceUnavailable(_) |
            RegistryError::RateLimitExceeded |
            RegistryError::ResourceUnavailable(_)
        )
    }

    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            RegistryError::ModelNotFound(_) |
            RegistryError::VersionNotFound(_) |
            RegistryError::DeploymentNotFound(_) => "not_found",
            
            RegistryError::ModelAlreadyExists(_) |
            RegistryError::Conflict(_) => "conflict",
            
            RegistryError::InvalidConfiguration(_) |
            RegistryError::ValidationError(_) => "validation",
            
            RegistryError::DatabaseError(_) => "database",
            
            RegistryError::StorageError(_) |
            RegistryError::IoError(_) => "storage",
            
            RegistryError::ArtifactVerificationFailed => "integrity",
            
            RegistryError::PermissionDenied(_) => "auth",
            
            RegistryError::DeploymentFailed(_) |
            RegistryError::RetirementFailed(_) => "deployment",
            
            RegistryError::RateLimitExceeded => "rate_limit",
            
            RegistryError::ServiceUnavailable(_) |
            RegistryError::ResourceUnavailable(_) => "unavailable",
            
            RegistryError::UnsupportedOperation(_) => "unsupported",
            
            RegistryError::SerializationError(_) => "serialization",
        }
    }
}

/// Result type for registry operations
pub type RegistryResult<T> = Result<T, RegistryError>;

/// Convert gRPC status to RegistryError
impl From<RegistryError> for tonic::Status {
    fn from(err: RegistryError) -> Self {
        let code = match &err {
            RegistryError::ModelNotFound(_) |
            RegistryError::VersionNotFound(_) |
            RegistryError::DeploymentNotFound(_) => tonic::Code::NotFound,
            
            RegistryError::ModelAlreadyExists(_) |
            RegistryError::Conflict(_) => tonic::Code::AlreadyExists,
            
            RegistryError::InvalidConfiguration(_) |
            RegistryError::ValidationError(_) => tonic::Code::InvalidArgument,
            
            RegistryError::PermissionDenied(_) => tonic::Code::PermissionDenied,
            
            RegistryError::RateLimitExceeded => tonic::Code::ResourceExhausted,
            
            RegistryError::ServiceUnavailable(_) |
            RegistryError::ResourceUnavailable(_) => tonic::Code::Unavailable,
            
            RegistryError::UnsupportedOperation(_) => tonic::Code::Unimplemented,
            
            RegistryError::ArtifactVerificationFailed => tonic::Code::DataLoss,
            
            _ => tonic::Code::Internal,
        };
        
        tonic::Status::new(code, err.to_string())
    }
}