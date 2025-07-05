use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Network error: {0}")]
    Network(#[from] tonic::transport::Error),
    
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Configuration error: {0}")]
    Configuration(#[from] config::ConfigError),
    
    #[error("Forecasting error: {0}")]
    Forecasting(String),
    
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("Data processing error: {0}")]
    DataProcessing(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    #[error("Timeout: {0}")]
    Timeout(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
    
    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Parse error: {0}")]
    Parse(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
}

impl From<Error> for tonic::Status {
    fn from(error: Error) -> Self {
        match error {
            Error::InvalidInput(msg) => tonic::Status::invalid_argument(msg),
            Error::InsufficientData(msg) => tonic::Status::failed_precondition(msg),
            Error::Timeout(msg) => tonic::Status::deadline_exceeded(msg),
            Error::Validation(msg) => tonic::Status::invalid_argument(msg),
            Error::Grpc(status) => status,
            _ => tonic::Status::internal(error.to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;