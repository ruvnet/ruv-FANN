use thiserror::Error;

pub type ServiceResult<T> = Result<T, ServiceError>;

#[derive(Error, Debug)]
pub enum ServiceError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Training failed: {0}")]
    TrainingFailed(String),
    
    #[error("Prediction failed: {0}")]
    PredictionFailed(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Neural network error: {0}")]
    NeuralNetwork(#[from] ruv_fann::NetworkError),
    
    #[error("Training error: {0}")]
    Training(#[from] ruv_fann::TrainingError),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Internal server error: {0}")]
    Internal(String),
}

impl From<ServiceError> for tonic::Status {
    fn from(error: ServiceError) -> Self {
        match error {
            ServiceError::ModelNotFound(msg) => {
                tonic::Status::not_found(msg)
            }
            ServiceError::InvalidInput(msg) => {
                tonic::Status::invalid_argument(msg)
            }
            ServiceError::TrainingFailed(msg) => {
                tonic::Status::internal(format!("Training failed: {}", msg))
            }
            ServiceError::PredictionFailed(msg) => {
                tonic::Status::internal(format!("Prediction failed: {}", msg))
            }
            ServiceError::Configuration(msg) => {
                tonic::Status::invalid_argument(format!("Configuration error: {}", msg))
            }
            _ => tonic::Status::internal(error.to_string()),
        }
    }
}