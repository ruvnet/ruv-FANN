pub mod neural_service {
    tonic::include_proto!("neural_service");
}

pub mod service;
pub mod model_manager;
pub mod conversion;
pub mod error;
pub mod config;

pub use service::NeuralServiceImpl;
pub use model_manager::ModelManager;
pub use error::{ServiceError, ServiceResult};
pub use config::ServiceConfig;