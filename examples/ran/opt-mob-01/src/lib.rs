//! OPT-MOB-01: Predictive Handover Trigger Model
//!
//! This module implements a neural network-based handover prediction system
//! that analyzes UE metrics to predict handover probability and target cells.
//!
//! ## Architecture
//!
//! - **Feature Engineering**: Preprocessing of UE metrics with time-series features
//! - **Neural Network**: ruv-FANN-based classifier for handover prediction
//! - **Backtesting**: Comprehensive evaluation framework with >90% accuracy target
//! - **Real-time Service**: gRPC-based prediction service
//!
//! ## Key Components
//!
//! - `HandoverPredictor`: Main prediction engine
//! - `FeatureExtractor`: Time-series feature engineering
//! - `BacktestingFramework`: Model evaluation and metrics
//! - `HandoverDataset`: Data management and preprocessing
//!
//! ## Usage
//!
//! ```rust
//! use opt_mob_01::{HandoverPredictor, FeatureExtractor, UeMetrics};
//!
//! // Create predictor with trained model
//! let predictor = HandoverPredictor::load_model("models/handover_v1.bin")?;
//!
//! // Prepare UE metrics
//! let metrics = UeMetrics::new("UE_001")
//!     .with_rsrp(-85.0)
//!     .with_sinr(12.0)
//!     .with_speed(60.0);
//!
//! // Predict handover
//! let prediction = predictor.predict(&metrics).await?;
//! println!("Handover probability: {:.2}%", prediction.ho_probability * 100.0);
//! ```

pub mod data;
pub mod features;
pub mod model;
pub mod prediction;
pub mod backtesting;
pub mod service;
pub mod utils;

// Re-export main types
pub use data::{UeMetrics, HandoverEvent, HandoverDataset, NeighborCell};
pub use features::{FeatureExtractor, FeatureVector, TimeSeriesFeatures};
pub use model::{HandoverModel, ModelConfig, TrainingConfig};
pub use prediction::{HandoverPredictor, PredictionResult, PredictionError};
pub use backtesting::{BacktestingFramework, BacktestResults, ModelMetrics};
pub use service::{HandoverPredictorService, ServiceConfig};

// Generated gRPC code
pub mod generated {
    tonic::include_proto!("handover_predictor");
}

// Re-export common types
pub use generated::*;

// Error types
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OptMobError {
    #[error("Model error: {0}")]
    Model(#[from] ruv_fann::NetworkError),
    
    #[error("Data error: {0}")]
    Data(String),
    
    #[error("Feature extraction error: {0}")]
    Features(String),
    
    #[error("Prediction error: {0}")]
    Prediction(#[from] PredictionError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Service error: {0}")]
    Service(#[from] tonic::Status),
}

pub type Result<T> = std::result::Result<T, OptMobError>;

// Constants
pub const DEFAULT_PREDICTION_HORIZON_SECONDS: i64 = 30;
pub const DEFAULT_HANDOVER_THRESHOLD: f64 = 0.5;
pub const MINIMUM_ACCURACY_TARGET: f64 = 0.90;
pub const DEFAULT_FEATURE_WINDOW_SIZE: usize = 10;
pub const DEFAULT_SAMPLING_RATE_MS: u64 = 1000;

// Configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptMobConfig {
    pub model_path: String,
    pub prediction_horizon_seconds: i64,
    pub handover_threshold: f64,
    pub feature_window_size: usize,
    pub sampling_rate_ms: u64,
    pub grpc_port: u16,
    pub max_batch_size: usize,
    pub enable_logging: bool,
}

impl Default for OptMobConfig {
    fn default() -> Self {
        Self {
            model_path: "models/handover_v1.bin".to_string(),
            prediction_horizon_seconds: DEFAULT_PREDICTION_HORIZON_SECONDS,
            handover_threshold: DEFAULT_HANDOVER_THRESHOLD,
            feature_window_size: DEFAULT_FEATURE_WINDOW_SIZE,
            sampling_rate_ms: DEFAULT_SAMPLING_RATE_MS,
            grpc_port: 50051,
            max_batch_size: 1000,
            enable_logging: true,
        }
    }
}