//! RAN Intelligence Platform
//! 
//! AI-powered RAN Intelligence & Automation Platform using ruv-FANN
//! for 5G network optimization and service assurance.

pub mod asa_5g;
pub mod common;
pub mod features;
pub mod models;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum RanError {
    #[error("ML model error: {0}")]
    ModelError(String),
    
    #[error("Feature engineering error: {0}")]
    FeatureError(String),
    
    #[error("Data processing error: {0}")]
    DataError(String),
    
    #[error("Network communication error: {0}")]
    NetworkError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("5G service assurance error: {0}")]
    ServiceAssuranceError(String),
}

pub type Result<T> = std::result::Result<T, RanError>;

/// Common types used across the RAN Intelligence Platform
pub mod types {
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UeId(pub String);

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CellId(pub String);

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ModelId(pub Uuid);

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TimeSeries {
        pub timestamp: DateTime<Utc>,
        pub values: Vec<f64>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SignalQuality {
        pub timestamp: DateTime<Utc>,
        pub ue_id: UeId,
        pub lte_rsrp: f64,
        pub lte_sinr: f64,
        pub nr_ssb_rsrp: Option<f64>,
        pub endc_setup_success_rate: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PredictionResult {
        pub timestamp: DateTime<Utc>,
        pub confidence: f64,
        pub metadata: serde_json::Value,
    }
}