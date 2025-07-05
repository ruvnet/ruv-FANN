//! Common utilities and types for the RAN Intelligence Platform

pub mod config;
pub mod database;
pub mod metrics;
pub mod utils;

use crate::types::*;
use crate::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Configuration for the RAN Intelligence Platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RanConfig {
    pub database_url: String,
    pub ml_service_endpoint: String,
    pub feature_store_path: String,
    pub model_registry_path: String,
    pub log_level: String,
    pub performance_targets: PerformanceTargets,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub prediction_accuracy_threshold: f64,
    pub latency_ms_p95: u64,
    pub throughput_predictions_per_sec: u64,
}

/// Trait for ML models in the RAN Intelligence Platform
#[async_trait]
pub trait RanModel {
    type Input;
    type Output;
    
    /// Train the model with the given data
    async fn train(&mut self, data: &[Self::Input]) -> Result<()>;
    
    /// Make predictions on the given input
    async fn predict(&self, input: &Self::Input) -> Result<Self::Output>;
    
    /// Validate the model performance
    async fn validate(&self, test_data: &[Self::Input]) -> Result<ModelMetrics>;
    
    /// Save the model to disk
    async fn save(&self, path: &str) -> Result<()>;
    
    /// Load the model from disk
    async fn load(path: &str) -> Result<Self> where Self: Sized;
}

/// Metrics for model performance evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub auc_roc: f64,
    pub confusion_matrix: Vec<Vec<u32>>,
    pub training_time_ms: u64,
    pub inference_time_ms: u64,
}

/// Feature engineering trait for time series data
#[async_trait]
pub trait FeatureEngineer {
    type Input;
    type Output;
    
    /// Extract features from raw time series data
    async fn extract_features(&self, data: &[Self::Input]) -> Result<Vec<Self::Output>>;
    
    /// Validate feature quality
    async fn validate_features(&self, features: &[Self::Output]) -> Result<()>;
}

/// Data ingestion trait for various data sources
#[async_trait]
pub trait DataIngester {
    type Input;
    
    /// Ingest data from the specified source
    async fn ingest(&self, source: &str) -> Result<Vec<Self::Input>>;
    
    /// Validate data quality
    async fn validate(&self, data: &[Self::Input]) -> Result<()>;
}

impl Default for RanConfig {
    fn default() -> Self {
        Self {
            database_url: "postgresql://localhost:5432/ran_intelligence".to_string(),
            ml_service_endpoint: "http://localhost:8080".to_string(),
            feature_store_path: "./data/features".to_string(),
            model_registry_path: "./models".to_string(),
            log_level: "info".to_string(),
            performance_targets: PerformanceTargets {
                prediction_accuracy_threshold: 0.8,
                latency_ms_p95: 100,
                throughput_predictions_per_sec: 1000,
            },
        }
    }
}