//! # SCell Manager - Predictive Carrier Aggregation for RAN Intelligence
//!
//! This crate implements OPT-RES-01: Predictive Carrier Aggregation SCell Manager
//! as specified in the RAN Intelligence Platform PRD.
//!
//! ## Overview
//!
//! The SCell Manager uses machine learning to predict when a UE will need
//! Secondary Cell (SCell) activation for Carrier Aggregation, achieving >80%
//! accuracy in predicting high throughput demand scenarios.
//!
//! ## Key Features
//!
//! - Real-time throughput demand prediction
//! - SCell activation recommendations
//! - Historical pattern analysis
//! - Performance monitoring and metrics
//! - gRPC API for integration
//!
//! ## Architecture
//!
//! The system consists of several key components:
//! - Prediction Engine: Uses ruv-FANN for ML inference
//! - Data Pipeline: Processes UE metrics and historical data
//! - Model Manager: Handles model training and versioning
//! - API Service: Provides gRPC endpoints for predictions

pub mod config;
pub mod data;
pub mod metrics;
pub mod model;
pub mod prediction;
pub mod service;
pub mod types;
pub mod utils;

// Re-export commonly used types
pub use config::SCellManagerConfig;
pub use metrics::MetricsCollector;
pub use model::SCellPredictionModel;
pub use prediction::PredictionEngine;
pub use service::SCellManagerService;
pub use types::*;

// Generated protobuf types
pub mod proto {
    tonic::include_proto!("scell_manager");
}

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main SCell Manager instance
#[derive(Debug, Clone)]
pub struct SCellManager {
    config: SCellManagerConfig,
    prediction_engine: Arc<RwLock<PredictionEngine>>,
    metrics_collector: Arc<MetricsCollector>,
}

impl SCellManager {
    /// Create a new SCell Manager instance
    pub async fn new(config: SCellManagerConfig) -> Result<Self> {
        let prediction_engine = Arc::new(RwLock::new(
            PredictionEngine::new(&config.model_config).await?
        ));
        
        let metrics_collector = Arc::new(MetricsCollector::new()?);
        
        Ok(Self {
            config,
            prediction_engine,
            metrics_collector,
        })
    }
    
    /// Get the prediction engine
    pub fn prediction_engine(&self) -> Arc<RwLock<PredictionEngine>> {
        self.prediction_engine.clone()
    }
    
    /// Get the metrics collector
    pub fn metrics_collector(&self) -> Arc<MetricsCollector> {
        self.metrics_collector.clone()
    }
    
    /// Get the configuration
    pub fn config(&self) -> &SCellManagerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_scell_manager_creation() {
        let config = SCellManagerConfig::default();
        let manager = SCellManager::new(config).await;
        assert!(manager.is_ok());
    }
}