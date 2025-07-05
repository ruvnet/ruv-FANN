pub mod slice_service {
    tonic::include_proto!("dni.slice");
}

pub mod network_intelligence {
    tonic::include_proto!("ran.network_intelligence");
}

pub mod common {
    tonic::include_proto!("ran.common");
}

pub mod service;
pub mod predictor;
pub mod monitor;
pub mod sla_manager;
pub mod optimizer;
pub mod alerting;
pub mod models;
pub mod error;
pub mod config;
pub mod metrics;
pub mod storage;
pub mod utils;

pub use service::SliceServiceImpl;
pub use predictor::{SlaBreachPredictor, PredictionEngine};
pub use monitor::SliceMonitor;
pub use sla_manager::SlaManager;
pub use optimizer::SliceOptimizer;
pub use alerting::AlertingService;
pub use models::{SliceModel, SliceData, ModelRegistry};
pub use error::{SliceError, SliceResult};
pub use config::SliceConfig;
pub use metrics::SliceMetricsCollector;
pub use storage::SliceStorage;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// Core DNI-SLICE-01 service providing network slice SLA breach prediction
pub struct DniSliceService {
    pub predictor: Arc<SlaBreachPredictor>,
    pub monitor: Arc<SliceMonitor>,
    pub sla_manager: Arc<SlaManager>,
    pub optimizer: Arc<SliceOptimizer>,
    pub alerting: Arc<AlertingService>,
    pub storage: Arc<SliceStorage>,
    pub metrics: Arc<SliceMetricsCollector>,
    pub config: SliceConfig,
}

impl DniSliceService {
    pub async fn new(config: SliceConfig) -> SliceResult<Self> {
        info!("Initializing DNI-SLICE-01 service");
        
        let storage = Arc::new(SliceStorage::new(&config.storage).await?);
        let metrics = Arc::new(SliceMetricsCollector::new());
        
        let predictor = Arc::new(SlaBreachPredictor::new(
            config.prediction.clone(),
            Arc::clone(&storage),
            Arc::clone(&metrics),
        ).await?);
        
        let monitor = Arc::new(SliceMonitor::new(
            config.monitoring.clone(),
            Arc::clone(&storage),
            Arc::clone(&metrics),
        ).await?);
        
        let sla_manager = Arc::new(SlaManager::new(
            config.sla.clone(),
            Arc::clone(&storage),
        ).await?);
        
        let optimizer = Arc::new(SliceOptimizer::new(
            config.optimization.clone(),
            Arc::clone(&storage),
            Arc::clone(&metrics),
        ).await?);
        
        let alerting = Arc::new(AlertingService::new(
            config.alerting.clone(),
            Arc::clone(&storage),
        ).await?);
        
        info!("DNI-SLICE-01 service initialized successfully");
        
        Ok(Self {
            predictor,
            monitor,
            sla_manager,
            optimizer,
            alerting,
            storage,
            metrics,
            config,
        })
    }
    
    pub async fn start_background_tasks(&self) -> SliceResult<()> {
        info!("Starting background tasks for DNI-SLICE-01");
        
        // Start monitoring tasks
        self.monitor.start_monitoring().await?;
        
        // Start prediction tasks
        self.predictor.start_prediction_loop().await?;
        
        // Start alerting tasks
        self.alerting.start_alerting().await?;
        
        info!("Background tasks started successfully");
        Ok(())
    }
    
    pub async fn shutdown(&self) -> SliceResult<()> {
        info!("Shutting down DNI-SLICE-01 service");
        
        // Stop all background tasks
        self.monitor.stop_monitoring().await?;
        self.predictor.stop_prediction_loop().await?;
        self.alerting.stop_alerting().await?;
        
        // Flush metrics and storage
        self.metrics.flush().await?;
        self.storage.flush().await?;
        
        info!("DNI-SLICE-01 service shutdown complete");
        Ok(())
    }
}

/// Constants for the DNI-SLICE-01 service
pub mod constants {
    pub const SERVICE_NAME: &str = "dni-slice-01";
    pub const SERVICE_VERSION: &str = "0.1.0";
    pub const DEFAULT_PREDICTION_HORIZON_MINUTES: i32 = 15;
    pub const DEFAULT_MONITORING_INTERVAL_SECONDS: i32 = 60;
    pub const DEFAULT_SLA_COMPLIANCE_THRESHOLD: f64 = 0.95;
    pub const DEFAULT_BREACH_PROBABILITY_THRESHOLD: f64 = 0.8;
    pub const MIN_HISTORICAL_DATA_POINTS: usize = 100;
    pub const MAX_CONCURRENT_PREDICTIONS: usize = 1000;
    pub const MODEL_RETRAIN_INTERVAL_HOURS: u64 = 24;
    
    // Slice types
    pub const SLICE_TYPE_EMBB: &str = "eMBB";
    pub const SLICE_TYPE_URLLC: &str = "URLLC";
    pub const SLICE_TYPE_MMTC: &str = "mMTC";
    
    // SLA metrics
    pub const METRIC_THROUGHPUT: &str = "THROUGHPUT";
    pub const METRIC_LATENCY: &str = "LATENCY";
    pub const METRIC_JITTER: &str = "JITTER";
    pub const METRIC_PACKET_LOSS: &str = "PACKET_LOSS";
    pub const METRIC_AVAILABILITY: &str = "AVAILABILITY";
    
    // Risk levels
    pub const RISK_LEVEL_LOW: &str = "LOW";
    pub const RISK_LEVEL_MEDIUM: &str = "MEDIUM";
    pub const RISK_LEVEL_HIGH: &str = "HIGH";
    pub const RISK_LEVEL_CRITICAL: &str = "CRITICAL";
    
    // Alert severities
    pub const SEVERITY_LOW: &str = "LOW";
    pub const SEVERITY_MEDIUM: &str = "MEDIUM";
    pub const SEVERITY_HIGH: &str = "HIGH";
    pub const SEVERITY_CRITICAL: &str = "CRITICAL";
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_service_initialization() {
        let config = SliceConfig::default();
        let service = DniSliceService::new(config).await;
        assert!(service.is_ok());
    }
    
    #[tokio::test]
    async fn test_background_tasks_lifecycle() {
        let config = SliceConfig::default();
        let service = DniSliceService::new(config).await.unwrap();
        
        // Start background tasks
        let start_result = service.start_background_tasks().await;
        assert!(start_result.is_ok());
        
        // Give tasks time to initialize
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Shutdown
        let shutdown_result = service.shutdown().await;
        assert!(shutdown_result.is_ok());
    }
}