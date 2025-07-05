//! Metrics collection for the Model Registry service

use prometheus::{
    Counter, Gauge, Histogram, HistogramVec, IntCounter, IntCounterVec, IntGauge, Registry,
};
use std::sync::Arc;

/// Metrics collector for the Model Registry service
#[derive(Clone)]
pub struct RegistryMetrics {
    /// Total number of registered models
    pub total_models: IntGauge,
    
    /// Total number of model versions
    pub total_versions: IntGauge,
    
    /// Total number of active deployments
    pub active_deployments: IntGauge,
    
    /// Model registration requests
    pub model_registrations: IntCounterVec,
    
    /// Model retrieval requests
    pub model_retrievals: IntCounterVec,
    
    /// Model search requests
    pub model_searches: IntCounter,
    
    /// Request duration histogram
    pub request_duration: HistogramVec,
    
    /// Storage operations
    pub storage_operations: IntCounterVec,
    
    /// Database operations
    pub database_operations: IntCounterVec,
    
    /// Error counters
    pub errors: IntCounterVec,
    
    /// Model artifact sizes
    pub artifact_sizes: Histogram,
    
    /// Registry size in bytes
    pub registry_size_bytes: IntGauge,
}

impl RegistryMetrics {
    /// Create new metrics instance
    pub fn new() -> anyhow::Result<Self> {
        let metrics = Self {
            total_models: IntGauge::new(
                "registry_total_models",
                "Total number of registered models"
            )?,
            
            total_versions: IntGauge::new(
                "registry_total_versions",
                "Total number of model versions"
            )?,
            
            active_deployments: IntGauge::new(
                "registry_active_deployments",
                "Total number of active deployments"
            )?,
            
            model_registrations: IntCounterVec::new(
                prometheus::Opts::new(
                    "registry_model_registrations_total",
                    "Total number of model registration requests"
                ),
                &["status"]
            )?,
            
            model_retrievals: IntCounterVec::new(
                prometheus::Opts::new(
                    "registry_model_retrievals_total",
                    "Total number of model retrieval requests"
                ),
                &["status", "include_artifact"]
            )?,
            
            model_searches: IntCounter::new(
                "registry_model_searches_total",
                "Total number of model search requests"
            )?,
            
            request_duration: HistogramVec::new(
                prometheus::HistogramOpts::new(
                    "registry_request_duration_seconds",
                    "Request duration in seconds"
                ).buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]),
                &["method", "status"]
            )?,
            
            storage_operations: IntCounterVec::new(
                prometheus::Opts::new(
                    "registry_storage_operations_total",
                    "Total number of storage operations"
                ),
                &["operation", "status"]
            )?,
            
            database_operations: IntCounterVec::new(
                prometheus::Opts::new(
                    "registry_database_operations_total",
                    "Total number of database operations"
                ),
                &["operation", "status"]
            )?,
            
            errors: IntCounterVec::new(
                prometheus::Opts::new(
                    "registry_errors_total",
                    "Total number of errors"
                ),
                &["error_type", "component"]
            )?,
            
            artifact_sizes: Histogram::new(
                prometheus::HistogramOpts::new(
                    "registry_artifact_sizes_bytes",
                    "Size of model artifacts in bytes"
                ).buckets(vec![
                    1024.0, 10240.0, 102400.0, 1048576.0, 10485760.0, 
                    104857600.0, 1073741824.0, 10737418240.0
                ])
            )?,
            
            registry_size_bytes: IntGauge::new(
                "registry_size_bytes",
                "Total size of the registry in bytes"
            )?,
        };
        
        Ok(metrics)
    }
    
    /// Register metrics with Prometheus registry
    pub fn register(&self, registry: &Registry) -> anyhow::Result<()> {
        registry.register(Box::new(self.total_models.clone()))?;
        registry.register(Box::new(self.total_versions.clone()))?;
        registry.register(Box::new(self.active_deployments.clone()))?;
        registry.register(Box::new(self.model_registrations.clone()))?;
        registry.register(Box::new(self.model_retrievals.clone()))?;
        registry.register(Box::new(self.model_searches.clone()))?;
        registry.register(Box::new(self.request_duration.clone()))?;
        registry.register(Box::new(self.storage_operations.clone()))?;
        registry.register(Box::new(self.database_operations.clone()))?;
        registry.register(Box::new(self.errors.clone()))?;
        registry.register(Box::new(self.artifact_sizes.clone()))?;
        registry.register(Box::new(self.registry_size_bytes.clone()))?;
        
        Ok(())
    }
    
    /// Record a successful model registration
    pub fn record_model_registration_success(&self) {
        self.model_registrations.with_label_values(&["success"]).inc();
    }
    
    /// Record a failed model registration
    pub fn record_model_registration_failure(&self) {
        self.model_registrations.with_label_values(&["failure"]).inc();
    }
    
    /// Record a model retrieval
    pub fn record_model_retrieval(&self, success: bool, include_artifact: bool) {
        let status = if success { "success" } else { "failure" };
        let artifact = if include_artifact { "true" } else { "false" };
        self.model_retrievals.with_label_values(&[status, artifact]).inc();
    }
    
    /// Record a model search
    pub fn record_model_search(&self) {
        self.model_searches.inc();
    }
    
    /// Record request duration
    pub fn record_request_duration(&self, method: &str, status: &str, duration: f64) {
        self.request_duration
            .with_label_values(&[method, status])
            .observe(duration);
    }
    
    /// Record storage operation
    pub fn record_storage_operation(&self, operation: &str, success: bool) {
        let status = if success { "success" } else { "failure" };
        self.storage_operations.with_label_values(&[operation, status]).inc();
    }
    
    /// Record database operation
    pub fn record_database_operation(&self, operation: &str, success: bool) {
        let status = if success { "success" } else { "failure" };
        self.database_operations.with_label_values(&[operation, status]).inc();
    }
    
    /// Record an error
    pub fn record_error(&self, error_type: &str, component: &str) {
        self.errors.with_label_values(&[error_type, component]).inc();
    }
    
    /// Record artifact size
    pub fn record_artifact_size(&self, size_bytes: f64) {
        self.artifact_sizes.observe(size_bytes);
    }
    
    /// Update registry metrics
    pub fn update_registry_stats(&self, total_models: i64, total_versions: i64, active_deployments: i64, size_bytes: i64) {
        self.total_models.set(total_models);
        self.total_versions.set(total_versions);
        self.active_deployments.set(active_deployments);
        self.registry_size_bytes.set(size_bytes);
    }
}

impl Default for RegistryMetrics {
    fn default() -> Self {
        Self::new().unwrap()
    }
}