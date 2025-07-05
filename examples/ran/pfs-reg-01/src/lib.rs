//! PFS-REG-01: Model Registry & Lifecycle Service for RAN Intelligence Platform
//!
//! This service provides comprehensive model registry functionality including:
//! - Model registration and metadata management
//! - Model versioning and lifecycle management
//! - Model deployment and retirement APIs
//! - Search and discovery functionality
//! - Performance monitoring and metrics
//!
//! Built on top of the neuro-divergent registry system with extensions for
//! RAN-specific requirements and enterprise-grade model lifecycle management.

#![warn(missing_docs, rust_2018_idioms)]
#![deny(unsafe_code)]

pub mod config;
pub mod error;
pub mod services;
pub mod storage;
pub mod database;
pub mod grpc;
pub mod metrics;

// Generated protobuf code
pub mod generated {
    #![allow(missing_docs)]
    include!("generated/pfs_reg_01.rs");
}

pub use config::RegistryConfig;
pub use error::{RegistryError, RegistryResult};
pub use services::registry_service::ModelRegistryService;
pub use storage::{ModelArtifactStorage, FilesystemStorage};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const SERVICE_NAME: &str = "pfs-reg-01";

/// Initialize tracing
pub fn init_tracing(log_level: &str) -> anyhow::Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
    
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_thread_ids(true)
                .with_line_number(true)
                .compact(),
        )
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(log_level)),
        )
        .init();
    
    Ok(())
}

/// Service health status
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// Service is healthy and operational
    Healthy,
    /// Service is degraded but operational
    Degraded,
    /// Service is unhealthy
    Unhealthy,
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Overall health status
    pub status: HealthStatus,
    /// Component-specific health checks
    pub components: std::collections::HashMap<String, ComponentHealth>,
    /// Timestamp of the health check
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Component health information
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    /// Component health status
    pub status: HealthStatus,
    /// Additional details about the component
    pub details: Option<String>,
    /// Component metrics
    pub metrics: std::collections::HashMap<String, f64>,
}

impl HealthCheck {
    /// Create a new health check
    pub fn new() -> Self {
        Self {
            status: HealthStatus::Healthy,
            components: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Add a component health check
    pub fn add_component(&mut self, name: String, health: ComponentHealth) {
        // Update overall status based on component status
        match health.status {
            HealthStatus::Unhealthy => self.status = HealthStatus::Unhealthy,
            HealthStatus::Degraded if self.status == HealthStatus::Healthy => {
                self.status = HealthStatus::Degraded;
            }
            _ => {}
        }
        
        self.components.insert(name, health);
    }
}

impl Default for HealthCheck {
    fn default() -> Self {
        Self::new()
    }
}