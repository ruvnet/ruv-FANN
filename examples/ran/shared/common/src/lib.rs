// RAN Intelligence Platform Common Library
//! Common utilities, types, and functionality shared across all RAN services.

pub mod config;
pub mod error;
pub mod logging;
pub mod metrics;
pub mod types;
pub mod utils;
pub mod grpc;

// Re-export commonly used items
pub use error::{RanError, RanResult};
pub use types::*;
pub use ran_proto as proto;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const SERVICE_NAME: &str = "ran-intelligence-platform";

// Default configuration values
pub mod defaults {
    pub const DEFAULT_GRPC_PORT: u16 = 50051;
    pub const DEFAULT_LOG_LEVEL: &str = "info";
    pub const DEFAULT_METRICS_PORT: u16 = 9090;
    pub const DEFAULT_DATABASE_URL: &str = "postgresql://localhost:5432/ran_intelligence";
    pub const DEFAULT_DATA_PATH: &str = "./data";
    pub const DEFAULT_MODEL_PATH: &str = "./models";
    
    // ML-related defaults
    pub const DEFAULT_BATCH_SIZE: usize = 32;
    pub const DEFAULT_LEARNING_RATE: f64 = 0.001;
    pub const DEFAULT_EPOCHS: usize = 100;
    pub const DEFAULT_VALIDATION_SPLIT: f64 = 0.2;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert_eq!(SERVICE_NAME, "ran-intelligence-platform");
    }
}