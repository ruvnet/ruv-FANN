//! PFS-DATA-01: File-based Data Ingestion Service for RAN Intelligence Platform
//!
//! This service provides batch data ingestion capabilities for processing CSV and JSON files
//! into normalized Parquet format with standardized schema for RAN intelligence applications.
//!
//! Key features:
//! - High-performance file processing with configurable concurrency
//! - Schema normalization with standard RAN data model
//! - Error handling with configurable error rate thresholds
//! - Real-time monitoring and metrics
//! - gRPC API for ingestion control
//! - Directory watching with automatic file discovery

pub mod config;
pub mod error;
pub mod ingestion;
pub mod monitoring;
pub mod schema;
pub mod service;
pub mod storage;
pub mod watcher;

pub use config::IngestionConfig;
pub use error::{IngestionError, IngestionResult};
pub use ingestion::IngestionEngine;
pub use monitoring::IngestionMonitor;
pub use schema::StandardSchema;
pub use service::DataIngestionServiceImpl;
pub use storage::ParquetWriter;
pub use watcher::DirectoryWatcher;

// Re-export gRPC types
pub mod proto {
    tonic::include_proto!("pfs.data.v1");
}

use proto::*;

/// Default configuration values
pub const DEFAULT_BATCH_SIZE: usize = 10000;
pub const DEFAULT_MAX_CONCURRENT_FILES: usize = 4;
pub const DEFAULT_MAX_ERROR_RATE: f64 = 0.01; // 1%
pub const DEFAULT_ROW_GROUP_SIZE: usize = 1000000; // 1M rows
pub const DEFAULT_COMPRESSION: &str = "snappy";

/// Standard RAN data columns
pub const TIMESTAMP_COLUMN: &str = "timestamp";
pub const CELL_ID_COLUMN: &str = "cell_id";
pub const KPI_NAME_COLUMN: &str = "kpi_name";
pub const KPI_VALUE_COLUMN: &str = "kpi_value";
pub const UE_ID_COLUMN: &str = "ue_id";
pub const SECTOR_ID_COLUMN: &str = "sector_id";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_constants() {
        assert!(DEFAULT_MAX_ERROR_RATE < 0.02);
        assert!(DEFAULT_BATCH_SIZE > 0);
        assert!(DEFAULT_MAX_CONCURRENT_FILES > 0);
    }
}