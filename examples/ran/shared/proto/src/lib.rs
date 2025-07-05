// RAN Intelligence Platform gRPC Protocol Definitions
//! This crate contains all Protocol Buffer definitions and generated gRPC code
//! for the RAN Intelligence Platform services.

// Common definitions
pub mod common {
    tonic::include_proto!("ran.common");
}

// Platform Foundation Services
pub mod ml_core {
    tonic::include_proto!("ran.ml_core");
}

pub mod data_ingestion {
    tonic::include_proto!("ran.data_ingestion");
}

pub mod feature_engineering {
    tonic::include_proto!("ran.feature_engineering");
}

pub mod model_registry {
    tonic::include_proto!("ran.model_registry");
}

// Predictive Optimization Services
pub mod predictive_optimization {
    tonic::include_proto!("ran.predictive_optimization");
}

// Service Assurance Services
pub mod service_assurance {
    tonic::include_proto!("ran.service_assurance");
}

// Network Intelligence Services
pub mod network_intelligence {
    tonic::include_proto!("ran.network_intelligence");
}

// Re-export commonly used types for convenience
pub use common::*;

// Helper functions for creating common message types
impl Metadata {
    pub fn new(service_name: &str, version: &str, request_id: &str) -> Self {
        Self {
            service_name: service_name.to_string(),
            version: version.to_string(),
            timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
            request_id: request_id.to_string(),
        }
    }
}

impl DataPoint {
    pub fn new(cell_id: &str, kpi_name: &str, kpi_value: f64) -> Self {
        Self {
            timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
            cell_id: cell_id.to_string(),
            kpi_name: kpi_name.to_string(),
            kpi_value,
            tags: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_tags(mut self, tags: std::collections::HashMap<String, String>) -> Self {
        self.tags = tags;
        self
    }
}

impl TimeSeriesData {
    pub fn new(series_id: &str, data_points: Vec<DataPoint>, metadata: Metadata) -> Self {
        Self {
            series_id: series_id.to_string(),
            data_points,
            metadata: Some(metadata),
        }
    }
}

impl ModelConfig {
    pub fn new(model_type: &str) -> Self {
        Self {
            model_type: model_type.to_string(),
            parameters: std::collections::HashMap::new(),
            input_features: Vec::new(),
            output_features: Vec::new(),
        }
    }
    
    pub fn with_parameter(mut self, key: &str, value: &str) -> Self {
        self.parameters.insert(key.to_string(), value.to_string());
        self
    }
    
    pub fn with_input_features(mut self, features: Vec<String>) -> Self {
        self.input_features = features;
        self
    }
    
    pub fn with_output_features(mut self, features: Vec<String>) -> Self {
        self.output_features = features;
        self
    }
}

impl HealthCheck {
    pub fn new(service_name: &str, status: &str) -> Self {
        Self {
            service_name: service_name.to_string(),
            status: status.to_string(),
            last_updated: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
            details: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_details(mut self, details: std::collections::HashMap<String, String>) -> Self {
        self.details = details;
        self
    }
}

impl ErrorResponse {
    pub fn new(error_code: &str, error_message: &str, service_name: &str) -> Self {
        Self {
            error_code: error_code.to_string(),
            error_message: error_message.to_string(),
            service_name: service_name.to_string(),
            timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
        }
    }
}

// Helper trait for converting system time to protobuf timestamp
trait TimestampHelper {
    fn from_system_time(time: std::time::SystemTime) -> prost_types::Timestamp;
}

impl TimestampHelper for prost_types::Timestamp {
    fn from_system_time(time: std::time::SystemTime) -> prost_types::Timestamp {
        let duration = time.duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0));
        
        prost_types::Timestamp {
            seconds: duration.as_secs() as i64,
            nanos: duration.subsec_nanos() as i32,
        }
    }
}

impl From<std::time::SystemTime> for prost_types::Timestamp {
    fn from(time: std::time::SystemTime) -> Self {
        prost_types::Timestamp::from_system_time(time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_creation() {
        let metadata = Metadata::new("test-service", "1.0.0", "req-123");
        assert_eq!(metadata.service_name, "test-service");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.request_id, "req-123");
        assert!(metadata.timestamp.is_some());
    }

    #[test]
    fn test_data_point_creation() {
        let dp = DataPoint::new("cell-001", "prb_utilization", 75.5);
        assert_eq!(dp.cell_id, "cell-001");
        assert_eq!(dp.kpi_name, "prb_utilization");
        assert_eq!(dp.kpi_value, 75.5);
        assert!(dp.timestamp.is_some());
    }

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::new("neural_network")
            .with_parameter("hidden_layers", "3")
            .with_parameter("learning_rate", "0.001")
            .with_input_features(vec!["rsrp".to_string(), "sinr".to_string()])
            .with_output_features(vec!["handover_probability".to_string()]);
        
        assert_eq!(config.model_type, "neural_network");
        assert_eq!(config.parameters.len(), 2);
        assert_eq!(config.input_features.len(), 2);
        assert_eq!(config.output_features.len(), 1);
    }
}