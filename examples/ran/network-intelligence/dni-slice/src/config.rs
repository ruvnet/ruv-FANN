use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceConfig {
    pub server: ServerConfig,
    pub prediction: PredictionConfig,
    pub monitoring: MonitoringConfig,
    pub sla: SlaConfig,
    pub optimization: OptimizationConfig,
    pub alerting: AlertingConfig,
    pub storage: StorageConfig,
    pub metrics: MetricsConfig,
    pub neural_network: NeuralNetworkConfig,
    pub feature_engineering: FeatureEngineeringConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: u32,
    pub connection_timeout_seconds: u64,
    pub request_timeout_seconds: u64,
    pub keepalive_timeout_seconds: u64,
    pub max_concurrent_streams: u32,
    pub initial_stream_window_size: u32,
    pub initial_connection_window_size: u32,
    pub max_frame_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    pub enabled: bool,
    pub default_horizon_minutes: i32,
    pub max_horizon_minutes: i32,
    pub min_historical_data_points: usize,
    pub max_concurrent_predictions: usize,
    pub prediction_interval_seconds: u64,
    pub model_retrain_interval_hours: u64,
    pub confidence_threshold: f64,
    pub breach_probability_threshold: f64,
    pub feature_importance_enabled: bool,
    pub uncertainty_quantification_enabled: bool,
    pub ensemble_voting_enabled: bool,
    pub cache_predictions: bool,
    pub cache_ttl_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enabled: bool,
    pub default_interval_seconds: i32,
    pub max_monitored_slices: usize,
    pub health_check_interval_seconds: u64,
    pub metrics_collection_interval_seconds: u64,
    pub data_retention_days: u32,
    pub streaming_buffer_size: usize,
    pub streaming_timeout_seconds: u64,
    pub anomaly_detection_enabled: bool,
    pub real_time_alerts: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaConfig {
    pub compliance_threshold: f64,
    pub default_availability_threshold: f64,
    pub default_latency_threshold_ms: f64,
    pub default_throughput_threshold_mbps: f64,
    pub default_packet_loss_threshold: f64,
    pub sla_evaluation_interval_seconds: u64,
    pub grace_period_minutes: u32,
    pub escalation_levels: Vec<String>,
    pub auto_remediation_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub enabled: bool,
    pub optimization_interval_hours: u64,
    pub max_optimization_suggestions: usize,
    pub cost_benefit_threshold: f64,
    pub roi_threshold: f64,
    pub simulation_enabled: bool,
    pub what_if_analysis_enabled: bool,
    pub multi_objective_optimization: bool,
    pub resource_constraints: ResourceConstraints,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    pub max_cpu_allocation_percent: f64,
    pub max_memory_allocation_percent: f64,
    pub max_bandwidth_allocation_percent: f64,
    pub max_concurrent_users_per_slice: u32,
    pub budget_constraints: bool,
    pub max_budget_per_slice: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    pub enabled: bool,
    pub alert_evaluation_interval_seconds: u64,
    pub max_alerts_per_slice: usize,
    pub alert_aggregation_window_seconds: u64,
    pub escalation_enabled: bool,
    pub auto_acknowledgment_timeout_minutes: u32,
    pub notification_channels: Vec<NotificationChannel>,
    pub severity_mapping: SeverityMapping,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    pub name: String,
    pub channel_type: String, // EMAIL, SLACK, WEBHOOK, SMS
    pub endpoint: String,
    pub enabled: bool,
    pub severity_filter: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeverityMapping {
    pub critical_threshold: f64,
    pub high_threshold: f64,
    pub medium_threshold: f64,
    pub low_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub storage_type: String, // MEMORY, FILE, DATABASE
    pub connection_string: Option<String>,
    pub database_url: Option<String>,
    pub file_path: Option<PathBuf>,
    pub max_connections: u32,
    pub connection_timeout_seconds: u64,
    pub data_retention_days: u32,
    pub backup_enabled: bool,
    pub backup_interval_hours: u64,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub collection_interval_seconds: u64,
    pub export_interval_seconds: u64,
    pub prometheus_enabled: bool,
    pub prometheus_port: u16,
    pub custom_metrics_enabled: bool,
    pub histogram_buckets: Vec<f64>,
    pub labels: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkConfig {
    pub model_type: String, // FANN, LSTM, GRU, TRANSFORMER
    pub hidden_layers: Vec<usize>,
    pub activation_function: String,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub dropout_rate: f64,
    pub regularization: f64,
    pub optimizer: String,
    pub loss_function: String,
    pub early_stopping_patience: usize,
    pub validation_split: f64,
    pub model_checkpointing: bool,
    pub ensemble_size: usize,
    pub cross_validation_folds: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    pub lookback_window_minutes: i32,
    pub feature_selection_enabled: bool,
    pub dimensionality_reduction_enabled: bool,
    pub normalization_method: String, // STANDARD, MINMAX, ROBUST
    pub outlier_detection_enabled: bool,
    pub outlier_threshold: f64,
    pub seasonal_decomposition: bool,
    pub trend_analysis: bool,
    pub lag_features: Vec<i32>,
    pub rolling_window_sizes: Vec<i32>,
    pub statistical_features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String, // JSON, PLAIN
    pub output: String, // STDOUT, FILE
    pub file_path: Option<PathBuf>,
    pub rotation_enabled: bool,
    pub max_file_size_mb: u64,
    pub max_files: u32,
    pub structured_logging: bool,
    pub trace_sampling_rate: f64,
}

impl Default for SliceConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            prediction: PredictionConfig::default(),
            monitoring: MonitoringConfig::default(),
            sla: SlaConfig::default(),
            optimization: OptimizationConfig::default(),
            alerting: AlertingConfig::default(),
            storage: StorageConfig::default(),
            metrics: MetricsConfig::default(),
            neural_network: NeuralNetworkConfig::default(),
            feature_engineering: FeatureEngineeringConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 50051,
            max_connections: 1000,
            connection_timeout_seconds: 30,
            request_timeout_seconds: 300,
            keepalive_timeout_seconds: 20,
            max_concurrent_streams: 100,
            initial_stream_window_size: 65536,
            initial_connection_window_size: 65536,
            max_frame_size: 16384,
        }
    }
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_horizon_minutes: 15,
            max_horizon_minutes: 60,
            min_historical_data_points: 100,
            max_concurrent_predictions: 1000,
            prediction_interval_seconds: 60,
            model_retrain_interval_hours: 24,
            confidence_threshold: 0.8,
            breach_probability_threshold: 0.8,
            feature_importance_enabled: true,
            uncertainty_quantification_enabled: true,
            ensemble_voting_enabled: true,
            cache_predictions: true,
            cache_ttl_seconds: 300,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_interval_seconds: 60,
            max_monitored_slices: 10000,
            health_check_interval_seconds: 30,
            metrics_collection_interval_seconds: 10,
            data_retention_days: 30,
            streaming_buffer_size: 1000,
            streaming_timeout_seconds: 30,
            anomaly_detection_enabled: true,
            real_time_alerts: true,
        }
    }
}

impl Default for SlaConfig {
    fn default() -> Self {
        Self {
            compliance_threshold: 0.95,
            default_availability_threshold: 99.9,
            default_latency_threshold_ms: 50.0,
            default_throughput_threshold_mbps: 100.0,
            default_packet_loss_threshold: 0.1,
            sla_evaluation_interval_seconds: 300,
            grace_period_minutes: 5,
            escalation_levels: vec![
                "LEVEL_1".to_string(),
                "LEVEL_2".to_string(),
                "LEVEL_3".to_string(),
            ],
            auto_remediation_enabled: false,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimization_interval_hours: 6,
            max_optimization_suggestions: 50,
            cost_benefit_threshold: 1.2,
            roi_threshold: 0.15,
            simulation_enabled: true,
            what_if_analysis_enabled: true,
            multi_objective_optimization: true,
            resource_constraints: ResourceConstraints::default(),
        }
    }
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_cpu_allocation_percent: 80.0,
            max_memory_allocation_percent: 80.0,
            max_bandwidth_allocation_percent: 80.0,
            max_concurrent_users_per_slice: 10000,
            budget_constraints: false,
            max_budget_per_slice: 100000.0,
        }
    }
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            alert_evaluation_interval_seconds: 60,
            max_alerts_per_slice: 100,
            alert_aggregation_window_seconds: 300,
            escalation_enabled: true,
            auto_acknowledgment_timeout_minutes: 30,
            notification_channels: vec![],
            severity_mapping: SeverityMapping::default(),
        }
    }
}

impl Default for SeverityMapping {
    fn default() -> Self {
        Self {
            critical_threshold: 0.95,
            high_threshold: 0.85,
            medium_threshold: 0.70,
            low_threshold: 0.50,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            storage_type: "MEMORY".to_string(),
            connection_string: None,
            database_url: None,
            file_path: None,
            max_connections: 10,
            connection_timeout_seconds: 30,
            data_retention_days: 30,
            backup_enabled: false,
            backup_interval_hours: 24,
            compression_enabled: true,
            encryption_enabled: false,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval_seconds: 10,
            export_interval_seconds: 60,
            prometheus_enabled: true,
            prometheus_port: 9090,
            custom_metrics_enabled: true,
            histogram_buckets: vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            labels: std::collections::HashMap::new(),
        }
    }
}

impl Default for NeuralNetworkConfig {
    fn default() -> Self {
        Self {
            model_type: "FANN".to_string(),
            hidden_layers: vec![64, 32, 16],
            activation_function: "SIGMOID".to_string(),
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            dropout_rate: 0.2,
            regularization: 0.01,
            optimizer: "ADAM".to_string(),
            loss_function: "MSE".to_string(),
            early_stopping_patience: 10,
            validation_split: 0.2,
            model_checkpointing: true,
            ensemble_size: 5,
            cross_validation_folds: 5,
        }
    }
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            lookback_window_minutes: 60,
            feature_selection_enabled: true,
            dimensionality_reduction_enabled: false,
            normalization_method: "STANDARD".to_string(),
            outlier_detection_enabled: true,
            outlier_threshold: 3.0,
            seasonal_decomposition: true,
            trend_analysis: true,
            lag_features: vec![1, 2, 3, 5, 10],
            rolling_window_sizes: vec![5, 10, 15, 30],
            statistical_features: vec![
                "MEAN".to_string(),
                "STD".to_string(),
                "MIN".to_string(),
                "MAX".to_string(),
                "MEDIAN".to_string(),
                "PERCENTILE_25".to_string(),
                "PERCENTILE_75".to_string(),
            ],
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "INFO".to_string(),
            format: "JSON".to_string(),
            output: "STDOUT".to_string(),
            file_path: None,
            rotation_enabled: true,
            max_file_size_mb: 100,
            max_files: 10,
            structured_logging: true,
            trace_sampling_rate: 0.1,
        }
    }
}

impl SliceConfig {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, config::ConfigError> {
        let settings = config::Config::builder()
            .add_source(config::File::from(path.as_ref()))
            .build()?;
        
        settings.try_deserialize()
    }
    
    pub fn from_env() -> Result<Self, config::ConfigError> {
        let settings = config::Config::builder()
            .add_source(config::Environment::with_prefix("DNI_SLICE"))
            .build()?;
        
        settings.try_deserialize()
    }
    
    pub fn merge_with_env(mut self) -> Result<Self, config::ConfigError> {
        let env_config = Self::from_env().unwrap_or_default();
        
        // Merge specific fields from environment
        if env_config.server.port != 50051 {
            self.server.port = env_config.server.port;
        }
        
        if !env_config.server.host.is_empty() && env_config.server.host != "0.0.0.0" {
            self.server.host = env_config.server.host;
        }
        
        Ok(self)
    }
    
    pub fn validate(&self) -> Result<(), crate::error::SliceError> {
        // Validate server config
        if self.server.port == 0 {
            return Err(crate::error::SliceError::Configuration(
                "Server port cannot be zero".to_string()
            ));
        }
        
        // Validate prediction config
        if self.prediction.default_horizon_minutes <= 0 {
            return Err(crate::error::SliceError::Configuration(
                "Prediction horizon must be positive".to_string()
            ));
        }
        
        if self.prediction.confidence_threshold < 0.0 || self.prediction.confidence_threshold > 1.0 {
            return Err(crate::error::SliceError::Configuration(
                "Confidence threshold must be between 0 and 1".to_string()
            ));
        }
        
        // Validate SLA config
        if self.sla.compliance_threshold < 0.0 || self.sla.compliance_threshold > 1.0 {
            return Err(crate::error::SliceError::Configuration(
                "SLA compliance threshold must be between 0 and 1".to_string()
            ));
        }
        
        // Validate neural network config
        if self.neural_network.hidden_layers.is_empty() {
            return Err(crate::error::SliceError::Configuration(
                "Neural network must have at least one hidden layer".to_string()
            ));
        }
        
        if self.neural_network.learning_rate <= 0.0 {
            return Err(crate::error::SliceError::Configuration(
                "Learning rate must be positive".to_string()
            ));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = SliceConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = SliceConfig::default();
        
        // Test invalid port
        config.server.port = 0;
        assert!(config.validate().is_err());
        
        // Reset and test invalid confidence threshold
        config = SliceConfig::default();
        config.prediction.confidence_threshold = 1.5;
        assert!(config.validate().is_err());
        
        // Reset and test invalid learning rate
        config = SliceConfig::default();
        config.neural_network.learning_rate = -0.1;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_config_serialization() {
        let config = SliceConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SliceConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.server.port, deserialized.server.port);
        assert_eq!(config.prediction.enabled, deserialized.prediction.enabled);
    }
}