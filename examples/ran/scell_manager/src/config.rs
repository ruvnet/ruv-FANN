//! Configuration management for the SCell Manager

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;

/// Main configuration for the SCell Manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCellManagerConfig {
    pub server: ServerConfig,
    pub model_config: ModelConfig,
    pub data_config: DataConfig,
    pub metrics_config: MetricsConfig,
    pub logging_config: LoggingConfig,
}

impl Default for SCellManagerConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            model_config: ModelConfig::default(),
            data_config: DataConfig::default(),
            metrics_config: MetricsConfig::default(),
            logging_config: LoggingConfig::default(),
        }
    }
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub bind_address: SocketAddr,
    pub max_connections: usize,
    pub request_timeout_seconds: u64,
    pub enable_tls: bool,
    pub tls_cert_path: Option<PathBuf>,
    pub tls_key_path: Option<PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:50051".parse().unwrap(),
            max_connections: 1000,
            request_timeout_seconds: 30,
            enable_tls: false,
            tls_cert_path: None,
            tls_key_path: None,
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_dir: PathBuf,
    pub default_model_id: String,
    pub feature_window_size: usize,
    pub prediction_horizon_seconds: i32,
    pub confidence_threshold: f32,
    pub throughput_threshold_mbps: f32,
    pub neural_network: NeuralNetworkConfig,
    pub training: TrainingConfig,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("./models"),
            default_model_id: "scell_predictor_v1".to_string(),
            feature_window_size: 10,
            prediction_horizon_seconds: 30,
            confidence_threshold: 0.7,
            throughput_threshold_mbps: 100.0,
            neural_network: NeuralNetworkConfig::default(),
            training: TrainingConfig::default(),
        }
    }
}

/// Neural network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkConfig {
    pub input_neurons: usize,
    pub hidden_layers: Vec<usize>,
    pub output_neurons: usize,
    pub activation_function: String,
    pub training_algorithm: String,
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
}

impl Default for NeuralNetworkConfig {
    fn default() -> Self {
        Self {
            input_neurons: 10, // 7 base features + 3 time features
            hidden_layers: vec![64, 32, 16],
            output_neurons: 2, // binary classification + throughput prediction
            activation_function: "sigmoid".to_string(),
            training_algorithm: "rprop".to_string(),
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 0.0001,
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub max_epochs: i32,
    pub min_error: f32,
    pub batch_size: usize,
    pub validation_split: f32,
    pub early_stopping_patience: i32,
    pub shuffle_training_data: bool,
    pub save_best_model: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 1000,
            min_error: 0.01,
            batch_size: 32,
            validation_split: 0.2,
            early_stopping_patience: 50,
            shuffle_training_data: true,
            save_best_model: true,
        }
    }
}

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub data_dir: PathBuf,
    pub training_data_path: PathBuf,
    pub validation_data_path: PathBuf,
    pub cache_size: usize,
    pub max_historical_records: usize,
    pub data_retention_days: i32,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            training_data_path: PathBuf::from("./data/training_data.parquet"),
            validation_data_path: PathBuf::from("./data/validation_data.parquet"),
            cache_size: 10000,
            max_historical_records: 100000,
            data_retention_days: 30,
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enable_prometheus: bool,
    pub prometheus_port: u16,
    pub metrics_prefix: String,
    pub collection_interval_seconds: u64,
    pub enable_detailed_metrics: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enable_prometheus: true,
            prometheus_port: 9090,
            metrics_prefix: "scell_manager".to_string(),
            collection_interval_seconds: 60,
            enable_detailed_metrics: true,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub output: LogOutput,
    pub enable_structured_logging: bool,
    pub log_file_path: Option<PathBuf>,
    pub max_log_file_size_mb: usize,
    pub max_log_files: usize,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "json".to_string(),
            output: LogOutput::Console,
            enable_structured_logging: true,
            log_file_path: Some(PathBuf::from("./logs/scell_manager.log")),
            max_log_file_size_mb: 100,
            max_log_files: 10,
        }
    }
}

/// Log output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogOutput {
    Console,
    File,
    Both,
}

impl SCellManagerConfig {
    /// Load configuration from file
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let config_str = std::fs::read_to_string(path)?;
        let config: SCellManagerConfig = serde_json::from_str(&config_str)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> anyhow::Result<()> {
        let config_str = serde_json::to_string_pretty(self)?;
        std::fs::write(path, config_str)?;
        Ok(())
    }
    
    /// Create configuration from environment variables
    pub fn from_env() -> anyhow::Result<Self> {
        let mut config = Self::default();
        
        // Override with environment variables
        if let Ok(addr) = std::env::var("SCELL_MANAGER_BIND_ADDRESS") {
            config.server.bind_address = addr.parse()?;
        }
        
        if let Ok(model_dir) = std::env::var("SCELL_MANAGER_MODEL_DIR") {
            config.model_config.model_dir = PathBuf::from(model_dir);
        }
        
        if let Ok(data_dir) = std::env::var("SCELL_MANAGER_DATA_DIR") {
            config.data_config.data_dir = PathBuf::from(data_dir);
        }
        
        if let Ok(log_level) = std::env::var("SCELL_MANAGER_LOG_LEVEL") {
            config.logging_config.level = log_level;
        }
        
        Ok(config)
    }
    
    /// Validate configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        // Validate server configuration
        if self.server.max_connections == 0 {
            return Err(anyhow::anyhow!("Max connections must be greater than 0"));
        }
        
        // Validate model configuration
        if self.model_config.neural_network.input_neurons == 0 {
            return Err(anyhow::anyhow!("Input neurons must be greater than 0"));
        }
        
        if self.model_config.neural_network.hidden_layers.is_empty() {
            return Err(anyhow::anyhow!("At least one hidden layer is required"));
        }
        
        if self.model_config.confidence_threshold < 0.0 || self.model_config.confidence_threshold > 1.0 {
            return Err(anyhow::anyhow!("Confidence threshold must be between 0.0 and 1.0"));
        }
        
        // Validate training configuration
        if self.model_config.training.validation_split < 0.0 || self.model_config.training.validation_split > 1.0 {
            return Err(anyhow::anyhow!("Validation split must be between 0.0 and 1.0"));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = SCellManagerConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_serialization() {
        let config = SCellManagerConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SCellManagerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.server.bind_address, deserialized.server.bind_address);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = SCellManagerConfig::default();
        config.server.max_connections = 0;
        assert!(config.validate().is_err());
        
        config.server.max_connections = 100;
        config.model_config.confidence_threshold = 1.5;
        assert!(config.validate().is_err());
    }
}