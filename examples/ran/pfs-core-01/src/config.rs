use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub server: ServerConfig,
    pub storage: StorageConfig,
    pub training: TrainingConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub request_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub models_directory: PathBuf,
    pub max_models: usize,
    pub cleanup_interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub max_concurrent_training: usize,
    pub default_max_epochs: u32,
    pub default_learning_rate: f64,
    pub default_desired_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 50051,
                max_connections: 1000,
                request_timeout_seconds: 300,
            },
            storage: StorageConfig {
                models_directory: PathBuf::from("./models"),
                max_models: 100,
                cleanup_interval_seconds: 3600,
            },
            training: TrainingConfig {
                max_concurrent_training: 4,
                default_max_epochs: 10000,
                default_learning_rate: 0.01,
                default_desired_error: 0.001,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: "json".to_string(),
            },
        }
    }
}

impl ServiceConfig {
    pub fn from_file(path: &str) -> Result<Self, config::ConfigError> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name(path))
            .add_source(config::Environment::with_prefix("PFS_CORE"))
            .build()?;
        
        settings.try_deserialize()
    }
    
    pub fn validate(&self) -> Result<(), String> {
        if self.server.port == 0 {
            return Err("Server port cannot be 0".to_string());
        }
        
        if self.storage.max_models == 0 {
            return Err("Max models cannot be 0".to_string());
        }
        
        if self.training.max_concurrent_training == 0 {
            return Err("Max concurrent training cannot be 0".to_string());
        }
        
        if self.training.default_learning_rate <= 0.0 {
            return Err("Default learning rate must be positive".to_string());
        }
        
        Ok(())
    }
}