use crate::types::{AlertConfig, ModelConfig, ModelType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub model: ModelConfig,
    pub forecasting: ForecastingConfig,
    pub alerts: AlertConfig,
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub request_timeout_ms: u64,
    pub max_request_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub connection_timeout_ms: u64,
    pub query_timeout_ms: u64,
    pub migration_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastingConfig {
    pub default_forecast_horizon_minutes: u32,
    pub max_forecast_horizon_minutes: u32,
    pub min_historical_data_points: usize,
    pub feature_engineering: FeatureEngineeringConfig,
    pub model_ensemble: bool,
    pub model_weights: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    pub enable_lag_features: bool,
    pub lag_periods: Vec<u32>,
    pub enable_moving_averages: bool,
    pub ma_windows: Vec<u32>,
    pub enable_seasonal_decomposition: bool,
    pub seasonal_periods: Vec<u32>,
    pub enable_fourier_features: bool,
    pub fourier_order: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub metrics_port: u16,
    pub log_level: String,
    pub enable_tracing: bool,
    pub jaeger_endpoint: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 50051,
                max_connections: 1000,
                request_timeout_ms: 30000,
                max_request_size: 1024 * 1024, // 1MB
            },
            database: DatabaseConfig {
                url: "postgresql://asa_qos:password@localhost/asa_qos".to_string(),
                max_connections: 20,
                connection_timeout_ms: 5000,
                query_timeout_ms: 10000,
                migration_path: Some("migrations".to_string()),
            },
            model: ModelConfig {
                model_type: ModelType::Lstm,
                input_features: vec![
                    "prb_utilization_dl".to_string(),
                    "active_volte_users".to_string(),
                    "competing_gbr_traffic_mbps".to_string(),
                    "current_jitter_ms".to_string(),
                    "packet_loss_rate".to_string(),
                    "delay_ms".to_string(),
                ],
                forecast_horizon_minutes: 5,
                training_window_hours: 24,
                learning_rate: 0.001,
                batch_size: 32,
                epochs: 100,
            },
            forecasting: ForecastingConfig {
                default_forecast_horizon_minutes: 5,
                max_forecast_horizon_minutes: 60,
                min_historical_data_points: 100,
                feature_engineering: FeatureEngineeringConfig {
                    enable_lag_features: true,
                    lag_periods: vec![1, 2, 3, 5, 10, 15, 30],
                    enable_moving_averages: true,
                    ma_windows: vec![5, 10, 15, 30],
                    enable_seasonal_decomposition: true,
                    seasonal_periods: vec![60, 1440], // 1 hour, 1 day in minutes
                    enable_fourier_features: true,
                    fourier_order: 5,
                },
                model_ensemble: true,
                model_weights: {
                    let mut weights = HashMap::new();
                    weights.insert("lstm".to_string(), 0.4);
                    weights.insert("gru".to_string(), 0.3);
                    weights.insert("transformer".to_string(), 0.2);
                    weights.insert("arima".to_string(), 0.1);
                    weights
                },
            },
            alerts: AlertConfig {
                jitter_threshold_ms: 20.0,
                quality_degradation_threshold: 0.7,
                prediction_confidence_threshold: 0.8,
                alert_cooldown_minutes: 5,
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                metrics_port: 9090,
                log_level: "info".to_string(),
                enable_tracing: true,
                jaeger_endpoint: Some("http://localhost:14268".to_string()),
            },
        }
    }
}

impl Config {
    pub fn from_file(path: &str) -> Result<Self, config::ConfigError> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name(path))
            .add_source(config::Environment::with_prefix("ASA_QOS"))
            .build()?;
        
        settings.try_deserialize()
    }
    
    pub fn validate(&self) -> Result<(), String> {
        if self.server.port == 0 {
            return Err("Server port cannot be 0".to_string());
        }
        
        if self.database.url.is_empty() {
            return Err("Database URL cannot be empty".to_string());
        }
        
        if self.model.input_features.is_empty() {
            return Err("Model must have at least one input feature".to_string());
        }
        
        if self.forecasting.default_forecast_horizon_minutes == 0 {
            return Err("Forecast horizon must be greater than 0".to_string());
        }
        
        if self.forecasting.max_forecast_horizon_minutes 
            < self.forecasting.default_forecast_horizon_minutes {
            return Err("Max forecast horizon must be >= default forecast horizon".to_string());
        }
        
        if self.alerts.jitter_threshold_ms <= 0.0 {
            return Err("Jitter threshold must be positive".to_string());
        }
        
        Ok(())
    }
}