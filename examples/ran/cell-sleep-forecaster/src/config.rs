//! Configuration module for Cell Sleep Mode Forecaster

use std::time::Duration;
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// Main forecasting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastingConfig {
    /// Minimum number of historical data points required for forecasting
    pub min_data_points: usize,
    
    /// Forecast horizon in minutes
    pub forecast_horizon_minutes: u32,
    
    /// Minimum confidence score for sleep window recommendations
    pub min_confidence_score: f64,
    
    /// Maximum risk score for sleep window recommendations
    pub max_risk_score: f64,
    
    /// Low traffic threshold (percentage)
    pub low_traffic_threshold: f64,
    
    /// Minimum sleep window duration in minutes
    pub min_sleep_duration_minutes: u32,
    
    /// Maximum sleep window duration in minutes
    pub max_sleep_duration_minutes: u32,
    
    /// Model retraining interval in hours
    pub model_retrain_interval_hours: u32,
    
    /// Database configuration
    pub database: DatabaseConfig,
    
    /// Network API configuration
    pub network: NetworkConfig,
    
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    
    /// Performance targets
    pub targets: PerformanceTargets,
}

impl Default for ForecastingConfig {
    fn default() -> Self {
        Self {
            min_data_points: 144, // 24 hours of 10-minute intervals
            forecast_horizon_minutes: 60,
            min_confidence_score: 0.8,
            max_risk_score: 0.2,
            low_traffic_threshold: 20.0, // 20% utilization
            min_sleep_duration_minutes: 15,
            max_sleep_duration_minutes: 120,
            model_retrain_interval_hours: 6,
            database: DatabaseConfig::default(),
            network: NetworkConfig::default(),
            monitoring: MonitoringConfig::default(),
            targets: PerformanceTargets::default(),
        }
    }
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub timeout_seconds: u64,
    pub retry_attempts: u32,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "postgresql://localhost/cell_sleep_forecaster".to_string(),
            max_connections: 10,
            timeout_seconds: 30,
            retry_attempts: 3,
        }
    }
}

/// Network API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub base_url: String,
    pub timeout_seconds: u64,
    pub retry_attempts: u32,
    pub rate_limit_requests_per_minute: u32,
    pub auth_token: Option<String>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080/api/v1".to_string(),
            timeout_seconds: 10,
            retry_attempts: 3,
            rate_limit_requests_per_minute: 100,
            auth_token: None,
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enabled: bool,
    pub metrics_port: u16,
    pub alert_thresholds: AlertThresholds,
    pub log_level: String,
    pub prometheus_namespace: String,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics_port: 9090,
            alert_thresholds: AlertThresholds::default(),
            log_level: "info".to_string(),
            prometheus_namespace: "cell_sleep_forecaster".to_string(),
        }
    }
}

/// Alert thresholds for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub mape_threshold: f64,
    pub detection_rate_threshold: f64,
    pub prediction_latency_ms: u64,
    pub error_rate_threshold: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            mape_threshold: 10.0,
            detection_rate_threshold: 95.0,
            prediction_latency_ms: 1000,
            error_rate_threshold: 5.0,
        }
    }
}

/// Performance targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub target_mape: f64,
    pub target_detection_rate: f64,
    pub target_prediction_latency_ms: u64,
    pub target_throughput_rps: u32,
    pub target_availability: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            target_mape: 10.0,
            target_detection_rate: 95.0,
            target_prediction_latency_ms: 1000,
            target_throughput_rps: 1000,
            target_availability: 99.9,
        }
    }
}

impl ForecastingConfig {
    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: ForecastingConfig = toml::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.min_data_points < 10 {
            return Err(anyhow::anyhow!("min_data_points must be at least 10"));
        }
        
        if self.forecast_horizon_minutes == 0 {
            return Err(anyhow::anyhow!("forecast_horizon_minutes must be greater than 0"));
        }
        
        if self.min_confidence_score < 0.0 || self.min_confidence_score > 1.0 {
            return Err(anyhow::anyhow!("min_confidence_score must be between 0.0 and 1.0"));
        }
        
        if self.max_risk_score < 0.0 || self.max_risk_score > 1.0 {
            return Err(anyhow::anyhow!("max_risk_score must be between 0.0 and 1.0"));
        }
        
        if self.low_traffic_threshold < 0.0 || self.low_traffic_threshold > 100.0 {
            return Err(anyhow::anyhow!("low_traffic_threshold must be between 0.0 and 100.0"));
        }
        
        if self.min_sleep_duration_minutes >= self.max_sleep_duration_minutes {
            return Err(anyhow::anyhow!("min_sleep_duration_minutes must be less than max_sleep_duration_minutes"));
        }
        
        if self.database.max_connections == 0 {
            return Err(anyhow::anyhow!("database max_connections must be greater than 0"));
        }
        
        if self.network.rate_limit_requests_per_minute == 0 {
            return Err(anyhow::anyhow!("network rate_limit_requests_per_minute must be greater than 0"));
        }
        
        Ok(())
    }
    
    /// Get forecast interval duration
    pub fn forecast_interval(&self) -> Duration {
        Duration::from_secs(self.forecast_horizon_minutes as u64 * 60)
    }
    
    /// Get model retrain interval duration
    pub fn retrain_interval(&self) -> Duration {
        Duration::from_secs(self.model_retrain_interval_hours as u64 * 3600)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_default_config() {
        let config = ForecastingConfig::default();
        
        assert_eq!(config.min_data_points, 144);
        assert_eq!(config.forecast_horizon_minutes, 60);
        assert_eq!(config.min_confidence_score, 0.8);
        assert_eq!(config.max_risk_score, 0.2);
        assert_eq!(config.low_traffic_threshold, 20.0);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = ForecastingConfig::default();
        
        // Valid config should pass
        assert!(config.validate().is_ok());
        
        // Invalid min_data_points
        config.min_data_points = 5;
        assert!(config.validate().is_err());
        config.min_data_points = 144;
        
        // Invalid forecast_horizon_minutes
        config.forecast_horizon_minutes = 0;
        assert!(config.validate().is_err());
        config.forecast_horizon_minutes = 60;
        
        // Invalid confidence score
        config.min_confidence_score = 1.5;
        assert!(config.validate().is_err());
        config.min_confidence_score = 0.8;
        
        // Invalid sleep duration
        config.min_sleep_duration_minutes = 120;
        config.max_sleep_duration_minutes = 60;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_config_file_operations() {
        let config = ForecastingConfig::default();
        
        // Create temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_str().unwrap();
        
        // Save config
        config.to_file(temp_path).unwrap();
        
        // Load config
        let loaded_config = ForecastingConfig::from_file(temp_path).unwrap();
        
        // Verify values match
        assert_eq!(config.min_data_points, loaded_config.min_data_points);
        assert_eq!(config.forecast_horizon_minutes, loaded_config.forecast_horizon_minutes);
        assert_eq!(config.min_confidence_score, loaded_config.min_confidence_score);
    }
    
    #[test]
    fn test_duration_calculations() {
        let config = ForecastingConfig::default();
        
        assert_eq!(config.forecast_interval(), Duration::from_secs(3600)); // 60 minutes
        assert_eq!(config.retrain_interval(), Duration::from_secs(21600)); // 6 hours
    }
}