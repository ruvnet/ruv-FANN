//! Configuration management for DNI-CAP-01 Capacity Cliff Forecaster

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::{ConfigurationError, ConfigurationResult};

/// Main configuration structure for the capacity forecaster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// General system configuration
    pub system: SystemConfig,
    /// Database configuration
    pub database: DatabaseConfig,
    /// Forecasting model configurations
    pub models: ModelConfigs,
    /// Monitoring and alerting configuration
    pub monitoring: MonitoringConfig,
    /// API service configuration
    pub api: ApiConfig,
    /// Capacity planning specific settings
    pub capacity_planning: CapacityPlanningConfig,
}

/// System-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Service name
    pub service_name: String,
    /// Service version
    pub version: String,
    /// Environment (dev, staging, prod)
    pub environment: String,
    /// Log level
    pub log_level: String,
    /// Number of worker threads
    pub worker_threads: usize,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval in seconds
    pub metrics_interval_seconds: u64,
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Database URL
    pub url: String,
    /// Maximum number of connections
    pub max_connections: u32,
    /// Connection timeout in seconds
    pub connect_timeout_seconds: u64,
    /// Query timeout in seconds
    pub query_timeout_seconds: u64,
    /// Enable query logging
    pub enable_query_logging: bool,
    /// Database migration configuration
    pub migration: MigrationConfig,
}

/// Database migration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConfig {
    /// Enable automatic migrations
    pub auto_migrate: bool,
    /// Backup before migration
    pub backup_before_migration: bool,
    /// Migration timeout in seconds
    pub migration_timeout_seconds: u64,
}

/// Model configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigs {
    /// LSTM model configuration
    pub lstm: LSTMConfig,
    /// ARIMA model configuration
    pub arima: ARIMAConfig,
    /// Polynomial regression configuration
    pub polynomial: PolynomialConfig,
    /// Exponential smoothing configuration
    pub exponential_smoothing: ExponentialSmoothingConfig,
    /// Ensemble model configuration
    pub ensemble: EnsembleConfig,
    /// Neural forecast configuration
    pub neural_forecast: NeuralForecastConfig,
}

/// LSTM model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMConfig {
    /// Number of LSTM layers
    pub layers: usize,
    /// Hidden layer size
    pub hidden_size: usize,
    /// Sequence length for input
    pub sequence_length: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Minimum improvement for early stopping
    pub min_improvement: f64,
}

/// ARIMA model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ARIMAConfig {
    /// Auto-regressive order (p)
    pub p: usize,
    /// Differencing order (d)
    pub d: usize,
    /// Moving average order (q)
    pub q: usize,
    /// Seasonal auto-regressive order (P)
    pub seasonal_p: usize,
    /// Seasonal differencing order (D)
    pub seasonal_d: usize,
    /// Seasonal moving average order (Q)
    pub seasonal_q: usize,
    /// Seasonal period
    pub seasonal_period: usize,
    /// Include trend
    pub include_trend: bool,
    /// Maximum iterations for fitting
    pub max_iterations: usize,
}

/// Polynomial regression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialConfig {
    /// Polynomial degree
    pub degree: usize,
    /// Regularization parameter
    pub regularization: f64,
    /// Include interaction terms
    pub include_interactions: bool,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Feature selection method
    pub feature_selection: String,
}

/// Exponential smoothing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialSmoothingConfig {
    /// Smoothing parameter alpha
    pub alpha: f64,
    /// Trend smoothing parameter beta
    pub beta: f64,
    /// Seasonal smoothing parameter gamma
    pub gamma: f64,
    /// Seasonal period
    pub seasonal_period: usize,
    /// Trend type (additive, multiplicative, none)
    pub trend_type: String,
    /// Seasonal type (additive, multiplicative, none)
    pub seasonal_type: String,
    /// Damping parameter
    pub damping_parameter: f64,
}

/// Ensemble model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Model weights
    pub model_weights: HashMap<String, f64>,
    /// Voting method (average, weighted_average, median)
    pub voting_method: String,
    /// Confidence threshold for predictions
    pub confidence_threshold: f64,
    /// Enable dynamic weight adjustment
    pub enable_dynamic_weights: bool,
    /// Performance window for weight adjustment
    pub performance_window_days: usize,
}

/// Neural forecast configuration using ruv-FANN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralForecastConfig {
    /// Network architecture
    pub architecture: Vec<usize>,
    /// Activation function
    pub activation_function: String,
    /// Training algorithm
    pub training_algorithm: String,
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum
    pub momentum: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Maximum training epochs
    pub max_epochs: usize,
    /// Target error for training
    pub target_error: f64,
    /// Enable bit-fail limit
    pub enable_bit_fail_limit: bool,
    /// Bit-fail limit
    pub bit_fail_limit: usize,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Metrics endpoint
    pub metrics_endpoint: String,
    /// Health check endpoint
    pub health_endpoint: String,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Notification settings
    pub notifications: NotificationConfig,
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Forecast accuracy threshold
    pub forecast_accuracy_threshold: f64,
    /// Model performance threshold
    pub model_performance_threshold: f64,
    /// Data quality threshold
    pub data_quality_threshold: f64,
    /// Capacity breach warning threshold (months ahead)
    pub capacity_breach_warning_months: f64,
    /// System latency threshold (milliseconds)
    pub system_latency_threshold_ms: u64,
    /// Error rate threshold
    pub error_rate_threshold: f64,
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Enable email notifications
    pub enable_email: bool,
    /// Email recipients
    pub email_recipients: Vec<String>,
    /// Enable Slack notifications
    pub enable_slack: bool,
    /// Slack webhook URL
    pub slack_webhook_url: Option<String>,
    /// Enable SMS notifications
    pub enable_sms: bool,
    /// SMS configuration
    pub sms_config: Option<SmsConfig>,
}

/// SMS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmsConfig {
    /// SMS provider
    pub provider: String,
    /// API key
    pub api_key: String,
    /// Phone numbers
    pub phone_numbers: Vec<String>,
}

/// API service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Service host
    pub host: String,
    /// Service port
    pub port: u16,
    /// Enable TLS
    pub enable_tls: bool,
    /// TLS certificate path
    pub tls_cert_path: Option<String>,
    /// TLS private key path
    pub tls_key_path: Option<String>,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
    /// Maximum request size in bytes
    pub max_request_size_bytes: usize,
    /// Enable CORS
    pub enable_cors: bool,
    /// CORS allowed origins
    pub cors_allowed_origins: Vec<String>,
    /// Authentication configuration
    pub auth: AuthConfig,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enable authentication
    pub enabled: bool,
    /// Authentication type (jwt, api_key, oauth2)
    pub auth_type: String,
    /// JWT configuration
    pub jwt: Option<JwtConfig>,
    /// API key configuration
    pub api_key: Option<ApiKeyConfig>,
}

/// JWT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtConfig {
    /// JWT secret key
    pub secret: String,
    /// Token expiration time in seconds
    pub expiration_seconds: u64,
    /// Issuer
    pub issuer: String,
    /// Audience
    pub audience: String,
}

/// API key configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    /// Valid API keys
    pub valid_keys: Vec<String>,
    /// Key rotation interval in days
    pub rotation_interval_days: u64,
}

/// Capacity planning specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityPlanningConfig {
    /// Default capacity threshold for breach detection
    pub default_capacity_threshold: f64,
    /// Maximum forecast horizon in months
    pub max_forecast_horizon_months: usize,
    /// Minimum historical data points required
    pub min_historical_data_points: usize,
    /// Target forecast accuracy (±months)
    pub target_accuracy_months: f64,
    /// Confidence level for predictions
    pub confidence_level: f64,
    /// Growth trend analysis settings
    pub growth_trend: GrowthTrendConfig,
    /// Investment planning settings
    pub investment_planning: InvestmentPlanningConfig,
}

/// Growth trend analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthTrendConfig {
    /// Minimum R-squared for trend validity
    pub min_r_squared: f64,
    /// Seasonal analysis window (months)
    pub seasonal_window_months: usize,
    /// Volatility calculation window (months)
    pub volatility_window_months: usize,
    /// Trend significance threshold
    pub trend_significance_threshold: f64,
}

/// Investment planning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestmentPlanningConfig {
    /// Default ROI threshold
    pub default_roi_threshold: f64,
    /// Risk tolerance levels
    pub risk_tolerance: HashMap<String, f64>,
    /// Cost estimation models
    pub cost_models: HashMap<String, CostModelConfig>,
    /// Benefit calculation parameters
    pub benefit_parameters: BenefitParametersConfig,
}

/// Cost model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModelConfig {
    /// Base cost
    pub base_cost: f64,
    /// Cost per unit
    pub cost_per_unit: f64,
    /// Scaling factor
    pub scaling_factor: f64,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
}

/// Benefit calculation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenefitParametersConfig {
    /// Revenue per capacity unit
    pub revenue_per_capacity_unit: f64,
    /// Cost savings per quality improvement
    pub cost_savings_per_quality_improvement: f64,
    /// Customer satisfaction impact factor
    pub customer_satisfaction_impact_factor: f64,
    /// Churn reduction factor
    pub churn_reduction_factor: f64,
}

impl Config {
    /// Load configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> ConfigurationResult<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|_| {
            ConfigurationError::file_not_found(path.as_ref().to_string_lossy().to_string())
        })?;

        let config: Config = toml::from_str(&content)
            .map_err(|e| ConfigurationError::parse_error(e.to_string()))?;

        config.validate()?;
        Ok(config)
    }

    /// Load configuration from environment variables
    pub fn from_env() -> ConfigurationResult<Self> {
        let mut config = Self::default();

        // Override with environment variables
        if let Ok(db_url) = std::env::var("DATABASE_URL") {
            config.database.url = db_url;
        }

        if let Ok(log_level) = std::env::var("LOG_LEVEL") {
            config.system.log_level = log_level;
        }

        if let Ok(port) = std::env::var("PORT") {
            config.api.port = port.parse().map_err(|_| {
                ConfigurationError::invalid_value("PORT".to_string(), port)
            })?;
        }

        config.validate()?;
        Ok(config)
    }

    /// Validate configuration
    pub fn validate(&self) -> ConfigurationResult<()> {
        // Validate system configuration
        if self.system.worker_threads == 0 {
            return Err(ConfigurationError::invalid_value(
                "system.worker_threads".to_string(),
                "0".to_string(),
            ));
        }

        // Validate database configuration
        if self.database.url.is_empty() {
            return Err(ConfigurationError::missing_required(
                "database.url".to_string(),
            ));
        }

        if self.database.max_connections == 0 {
            return Err(ConfigurationError::invalid_value(
                "database.max_connections".to_string(),
                "0".to_string(),
            ));
        }

        // Validate capacity planning configuration
        if self.capacity_planning.default_capacity_threshold <= 0.0
            || self.capacity_planning.default_capacity_threshold > 1.0
        {
            return Err(ConfigurationError::invalid_value(
                "capacity_planning.default_capacity_threshold".to_string(),
                self.capacity_planning.default_capacity_threshold.to_string(),
            ));
        }

        if self.capacity_planning.max_forecast_horizon_months == 0 {
            return Err(ConfigurationError::invalid_value(
                "capacity_planning.max_forecast_horizon_months".to_string(),
                "0".to_string(),
            ));
        }

        // Validate model configurations
        if self.models.lstm.hidden_size == 0 {
            return Err(ConfigurationError::invalid_value(
                "models.lstm.hidden_size".to_string(),
                "0".to_string(),
            ));
        }

        if self.models.polynomial.degree == 0 {
            return Err(ConfigurationError::invalid_value(
                "models.polynomial.degree".to_string(),
                "0".to_string(),
            ));
        }

        Ok(())
    }

    /// Get database URL
    pub fn database_url(&self) -> &str {
        &self.database.url
    }

    /// Get API listen address
    pub fn api_listen_address(&self) -> String {
        format!("{}:{}", self.api.host, self.api.port)
    }

    /// Check if monitoring is enabled
    pub fn is_monitoring_enabled(&self) -> bool {
        self.monitoring.enabled
    }

    /// Get capacity threshold
    pub fn capacity_threshold(&self) -> f64 {
        self.capacity_planning.default_capacity_threshold
    }

    /// Get maximum forecast horizon
    pub fn max_forecast_horizon(&self) -> usize {
        self.capacity_planning.max_forecast_horizon_months
    }

    /// Get minimum historical data points
    pub fn min_historical_data_points(&self) -> usize {
        self.capacity_planning.min_historical_data_points
    }

    /// Get target accuracy
    pub fn target_accuracy(&self) -> f64 {
        self.capacity_planning.target_accuracy_months
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            database: DatabaseConfig::default(),
            models: ModelConfigs::default(),
            monitoring: MonitoringConfig::default(),
            api: ApiConfig::default(),
            capacity_planning: CapacityPlanningConfig::default(),
        }
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            service_name: "dni-cap-01".to_string(),
            version: crate::VERSION.to_string(),
            environment: "dev".to_string(),
            log_level: "info".to_string(),
            worker_threads: num_cpus::get(),
            enable_metrics: true,
            metrics_interval_seconds: 60,
        }
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "postgres://localhost/dni_cap_01".to_string(),
            max_connections: 10,
            connect_timeout_seconds: 30,
            query_timeout_seconds: 30,
            enable_query_logging: false,
            migration: MigrationConfig::default(),
        }
    }
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            auto_migrate: true,
            backup_before_migration: true,
            migration_timeout_seconds: 300,
        }
    }
}

impl Default for ModelConfigs {
    fn default() -> Self {
        Self {
            lstm: LSTMConfig::default(),
            arima: ARIMAConfig::default(),
            polynomial: PolynomialConfig::default(),
            exponential_smoothing: ExponentialSmoothingConfig::default(),
            ensemble: EnsembleConfig::default(),
            neural_forecast: NeuralForecastConfig::default(),
        }
    }
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            layers: 2,
            hidden_size: 128,
            sequence_length: 12,
            dropout: 0.2,
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            early_stopping_patience: 10,
            min_improvement: 0.001,
        }
    }
}

impl Default for ARIMAConfig {
    fn default() -> Self {
        Self {
            p: 2,
            d: 1,
            q: 2,
            seasonal_p: 1,
            seasonal_d: 1,
            seasonal_q: 1,
            seasonal_period: 12,
            include_trend: true,
            max_iterations: 1000,
        }
    }
}

impl Default for PolynomialConfig {
    fn default() -> Self {
        Self {
            degree: 3,
            regularization: 0.1,
            include_interactions: false,
            cv_folds: 5,
            feature_selection: "none".to_string(),
        }
    }
}

impl Default for ExponentialSmoothingConfig {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            beta: 0.1,
            gamma: 0.1,
            seasonal_period: 12,
            trend_type: "additive".to_string(),
            seasonal_type: "additive".to_string(),
            damping_parameter: 0.98,
        }
    }
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        let mut model_weights = HashMap::new();
        model_weights.insert("lstm".to_string(), 0.3);
        model_weights.insert("arima".to_string(), 0.25);
        model_weights.insert("polynomial".to_string(), 0.2);
        model_weights.insert("exponential_smoothing".to_string(), 0.15);
        model_weights.insert("neural_forecast".to_string(), 0.1);

        Self {
            model_weights,
            voting_method: "weighted_average".to_string(),
            confidence_threshold: 0.7,
            enable_dynamic_weights: true,
            performance_window_days: 30,
        }
    }
}

impl Default for NeuralForecastConfig {
    fn default() -> Self {
        Self {
            architecture: vec![24, 48, 24, 12, 1],
            activation_function: "sigmoid".to_string(),
            training_algorithm: "rprop".to_string(),
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0001,
            max_epochs: 1000,
            target_error: 0.001,
            enable_bit_fail_limit: true,
            bit_fail_limit: 10,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics_endpoint: "/metrics".to_string(),
            health_endpoint: "/health".to_string(),
            alert_thresholds: AlertThresholds::default(),
            notifications: NotificationConfig::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            forecast_accuracy_threshold: 0.8,
            model_performance_threshold: 0.7,
            data_quality_threshold: 0.9,
            capacity_breach_warning_months: 3.0,
            system_latency_threshold_ms: 1000,
            error_rate_threshold: 0.05,
        }
    }
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enable_email: false,
            email_recipients: vec![],
            enable_slack: false,
            slack_webhook_url: None,
            enable_sms: false,
            sms_config: None,
        }
    }
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            enable_tls: false,
            tls_cert_path: None,
            tls_key_path: None,
            request_timeout_seconds: 30,
            max_request_size_bytes: 1024 * 1024, // 1MB
            enable_cors: true,
            cors_allowed_origins: vec!["*".to_string()],
            auth: AuthConfig::default(),
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            auth_type: "api_key".to_string(),
            jwt: None,
            api_key: None,
        }
    }
}

impl Default for CapacityPlanningConfig {
    fn default() -> Self {
        Self {
            default_capacity_threshold: crate::DEFAULT_CAPACITY_THRESHOLD,
            max_forecast_horizon_months: crate::MAX_FORECAST_HORIZON,
            min_historical_data_points: crate::MIN_HISTORICAL_POINTS,
            target_accuracy_months: crate::TARGET_ACCURACY_MONTHS,
            confidence_level: 0.95,
            growth_trend: GrowthTrendConfig::default(),
            investment_planning: InvestmentPlanningConfig::default(),
        }
    }
}

impl Default for GrowthTrendConfig {
    fn default() -> Self {
        Self {
            min_r_squared: 0.5,
            seasonal_window_months: 12,
            volatility_window_months: 6,
            trend_significance_threshold: 0.05,
        }
    }
}

impl Default for InvestmentPlanningConfig {
    fn default() -> Self {
        let mut risk_tolerance = HashMap::new();
        risk_tolerance.insert("low".to_string(), 0.1);
        risk_tolerance.insert("medium".to_string(), 0.3);
        risk_tolerance.insert("high".to_string(), 0.5);
        risk_tolerance.insert("critical".to_string(), 0.8);

        let mut cost_models = HashMap::new();
        cost_models.insert(
            "new_cell_site".to_string(),
            CostModelConfig {
                base_cost: 250000.0,
                cost_per_unit: 1000.0,
                scaling_factor: 1.2,
                parameters: HashMap::new(),
            },
        );
        cost_models.insert(
            "equipment_upgrade".to_string(),
            CostModelConfig {
                base_cost: 50000.0,
                cost_per_unit: 500.0,
                scaling_factor: 1.1,
                parameters: HashMap::new(),
            },
        );

        Self {
            default_roi_threshold: 2.0,
            risk_tolerance,
            cost_models,
            benefit_parameters: BenefitParametersConfig::default(),
        }
    }
}

impl Default for BenefitParametersConfig {
    fn default() -> Self {
        Self {
            revenue_per_capacity_unit: 1000.0,
            cost_savings_per_quality_improvement: 500.0,
            customer_satisfaction_impact_factor: 0.1,
            churn_reduction_factor: 0.05,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.system.service_name, "dni-cap-01");
        assert_eq!(config.capacity_planning.default_capacity_threshold, 0.8);
        assert_eq!(config.capacity_planning.max_forecast_horizon_months, 24);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        
        // Test invalid capacity threshold
        config.capacity_planning.default_capacity_threshold = 1.5;
        assert!(config.validate().is_err());
        
        // Test invalid worker threads
        config.capacity_planning.default_capacity_threshold = 0.8;
        config.system.worker_threads = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_from_file() {
        let toml_content = r#"
[system]
service_name = "test-service"
version = "1.0.0"
environment = "test"
log_level = "debug"
worker_threads = 4
enable_metrics = true
metrics_interval_seconds = 30

[database]
url = "postgres://test:test@localhost/test"
max_connections = 5
connect_timeout_seconds = 10
query_timeout_seconds = 20
enable_query_logging = true

[capacity_planning]
default_capacity_threshold = 0.75
max_forecast_horizon_months = 18
min_historical_data_points = 10
target_accuracy_months = 1.5
confidence_level = 0.9
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();

        let config = Config::from_file(temp_file.path()).unwrap();
        assert_eq!(config.system.service_name, "test-service");
        assert_eq!(config.database.max_connections, 5);
        assert_eq!(config.capacity_planning.default_capacity_threshold, 0.75);
        assert_eq!(config.capacity_planning.max_forecast_horizon_months, 18);
    }

    #[test]
    fn test_config_methods() {
        let config = Config::default();
        
        assert_eq!(config.database_url(), "postgres://localhost/dni_cap_01");
        assert_eq!(config.api_listen_address(), "0.0.0.0:8080");
        assert!(config.is_monitoring_enabled());
        assert_eq!(config.capacity_threshold(), 0.8);
        assert_eq!(config.max_forecast_horizon(), 24);
        assert_eq!(config.min_historical_data_points(), 12);
        assert_eq!(config.target_accuracy(), 2.0);
    }
}