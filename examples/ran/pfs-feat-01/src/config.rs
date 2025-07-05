use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the feature engineering agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineConfig {
    /// Service configuration
    pub service: ServiceConfig,
    
    /// Default feature configuration
    pub default_features: FeatureConfig,
    
    /// Data processing configuration
    pub processing: ProcessingConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// gRPC server host
    pub host: String,
    
    /// gRPC server port
    pub port: u16,
    
    /// Service name
    pub name: String,
    
    /// Service version
    pub version: String,
    
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
}

/// Feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Lag features configuration
    pub lag_features: LagFeatureConfig,
    
    /// Rolling window statistics configuration
    pub rolling_window: RollingWindowConfig,
    
    /// Time-based features configuration
    pub time_features: TimeBasedFeatureConfig,
    
    /// Output configuration
    pub output: OutputConfig,
}

/// Lag features configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagFeatureConfig {
    /// Whether lag features are enabled
    pub enabled: bool,
    
    /// List of lag periods (e.g., [1, 2, 3, 6, 12, 24])
    pub lag_periods: Vec<i32>,
    
    /// Target columns for lag features
    pub target_columns: Vec<String>,
}

/// Rolling window statistics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingWindowConfig {
    /// Whether rolling window features are enabled
    pub enabled: bool,
    
    /// Window sizes (e.g., [3, 6, 12, 24, 48])
    pub window_sizes: Vec<i32>,
    
    /// Statistics to calculate (e.g., ["mean", "std", "min", "max", "median"])
    pub statistics: Vec<String>,
    
    /// Target columns for rolling window features
    pub target_columns: Vec<String>,
}

/// Time-based features configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBasedFeatureConfig {
    /// Whether time-based features are enabled
    pub enabled: bool,
    
    /// List of time features to generate
    pub features: Vec<String>,
    
    /// Name of the timestamp column
    pub timestamp_column: String,
    
    /// Timezone for time calculations
    pub timezone: String,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output format ("parquet", "csv", "json")
    pub format: String,
    
    /// Compression method ("snappy", "gzip", "lz4", "brotli")
    pub compression: String,
    
    /// Whether to include metadata in output
    pub include_metadata: bool,
    
    /// Whether to validate schema before writing
    pub validate_schema: bool,
}

/// Data processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Maximum number of parallel jobs
    pub max_parallel_jobs: usize,
    
    /// Batch size for processing
    pub batch_size: usize,
    
    /// Memory limit in MB
    pub memory_limit_mb: u64,
    
    /// Temporary directory for processing
    pub temp_directory: String,
    
    /// Whether to enable streaming mode
    pub streaming_mode: bool,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level ("trace", "debug", "info", "warn", "error")
    pub level: String,
    
    /// Log format ("json", "text")
    pub format: String,
    
    /// Output destination ("stdout", "stderr", "file")
    pub output: String,
    
    /// Log file path (if output is "file")
    pub file_path: Option<String>,
}

impl Default for FeatureEngineConfig {
    fn default() -> Self {
        Self {
            service: ServiceConfig {
                host: "127.0.0.1".to_string(),
                port: 50051,
                name: "feature-engineering-service".to_string(),
                version: "0.1.0".to_string(),
                max_concurrent_requests: 100,
                request_timeout_seconds: 300,
            },
            default_features: FeatureConfig {
                lag_features: LagFeatureConfig {
                    enabled: true,
                    lag_periods: vec![1, 2, 3, 6, 12, 24],
                    target_columns: vec!["kpi_value".to_string()],
                },
                rolling_window: RollingWindowConfig {
                    enabled: true,
                    window_sizes: vec![3, 6, 12, 24, 48],
                    statistics: vec![
                        "mean".to_string(),
                        "std".to_string(),
                        "min".to_string(),
                        "max".to_string(),
                        "median".to_string(),
                    ],
                    target_columns: vec!["kpi_value".to_string()],
                },
                time_features: TimeBasedFeatureConfig {
                    enabled: true,
                    features: vec![
                        "hour_of_day".to_string(),
                        "day_of_week".to_string(),
                        "is_weekend".to_string(),
                    ],
                    timestamp_column: "timestamp".to_string(),
                    timezone: "UTC".to_string(),
                },
                output: OutputConfig {
                    format: "parquet".to_string(),
                    compression: "snappy".to_string(),
                    include_metadata: true,
                    validate_schema: true,
                },
            },
            processing: ProcessingConfig {
                max_parallel_jobs: num_cpus::get(),
                batch_size: 1000,
                memory_limit_mb: 2048,
                temp_directory: "/tmp/feature-engineering".to_string(),
                streaming_mode: false,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: "json".to_string(),
                output: "stdout".to_string(),
                file_path: None,
            },
        }
    }
}

impl FeatureEngineConfig {
    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config_str = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&config_str)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let config_str = toml::to_string_pretty(self)?;
        std::fs::write(path, config_str)?;
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate service configuration
        if self.service.host.is_empty() {
            return Err("Service host cannot be empty".to_string());
        }
        
        if self.service.port == 0 {
            return Err("Service port must be greater than 0".to_string());
        }
        
        // Validate feature configuration
        if self.default_features.lag_features.enabled && self.default_features.lag_features.lag_periods.is_empty() {
            return Err("Lag periods cannot be empty when lag features are enabled".to_string());
        }
        
        if self.default_features.rolling_window.enabled && self.default_features.rolling_window.window_sizes.is_empty() {
            return Err("Window sizes cannot be empty when rolling window features are enabled".to_string());
        }
        
        if self.default_features.time_features.enabled && self.default_features.time_features.timestamp_column.is_empty() {
            return Err("Timestamp column cannot be empty when time features are enabled".to_string());
        }
        
        // Validate output format
        match self.default_features.output.format.as_str() {
            "parquet" | "csv" | "json" => {}
            _ => return Err(format!("Unsupported output format: {}", self.default_features.output.format)),
        }
        
        // Validate compression
        match self.default_features.output.compression.as_str() {
            "snappy" | "gzip" | "lz4" | "brotli" => {}
            _ => return Err(format!("Unsupported compression: {}", self.default_features.output.compression)),
        }
        
        // Validate processing configuration
        if self.processing.max_parallel_jobs == 0 {
            return Err("Max parallel jobs must be greater than 0".to_string());
        }
        
        if self.processing.batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }
        
        Ok(())
    }
}

/// RAN-specific feature configurations
pub struct RanFeatureTemplates;

impl RanFeatureTemplates {
    /// Configuration for RAN KPI features
    pub fn ran_kpi_features() -> FeatureConfig {
        FeatureConfig {
            lag_features: LagFeatureConfig {
                enabled: true,
                lag_periods: vec![1, 2, 3, 6, 12, 24, 48], // Up to 48 hours for daily patterns
                target_columns: vec![
                    "prb_utilization_dl".to_string(),
                    "prb_utilization_ul".to_string(),
                    "active_users".to_string(),
                    "throughput_dl".to_string(),
                    "throughput_ul".to_string(),
                    "rsrp_avg".to_string(),
                    "sinr_avg".to_string(),
                ],
            },
            rolling_window: RollingWindowConfig {
                enabled: true,
                window_sizes: vec![3, 6, 12, 24, 48, 72], // Up to 72 hours for weekly patterns
                statistics: vec![
                    "mean".to_string(),
                    "std".to_string(),
                    "min".to_string(),
                    "max".to_string(),
                    "median".to_string(),
                    "q25".to_string(),
                    "q75".to_string(),
                ],
                target_columns: vec![
                    "prb_utilization_dl".to_string(),
                    "prb_utilization_ul".to_string(),
                    "active_users".to_string(),
                    "throughput_dl".to_string(),
                    "throughput_ul".to_string(),
                    "rsrp_avg".to_string(),
                    "sinr_avg".to_string(),
                ],
            },
            time_features: TimeBasedFeatureConfig {
                enabled: true,
                features: vec![
                    "hour_of_day".to_string(),
                    "day_of_week".to_string(),
                    "is_weekend".to_string(),
                    "is_business_hour".to_string(),
                    "is_peak_hour".to_string(),
                ],
                timestamp_column: "timestamp".to_string(),
                timezone: "UTC".to_string(),
            },
            output: OutputConfig {
                format: "parquet".to_string(),
                compression: "snappy".to_string(),
                include_metadata: true,
                validate_schema: true,
            },
        }
    }
    
    /// Configuration for handover prediction features
    pub fn handover_prediction_features() -> FeatureConfig {
        FeatureConfig {
            lag_features: LagFeatureConfig {
                enabled: true,
                lag_periods: vec![1, 2, 3, 5, 10, 15, 30], // Short-term lags for mobility
                target_columns: vec![
                    "serving_rsrp".to_string(),
                    "serving_sinr".to_string(),
                    "neighbor_rsrp_best".to_string(),
                    "ue_speed_kmh".to_string(),
                    "cqi".to_string(),
                ],
            },
            rolling_window: RollingWindowConfig {
                enabled: true,
                window_sizes: vec![3, 5, 10, 15, 30, 60], // Short windows for mobility
                statistics: vec![
                    "mean".to_string(),
                    "std".to_string(),
                    "min".to_string(),
                    "max".to_string(),
                    "slope".to_string(), // Trend information
                ],
                target_columns: vec![
                    "serving_rsrp".to_string(),
                    "serving_sinr".to_string(),
                    "neighbor_rsrp_best".to_string(),
                    "ue_speed_kmh".to_string(),
                    "cqi".to_string(),
                ],
            },
            time_features: TimeBasedFeatureConfig {
                enabled: true,
                features: vec![
                    "hour_of_day".to_string(),
                    "day_of_week".to_string(),
                    "is_weekend".to_string(),
                ],
                timestamp_column: "timestamp".to_string(),
                timezone: "UTC".to_string(),
            },
            output: OutputConfig {
                format: "parquet".to_string(),
                compression: "snappy".to_string(),
                include_metadata: true,
                validate_schema: true,
            },
        }
    }
    
    /// Configuration for interference detection features
    pub fn interference_detection_features() -> FeatureConfig {
        FeatureConfig {
            lag_features: LagFeatureConfig {
                enabled: true,
                lag_periods: vec![1, 2, 3, 6, 12, 24], // Medium-term lags for interference patterns
                target_columns: vec![
                    "noise_floor_pusch".to_string(),
                    "noise_floor_pucch".to_string(),
                    "prb_utilization_ul".to_string(),
                    "sinr_avg".to_string(),
                    "bler_ul".to_string(),
                ],
            },
            rolling_window: RollingWindowConfig {
                enabled: true,
                window_sizes: vec![6, 12, 24, 48, 72], // Longer windows for interference patterns
                statistics: vec![
                    "mean".to_string(),
                    "std".to_string(),
                    "min".to_string(),
                    "max".to_string(),
                    "median".to_string(),
                    "iqr".to_string(), // Interquartile range for anomaly detection
                ],
                target_columns: vec![
                    "noise_floor_pusch".to_string(),
                    "noise_floor_pucch".to_string(),
                    "prb_utilization_ul".to_string(),
                    "sinr_avg".to_string(),
                    "bler_ul".to_string(),
                ],
            },
            time_features: TimeBasedFeatureConfig {
                enabled: true,
                features: vec![
                    "hour_of_day".to_string(),
                    "day_of_week".to_string(),
                    "is_weekend".to_string(),
                    "is_business_hour".to_string(),
                ],
                timestamp_column: "timestamp".to_string(),
                timezone: "UTC".to_string(),
            },
            output: OutputConfig {
                format: "parquet".to_string(),
                compression: "snappy".to_string(),
                include_metadata: true,
                validate_schema: true,
            },
        }
    }
}