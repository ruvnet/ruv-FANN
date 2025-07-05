pub mod config;
pub mod error;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};

pub use config::*;
pub use error::*;

/// Simplified time-series feature generation agent
#[derive(Debug, Clone)]
pub struct FeatureEngineeringAgent {
    config: Arc<FeatureEngineConfig>,
    stats: Arc<RwLock<HashMap<String, FeatureGenerationStats>>>,
}

impl FeatureEngineeringAgent {
    /// Create a new feature engineering agent
    pub fn new(config: FeatureEngineConfig) -> Self {
        Self {
            config: Arc::new(config),
            stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Generate features for a single time-series
    #[instrument(skip(self, input_path, output_path))]
    pub async fn generate_features(
        &self,
        time_series_id: &str,
        input_path: &Path,
        output_path: &Path,
        feature_config: &FeatureConfig,
    ) -> Result<FeatureGenerationResult> {
        let start_time = Instant::now();
        
        info!(
            "Starting feature generation for time-series '{}', input: {:?}, output: {:?}",
            time_series_id, input_path, output_path
        );

        // Load input data
        let mut input_df = self.load_input_data(input_path)
            .context("Failed to load input data")?;

        debug!("Loaded {} rows from input file {:?}", input_df.height(), input_path);

        let original_width = input_df.width();

        // Generate lag features
        if feature_config.lag_features.enabled {
            input_df = self.generate_lag_features(input_df, &feature_config.lag_features)
                .context("Failed to generate lag features")?;
        }

        // Generate rolling window features (simplified)
        if feature_config.rolling_window.enabled {
            input_df = self.generate_rolling_window_features(input_df, &feature_config.rolling_window)
                .context("Failed to generate rolling window features")?;
        }

        // Generate time-based features
        if feature_config.time_features.enabled {
            input_df = self.generate_time_based_features(input_df, &feature_config.time_features)
                .context("Failed to generate time-based features")?;
        }

        debug!("Generated {} features, output has {} rows", input_df.width() - original_width, input_df.height());

        // Save output
        self.save_output_data(&input_df, output_path, &feature_config.output)
            .context("Failed to save output data")?;

        let processing_time = start_time.elapsed();
        
        let stats = FeatureGenerationStats {
            processing_time_ms: processing_time.as_millis() as u64,
            input_rows: input_df.height() as u64,
            output_rows: input_df.height() as u64,
            features_generated: (input_df.width() - original_width) as u32,
            memory_usage_mb: self.estimate_memory_usage(&input_df),
            feature_names: input_df.get_column_names().iter().map(|s| s.to_string()).collect(),
        };

        // Store stats
        self.stats.write().await.insert(time_series_id.to_string(), stats.clone());

        info!(
            "Feature generation completed for '{}' in {}ms",
            time_series_id,
            processing_time.as_millis()
        );

        Ok(FeatureGenerationResult {
            time_series_id: time_series_id.to_string(),
            output_path: output_path.to_path_buf(),
            stats,
            generated_features: input_df.get_column_names().iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Load input data from Parquet file
    fn load_input_data(&self, input_path: &Path) -> Result<DataFrame> {
        let df = LazyFrame::scan_parquet(input_path, Default::default())?
            .collect()?;
        Ok(df)
    }

    /// Generate lag features (simplified)
    fn generate_lag_features(
        &self,
        mut df: DataFrame,
        lag_config: &LagFeatureConfig,
    ) -> Result<DataFrame, FeatureEngineError> {
        for column in &lag_config.target_columns {
            for &lag_period in &lag_config.lag_periods {
                let lag_col_name = format!("{}_lag_{}", column, lag_period);
                
                if let Ok(col) = df.column(column) {
                    let lagged_values: Vec<Option<f64>> = (0..df.height())
                        .map(|i| {
                            if i >= lag_period as usize {
                                col.get(i - lag_period as usize).ok()
                                    .and_then(|v| v.try_extract::<f64>().ok())
                            } else {
                                None
                            }
                        })
                        .collect();
                    
                    let lagged_series = Series::new(&lag_col_name, lagged_values);
                    df = df.with_column(lagged_series).map_err(FeatureEngineError::Polars)?;
                }
            }
        }
        Ok(df)
    }

    /// Generate rolling window features (simplified)
    fn generate_rolling_window_features(
        &self,
        mut df: DataFrame,
        rolling_config: &RollingWindowConfig,
    ) -> Result<DataFrame, FeatureEngineError> {
        for column in &rolling_config.target_columns {
            for &window_size in &rolling_config.window_sizes {
                for stat in &rolling_config.statistics {
                    let feature_name = format!("{}_{}_{}w", column, stat, window_size);
                    
                    if let Ok(col) = df.column(column) {
                        let rolling_values: Vec<Option<f64>> = (0..df.height())
                            .map(|i| {
                                let start = if i >= window_size as usize { i - window_size as usize + 1 } else { 0 };
                                let end = i + 1;
                                
                                let window_values: Vec<f64> = (start..end)
                                    .filter_map(|j| {
                                        col.get(j).ok()
                                            .and_then(|v| v.try_extract::<f64>().ok())
                                    })
                                    .collect();
                                
                                if window_values.is_empty() {
                                    None
                                } else {
                                    match stat.as_str() {
                                        "mean" => Some(window_values.iter().sum::<f64>() / window_values.len() as f64),
                                        "min" => window_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)).into(),
                                        "max" => window_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)).into(),
                                        "std" => {
                                            let mean = window_values.iter().sum::<f64>() / window_values.len() as f64;
                                            let variance = window_values.iter()
                                                .map(|x| (x - mean).powi(2))
                                                .sum::<f64>() / window_values.len() as f64;
                                            Some(variance.sqrt())
                                        },
                                        _ => None,
                                    }
                                }
                            })
                            .collect();
                        
                        let rolling_series = Series::new(&feature_name, rolling_values);
                        df = df.with_column(rolling_series);
                    }
                }
            }
        }
        Ok(df)
    }

    /// Generate time-based features (simplified)
    fn generate_time_based_features(
        &self,
        mut df: DataFrame,
        time_config: &TimeBasedFeatureConfig,
    ) -> Result<DataFrame> {
        if let Ok(_timestamp_col) = df.column(&time_config.timestamp_column) {
            for feature in &time_config.features {
                match feature.as_str() {
                    "hour_of_day" => {
                        // Add simplified hour feature as constant for now
                        let hour_values: Vec<i32> = (0..df.height()).map(|i| (i % 24) as i32).collect();
                        let hour_series = Series::new("hour_of_day", hour_values);
                        df = df.with_column(hour_series);
                    }
                    "day_of_week" => {
                        // Add simplified day of week as constant for now
                        let dow_values: Vec<i32> = (0..df.height()).map(|i| (i % 7) as i32).collect();
                        let dow_series = Series::new("day_of_week", dow_values);
                        df = df.with_column(dow_series);
                    }
                    "is_weekend" => {
                        // Add simplified weekend flag
                        let weekend_values: Vec<bool> = (0..df.height()).map(|i| (i % 7) >= 5).collect();
                        let weekend_series = Series::new("is_weekend", weekend_values);
                        df = df.with_column(weekend_series);
                    }
                    _ => {
                        warn!("Unknown time-based feature: {}", feature);
                    }
                }
            }
        }
        Ok(df)
    }

    /// Save output data
    fn save_output_data(
        &self,
        df: &DataFrame,
        output_path: &Path,
        output_config: &OutputConfig,
    ) -> Result<()> {
        // Ensure output directory exists
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create output directory")?;
        }

        match output_config.format.as_str() {
            "parquet" => {
                let mut file = std::fs::File::create(output_path)?;
                ParquetWriter::new(&mut file).finish(&mut df.clone())?;
            }
            "csv" => {
                let mut file = std::fs::File::create(output_path)?;
                CsvWriter::new(&mut file).finish(&mut df.clone())?;
            }
            _ => {
                return Err(anyhow::anyhow!("Unsupported output format: {}", output_config.format));
            }
        }

        Ok(())
    }

    /// Estimate memory usage in MB
    fn estimate_memory_usage(&self, df: &DataFrame) -> u64 {
        // Simple estimation based on DataFrame size
        let estimated_bytes = df.height() * df.width() * 8; // Assume 8 bytes per value
        (estimated_bytes / 1024 / 1024) as u64
    }

    /// Get statistics for a time-series
    pub async fn get_stats(&self, time_series_id: &str) -> Option<FeatureGenerationStats> {
        self.stats.read().await.get(time_series_id).cloned()
    }
}

/// Result of feature generation for a single time-series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGenerationResult {
    pub time_series_id: String,
    pub output_path: std::path::PathBuf,
    pub stats: FeatureGenerationStats,
    pub generated_features: Vec<String>,
}

/// Feature generation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGenerationStats {
    pub processing_time_ms: u64,
    pub input_rows: u64,
    pub output_rows: u64,
    pub features_generated: u32,
    pub memory_usage_mb: u64,
    pub feature_names: Vec<String>,
}