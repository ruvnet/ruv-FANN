pub mod config;
pub mod engine;
pub mod features;
pub mod grpc_service;
pub mod stats;
pub mod validation;
pub mod error;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};

pub use config::*;
pub use engine::*;
pub use error::*;
pub use features::*;
pub use grpc_service::*;
pub use stats::*;
pub use validation::*;

/// Time-series feature generation agent
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
        let input_df = self.load_input_data(input_path)
            .context("Failed to load input data")?;

        debug!(
            "Loaded {} rows from input file {:?}",
            input_df.height(),
            input_path
        );

        // Generate features
        let feature_df = self.generate_time_series_features(&input_df, feature_config)
            .context("Failed to generate features")?;

        debug!(
            "Generated {} features, output has {} rows",
            feature_df.width() - input_df.width(),
            feature_df.height()
        );

        // Save output
        self.save_output_data(&feature_df, output_path, &feature_config.output)
            .context("Failed to save output data")?;

        let processing_time = start_time.elapsed();
        
        let stats = FeatureGenerationStats {
            processing_time_ms: processing_time.as_millis() as u64,
            input_rows: input_df.height() as u64,
            output_rows: feature_df.height() as u64,
            features_generated: (feature_df.width() - input_df.width()) as u32,
            memory_usage_mb: self.estimate_memory_usage(&feature_df),
            feature_names: feature_df.get_column_names().iter().map(|s| s.to_string()).collect(),
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
            generated_features: feature_df.get_column_names().iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Generate features for multiple time-series in batch
    #[instrument(skip(self, input_directory, output_directory))]
    pub async fn generate_batch_features(
        &self,
        time_series_ids: &[String],
        input_directory: &Path,
        output_directory: &Path,
        feature_config: &FeatureConfig,
        max_parallel_jobs: usize,
    ) -> Result<BatchFeatureGenerationResult> {
        let start_time = Instant::now();
        
        info!(
            "Starting batch feature generation for {} time-series, max parallel jobs: {}",
            time_series_ids.len(),
            max_parallel_jobs
        );

        // Ensure output directory exists
        std::fs::create_dir_all(output_directory)
            .context("Failed to create output directory")?;

        // Process in parallel chunks
        let chunk_size = std::cmp::max(1, time_series_ids.len() / max_parallel_jobs);
        let results: Vec<Result<FeatureGenerationResult>> = time_series_ids
            .par_chunks(chunk_size)
            .map(|chunk| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let mut chunk_results = Vec::new();
                    for ts_id in chunk {
                        let input_path = input_directory.join(format!("{}.parquet", ts_id));
                        let output_path = output_directory.join(format!("{}_features.parquet", ts_id));
                        
                        match self.generate_features(ts_id, &input_path, &output_path, feature_config).await {
                            Ok(result) => chunk_results.push(result),
                            Err(e) => {
                                error!("Failed to generate features for '{}': {}", ts_id, e);
                                return Err(e);
                            }
                        }
                    }
                    Ok(chunk_results)
                })
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        let processing_time = start_time.elapsed();
        
        let successful_count = results.len();
        let failed_count = time_series_ids.len() - successful_count;
        
        let batch_stats = BatchProcessingStats {
            total_processing_time_ms: processing_time.as_millis() as u64,
            total_time_series: time_series_ids.len() as u32,
            successful_series: successful_count as u32,
            failed_series: failed_count as u32,
            total_input_rows: results.iter().map(|r| r.stats.input_rows).sum(),
            total_output_rows: results.iter().map(|r| r.stats.output_rows).sum(),
            peak_memory_usage_mb: results.iter().map(|r| r.stats.memory_usage_mb).max().unwrap_or(0),
        };

        info!(
            "Batch feature generation completed: {} successful, {} failed, in {}ms",
            successful_count,
            failed_count,
            processing_time.as_millis()
        );

        Ok(BatchFeatureGenerationResult {
            results,
            batch_stats,
        })
    }

    /// Load input data from Parquet file
    fn load_input_data(&self, input_path: &Path) -> Result<DataFrame> {
        let df = LazyFrame::scan_parquet(input_path, Default::default())?
            .collect()?;
        Ok(df)
    }

    /// Generate time-series features
    fn generate_time_series_features(
        &self,
        input_df: &DataFrame,
        feature_config: &FeatureConfig,
    ) -> Result<DataFrame> {
        let mut df = input_df.clone();

        // Generate lag features
        if feature_config.lag_features.enabled {
            df = self.generate_lag_features(df, &feature_config.lag_features)?;
        }

        // Generate rolling window statistics
        if feature_config.rolling_window.enabled {
            df = self.generate_rolling_window_features(df, &feature_config.rolling_window)?;
        }

        // Generate time-based features
        if feature_config.time_features.enabled {
            df = self.generate_time_based_features(df, &feature_config.time_features)?;
        }

        Ok(df)
    }

    /// Generate lag features
    fn generate_lag_features(
        &self,
        mut df: DataFrame,
        lag_config: &LagFeatureConfig,
    ) -> Result<DataFrame> {
        for column in &lag_config.target_columns {
            for &lag_period in &lag_config.lag_periods {
                let lag_col_name = format!("{}_lag_{}", column, lag_period);
                
                if let Ok(col) = df.column(column) {
                    let lagged_col = col.shift(lag_period);
                    df = df.with_column(lagged_col.alias(&lag_col_name))
                        .context("Failed to add lag feature")?;
                }
            }
        }
        Ok(df)
    }

    /// Generate rolling window statistics
    fn generate_rolling_window_features(
        &self,
        mut df: DataFrame,
        rolling_config: &RollingWindowConfig,
    ) -> Result<DataFrame> {
        for column in &rolling_config.target_columns {
            for &window_size in &rolling_config.window_sizes {
                for stat in &rolling_config.statistics {
                    let feature_name = format!("{}_{}_{}w", column, stat, window_size);
                    
                    if let Ok(col) = df.column(column) {
                        let rolling_col = match stat.as_str() {
                            "mean" => col.rolling_mean(RollingOptions::default().window_size(Duration::parse(&format!("{}i", window_size)))),
                            "std" => col.rolling_std(RollingOptions::default().window_size(Duration::parse(&format!("{}i", window_size)))),
                            "min" => col.rolling_min(RollingOptions::default().window_size(Duration::parse(&format!("{}i", window_size)))),
                            "max" => col.rolling_max(RollingOptions::default().window_size(Duration::parse(&format!("{}i", window_size)))),
                            "median" => col.rolling_median(RollingOptions::default().window_size(Duration::parse(&format!("{}i", window_size)))),
                            _ => continue,
                        };

                        if let Ok(rolling_col) = rolling_col {
                            df = df.with_column(rolling_col.alias(&feature_name))
                                .context("Failed to add rolling window feature")?;
                        }
                    }
                }
            }
        }
        Ok(df)
    }

    /// Generate time-based features
    fn generate_time_based_features(
        &self,
        mut df: DataFrame,
        time_config: &TimeBasedFeatureConfig,
    ) -> Result<DataFrame> {
        if let Ok(timestamp_col) = df.column(&time_config.timestamp_column) {
            for feature in &time_config.features {
                match feature.as_str() {
                    "hour_of_day" => {
                        let hour_col = timestamp_col.hour().map_err(|e| anyhow::anyhow!("Failed to extract hour: {}", e))?;
                        df = df.with_column(hour_col.alias("hour_of_day"))
                            .context("Failed to add hour_of_day feature")?;
                    }
                    "day_of_week" => {
                        let dow_col = timestamp_col.weekday().map_err(|e| anyhow::anyhow!("Failed to extract day of week: {}", e))?;
                        df = df.with_column(dow_col.alias("day_of_week"))
                            .context("Failed to add day_of_week feature")?;
                    }
                    "is_weekend" => {
                        let weekend_col = timestamp_col.weekday().map_err(|e| anyhow::anyhow!("Failed to extract weekday: {}", e))?
                            .gt_eq(lit(5i32));
                        df = df.with_column(weekend_col.alias("is_weekend"))
                            .context("Failed to add is_weekend feature")?;
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
                let mut writer = ParquetWriter::new(std::fs::File::create(output_path)?)
                    .with_compression(match output_config.compression.as_str() {
                        "snappy" => ParquetCompression::Snappy,
                        "gzip" => ParquetCompression::Gzip(None),
                        "lz4" => ParquetCompression::Lz4Raw,
                        "brotli" => ParquetCompression::Brotli(None),
                        _ => ParquetCompression::Snappy,
                    });
                
                writer.finish(df)?;
            }
            "csv" => {
                let mut writer = CsvWriter::new(std::fs::File::create(output_path)?)
                    .include_header(true);
                writer.finish(df)?;
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

/// Result of batch feature generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchFeatureGenerationResult {
    pub results: Vec<FeatureGenerationResult>,
    pub batch_stats: BatchProcessingStats,
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

/// Batch processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingStats {
    pub total_processing_time_ms: u64,
    pub total_time_series: u32,
    pub successful_series: u32,
    pub failed_series: u32,
    pub total_input_rows: u64,
    pub total_output_rows: u64,
    pub peak_memory_usage_mb: u64,
}