use crate::config::*;
use crate::error::*;
use crate::features::*;
use crate::stats::*;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use polars::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument, warn};

/// Core feature engineering engine
#[derive(Debug, Clone)]
pub struct FeatureEngine {
    config: Arc<FeatureEngineConfig>,
    feature_generators: Arc<RwLock<HashMap<String, Box<dyn FeatureGenerator>>>>,
    stats_collector: Arc<StatsCollector>,
}

impl FeatureEngine {
    /// Create a new feature engine
    pub fn new(config: FeatureEngineConfig) -> Result<Self> {
        let feature_generators = Self::initialize_feature_generators(&config)?;
        
        Ok(Self {
            config: Arc::new(config),
            feature_generators: Arc::new(RwLock::new(feature_generators)),
            stats_collector: Arc::new(StatsCollector::new()),
        })
    }

    /// Initialize feature generators
    fn initialize_feature_generators(
        config: &FeatureEngineConfig,
    ) -> Result<HashMap<String, Box<dyn FeatureGenerator>>> {
        let mut generators: HashMap<String, Box<dyn FeatureGenerator>> = HashMap::new();
        
        // Add lag feature generator
        if config.default_features.lag_features.enabled {
            generators.insert(
                "lag".to_string(),
                Box::new(LagFeatureGenerator::new(config.default_features.lag_features.clone())),
            );
        }
        
        // Add rolling window feature generator
        if config.default_features.rolling_window.enabled {
            generators.insert(
                "rolling_window".to_string(),
                Box::new(RollingWindowFeatureGenerator::new(
                    config.default_features.rolling_window.clone(),
                )),
            );
        }
        
        // Add time-based feature generator
        if config.default_features.time_features.enabled {
            generators.insert(
                "time_based".to_string(),
                Box::new(TimeBasedFeatureGenerator::new(
                    config.default_features.time_features.clone(),
                )),
            );
        }
        
        Ok(generators)
    }

    /// Process a single time-series
    #[instrument(skip(self, input_df))]
    pub async fn process_time_series(
        &self,
        input_df: DataFrame,
        feature_config: &FeatureConfig,
    ) -> FeatureEngineResult<DataFrame> {
        let start_time = Instant::now();
        
        // Validate input DataFrame
        self.validate_input_dataframe(&input_df)?;
        
        let mut result_df = input_df.clone();
        let generators = self.feature_generators.read().await;
        
        // Apply lag features
        if feature_config.lag_features.enabled {
            if let Some(generator) = generators.get("lag") {
                result_df = generator.generate_features(result_df, &feature_config.lag_features)?;
                debug!("Applied lag features");
            }
        }
        
        // Apply rolling window features
        if feature_config.rolling_window.enabled {
            if let Some(generator) = generators.get("rolling_window") {
                result_df = generator.generate_features(result_df, &feature_config.rolling_window)?;
                debug!("Applied rolling window features");
            }
        }
        
        // Apply time-based features
        if feature_config.time_features.enabled {
            if let Some(generator) = generators.get("time_based") {
                result_df = generator.generate_features(result_df, &feature_config.time_features)?;
                debug!("Applied time-based features");
            }
        }
        
        // Validate output DataFrame
        self.validate_output_dataframe(&result_df, &input_df)?;
        
        let processing_time = start_time.elapsed();
        
        // Record statistics
        self.stats_collector.record_processing_time(processing_time).await;
        self.stats_collector.record_rows_processed(input_df.height()).await;
        
        debug!(
            "Time-series processing completed in {}ms, {} -> {} rows",
            processing_time.as_millis(),
            input_df.height(),
            result_df.height()
        );
        
        Ok(result_df)
    }

    /// Process multiple time-series in parallel
    #[instrument(skip(self, input_dataframes))]
    pub async fn process_batch_time_series(
        &self,
        input_dataframes: Vec<(String, DataFrame)>,
        feature_config: &FeatureConfig,
    ) -> FeatureEngineResult<Vec<(String, DataFrame)>> {
        let start_time = Instant::now();
        
        info!("Starting batch processing of {} time-series", input_dataframes.len());
        
        // Process in parallel chunks
        let max_parallel = self.config.processing.max_parallel_jobs;
        let chunk_size = std::cmp::max(1, input_dataframes.len() / max_parallel);
        
        let mut results = Vec::new();
        
        for chunk in input_dataframes.chunks(chunk_size) {
            let chunk_results = self.process_chunk(chunk, feature_config).await?;
            results.extend(chunk_results);
        }
        
        let processing_time = start_time.elapsed();
        
        info!(
            "Batch processing completed in {}ms, processed {} time-series",
            processing_time.as_millis(),
            results.len()
        );
        
        Ok(results)
    }

    /// Process a chunk of time-series
    async fn process_chunk(
        &self,
        chunk: &[(String, DataFrame)],
        feature_config: &FeatureConfig,
    ) -> FeatureEngineResult<Vec<(String, DataFrame)>> {
        let mut results = Vec::new();
        
        for (ts_id, df) in chunk {
            match self.process_time_series(df.clone(), feature_config).await {
                Ok(result_df) => {
                    results.push((ts_id.clone(), result_df));
                }
                Err(e) => {
                    warn!("Failed to process time-series '{}': {}", ts_id, e);
                    return Err(e);
                }
            }
        }
        
        Ok(results)
    }

    /// Validate input DataFrame
    fn validate_input_dataframe(&self, df: &DataFrame) -> FeatureEngineResult<()> {
        // Check if DataFrame is empty
        if df.height() == 0 {
            return Err(FeatureEngineError::validation("Input DataFrame is empty"));
        }
        
        // Check for required columns
        let required_columns = vec!["timestamp", "kpi_value"];
        for col in required_columns {
            if !df.get_column_names().contains(&col) {
                return Err(FeatureEngineError::missing_column(col));
            }
        }
        
        // Check timestamp column format
        if let Ok(timestamp_col) = df.column("timestamp") {
            if !matches!(timestamp_col.dtype(), DataType::Datetime(_, _)) {
                return Err(FeatureEngineError::validation(
                    "Timestamp column must be of datetime type"
                ));
            }
        }
        
        // Check for null values in critical columns
        for col_name in ["timestamp", "kpi_value"] {
            if let Ok(col) = df.column(col_name) {
                if col.null_count() > 0 {
                    warn!("Column '{}' contains {} null values", col_name, col.null_count());
                }
            }
        }
        
        Ok(())
    }

    /// Validate output DataFrame
    fn validate_output_dataframe(
        &self,
        output_df: &DataFrame,
        input_df: &DataFrame,
    ) -> FeatureEngineResult<()> {
        // Check if output has same number of rows as input
        if output_df.height() != input_df.height() {
            return Err(FeatureEngineError::validation(
                format!(
                    "Output DataFrame has different number of rows than input: {} vs {}",
                    output_df.height(),
                    input_df.height()
                )
            ));
        }
        
        // Check if output has more columns than input (features were added)
        if output_df.width() <= input_df.width() {
            return Err(FeatureEngineError::validation(
                "Output DataFrame should have more columns than input (features should be added)"
            ));
        }
        
        // Check if all original columns are preserved
        for col_name in input_df.get_column_names() {
            if !output_df.get_column_names().contains(&col_name) {
                return Err(FeatureEngineError::validation(
                    format!("Original column '{}' was not preserved in output", col_name)
                ));
            }
        }
        
        Ok(())
    }

    /// Get processing statistics
    pub async fn get_statistics(&self) -> ProcessingStatistics {
        self.stats_collector.get_statistics().await
    }

    /// Reset statistics
    pub async fn reset_statistics(&self) {
        self.stats_collector.reset_statistics().await;
    }

    /// Add a custom feature generator
    pub async fn add_feature_generator(
        &self,
        name: String,
        generator: Box<dyn FeatureGenerator>,
    ) -> FeatureEngineResult<()> {
        let mut generators = self.feature_generators.write().await;
        generators.insert(name, generator);
        Ok(())
    }

    /// Remove a feature generator
    pub async fn remove_feature_generator(&self, name: &str) -> FeatureEngineResult<()> {
        let mut generators = self.feature_generators.write().await;
        if generators.remove(name).is_none() {
            return Err(FeatureEngineError::validation(
                format!("Feature generator '{}' not found", name)
            ));
        }
        Ok(())
    }

    /// List available feature generators
    pub async fn list_feature_generators(&self) -> Vec<String> {
        let generators = self.feature_generators.read().await;
        generators.keys().cloned().collect()
    }

    /// Get configuration
    pub fn get_config(&self) -> &FeatureEngineConfig {
        &self.config
    }

    /// Health check
    pub async fn health_check(&self) -> FeatureEngineResult<HealthStatus> {
        let stats = self.get_statistics().await;
        let generators = self.list_feature_generators().await;
        
        Ok(HealthStatus {
            is_healthy: true,
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: stats.uptime_seconds,
            active_generators: generators,
            total_processed: stats.total_rows_processed,
            average_processing_time_ms: stats.average_processing_time_ms,
        })
    }
}

/// Health status information
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub is_healthy: bool,
    pub version: String,
    pub uptime_seconds: u64,
    pub active_generators: Vec<String>,
    pub total_processed: u64,
    pub average_processing_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use polars::prelude::*;
    
    fn create_test_config() -> FeatureEngineConfig {
        FeatureEngineConfig::default()
    }
    
    fn create_test_dataframe() -> DataFrame {
        let timestamps = vec![
            Utc::now() - chrono::Duration::hours(3),
            Utc::now() - chrono::Duration::hours(2),
            Utc::now() - chrono::Duration::hours(1),
            Utc::now(),
        ];
        
        df! {
            "timestamp" => timestamps,
            "kpi_value" => [10.0, 15.0, 12.0, 18.0],
            "cell_id" => ["Cell_A", "Cell_A", "Cell_A", "Cell_A"],
        }.unwrap()
    }
    
    #[tokio::test]
    async fn test_feature_engine_creation() {
        let config = create_test_config();
        let engine = FeatureEngine::new(config).unwrap();
        
        let generators = engine.list_feature_generators().await;
        assert!(!generators.is_empty());
        assert!(generators.contains(&"lag".to_string()));
        assert!(generators.contains(&"rolling_window".to_string()));
        assert!(generators.contains(&"time_based".to_string()));
    }
    
    #[tokio::test]
    async fn test_process_time_series() {
        let config = create_test_config();
        let engine = FeatureEngine::new(config).unwrap();
        
        let input_df = create_test_dataframe();
        let input_width = input_df.width();
        
        let result = engine.process_time_series(
            input_df.clone(),
            &engine.config.default_features,
        ).await.unwrap();
        
        // Check that features were added
        assert!(result.width() > input_width);
        assert_eq!(result.height(), input_df.height());
    }
    
    #[tokio::test]
    async fn test_validate_input_dataframe() {
        let config = create_test_config();
        let engine = FeatureEngine::new(config).unwrap();
        
        // Test valid DataFrame
        let valid_df = create_test_dataframe();
        assert!(engine.validate_input_dataframe(&valid_df).is_ok());
        
        // Test empty DataFrame
        let empty_df = df! {
            "timestamp" => Vec::<DateTime<Utc>>::new(),
            "kpi_value" => Vec::<f64>::new(),
        }.unwrap();
        assert!(engine.validate_input_dataframe(&empty_df).is_err());
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let config = create_test_config();
        let engine = FeatureEngine::new(config).unwrap();
        
        let health = engine.health_check().await.unwrap();
        assert!(health.is_healthy);
        assert!(!health.version.is_empty());
        assert!(!health.active_generators.is_empty());
    }
}