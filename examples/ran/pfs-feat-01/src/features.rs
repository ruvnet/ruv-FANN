use crate::config::*;
use crate::error::*;

use polars::prelude::*;
use std::collections::HashMap;
use tracing::{debug, instrument};

/// Trait for feature generators
pub trait FeatureGenerator: Send + Sync {
    /// Generate features and add them to the DataFrame
    fn generate_features(
        &self,
        df: DataFrame,
        config: &dyn FeatureGeneratorConfig,
    ) -> FeatureEngineResult<DataFrame>;
    
    /// Get the name of this feature generator
    fn name(&self) -> &'static str;
    
    /// Get feature names that will be generated
    fn feature_names(&self, config: &dyn FeatureGeneratorConfig) -> Vec<String>;
}

/// Trait for feature generator configurations
pub trait FeatureGeneratorConfig {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl FeatureGeneratorConfig for LagFeatureConfig {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl FeatureGeneratorConfig for RollingWindowConfig {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl FeatureGeneratorConfig for TimeBasedFeatureConfig {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Lag feature generator
#[derive(Debug, Clone)]
pub struct LagFeatureGenerator {
    config: LagFeatureConfig,
}

impl LagFeatureGenerator {
    pub fn new(config: LagFeatureConfig) -> Self {
        Self { config }
    }
}

impl FeatureGenerator for LagFeatureGenerator {
    #[instrument(skip(self, df))]
    fn generate_features(
        &self,
        mut df: DataFrame,
        config: &dyn FeatureGeneratorConfig,
    ) -> FeatureEngineResult<DataFrame> {
        let lag_config = config.as_any()
            .downcast_ref::<LagFeatureConfig>()
            .ok_or_else(|| FeatureEngineError::invalid_feature_config("Invalid lag feature config"))?;
        
        debug!("Generating lag features with {} periods for {} columns",
               lag_config.lag_periods.len(),
               lag_config.target_columns.len());
        
        for column in &lag_config.target_columns {
            // Check if column exists
            if !df.get_column_names().contains(&column.as_str()) {
                debug!("Skipping lag features for missing column: {}", column);
                continue;
            }
            
            for &lag_period in &lag_config.lag_periods {
                let lag_col_name = format!("{}_lag_{}", column, lag_period);
                
                let col = df.column(column)
                    .map_err(|e| FeatureEngineError::data_processing(format!("Failed to get column {}: {}", column, e)))?;
                
                let lagged_col = col.shift(lag_period);
                
                df = df.with_column(lagged_col.alias(&lag_col_name))
                    .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add lag feature {}: {}", lag_col_name, e)))?;
                
                debug!("Added lag feature: {}", lag_col_name);
            }
        }
        
        Ok(df)
    }
    
    fn name(&self) -> &'static str {
        "lag"
    }
    
    fn feature_names(&self, config: &dyn FeatureGeneratorConfig) -> Vec<String> {
        let lag_config = config.as_any()
            .downcast_ref::<LagFeatureConfig>()
            .unwrap();
        
        let mut names = Vec::new();
        for column in &lag_config.target_columns {
            for &lag_period in &lag_config.lag_periods {
                names.push(format!("{}_lag_{}", column, lag_period));
            }
        }
        names
    }
}

/// Rolling window feature generator
#[derive(Debug, Clone)]
pub struct RollingWindowFeatureGenerator {
    config: RollingWindowConfig,
}

impl RollingWindowFeatureGenerator {
    pub fn new(config: RollingWindowConfig) -> Self {
        Self { config }
    }
    
    fn create_rolling_options(window_size: i32) -> RollingOptions {
        RollingOptions {
            window_size: Duration::parse(&format!("{}i", window_size)).unwrap(),
            weights: None,
            min_periods: Some(1),
            center: false,
            by: None,
            closed_window: None,
        }
    }
}

impl FeatureGenerator for RollingWindowFeatureGenerator {
    #[instrument(skip(self, df))]
    fn generate_features(
        &self,
        mut df: DataFrame,
        config: &dyn FeatureGeneratorConfig,
    ) -> FeatureEngineResult<DataFrame> {
        let rolling_config = config.as_any()
            .downcast_ref::<RollingWindowConfig>()
            .ok_or_else(|| FeatureEngineError::invalid_feature_config("Invalid rolling window config"))?;
        
        debug!("Generating rolling window features with {} window sizes for {} columns",
               rolling_config.window_sizes.len(),
               rolling_config.target_columns.len());
        
        for column in &rolling_config.target_columns {
            // Check if column exists
            if !df.get_column_names().contains(&column.as_str()) {
                debug!("Skipping rolling window features for missing column: {}", column);
                continue;
            }
            
            for &window_size in &rolling_config.window_sizes {
                for stat in &rolling_config.statistics {
                    let feature_name = format!("{}_{}_{}w", column, stat, window_size);
                    let rolling_opts = Self::create_rolling_options(window_size);
                    
                    let col = df.column(column)
                        .map_err(|e| FeatureEngineError::data_processing(format!("Failed to get column {}: {}", column, e)))?;
                    
                    let rolling_col = match stat.as_str() {
                        "mean" => col.rolling_mean(rolling_opts),
                        "std" => col.rolling_std(rolling_opts),
                        "min" => col.rolling_min(rolling_opts),
                        "max" => col.rolling_max(rolling_opts),
                        "median" => col.rolling_median(rolling_opts),
                        "sum" => col.rolling_sum(rolling_opts),
                        "var" => col.rolling_var(rolling_opts),
                        "q25" => col.rolling_quantile(rolling_opts, 0.25, QuantileInterpolOptions::default()),
                        "q75" => col.rolling_quantile(rolling_opts, 0.75, QuantileInterpolOptions::default()),
                        "q90" => col.rolling_quantile(rolling_opts, 0.90, QuantileInterpolOptions::default()),
                        "q95" => col.rolling_quantile(rolling_opts, 0.95, QuantileInterpolOptions::default()),
                        "iqr" => {
                            // Interquartile range: Q75 - Q25
                            let q75 = col.rolling_quantile(rolling_opts.clone(), 0.75, QuantileInterpolOptions::default())?;
                            let q25 = col.rolling_quantile(rolling_opts, 0.25, QuantileInterpolOptions::default())?;
                            Ok((&q75 - &q25)?)
                        },
                        "skew" => col.rolling_skew(rolling_opts, true),
                        _ => {
                            debug!("Unsupported rolling statistic: {}", stat);
                            continue;
                        }
                    };
                    
                    match rolling_col {
                        Ok(col_result) => {
                            df = df.with_column(col_result.alias(&feature_name))
                                .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add rolling feature {}: {}", feature_name, e)))?;
                            debug!("Added rolling window feature: {}", feature_name);
                        }
                        Err(e) => {
                            debug!("Failed to compute rolling {} for {}: {}", stat, column, e);
                        }
                    }
                }
            }
        }
        
        Ok(df)
    }
    
    fn name(&self) -> &'static str {
        "rolling_window"
    }
    
    fn feature_names(&self, config: &dyn FeatureGeneratorConfig) -> Vec<String> {
        let rolling_config = config.as_any()
            .downcast_ref::<RollingWindowConfig>()
            .unwrap();
        
        let mut names = Vec::new();
        for column in &rolling_config.target_columns {
            for &window_size in &rolling_config.window_sizes {
                for stat in &rolling_config.statistics {
                    names.push(format!("{}_{}_{}w", column, stat, window_size));
                }
            }
        }
        names
    }
}

/// Time-based feature generator
#[derive(Debug, Clone)]
pub struct TimeBasedFeatureGenerator {
    config: TimeBasedFeatureConfig,
}

impl TimeBasedFeatureGenerator {
    pub fn new(config: TimeBasedFeatureConfig) -> Self {
        Self { config }
    }
}

impl FeatureGenerator for TimeBasedFeatureGenerator {
    #[instrument(skip(self, df))]
    fn generate_features(
        &self,
        mut df: DataFrame,
        config: &dyn FeatureGeneratorConfig,
    ) -> FeatureEngineResult<DataFrame> {
        let time_config = config.as_any()
            .downcast_ref::<TimeBasedFeatureConfig>()
            .ok_or_else(|| FeatureEngineError::invalid_feature_config("Invalid time-based feature config"))?;
        
        debug!("Generating time-based features: {:?}", time_config.features);
        
        // Check if timestamp column exists
        if !df.get_column_names().contains(&time_config.timestamp_column.as_str()) {
            return Err(FeatureEngineError::missing_column(&time_config.timestamp_column));
        }
        
        let timestamp_col = df.column(&time_config.timestamp_column)
            .map_err(|e| FeatureEngineError::data_processing(format!("Failed to get timestamp column: {}", e)))?;
        
        for feature in &time_config.features {
            match feature.as_str() {
                "hour_of_day" => {
                    let hour_col = timestamp_col.hour()
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to extract hour: {}", e)))?;
                    df = df.with_column(hour_col.alias("hour_of_day"))
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add hour_of_day: {}", e)))?;
                    debug!("Added hour_of_day feature");
                }
                "day_of_week" => {
                    let dow_col = timestamp_col.weekday()
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to extract day of week: {}", e)))?;
                    df = df.with_column(dow_col.alias("day_of_week"))
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add day_of_week: {}", e)))?;
                    debug!("Added day_of_week feature");
                }
                "day_of_month" => {
                    let dom_col = timestamp_col.day()
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to extract day of month: {}", e)))?;
                    df = df.with_column(dom_col.alias("day_of_month"))
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add day_of_month: {}", e)))?;
                    debug!("Added day_of_month feature");
                }
                "month" => {
                    let month_col = timestamp_col.month()
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to extract month: {}", e)))?;
                    df = df.with_column(month_col.alias("month"))
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add month: {}", e)))?;
                    debug!("Added month feature");
                }
                "quarter" => {
                    let quarter_col = timestamp_col.quarter()
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to extract quarter: {}", e)))?;
                    df = df.with_column(quarter_col.alias("quarter"))
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add quarter: {}", e)))?;
                    debug!("Added quarter feature");
                }
                "year" => {
                    let year_col = timestamp_col.year()
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to extract year: {}", e)))?;
                    df = df.with_column(year_col.alias("year"))
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add year: {}", e)))?;
                    debug!("Added year feature");
                }
                "is_weekend" => {
                    let weekend_col = timestamp_col.weekday()
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to extract weekday: {}", e)))?
                        .gt_eq(lit(6i32)); // Saturday = 6, Sunday = 7
                    df = df.with_column(weekend_col.alias("is_weekend"))
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add is_weekend: {}", e)))?;
                    debug!("Added is_weekend feature");
                }
                "is_business_hour" => {
                    // Business hours: 9 AM to 5 PM (9-17)
                    let hour_col = timestamp_col.hour()
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to extract hour: {}", e)))?;
                    let business_hour_col = hour_col.gt_eq(lit(9i32)).and(hour_col.lt(lit(17i32)));
                    df = df.with_column(business_hour_col.alias("is_business_hour"))
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add is_business_hour: {}", e)))?;
                    debug!("Added is_business_hour feature");
                }
                "is_peak_hour" => {
                    // Peak hours: 8-10 AM and 5-7 PM
                    let hour_col = timestamp_col.hour()
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to extract hour: {}", e)))?;
                    let morning_peak = hour_col.gt_eq(lit(8i32)).and(hour_col.lt(lit(10i32)));
                    let evening_peak = hour_col.gt_eq(lit(17i32)).and(hour_col.lt(lit(19i32)));
                    let peak_hour_col = morning_peak.or(evening_peak);
                    df = df.with_column(peak_hour_col.alias("is_peak_hour"))
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add is_peak_hour: {}", e)))?;
                    debug!("Added is_peak_hour feature");
                }
                "minute_of_hour" => {
                    let minute_col = timestamp_col.minute()
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to extract minute: {}", e)))?;
                    df = df.with_column(minute_col.alias("minute_of_hour"))
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add minute_of_hour: {}", e)))?;
                    debug!("Added minute_of_hour feature");
                }
                "second_of_minute" => {
                    let second_col = timestamp_col.second()
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to extract second: {}", e)))?;
                    df = df.with_column(second_col.alias("second_of_minute"))
                        .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add second_of_minute: {}", e)))?;
                    debug!("Added second_of_minute feature");
                }
                _ => {
                    debug!("Unknown time-based feature: {}", feature);
                }
            }
        }
        
        Ok(df)
    }
    
    fn name(&self) -> &'static str {
        "time_based"
    }
    
    fn feature_names(&self, config: &dyn FeatureGeneratorConfig) -> Vec<String> {
        let time_config = config.as_any()
            .downcast_ref::<TimeBasedFeatureConfig>()
            .unwrap();
        
        time_config.features.clone()
    }
}

/// Statistical feature generator for RAN-specific metrics
#[derive(Debug, Clone)]
pub struct RanStatisticalFeatureGenerator {
    target_columns: Vec<String>,
}

impl RanStatisticalFeatureGenerator {
    pub fn new(target_columns: Vec<String>) -> Self {
        Self { target_columns }
    }
}

impl FeatureGenerator for RanStatisticalFeatureGenerator {
    fn generate_features(
        &self,
        mut df: DataFrame,
        _config: &dyn FeatureGeneratorConfig,
    ) -> FeatureEngineResult<DataFrame> {
        debug!("Generating RAN statistical features for {} columns", self.target_columns.len());
        
        for column in &self.target_columns {
            if !df.get_column_names().contains(&column.as_str()) {
                debug!("Skipping statistical features for missing column: {}", column);
                continue;
            }
            
            let col = df.column(column)
                .map_err(|e| FeatureEngineError::data_processing(format!("Failed to get column {}: {}", column, e)))?;
            
            // Rate of change
            let diff_col = col.diff(1, NullBehavior::Ignore);
            df = df.with_column(diff_col.alias(&format!("{}_diff", column)))
                .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add diff feature: {}", e)))?;
            
            // Percentage change
            let pct_change_col = col.pct_change(Some(Expr::val(1.into())));
            df = df.with_column(pct_change_col.alias(&format!("{}_pct_change", column)))
                .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add pct_change feature: {}", e)))?;
            
            // Z-score (standardization)
            let mean_val = col.mean().unwrap_or(0.0);
            let std_val = col.std(1).unwrap_or(1.0);
            if std_val > 0.0 {
                let zscore_col = (col - lit(mean_val)) / lit(std_val);
                df = df.with_column(zscore_col.alias(&format!("{}_zscore", column)))
                    .map_err(|e| FeatureEngineError::feature_generation(format!("Failed to add zscore feature: {}", e)))?;
            }
            
            debug!("Added statistical features for column: {}", column);
        }
        
        Ok(df)
    }
    
    fn name(&self) -> &'static str {
        "ran_statistical"
    }
    
    fn feature_names(&self, _config: &dyn FeatureGeneratorConfig) -> Vec<String> {
        let mut names = Vec::new();
        for column in &self.target_columns {
            names.push(format!("{}_diff", column));
            names.push(format!("{}_pct_change", column));
            names.push(format!("{}_zscore", column));
        }
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    
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
    
    #[test]
    fn test_lag_feature_generator() {
        let config = LagFeatureConfig {
            enabled: true,
            lag_periods: vec![1, 2],
            target_columns: vec!["kpi_value".to_string()],
        };
        
        let generator = LagFeatureGenerator::new(config.clone());
        let df = create_test_dataframe();
        let original_width = df.width();
        
        let result = generator.generate_features(df, &config).unwrap();
        
        // Should have added 2 lag features
        assert_eq!(result.width(), original_width + 2);
        assert!(result.get_column_names().contains(&"kpi_value_lag_1"));
        assert!(result.get_column_names().contains(&"kpi_value_lag_2"));
    }
    
    #[test]
    fn test_time_based_feature_generator() {
        let config = TimeBasedFeatureConfig {
            enabled: true,
            features: vec!["hour_of_day".to_string(), "day_of_week".to_string(), "is_weekend".to_string()],
            timestamp_column: "timestamp".to_string(),
            timezone: "UTC".to_string(),
        };
        
        let generator = TimeBasedFeatureGenerator::new(config.clone());
        let df = create_test_dataframe();
        let original_width = df.width();
        
        let result = generator.generate_features(df, &config).unwrap();
        
        // Should have added 3 time-based features
        assert_eq!(result.width(), original_width + 3);
        assert!(result.get_column_names().contains(&"hour_of_day"));
        assert!(result.get_column_names().contains(&"day_of_week"));
        assert!(result.get_column_names().contains(&"is_weekend"));
    }
    
    #[test]
    fn test_rolling_window_feature_generator() {
        let config = RollingWindowConfig {
            enabled: true,
            window_sizes: vec![2, 3],
            statistics: vec!["mean".to_string(), "std".to_string()],
            target_columns: vec!["kpi_value".to_string()],
        };
        
        let generator = RollingWindowFeatureGenerator::new(config.clone());
        let df = create_test_dataframe();
        let original_width = df.width();
        
        let result = generator.generate_features(df, &config).unwrap();
        
        // Should have added 4 rolling window features (2 windows * 2 stats)
        assert_eq!(result.width(), original_width + 4);
        assert!(result.get_column_names().contains(&"kpi_value_mean_2w"));
        assert!(result.get_column_names().contains(&"kpi_value_std_2w"));
        assert!(result.get_column_names().contains(&"kpi_value_mean_3w"));
        assert!(result.get_column_names().contains(&"kpi_value_std_3w"));
    }
}