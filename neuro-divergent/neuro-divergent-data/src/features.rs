//! # Feature Engineering for Time Series
//!
//! This module provides comprehensive feature engineering capabilities specifically designed
//! for time series forecasting, including lag features, rolling statistics, temporal features,
//! and Fourier transformations.

use crate::{DataPipelineError, Result, TimeSeriesData, DataPoint};
use num_traits::Float;
use chrono::{DateTime, Utc, Datelike, Timelike, Weekday};
use ndarray::{Array1, Array2};
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::HashMap;
use std::marker::PhantomData;

#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for feature engineering
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct FeatureConfig<T: Float> {
    /// Lag features configuration
    pub lag_features: Option<LagConfig>,
    /// Rolling statistics configuration
    pub rolling_features: Option<RollingConfig>,
    /// Temporal features configuration
    pub temporal_features: Option<TemporalConfig>,
    /// Fourier features configuration
    pub fourier_features: Option<FourierConfig<T>>,
    /// Exogenous features configuration
    pub exogenous_features: Option<ExogenousConfig>,
}

impl<T: Float> Default for FeatureConfig<T> {
    fn default() -> Self {
        Self {
            lag_features: Some(LagConfig::default()),
            rolling_features: Some(RollingConfig::default()),
            temporal_features: Some(TemporalConfig::default()),
            fourier_features: None,
            exogenous_features: None,
        }
    }
}

/// Configuration for lag features
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct LagConfig {
    /// List of lag periods to use
    pub lags: Vec<usize>,
    /// Whether to include the target value at each lag
    pub include_target: bool,
    /// Whether to include differences between lags
    pub include_differences: bool,
}

impl Default for LagConfig {
    fn default() -> Self {
        Self {
            lags: vec![1, 2, 3, 7, 14, 21, 28], // Common time series lags
            include_target: true,
            include_differences: false,
        }
    }
}

/// Configuration for rolling window statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct RollingConfig {
    /// Window sizes for rolling statistics
    pub windows: Vec<usize>,
    /// Statistics to compute
    pub statistics: Vec<RollingStatistic>,
    /// Minimum number of observations in window
    pub min_periods: Option<usize>,
}

impl Default for RollingConfig {
    fn default() -> Self {
        Self {
            windows: vec![7, 14, 30, 60, 90], // Common rolling windows
            statistics: vec![
                RollingStatistic::Mean,
                RollingStatistic::Std,
                RollingStatistic::Min,
                RollingStatistic::Max,
                RollingStatistic::Median,
            ],
            min_periods: None,
        }
    }
}

/// Types of rolling statistics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum RollingStatistic {
    Mean,
    Std,
    Var,
    Min,
    Max,
    Median,
    Quantile25,
    Quantile75,
    Skewness,
    Kurtosis,
}

/// Configuration for temporal features
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct TemporalConfig {
    /// Include day of week features
    pub day_of_week: bool,
    /// Include month features
    pub month: bool,
    /// Include quarter features
    pub quarter: bool,
    /// Include day of year features
    pub day_of_year: bool,
    /// Include hour of day features
    pub hour_of_day: bool,
    /// Include minute of hour features
    pub minute_of_hour: bool,
    /// Include holiday features
    pub holidays: bool,
    /// Include business day indicator
    pub business_day: bool,
    /// Cyclic encoding for temporal features
    pub cyclic_encoding: bool,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            day_of_week: true,
            month: true,
            quarter: true,
            day_of_year: false,
            hour_of_day: true,
            minute_of_hour: false,
            holidays: false,
            business_day: true,
            cyclic_encoding: true,
        }
    }
}

/// Configuration for Fourier features
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct FourierConfig<T: Float> {
    /// Number of Fourier terms for each seasonal period
    pub seasonal_periods: Vec<usize>,
    /// Number of Fourier terms per period
    pub fourier_terms: Vec<usize>,
    /// Whether to normalize Fourier coefficients
    pub normalize: bool,
    /// Phantom data for type parameter
    #[cfg_attr(feature = "serde_support", serde(skip))]
    _phantom: PhantomData<T>,
}

impl<T: Float> FourierConfig<T> {
    /// Create a new Fourier configuration
    pub fn new(seasonal_periods: Vec<usize>, fourier_terms: Vec<usize>) -> Self {
        Self {
            seasonal_periods,
            fourier_terms,
            normalize: true,
            _phantom: PhantomData,
        }
    }
}

/// Configuration for exogenous features
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct ExogenousConfig {
    /// Names of exogenous features to include
    pub feature_names: Vec<String>,
    /// Whether to include lag features of exogenous variables
    pub include_lags: bool,
    /// Lag periods for exogenous features
    pub exogenous_lags: Vec<usize>,
    /// Whether to include rolling statistics of exogenous features
    pub include_rolling: bool,
}

impl Default for ExogenousConfig {
    fn default() -> Self {
        Self {
            feature_names: Vec::new(),
            include_lags: false,
            exogenous_lags: vec![1, 2, 3],
            include_rolling: false,
        }
    }
}

/// Feature matrix containing all engineered features
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct FeatureMatrix<T: Float> {
    /// Feature values organized as (n_samples, n_features)
    pub features: Array2<T>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Timestamps corresponding to each sample
    pub timestamps: Vec<DateTime<Utc>>,
    /// Series ID
    pub series_id: String,
}

impl<T: Float> FeatureMatrix<T> {
    /// Create a new feature matrix
    pub fn new(
        features: Array2<T>,
        feature_names: Vec<String>,
        timestamps: Vec<DateTime<Utc>>,
        series_id: String,
    ) -> Result<Self> {
        if features.nrows() != timestamps.len() {
            return Err(DataPipelineError::IncompatibleDimensions {
                expected: format!("features.nrows()={}", features.nrows()),
                actual: format!("timestamps.len()={}", timestamps.len()),
            });
        }
        
        if features.ncols() != feature_names.len() {
            return Err(DataPipelineError::IncompatibleDimensions {
                expected: format!("features.ncols()={}", features.ncols()),
                actual: format!("feature_names.len()={}", feature_names.len()),
            });
        }
        
        Ok(Self {
            features,
            feature_names,
            timestamps,
            series_id,
        })
    }
    
    /// Get the number of samples
    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }
    
    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }
    
    /// Get feature values for a specific sample
    pub fn get_sample(&self, index: usize) -> Option<Array1<T>> {
        if index < self.n_samples() {
            Some(self.features.row(index).to_owned())
        } else {
            None
        }
    }
    
    /// Get a specific feature column
    pub fn get_feature(&self, name: &str) -> Option<Array1<T>> {
        self.feature_names.iter()
            .position(|n| n == name)
            .map(|idx| self.features.column(idx).to_owned())
    }
}

/// Main feature engineering engine
#[derive(Debug, Clone)]
pub struct FeatureEngine<T: Float> {
    config: FeatureConfig<T>,
    fitted: bool,
    feature_stats: HashMap<String, FeatureStats<T>>,
}

/// Statistics for features (used for normalization/scaling)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
struct FeatureStats<T: Float> {
    mean: T,
    std: T,
    min: T,
    max: T,
}

impl<T: Float> FeatureEngine<T> {
    /// Create a new feature engine with default configuration
    pub fn new() -> Self {
        Self {
            config: FeatureConfig::default(),
            fitted: false,
            feature_stats: HashMap::new(),
        }
    }
    
    /// Create a feature engine with custom configuration
    pub fn with_config(config: FeatureConfig<T>) -> Self {
        Self {
            config,
            fitted: false,
            feature_stats: HashMap::new(),
        }
    }
    
    /// Configure lag features
    pub fn with_lag_features(mut self, lags: Vec<usize>) -> Self {
        self.config.lag_features = Some(LagConfig {
            lags,
            include_target: true,
            include_differences: false,
        });
        self
    }
    
    /// Configure rolling features
    pub fn with_rolling_features(mut self, windows: Vec<usize>) -> Self {
        self.config.rolling_features = Some(RollingConfig {
            windows,
            statistics: vec![
                RollingStatistic::Mean,
                RollingStatistic::Std,
                RollingStatistic::Min,
                RollingStatistic::Max,
            ],
            min_periods: None,
        });
        self
    }
    
    /// Configure temporal features
    pub fn with_temporal_features(mut self, enabled: bool) -> Self {
        if enabled {
            self.config.temporal_features = Some(TemporalConfig::default());
        } else {
            self.config.temporal_features = None;
        }
        self
    }
    
    /// Configure Fourier features
    pub fn with_fourier_features(mut self, seasonal_periods: Vec<usize>, fourier_terms: Vec<usize>) -> Self {
        self.config.fourier_features = Some(FourierConfig::new(seasonal_periods, fourier_terms));
        self
    }
    
    /// Generate features for a time series
    pub fn generate_features(&self, data: &TimeSeriesData<T>) -> Result<FeatureMatrix<T>> {
        let mut all_features = Vec::new();
        let mut all_feature_names = Vec::new();
        
        // Generate lag features
        if let Some(ref lag_config) = self.config.lag_features {
            let (lag_features, lag_names) = self.generate_lag_features(data, lag_config)?;
            all_features.push(lag_features);
            all_feature_names.extend(lag_names);
        }
        
        // Generate rolling features
        if let Some(ref rolling_config) = self.config.rolling_features {
            let (rolling_features, rolling_names) = self.generate_rolling_features(data, rolling_config)?;
            all_features.push(rolling_features);
            all_feature_names.extend(rolling_names);
        }
        
        // Generate temporal features
        if let Some(ref temporal_config) = self.config.temporal_features {
            let (temporal_features, temporal_names) = self.generate_temporal_features(data, temporal_config)?;
            all_features.push(temporal_features);
            all_feature_names.extend(temporal_names);
        }
        
        // Generate Fourier features
        if let Some(ref fourier_config) = self.config.fourier_features {
            let (fourier_features, fourier_names) = self.generate_fourier_features(data, fourier_config)?;
            all_features.push(fourier_features);
            all_feature_names.extend(fourier_names);
        }
        
        // Generate exogenous features
        if let Some(ref exog_config) = self.config.exogenous_features {
            let (exog_features, exog_names) = self.generate_exogenous_features(data, exog_config)?;
            all_features.push(exog_features);
            all_feature_names.extend(exog_names);
        }
        
        // Combine all features
        let combined_features = self.combine_feature_matrices(all_features)?;
        
        // Determine valid timestamps (after any lag/rolling window requirements)
        let max_lag = self.config.lag_features.as_ref()
            .map(|cfg| cfg.lags.iter().cloned().max().unwrap_or(0))
            .unwrap_or(0);
            
        let max_window = self.config.rolling_features.as_ref()
            .map(|cfg| cfg.windows.iter().cloned().max().unwrap_or(0))
            .unwrap_or(0);
            
        let skip_rows = std::cmp::max(max_lag, max_window);
        let valid_timestamps = if skip_rows < data.timestamps().len() {
            data.timestamps()[skip_rows..].to_vec()
        } else {
            Vec::new()
        };
        
        FeatureMatrix::new(
            combined_features,
            all_feature_names,
            valid_timestamps,
            data.series_id.clone(),
        )
    }
    
    /// Generate lag features
    fn generate_lag_features(
        &self,
        data: &TimeSeriesData<T>,
        config: &LagConfig,
    ) -> Result<(Array2<T>, Vec<String>)> {
        let values = data.values();
        let n_samples = values.len();
        let max_lag = config.lags.iter().cloned().max().unwrap_or(0);
        
        if n_samples <= max_lag {
            return Err(DataPipelineError::InvalidFormat {
                message: format!("Insufficient data for max lag {}: need > {}, got {}", 
                               max_lag, max_lag, n_samples),
            });
        }
        
        let valid_samples = n_samples - max_lag;
        let mut feature_names = Vec::new();
        let mut features = Vec::new();
        
        for &lag in &config.lags {
            if config.include_target {
                feature_names.push(format!("lag_{}", lag));
                let mut lag_values = Vec::with_capacity(valid_samples);
                
                for i in max_lag..n_samples {
                    if i >= lag {
                        lag_values.push(values[i - lag]);
                    } else {
                        lag_values.push(T::zero()); // Padding for early values
                    }
                }
                features.push(lag_values);
            }
        }
        
        // Generate difference features if requested
        if config.include_differences {
            for i in 0..config.lags.len() {
                for j in (i + 1)..config.lags.len() {
                    let lag1 = config.lags[i];
                    let lag2 = config.lags[j];
                    
                    feature_names.push(format!("lag_diff_{}_{}", lag1, lag2));
                    let mut diff_values = Vec::with_capacity(valid_samples);
                    
                    for k in max_lag..n_samples {
                        let val1 = if k >= lag1 { values[k - lag1] } else { T::zero() };
                        let val2 = if k >= lag2 { values[k - lag2] } else { T::zero() };
                        diff_values.push(val1 - val2);
                    }
                    features.push(diff_values);
                }
            }
        }
        
        // Convert to Array2
        let n_features = features.len();
        if n_features == 0 {
            return Ok((Array2::zeros((valid_samples, 0)), feature_names));
        }
        
        let mut feature_matrix = Array2::zeros((valid_samples, n_features));
        for (col_idx, feature_col) in features.iter().enumerate() {
            for (row_idx, &value) in feature_col.iter().enumerate() {
                feature_matrix[[row_idx, col_idx]] = value;
            }
        }
        
        Ok((feature_matrix, feature_names))
    }
    
    /// Generate rolling window statistics
    fn generate_rolling_features(
        &self,
        data: &TimeSeriesData<T>,
        config: &RollingConfig,
    ) -> Result<(Array2<T>, Vec<String>)> {
        let values = data.values();
        let n_samples = values.len();
        let max_window = config.windows.iter().cloned().max().unwrap_or(0);
        
        if n_samples <= max_window {
            return Err(DataPipelineError::InvalidFormat {
                message: format!("Insufficient data for max window {}: need > {}, got {}", 
                               max_window, max_window, n_samples),
            });
        }
        
        let valid_samples = n_samples - max_window;
        let mut feature_names = Vec::new();
        let mut features = Vec::new();
        
        for &window in &config.windows {
            for &stat in &config.statistics {
                feature_names.push(format!("rolling_{}_{}", Self::stat_name(stat), window));
                let mut stat_values = Vec::with_capacity(valid_samples);
                
                for i in max_window..n_samples {
                    let start_idx = if i >= window { i - window + 1 } else { 0 };
                    let window_data = &values[start_idx..=i];
                    
                    let min_periods = config.min_periods.unwrap_or(1);
                    if window_data.len() >= min_periods {
                        let stat_value = self.compute_rolling_statistic(window_data, stat)?;
                        stat_values.push(stat_value);
                    } else {
                        stat_values.push(T::nan()); // Not enough data
                    }
                }
                features.push(stat_values);
            }
        }
        
        // Convert to Array2
        let n_features = features.len();
        if n_features == 0 {
            return Ok((Array2::zeros((valid_samples, 0)), feature_names));
        }
        
        let mut feature_matrix = Array2::zeros((valid_samples, n_features));
        for (col_idx, feature_col) in features.iter().enumerate() {
            for (row_idx, &value) in feature_col.iter().enumerate() {
                feature_matrix[[row_idx, col_idx]] = value;
            }
        }
        
        Ok((feature_matrix, feature_names))
    }
    
    /// Generate temporal features from timestamps
    fn generate_temporal_features(
        &self,
        data: &TimeSeriesData<T>,
        config: &TemporalConfig,
    ) -> Result<(Array2<T>, Vec<String>)> {
        let timestamps = data.timestamps();
        let n_samples = timestamps.len();
        let mut feature_names = Vec::new();
        let mut features = Vec::new();
        
        // Day of week features
        if config.day_of_week {
            if config.cyclic_encoding {
                feature_names.extend_from_slice(&["dow_sin".to_string(), "dow_cos".to_string()]);
                let mut dow_sin = Vec::with_capacity(n_samples);
                let mut dow_cos = Vec::with_capacity(n_samples);
                
                for timestamp in &timestamps {
                    let dow = timestamp.weekday().number_from_monday() as f64 - 1.0; // 0-6
                    let angle = 2.0 * std::f64::consts::PI * dow / 7.0;
                    dow_sin.push(T::from(angle.sin()).unwrap());
                    dow_cos.push(T::from(angle.cos()).unwrap());
                }
                features.push(dow_sin);
                features.push(dow_cos);
            } else {
                for day in 0..7 {
                    feature_names.push(format!("dow_{}", day));
                    let mut dow_feature = Vec::with_capacity(n_samples);
                    
                    for timestamp in &timestamps {
                        let dow = (timestamp.weekday().number_from_monday() as usize - 1) % 7;
                        dow_feature.push(if dow == day { T::one() } else { T::zero() });
                    }
                    features.push(dow_feature);
                }
            }
        }
        
        // Month features
        if config.month {
            if config.cyclic_encoding {
                feature_names.extend_from_slice(&["month_sin".to_string(), "month_cos".to_string()]);
                let mut month_sin = Vec::with_capacity(n_samples);
                let mut month_cos = Vec::with_capacity(n_samples);
                
                for timestamp in &timestamps {
                    let month = timestamp.month() as f64 - 1.0; // 0-11
                    let angle = 2.0 * std::f64::consts::PI * month / 12.0;
                    month_sin.push(T::from(angle.sin()).unwrap());
                    month_cos.push(T::from(angle.cos()).unwrap());
                }
                features.push(month_sin);
                features.push(month_cos);
            } else {
                for month in 1..=12 {
                    feature_names.push(format!("month_{}", month));
                    let mut month_feature = Vec::with_capacity(n_samples);
                    
                    for timestamp in &timestamps {
                        month_feature.push(if timestamp.month() == month { T::one() } else { T::zero() });
                    }
                    features.push(month_feature);
                }
            }
        }
        
        // Quarter features
        if config.quarter {
            feature_names.extend_from_slice(&["quarter_1".to_string(), "quarter_2".to_string(), 
                                           "quarter_3".to_string(), "quarter_4".to_string()]);
            
            for quarter in 1..=4 {
                let mut quarter_feature = Vec::with_capacity(n_samples);
                
                for timestamp in &timestamps {
                    let ts_quarter = (timestamp.month() - 1) / 3 + 1;
                    quarter_feature.push(if ts_quarter == quarter { T::one() } else { T::zero() });
                }
                features.push(quarter_feature);
            }
        }
        
        // Hour of day features
        if config.hour_of_day {
            if config.cyclic_encoding {
                feature_names.extend_from_slice(&["hour_sin".to_string(), "hour_cos".to_string()]);
                let mut hour_sin = Vec::with_capacity(n_samples);
                let mut hour_cos = Vec::with_capacity(n_samples);
                
                for timestamp in &timestamps {
                    let hour = timestamp.hour() as f64;
                    let angle = 2.0 * std::f64::consts::PI * hour / 24.0;
                    hour_sin.push(T::from(angle.sin()).unwrap());
                    hour_cos.push(T::from(angle.cos()).unwrap());
                }
                features.push(hour_sin);
                features.push(hour_cos);
            } else {
                for hour in 0..24 {
                    feature_names.push(format!("hour_{}", hour));
                    let mut hour_feature = Vec::with_capacity(n_samples);
                    
                    for timestamp in &timestamps {
                        hour_feature.push(if timestamp.hour() as usize == hour { T::one() } else { T::zero() });
                    }
                    features.push(hour_feature);
                }
            }
        }
        
        // Business day indicator
        if config.business_day {
            feature_names.push("is_business_day".to_string());
            let mut business_day_feature = Vec::with_capacity(n_samples);
            
            for timestamp in &timestamps {
                let is_business_day = match timestamp.weekday() {
                    Weekday::Sat | Weekday::Sun => T::zero(),
                    _ => T::one(),
                };
                business_day_feature.push(is_business_day);
            }
            features.push(business_day_feature);
        }
        
        // Convert to Array2
        let n_features = features.len();
        if n_features == 0 {
            return Ok((Array2::zeros((n_samples, 0)), feature_names));
        }
        
        let mut feature_matrix = Array2::zeros((n_samples, n_features));
        for (col_idx, feature_col) in features.iter().enumerate() {
            for (row_idx, &value) in feature_col.iter().enumerate() {
                feature_matrix[[row_idx, col_idx]] = value;
            }
        }
        
        Ok((feature_matrix, feature_names))
    }
    
    /// Generate Fourier features for seasonal patterns
    fn generate_fourier_features(
        &self,
        data: &TimeSeriesData<T>,
        config: &FourierConfig<T>,
    ) -> Result<(Array2<T>, Vec<String>)> {
        let n_samples = data.len();
        let mut feature_names = Vec::new();
        let mut features = Vec::new();
        
        for (i, &period) in config.seasonal_periods.iter().enumerate() {
            let n_terms = if i < config.fourier_terms.len() {
                config.fourier_terms[i]
            } else {
                config.fourier_terms.last().cloned().unwrap_or(1)
            };
            
            for k in 1..=n_terms {
                // Sine component
                feature_names.push(format!("fourier_sin_{}_{}", period, k));
                let mut sin_feature = Vec::with_capacity(n_samples);
                
                // Cosine component
                feature_names.push(format!("fourier_cos_{}_{}", period, k));
                let mut cos_feature = Vec::with_capacity(n_samples);
                
                for t in 0..n_samples {
                    let angle = 2.0 * std::f64::consts::PI * k as f64 * t as f64 / period as f64;
                    sin_feature.push(T::from(angle.sin()).unwrap());
                    cos_feature.push(T::from(angle.cos()).unwrap());
                }
                
                features.push(sin_feature);
                features.push(cos_feature);
            }
        }
        
        // Convert to Array2
        let n_features = features.len();
        if n_features == 0 {
            return Ok((Array2::zeros((n_samples, 0)), feature_names));
        }
        
        let mut feature_matrix = Array2::zeros((n_samples, n_features));
        for (col_idx, feature_col) in features.iter().enumerate() {
            for (row_idx, &value) in feature_col.iter().enumerate() {
                feature_matrix[[row_idx, col_idx]] = value;
            }
        }
        
        Ok((feature_matrix, feature_names))
    }
    
    /// Generate features from exogenous variables
    fn generate_exogenous_features(
        &self,
        data: &TimeSeriesData<T>,
        config: &ExogenousConfig,
    ) -> Result<(Array2<T>, Vec<String>)> {
        if config.feature_names.is_empty() {
            return Ok((Array2::zeros((data.len(), 0)), Vec::new()));
        }
        
        // This is a placeholder implementation - in practice, exogenous features
        // would be extracted from the data_points' exogenous field
        let n_samples = data.len();
        let n_exog_features = config.feature_names.len();
        
        let mut feature_names = config.feature_names.clone();
        let mut feature_matrix = Array2::zeros((n_samples, n_exog_features));
        
        // Extract exogenous features from data points
        for (row_idx, point) in data.data_points.iter().enumerate() {
            if let Some(ref exog) = point.exogenous {
                for (col_idx, &value) in exog.iter().take(n_exog_features).enumerate() {
                    feature_matrix[[row_idx, col_idx]] = value;
                }
            }
        }
        
        Ok((feature_matrix, feature_names))
    }
    
    /// Compute rolling window statistic
    fn compute_rolling_statistic(&self, window_data: &[T], stat: RollingStatistic) -> Result<T> {
        if window_data.is_empty() {
            return Ok(T::nan());
        }
        
        match stat {
            RollingStatistic::Mean => {
                let sum: T = window_data.iter().fold(T::zero(), |acc, &x| acc + x);
                Ok(sum / T::from(window_data.len()).unwrap())
            }
            RollingStatistic::Std => {
                let mean = self.compute_rolling_statistic(window_data, RollingStatistic::Mean)?;
                let variance: T = window_data.iter()
                    .map(|&x| (x - mean) * (x - mean))
                    .fold(T::zero(), |acc, x| acc + x) / T::from(window_data.len()).unwrap();
                Ok(variance.sqrt())
            }
            RollingStatistic::Var => {
                let mean = self.compute_rolling_statistic(window_data, RollingStatistic::Mean)?;
                let variance: T = window_data.iter()
                    .map(|&x| (x - mean) * (x - mean))
                    .fold(T::zero(), |acc, x| acc + x) / T::from(window_data.len()).unwrap();
                Ok(variance)
            }
            RollingStatistic::Min => {
                Ok(*window_data.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap())
            }
            RollingStatistic::Max => {
                Ok(*window_data.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap())
            }
            RollingStatistic::Median => {
                let mut sorted_data = window_data.to_vec();
                sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = sorted_data.len() / 2;
                if sorted_data.len() % 2 == 0 {
                    Ok((sorted_data[mid - 1] + sorted_data[mid]) / T::from(2.0).unwrap())
                } else {
                    Ok(sorted_data[mid])
                }
            }
            RollingStatistic::Quantile25 => {
                let mut sorted_data = window_data.to_vec();
                sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let index = (sorted_data.len() as f64 * 0.25) as usize;
                Ok(sorted_data[index.min(sorted_data.len() - 1)])
            }
            RollingStatistic::Quantile75 => {
                let mut sorted_data = window_data.to_vec();
                sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let index = (sorted_data.len() as f64 * 0.75) as usize;
                Ok(sorted_data[index.min(sorted_data.len() - 1)])
            }
            RollingStatistic::Skewness => {
                let mean = self.compute_rolling_statistic(window_data, RollingStatistic::Mean)?;
                let std = self.compute_rolling_statistic(window_data, RollingStatistic::Std)?;
                
                if std == T::zero() {
                    return Ok(T::zero());
                }
                
                let n = T::from(window_data.len()).unwrap();
                let skewness: T = window_data.iter()
                    .map(|&x| {
                        let z = (x - mean) / std;
                        z * z * z
                    })
                    .fold(T::zero(), |acc, x| acc + x) / n;
                Ok(skewness)
            }
            RollingStatistic::Kurtosis => {
                let mean = self.compute_rolling_statistic(window_data, RollingStatistic::Mean)?;
                let std = self.compute_rolling_statistic(window_data, RollingStatistic::Std)?;
                
                if std == T::zero() {
                    return Ok(T::zero());
                }
                
                let n = T::from(window_data.len()).unwrap();
                let kurtosis: T = window_data.iter()
                    .map(|&x| {
                        let z = (x - mean) / std;
                        z * z * z * z
                    })
                    .fold(T::zero(), |acc, x| acc + x) / n - T::from(3.0).unwrap();
                Ok(kurtosis)
            }
        }
    }
    
    /// Get the name of a rolling statistic
    fn stat_name(stat: RollingStatistic) -> &'static str {
        match stat {
            RollingStatistic::Mean => "mean",
            RollingStatistic::Std => "std",
            RollingStatistic::Var => "var",
            RollingStatistic::Min => "min",
            RollingStatistic::Max => "max",
            RollingStatistic::Median => "median",
            RollingStatistic::Quantile25 => "q25",
            RollingStatistic::Quantile75 => "q75",
            RollingStatistic::Skewness => "skew",
            RollingStatistic::Kurtosis => "kurt",
        }
    }
    
    /// Combine multiple feature matrices horizontally
    fn combine_feature_matrices(&self, matrices: Vec<Array2<T>>) -> Result<Array2<T>> {
        if matrices.is_empty() {
            return Err(DataPipelineError::InvalidFormat {
                message: "No feature matrices to combine".to_string(),
            });
        }
        
        let n_rows = matrices[0].nrows();
        let total_cols: usize = matrices.iter().map(|m| m.ncols()).sum();
        
        // Verify all matrices have the same number of rows
        for matrix in &matrices {
            if matrix.nrows() != n_rows {
                return Err(DataPipelineError::IncompatibleDimensions {
                    expected: format!("nrows={}", n_rows),
                    actual: format!("nrows={}", matrix.nrows()),
                });
            }
        }
        
        let mut combined = Array2::zeros((n_rows, total_cols));
        let mut col_offset = 0;
        
        for matrix in matrices {
            let cols = matrix.ncols();
            if cols > 0 {
                combined.slice_mut(s![.., col_offset..col_offset + cols])
                    .assign(&matrix);
                col_offset += cols;
            }
        }
        
        Ok(combined)
    }
}

impl<T: Float> Default for FeatureEngine<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TimeSeriesDatasetBuilder, DataPoint};
    use chrono::TimeZone;
    use approx::assert_relative_eq;
    
    fn create_test_series() -> TimeSeriesData<f64> {
        let mut timestamps = Vec::new();
        let mut values = Vec::new();
        
        // Create 30 days of daily data
        for i in 0..30 {
            timestamps.push(
                chrono::Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap() + 
                chrono::Duration::days(i)
            );
            values.push((i as f64 + 1.0) * 10.0); // Simple linear trend
        }
        
        TimeSeriesDatasetBuilder::new("test_series".to_string())
            .with_frequency("D".to_string())
            .with_values(values)
            .with_timestamps(timestamps)
            .build()
            .unwrap()
    }
    
    #[test]
    fn test_lag_features() {
        let data = create_test_series();
        let feature_engine = FeatureEngine::new().with_lag_features(vec![1, 2, 3]);
        
        let features = feature_engine.generate_features(&data).unwrap();
        
        // Should have 3 lag features
        assert_eq!(features.n_features(), 7); // 3 lags + 4 rolling stats * 1 window
        assert!(features.feature_names.contains(&"lag_1".to_string()));
        assert!(features.feature_names.contains(&"lag_2".to_string()));
        assert!(features.feature_names.contains(&"lag_3".to_string()));
        
        // Check first few lag_1 values
        let lag_1_column = features.get_feature("lag_1").unwrap();
        assert_relative_eq!(lag_1_column[0], 270.0, epsilon = 1e-10); // lag_1 at position max_lag(3) should be value at position 2
    }
    
    #[test]
    fn test_rolling_features() {
        let data = create_test_series();
        let feature_engine = FeatureEngine::new()
            .with_lag_features(vec![]) // Disable lag features
            .with_rolling_features(vec![7]);
        
        let features = feature_engine.generate_features(&data).unwrap();
        
        // Should have rolling features for window 7
        assert!(features.feature_names.iter().any(|name| name.contains("rolling_mean_7")));
        assert!(features.feature_names.iter().any(|name| name.contains("rolling_std_7")));
        assert!(features.feature_names.iter().any(|name| name.contains("rolling_min_7")));
        assert!(features.feature_names.iter().any(|name| name.contains("rolling_max_7")));
    }
    
    #[test]
    fn test_temporal_features() {
        let data = create_test_series();
        let feature_engine = FeatureEngine::new()
            .with_lag_features(vec![]) // Disable other features
            .with_rolling_features(vec![])
            .with_temporal_features(true);
        
        let features = feature_engine.generate_features(&data).unwrap();
        
        // Should have temporal features
        assert!(features.feature_names.iter().any(|name| name.contains("dow")));
        assert!(features.feature_names.iter().any(|name| name.contains("month")));
        assert!(features.feature_names.iter().any(|name| name.contains("quarter")));
        assert!(features.feature_names.iter().any(|name| name.contains("is_business_day")));
    }
    
    #[test]
    fn test_fourier_features() {
        let data = create_test_series();
        let feature_engine = FeatureEngine::new()
            .with_lag_features(vec![])
            .with_rolling_features(vec![])
            .with_temporal_features(false)
            .with_fourier_features(vec![7], vec![2]); // Weekly seasonality, 2 terms
        
        let features = feature_engine.generate_features(&data).unwrap();
        
        // Should have 4 Fourier features (2 terms * 2 components each)
        let fourier_features: Vec<_> = features.feature_names.iter()
            .filter(|name| name.contains("fourier"))
            .collect();
        assert_eq!(fourier_features.len(), 4);
        
        assert!(features.feature_names.iter().any(|name| name.contains("fourier_sin_7_1")));
        assert!(features.feature_names.iter().any(|name| name.contains("fourier_cos_7_1")));
        assert!(features.feature_names.iter().any(|name| name.contains("fourier_sin_7_2")));
        assert!(features.feature_names.iter().any(|name| name.contains("fourier_cos_7_2")));
    }
    
    #[test]
    fn test_rolling_statistics_computation() {
        let feature_engine = FeatureEngine::<f64>::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Test mean
        let mean = feature_engine.compute_rolling_statistic(&data, RollingStatistic::Mean).unwrap();
        assert_relative_eq!(mean, 3.0, epsilon = 1e-10);
        
        // Test min/max
        let min = feature_engine.compute_rolling_statistic(&data, RollingStatistic::Min).unwrap();
        let max = feature_engine.compute_rolling_statistic(&data, RollingStatistic::Max).unwrap();
        assert_relative_eq!(min, 1.0, epsilon = 1e-10);
        assert_relative_eq!(max, 5.0, epsilon = 1e-10);
        
        // Test median
        let median = feature_engine.compute_rolling_statistic(&data, RollingStatistic::Median).unwrap();
        assert_relative_eq!(median, 3.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_feature_matrix_creation() {
        let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];
        let timestamps = vec![
            chrono::Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        ];
        
        let matrix = FeatureMatrix::new(
            features,
            feature_names,
            timestamps,
            "test".to_string(),
        ).unwrap();
        
        assert_eq!(matrix.n_samples(), 3);
        assert_eq!(matrix.n_features(), 2);
        assert_eq!(matrix.feature_names.len(), 2);
        assert_eq!(matrix.timestamps.len(), 3);
        
        // Test getting specific feature
        let feature1 = matrix.get_feature("feature1").unwrap();
        assert_relative_eq!(feature1[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(feature1[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(feature1[2], 5.0, epsilon = 1e-10);
    }
}