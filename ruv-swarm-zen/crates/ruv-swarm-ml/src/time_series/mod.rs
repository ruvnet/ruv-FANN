//! Time series data processing and transformations
//!
//! This module provides utilities for time series data preprocessing,
//! feature engineering, and transformations.

use alloc::{
    boxed::Box,
    collections::BTreeMap as HashMap,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::cmp::Ordering;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Time series data structure
#[derive(Clone, Debug)]
pub struct TimeSeriesData {
    pub values: Vec<f32>,
    pub timestamps: Vec<f64>,
    pub frequency: String,
    pub unique_id: String,
}

impl TimeSeriesData {
    /// Calculate mean of values
    pub fn mean(&self) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f32>() / self.values.len() as f32
    }

    /// Calculate standard deviation of values
    pub fn std_dev(&self) -> f32 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = self.values.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / (self.values.len() - 1) as f32;
        variance.sqrt()
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmTimeSeriesData {
    inner: TimeSeriesData,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmTimeSeriesData {
    #[wasm_bindgen(constructor)]
    pub fn new(
        values: Vec<f32>,
        timestamps: Vec<f64>,
        frequency: String,
        unique_id: String,
    ) -> Self {
        Self {
            inner: TimeSeriesData {
                values,
                timestamps,
                frequency,
                unique_id,
            },
        }
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.values.len()
    }

    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> f32 {
        self.inner.mean()
    }

    #[wasm_bindgen(getter)]
    pub fn std_dev(&self) -> f32 {
        self.inner.std_dev()
    }

    pub fn get_values(&self) -> Vec<f32> {
        self.inner.values.clone()
    }

    pub fn get_timestamps(&self) -> Vec<f64> {
        self.inner.timestamps.clone()
    }
}

/// Time series processor for data transformations
pub struct TimeSeriesProcessor {
    scalers: HashMap<String, Scaler>,
    transformations: Vec<TransformationStep>,
}

/// Scaler information
#[derive(Clone, Debug)]
pub struct Scaler {
    pub scaler_type: ScalerType,
    pub fitted: bool,
    pub parameters: ScalerParameters,
}

/// Scaler types
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalerType {
    MinMax,
    Standard,
    Robust,
    MaxAbs,
}

/// Scaler parameters
#[derive(Clone, Debug)]
pub enum ScalerParameters {
    MinMax { min: f32, max: f32 },
    Standard { mean: f32, std: f32 },
    Robust { median: f32, iqr: f32 },
    MaxAbs { max_abs: f32 },
}

/// Transformation step record
#[derive(Clone, Debug)]
pub struct TransformationStep {
    pub step_type: TransformationType,
    pub parameters: TransformationParameters,
    pub applied_at: f64,
}

/// Transformation types
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TransformationType {
    Normalize,
    Standardize,
    Log,
    Difference,
    BoxCox,
    MovingAverage,
    ExponentialSmoothing,
}

/// Transformation parameters
#[derive(Clone, Debug)]
pub enum TransformationParameters {
    None,
    Difference { order: usize },
    MovingAverage { window: usize },
    ExponentialSmoothing { alpha: f32 },
    BoxCox { lambda: f32 },
}

impl TimeSeriesProcessor {
    /// Create a new time series processor
    pub fn new() -> Self {
        Self {
            scalers: HashMap::new(),
            transformations: Vec::new(),
        }
    }

    /// Add a scaler
    pub fn add_scaler(&mut self, name: String, scaler_type: ScalerType) {
        let scaler = Scaler {
            scaler_type,
            fitted: false,
            parameters: match scaler_type {
                ScalerType::MinMax => ScalerParameters::MinMax { min: 0.0, max: 1.0 },
                ScalerType::Standard => ScalerParameters::Standard {
                    mean: 0.0,
                    std: 1.0,
                },
                ScalerType::Robust => ScalerParameters::Robust {
                    median: 0.0,
                    iqr: 1.0,
                },
                ScalerType::MaxAbs => ScalerParameters::MaxAbs { max_abs: 1.0 },
            },
        };
        self.scalers.insert(name, scaler);
    }

    /// Fit and transform data
    pub fn fit_transform(
        &mut self,
        mut data: TimeSeriesData,
        transformations: Vec<TransformationType>,
    ) -> Result<TimeSeriesData, String> {
        for transform_type in transformations {
            data = self.apply_transformation(data, transform_type)?;
        }
        Ok(data)
    }

    /// Apply a transformation
    fn apply_transformation(
        &mut self,
        data: TimeSeriesData,
        transform_type: TransformationType,
    ) -> Result<TimeSeriesData, String> {
        let transformed_data = match transform_type {
            TransformationType::Normalize => self.normalize_data(data)?,
            TransformationType::Standardize => self.standardize_data(data)?,
            TransformationType::Log => self.log_transform_data(data)?,
            TransformationType::Difference => self.difference_data(data, 1)?,
            TransformationType::MovingAverage => self.moving_average_data(data, 3)?,
            TransformationType::ExponentialSmoothing => {
                self.exponential_smoothing_data(data, 0.3)?
            }
            TransformationType::BoxCox => self.box_cox_transform(data, 0.0)?,
        };

        // Record transformation
        self.transformations.push(TransformationStep {
            step_type: transform_type,
            parameters: TransformationParameters::None,
            applied_at: 0.0, // TODO: Add proper timestamp
        });

        Ok(transformed_data)
    }

    /// Normalize data to [0, 1] range
    fn normalize_data(&self, mut data: TimeSeriesData) -> Result<TimeSeriesData, String> {
        if data.values.is_empty() {
            return Ok(data);
        }

        let min_val = data.values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range > 0.0 {
            data.values = data.values.iter().map(|&x| (x - min_val) / range).collect();
        }

        Ok(data)
    }

    /// Standardize data to zero mean and unit variance
    fn standardize_data(&self, mut data: TimeSeriesData) -> Result<TimeSeriesData, String> {
        if data.values.len() < 2 {
            return Ok(data);
        }

        let mean = data.values.iter().sum::<f32>() / data.values.len() as f32;
        let variance =
            data.values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.values.len() as f32;
        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            data.values = data.values.iter().map(|&x| (x - mean) / std_dev).collect();
        }

        Ok(data)
    }

    /// Apply log transformation
    fn log_transform_data(&self, mut data: TimeSeriesData) -> Result<TimeSeriesData, String> {
        // Check for non-positive values
        if data.values.iter().any(|&x| x <= 0.0) {
            // Shift data to make all values positive
            let min_val = data.values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let shift = if min_val <= 0.0 { 1.0 - min_val } else { 0.0 };

            data.values = data.values.iter().map(|&x| (x + shift).ln()).collect();
        } else {
            data.values = data.values.iter().map(|&x| x.ln()).collect();
        }

        Ok(data)
    }

    /// Difference the data
    fn difference_data(
        &self,
        mut data: TimeSeriesData,
        order: usize,
    ) -> Result<TimeSeriesData, String> {
        if data.values.len() <= order {
            return Err("Not enough data points for differencing".to_string());
        }

        for _ in 0..order {
            let differenced: Vec<f32> = data.values.windows(2).map(|w| w[1] - w[0]).collect();

            data.values = differenced;
            data.timestamps = data.timestamps[1..].to_vec();
        }

        Ok(data)
    }

    /// Apply moving average smoothing
    fn moving_average_data(
        &self,
        mut data: TimeSeriesData,
        window: usize,
    ) -> Result<TimeSeriesData, String> {
        if window == 0 || window > data.values.len() {
            return Err("Invalid window size".to_string());
        }

        let smoothed: Vec<f32> = data
            .values
            .windows(window)
            .map(|w| w.iter().sum::<f32>() / window as f32)
            .collect();

        // Adjust timestamps to match smoothed data length
        let timestamp_offset = (window - 1) / 2;
        let smoothed_len = smoothed.len();
        data.values = smoothed;
        data.timestamps =
            data.timestamps[timestamp_offset..timestamp_offset + smoothed_len].to_vec();

        Ok(data)
    }

    /// Apply exponential smoothing
    fn exponential_smoothing_data(
        &self,
        mut data: TimeSeriesData,
        alpha: f32,
    ) -> Result<TimeSeriesData, String> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err("Alpha must be between 0 and 1".to_string());
        }

        if data.values.is_empty() {
            return Ok(data);
        }

        let mut smoothed = vec![data.values[0]];

        for i in 1..data.values.len() {
            let smooth_val = alpha * data.values[i] + (1.0 - alpha) * smoothed[i - 1];
            smoothed.push(smooth_val);
        }

        data.values = smoothed;
        Ok(data)
    }

    /// Apply Box-Cox transformation
    fn box_cox_transform(
        &self,
        mut data: TimeSeriesData,
        lambda: f32,
    ) -> Result<TimeSeriesData, String> {
        // Ensure all values are positive
        let min_val = data.values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let shift = if min_val <= 0.0 { 1.0 - min_val } else { 0.0 };

        data.values = data
            .values
            .iter()
            .map(|&x| {
                let shifted = x + shift;
                if lambda.abs() < 1e-6 {
                    shifted.ln()
                } else {
                    (shifted.powf(lambda) - 1.0) / lambda
                }
            })
            .collect();

        Ok(data)
    }

    /// Detect seasonality in the data
    pub fn detect_seasonality(&self, data: &TimeSeriesData) -> SeasonalityInfo {
        // Simple seasonality detection using autocorrelation
        let acf = self.autocorrelation(&data.values, data.values.len() / 2);

        // Find peaks in ACF
        let mut seasonal_periods = Vec::new();
        for i in 1..acf.len() - 1 {
            if acf[i] > acf[i - 1] && acf[i] > acf[i + 1] && acf[i] > 0.3 {
                seasonal_periods.push(i);
            }
        }

        // Estimate trend strength
        let trend_strength = self.estimate_trend_strength(&data.values);

        SeasonalityInfo {
            has_trend: trend_strength > 0.3,
            has_seasonality: !seasonal_periods.is_empty(),
            seasonal_strength: if seasonal_periods.is_empty() {
                0.0
            } else {
                0.5
            }, // Placeholder
            residual_strength: 0.2, // Placeholder
            seasonal_periods,
            trend_strength,
        }
    }

    /// Calculate autocorrelation function
    fn autocorrelation(&self, values: &[f32], max_lag: usize) -> Vec<f32> {
        let n = values.len();
        let mean = values.iter().sum::<f32>() / n as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

        if variance == 0.0 {
            return vec![1.0; max_lag.min(n)];
        }

        (0..max_lag.min(n))
            .map(|lag| {
                let covariance = (0..n - lag)
                    .map(|i| (values[i] - mean) * (values[i + lag] - mean))
                    .sum::<f32>()
                    / (n - lag) as f32;
                covariance / variance
            })
            .collect()
    }

    /// Estimate trend strength
    fn estimate_trend_strength(&self, values: &[f32]) -> f32 {
        if values.len() < 3 {
            return 0.0;
        }

        // Simple linear trend estimation
        let n = values.len() as f32;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f32>() / n;

        let mut num = 0.0;
        let mut den = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f32;
            num += (x - x_mean) * (y - y_mean);
            den += (x - x_mean).powi(2);
        }

        if den == 0.0 {
            return 0.0;
        }

        let slope = num / den;
        let intercept = y_mean - slope * x_mean;

        // Calculate R-squared
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let y_pred = slope * i as f32 + intercept;
            ss_res += (y - y_pred).powi(2);
            ss_tot += (y - y_mean).powi(2);
        }

        if ss_tot == 0.0 {
            return 0.0;
        }

        (1.0 - ss_res / ss_tot).max(0.0).sqrt()
    }
}

/// Seasonality information
#[derive(Clone, Debug)]
pub struct SeasonalityInfo {
    pub has_trend: bool,
    pub has_seasonality: bool,
    pub seasonal_periods: Vec<usize>,
    pub trend_strength: f32,
    pub seasonal_strength: f32,
    pub residual_strength: f32,
}

/// Feature engineering utilities
pub struct FeatureEngineering;

impl FeatureEngineering {
    /// Create lag features
    pub fn create_lag_features(data: &TimeSeriesData, lags: &[usize]) -> Vec<Vec<f32>> {
        let n = data.values.len();
        let max_lag = *lags.iter().max().unwrap_or(&0);

        if max_lag >= n {
            return Vec::new();
        }

        let mut features = Vec::new();

        for &lag in lags {
            let lag_feature: Vec<f32> = (lag..n).map(|i| data.values[i - lag]).collect();
            features.push(lag_feature);
        }

        features
    }

    /// Create rolling window features
    pub fn create_rolling_features(
        data: &TimeSeriesData,
        windows: &[usize],
    ) -> HashMap<String, Vec<f32>> {
        let mut features = HashMap::new();

        for &window in windows {
            if window > data.values.len() {
                continue;
            }

            // Rolling mean
            let rolling_mean: Vec<f32> = data
                .values
                .windows(window)
                .map(|w| w.iter().sum::<f32>() / window as f32)
                .collect();
            features.insert(format!("rolling_mean_{}", window), rolling_mean);

            // Rolling std
            let rolling_std: Vec<f32> = data
                .values
                .windows(window)
                .map(|w| {
                    let mean = w.iter().sum::<f32>() / window as f32;
                    let variance =
                        w.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / window as f32;
                    variance.sqrt()
                })
                .collect();
            features.insert(format!("rolling_std_{}", window), rolling_std);
        }

        features
    }

    /// Create date/time features
    pub fn create_datetime_features(timestamps: &[f64]) -> HashMap<String, Vec<f32>> {
        let mut features = HashMap::new();

        // Extract hour of day, day of week, etc. from timestamps
        // This is a simplified version - real implementation would use proper date/time parsing

        let hour_of_day: Vec<f32> = timestamps
            .iter()
            .map(|&ts| ((ts / 3600.0) % 24.0) as f32)
            .collect();
        features.insert("hour_of_day".to_string(), hour_of_day);

        let day_of_week: Vec<f32> = timestamps
            .iter()
            .map(|&ts| ((ts / 86400.0) % 7.0) as f32)
            .collect();
        features.insert("day_of_week".to_string(), day_of_week);

        features
    }
}

impl Default for TimeSeriesProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization() {
        let data = TimeSeriesData {
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            timestamps: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            frequency: "D".to_string(),
            unique_id: "test".to_string(),
        };

        let mut processor = TimeSeriesProcessor::new();
        let normalized = processor.normalize_data(data).unwrap();

        assert_eq!(normalized.values[0], 0.0);
        assert_eq!(normalized.values[4], 1.0);
    }

    #[test]
    fn test_differencing() {
        let data = TimeSeriesData {
            values: vec![1.0, 3.0, 2.0, 5.0, 4.0],
            timestamps: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            frequency: "D".to_string(),
            unique_id: "test".to_string(),
        };

        let processor = TimeSeriesProcessor::new();
        let differenced = processor.difference_data(data, 1).unwrap();

        assert_eq!(differenced.values.len(), 4);
        assert_eq!(differenced.values[0], 2.0); // 3 - 1
        assert_eq!(differenced.values[1], -1.0); // 2 - 3
    }
}
