//! Comprehensive unit tests for the data pipeline
//!
//! This module tests data preprocessing, validation, feature engineering,
//! and all data transformation functionality.

use neuro_divergent::prelude::*;
use neuro_divergent::{AccuracyMetrics, NeuroDivergentError, NeuroDivergentResult};
use neuro_divergent::data::{TimeSeriesDataFrame, TimeSeriesSchema};
use num_traits::Float;
use polars::prelude::*;
use proptest::prelude::*;
use approx::assert_relative_eq;
use std::collections::HashMap;
use chrono::{DateTime, TimeZone, Utc, NaiveDate};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;
use std::io::Write;

// ============================================================================
// Mock Data Processing Components for Testing
// ============================================================================

/// Mock standard scaler for data normalization
#[derive(Clone, Debug)]
struct MockStandardScaler<T: Float> {
    mean: Option<T>,
    std: Option<T>,
    fitted: bool,
}

impl<T: Float> MockStandardScaler<T> {
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
            fitted: false,
        }
    }
    
    pub fn fit(&mut self, data: &[T]) -> NeuroDivergentResult<()> {
        if data.is_empty() {
            return Err(NeuroDivergentError::data("Cannot fit scaler on empty data"));
        }
        
        let n = T::from(data.len()).unwrap();
        let mean = data.iter().copied().fold(T::zero(), |acc, x| acc + x) / n;
        
        let variance = data.iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x) / n;
            
        let std = variance.sqrt();
        
        self.mean = Some(mean);
        self.std = Some(std);
        self.fitted = true;
        
        Ok(())
    }
    
    pub fn transform(&self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
        if !self.fitted {
            return Err(NeuroDivergentError::data("Scaler has not been fitted"));
        }
        
        let mean = self.mean.unwrap();
        let std = self.std.unwrap();
        
        if std == T::zero() {
            return Ok(vec![T::zero(); data.len()]);
        }
        
        let transformed = data.iter()
            .map(|&x| (x - mean) / std)
            .collect();
            
        Ok(transformed)
    }
    
    pub fn fit_transform(&mut self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
        self.fit(data)?;
        self.transform(data)
    }
    
    pub fn inverse_transform(&self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
        if !self.fitted {
            return Err(NeuroDivergentError::data("Scaler has not been fitted"));
        }
        
        let mean = self.mean.unwrap();
        let std = self.std.unwrap();
        
        let transformed = data.iter()
            .map(|&x| x * std + mean)
            .collect();
            
        Ok(transformed)
    }
}

/// Mock Min-Max scaler
#[derive(Clone, Debug)]
struct MockMinMaxScaler<T: Float> {
    min_val: Option<T>,
    max_val: Option<T>,
    feature_range: (T, T),
    fitted: bool,
}

impl<T: Float> MockMinMaxScaler<T> {
    pub fn new(feature_range: (T, T)) -> Self {
        Self {
            min_val: None,
            max_val: None,
            feature_range,
            fitted: false,
        }
    }
    
    pub fn fit(&mut self, data: &[T]) -> NeuroDivergentResult<()> {
        if data.is_empty() {
            return Err(NeuroDivergentError::data("Cannot fit scaler on empty data"));
        }
        
        let min_val = data.iter().fold(data[0], |min, &x| min.min(x));
        let max_val = data.iter().fold(data[0], |max, &x| max.max(x));
        
        self.min_val = Some(min_val);
        self.max_val = Some(max_val);
        self.fitted = true;
        
        Ok(())
    }
    
    pub fn transform(&self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
        if !self.fitted {
            return Err(NeuroDivergentError::data("Scaler has not been fitted"));
        }
        
        let min_val = self.min_val.unwrap();
        let max_val = self.max_val.unwrap();
        
        if max_val == min_val {
            return Ok(vec![self.feature_range.0; data.len()]);
        }
        
        let scale = (self.feature_range.1 - self.feature_range.0) / (max_val - min_val);
        let transformed = data.iter()
            .map(|&x| (x - min_val) * scale + self.feature_range.0)
            .collect();
            
        Ok(transformed)
    }
    
    pub fn fit_transform(&mut self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
        self.fit(data)?;
        self.transform(data)
    }
}

/// Mock data imputer for handling missing values
#[derive(Clone, Debug)]
enum MockImputationStrategy {
    Mean,
    Median,
    Mode,
    Constant(f64),
    Forward,
    Backward,
}

#[derive(Clone, Debug)]
struct MockDataImputer<T: Float> {
    strategy: MockImputationStrategy,
    fill_value: Option<T>,
}

impl<T: Float> MockDataImputer<T> {
    pub fn new(strategy: MockImputationStrategy) -> Self {
        Self {
            strategy,
            fill_value: None,
        }
    }
    
    pub fn fit(&mut self, data: &[Option<T>]) -> NeuroDivergentResult<()> {
        let valid_data: Vec<T> = data.iter().filter_map(|&x| x).collect();
        
        if valid_data.is_empty() {
            return Err(NeuroDivergentError::data("No valid data to fit imputer"));
        }
        
        match self.strategy {
            MockImputationStrategy::Mean => {
                let sum = valid_data.iter().copied().fold(T::zero(), |acc, x| acc + x);
                self.fill_value = Some(sum / T::from(valid_data.len()).unwrap());
            },
            MockImputationStrategy::Median => {
                let mut sorted = valid_data;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = sorted.len() / 2;
                self.fill_value = Some(sorted[mid]);
            },
            MockImputationStrategy::Constant(val) => {
                self.fill_value = Some(T::from(*val).unwrap());
            },
            _ => {
                // For Forward/Backward, no fitting needed
            }
        }
        
        Ok(())
    }
    
    pub fn transform(&self, data: &[Option<T>]) -> NeuroDivergentResult<Vec<T>> {
        match self.strategy {
            MockImputationStrategy::Forward => {
                let mut result = Vec::new();
                let mut last_valid = T::zero();
                
                for &value in data {
                    match value {
                        Some(val) => {
                            last_valid = val;
                            result.push(val);
                        },
                        None => {
                            result.push(last_valid);
                        }
                    }
                }
                Ok(result)
            },
            MockImputationStrategy::Backward => {
                let mut result = vec![T::zero(); data.len()];
                let mut next_valid = T::zero();
                
                for (i, &value) in data.iter().enumerate().rev() {
                    match value {
                        Some(val) => {
                            next_valid = val;
                            result[i] = val;
                        },
                        None => {
                            result[i] = next_valid;
                        }
                    }
                }
                Ok(result)
            },
            _ => {
                let fill_val = self.fill_value.ok_or_else(|| 
                    NeuroDivergentError::data("Imputer has not been fitted"))?;
                    
                Ok(data.iter().map(|&x| x.unwrap_or(fill_val)).collect())
            }
        }
    }
    
    pub fn fit_transform(&mut self, data: &[Option<T>]) -> NeuroDivergentResult<Vec<T>> {
        self.fit(data)?;
        self.transform(data)
    }
}

/// Mock feature engineering utilities
struct MockFeatureEngineer;

impl MockFeatureEngineer {
    /// Create lagged features
    pub fn create_lags<T: Float + Copy>(data: &[T], lags: &[usize]) -> HashMap<String, Vec<Option<T>>> {
        let mut features = HashMap::new();
        
        for &lag in lags {
            let mut lagged = vec![None; data.len()];
            for i in lag..data.len() {
                lagged[i] = Some(data[i - lag]);
            }
            features.insert(format!("lag_{}", lag), lagged);
        }
        
        features
    }
    
    /// Create rolling window features
    pub fn rolling_mean<T: Float + Copy>(data: &[T], window: usize) -> Vec<Option<T>> {
        let mut result = vec![None; data.len()];
        
        for i in window-1..data.len() {
            let sum = data[i+1-window..=i].iter().copied().fold(T::zero(), |acc, x| acc + x);
            result[i] = Some(sum / T::from(window).unwrap());
        }
        
        result
    }
    
    /// Create exponential moving average
    pub fn exponential_moving_average<T: Float + Copy>(data: &[T], alpha: T) -> Vec<T> {
        let mut result = vec![T::zero(); data.len()];
        if !data.is_empty() {
            result[0] = data[0];
            for i in 1..data.len() {
                result[i] = alpha * data[i] + (T::one() - alpha) * result[i-1];
            }
        }
        result
    }
    
    /// Create seasonal difference features
    pub fn seasonal_difference<T: Float + Copy>(data: &[T], period: usize) -> Vec<Option<T>> {
        let mut result = vec![None; data.len()];
        
        for i in period..data.len() {
            result[i] = Some(data[i] - data[i - period]);
        }
        
        result
    }
}

/// Mock outlier detector
#[derive(Clone, Debug)]
struct MockOutlierDetector<T: Float> {
    threshold: T,
    method: OutlierMethod,
}

#[derive(Clone, Debug)]
enum OutlierMethod {
    ZScore,
    IQR,
    ModifiedZScore,
}

impl<T: Float> MockOutlierDetector<T> {
    pub fn new(method: OutlierMethod, threshold: T) -> Self {
        Self { method, threshold }
    }
    
    pub fn detect_outliers(&self, data: &[T]) -> NeuroDivergentResult<Vec<bool>> {
        match self.method {
            OutlierMethod::ZScore => {
                let n = T::from(data.len()).unwrap();
                let mean = data.iter().copied().fold(T::zero(), |acc, x| acc + x) / n;
                let variance = data.iter()
                    .map(|&x| {
                        let diff = x - mean;
                        diff * diff
                    })
                    .fold(T::zero(), |acc, x| acc + x) / n;
                let std = variance.sqrt();
                
                if std == T::zero() {
                    return Ok(vec![false; data.len()]);
                }
                
                Ok(data.iter()
                    .map(|&x| ((x - mean) / std).abs() > self.threshold)
                    .collect())
            },
            OutlierMethod::IQR => {
                let mut sorted = data.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                
                let q1_idx = sorted.len() / 4;
                let q3_idx = 3 * sorted.len() / 4;
                let q1 = sorted[q1_idx];
                let q3 = sorted[q3_idx];
                let iqr = q3 - q1;
                
                let lower_bound = q1 - self.threshold * iqr;
                let upper_bound = q3 + self.threshold * iqr;
                
                Ok(data.iter()
                    .map(|&x| x < lower_bound || x > upper_bound)
                    .collect())
            },
            OutlierMethod::ModifiedZScore => {
                // Using median absolute deviation
                let mut sorted = data.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = sorted[sorted.len() / 2];
                
                let mad = {
                    let mut deviations: Vec<T> = data.iter()
                        .map(|&x| (x - median).abs())
                        .collect();
                    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    deviations[deviations.len() / 2]
                };
                
                if mad == T::zero() {
                    return Ok(vec![false; data.len()]);
                }
                
                let modified_z_factor = T::from(0.6745).unwrap(); // 0.6745 is the 0.75 quantile of standard normal
                
                Ok(data.iter()
                    .map(|&x| {
                        let modified_z = modified_z_factor * (x - median).abs() / mad;
                        modified_z > self.threshold
                    })
                    .collect())
            }
        }
    }
}

// ============================================================================
// Data Loading and I/O Tests
// ============================================================================

#[cfg(test)]
mod io_tests {
    use super::*;

    #[test]
    fn test_time_series_dataframe_from_polars() {
        let data = df! {
            "unique_id" => ["A", "A", "B", "B"],
            "ds" => [1, 2, 1, 2],
            "y" => [10.0, 11.0, 20.0, 21.0],
        }.unwrap();
        
        let schema = TimeSeriesSchema::default();
        let ts_df = TimeSeriesDataFrame::<f64>::from_polars(data, schema, None);
        
        assert!(ts_df.is_ok());
        let ts_df = ts_df.unwrap();
        assert_eq!(ts_df.shape(), (4, 3));
        assert_eq!(ts_df.n_series(), 2);
    }

    #[test]
    fn test_csv_export_import_roundtrip() {
        let original_data = df! {
            "unique_id" => ["series_1", "series_1", "series_2", "series_2"],
            "ds" => [1, 2, 1, 2],
            "y" => [1.5, 2.5, 3.5, 4.5],
        }.unwrap();
        
        let schema = TimeSeriesSchema::default();
        let ts_df = TimeSeriesDataFrame::<f64>::from_polars(original_data, schema.clone(), None).unwrap();
        
        // Create temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path();
        
        // Export to CSV
        let export_result = ts_df.to_csv(temp_path);
        assert!(export_result.is_ok());
        
        // Import from CSV
        let imported_ts_df = TimeSeriesDataFrame::<f64>::from_csv(temp_path, schema, None);
        assert!(imported_ts_df.is_ok());
        
        let imported = imported_ts_df.unwrap();
        assert_eq!(imported.shape(), ts_df.shape());
    }

    #[test]
    fn test_schema_validation_with_missing_columns() {
        let invalid_data = df! {
            "unique_id" => ["A", "B"],
            "ds" => [1, 2],
            // Missing "y" column
        }.unwrap();
        
        let schema = TimeSeriesSchema::default();
        let result = TimeSeriesDataFrame::<f64>::from_polars(invalid_data, schema, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_schema_with_exogenous_features() {
        let data = df! {
            "unique_id" => ["A", "A", "B", "B"],
            "ds" => [1, 2, 1, 2],
            "y" => [10.0, 11.0, 20.0, 21.0],
            "temperature" => [25.0, 26.0, 30.0, 31.0],
            "humidity" => [60.0, 65.0, 70.0, 75.0],
        }.unwrap();
        
        let schema = TimeSeriesSchema::new("unique_id", "ds", "y")
            .with_exogenous_features(vec!["temperature".to_string(), "humidity".to_string()]);
        
        let ts_df = TimeSeriesDataFrame::<f64>::from_polars(data, schema, None);
        assert!(ts_df.is_ok());
        
        let ts_df = ts_df.unwrap();
        assert_eq!(ts_df.shape(), (4, 5));
    }
}

// ============================================================================
// Preprocessing Tests
// ============================================================================

#[cfg(test)]
mod preprocessing_tests {
    use super::*;

    #[test]
    fn test_standard_scaler() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mut scaler = MockStandardScaler::new();
        
        // Fit and transform
        let scaled = scaler.fit_transform(&data).unwrap();
        
        // Mean should be approximately 0
        let mean = scaled.iter().copied().sum::<f64>() / scaled.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
        
        // Std should be approximately 1
        let variance = scaled.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>() / scaled.len() as f64;
        let std = variance.sqrt();
        assert_relative_eq!(std, 1.0, epsilon = 1e-10);
        
        // Inverse transform should recover original
        let inverse = scaler.inverse_transform(&scaled).unwrap();
        for (orig, inv) in data.iter().zip(inverse.iter()) {
            assert_relative_eq!(orig, inv, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_minmax_scaler() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let feature_range = (0.0, 1.0);
        let mut scaler = MockMinMaxScaler::new(feature_range);
        
        let scaled = scaler.fit_transform(&data).unwrap();
        
        // Check range
        let min_scaled = scaled.iter().fold(scaled[0], |min, &x| min.min(x));
        let max_scaled = scaled.iter().fold(scaled[0], |max, &x| max.max(x));
        
        assert_relative_eq!(min_scaled, 0.0, epsilon = 1e-10);
        assert_relative_eq!(max_scaled, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scaler_with_constant_data() {
        let data = vec![5.0f64; 10]; // All same value
        let mut scaler = MockStandardScaler::new();
        
        let scaled = scaler.fit_transform(&data).unwrap();
        
        // Should return all zeros for constant data
        for &value in &scaled {
            assert_relative_eq!(value, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_scaler_with_empty_data() {
        let data: Vec<f64> = vec![];
        let mut scaler = MockStandardScaler::new();
        
        let result = scaler.fit(&data);
        assert!(result.is_err());
    }
}

// ============================================================================
// Imputation Tests
// ============================================================================

#[cfg(test)]
mod imputation_tests {
    use super::*;

    #[test]
    fn test_mean_imputation() {
        let data = vec![Some(1.0f64), None, Some(3.0), Some(4.0), None];
        let mut imputer = MockDataImputer::new(MockImputationStrategy::Mean);
        
        let imputed = imputer.fit_transform(&data).unwrap();
        
        // Mean of [1.0, 3.0, 4.0] = 8.0/3 ≈ 2.667
        let expected_fill = (1.0 + 3.0 + 4.0) / 3.0;
        
        assert_relative_eq!(imputed[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(imputed[1], expected_fill, epsilon = 1e-10);
        assert_relative_eq!(imputed[2], 3.0, epsilon = 1e-10);
        assert_relative_eq!(imputed[3], 4.0, epsilon = 1e-10);
        assert_relative_eq!(imputed[4], expected_fill, epsilon = 1e-10);
    }

    #[test]
    fn test_forward_fill_imputation() {
        let data = vec![Some(1.0f64), None, None, Some(4.0), None];
        let imputer = MockDataImputer::new(MockImputationStrategy::Forward);
        
        let imputed = imputer.transform(&data).unwrap();
        
        assert_relative_eq!(imputed[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(imputed[1], 1.0, epsilon = 1e-10); // Forward filled
        assert_relative_eq!(imputed[2], 1.0, epsilon = 1e-10); // Forward filled
        assert_relative_eq!(imputed[3], 4.0, epsilon = 1e-10);
        assert_relative_eq!(imputed[4], 4.0, epsilon = 1e-10); // Forward filled
    }

    #[test]
    fn test_backward_fill_imputation() {
        let data = vec![None, Some(2.0f64), None, Some(4.0), None];
        let imputer = MockDataImputer::new(MockImputationStrategy::Backward);
        
        let imputed = imputer.transform(&data).unwrap();
        
        assert_relative_eq!(imputed[0], 2.0, epsilon = 1e-10); // Backward filled
        assert_relative_eq!(imputed[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(imputed[2], 4.0, epsilon = 1e-10); // Backward filled
        assert_relative_eq!(imputed[3], 4.0, epsilon = 1e-10);
        assert_relative_eq!(imputed[4], 0.0, epsilon = 1e-10); // No backward value
    }

    #[test]
    fn test_constant_imputation() {
        let data = vec![Some(1.0f64), None, Some(3.0), None];
        let mut imputer = MockDataImputer::new(MockImputationStrategy::Constant(-1.0));
        
        let imputed = imputer.fit_transform(&data).unwrap();
        
        assert_relative_eq!(imputed[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(imputed[1], -1.0, epsilon = 1e-10);
        assert_relative_eq!(imputed[2], 3.0, epsilon = 1e-10);
        assert_relative_eq!(imputed[3], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_imputation_all_missing() {
        let data = vec![None, None, None];
        let mut imputer = MockDataImputer::new(MockImputationStrategy::Mean);
        
        let result = imputer.fit(&data);
        assert!(result.is_err());
    }
}

// ============================================================================
// Feature Engineering Tests
// ============================================================================

#[cfg(test)]
mod feature_engineering_tests {
    use super::*;

    #[test]
    fn test_lag_features() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let lags = vec![1, 2];
        
        let lag_features = MockFeatureEngineer::create_lags(&data, &lags);
        
        assert!(lag_features.contains_key("lag_1"));
        assert!(lag_features.contains_key("lag_2"));
        
        let lag_1 = &lag_features["lag_1"];
        assert_eq!(lag_1[0], None);
        assert_eq!(lag_1[1], Some(1.0));
        assert_eq!(lag_1[2], Some(2.0));
        assert_eq!(lag_1[3], Some(3.0));
        assert_eq!(lag_1[4], Some(4.0));
        
        let lag_2 = &lag_features["lag_2"];
        assert_eq!(lag_2[0], None);
        assert_eq!(lag_2[1], None);
        assert_eq!(lag_2[2], Some(1.0));
        assert_eq!(lag_2[3], Some(2.0));
        assert_eq!(lag_2[4], Some(3.0));
    }

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let window = 3;
        
        let rolling_means = MockFeatureEngineer::rolling_mean(&data, window);
        
        assert_eq!(rolling_means[0], None);
        assert_eq!(rolling_means[1], None);
        assert_relative_eq!(rolling_means[2].unwrap(), 2.0, epsilon = 1e-10); // (1+2+3)/3
        assert_relative_eq!(rolling_means[3].unwrap(), 3.0, epsilon = 1e-10); // (2+3+4)/3
        assert_relative_eq!(rolling_means[4].unwrap(), 4.0, epsilon = 1e-10); // (3+4+5)/3
    }

    #[test]
    fn test_exponential_moving_average() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let alpha = 0.3;
        
        let ema = MockFeatureEngineer::exponential_moving_average(&data, alpha);
        
        assert_relative_eq!(ema[0], 1.0, epsilon = 1e-10);
        // ema[1] = 0.3 * 2.0 + 0.7 * 1.0 = 1.3
        assert_relative_eq!(ema[1], 1.3, epsilon = 1e-10);
        // ema[2] = 0.3 * 3.0 + 0.7 * 1.3 = 1.81
        assert_relative_eq!(ema[2], 1.81, epsilon = 1e-10);
    }

    #[test]
    fn test_seasonal_difference() {
        let data = vec![10.0f64, 15.0, 20.0, 25.0, 30.0, 35.0];
        let period = 3;
        
        let seasonal_diff = MockFeatureEngineer::seasonal_difference(&data, period);
        
        assert_eq!(seasonal_diff[0], None);
        assert_eq!(seasonal_diff[1], None);
        assert_eq!(seasonal_diff[2], None);
        assert_relative_eq!(seasonal_diff[3].unwrap(), 15.0, epsilon = 1e-10); // 25 - 10
        assert_relative_eq!(seasonal_diff[4].unwrap(), 15.0, epsilon = 1e-10); // 30 - 15
        assert_relative_eq!(seasonal_diff[5].unwrap(), 15.0, epsilon = 1e-10); // 35 - 20
    }
}

// ============================================================================
// Outlier Detection Tests
// ============================================================================

#[cfg(test)]
mod outlier_detection_tests {
    use super::*;

    #[test]
    fn test_zscore_outlier_detection() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100.0 is outlier
        let detector = MockOutlierDetector::new(OutlierMethod::ZScore, 2.0);
        
        let outliers = detector.detect_outliers(&data).unwrap();
        
        // Only the last value (100.0) should be flagged as outlier
        assert!(!outliers[0]);
        assert!(!outliers[1]);
        assert!(!outliers[2]);
        assert!(!outliers[3]);
        assert!(!outliers[4]);
        assert!(outliers[5]);
    }

    #[test]
    fn test_iqr_outlier_detection() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0];
        let detector = MockOutlierDetector::new(OutlierMethod::IQR, 1.5);
        
        let outliers = detector.detect_outliers(&data).unwrap();
        
        // The last value should be detected as outlier
        assert!(outliers[outliers.len() - 1]);
    }

    #[test]
    fn test_outlier_detection_no_outliers() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0]; // No outliers
        let detector = MockOutlierDetector::new(OutlierMethod::ZScore, 2.0);
        
        let outliers = detector.detect_outliers(&data).unwrap();
        
        // No values should be flagged as outliers
        for &is_outlier in &outliers {
            assert!(!is_outlier);
        }
    }

    #[test]
    fn test_outlier_detection_constant_data() {
        let data = vec![5.0f64; 10]; // All same value
        let detector = MockOutlierDetector::new(OutlierMethod::ZScore, 2.0);
        
        let outliers = detector.detect_outliers(&data).unwrap();
        
        // No values should be flagged as outliers in constant data
        for &is_outlier in &outliers {
            assert!(!is_outlier);
        }
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;

    proptest! {
        #[test]
        fn test_standard_scaler_properties(
            data in prop::collection::vec(0.0f64..1000.0, 1..100)
        ) {
            let mut scaler = MockStandardScaler::new();
            let scaled = scaler.fit_transform(&data).unwrap();
            
            // Mean should be close to zero
            let mean = scaled.iter().copied().sum::<f64>() / scaled.len() as f64;
            assert!((mean.abs() < 1e-10) || data.iter().all(|&x| x == data[0])); // constant data exception
            
            // Inverse transform should recover original (approximately)
            let inverse = scaler.inverse_transform(&scaled).unwrap();
            for (orig, inv) in data.iter().zip(inverse.iter()) {
                assert_relative_eq!(orig, inv, epsilon = 1e-8);
            }
        }

        #[test]
        fn test_minmax_scaler_properties(
            data in prop::collection::vec(-1000.0f64..1000.0, 1..100),
            min_range in -10.0f64..0.0,
            max_range in 1.0f64..10.0
        ) {
            prop_assume!(min_range < max_range);
            prop_assume!(!data.iter().all(|&x| x == data[0])); // Avoid constant data
            
            let feature_range = (min_range, max_range);
            let mut scaler = MockMinMaxScaler::new(feature_range);
            let scaled = scaler.fit_transform(&data).unwrap();
            
            // All values should be within the specified range
            for &value in &scaled {
                assert!(value >= min_range - 1e-10 && value <= max_range + 1e-10);
            }
            
            // Min and max should match the range (approximately)
            let min_scaled = scaled.iter().fold(scaled[0], |min, &x| min.min(x));
            let max_scaled = scaled.iter().fold(scaled[0], |max, &x| max.max(x));
            
            assert_relative_eq!(min_scaled, min_range, epsilon = 1e-8);
            assert_relative_eq!(max_scaled, max_range, epsilon = 1e-8);
        }

        #[test]
        fn test_imputation_preserves_valid_values(
            data in prop::collection::vec(prop::option::of(0.0f64..100.0), 1..50)
        ) {
            prop_assume!(data.iter().any(|x| x.is_some())); // At least one valid value
            
            let mut imputer = MockDataImputer::new(MockImputationStrategy::Mean);
            let imputed = imputer.fit_transform(&data).unwrap();
            
            // All originally valid values should be preserved
            for (original, &imputed_val) in data.iter().zip(imputed.iter()) {
                if let Some(orig_val) = original {
                    assert_relative_eq!(imputed_val, *orig_val, epsilon = 1e-10);
                }
            }
            
            // All values should be finite
            for &value in &imputed {
                assert!(value.is_finite());
            }
        }

        #[test]
        fn test_rolling_mean_properties(
            data in prop::collection::vec(0.0f64..100.0, 5..50),
            window in 1usize..10
        ) {
            let rolling_means = MockFeatureEngineer::rolling_mean(&data, window);
            
            // Length should match input
            assert_eq!(rolling_means.len(), data.len());
            
            // First (window-1) values should be None
            for i in 0..window-1 {
                assert_eq!(rolling_means[i], None);
            }
            
            // Remaining values should be valid
            for i in window-1..rolling_means.len() {
                assert!(rolling_means[i].is_some());
                let mean_val = rolling_means[i].unwrap();
                assert!(mean_val.is_finite());
                
                // Mean should be within the range of window values
                let window_data = &data[i+1-window..=i];
                let min_in_window = window_data.iter().fold(window_data[0], |min, &x| min.min(x));
                let max_in_window = window_data.iter().fold(window_data[0], |max, &x| max.max(x));
                
                assert!(mean_val >= min_in_window && mean_val <= max_in_window);
            }
        }
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_data_preprocessing_pipeline() {
        // Create sample data with missing values and outliers
        let raw_data = vec![
            Some(1.0f64), None, Some(3.0), Some(100.0), // 100.0 is outlier
            Some(5.0), Some(6.0), None, Some(8.0),
        ];
        
        // Step 1: Detect outliers (before imputation)
        let detector = MockOutlierDetector::new(OutlierMethod::ZScore, 2.0);
        
        // Step 2: Impute missing values
        let mut imputer = MockDataImputer::new(MockImputationStrategy::Mean);
        let imputed = imputer.fit_transform(&raw_data).unwrap();
        
        // Step 3: Remove outliers after detection
        let outlier_flags = detector.detect_outliers(&imputed).unwrap();
        let clean_data: Vec<f64> = imputed.iter().zip(outlier_flags.iter())
            .filter_map(|(&val, &is_outlier)| if !is_outlier { Some(val) } else { None })
            .collect();
        
        // Step 4: Scale the clean data
        let mut scaler = MockStandardScaler::new();
        let scaled = scaler.fit_transform(&clean_data).unwrap();
        
        // Step 5: Create features
        let lag_features = MockFeatureEngineer::create_lags(&clean_data, &[1, 2]);
        let rolling_features = MockFeatureEngineer::rolling_mean(&clean_data, 3);
        
        // Verify the pipeline worked
        assert!(scaled.len() > 0);
        assert!(scaled.len() < imputed.len()); // Outliers removed
        
        // Verify features were created
        assert!(lag_features.contains_key("lag_1"));
        assert!(lag_features.contains_key("lag_2"));
        assert_eq!(rolling_features.len(), clean_data.len());
        
        // Verify scaled data properties
        let mean = scaled.iter().copied().sum::<f64>() / scaled.len() as f64;
        assert!(mean.abs() < 1e-8); // Mean should be close to zero
    }

    #[test]
    fn test_time_series_data_workflow() {
        // Create time series DataFrame
        let data = df! {
            "unique_id" => ["A", "A", "A", "A", "B", "B", "B", "B"],
            "ds" => [1, 2, 3, 4, 1, 2, 3, 4],
            "y" => [10.0, 12.0, 14.0, 16.0, 20.0, 22.0, 24.0, 26.0],
            "temperature" => [25.0, 26.0, 27.0, 28.0, 30.0, 31.0, 32.0, 33.0],
        }.unwrap();
        
        let schema = TimeSeriesSchema::new("unique_id", "ds", "y")
            .with_exogenous_features(vec!["temperature".to_string()]);
        
        let ts_df = TimeSeriesDataFrame::<f64>::from_polars(data, schema, None).unwrap();
        
        // Filter by series
        let series_a = ts_df.filter_by_id("A").unwrap();
        assert_eq!(series_a.shape().0, 4);
        
        let series_b = ts_df.filter_by_id("B").unwrap();
        assert_eq!(series_b.shape().0, 4);
        
        // Extract unique IDs
        let unique_ids = ts_df.unique_ids().unwrap();
        assert_eq!(unique_ids.len(), 2);
        assert!(unique_ids.contains(&"A".to_string()));
        assert!(unique_ids.contains(&"B".to_string()));
        
        // Verify shapes
        assert_eq!(ts_df.shape(), (8, 4)); // 8 rows, 4 columns (id, ds, y, temperature)
        assert_eq!(ts_df.n_series(), 2);
    }
}