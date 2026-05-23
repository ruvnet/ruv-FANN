//! # Time Series Preprocessing
//!
//! This module provides comprehensive preprocessing capabilities for time series data,
//! including various scaling techniques, transformations, and normalization strategies.

use crate::{DataPipelineError, Result, TimeSeriesData, DataTransform, FittableTransform};
use num_traits::Float;
use ndarray::{Array1, Array2, ArrayView1};
use std::collections::HashMap;
use std::marker::PhantomData;

#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

/// Trait for data scalers that can normalize time series values
pub trait Scaler<T: Float> {
    /// Fit the scaler to the data
    fn fit(&mut self, data: &[T]) -> Result<()>;
    
    /// Transform the data using fitted parameters
    fn transform(&self, data: &[T]) -> Result<Vec<T>>;
    
    /// Inverse transform to get back original scale
    fn inverse_transform(&self, data: &[T]) -> Result<Vec<T>>;
    
    /// Fit and transform in one step
    fn fit_transform(&mut self, data: &[T]) -> Result<Vec<T>> {
        self.fit(data)?;
        self.transform(data)
    }
    
    /// Check if the scaler has been fitted
    fn is_fitted(&self) -> bool;
}

/// Standard scaler that normalizes data to have zero mean and unit variance
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct StandardScaler<T: Float> {
    mean: Option<T>,
    std: Option<T>,
    with_mean: bool,
    with_std: bool,
}

impl<T: Float> Default for StandardScaler<T> {
    fn default() -> Self {
        Self {
            mean: None,
            std: None,
            with_mean: true,
            with_std: true,
        }
    }
}

impl<T: Float> StandardScaler<T> {
    /// Create a new standard scaler
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a scaler that only centers (subtracts mean)
    pub fn new_center_only() -> Self {
        Self {
            mean: None,
            std: None,
            with_mean: true,
            with_std: false,
        }
    }
    
    /// Create a scaler that only scales (divides by std)
    pub fn new_scale_only() -> Self {
        Self {
            mean: None,
            std: None,
            with_mean: false,
            with_std: true,
        }
    }
    
    /// Get the fitted mean
    pub fn mean(&self) -> Option<T> {
        self.mean
    }
    
    /// Get the fitted standard deviation  
    pub fn std(&self) -> Option<T> {
        self.std
    }
}

impl<T: Float> Scaler<T> for StandardScaler<T> {
    fn fit(&mut self, data: &[T]) -> Result<()> {
        if data.is_empty() {
            return Err(DataPipelineError::InvalidFormat {
                message: "Cannot fit scaler on empty data".to_string(),
            });
        }
        
        if self.with_mean {
            let sum = data.iter().fold(T::zero(), |acc, &x| acc + x);
            self.mean = Some(sum / T::from(data.len()).unwrap());
        }
        
        if self.with_std {
            let mean = if self.with_mean {
                self.mean.unwrap()
            } else {
                T::zero()
            };
            
            let variance = data.iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x) / T::from(data.len()).unwrap();
            
            self.std = Some(variance.sqrt());
            
            // Prevent division by zero
            if self.std.unwrap() == T::zero() {
                self.std = Some(T::one());
            }
        }
        
        Ok(())
    }
    
    fn transform(&self, data: &[T]) -> Result<Vec<T>> {
        if !self.is_fitted() {
            return Err(DataPipelineError::ValidationFailed {
                reason: "Scaler has not been fitted".to_string(),
            });
        }
        
        let mut result = data.to_vec();
        
        if self.with_mean {
            let mean = self.mean.unwrap();
            for value in &mut result {
                *value = *value - mean;
            }
        }
        
        if self.with_std {
            let std = self.std.unwrap();
            for value in &mut result {
                *value = *value / std;
            }
        }
        
        Ok(result)
    }
    
    fn inverse_transform(&self, data: &[T]) -> Result<Vec<T>> {
        if !self.is_fitted() {
            return Err(DataPipelineError::ValidationFailed {
                reason: "Scaler has not been fitted".to_string(),
            });
        }
        
        let mut result = data.to_vec();
        
        if self.with_std {
            let std = self.std.unwrap();
            for value in &mut result {
                *value = *value * std;
            }
        }
        
        if self.with_mean {
            let mean = self.mean.unwrap();
            for value in &mut result {
                *value = *value + mean;
            }
        }
        
        Ok(result)
    }
    
    fn is_fitted(&self) -> bool {
        (!self.with_mean || self.mean.is_some()) && (!self.with_std || self.std.is_some())
    }
}

/// MinMax scaler that normalizes data to a specified range [min_val, max_val]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct MinMaxScaler<T: Float> {
    data_min: Option<T>,
    data_max: Option<T>,
    feature_range: (T, T),
}

impl<T: Float> Default for MinMaxScaler<T> {
    fn default() -> Self {
        Self {
            data_min: None,
            data_max: None,
            feature_range: (T::zero(), T::one()),
        }
    }
}

impl<T: Float> MinMaxScaler<T> {
    /// Create a new MinMax scaler with default range [0, 1]
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a MinMax scaler with custom range
    pub fn with_range(min_val: T, max_val: T) -> Self {
        Self {
            data_min: None,
            data_max: None,
            feature_range: (min_val, max_val),
        }
    }
    
    /// Get the fitted data minimum
    pub fn data_min(&self) -> Option<T> {
        self.data_min
    }
    
    /// Get the fitted data maximum
    pub fn data_max(&self) -> Option<T> {
        self.data_max
    }
    
    /// Get the feature range
    pub fn feature_range(&self) -> (T, T) {
        self.feature_range
    }
}

impl<T: Float> Scaler<T> for MinMaxScaler<T> {
    fn fit(&mut self, data: &[T]) -> Result<()> {
        if data.is_empty() {
            return Err(DataPipelineError::InvalidFormat {
                message: "Cannot fit scaler on empty data".to_string(),
            });
        }
        
        self.data_min = Some(*data.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap());
        self.data_max = Some(*data.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap());
        
        Ok(())
    }
    
    fn transform(&self, data: &[T]) -> Result<Vec<T>> {
        if !self.is_fitted() {
            return Err(DataPipelineError::ValidationFailed {
                reason: "Scaler has not been fitted".to_string(),
            });
        }
        
        let data_min = self.data_min.unwrap();
        let data_max = self.data_max.unwrap();
        let scale = data_max - data_min;
        
        if scale == T::zero() {
            // All values are the same, return the minimum of the feature range
            return Ok(vec![self.feature_range.0; data.len()]);
        }
        
        let (min_range, max_range) = self.feature_range;
        let range_scale = max_range - min_range;
        
        let result = data.iter()
            .map(|&x| {
                let normalized = (x - data_min) / scale;
                normalized * range_scale + min_range
            })
            .collect();
        
        Ok(result)
    }
    
    fn inverse_transform(&self, data: &[T]) -> Result<Vec<T>> {
        if !self.is_fitted() {
            return Err(DataPipelineError::ValidationFailed {
                reason: "Scaler has not been fitted".to_string(),
            });
        }
        
        let data_min = self.data_min.unwrap();
        let data_max = self.data_max.unwrap();
        let scale = data_max - data_min;
        
        let (min_range, max_range) = self.feature_range;
        let range_scale = max_range - min_range;
        
        if range_scale == T::zero() {
            return Ok(vec![data_min; data.len()]);
        }
        
        let result = data.iter()
            .map(|&x| {
                let normalized = (x - min_range) / range_scale;
                normalized * scale + data_min
            })
            .collect();
        
        Ok(result)
    }
    
    fn is_fitted(&self) -> bool {
        self.data_min.is_some() && self.data_max.is_some()
    }
}

/// Robust scaler that uses median and interquartile range for scaling
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct RobustScaler<T: Float> {
    median: Option<T>,
    iqr: Option<T>,
    with_centering: bool,
    with_scaling: bool,
}

impl<T: Float> Default for RobustScaler<T> {
    fn default() -> Self {
        Self {
            median: None,
            iqr: None,
            with_centering: true,
            with_scaling: true,
        }
    }
}

impl<T: Float> RobustScaler<T> {
    /// Create a new robust scaler
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a robust scaler with custom centering and scaling options
    pub fn with_options(with_centering: bool, with_scaling: bool) -> Self {
        Self {
            median: None,
            iqr: None,
            with_centering,
            with_scaling,
        }
    }
    
    /// Calculate quantile from sorted data
    fn quantile(sorted_data: &[T], q: f64) -> T {
        let n = sorted_data.len();
        let index = q * (n - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        
        if lower == upper {
            sorted_data[lower]
        } else {
            let weight = T::from(index - index.floor()).unwrap();
            sorted_data[lower] * (T::one() - weight) + sorted_data[upper] * weight
        }
    }
}

impl<T: Float> Scaler<T> for RobustScaler<T> {
    fn fit(&mut self, data: &[T]) -> Result<()> {
        if data.is_empty() {
            return Err(DataPipelineError::InvalidFormat {
                message: "Cannot fit scaler on empty data".to_string(),
            });
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        if self.with_centering {
            self.median = Some(Self::quantile(&sorted_data, 0.5));
        }
        
        if self.with_scaling {
            let q75 = Self::quantile(&sorted_data, 0.75);
            let q25 = Self::quantile(&sorted_data, 0.25);
            self.iqr = Some(q75 - q25);
            
            // Prevent division by zero
            if self.iqr.unwrap() == T::zero() {
                self.iqr = Some(T::one());
            }
        }
        
        Ok(())
    }
    
    fn transform(&self, data: &[T]) -> Result<Vec<T>> {
        if !self.is_fitted() {
            return Err(DataPipelineError::ValidationFailed {
                reason: "Scaler has not been fitted".to_string(),
            });
        }
        
        let mut result = data.to_vec();
        
        if self.with_centering {
            let median = self.median.unwrap();
            for value in &mut result {
                *value = *value - median;
            }
        }
        
        if self.with_scaling {
            let iqr = self.iqr.unwrap();
            for value in &mut result {
                *value = *value / iqr;
            }
        }
        
        Ok(result)
    }
    
    fn inverse_transform(&self, data: &[T]) -> Result<Vec<T>> {
        if !self.is_fitted() {
            return Err(DataPipelineError::ValidationFailed {
                reason: "Scaler has not been fitted".to_string(),
            });
        }
        
        let mut result = data.to_vec();
        
        if self.with_scaling {
            let iqr = self.iqr.unwrap();
            for value in &mut result {
                *value = *value * iqr;
            }
        }
        
        if self.with_centering {
            let median = self.median.unwrap();
            for value in &mut result {
                *value = *value + median;
            }
        }
        
        Ok(result)
    }
    
    fn is_fitted(&self) -> bool {
        (!self.with_centering || self.median.is_some()) && 
        (!self.with_scaling || self.iqr.is_some())
    }
}

/// Quantile transformer that maps features to a uniform or normal distribution
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct QuantileTransformer<T: Float> {
    quantiles: Option<Vec<T>>,
    references: Option<Vec<T>>,
    n_quantiles: usize,
    output_distribution: QuantileDistribution,
    subsample: Option<usize>,
}

/// Output distribution for quantile transformer
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum QuantileDistribution {
    Uniform,
    Normal,
}

impl<T: Float> Default for QuantileTransformer<T> {
    fn default() -> Self {
        Self {
            quantiles: None,
            references: None,
            n_quantiles: 1000,
            output_distribution: QuantileDistribution::Uniform,
            subsample: Some(100000),
        }
    }
}

impl<T: Float> QuantileTransformer<T> {
    /// Create a new quantile transformer
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a quantile transformer with specified parameters
    pub fn with_params(
        n_quantiles: usize,
        output_distribution: QuantileDistribution,
        subsample: Option<usize>,
    ) -> Self {
        Self {
            quantiles: None,
            references: None,
            n_quantiles,
            output_distribution,
            subsample,
        }
    }
}

impl<T: Float> Scaler<T> for QuantileTransformer<T> {
    fn fit(&mut self, data: &[T]) -> Result<()> {
        if data.is_empty() {
            return Err(DataPipelineError::InvalidFormat {
                message: "Cannot fit transformer on empty data".to_string(),
            });
        }
        
        // Subsample if specified and data is large
        let sample_data = if let Some(subsample_size) = self.subsample {
            if data.len() > subsample_size {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                let mut sample: Vec<T> = data.to_vec();
                sample.shuffle(&mut rng);
                sample.truncate(subsample_size);
                sample
            } else {
                data.to_vec()
            }
        } else {
            data.to_vec()
        };
        
        let mut sorted_data = sample_data;
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        // Calculate quantiles
        let mut quantiles = Vec::with_capacity(self.n_quantiles);
        for i in 0..self.n_quantiles {
            let q = i as f64 / (self.n_quantiles - 1) as f64;
            let quantile = RobustScaler::quantile(&sorted_data, q);
            quantiles.push(quantile);
        }
        
        // Calculate reference values for output distribution
        let mut references = Vec::with_capacity(self.n_quantiles);
        match self.output_distribution {
            QuantileDistribution::Uniform => {
                for i in 0..self.n_quantiles {
                    let ref_val = T::from(i as f64 / (self.n_quantiles - 1) as f64).unwrap();
                    references.push(ref_val);
                }
            }
            QuantileDistribution::Normal => {
                for i in 0..self.n_quantiles {
                    let p = i as f64 / (self.n_quantiles - 1) as f64;
                    // Approximate normal quantile using inverse error function
                    let z = if p <= 0.0 {
                        T::from(-6.0).unwrap()
                    } else if p >= 1.0 {
                        T::from(6.0).unwrap()
                    } else {
                        // Simple approximation for normal quantile
                        let t = T::from(2.0 * p - 1.0).unwrap();
                        t * T::from(1.4142135623730951).unwrap() // sqrt(2)
                    };
                    references.push(z);
                }
            }
        }
        
        self.quantiles = Some(quantiles);
        self.references = Some(references);
        
        Ok(())
    }
    
    fn transform(&self, data: &[T]) -> Result<Vec<T>> {
        if !self.is_fitted() {
            return Err(DataPipelineError::ValidationFailed {
                reason: "Transformer has not been fitted".to_string(),
            });
        }
        
        let quantiles = self.quantiles.as_ref().unwrap();
        let references = self.references.as_ref().unwrap();
        
        let mut result = Vec::with_capacity(data.len());
        
        for &value in data {
            // Find the appropriate quantile
            let mut lower_idx = 0;
            let mut upper_idx = quantiles.len() - 1;
            
            // Binary search for the quantile
            while lower_idx < upper_idx - 1 {
                let mid_idx = (lower_idx + upper_idx) / 2;
                if value <= quantiles[mid_idx] {
                    upper_idx = mid_idx;
                } else {
                    lower_idx = mid_idx;
                }
            }
            
            // Interpolate between quantiles
            let transformed_value = if quantiles[lower_idx] == quantiles[upper_idx] {
                references[lower_idx]
            } else {
                let weight = (value - quantiles[lower_idx]) / (quantiles[upper_idx] - quantiles[lower_idx]);
                references[lower_idx] * (T::one() - weight) + references[upper_idx] * weight
            };
            
            result.push(transformed_value);
        }
        
        Ok(result)
    }
    
    fn inverse_transform(&self, data: &[T]) -> Result<Vec<T>> {
        if !self.is_fitted() {
            return Err(DataPipelineError::ValidationFailed {
                reason: "Transformer has not been fitted".to_string(),
            });
        }
        
        let quantiles = self.quantiles.as_ref().unwrap();
        let references = self.references.as_ref().unwrap();
        
        let mut result = Vec::with_capacity(data.len());
        
        for &value in data {
            // Find the appropriate reference
            let mut lower_idx = 0;
            let mut upper_idx = references.len() - 1;
            
            // Binary search for the reference
            while lower_idx < upper_idx - 1 {
                let mid_idx = (lower_idx + upper_idx) / 2;
                if value <= references[mid_idx] {
                    upper_idx = mid_idx;
                } else {
                    lower_idx = mid_idx;
                }
            }
            
            // Interpolate between quantiles
            let original_value = if references[lower_idx] == references[upper_idx] {
                quantiles[lower_idx]
            } else {
                let weight = (value - references[lower_idx]) / (references[upper_idx] - references[lower_idx]);
                quantiles[lower_idx] * (T::one() - weight) + quantiles[upper_idx] * weight
            };
            
            result.push(original_value);
        }
        
        Ok(result)
    }
    
    fn is_fitted(&self) -> bool {
        self.quantiles.is_some() && self.references.is_some()
    }
}

/// Differencing transformation for making time series stationary
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct DifferencingTransformer<T: Float> {
    order: usize,
    seasonal_periods: Option<usize>,
    original_values: Option<Vec<T>>,
}

impl<T: Float> DifferencingTransformer<T> {
    /// Create a new differencing transformer
    pub fn new(order: usize) -> Self {
        Self {
            order,
            seasonal_periods: None,
            original_values: None,
        }
    }
    
    /// Create a seasonal differencing transformer
    pub fn seasonal(order: usize, seasonal_periods: usize) -> Self {
        Self {
            order,
            seasonal_periods: Some(seasonal_periods),
            original_values: None,
        }
    }
    
    /// Apply differencing to data
    pub fn difference(&self, data: &[T]) -> Vec<T> {
        if data.len() <= self.order {
            return Vec::new();
        }
        
        let mut result = data.to_vec();
        
        for _ in 0..self.order {
            if let Some(periods) = self.seasonal_periods {
                // Seasonal differencing
                if result.len() <= periods {
                    break;
                }
                let mut seasonal_diff = Vec::with_capacity(result.len() - periods);
                for i in periods..result.len() {
                    seasonal_diff.push(result[i] - result[i - periods]);
                }
                result = seasonal_diff;
            } else {
                // Regular differencing
                if result.len() <= 1 {
                    break;
                }
                let mut diff = Vec::with_capacity(result.len() - 1);
                for i in 1..result.len() {
                    diff.push(result[i] - result[i - 1]);
                }
                result = diff;
            }
        }
        
        result
    }
    
    /// Integrate (inverse difference) the data
    pub fn integrate(&self, diff_data: &[T], initial_values: &[T]) -> Result<Vec<T>> {
        if initial_values.is_empty() {
            return Err(DataPipelineError::InvalidFormat {
                message: "Initial values required for integration".to_string(),
            });
        }
        
        let mut result = initial_values.to_vec();
        result.extend_from_slice(diff_data);
        
        for _ in 0..self.order {
            if let Some(periods) = self.seasonal_periods {
                // Seasonal integration
                for i in periods..result.len() {
                    result[i] = result[i] + result[i - periods];
                }
            } else {
                // Regular integration
                for i in 1..result.len() {
                    result[i] = result[i] + result[i - 1];
                }
            }
        }
        
        // Remove the prepended initial values
        let start_idx = if let Some(periods) = self.seasonal_periods {
            periods * self.order
        } else {
            self.order
        };
        
        if start_idx < result.len() {
            Ok(result[start_idx..].to_vec())
        } else {
            Ok(Vec::new())
        }
    }
}

impl<T: Float> DataTransform<T> for DifferencingTransformer<T> {
    fn transform(&self, data: &TimeSeriesData<T>) -> Result<TimeSeriesData<T>> {
        let values = data.values();
        let differenced = self.difference(&values);
        
        if differenced.is_empty() {
            return Err(DataPipelineError::InvalidFormat {
                message: "Insufficient data for differencing".to_string(),
            });
        }
        
        let mut transformed = data.clone();
        transformed.data_points.clear();
        
        // Skip the first few timestamps based on differencing order
        let skip_count = if let Some(periods) = self.seasonal_periods {
            periods * self.order
        } else {
            self.order
        };
        
        for (i, &value) in differenced.iter().enumerate() {
            if skip_count + i < data.timestamps().len() {
                let timestamp = data.timestamps()[skip_count + i];
                transformed.add_point(crate::DataPoint::new(timestamp, value));
            }
        }
        
        Ok(transformed)
    }
}

impl<T: Float> FittableTransform<T> for DifferencingTransformer<T> {
    fn fit(&mut self, data: &TimeSeriesData<T>) -> Result<()> {
        // Store original values for inverse transformation
        let values = data.values();
        let keep_count = if let Some(periods) = self.seasonal_periods {
            periods * self.order
        } else {
            self.order
        };
        
        if values.len() > keep_count {
            self.original_values = Some(values[..keep_count].to_vec());
        } else {
            self.original_values = Some(values);
        }
        
        Ok(())
    }
}

/// Log transformation for handling exponential trends
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct LogTransformer<T: Float> {
    base: T,
    offset: T,
}

impl<T: Float> Default for LogTransformer<T> {
    fn default() -> Self {
        Self {
            base: T::from(std::f64::consts::E).unwrap(), // Natural log
            offset: T::zero(),
        }
    }
}

impl<T: Float> LogTransformer<T> {
    /// Create a natural log transformer
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a log transformer with specified base
    pub fn with_base(base: T) -> Self {
        Self {
            base,
            offset: T::zero(),
        }
    }
    
    /// Create a log transformer with offset for handling zeros/negatives
    pub fn with_offset(offset: T) -> Self {
        Self {
            base: T::from(std::f64::consts::E).unwrap(),
            offset,
        }
    }
    
    /// Transform data using logarithm
    pub fn transform_values(&self, data: &[T]) -> Result<Vec<T>> {
        let mut result = Vec::with_capacity(data.len());
        
        for &value in data {
            let adjusted_value = value + self.offset;
            if adjusted_value <= T::zero() {
                return Err(DataPipelineError::InvalidFormat {
                    message: format!("Cannot take log of non-positive value: {}", 
                                   adjusted_value.to_f64().unwrap_or(0.0)),
                });
            }
            
            let log_value = if self.base == T::from(std::f64::consts::E).unwrap() {
                adjusted_value.ln()
            } else {
                adjusted_value.ln() / self.base.ln()
            };
            
            result.push(log_value);
        }
        
        Ok(result)
    }
    
    /// Inverse transform using exponential
    pub fn inverse_transform_values(&self, data: &[T]) -> Result<Vec<T>> {
        let mut result = Vec::with_capacity(data.len());
        
        for &value in data {
            let exp_value = if self.base == T::from(std::f64::consts::E).unwrap() {
                value.exp()
            } else {
                self.base.powf(value)
            };
            
            result.push(exp_value - self.offset);
        }
        
        Ok(result)
    }
}

impl<T: Float> DataTransform<T> for LogTransformer<T> {
    fn transform(&self, data: &TimeSeriesData<T>) -> Result<TimeSeriesData<T>> {
        let values = data.values();
        let transformed_values = self.transform_values(&values)?;
        
        let mut transformed = data.clone();
        for (point, &new_value) in transformed.data_points.iter_mut().zip(transformed_values.iter()) {
            point.value = new_value;
        }
        
        Ok(transformed)
    }
}

/// Box-Cox transformation for variance stabilization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct BoxCoxTransformer<T: Float> {
    lambda: Option<T>,
    fitted_lambda: Option<T>,
}

impl<T: Float> BoxCoxTransformer<T> {
    /// Create a Box-Cox transformer with automatic lambda selection
    pub fn new() -> Self {
        Self {
            lambda: None,
            fitted_lambda: None,
        }
    }
    
    /// Create a Box-Cox transformer with fixed lambda
    pub fn with_lambda(lambda: T) -> Self {
        Self {
            lambda: Some(lambda),
            fitted_lambda: Some(lambda),
        }
    }
    
    /// Box-Cox transformation function
    fn box_cox(value: T, lambda: T) -> T {
        if lambda == T::zero() {
            value.ln()
        } else {
            (value.powf(lambda) - T::one()) / lambda
        }
    }
    
    /// Inverse Box-Cox transformation
    fn inv_box_cox(value: T, lambda: T) -> T {
        if lambda == T::zero() {
            value.exp()
        } else {
            (value * lambda + T::one()).powf(T::one() / lambda)
        }
    }
    
    /// Find optimal lambda using log-likelihood
    fn find_optimal_lambda(&self, data: &[T]) -> Result<T> {
        let lambdas = Array1::linspace(-2.0, 2.0, 81);
        let mut best_lambda = T::zero();
        let mut best_likelihood = T::neg_infinity();
        
        for &lambda_f64 in lambdas.iter() {
            let lambda = T::from(lambda_f64).unwrap();
            
            // Transform data
            let mut transformed = Vec::new();
            let mut valid = true;
            
            for &value in data {
                if value <= T::zero() {
                    valid = false;
                    break;
                }
                transformed.push(Self::box_cox(value, lambda));
            }
            
            if !valid {
                continue;
            }
            
            // Calculate log-likelihood (simplified)
            let n = T::from(transformed.len()).unwrap();
            let mean = transformed.iter().fold(T::zero(), |acc, &x| acc + x) / n;
            let variance = transformed.iter()
                .map(|&x| (x - mean) * (x - mean))
                .fold(T::zero(), |acc, x| acc + x) / n;
            
            if variance <= T::zero() {
                continue;
            }
            
            let log_likelihood = -n / T::from(2.0).unwrap() * variance.ln() + 
                                (lambda - T::one()) * data.iter().fold(T::zero(), |acc, &x| acc + x.ln());
            
            if log_likelihood > best_likelihood {
                best_likelihood = log_likelihood;
                best_lambda = lambda;
            }
        }
        
        Ok(best_lambda)
    }
}

impl<T: Float> DataTransform<T> for BoxCoxTransformer<T> {
    fn transform(&self, data: &TimeSeriesData<T>) -> Result<TimeSeriesData<T>> {
        if self.fitted_lambda.is_none() {
            return Err(DataPipelineError::ValidationFailed {
                reason: "Transformer has not been fitted".to_string(),
            });
        }
        
        let lambda = self.fitted_lambda.unwrap();
        let values = data.values();
        
        // Check for non-positive values
        for &value in &values {
            if value <= T::zero() {
                return Err(DataPipelineError::InvalidFormat {
                    message: "Box-Cox transformation requires positive values".to_string(),
                });
            }
        }
        
        let transformed_values: Vec<T> = values.iter()
            .map(|&x| Self::box_cox(x, lambda))
            .collect();
        
        let mut transformed = data.clone();
        for (point, &new_value) in transformed.data_points.iter_mut().zip(transformed_values.iter()) {
            point.value = new_value;
        }
        
        Ok(transformed)
    }
}

impl<T: Float> FittableTransform<T> for BoxCoxTransformer<T> {
    fn fit(&mut self, data: &TimeSeriesData<T>) -> Result<()> {
        if self.lambda.is_some() {
            // Use fixed lambda
            self.fitted_lambda = self.lambda;
        } else {
            // Find optimal lambda
            let values = data.values();
            let optimal_lambda = self.find_optimal_lambda(&values)?;
            self.fitted_lambda = Some(optimal_lambda);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeSeriesDatasetBuilder;
    use chrono::TimeZone;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_standard_scaler() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut scaler = StandardScaler::new();
        
        scaler.fit(&data).unwrap();
        let scaled = scaler.transform(&data).unwrap();
        let unscaled = scaler.inverse_transform(&scaled).unwrap();
        
        // Check mean is approximately 0
        let mean: f64 = scaled.iter().sum::<f64>() / scaled.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
        
        // Check std is approximately 1
        let variance: f64 = scaled.iter().map(|&x| x * x).sum::<f64>() / scaled.len() as f64;
        assert_relative_eq!(variance.sqrt(), 1.0, epsilon = 1e-10);
        
        // Check inverse transformation
        for (original, recovered) in data.iter().zip(unscaled.iter()) {
            assert_relative_eq!(original, recovered, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_minmax_scaler() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut scaler = MinMaxScaler::new();
        
        scaler.fit(&data).unwrap();
        let scaled = scaler.transform(&data).unwrap();
        let unscaled = scaler.inverse_transform(&scaled).unwrap();
        
        // Check range is [0, 1]
        assert_relative_eq!(scaled[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(scaled[4], 1.0, epsilon = 1e-10);
        
        // Check inverse transformation
        for (original, recovered) in data.iter().zip(unscaled.iter()) {
            assert_relative_eq!(original, recovered, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_robust_scaler() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // with outlier
        let mut scaler = RobustScaler::new();
        
        scaler.fit(&data).unwrap();
        let scaled = scaler.transform(&data).unwrap();
        let unscaled = scaler.inverse_transform(&scaled).unwrap();
        
        // Median should be 0
        let median = 3.0; // Original median
        let scaled_median_idx = 2; // Index of median in scaled data
        assert_relative_eq!(scaled[scaled_median_idx], 0.0, epsilon = 1e-10);
        
        // Check inverse transformation
        for (original, recovered) in data.iter().zip(unscaled.iter()) {
            assert_relative_eq!(original, recovered, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_differencing_transformer() {
        let timestamps = vec![
            chrono::Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 4).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 5).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 3.0, 6.0, 10.0, 15.0]; // Differences: [2, 3, 4, 5]
        
        let series = TimeSeriesDatasetBuilder::new("test".to_string())
            .with_frequency("D".to_string())
            .with_values(values)
            .with_timestamps(timestamps)
            .build()
            .unwrap();
        
        let mut transformer = DifferencingTransformer::new(1);
        let transformed = transformer.fit_transform(&series).unwrap();
        
        let expected_diffs = vec![2.0, 3.0, 4.0, 5.0];
        let actual_diffs = transformed.values();
        
        assert_eq!(actual_diffs.len(), expected_diffs.len());
        for (expected, actual) in expected_diffs.iter().zip(actual_diffs.iter()) {
            assert_relative_eq!(expected, actual, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_log_transformer() {
        let timestamps = vec![
            chrono::Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        ];
        let values = vec![1.0, std::f64::consts::E, std::f64::consts::E * std::f64::consts::E];
        
        let series = TimeSeriesDatasetBuilder::new("test".to_string())
            .with_frequency("D".to_string())
            .with_values(values.clone())
            .with_timestamps(timestamps)
            .build()
            .unwrap();
        
        let transformer = LogTransformer::new();
        let transformed = transformer.transform(&series).unwrap();
        
        let log_values = transformed.values();
        assert_relative_eq!(log_values[0], 0.0, epsilon = 1e-10); // ln(1) = 0
        assert_relative_eq!(log_values[1], 1.0, epsilon = 1e-10); // ln(e) = 1
        assert_relative_eq!(log_values[2], 2.0, epsilon = 1e-10); // ln(e^2) = 2
        
        // Test inverse transformation
        let recovered_values = transformer.inverse_transform_values(&log_values).unwrap();
        for (original, recovered) in values.iter().zip(recovered_values.iter()) {
            assert_relative_eq!(original, recovered, epsilon = 1e-10);
        }
    }
}