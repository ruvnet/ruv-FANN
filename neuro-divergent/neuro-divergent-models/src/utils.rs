//! Utility functions for neural forecasting models
//!
//! This module provides common utility functions used across different models,
//! including mathematical operations, data preprocessing, and validation helpers.

use num_traits::Float;
use std::collections::HashMap;
use crate::errors::{NeuroDivergentError, NeuroDivergentResult};

/// Mathematical utility functions
pub mod math {
    use super::*;
    
    /// Calculate mean squared error
    pub fn mse<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(actual: &[T], predicted: &[T]) -> NeuroDivergentResult<T> {
        if actual.len() != predicted.len() {
            return Err(NeuroDivergentError::dimension_mismatch(actual.len(), predicted.len()));
        }
        
        if actual.is_empty() {
            return Err(NeuroDivergentError::data("Cannot score empty forecasts"));
        }
        validation::validate_finite(actual, "actual")?;
        validation::validate_finite(predicted, "predicted")?;

        let sum = actual.iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| {
                let diff = a - p;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);
            
        Ok(sum / T::from(actual.len()).unwrap())
    }
    
    /// Calculate mean absolute error
    pub fn mae<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(actual: &[T], predicted: &[T]) -> NeuroDivergentResult<T> {
        if actual.len() != predicted.len() {
            return Err(NeuroDivergentError::dimension_mismatch(actual.len(), predicted.len()));
        }
        
        if actual.is_empty() {
            return Err(NeuroDivergentError::data("Cannot score empty forecasts"));
        }
        validation::validate_finite(actual, "actual")?;
        validation::validate_finite(predicted, "predicted")?;

        let sum = actual.iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| (a - p).abs())
            .fold(T::zero(), |acc, x| acc + x);
            
        Ok(sum / T::from(actual.len()).unwrap())
    }
    
    /// Calculate root mean squared error
    pub fn rmse<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(actual: &[T], predicted: &[T]) -> NeuroDivergentResult<T> {
        let mse_value = mse(actual, predicted)?;
        Ok(mse_value.sqrt())
    }
    
    /// Apply softmax function
    pub fn softmax<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(inputs: &[T]) -> Vec<T> {
        let Some(&first) = inputs.first() else {
            return Vec::new();
        };
        let max_val = inputs.iter().fold(first, |max, &x| max.max(x));
        let exp_values: Vec<T> = inputs.iter()
            .map(|&x| (x - max_val).exp())
            .collect();
        let sum_exp: T = exp_values.iter().copied().sum();
        
        exp_values.iter()
            .map(|&exp_val| exp_val / sum_exp)
            .collect()
    }
    
    /// Calculate dot product of two vectors
    pub fn dot_product<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(a: &[T], b: &[T]) -> NeuroDivergentResult<T> {
        if a.len() != b.len() {
            return Err(NeuroDivergentError::dimension_mismatch(a.len(), b.len()));
        }
        
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
    }
    
    /// Matrix multiplication for 2D vectors
    pub fn matrix_multiply<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(a: &[Vec<T>], b: &[Vec<T>]) -> NeuroDivergentResult<Vec<Vec<T>>> {
        if a.is_empty() || b.is_empty() {
            return Err(NeuroDivergentError::data("Cannot multiply empty matrices"));
        }
        
        let rows_a = a.len();
        let cols_a = a[0].len();
        let rows_b = b.len();
        let cols_b = b[0].len();
        
        if cols_a != rows_b {
            return Err(NeuroDivergentError::dimension_mismatch(cols_a, rows_b));
        }
        
        if cols_a == 0 || cols_b == 0
            || a.iter().any(|row| row.len() != cols_a)
            || b.iter().any(|row| row.len() != cols_b)
        {
            return Err(NeuroDivergentError::data("Matrices must be nonempty and rectangular"));
        }
        let mut result = vec![vec![T::zero(); cols_b]; rows_a];
        for (a_row, result_row) in a.iter().zip(result.iter_mut()) {
            for (&a_value, b_row) in a_row.iter().zip(b.iter()) {
                for (value, &b_value) in result_row.iter_mut().zip(b_row.iter()) {
                    *value = *value + a_value * b_value;
                }
            }
        }
        
        Ok(result)
    }
}

/// Data preprocessing utilities
pub mod preprocessing {
    use super::*;
    
    /// Min-Max scaler
    #[derive(Debug, Clone)]
    pub struct MinMaxScaler<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
        min_val: Option<T>,
        max_val: Option<T>,
        range: (T, T), // (min_range, max_range)
    }
    
    impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> MinMaxScaler<T> {
        pub fn new() -> Self {
            Self {
                min_val: None,
                max_val: None,
                range: (T::zero(), T::one()),
            }
        }
        
        pub fn with_range(mut self, min_range: T, max_range: T) -> Self {
            self.range = (min_range, max_range);
            self
        }
        
        pub fn fit(&mut self, data: &[T]) -> NeuroDivergentResult<()> {
            if data.is_empty() {
                return Err(NeuroDivergentError::data("Cannot fit scaler on empty data"));
            }
            
            self.min_val = Some(data.iter().fold(data[0], |min, &x| min.min(x)));
            self.max_val = Some(data.iter().fold(data[0], |max, &x| max.max(x)));
            
            Ok(())
        }
        
        pub fn transform(&self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
            let min_val = self.min_val.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            let max_val = self.max_val.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            
            if max_val == min_val {
                return Ok(vec![self.range.0; data.len()]);
            }
            
            let scale = (self.range.1 - self.range.0) / (max_val - min_val);
            let transformed = data.iter()
                .map(|&x| (x - min_val) * scale + self.range.0)
                .collect();
                
            Ok(transformed)
        }
        
        pub fn inverse_transform(&self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
            let min_val = self.min_val.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            let max_val = self.max_val.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            
            if max_val == min_val {
                return Ok(vec![min_val; data.len()]);
            }
            
            let scale = (max_val - min_val) / (self.range.1 - self.range.0);
            let transformed = data.iter()
                .map(|&x| (x - self.range.0) * scale + min_val)
                .collect();
                
            Ok(transformed)
        }
        
        pub fn fit_transform(&mut self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
            self.fit(data)?;
            self.transform(data)
        }
    }
    
    /// Standard scaler (Z-score normalization)
    #[derive(Debug, Clone)]
    pub struct StandardScaler<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
        mean: Option<T>,
        std: Option<T>,
    }
    
    impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> StandardScaler<T> {
        pub fn new() -> Self {
            Self {
                mean: None,
                std: None,
            }
        }
        
        pub fn fit(&mut self, data: &[T]) -> NeuroDivergentResult<()> {
            if data.is_empty() {
                return Err(NeuroDivergentError::data("Cannot fit scaler on empty data"));
            }
            
            let n = T::from(data.len()).unwrap();
            let mean = data.iter().copied().sum::<T>() / n;
            
            let variance = data.iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<T>() / n;
                
            let std = variance.sqrt();
            
            self.mean = Some(mean);
            self.std = Some(std);
            
            Ok(())
        }
        
        pub fn transform(&self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
            let mean = self.mean.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            let std = self.std.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            
            if std == T::zero() {
                return Ok(vec![T::zero(); data.len()]);
            }
            
            let transformed = data.iter()
                .map(|&x| (x - mean) / std)
                .collect();
                
            Ok(transformed)
        }
        
        pub fn inverse_transform(&self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
            let mean = self.mean.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            let std = self.std.ok_or_else(|| 
                NeuroDivergentError::state("Scaler has not been fitted"))?;
            
            let transformed = data.iter()
                .map(|&x| x * std + mean)
                .collect();
                
            Ok(transformed)
        }
        
        pub fn fit_transform(&mut self, data: &[T]) -> NeuroDivergentResult<Vec<T>> {
            self.fit(data)?;
            self.transform(data)
        }
    }
}

/// Validation utilities
pub mod validation {
    use super::*;
    
    /// Validate input dimensions
    pub fn validate_dimensions<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
        input: &[T], 
        expected_size: usize, 
        name: &str
    ) -> NeuroDivergentResult<()> {
        if input.len() != expected_size {
            return Err(NeuroDivergentError::dimension_mismatch(expected_size, input.len()));
        }
        Ok(())
    }
    
    /// Validate that vectors have the same length
    pub fn validate_same_length<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
        a: &[T], 
        b: &[T], 
        name_a: &str, 
        name_b: &str
    ) -> NeuroDivergentResult<()> {
        if a.len() != b.len() {
            return Err(NeuroDivergentError::dimension_mismatch(a.len(), b.len()));
        }
        Ok(())
    }
    
    /// Validate that all values are finite
    pub fn validate_finite<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(data: &[T], name: &str) -> NeuroDivergentResult<()> {
        for (i, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                return Err(NeuroDivergentError::data(
                    format!("Non-finite value at index {} in {}: {:?}", i, name, value)
                ));
            }
        }
        Ok(())
    }
    
    /// Validate that values are in a specific range
    pub fn validate_range<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
        data: &[T], 
        min_val: T, 
        max_val: T, 
        name: &str
    ) -> NeuroDivergentResult<()> {
        for (i, &value) in data.iter().enumerate() {
            if value < min_val || value > max_val {
                return Err(NeuroDivergentError::data(
                    format!("Value at index {} in {} out of range [{:?}, {:?}]: {:?}", 
                           i, name, min_val, max_val, value)
                ));
            }
        }
        Ok(())
    }
}

/// Time series specific utilities
pub mod timeseries {
    use super::*;
    
    /// Create sliding windows from a time series
    pub fn create_sliding_windows<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
        data: &[T], 
        window_size: usize, 
        step_size: usize
    ) -> Vec<Vec<T>> {
        if window_size == 0 || step_size == 0 || window_size > data.len() {
            return Vec::new();
        }
        data.windows(window_size).step_by(step_size).map(<[T]>::to_vec).collect()
    }
    
    /// Split time series into train/validation sets
    pub fn train_validation_split<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
        data: &[T], 
        validation_split: f64
    ) -> NeuroDivergentResult<(Vec<T>, Vec<T>)> {
        if !validation_split.is_finite() || !(0.0..1.0).contains(&validation_split) {
            return Err(NeuroDivergentError::config(
                "Validation split must be in range [0, 1)"
            ));
        }
        
        let split_index = ((1.0 - validation_split) * data.len() as f64) as usize;
        
        Ok((
            data[..split_index].to_vec(),
            data[split_index..].to_vec(),
        ))
    }
    
    /// Calculate seasonal decomposition (simple moving average)
    pub fn simple_seasonal_decompose<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>(
        data: &[T], 
        period: usize
    ) -> NeuroDivergentResult<(Vec<T>, Vec<T>)> {
        if period == 0 || period > data.len() {
            return Err(NeuroDivergentError::config(
                format!("Invalid period {} for data length {}", period, data.len())
            ));
        }
        
        let mut trend = Vec::with_capacity(data.len());
        let mut seasonal = Vec::with_capacity(data.len());
        
        // Simple moving average for trend
        for i in 0..data.len() {
            let start = i.saturating_sub(period / 2);
            let end = (i + period / 2 + 1).min(data.len());
            
            let sum: T = data[start..end].iter().copied().sum();
            let count = T::from(end - start).unwrap();
            let avg = sum / count;
            
            trend.push(avg);
            seasonal.push(data[i] - avg);
        }
        
        Ok((trend, seasonal))
    }
}

#[cfg(test)]
#[path = "utility_tests.rs"]
mod tests;
