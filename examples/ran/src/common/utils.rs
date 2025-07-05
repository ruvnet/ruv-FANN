//! Utility functions for the RAN Intelligence Platform

use crate::{Result, RanError};
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::HashMap;

/// Utility functions for data processing
pub struct DataUtils;

impl DataUtils {
    /// Normalize a value to 0-1 range
    pub fn normalize(value: f64, min: f64, max: f64) -> f64 {
        if max <= min {
            return 0.0;
        }
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }
    
    /// Denormalize a value from 0-1 range
    pub fn denormalize(normalized: f64, min: f64, max: f64) -> f64 {
        min + normalized * (max - min)
    }
    
    /// Calculate moving average
    pub fn moving_average(values: &[f64], window_size: usize) -> Vec<f64> {
        if values.is_empty() || window_size == 0 {
            return Vec::new();
        }
        
        let mut result = Vec::new();
        for i in 0..values.len() {
            let start = if i >= window_size - 1 { i - window_size + 1 } else { 0 };
            let window = &values[start..=i];
            let avg = window.iter().sum::<f64>() / window.len() as f64;
            result.push(avg);
        }
        result
    }
    
    /// Calculate standard deviation
    pub fn std_deviation(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
    
    /// Convert JSON value to f64
    pub fn json_to_f64(value: &Value) -> Result<f64> {
        match value {
            Value::Number(n) => n.as_f64().ok_or_else(|| 
                RanError::DataError("Invalid number format".to_string())),
            Value::String(s) => s.parse::<f64>().map_err(|e| 
                RanError::DataError(format!("Failed to parse number: {}", e))),
            _ => Err(RanError::DataError("Value is not a number".to_string())),
        }
    }
    
    /// Generate UUID string
    pub fn generate_uuid() -> String {
        uuid::Uuid::new_v4().to_string()
    }
    
    /// Current timestamp
    pub fn now() -> DateTime<Utc> {
        Utc::now()
    }
}

/// Time series utilities
pub struct TimeSeriesUtils;

impl TimeSeriesUtils {
    /// Calculate trend direction from time series data
    pub fn calculate_trend(values: &[f64]) -> TrendDirection {
        if values.len() < 2 {
            return TrendDirection::Unknown;
        }
        
        let mut increasing = 0;
        let mut decreasing = 0;
        
        for i in 1..values.len() {
            if values[i] > values[i-1] {
                increasing += 1;
            } else if values[i] < values[i-1] {
                decreasing += 1;
            }
        }
        
        let total_changes = increasing + decreasing;
        if total_changes == 0 {
            return TrendDirection::Stable;
        }
        
        let increasing_ratio = increasing as f64 / total_changes as f64;
        
        match increasing_ratio {
            r if r > 0.7 => TrendDirection::Increasing,
            r if r < 0.3 => TrendDirection::Decreasing,
            _ => TrendDirection::Stable,
        }
    }
    
    /// Detect anomalies using simple threshold method
    pub fn detect_anomalies(values: &[f64], threshold_multiplier: f64) -> Vec<usize> {
        if values.is_empty() {
            return Vec::new();
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev = DataUtils::std_deviation(values);
        let threshold = std_dev * threshold_multiplier;
        
        values.iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                if (value - mean).abs() > threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Calculate correlation between two time series
    pub fn correlation(x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() || x.is_empty() {
            return Err(RanError::DataError("Time series must have same length".to_string()));
        }
        
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        
        let numerator: f64 = x.iter().zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();
        
        let sum_sq_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Unknown,
}

/// Configuration validation utilities
pub struct ValidationUtils;

impl ValidationUtils {
    /// Validate signal strength values
    pub fn validate_rsrp(rsrp: f64) -> Result<()> {
        if rsrp < -140.0 || rsrp > -44.0 {
            return Err(RanError::DataError(format!("RSRP {} out of valid range [-140, -44] dBm", rsrp)));
        }
        Ok(())
    }
    
    /// Validate SINR values
    pub fn validate_sinr(sinr: f64) -> Result<()> {
        if sinr < -20.0 || sinr > 40.0 {
            return Err(RanError::DataError(format!("SINR {} out of valid range [-20, 40] dB", sinr)));
        }
        Ok(())
    }
    
    /// Validate success rate
    pub fn validate_success_rate(rate: f64) -> Result<()> {
        if rate < 0.0 || rate > 1.0 {
            return Err(RanError::DataError(format!("Success rate {} out of valid range [0, 1]", rate)));
        }
        Ok(())
    }
    
    /// Validate probability
    pub fn validate_probability(prob: f64) -> Result<()> {
        if prob < 0.0 || prob > 1.0 {
            return Err(RanError::DataError(format!("Probability {} out of valid range [0, 1]", prob)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_normalize() {
        assert_eq!(DataUtils::normalize(5.0, 0.0, 10.0), 0.5);
        assert_eq!(DataUtils::normalize(0.0, 0.0, 10.0), 0.0);
        assert_eq!(DataUtils::normalize(10.0, 0.0, 10.0), 1.0);
        assert_eq!(DataUtils::normalize(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(DataUtils::normalize(15.0, 0.0, 10.0), 1.0);
    }
    
    #[test]
    fn test_moving_average() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = DataUtils::moving_average(&values, 3);
        assert_eq!(ma, vec![1.0, 1.5, 2.0, 3.0, 4.0]);
    }
    
    #[test]
    fn test_trend_calculation() {
        let increasing = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(TimeSeriesUtils::calculate_trend(&increasing), TrendDirection::Increasing);
        
        let decreasing = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(TimeSeriesUtils::calculate_trend(&decreasing), TrendDirection::Decreasing);
        
        let stable = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        assert_eq!(TimeSeriesUtils::calculate_trend(&stable), TrendDirection::Stable);
    }
}