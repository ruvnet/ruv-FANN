//! Feature engineering module for the RAN Intelligence Platform

use crate::{Result, RanError};
use crate::types::*;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Feature engineering service
pub struct FeatureEngineeringService {
    config: FeatureConfig,
}

/// Configuration for feature engineering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    pub window_sizes: Vec<u32>,
    pub lag_features: Vec<u32>,
    pub statistical_features: bool,
    pub time_features: bool,
}

impl FeatureEngineeringService {
    pub fn new(config: FeatureConfig) -> Self {
        Self { config }
    }
    
    /// Extract features from time series data
    pub async fn extract_features(&self, data: &[TimeSeries]) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        for ts in data {
            // Add basic statistics
            features.extend(self.calculate_statistics(&ts.values)?);
            
            // Add lag features
            features.extend(self.calculate_lag_features(&ts.values)?);
            
            // Add time-based features
            if self.config.time_features {
                features.extend(self.calculate_time_features(ts.timestamp)?);
            }
        }
        
        Ok(features)
    }
    
    fn calculate_statistics(&self, values: &[f64]) -> Result<Vec<f64>> {
        if values.is_empty() {
            return Ok(vec![0.0; 4]); // mean, std, min, max
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        Ok(vec![mean, std_dev, min, max])
    }
    
    fn calculate_lag_features(&self, values: &[f64]) -> Result<Vec<f64>> {
        let mut lag_features = Vec::new();
        
        for &lag in &self.config.lag_features {
            if values.len() > lag as usize {
                let lag_value = values[values.len() - 1 - lag as usize];
                lag_features.push(lag_value);
            } else {
                lag_features.push(0.0);
            }
        }
        
        Ok(lag_features)
    }
    
    fn calculate_time_features(&self, timestamp: DateTime<Utc>) -> Result<Vec<f64>> {
        let hour = timestamp.hour() as f64 / 24.0;
        let day_of_week = timestamp.weekday().num_days_from_monday() as f64 / 7.0;
        let day_of_month = timestamp.day() as f64 / 31.0;
        
        Ok(vec![hour, day_of_week, day_of_month])
    }
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            window_sizes: vec![5, 10, 30],
            lag_features: vec![1, 2, 5],
            statistical_features: true,
            time_features: true,
        }
    }
}