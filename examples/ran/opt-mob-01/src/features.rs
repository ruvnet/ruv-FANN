//! Feature extraction and preprocessing for handover prediction
//!
//! This module implements comprehensive feature engineering for UE metrics,
//! including time-series features, statistical aggregations, and domain-specific
//! handover predictors.

use crate::data::{UeMetrics, NeighborCell};
use crate::{OptMobError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Feature vector for handover prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub features: Vec<f64>,
    pub feature_names: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub ue_id: String,
}

/// Time-series features extracted from UE metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesFeatures {
    // Current measurements
    pub current_rsrp: f64,
    pub current_sinr: f64,
    pub current_rsrq: f64,
    pub current_speed: f64,
    pub current_neighbor_rsrp: f64,
    
    // Lag features (t-1, t-2, ...)
    pub rsrp_lag_1: f64,
    pub rsrp_lag_2: f64,
    pub rsrp_lag_3: f64,
    pub sinr_lag_1: f64,
    pub sinr_lag_2: f64,
    pub sinr_lag_3: f64,
    
    // Rolling window statistics (5-sample window)
    pub rsrp_mean_5: f64,
    pub rsrp_std_5: f64,
    pub rsrp_min_5: f64,
    pub rsrp_max_5: f64,
    pub rsrp_trend_5: f64,        // Linear trend slope
    
    pub sinr_mean_5: f64,
    pub sinr_std_5: f64,
    pub sinr_min_5: f64,
    pub sinr_max_5: f64,
    pub sinr_trend_5: f64,
    
    // Rolling window statistics (10-sample window)
    pub rsrp_mean_10: f64,
    pub rsrp_std_10: f64,
    pub rsrp_trend_10: f64,
    
    pub sinr_mean_10: f64,
    pub sinr_std_10: f64,
    pub sinr_trend_10: f64,
    
    // Velocity and acceleration features
    pub rsrp_velocity: f64,       // Rate of change
    pub rsrp_acceleration: f64,   // Second derivative
    pub sinr_velocity: f64,
    pub sinr_acceleration: f64,
    
    // Neighbor-related features
    pub neighbor_rsrp_mean: f64,
    pub neighbor_rsrp_max: f64,
    pub best_neighbor_rsrp: f64,
    pub rsrp_delta: f64,          // Best neighbor - serving
    pub sinr_delta: f64,          // Best neighbor - serving
    pub neighbor_count: f64,
    
    // Handover trigger features
    pub a3_margin: f64,           // How close to A3 threshold
    pub time_below_threshold: f64, // Time serving cell below threshold
    pub handover_urgency: f64,    // Composite urgency score
    
    // Mobility features
    pub speed_category: f64,      // 0=stationary, 1=walking, 2=driving, 3=high-speed
    pub speed_trend: f64,         // Speed trend
    pub direction_stability: f64, // How stable is movement direction
    
    // Time-based features
    pub hour_of_day: f64,
    pub day_of_week: f64,
    pub is_weekend: f64,
    pub is_busy_hour: f64,
    
    // Cell load features
    pub serving_load: f64,
    pub neighbor_load_min: f64,
    pub load_balance_factor: f64,
    
    // Technology and band features
    pub is_5g: f64,
    pub is_lte: f64,
    pub frequency_band_encoded: f64,
    
    // Historical handover features
    pub handover_history_count: f64,
    pub time_since_last_handover: f64,
    pub handover_success_rate: f64,
}

/// Feature extractor for handover prediction
pub struct FeatureExtractor {
    window_size: usize,
    metrics_buffer: VecDeque<UeMetrics>,
    feature_names: Vec<String>,
}

impl FeatureExtractor {
    /// Create a new feature extractor with specified window size
    pub fn new(window_size: usize) -> Self {
        let feature_names = Self::generate_feature_names();
        Self {
            window_size,
            metrics_buffer: VecDeque::with_capacity(window_size),
            feature_names,
        }
    }
    
    /// Add new UE metrics to the buffer
    pub fn add_metrics(&mut self, metrics: UeMetrics) {
        if self.metrics_buffer.len() >= self.window_size {
            self.metrics_buffer.pop_front();
        }
        self.metrics_buffer.push_back(metrics);
    }
    
    /// Extract features from current buffer
    pub fn extract_features(&self) -> Result<FeatureVector> {
        if self.metrics_buffer.is_empty() {
            return Err(OptMobError::Features("No metrics in buffer".to_string()));
        }
        
        let latest_metrics = self.metrics_buffer.back().unwrap();
        let features = self.extract_time_series_features()?;
        
        Ok(FeatureVector {
            features: features.to_vec(),
            feature_names: self.feature_names.clone(),
            timestamp: latest_metrics.timestamp,
            ue_id: latest_metrics.ue_id.clone(),
        })
    }
    
    /// Extract time-series features from metrics buffer
    fn extract_time_series_features(&self) -> Result<Vec<f64>> {
        let current = self.metrics_buffer.back().unwrap();
        let mut features = Vec::new();
        
        // Current measurements
        features.push(current.serving_rsrp);
        features.push(current.serving_sinr);
        features.push(current.serving_rsrq);
        features.push(current.ue_speed_kmh);
        features.push(current.neighbor_rsrp_best);
        
        // Lag features
        features.extend(self.extract_lag_features());
        
        // Rolling window statistics
        features.extend(self.extract_rolling_statistics());
        
        // Velocity and acceleration
        features.extend(self.extract_velocity_acceleration());
        
        // Neighbor features
        features.extend(self.extract_neighbor_features());
        
        // Handover trigger features
        features.extend(self.extract_handover_trigger_features());
        
        // Mobility features
        features.extend(self.extract_mobility_features());
        
        // Time-based features
        features.extend(self.extract_time_features());
        
        // Cell and technology features
        features.extend(self.extract_cell_tech_features());
        
        Ok(features)
    }
    
    /// Extract lag features (t-1, t-2, t-3)
    fn extract_lag_features(&self) -> Vec<f64> {
        let mut features = Vec::new();
        let buffer_len = self.metrics_buffer.len();
        
        // RSRP lags
        features.push(self.get_lag_value(1, |m| m.serving_rsrp));
        features.push(self.get_lag_value(2, |m| m.serving_rsrp));
        features.push(self.get_lag_value(3, |m| m.serving_rsrp));
        
        // SINR lags
        features.push(self.get_lag_value(1, |m| m.serving_sinr));
        features.push(self.get_lag_value(2, |m| m.serving_sinr));
        features.push(self.get_lag_value(3, |m| m.serving_sinr));
        
        features
    }
    
    /// Get lag value for a specific offset
    fn get_lag_value<F>(&self, lag: usize, extractor: F) -> f64
    where
        F: Fn(&UeMetrics) -> f64,
    {
        if lag >= self.metrics_buffer.len() {
            return 0.0; // Default value for missing lags
        }
        
        let index = self.metrics_buffer.len() - 1 - lag;
        extractor(&self.metrics_buffer[index])
    }
    
    /// Extract rolling window statistics
    fn extract_rolling_statistics(&self) -> Vec<f64> {
        let mut features = Vec::new();
        
        // 5-sample window statistics
        let rsrp_5 = self.get_windowed_values(5, |m| m.serving_rsrp);
        features.extend(self.calculate_window_stats(&rsrp_5));
        
        let sinr_5 = self.get_windowed_values(5, |m| m.serving_sinr);
        features.extend(self.calculate_window_stats(&sinr_5));
        
        // 10-sample window statistics
        let rsrp_10 = self.get_windowed_values(10, |m| m.serving_rsrp);
        features.extend(self.calculate_window_stats(&rsrp_10));
        
        let sinr_10 = self.get_windowed_values(10, |m| m.serving_sinr);
        features.extend(self.calculate_window_stats(&sinr_10));
        
        features
    }
    
    /// Get windowed values for a specific metric
    fn get_windowed_values<F>(&self, window: usize, extractor: F) -> Vec<f64>
    where
        F: Fn(&UeMetrics) -> f64,
    {
        let start = if window >= self.metrics_buffer.len() {
            0
        } else {
            self.metrics_buffer.len() - window
        };
        
        (start..self.metrics_buffer.len())
            .map(|i| extractor(&self.metrics_buffer[i]))
            .collect()
    }
    
    /// Calculate statistics for a window of values
    fn calculate_window_stats(&self, values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return vec![0.0, 0.0, 0.0, 0.0, 0.0]; // mean, std, min, max, trend
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Linear trend (slope)
        let trend = if values.len() > 1 {
            let n = values.len() as f64;
            let x_mean = (n - 1.0) / 2.0;
            let numerator: f64 = values.iter().enumerate()
                .map(|(i, &y)| (i as f64 - x_mean) * (y - mean))
                .sum();
            let denominator: f64 = (0..values.len())
                .map(|i| (i as f64 - x_mean).powi(2))
                .sum();
            
            if denominator != 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        vec![mean, std, min, max, trend]
    }
    
    /// Extract velocity and acceleration features
    fn extract_velocity_acceleration(&self) -> Vec<f64> {
        let mut features = Vec::new();
        
        // RSRP velocity (rate of change)
        let rsrp_velocity = if self.metrics_buffer.len() >= 2 {
            let current = self.metrics_buffer.back().unwrap().serving_rsrp;
            let previous = self.metrics_buffer[self.metrics_buffer.len() - 2].serving_rsrp;
            current - previous
        } else {
            0.0
        };
        
        // RSRP acceleration (second derivative)
        let rsrp_acceleration = if self.metrics_buffer.len() >= 3 {
            let current = self.metrics_buffer.back().unwrap().serving_rsrp;
            let prev1 = self.metrics_buffer[self.metrics_buffer.len() - 2].serving_rsrp;
            let prev2 = self.metrics_buffer[self.metrics_buffer.len() - 3].serving_rsrp;
            (current - prev1) - (prev1 - prev2)
        } else {
            0.0
        };
        
        // Similar for SINR
        let sinr_velocity = if self.metrics_buffer.len() >= 2 {
            let current = self.metrics_buffer.back().unwrap().serving_sinr;
            let previous = self.metrics_buffer[self.metrics_buffer.len() - 2].serving_sinr;
            current - previous
        } else {
            0.0
        };
        
        let sinr_acceleration = if self.metrics_buffer.len() >= 3 {
            let current = self.metrics_buffer.back().unwrap().serving_sinr;
            let prev1 = self.metrics_buffer[self.metrics_buffer.len() - 2].serving_sinr;
            let prev2 = self.metrics_buffer[self.metrics_buffer.len() - 3].serving_sinr;
            (current - prev1) - (prev1 - prev2)
        } else {
            0.0
        };
        
        features.push(rsrp_velocity);
        features.push(rsrp_acceleration);
        features.push(sinr_velocity);
        features.push(sinr_acceleration);
        
        features
    }
    
    /// Extract neighbor cell features
    fn extract_neighbor_features(&self) -> Vec<f64> {
        let current = self.metrics_buffer.back().unwrap();
        let mut features = Vec::new();
        
        // Neighbor statistics
        if !current.neighbor_cells.is_empty() {
            let neighbor_rsrps: Vec<f64> = current.neighbor_cells.iter()
                .map(|n| n.rsrp)
                .collect();
            
            let neighbor_mean = neighbor_rsrps.iter().sum::<f64>() / neighbor_rsrps.len() as f64;
            let neighbor_max = neighbor_rsrps.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let best_neighbor = neighbor_max;
            
            features.push(neighbor_mean);
            features.push(neighbor_max);
            features.push(best_neighbor);
        } else {
            features.push(current.neighbor_rsrp_best);
            features.push(current.neighbor_rsrp_best);
            features.push(current.neighbor_rsrp_best);
        }
        
        // Deltas
        features.push(current.neighbor_rsrp_best - current.serving_rsrp);
        features.push(current.neighbor_sinr_best - current.serving_sinr);
        features.push(current.neighbor_cells.len() as f64);
        
        features
    }
    
    /// Extract handover trigger features
    fn extract_handover_trigger_features(&self) -> Vec<f64> {
        let current = self.metrics_buffer.back().unwrap();
        let mut features = Vec::new();
        
        // A3 margin (how close to A3 threshold)
        let a3_offset = 3.0; // dB
        let a3_hysteresis = 0.5; // dB
        let a3_margin = (current.neighbor_rsrp_best - current.serving_rsrp) - a3_offset - a3_hysteresis;
        features.push(a3_margin);
        
        // Time below threshold (simplified)
        let rsrp_threshold = -100.0; // dBm
        let time_below = if current.serving_rsrp < rsrp_threshold {
            1.0
        } else {
            0.0
        };
        features.push(time_below);
        
        // Handover urgency (composite score)
        let urgency = self.calculate_handover_urgency(current);
        features.push(urgency);
        
        features
    }
    
    /// Calculate handover urgency score
    fn calculate_handover_urgency(&self, metrics: &UeMetrics) -> f64 {
        let mut urgency = 0.0;
        
        // Poor serving cell quality
        if metrics.serving_rsrp < -110.0 {
            urgency += 0.3;
        }
        if metrics.serving_sinr < 0.0 {
            urgency += 0.2;
        }
        
        // Strong neighbor
        if metrics.neighbor_rsrp_best > metrics.serving_rsrp + 6.0 {
            urgency += 0.3;
        }
        
        // High mobility
        if metrics.ue_speed_kmh > 60.0 {
            urgency += 0.2;
        }
        
        urgency.min(1.0)
    }
    
    /// Extract mobility-related features
    fn extract_mobility_features(&self) -> Vec<f64> {
        let current = self.metrics_buffer.back().unwrap();
        let mut features = Vec::new();
        
        // Speed category
        let speed_category = if current.ue_speed_kmh < 5.0 {
            0.0 // Stationary
        } else if current.ue_speed_kmh < 30.0 {
            1.0 // Walking/slow
        } else if current.ue_speed_kmh < 80.0 {
            2.0 // Driving
        } else {
            3.0 // High-speed
        };
        features.push(speed_category);
        
        // Speed trend
        let speed_trend = if self.metrics_buffer.len() >= 2 {
            let current_speed = current.ue_speed_kmh;
            let previous_speed = self.metrics_buffer[self.metrics_buffer.len() - 2].ue_speed_kmh;
            (current_speed - previous_speed) / previous_speed.max(1.0)
        } else {
            0.0
        };
        features.push(speed_trend);
        
        // Direction stability (simplified)
        let direction_stability = if current.ue_bearing_degrees.is_some() {
            0.8 // Assume stable for now
        } else {
            0.5 // Unknown
        };
        features.push(direction_stability);
        
        features
    }
    
    /// Extract time-based features
    fn extract_time_features(&self) -> Vec<f64> {
        let current = self.metrics_buffer.back().unwrap();
        let mut features = Vec::new();
        
        let hour = current.timestamp.hour() as f64;
        let day_of_week = current.timestamp.weekday().num_days_from_monday() as f64;
        let is_weekend = if day_of_week >= 5.0 { 1.0 } else { 0.0 };
        let is_busy_hour = if hour >= 8.0 && hour <= 10.0 || hour >= 17.0 && hour <= 19.0 {
            1.0
        } else {
            0.0
        };
        
        features.push(hour / 24.0); // Normalize to [0, 1]
        features.push(day_of_week / 7.0); // Normalize to [0, 1]
        features.push(is_weekend);
        features.push(is_busy_hour);
        
        features
    }
    
    /// Extract cell and technology features
    fn extract_cell_tech_features(&self) -> Vec<f64> {
        let current = self.metrics_buffer.back().unwrap();
        let mut features = Vec::new();
        
        // Cell load
        let serving_load = current.serving_prb_usage.unwrap_or(0.5);
        features.push(serving_load);
        
        // Neighbor load (simplified)
        let neighbor_load_min = current.neighbor_cells.iter()
            .filter_map(|n| n.cell_load_percent)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.5);
        features.push(neighbor_load_min);
        
        // Load balance factor
        let load_balance = serving_load - neighbor_load_min;
        features.push(load_balance);
        
        // Technology flags
        let is_5g = if current.technology.contains("5G") { 1.0 } else { 0.0 };
        let is_lte = if current.technology.contains("LTE") { 1.0 } else { 0.0 };
        features.push(is_5g);
        features.push(is_lte);
        
        // Frequency band (simplified encoding)
        let band_num = current.frequency_band.chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<f64>()
            .unwrap_or(1.0);
        features.push(band_num / 100.0); // Normalize
        
        // Historical features (simplified)
        features.push(0.0); // handover_history_count
        features.push(0.0); // time_since_last_handover
        features.push(0.95); // handover_success_rate
        
        features
    }
    
    /// Generate feature names
    fn generate_feature_names() -> Vec<String> {
        vec![
            // Current measurements
            "serving_rsrp".to_string(),
            "serving_sinr".to_string(),
            "serving_rsrq".to_string(),
            "ue_speed_kmh".to_string(),
            "neighbor_rsrp_best".to_string(),
            
            // Lag features
            "rsrp_lag_1".to_string(),
            "rsrp_lag_2".to_string(),
            "rsrp_lag_3".to_string(),
            "sinr_lag_1".to_string(),
            "sinr_lag_2".to_string(),
            "sinr_lag_3".to_string(),
            
            // 5-sample window RSRP
            "rsrp_mean_5".to_string(),
            "rsrp_std_5".to_string(),
            "rsrp_min_5".to_string(),
            "rsrp_max_5".to_string(),
            "rsrp_trend_5".to_string(),
            
            // 5-sample window SINR
            "sinr_mean_5".to_string(),
            "sinr_std_5".to_string(),
            "sinr_min_5".to_string(),
            "sinr_max_5".to_string(),
            "sinr_trend_5".to_string(),
            
            // 10-sample window RSRP
            "rsrp_mean_10".to_string(),
            "rsrp_std_10".to_string(),
            "rsrp_min_10".to_string(),
            "rsrp_max_10".to_string(),
            "rsrp_trend_10".to_string(),
            
            // 10-sample window SINR
            "sinr_mean_10".to_string(),
            "sinr_std_10".to_string(),
            "sinr_min_10".to_string(),
            "sinr_max_10".to_string(),
            "sinr_trend_10".to_string(),
            
            // Velocity and acceleration
            "rsrp_velocity".to_string(),
            "rsrp_acceleration".to_string(),
            "sinr_velocity".to_string(),
            "sinr_acceleration".to_string(),
            
            // Neighbor features
            "neighbor_rsrp_mean".to_string(),
            "neighbor_rsrp_max".to_string(),
            "best_neighbor_rsrp".to_string(),
            "rsrp_delta".to_string(),
            "sinr_delta".to_string(),
            "neighbor_count".to_string(),
            
            // Handover trigger features
            "a3_margin".to_string(),
            "time_below_threshold".to_string(),
            "handover_urgency".to_string(),
            
            // Mobility features
            "speed_category".to_string(),
            "speed_trend".to_string(),
            "direction_stability".to_string(),
            
            // Time features
            "hour_of_day".to_string(),
            "day_of_week".to_string(),
            "is_weekend".to_string(),
            "is_busy_hour".to_string(),
            
            // Cell and technology features
            "serving_load".to_string(),
            "neighbor_load_min".to_string(),
            "load_balance_factor".to_string(),
            "is_5g".to_string(),
            "is_lte".to_string(),
            "frequency_band_encoded".to_string(),
            
            // Historical features
            "handover_history_count".to_string(),
            "time_since_last_handover".to_string(),
            "handover_success_rate".to_string(),
        ]
    }
    
    /// Get feature names
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
    
    /// Get feature count
    pub fn feature_count(&self) -> usize {
        self.feature_names.len()
    }
    
    /// Reset the feature extractor
    pub fn reset(&mut self) {
        self.metrics_buffer.clear();
    }
    
    /// Check if buffer is ready for feature extraction
    pub fn is_ready(&self) -> bool {
        !self.metrics_buffer.is_empty()
    }
    
    /// Get buffer length
    pub fn buffer_length(&self) -> usize {
        self.metrics_buffer.len()
    }
}

impl FeatureVector {
    /// Convert to ndarray Array1
    pub fn to_array(&self) -> Array1<f64> {
        Array1::from_vec(self.features.clone())
    }
    
    /// Convert to ndarray Array2 (single sample)
    pub fn to_matrix(&self) -> Array2<f64> {
        let mut matrix = Array2::zeros((1, self.features.len()));
        for (i, &value) in self.features.iter().enumerate() {
            matrix[[0, i]] = value;
        }
        matrix
    }
    
    /// Normalize features using z-score normalization
    pub fn normalize(&mut self, means: &[f64], stds: &[f64]) -> Result<()> {
        if self.features.len() != means.len() || self.features.len() != stds.len() {
            return Err(OptMobError::Features(
                "Feature count mismatch for normalization".to_string()
            ));
        }
        
        for (i, feature) in self.features.iter_mut().enumerate() {
            if stds[i] != 0.0 {
                *feature = (*feature - means[i]) / stds[i];
            }
        }
        
        Ok(())
    }
    
    /// Validate feature vector
    pub fn validate(&self) -> Result<()> {
        if self.features.is_empty() {
            return Err(OptMobError::Features("Empty feature vector".to_string()));
        }
        
        if self.features.len() != self.feature_names.len() {
            return Err(OptMobError::Features(
                "Feature count mismatch with feature names".to_string()
            ));
        }
        
        // Check for NaN or infinite values
        for (i, &feature) in self.features.iter().enumerate() {
            if !feature.is_finite() {
                return Err(OptMobError::Features(
                    format!("Invalid feature value at index {}: {}", i, feature)
                ));
            }
        }
        
        Ok(())
    }
}

/// Feature normalization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub feature_names: Vec<String>,
}

impl NormalizationParams {
    /// Calculate normalization parameters from feature vectors
    pub fn from_features(features: &[FeatureVector]) -> Result<Self> {
        if features.is_empty() {
            return Err(OptMobError::Features("No features provided".to_string()));
        }
        
        let feature_count = features[0].features.len();
        let mut means = vec![0.0; feature_count];
        let mut stds = vec![0.0; feature_count];
        
        // Calculate means
        for feature_vec in features {
            for (i, &value) in feature_vec.features.iter().enumerate() {
                means[i] += value;
            }
        }
        
        for mean in means.iter_mut() {
            *mean /= features.len() as f64;
        }
        
        // Calculate standard deviations
        for feature_vec in features {
            for (i, &value) in feature_vec.features.iter().enumerate() {
                stds[i] += (value - means[i]).powi(2);
            }
        }
        
        for std in stds.iter_mut() {
            *std = (*std / features.len() as f64).sqrt();
            if *std == 0.0 {
                *std = 1.0; // Avoid division by zero
            }
        }
        
        Ok(Self {
            means,
            stds,
            feature_names: features[0].feature_names.clone(),
        })
    }
    
    /// Apply normalization to a feature vector
    pub fn normalize(&self, features: &mut FeatureVector) -> Result<()> {
        features.normalize(&self.means, &self.stds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::UeMetrics;
    
    #[test]
    fn test_feature_extractor_creation() {
        let extractor = FeatureExtractor::new(10);
        assert_eq!(extractor.window_size, 10);
        assert_eq!(extractor.feature_count(), 57); // Expected number of features
    }
    
    #[test]
    fn test_feature_extraction() {
        let mut extractor = FeatureExtractor::new(5);
        
        // Add some sample metrics
        for i in 0..3 {
            let metrics = UeMetrics::new(&format!("UE_{}", i), &format!("Cell_{}", i))
                .with_rsrp(-90.0 - i as f64)
                .with_sinr(10.0 + i as f64)
                .with_speed(30.0 + i as f64 * 10.0);
            
            extractor.add_metrics(metrics);
        }
        
        let features = extractor.extract_features().unwrap();
        assert_eq!(features.features.len(), extractor.feature_count());
        assert!(features.validate().is_ok());
    }
    
    #[test]
    fn test_normalization_params() {
        let feature_vec = FeatureVector {
            features: vec![1.0, 2.0, 3.0],
            feature_names: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
            timestamp: chrono::Utc::now(),
            ue_id: "test".to_string(),
        };
        
        let params = NormalizationParams::from_features(&[feature_vec]).unwrap();
        assert_eq!(params.means.len(), 3);
        assert_eq!(params.stds.len(), 3);
    }
}