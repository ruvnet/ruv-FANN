//! Feature extraction module for interference classification
//! 
//! This module extracts features from noise floor measurements and cell parameters
//! to create feature vectors suitable for neural network classification.

use crate::{
    NoiseFloorMeasurement, CellParameters, InterferenceClassifierError, Result,
    FEATURE_VECTOR_SIZE, MINIMUM_MEASUREMENT_WINDOW, MAXIMUM_MEASUREMENT_WINDOW,
};
use std::collections::HashMap;
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Feature extractor for interference classification
pub struct FeatureExtractor {
    feature_names: Vec<String>,
    normalization_stats: HashMap<String, (f64, f64)>, // mean, std
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new() -> Self {
        let feature_names = vec![
            // Raw measurements statistics
            "noise_floor_pusch_mean".to_string(),
            "noise_floor_pusch_std".to_string(),
            "noise_floor_pusch_min".to_string(),
            "noise_floor_pusch_max".to_string(),
            "noise_floor_pucch_mean".to_string(),
            "noise_floor_pucch_std".to_string(),
            "noise_floor_pucch_min".to_string(),
            "noise_floor_pucch_max".to_string(),
            
            // Cell RET statistics
            "cell_ret_mean".to_string(),
            "cell_ret_std".to_string(),
            "cell_ret_trend".to_string(),
            
            // RSRP/SINR statistics
            "rsrp_mean".to_string(),
            "rsrp_std".to_string(),
            "sinr_mean".to_string(),
            "sinr_std".to_string(),
            
            // Load-based features
            "active_users_mean".to_string(),
            "active_users_std".to_string(),
            "prb_utilization_mean".to_string(),
            "prb_utilization_std".to_string(),
            
            // Temporal features
            "measurement_count".to_string(),
            "time_span_hours".to_string(),
            "peak_noise_time".to_string(),
            
            // Correlation features
            "pusch_pucch_correlation".to_string(),
            "noise_users_correlation".to_string(),
            "noise_prb_correlation".to_string(),
            
            // Cell parameter features
            "frequency_band_encoded".to_string(),
            "tx_power_normalized".to_string(),
            "antenna_count_normalized".to_string(),
            "bandwidth_normalized".to_string(),
            "technology_encoded".to_string(),
            
            // Advanced features
            "interference_signature".to_string(),
        ];
        
        Self {
            feature_names,
            normalization_stats: HashMap::new(),
        }
    }
    
    /// Extract features from measurements and cell parameters
    pub fn extract_features(
        &self,
        measurements: &[NoiseFloorMeasurement],
        cell_params: &CellParameters,
    ) -> Result<Vec<f64>> {
        if measurements.len() < MINIMUM_MEASUREMENT_WINDOW {
            return Err(InterferenceClassifierError::InvalidInputError(
                format!("Insufficient measurements: {} < {}", 
                    measurements.len(), MINIMUM_MEASUREMENT_WINDOW)
            ));
        }
        
        if measurements.len() > MAXIMUM_MEASUREMENT_WINDOW {
            return Err(InterferenceClassifierError::InvalidInputError(
                format!("Too many measurements: {} > {}", 
                    measurements.len(), MAXIMUM_MEASUREMENT_WINDOW)
            ));
        }
        
        let mut features = Vec::with_capacity(FEATURE_VECTOR_SIZE);
        
        // Extract raw measurement statistics
        features.extend(self.extract_noise_floor_features(measurements)?);
        features.extend(self.extract_cell_ret_features(measurements)?);
        features.extend(self.extract_rsrp_sinr_features(measurements)?);
        features.extend(self.extract_load_features(measurements)?);
        features.extend(self.extract_temporal_features(measurements)?);
        features.extend(self.extract_correlation_features(measurements)?);
        features.extend(self.extract_cell_parameter_features(cell_params)?);
        features.extend(self.extract_advanced_features(measurements)?);
        
        // Pad or truncate to exact size
        features.resize(FEATURE_VECTOR_SIZE, 0.0);
        
        Ok(features)
    }
    
    /// Extract noise floor related features
    fn extract_noise_floor_features(&self, measurements: &[NoiseFloorMeasurement]) -> Result<Vec<f64>> {
        let pusch_values: Vec<f64> = measurements.iter()
            .map(|m| m.noise_floor_pusch)
            .collect();
        let pucch_values: Vec<f64> = measurements.iter()
            .map(|m| m.noise_floor_pucch)
            .collect();
        
        let mut features = Vec::new();
        
        // PUSCH statistics
        features.push(self.calculate_mean(&pusch_values));
        features.push(self.calculate_std(&pusch_values));
        features.push(self.calculate_min(&pusch_values));
        features.push(self.calculate_max(&pusch_values));
        
        // PUCCH statistics
        features.push(self.calculate_mean(&pucch_values));
        features.push(self.calculate_std(&pucch_values));
        features.push(self.calculate_min(&pucch_values));
        features.push(self.calculate_max(&pucch_values));
        
        Ok(features)
    }
    
    /// Extract cell RET features
    fn extract_cell_ret_features(&self, measurements: &[NoiseFloorMeasurement]) -> Result<Vec<f64>> {
        let ret_values: Vec<f64> = measurements.iter()
            .map(|m| m.cell_ret)
            .collect();
        
        let mut features = Vec::new();
        features.push(self.calculate_mean(&ret_values));
        features.push(self.calculate_std(&ret_values));
        features.push(self.calculate_trend(&ret_values));
        
        Ok(features)
    }
    
    /// Extract RSRP/SINR features
    fn extract_rsrp_sinr_features(&self, measurements: &[NoiseFloorMeasurement]) -> Result<Vec<f64>> {
        let rsrp_values: Vec<f64> = measurements.iter()
            .map(|m| m.rsrp)
            .collect();
        let sinr_values: Vec<f64> = measurements.iter()
            .map(|m| m.sinr)
            .collect();
        
        let mut features = Vec::new();
        features.push(self.calculate_mean(&rsrp_values));
        features.push(self.calculate_std(&rsrp_values));
        features.push(self.calculate_mean(&sinr_values));
        features.push(self.calculate_std(&sinr_values));
        
        Ok(features)
    }
    
    /// Extract load-based features
    fn extract_load_features(&self, measurements: &[NoiseFloorMeasurement]) -> Result<Vec<f64>> {
        let user_values: Vec<f64> = measurements.iter()
            .map(|m| m.active_users as f64)
            .collect();
        let prb_values: Vec<f64> = measurements.iter()
            .map(|m| m.prb_utilization)
            .collect();
        
        let mut features = Vec::new();
        features.push(self.calculate_mean(&user_values));
        features.push(self.calculate_std(&user_values));
        features.push(self.calculate_mean(&prb_values));
        features.push(self.calculate_std(&prb_values));
        
        Ok(features)
    }
    
    /// Extract temporal features
    fn extract_temporal_features(&self, measurements: &[NoiseFloorMeasurement]) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        // Measurement count
        features.push(measurements.len() as f64);
        
        // Time span
        if measurements.len() > 1 {
            let first_time = measurements.first().unwrap().timestamp;
            let last_time = measurements.last().unwrap().timestamp;
            let time_span = (last_time - first_time).num_seconds() as f64 / 3600.0; // hours
            features.push(time_span);
        } else {
            features.push(0.0);
        }
        
        // Peak noise time (hour of day when max noise occurred)
        let max_noise_idx = measurements.iter()
            .enumerate()
            .max_by(|a, b| a.1.noise_floor_pusch.partial_cmp(&b.1.noise_floor_pusch).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let peak_hour = measurements[max_noise_idx].timestamp.hour() as f64;
        features.push(peak_hour);
        
        Ok(features)
    }
    
    /// Extract correlation features
    fn extract_correlation_features(&self, measurements: &[NoiseFloorMeasurement]) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        // PUSCH-PUCCH correlation
        let pusch_values: Vec<f64> = measurements.iter().map(|m| m.noise_floor_pusch).collect();
        let pucch_values: Vec<f64> = measurements.iter().map(|m| m.noise_floor_pucch).collect();
        features.push(self.calculate_correlation(&pusch_values, &pucch_values));
        
        // Noise-Users correlation
        let user_values: Vec<f64> = measurements.iter().map(|m| m.active_users as f64).collect();
        features.push(self.calculate_correlation(&pusch_values, &user_values));
        
        // Noise-PRB correlation
        let prb_values: Vec<f64> = measurements.iter().map(|m| m.prb_utilization).collect();
        features.push(self.calculate_correlation(&pusch_values, &prb_values));
        
        Ok(features)
    }
    
    /// Extract cell parameter features
    fn extract_cell_parameter_features(&self, cell_params: &CellParameters) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        // Frequency band encoding (hash-based)
        let freq_encoded = self.encode_frequency_band(&cell_params.frequency_band);
        features.push(freq_encoded);
        
        // Normalized TX power
        features.push(cell_params.tx_power / 60.0); // Normalize to typical max power
        
        // Normalized antenna count
        features.push(cell_params.antenna_count as f64 / 8.0); // Normalize to typical max
        
        // Normalized bandwidth
        features.push(cell_params.bandwidth_mhz / 100.0); // Normalize to typical max
        
        // Technology encoding
        let tech_encoded = self.encode_technology(&cell_params.technology);
        features.push(tech_encoded);
        
        Ok(features)
    }
    
    /// Extract advanced interference signature features
    fn extract_advanced_features(&self, measurements: &[NoiseFloorMeasurement]) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        // Interference signature based on noise floor pattern
        let signature = self.calculate_interference_signature(measurements);
        features.push(signature);
        
        Ok(features)
    }
    
    /// Calculate mean of values
    fn calculate_mean(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }
    
    /// Calculate standard deviation of values
    fn calculate_std(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = self.calculate_mean(values);
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        variance.sqrt()
    }
    
    /// Calculate minimum value
    fn calculate_min(&self, values: &[f64]) -> f64 {
        values.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    }
    
    /// Calculate maximum value
    fn calculate_max(&self, values: &[f64]) -> f64 {
        values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    }
    
    /// Calculate trend (slope of linear regression)
    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let n = values.len() as f64;
        let x_sum = (0..values.len()).map(|i| i as f64).sum::<f64>();
        let y_sum = values.iter().sum::<f64>();
        let xy_sum = values.iter().enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum::<f64>();
        let x2_sum = (0..values.len()).map(|i| (i as f64).powi(2)).sum::<f64>();
        
        let denominator = n * x2_sum - x_sum.powi(2);
        if denominator.abs() < 1e-10 {
            return 0.0;
        }
        
        (n * xy_sum - x_sum * y_sum) / denominator
    }
    
    /// Calculate correlation between two value series
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }
        
        let mean_x = self.calculate_mean(x);
        let mean_y = self.calculate_mean(y);
        
        let numerator = x.iter().zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>();
        
        let sum_sq_x = x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>();
        let sum_sq_y = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>();
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator.abs() < 1e-10 {
            return 0.0;
        }
        
        numerator / denominator
    }
    
    /// Encode frequency band to numerical value
    fn encode_frequency_band(&self, band: &str) -> f64 {
        match band {
            "B1" => 0.1,
            "B3" => 0.2,
            "B7" => 0.3,
            "B20" => 0.4,
            "B28" => 0.5,
            "n1" => 0.6,
            "n3" => 0.7,
            "n7" => 0.8,
            "n28" => 0.9,
            _ => 0.0,
        }
    }
    
    /// Encode technology to numerical value
    fn encode_technology(&self, tech: &str) -> f64 {
        match tech.to_uppercase().as_str() {
            "LTE" => 0.5,
            "NR" => 1.0,
            "5G" => 1.0,
            _ => 0.0,
        }
    }
    
    /// Calculate interference signature based on noise floor patterns
    fn calculate_interference_signature(&self, measurements: &[NoiseFloorMeasurement]) -> f64 {
        // This is a simplified signature calculation
        // In practice, this would use more sophisticated pattern recognition
        let noise_values: Vec<f64> = measurements.iter()
            .map(|m| m.noise_floor_pusch)
            .collect();
        
        if noise_values.len() < 3 {
            return 0.0;
        }
        
        // Calculate pattern irregularity
        let mut irregularity = 0.0;
        for i in 1..noise_values.len() - 1 {
            let prev = noise_values[i - 1];
            let curr = noise_values[i];
            let next = noise_values[i + 1];
            
            // Look for sudden spikes or drops
            let spike = (curr - prev).abs() + (curr - next).abs();
            irregularity += spike;
        }
        
        irregularity / (noise_values.len() - 2) as f64
    }
    
    /// Get feature names
    pub fn get_feature_names(&self) -> &[String] {
        &self.feature_names
    }
    
    /// Fit normalization parameters from training data
    pub fn fit_normalization(&mut self, feature_matrix: &Array2<f64>) -> Result<()> {
        if feature_matrix.nrows() == 0 {
            return Err(InterferenceClassifierError::InvalidInputError(
                "Empty feature matrix".to_string()
            ));
        }
        
        for (i, feature_name) in self.feature_names.iter().enumerate() {
            if i >= feature_matrix.ncols() {
                break;
            }
            
            let column = feature_matrix.column(i);
            let mean = column.mean().unwrap_or(0.0);
            let std = column.std(0.0);
            
            self.normalization_stats.insert(feature_name.clone(), (mean, std));
        }
        
        Ok(())
    }
    
    /// Normalize feature vector
    pub fn normalize_features(&self, features: &mut Vec<f64>) -> Result<()> {
        for (i, feature) in features.iter_mut().enumerate() {
            if i >= self.feature_names.len() {
                break;
            }
            
            let feature_name = &self.feature_names[i];
            if let Some((mean, std)) = self.normalization_stats.get(feature_name) {
                if *std > 1e-10 {
                    *feature = (*feature - mean) / std;
                } else {
                    *feature = 0.0;
                }
            }
        }
        
        Ok(())
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    
    fn create_test_measurement() -> NoiseFloorMeasurement {
        NoiseFloorMeasurement {
            timestamp: Utc::now(),
            noise_floor_pusch: -100.0,
            noise_floor_pucch: -102.0,
            cell_ret: 0.05,
            rsrp: -80.0,
            sinr: 15.0,
            active_users: 50,
            prb_utilization: 0.6,
        }
    }
    
    fn create_test_cell_params() -> CellParameters {
        CellParameters {
            cell_id: "test_cell_001".to_string(),
            frequency_band: "B1".to_string(),
            tx_power: 43.0,
            antenna_count: 4,
            bandwidth_mhz: 20.0,
            technology: "LTE".to_string(),
        }
    }
    
    #[test]
    fn test_feature_extraction() {
        let extractor = FeatureExtractor::new();
        let measurements = vec![create_test_measurement(); 20];
        let cell_params = create_test_cell_params();
        
        let features = extractor.extract_features(&measurements, &cell_params).unwrap();
        assert_eq!(features.len(), FEATURE_VECTOR_SIZE);
    }
    
    #[test]
    fn test_insufficient_measurements() {
        let extractor = FeatureExtractor::new();
        let measurements = vec![create_test_measurement(); 5]; // Too few
        let cell_params = create_test_cell_params();
        
        let result = extractor.extract_features(&measurements, &cell_params);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_statistical_calculations() {
        let extractor = FeatureExtractor::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_eq!(extractor.calculate_mean(&values), 3.0);
        assert!((extractor.calculate_std(&values) - 1.58).abs() < 0.01);
        assert_eq!(extractor.calculate_min(&values), 1.0);
        assert_eq!(extractor.calculate_max(&values), 5.0);
    }
    
    #[test]
    fn test_correlation_calculation() {
        let extractor = FeatureExtractor::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let correlation = extractor.calculate_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 0.01); // Perfect positive correlation
    }
}