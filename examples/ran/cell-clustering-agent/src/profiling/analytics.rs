//! Analytics Engine for Statistical Analysis
//! 
//! This module provides comprehensive statistical analysis capabilities
//! for cell profiling in the RAN Intelligence Platform.

use anyhow::Result;
use crate::ProfileStatistics;

/// Statistical analytics engine
#[derive(Debug, Clone)]
pub struct AnalyticsEngine {
    pub enable_advanced_stats: bool,
}

impl AnalyticsEngine {
    /// Create a new analytics engine
    pub fn new() -> Result<Self> {
        Ok(Self {
            enable_advanced_stats: true,
        })
    }
    
    /// Calculate comprehensive statistics for PRB utilization data
    pub fn calculate_statistics(&self, hourly_data: &[f64]) -> Result<ProfileStatistics> {
        if hourly_data.is_empty() {
            return Ok(ProfileStatistics {
                mean_utilization: 0.0,
                std_utilization: 0.0,
                min_utilization: 0.0,
                max_utilization: 0.0,
                median_utilization: 0.0,
                percentile_95: 0.0,
                trend_slope: 0.0,
                seasonality_strength: 0.0,
                autocorrelation: 0.0,
                coefficient_of_variation: 0.0,
            });
        }
        
        // Convert to 0-1 range
        let data: Vec<f64> = hourly_data.iter().map(|&x| x / 100.0).collect();
        
        // Basic statistics
        let mean_utilization = self.calculate_mean(&data)?;
        let std_utilization = self.calculate_std(&data, mean_utilization)?;
        let min_utilization = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max_utilization = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let median_utilization = self.calculate_median(&data)?;
        let percentile_95 = self.calculate_percentile(&data, 0.95)?;
        
        // Advanced statistics
        let trend_slope = if self.enable_advanced_stats {
            self.calculate_trend_slope(&data)?
        } else {
            0.0
        };
        
        let seasonality_strength = if self.enable_advanced_stats {
            self.calculate_seasonality_strength(&data)?
        } else {
            0.0
        };
        
        let autocorrelation = if self.enable_advanced_stats {
            self.calculate_autocorrelation(&data, 1)?
        } else {
            0.0
        };
        
        let coefficient_of_variation = if mean_utilization > 0.0 {
            std_utilization / mean_utilization
        } else {
            0.0
        };
        
        Ok(ProfileStatistics {
            mean_utilization,
            std_utilization,
            min_utilization,
            max_utilization,
            median_utilization,
            percentile_95,
            trend_slope,
            seasonality_strength,
            autocorrelation,
            coefficient_of_variation,
        })
    }
    
    /// Calculate typical daily pattern (smoothed)
    pub fn calculate_typical_pattern(&self, hourly_data: &[f64]) -> Result<Vec<f64>> {
        if hourly_data.len() != 24 {
            return Ok(hourly_data.to_vec());
        }
        
        // Apply simple moving average smoothing
        let window_size = 3;
        let mut smoothed = Vec::new();
        
        for i in 0..hourly_data.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = std::cmp::min(i + window_size / 2 + 1, hourly_data.len());
            
            let window_sum: f64 = hourly_data[start..end].iter().sum();
            let window_avg = window_sum / (end - start) as f64;
            
            smoothed.push(window_avg);
        }
        
        Ok(smoothed)
    }
    
    /// Calculate mean
    fn calculate_mean(&self, data: &[f64]) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        Ok(data.iter().sum::<f64>() / data.len() as f64)
    }
    
    /// Calculate standard deviation
    fn calculate_std(&self, data: &[f64], mean: f64) -> Result<f64> {
        if data.len() <= 1 {
            return Ok(0.0);
        }
        
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;
        
        Ok(variance.sqrt())
    }
    
    /// Calculate median
    fn calculate_median(&self, data: &[f64]) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = sorted_data.len();
        if len % 2 == 0 {
            Ok((sorted_data[len / 2 - 1] + sorted_data[len / 2]) / 2.0)
        } else {
            Ok(sorted_data[len / 2])
        }
    }
    
    /// Calculate percentile
    fn calculate_percentile(&self, data: &[f64], percentile: f64) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (percentile * (sorted_data.len() - 1) as f64).round() as usize;
        Ok(sorted_data[index.min(sorted_data.len() - 1)])
    }
    
    /// Calculate trend slope using linear regression
    fn calculate_trend_slope(&self, data: &[f64]) -> Result<f64> {
        if data.len() < 2 {
            return Ok(0.0);
        }
        
        let n = data.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = data.iter().sum::<f64>() / n;
        
        let numerator: f64 = data.iter().enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();
        
        let denominator: f64 = data.iter().enumerate()
            .map(|(i, _)| (i as f64 - x_mean).powi(2))
            .sum();
        
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
    
    /// Calculate seasonality strength
    fn calculate_seasonality_strength(&self, data: &[f64]) -> Result<f64> {
        if data.len() != 24 {
            return Ok(0.0);
        }
        
        // Simple seasonality measure based on hourly pattern variance
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        
        if mean == 0.0 {
            return Ok(0.0);
        }
        
        // Coefficient of variation as seasonality proxy
        let cv = variance.sqrt() / mean;
        Ok(cv.min(2.0)) // Cap at 2.0 for normalization
    }
    
    /// Calculate autocorrelation at given lag
    fn calculate_autocorrelation(&self, data: &[f64], lag: usize) -> Result<f64> {
        if data.len() <= lag {
            return Ok(0.0);
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        
        if variance == 0.0 {
            return Ok(0.0);
        }
        
        let covariance = data.iter().take(data.len() - lag).enumerate()
            .map(|(i, &x)| (x - mean) * (data[i + lag] - mean))
            .sum::<f64>() / (data.len() - lag) as f64;
        
        Ok(covariance / variance)
    }
    
    /// Detect peak hours in the data
    pub fn detect_peak_hours(&self, hourly_data: &[f64]) -> Result<Vec<usize>> {
        if hourly_data.len() != 24 {
            return Ok(Vec::new());
        }
        
        let mean = self.calculate_mean(hourly_data)?;
        let std = self.calculate_std(hourly_data, mean)?;
        let threshold = mean + std;
        
        let peak_hours: Vec<usize> = hourly_data.iter().enumerate()
            .filter(|(_, &value)| value > threshold)
            .map(|(hour, _)| hour)
            .collect();
        
        Ok(peak_hours)
    }
    
    /// Calculate traffic distribution by time periods
    pub fn calculate_traffic_distribution(&self, hourly_data: &[f64]) -> Result<TrafficDistribution> {
        if hourly_data.len() != 24 {
            return Ok(TrafficDistribution::default());
        }
        
        let night_traffic = hourly_data[0..6].iter().sum::<f64>() / 6.0;  // 00-05
        let morning_traffic = hourly_data[6..12].iter().sum::<f64>() / 6.0; // 06-11
        let afternoon_traffic = hourly_data[12..18].iter().sum::<f64>() / 6.0; // 12-17
        let evening_traffic = hourly_data[18..24].iter().sum::<f64>() / 6.0; // 18-23
        
        let total_traffic = night_traffic + morning_traffic + afternoon_traffic + evening_traffic;
        
        Ok(TrafficDistribution {
            night_percentage: if total_traffic > 0.0 { night_traffic / total_traffic } else { 0.0 },
            morning_percentage: if total_traffic > 0.0 { morning_traffic / total_traffic } else { 0.0 },
            afternoon_percentage: if total_traffic > 0.0 { afternoon_traffic / total_traffic } else { 0.0 },
            evening_percentage: if total_traffic > 0.0 { evening_traffic / total_traffic } else { 0.0 },
        })
    }
}

/// Traffic distribution across time periods
#[derive(Debug, Clone)]
pub struct TrafficDistribution {
    pub night_percentage: f64,     // 00-05
    pub morning_percentage: f64,   // 06-11
    pub afternoon_percentage: f64, // 12-17
    pub evening_percentage: f64,   // 18-23
}

impl Default for TrafficDistribution {
    fn default() -> Self {
        Self {
            night_percentage: 0.25,
            morning_percentage: 0.25,
            afternoon_percentage: 0.25,
            evening_percentage: 0.25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_analytics_engine_creation() {
        let engine = AnalyticsEngine::new().unwrap();
        assert!(engine.enable_advanced_stats);
    }
    
    #[test]
    fn test_statistics_calculation() {
        let engine = AnalyticsEngine::new().unwrap();
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        
        let stats = engine.calculate_statistics(&data).unwrap();
        
        assert!((stats.mean_utilization - 0.3).abs() < 1e-10); // 30% in 0-1 range
        assert!(stats.min_utilization == 0.1); // 10% in 0-1 range
        assert!(stats.max_utilization == 0.5); // 50% in 0-1 range
    }
    
    #[test]
    fn test_typical_pattern_calculation() {
        let engine = AnalyticsEngine::new().unwrap();
        let hourly_data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        
        let pattern = engine.calculate_typical_pattern(&hourly_data).unwrap();
        
        assert_eq!(pattern.len(), 24);
        // First element should be smoothed
        assert!(pattern[0] != hourly_data[0]);
    }
    
    #[test]
    fn test_peak_hours_detection() {
        let engine = AnalyticsEngine::new().unwrap();
        let mut hourly_data = vec![30.0; 24];
        hourly_data[8] = 80.0;  // Morning peak
        hourly_data[19] = 85.0; // Evening peak
        
        let peak_hours = engine.detect_peak_hours(&hourly_data).unwrap();
        
        assert!(peak_hours.contains(&8));
        assert!(peak_hours.contains(&19));
    }
    
    #[test]
    fn test_traffic_distribution() {
        let engine = AnalyticsEngine::new().unwrap();
        let hourly_data = vec![
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, // Night: 00-05
            50.0, 50.0, 50.0, 50.0, 50.0, 50.0, // Morning: 06-11
            30.0, 30.0, 30.0, 30.0, 30.0, 30.0, // Afternoon: 12-17
            70.0, 70.0, 70.0, 70.0, 70.0, 70.0, // Evening: 18-23
        ];
        
        let distribution = engine.calculate_traffic_distribution(&hourly_data).unwrap();
        
        // Evening should have highest percentage (70% average)
        assert!(distribution.evening_percentage > distribution.morning_percentage);
        assert!(distribution.morning_percentage > distribution.afternoon_percentage);
        assert!(distribution.afternoon_percentage > distribution.night_percentage);
    }
}