//! Cell behavior profiling and analysis engine
//!
//! This module provides automated cell profiling capabilities, generating detailed
//! behavior profiles, anomaly detection, and strategic recommendations for RAN optimization.

use crate::{
    Anomaly, CellProfile, Cluster, ProfileStatistics, PrbUtilizationVector,
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

pub mod analyzer;
pub mod anomaly_detector;
pub mod pattern_matcher;
pub mod recommender;

pub use analyzer::*;
pub use anomaly_detector::*;
pub use pattern_matcher::*;
pub use recommender::*;

/// Main profiling engine for cell behavior analysis
pub struct ProfilingEngine {
    pub analyzer: BehaviorAnalyzer,
    pub anomaly_detector: AnomalyDetector,
    pub pattern_matcher: PatternMatcher,
    pub recommender: StrategicRecommender,
}

impl ProfilingEngine {
    /// Create a new profiling engine
    pub fn new() -> Result<Self> {
        Ok(Self {
            analyzer: BehaviorAnalyzer::new(),
            anomaly_detector: AnomalyDetector::new(),
            pattern_matcher: PatternMatcher::new(),
            recommender: StrategicRecommender::new(),
        })
    }

    /// Generate comprehensive cell profiles from clustering results
    pub async fn generate_profiles(
        &self,
        prb_vectors: &[PrbUtilizationVector],
        clusters: &[Cluster],
    ) -> Result<Vec<CellProfile>> {
        log::info!("Generating cell profiles for {} clusters", clusters.len());
        
        let mut profiles = Vec::new();
        
        // Create cell ID to PRB vector mapping
        let prb_map: HashMap<String, &PrbUtilizationVector> = prb_vectors.iter()
            .map(|v| (v.cell_id.clone(), v))
            .collect();
        
        // Generate profile for each cell in each cluster
        for cluster in clusters {
            for cell_id in &cluster.cell_ids {
                if let Some(&prb_vector) = prb_map.get(cell_id) {
                    let profile = self.generate_cell_profile(
                        prb_vector,
                        cluster,
                        &prb_map,
                    ).await?;
                    profiles.push(profile);
                }
            }
        }
        
        log::info!("Generated {} cell profiles", profiles.len());
        Ok(profiles)
    }

    /// Generate a comprehensive profile for a single cell
    async fn generate_cell_profile(
        &self,
        prb_vector: &PrbUtilizationVector,
        cluster: &Cluster,
        prb_map: &HashMap<String, &PrbUtilizationVector>,
    ) -> Result<CellProfile> {
        let cell_id = &prb_vector.cell_id;
        
        // Analyze cell behavior patterns
        let behavior_analysis = self.analyzer.analyze_behavior(prb_vector).await?;
        
        // Calculate comprehensive statistics
        let statistics = self.calculate_profile_statistics(prb_vector)?;
        
        // Detect anomalies in the cell's behavior
        let anomalies = self.anomaly_detector.detect_anomalies(prb_vector, cluster).await?;
        
        // Match behavior patterns
        let pattern_match = self.pattern_matcher.match_patterns(&behavior_analysis).await?;
        
        // Generate strategic recommendations
        let recommendations = self.recommender.generate_recommendations(
            prb_vector,
            &behavior_analysis,
            &anomalies,
            cluster,
        ).await?;
        
        // Calculate cluster confidence (how well this cell fits in its cluster)
        let cluster_confidence = self.calculate_cluster_confidence(
            prb_vector,
            cluster,
            prb_map,
        )?;
        
        // Determine typical 24-hour pattern
        let typical_pattern = self.extract_typical_pattern(prb_vector, cluster, prb_map)?;
        
        Ok(CellProfile {
            cell_id: cell_id.clone(),
            cluster_id: cluster.id,
            cluster_confidence,
            behavior_type: pattern_match.primary_pattern,
            typical_pattern,
            statistics,
            anomalies,
            recommendations,
            last_updated: Utc::now(),
        })
    }

    /// Calculate comprehensive profile statistics
    fn calculate_profile_statistics(&self, prb_vector: &PrbUtilizationVector) -> Result<ProfileStatistics> {
        let utilization = &prb_vector.hourly_prb_utilization;
        
        // Basic statistics
        let mean_utilization = utilization.iter().sum::<f64>() / utilization.len() as f64;
        let variance = utilization.iter()
            .map(|x| (x - mean_utilization).powi(2))
            .sum::<f64>() / utilization.len() as f64;
        let std_utilization = variance.sqrt();
        
        let min_utilization = utilization.iter().copied().fold(f64::INFINITY, f64::min);
        let max_utilization = utilization.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        
        // Median and percentiles
        let mut sorted_utilization = utilization.clone();
        sorted_utilization.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median_utilization = if sorted_utilization.is_empty() {
            0.0
        } else if sorted_utilization.len() % 2 == 0 {
            let mid = sorted_utilization.len() / 2;
            (sorted_utilization[mid - 1] + sorted_utilization[mid]) / 2.0
        } else {
            sorted_utilization[sorted_utilization.len() / 2]
        };
        
        let percentile_95_idx = (0.95 * sorted_utilization.len() as f64) as usize;
        let percentile_95 = sorted_utilization.get(percentile_95_idx.min(sorted_utilization.len() - 1))
            .copied()
            .unwrap_or(0.0);
        
        // Trend analysis
        let trend_slope = self.calculate_trend_slope(utilization)?;
        
        // Seasonality analysis
        let seasonality_strength = self.calculate_seasonality_strength(utilization)?;
        
        // Autocorrelation
        let autocorrelation = self.calculate_autocorrelation(utilization, 1)?;
        
        // Coefficient of variation
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

    /// Calculate how well a cell fits in its assigned cluster
    fn calculate_cluster_confidence(
        &self,
        prb_vector: &PrbUtilizationVector,
        cluster: &Cluster,
        prb_map: &HashMap<String, &PrbUtilizationVector>,
    ) -> Result<f64> {
        let cell_pattern = &prb_vector.hourly_prb_utilization;
        
        // Calculate similarity to cluster centroid (if available in meaningful form)
        // For now, calculate similarity to other cells in the cluster
        let mut similarities = Vec::new();
        
        for other_cell_id in &cluster.cell_ids {
            if other_cell_id != &prb_vector.cell_id {
                if let Some(&other_prb_vector) = prb_map.get(other_cell_id) {
                    let other_pattern = &other_prb_vector.hourly_prb_utilization;
                    let similarity = self.calculate_pattern_similarity(cell_pattern, other_pattern)?;
                    similarities.push(similarity);
                }
            }
        }
        
        // Return average similarity to cluster members
        if similarities.is_empty() {
            Ok(1.0) // Single-cell cluster
        } else {
            Ok(similarities.iter().sum::<f64>() / similarities.len() as f64)
        }
    }

    /// Extract typical 24-hour pattern for the cell
    fn extract_typical_pattern(
        &self,
        prb_vector: &PrbUtilizationVector,
        cluster: &Cluster,
        prb_map: &HashMap<String, &PrbUtilizationVector>,
    ) -> Result<Vec<f64>> {
        // For now, return the cell's own pattern
        // In a more sophisticated implementation, this could be an average
        // of similar cells or a smoothed version
        let mut typical_pattern = prb_vector.hourly_prb_utilization.clone();
        
        // Apply smoothing to reduce noise
        typical_pattern = self.smooth_pattern(&typical_pattern, 2)?;
        
        Ok(typical_pattern)
    }

    /// Calculate pattern similarity between two 24-hour patterns
    fn calculate_pattern_similarity(&self, pattern1: &[f64], pattern2: &[f64]) -> Result<f64> {
        if pattern1.len() != pattern2.len() {
            return Ok(0.0);
        }
        
        // Use correlation coefficient as similarity measure
        let mean1 = pattern1.iter().sum::<f64>() / pattern1.len() as f64;
        let mean2 = pattern2.iter().sum::<f64>() / pattern2.len() as f64;
        
        let numerator: f64 = pattern1.iter().zip(pattern2.iter())
            .map(|(x, y)| (x - mean1) * (y - mean2))
            .sum();
        
        let sum_sq1: f64 = pattern1.iter().map(|x| (x - mean1).powi(2)).sum();
        let sum_sq2: f64 = pattern2.iter().map(|x| (x - mean2).powi(2)).sum();
        
        let denominator = (sum_sq1 * sum_sq2).sqrt();
        
        if denominator > 0.0 {
            Ok((numerator / denominator).abs()) // Use absolute correlation
        } else {
            Ok(0.0)
        }
    }

    /// Calculate trend slope using linear regression
    fn calculate_trend_slope(&self, values: &[f64]) -> Result<f64> {
        if values.len() < 2 {
            return Ok(0.0);
        }
        
        let n = values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;
        
        let numerator: f64 = values.iter().enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();
        
        let denominator: f64 = values.iter().enumerate()
            .map(|(i, _)| (i as f64 - x_mean).powi(2))
            .sum();
        
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Calculate seasonality strength for 24-hour pattern
    fn calculate_seasonality_strength(&self, values: &[f64]) -> Result<f64> {
        if values.len() != 24 {
            return Ok(0.0);
        }
        
        // Calculate variance between different parts of the day
        let mut period_means = Vec::new();
        
        // Split day into 4 periods: night (0-5), morning (6-11), day (12-17), evening (18-23)
        let periods = [(0, 6), (6, 12), (12, 18), (18, 24)];
        
        for (start, end) in periods {
            let period_values: Vec<f64> = values[start..end].to_vec();
            let period_mean = period_values.iter().sum::<f64>() / period_values.len() as f64;
            period_means.push(period_mean);
        }
        
        // Calculate variance of period means
        let overall_mean = period_means.iter().sum::<f64>() / period_means.len() as f64;
        let period_variance = period_means.iter()
            .map(|x| (x - overall_mean).powi(2))
            .sum::<f64>() / period_means.len() as f64;
        
        // Calculate total variance
        let overall_value_mean = values.iter().sum::<f64>() / values.len() as f64;
        let total_variance = values.iter()
            .map(|x| (x - overall_value_mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        // Seasonality strength as ratio of period variance to total variance
        if total_variance > 0.0 {
            Ok(period_variance / total_variance)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate autocorrelation at specified lag
    fn calculate_autocorrelation(&self, values: &[f64], lag: usize) -> Result<f64> {
        if values.len() <= lag {
            return Ok(0.0);
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        
        if variance == 0.0 {
            return Ok(0.0);
        }
        
        let covariance = values.iter().take(values.len() - lag).enumerate()
            .map(|(i, &x)| (x - mean) * (values[i + lag] - mean))
            .sum::<f64>() / (values.len() - lag) as f64;
        
        Ok(covariance / variance)
    }

    /// Apply smoothing to reduce noise in patterns
    fn smooth_pattern(&self, pattern: &[f64], window_size: usize) -> Result<Vec<f64>> {
        if pattern.len() < window_size || window_size == 0 {
            return Ok(pattern.to_vec());
        }
        
        let mut smoothed = Vec::new();
        
        for i in 0..pattern.len() {
            let start = if i >= window_size / 2 {
                i - window_size / 2
            } else {
                0
            };
            let end = (i + window_size / 2 + 1).min(pattern.len());
            
            let window_sum: f64 = pattern[start..end].iter().sum();
            let window_mean = window_sum / (end - start) as f64;
            smoothed.push(window_mean);
        }
        
        Ok(smoothed)
    }

    /// Generate profile summary for reporting
    pub fn generate_profile_summary(&self, profiles: &[CellProfile]) -> Result<ProfileSummary> {
        if profiles.is_empty() {
            return Ok(ProfileSummary::default());
        }
        
        let total_cells = profiles.len();
        
        // Count behavior types
        let mut behavior_counts = HashMap::new();
        for profile in profiles {
            *behavior_counts.entry(profile.behavior_type.clone()).or_insert(0) += 1;
        }
        
        // Count anomalies
        let total_anomalies: usize = profiles.iter().map(|p| p.anomalies.len()).sum();
        let cells_with_anomalies = profiles.iter().filter(|p| !p.anomalies.is_empty()).count();
        
        // Calculate average confidence
        let avg_confidence = profiles.iter().map(|p| p.cluster_confidence).sum::<f64>() / total_cells as f64;
        
        // Calculate utilization statistics
        let avg_utilization = profiles.iter()
            .map(|p| p.statistics.mean_utilization)
            .sum::<f64>() / total_cells as f64;
        
        let peak_utilization = profiles.iter()
            .map(|p| p.statistics.max_utilization)
            .fold(0.0, f64::max);
        
        Ok(ProfileSummary {
            total_cells,
            behavior_counts,
            total_anomalies,
            cells_with_anomalies,
            avg_confidence,
            avg_utilization,
            peak_utilization,
        })
    }
}

/// Profile summary for reporting and analysis
#[derive(Debug, Clone, Default)]
pub struct ProfileSummary {
    pub total_cells: usize,
    pub behavior_counts: HashMap<String, usize>,
    pub total_anomalies: usize,
    pub cells_with_anomalies: usize,
    pub avg_confidence: f64,
    pub avg_utilization: f64,
    pub peak_utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CellMetadata, Location};

    #[test]
    fn test_profiling_engine_creation() {
        let engine = ProfilingEngine::new().unwrap();
        assert!(true); // Engine created successfully
    }

    #[test]
    fn test_trend_slope_calculation() {
        let engine = ProfilingEngine::new().unwrap();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let slope = engine.calculate_trend_slope(&values).unwrap();
        assert!((slope - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pattern_similarity() {
        let engine = ProfilingEngine::new().unwrap();
        let pattern1 = vec![1.0, 2.0, 3.0, 4.0];
        let pattern2 = vec![2.0, 4.0, 6.0, 8.0]; // Perfect correlation
        
        let similarity = engine.calculate_pattern_similarity(&pattern1, &pattern2).unwrap();
        assert!((similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_seasonality_calculation() {
        let engine = ProfilingEngine::new().unwrap();
        
        // Create pattern with clear seasonality
        let mut pattern = vec![10.0; 24]; // Base level
        // Add morning peak
        for i in 6..10 {
            pattern[i] = 80.0;
        }
        // Add evening peak
        for i in 18..22 {
            pattern[i] = 90.0;
        }
        
        let seasonality = engine.calculate_seasonality_strength(&pattern).unwrap();
        assert!(seasonality > 0.0);
    }

    #[test]
    fn test_autocorrelation() {
        let engine = ProfilingEngine::new().unwrap();
        let values = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]; // Alternating pattern
        let autocorr = engine.calculate_autocorrelation(&values, 1).unwrap();
        assert!(autocorr < 0.0); // Should be negative for alternating pattern
    }

    #[test]
    fn test_smoothing() {
        let engine = ProfilingEngine::new().unwrap();
        let noisy_pattern = vec![1.0, 10.0, 2.0, 9.0, 3.0]; // Noisy data
        let smoothed = engine.smooth_pattern(&noisy_pattern, 3).unwrap();
        
        assert_eq!(smoothed.len(), noisy_pattern.len());
        // Smoothed values should be less extreme than original
        assert!(smoothed[1] < noisy_pattern[1]);
        assert!(smoothed[1] > noisy_pattern[0]);
    }

    fn create_test_prb_vector() -> PrbUtilizationVector {
        PrbUtilizationVector {
            cell_id: "test_cell_001".to_string(),
            date: "2024-01-01".to_string(),
            hourly_prb_utilization: vec![
                20.0, 15.0, 10.0, 8.0, 12.0, 18.0, // 00-05: Night
                35.0, 55.0, 70.0, 80.0, 75.0, 65.0, // 06-11: Morning peak
                60.0, 58.0, 62.0, 65.0, 68.0, 72.0, // 12-17: Day
                85.0, 90.0, 88.0, 82.0, 45.0, 30.0, // 18-23: Evening peak
            ],
            metadata: CellMetadata {
                cell_id: "test_cell_001".to_string(),
                site_id: "site_001".to_string(),
                technology: "5G".to_string(),
                frequency_band: "n78".to_string(),
                tx_power: 43.0,
                antenna_count: 64,
                bandwidth_mhz: 100.0,
                location: Location {
                    latitude: 37.7749,
                    longitude: -122.4194,
                    altitude: 100.0,
                },
                cell_type: "macro".to_string(),
                height_meters: 30.0,
                environment: "urban".to_string(),
            },
            additional_features: vec![0.85, 0.92, 0.78],
        }
    }
}