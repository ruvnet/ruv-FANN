//! Clustering Metrics Calculator
//! 
//! This module provides comprehensive metrics for evaluating clustering quality
//! and performance in the RAN Intelligence Platform.

use anyhow::Result;
use std::collections::HashMap;
use crate::{FeatureVector, ClusteringMetrics};

/// Metrics calculator for clustering evaluation
#[derive(Debug, Clone)]
pub struct MetricsCalculator {
    pub performance_tracking: bool,
    pub quality_assessment: bool,
}

impl MetricsCalculator {
    /// Create a new metrics calculator
    pub fn new() -> Self {
        Self {
            performance_tracking: true,
            quality_assessment: true,
        }
    }
    
    /// Calculate comprehensive clustering metrics
    pub fn calculate_metrics(
        &self,
        features: &[FeatureVector],
        clusters: &[crate::Cluster],
    ) -> Result<ClusteringMetrics> {
        let start_time = std::time::Instant::now();
        
        if features.is_empty() || clusters.is_empty() {
            return Ok(ClusteringMetrics {
                silhouette_score: 0.0,
                calinski_harabasz_score: 0.0,
                davies_bouldin_score: 0.0,
                inertia: 0.0,
                num_clusters: 0,
                num_cells: 0,
                cluster_separation: 0.0,
                cluster_cohesion: 0.0,
                stability_score: 0.0,
                coverage: 0.0,
            });
        }
        
        // Basic metrics
        let num_clusters = clusters.len();
        let num_cells = features.len();
        
        // Calculate silhouette score
        let silhouette_score = self.calculate_silhouette_score(features, clusters)?;
        
        // Calculate Calinski-Harabasz score (variance ratio criterion)
        let calinski_harabasz_score = self.calculate_calinski_harabasz_score(features, clusters)?;
        
        // Calculate Davies-Bouldin score
        let davies_bouldin_score = self.calculate_davies_bouldin_score(features, clusters)?;
        
        // Calculate inertia (within-cluster sum of squares)
        let inertia = self.calculate_inertia(clusters)?;
        
        // Calculate cluster separation
        let cluster_separation = self.calculate_cluster_separation(clusters)?;
        
        // Calculate cluster cohesion
        let cluster_cohesion = self.calculate_cluster_cohesion(features, clusters)?;
        
        // Calculate stability score
        let stability_score = self.calculate_stability_score(features, clusters)?;
        
        // Calculate coverage (percentage of points assigned to clusters)
        let coverage = self.calculate_coverage(features, clusters)?;
        
        let processing_time = start_time.elapsed().as_millis() as f64;
        
        log::info!("Clustering metrics calculated in {}ms", processing_time);
        log::info!("Silhouette Score: {:.3}, Calinski-Harabasz: {:.3}, Davies-Bouldin: {:.3}", 
                  silhouette_score, calinski_harabasz_score, davies_bouldin_score);
        
        Ok(ClusteringMetrics {
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score,
            inertia,
            num_clusters,
            num_cells,
            cluster_separation,
            cluster_cohesion,
            stability_score,
            coverage,
        })
    }
    
    /// Calculate silhouette score
    fn calculate_silhouette_score(
        &self,
        features: &[FeatureVector],
        clusters: &[crate::Cluster],
    ) -> Result<f64> {
        if clusters.len() <= 1 {
            return Ok(0.0);
        }
        
        let mut total_silhouette = 0.0;
        let mut count = 0;
        
        // Create cell ID to cluster mapping
        let cell_to_cluster: HashMap<String, usize> = clusters.iter()
            .enumerate()
            .flat_map(|(cluster_idx, cluster)| {
                cluster.cell_ids.iter().map(move |cell_id| (cell_id.clone(), cluster_idx))
            })
            .collect();
        
        for feature in features {
            if let Some(&cluster_idx) = cell_to_cluster.get(&feature.cell_id) {
                let cluster = &clusters[cluster_idx];
                
                // Calculate a(i) - average distance to points in same cluster
                let same_cluster_features: Vec<&FeatureVector> = features.iter()
                    .filter(|f| cluster.cell_ids.contains(&f.cell_id) && f.cell_id != feature.cell_id)
                    .collect();
                
                let a = if same_cluster_features.is_empty() {
                    0.0
                } else {
                    same_cluster_features.iter()
                        .map(|f| euclidean_distance(&feature.features, &f.features).unwrap_or(0.0))
                        .sum::<f64>() / same_cluster_features.len() as f64
                };
                
                // Calculate b(i) - minimum average distance to points in other clusters
                let mut min_b = f64::INFINITY;
                
                for (other_cluster_idx, other_cluster) in clusters.iter().enumerate() {
                    if other_cluster_idx != cluster_idx {
                        let other_cluster_features: Vec<&FeatureVector> = features.iter()
                            .filter(|f| other_cluster.cell_ids.contains(&f.cell_id))
                            .collect();
                        
                        if !other_cluster_features.is_empty() {
                            let avg_distance = other_cluster_features.iter()
                                .map(|f| euclidean_distance(&feature.features, &f.features).unwrap_or(0.0))
                                .sum::<f64>() / other_cluster_features.len() as f64;
                            
                            min_b = min_b.min(avg_distance);
                        }
                    }
                }
                
                let b = if min_b == f64::INFINITY { 0.0 } else { min_b };
                
                // Calculate silhouette score for this point
                let silhouette = if a == 0.0 && b == 0.0 {
                    0.0
                } else {
                    (b - a) / f64::max(a, b)
                };
                
                total_silhouette += silhouette;
                count += 1;
            }
        }
        
        Ok(if count > 0 { total_silhouette / count as f64 } else { 0.0 })
    }
    
    /// Calculate Calinski-Harabasz score (variance ratio criterion)
    fn calculate_calinski_harabasz_score(
        &self,
        features: &[FeatureVector],
        clusters: &[crate::Cluster],
    ) -> Result<f64> {
        if clusters.len() <= 1 {
            return Ok(0.0);
        }
        
        let n_samples = features.len();
        let n_clusters = clusters.len();
        
        // Calculate overall centroid
        let overall_centroid = self.calculate_overall_centroid(features)?;
        
        // Calculate between-cluster sum of squares
        let mut between_ss = 0.0;
        for cluster in clusters {
            let cluster_size = cluster.size as f64;
            let distance = euclidean_distance(&cluster.centroid, &overall_centroid)?;
            between_ss += cluster_size * distance.powi(2);
        }
        
        // Calculate within-cluster sum of squares
        let within_ss: f64 = clusters.iter().map(|c| c.inertia).sum();
        
        if within_ss == 0.0 {
            return Ok(0.0);
        }
        
        // Calinski-Harabasz score
        let score = (between_ss / (n_clusters - 1) as f64) / (within_ss / (n_samples - n_clusters) as f64);
        
        Ok(score)
    }
    
    /// Calculate Davies-Bouldin score
    fn calculate_davies_bouldin_score(
        &self,
        _features: &[FeatureVector],
        clusters: &[crate::Cluster],
    ) -> Result<f64> {
        if clusters.len() <= 1 {
            return Ok(0.0);
        }
        
        let mut total_db = 0.0;
        
        for i in 0..clusters.len() {
            let mut max_db = 0.0;
            
            for j in 0..clusters.len() {
                if i != j {
                    let cluster_i = &clusters[i];
                    let cluster_j = &clusters[j];
                    
                    // Average within-cluster distances (approximated as sqrt(inertia/size))
                    let s_i = if cluster_i.size > 0 {
                        (cluster_i.inertia / cluster_i.size as f64).sqrt()
                    } else {
                        0.0
                    };
                    
                    let s_j = if cluster_j.size > 0 {
                        (cluster_j.inertia / cluster_j.size as f64).sqrt()
                    } else {
                        0.0
                    };
                    
                    // Distance between cluster centroids
                    let d_ij = euclidean_distance(&cluster_i.centroid, &cluster_j.centroid)?;
                    
                    if d_ij > 0.0 {
                        let db_ij = (s_i + s_j) / d_ij;
                        max_db = max_db.max(db_ij);
                    }
                }
            }
            
            total_db += max_db;
        }
        
        Ok(total_db / clusters.len() as f64)
    }
    
    /// Calculate total inertia (within-cluster sum of squares)
    fn calculate_inertia(&self, clusters: &[crate::Cluster]) -> Result<f64> {
        Ok(clusters.iter().map(|c| c.inertia).sum())
    }
    
    /// Calculate cluster separation (average distance between cluster centroids)
    fn calculate_cluster_separation(&self, clusters: &[crate::Cluster]) -> Result<f64> {
        if clusters.len() <= 1 {
            return Ok(0.0);
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let distance = euclidean_distance(&clusters[i].centroid, &clusters[j].centroid)?;
                total_distance += distance;
                count += 1;
            }
        }
        
        Ok(if count > 0 { total_distance / count as f64 } else { 0.0 })
    }
    
    /// Calculate cluster cohesion (average within-cluster distance)
    fn calculate_cluster_cohesion(
        &self,
        features: &[FeatureVector],
        clusters: &[crate::Cluster],
    ) -> Result<f64> {
        if clusters.is_empty() {
            return Ok(0.0);
        }
        
        let mut total_cohesion = 0.0;
        let mut cluster_count = 0;
        
        for cluster in clusters {
            if cluster.size > 1 {
                // Get features for this cluster
                let cluster_features: Vec<&FeatureVector> = features.iter()
                    .filter(|f| cluster.cell_ids.contains(&f.cell_id))
                    .collect();
                
                if cluster_features.len() > 1 {
                    let mut total_distance = 0.0;
                    let mut pair_count = 0;
                    
                    for i in 0..cluster_features.len() {
                        for j in (i + 1)..cluster_features.len() {
                            let distance = euclidean_distance(
                                &cluster_features[i].features,
                                &cluster_features[j].features
                            )?;
                            total_distance += distance;
                            pair_count += 1;
                        }
                    }
                    
                    if pair_count > 0 {
                        total_cohesion += total_distance / pair_count as f64;
                        cluster_count += 1;
                    }
                }
            }
        }
        
        Ok(if cluster_count > 0 { total_cohesion / cluster_count as f64 } else { 0.0 })
    }
    
    /// Calculate stability score (consistency of cluster assignments)
    fn calculate_stability_score(
        &self,
        features: &[FeatureVector],
        clusters: &[crate::Cluster],
    ) -> Result<f64> {
        // Simplified stability measure based on cluster density
        if clusters.is_empty() {
            return Ok(0.0);
        }
        
        let mut total_density = 0.0;
        for cluster in clusters {
            total_density += cluster.density;
        }
        
        let average_density = total_density / clusters.len() as f64;
        
        // Normalize to 0-1 range
        Ok(average_density.min(1.0).max(0.0))
    }
    
    /// Calculate coverage (percentage of points assigned to meaningful clusters)
    fn calculate_coverage(
        &self,
        features: &[FeatureVector],
        clusters: &[crate::Cluster],
    ) -> Result<f64> {
        if features.is_empty() {
            return Ok(0.0);
        }
        
        let total_assigned: usize = clusters.iter().map(|c| c.size).sum();
        Ok(total_assigned as f64 / features.len() as f64)
    }
    
    /// Calculate overall centroid of all features
    fn calculate_overall_centroid(&self, features: &[FeatureVector]) -> Result<Vec<f64>> {
        if features.is_empty() {
            return Ok(Vec::new());
        }
        
        let dim = features[0].features.len();
        let mut centroid = vec![0.0; dim];
        
        for feature in features {
            for (i, &val) in feature.features.iter().enumerate() {
                centroid[i] += val;
            }
        }
        
        for coord in &mut centroid {
            *coord /= features.len() as f64;
        }
        
        Ok(centroid)
    }
}

/// Calculate Euclidean distance between two vectors
fn euclidean_distance(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(anyhow::anyhow!("Vector dimensions don't match: {} vs {}", a.len(), b.len()));
    }
    
    let distance = a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    
    Ok(distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeatureVector, Cluster, ClusterCharacteristics};
    
    #[test]
    fn test_metrics_calculation() {
        let calculator = MetricsCalculator::new();
        
        // Create test features
        let features = vec![
            FeatureVector {
                cell_id: "cell1".to_string(),
                features: vec![1.0, 2.0, 3.0],
                feature_names: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
                normalized: false,
            },
            FeatureVector {
                cell_id: "cell2".to_string(),
                features: vec![1.1, 2.1, 3.1],
                feature_names: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
                normalized: false,
            },
        ];
        
        // Create test clusters
        let clusters = vec![
            Cluster {
                id: 0,
                name: "Cluster_0".to_string(),
                behavior_pattern: "test".to_string(),
                cell_ids: vec!["cell1".to_string(), "cell2".to_string()],
                characteristics: ClusterCharacteristics {
                    primary_pattern: "test".to_string(),
                    avg_utilization: 0.5,
                    peak_utilization: 0.8,
                    utilization_variance: 100.0,
                    peak_hours: "none".to_string(),
                    load_profile: "medium".to_string(),
                    predictability: 0.7,
                    anomaly_level: "low".to_string(),
                    dominant_patterns: vec!["test".to_string()],
                },
                centroid: vec![1.05, 2.05, 3.05],
                inertia: 0.1,
                size: 2,
                density: 0.8,
                silhouette_score: 0.5,
            },
        ];
        
        let metrics = calculator.calculate_metrics(&features, &clusters).unwrap();
        
        assert_eq!(metrics.num_clusters, 1);
        assert_eq!(metrics.num_cells, 2);
        assert!(metrics.coverage > 0.0);
    }
    
    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let distance = euclidean_distance(&a, &b).unwrap();
        let expected = ((4.0 - 1.0).powi(2) + (5.0 - 2.0).powi(2) + (6.0 - 3.0).powi(2)).sqrt();
        
        assert!((distance - expected).abs() < 1e-10);
    }
}