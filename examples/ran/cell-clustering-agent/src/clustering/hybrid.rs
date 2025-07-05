//! Hybrid clustering implementation combining multiple algorithms for optimal cell behavior analysis

use crate::{ClusteringConfig, ClusteringAlgorithm};
use super::{KMeansClusterer, DBSCANClusterer, HierarchicalClusterer};
use anyhow::{anyhow, Result};
use ndarray::Array2;
use std::collections::HashMap;

/// Hybrid clustering implementation
pub struct HybridClusterer {
    pub primary_algorithm: ClusteringAlgorithm,
    pub secondary_algorithm: ClusteringAlgorithm,
    pub consensus_threshold: f64,
    pub auto_select_algorithm: bool,
}

impl HybridClusterer {
    /// Create a new hybrid clusterer
    pub fn new() -> Self {
        Self {
            primary_algorithm: ClusteringAlgorithm::KMeans,
            secondary_algorithm: ClusteringAlgorithm::DBSCAN,
            consensus_threshold: 0.7,
            auto_select_algorithm: true,
        }
    }

    /// Perform hybrid clustering
    pub async fn cluster(
        &mut self,
        data: &Array2<f64>,
        config: &ClusteringConfig,
    ) -> Result<Vec<i32>> {
        let (n_samples, n_features) = data.dim();
        
        log::info!("Starting hybrid clustering with {} samples, {} features", n_samples, n_features);
        
        if self.auto_select_algorithm {
            self.select_optimal_algorithms(data, config).await?;
        }

        // Run multiple clustering algorithms
        let results = self.run_multiple_algorithms(data, config).await?;
        
        // Combine results using ensemble method
        let final_assignments = self.ensemble_clustering(&results, n_samples)?;
        
        log::info!("Hybrid clustering completed with {} final clusters", 
            self.count_unique_clusters(&final_assignments));
        
        Ok(final_assignments)
    }

    /// Automatically select optimal algorithms based on data characteristics
    async fn select_optimal_algorithms(&mut self, data: &Array2<f64>, config: &ClusteringConfig) -> Result<()> {
        let characteristics = self.analyze_data_characteristics(data)?;
        
        log::info!("Data characteristics: density={:.3}, separability={:.3}, noise_level={:.3}",
            characteristics.density, characteristics.separability, characteristics.noise_level);
        
        // Select primary algorithm based on data characteristics
        if characteristics.noise_level > 0.3 {
            // High noise - DBSCAN is better
            self.primary_algorithm = ClusteringAlgorithm::DBSCAN;
            self.secondary_algorithm = ClusteringAlgorithm::Hierarchical;
        } else if characteristics.separability < 0.5 {
            // Poor separability - try hierarchical first
            self.primary_algorithm = ClusteringAlgorithm::Hierarchical;
            self.secondary_algorithm = ClusteringAlgorithm::KMeans;
        } else if characteristics.density > 0.7 {
            // High density - K-means should work well
            self.primary_algorithm = ClusteringAlgorithm::KMeans;
            self.secondary_algorithm = ClusteringAlgorithm::DBSCAN;
        } else {
            // Mixed characteristics - use K-means + Hierarchical
            self.primary_algorithm = ClusteringAlgorithm::KMeans;
            self.secondary_algorithm = ClusteringAlgorithm::Hierarchical;
        }
        
        log::info!("Selected algorithms: primary={:?}, secondary={:?}", 
            self.primary_algorithm, self.secondary_algorithm);
        
        Ok(())
    }

    /// Analyze data characteristics to guide algorithm selection
    fn analyze_data_characteristics(&self, data: &Array2<f64>) -> Result<DataCharacteristics> {
        let (n_samples, n_features) = data.dim();
        
        // Calculate pairwise distances for analysis
        let mut distances = Vec::new();
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let distance = self.euclidean_distance(&data.row(i), &data.row(j));
                distances.push(distance);
            }
        }
        
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate density (inverse of median distance)
        let median_distance = if distances.is_empty() {
            1.0
        } else {
            distances[distances.len() / 2]
        };
        let density = if median_distance > 0.0 {
            1.0 / (1.0 + median_distance)
        } else {
            1.0
        };
        
        // Calculate separability (ratio of 90th percentile to 10th percentile distance)
        let p10_distance = distances.get(distances.len() / 10).unwrap_or(&0.0);
        let p90_distance = distances.get(9 * distances.len() / 10).unwrap_or(&1.0);
        let separability = if *p90_distance > 0.0 {
            p10_distance / p90_distance
        } else {
            0.0
        };
        
        // Estimate noise level based on distance distribution
        let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance = distances.iter()
            .map(|d| (d - mean_distance).powi(2))
            .sum::<f64>() / distances.len() as f64;
        let coefficient_of_variation = if mean_distance > 0.0 {
            variance.sqrt() / mean_distance
        } else {
            0.0
        };
        let noise_level = coefficient_of_variation.min(1.0);
        
        Ok(DataCharacteristics {
            density,
            separability,
            noise_level,
            n_samples,
            n_features,
        })
    }

    /// Run multiple clustering algorithms
    async fn run_multiple_algorithms(
        &self,
        data: &Array2<f64>,
        config: &ClusteringConfig,
    ) -> Result<Vec<ClusteringResult>> {
        let mut results = Vec::new();
        
        // Run primary algorithm
        let primary_result = self.run_single_algorithm(data, config, &self.primary_algorithm).await?;
        results.push(ClusteringResult {
            algorithm: self.primary_algorithm.clone(),
            assignments: primary_result.assignments,
            quality_score: primary_result.quality_score,
            weight: 1.0,
        });
        
        // Run secondary algorithm
        let secondary_result = self.run_single_algorithm(data, config, &self.secondary_algorithm).await?;
        results.push(ClusteringResult {
            algorithm: self.secondary_algorithm.clone(),
            assignments: secondary_result.assignments,
            quality_score: secondary_result.quality_score,
            weight: 0.8, // Slightly lower weight for secondary
        });
        
        // Run additional algorithm if data characteristics suggest it
        if data.nrows() > 100 {
            let tertiary_algorithm = match (&self.primary_algorithm, &self.secondary_algorithm) {
                (ClusteringAlgorithm::KMeans, ClusteringAlgorithm::DBSCAN) => ClusteringAlgorithm::Hierarchical,
                (ClusteringAlgorithm::KMeans, ClusteringAlgorithm::Hierarchical) => ClusteringAlgorithm::DBSCAN,
                (ClusteringAlgorithm::DBSCAN, ClusteringAlgorithm::Hierarchical) => ClusteringAlgorithm::KMeans,
                _ => ClusteringAlgorithm::KMeans,
            };
            
            let tertiary_result = self.run_single_algorithm(data, config, &tertiary_algorithm).await?;
            results.push(ClusteringResult {
                algorithm: tertiary_algorithm,
                assignments: tertiary_result.assignments,
                quality_score: tertiary_result.quality_score,
                weight: 0.6,
            });
        }
        
        Ok(results)
    }

    /// Run a single clustering algorithm
    async fn run_single_algorithm(
        &self,
        data: &Array2<f64>,
        config: &ClusteringConfig,
        algorithm: &ClusteringAlgorithm,
    ) -> Result<SingleResult> {
        let assignments = match algorithm {
            ClusteringAlgorithm::KMeans => {
                let mut kmeans = KMeansClusterer::new();
                kmeans.cluster(data, config).await?
            }
            ClusteringAlgorithm::DBSCAN => {
                let mut dbscan = DBSCANClusterer::new();
                dbscan.cluster(data, config).await?
            }
            ClusteringAlgorithm::Hierarchical => {
                let mut hierarchical = HierarchicalClusterer::new();
                hierarchical.cluster(data, config).await?
            }
            _ => {
                // Fallback to K-means for unsupported algorithms
                let mut kmeans = KMeansClusterer::new();
                kmeans.cluster(data, config).await?
            }
        };
        
        let quality_score = self.calculate_quality_score(data, &assignments)?;
        
        Ok(SingleResult {
            assignments,
            quality_score,
        })
    }

    /// Combine multiple clustering results using ensemble method
    fn ensemble_clustering(&self, results: &[ClusteringResult], n_samples: usize) -> Result<Vec<i32>> {
        if results.is_empty() {
            return Err(anyhow!("No clustering results to combine"));
        }
        
        // Use consensus clustering approach
        let consensus_assignments = self.consensus_clustering(results, n_samples)?;
        
        // If consensus is too low, use the best single result
        let consensus_quality = self.calculate_consensus_quality(&consensus_assignments, results)?;
        
        if consensus_quality < self.consensus_threshold {
            log::info!("Consensus quality {:.3} below threshold {:.3}, using best single result", 
                consensus_quality, self.consensus_threshold);
            
            // Find the best single result
            let best_result = results.iter()
                .max_by(|a, b| (a.quality_score * a.weight).partial_cmp(&(b.quality_score * b.weight)).unwrap())
                .unwrap();
            
            return Ok(best_result.assignments.clone());
        }
        
        Ok(consensus_assignments)
    }

    /// Perform consensus clustering
    fn consensus_clustering(&self, results: &[ClusteringResult], n_samples: usize) -> Result<Vec<i32>> {
        // Create co-association matrix
        let mut co_association = vec![vec![0.0; n_samples]; n_samples];
        let mut total_weight = 0.0;
        
        for result in results {
            total_weight += result.weight;
            
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i != j && result.assignments[i] == result.assignments[j] && result.assignments[i] >= 0 {
                        co_association[i][j] += result.weight;
                    }
                }
            }
        }
        
        // Normalize co-association matrix
        for i in 0..n_samples {
            for j in 0..n_samples {
                co_association[i][j] /= total_weight;
            }
        }
        
        // Apply hierarchical clustering to co-association matrix
        let consensus_assignments = self.cluster_coassociation_matrix(&co_association)?;
        
        Ok(consensus_assignments)
    }

    /// Cluster the co-association matrix to get final assignments
    fn cluster_coassociation_matrix(&self, co_association: &[Vec<f64>]) -> Result<Vec<i32>> {
        let n_samples = co_association.len();
        let mut assignments = vec![0i32; n_samples];
        let mut visited = vec![false; n_samples];
        let mut cluster_id = 0;
        
        // Use a simple connected components approach
        for i in 0..n_samples {
            if visited[i] {
                continue;
            }
            
            // Start new cluster
            let mut cluster_members = Vec::new();
            let mut queue = vec![i];
            
            while let Some(current) = queue.pop() {
                if visited[current] {
                    continue;
                }
                
                visited[current] = true;
                cluster_members.push(current);
                
                // Find connected points (co-association > threshold)
                for j in 0..n_samples {
                    if !visited[j] && co_association[current][j] > 0.5 {
                        queue.push(j);
                    }
                }
            }
            
            // Assign cluster ID to all members
            for &member in &cluster_members {
                assignments[member] = cluster_id;
            }
            
            cluster_id += 1;
        }
        
        Ok(assignments)
    }

    /// Calculate quality score for clustering result
    fn calculate_quality_score(&self, data: &Array2<f64>, assignments: &[i32]) -> Result<f64> {
        // Simple silhouette score calculation
        let n_samples = data.nrows();
        let mut silhouette_scores = Vec::new();
        
        for i in 0..n_samples {
            let cluster_i = assignments[i];
            
            if cluster_i < 0 {
                continue; // Skip noise points
            }
            
            // Calculate a(i) - average distance to points in same cluster
            let mut same_cluster_distances = Vec::new();
            for j in 0..n_samples {
                if i != j && assignments[j] == cluster_i {
                    let distance = self.euclidean_distance(&data.row(i), &data.row(j));
                    same_cluster_distances.push(distance);
                }
            }
            
            let a_i = if same_cluster_distances.is_empty() {
                0.0
            } else {
                same_cluster_distances.iter().sum::<f64>() / same_cluster_distances.len() as f64
            };
            
            // Calculate b(i) - minimum average distance to points in other clusters
            let unique_clusters: std::collections::HashSet<i32> = assignments.iter()
                .filter(|&&c| c != cluster_i && c >= 0)
                .copied()
                .collect();
            
            let mut min_avg_distance = f64::INFINITY;
            for other_cluster in unique_clusters {
                let mut other_cluster_distances = Vec::new();
                for j in 0..n_samples {
                    if assignments[j] == other_cluster {
                        let distance = self.euclidean_distance(&data.row(i), &data.row(j));
                        other_cluster_distances.push(distance);
                    }
                }
                
                if !other_cluster_distances.is_empty() {
                    let avg_distance = other_cluster_distances.iter().sum::<f64>() / other_cluster_distances.len() as f64;
                    min_avg_distance = min_avg_distance.min(avg_distance);
                }
            }
            
            let b_i = if min_avg_distance == f64::INFINITY {
                0.0
            } else {
                min_avg_distance
            };
            
            // Calculate silhouette score
            let silhouette = if a_i.max(b_i) == 0.0 {
                0.0
            } else {
                (b_i - a_i) / a_i.max(b_i)
            };
            
            silhouette_scores.push(silhouette);
        }
        
        Ok(if silhouette_scores.is_empty() {
            0.0
        } else {
            silhouette_scores.iter().sum::<f64>() / silhouette_scores.len() as f64
        })
    }

    /// Calculate consensus quality
    fn calculate_consensus_quality(&self, consensus: &[i32], results: &[ClusteringResult]) -> Result<f64> {
        if results.is_empty() {
            return Ok(0.0);
        }
        
        let mut total_agreement = 0.0;
        let mut total_weight = 0.0;
        
        for result in results {
            let agreement = self.calculate_agreement(consensus, &result.assignments)?;
            total_agreement += agreement * result.weight;
            total_weight += result.weight;
        }
        
        Ok(total_agreement / total_weight)
    }

    /// Calculate agreement between two clustering assignments
    fn calculate_agreement(&self, assignments1: &[i32], assignments2: &[i32]) -> Result<f64> {
        if assignments1.len() != assignments2.len() {
            return Err(anyhow!("Assignment vectors must have same length"));
        }
        
        let n_samples = assignments1.len();
        let mut agreements = 0;
        let mut total_pairs = 0;
        
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let same_cluster_1 = assignments1[i] == assignments1[j] && assignments1[i] >= 0;
                let same_cluster_2 = assignments2[i] == assignments2[j] && assignments2[i] >= 0;
                
                if same_cluster_1 == same_cluster_2 {
                    agreements += 1;
                }
                total_pairs += 1;
            }
        }
        
        Ok(if total_pairs > 0 {
            agreements as f64 / total_pairs as f64
        } else {
            0.0
        })
    }

    /// Calculate Euclidean distance
    fn euclidean_distance(&self, a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Count unique clusters in assignments
    fn count_unique_clusters(&self, assignments: &[i32]) -> usize {
        let unique_clusters: std::collections::HashSet<i32> = assignments.iter()
            .filter(|&&c| c >= 0)
            .copied()
            .collect();
        unique_clusters.len()
    }
}

/// Data characteristics for algorithm selection
#[derive(Debug)]
struct DataCharacteristics {
    density: f64,
    separability: f64,
    noise_level: f64,
    n_samples: usize,
    n_features: usize,
}

/// Single clustering result
#[derive(Debug)]
struct SingleResult {
    assignments: Vec<i32>,
    quality_score: f64,
}

/// Clustering result with metadata
#[derive(Debug)]
struct ClusteringResult {
    algorithm: ClusteringAlgorithm,
    assignments: Vec<i32>,
    quality_score: f64,
    weight: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_hybrid_creation() {
        let clusterer = HybridClusterer::new();
        assert!(matches!(clusterer.primary_algorithm, ClusteringAlgorithm::KMeans));
        assert!(matches!(clusterer.secondary_algorithm, ClusteringAlgorithm::DBSCAN));
        assert!(clusterer.auto_select_algorithm);
    }

    #[test]
    fn test_data_characteristics() {
        let clusterer = HybridClusterer::new();
        let data = Array2::from_shape_vec((4, 2), vec![
            0.0, 0.0,
            1.0, 1.0,
            5.0, 5.0,
            6.0, 6.0,
        ]).unwrap();

        let characteristics = clusterer.analyze_data_characteristics(&data).unwrap();
        assert_eq!(characteristics.n_samples, 4);
        assert_eq!(characteristics.n_features, 2);
        assert!(characteristics.density > 0.0);
        assert!(characteristics.density <= 1.0);
    }

    #[test]
    fn test_agreement_calculation() {
        let clusterer = HybridClusterer::new();
        let assignments1 = vec![0, 0, 1, 1];
        let assignments2 = vec![1, 1, 0, 0]; // Same clustering, different labels

        let agreement = clusterer.calculate_agreement(&assignments1, &assignments2).unwrap();
        assert!((agreement - 1.0).abs() < 1e-10); // Should be perfect agreement
    }

    #[tokio::test]
    async fn test_hybrid_clustering() {
        let mut clusterer = HybridClusterer::new();
        clusterer.auto_select_algorithm = false; // Use default algorithms
        
        let data = Array2::from_shape_vec((6, 2), vec![
            1.0, 1.0,   // Cluster 1
            1.5, 1.5,   // Cluster 1
            2.0, 2.0,   // Cluster 1
            8.0, 8.0,   // Cluster 2
            8.5, 8.5,   // Cluster 2
            9.0, 9.0,   // Cluster 2
        ]).unwrap();

        let config = ClusteringConfig {
            num_clusters: 2,
            auto_tune: false,
            ..Default::default()
        };

        let assignments = clusterer.cluster(&data, &config).await.unwrap();
        assert_eq!(assignments.len(), 6);

        // Should produce meaningful clustering
        let unique_clusters: std::collections::HashSet<i32> = assignments.iter()
            .filter(|&&c| c >= 0)
            .copied()
            .collect();
        assert!(unique_clusters.len() >= 1);
    }

    #[test]
    fn test_quality_score() {
        let clusterer = HybridClusterer::new();
        let data = Array2::from_shape_vec((4, 2), vec![
            0.0, 0.0,
            0.1, 0.1,
            5.0, 5.0,
            5.1, 5.1,
        ]).unwrap();

        let assignments = vec![0, 0, 1, 1]; // Perfect clustering
        let quality = clusterer.calculate_quality_score(&data, &assignments).unwrap();
        
        // Should have high quality score
        assert!(quality > 0.5);
    }
}