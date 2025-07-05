//! DBSCAN clustering implementation for density-based cell behavior analysis

use crate::ClusteringConfig;
use anyhow::{anyhow, Result};
use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;
use std::collections::HashMap;

/// DBSCAN clustering implementation
pub struct DBSCANClusterer {
    pub eps: f64,
    pub min_samples: usize,
    pub distance_metric: String,
}

impl DBSCANClusterer {
    /// Create a new DBSCAN clusterer
    pub fn new() -> Self {
        Self {
            eps: 0.5,
            min_samples: 3,
            distance_metric: "euclidean".to_string(),
        }
    }

    /// Perform DBSCAN clustering
    pub async fn cluster(
        &mut self,
        data: &Array2<f64>,
        config: &ClusteringConfig,
    ) -> Result<Vec<i32>> {
        let (n_samples, _) = data.dim();
        
        // Update parameters from config
        self.eps = config.eps;
        self.min_samples = config.min_samples;
        self.distance_metric = config.distance_metric.clone();

        if config.auto_tune {
            self.auto_tune_parameters(data).await?;
        }

        log::info!("Starting DBSCAN clustering with eps={:.3}, min_samples={}", self.eps, self.min_samples);

        // Initialize labels: -1 for noise, 0 for unvisited, >0 for cluster ID
        let mut labels = vec![-1i32; n_samples];
        let mut visited = vec![false; n_samples];
        let mut cluster_id = 0i32;

        // Process each point
        for i in 0..n_samples {
            if visited[i] {
                continue;
            }

            visited[i] = true;
            let neighbors = self.find_neighbors(data, i)?;

            if neighbors.len() < self.min_samples {
                // Point is noise
                labels[i] = -1;
            } else {
                // Start new cluster
                cluster_id += 1;
                labels[i] = cluster_id;
                
                // Expand cluster
                self.expand_cluster(data, &neighbors, cluster_id, &mut labels, &mut visited)?;
            }
        }

        // Convert noise points (-1) to a separate cluster or keep as noise
        let unique_clusters: std::collections::HashSet<i32> = labels.iter()
            .filter(|&&label| label > 0)
            .copied()
            .collect();

        log::info!("DBSCAN clustering completed: {} clusters, {} noise points", 
            unique_clusters.len(),
            labels.iter().filter(|&&label| label == -1).count()
        );

        Ok(labels)
    }

    /// Find neighbors within eps distance
    fn find_neighbors(&self, data: &Array2<f64>, point_idx: usize) -> Result<Vec<usize>> {
        let (n_samples, _) = data.dim();
        let mut neighbors = Vec::new();

        for i in 0..n_samples {
            if i != point_idx {
                let distance = self.calculate_distance(&data.row(point_idx), &data.row(i));
                if distance <= self.eps {
                    neighbors.push(i);
                }
            }
        }

        Ok(neighbors)
    }

    /// Expand cluster by adding density-reachable points
    fn expand_cluster(
        &self,
        data: &Array2<f64>,
        seed_neighbors: &[usize],
        cluster_id: i32,
        labels: &mut [i32],
        visited: &mut [bool],
    ) -> Result<()> {
        let mut neighbors_queue = seed_neighbors.to_vec();
        let mut i = 0;

        while i < neighbors_queue.len() {
            let neighbor_idx = neighbors_queue[i];
            i += 1;

            if !visited[neighbor_idx] {
                visited[neighbor_idx] = true;
                let neighbor_neighbors = self.find_neighbors(data, neighbor_idx)?;

                if neighbor_neighbors.len() >= self.min_samples {
                    // Add new neighbors to the queue
                    for &new_neighbor in &neighbor_neighbors {
                        if !neighbors_queue.contains(&new_neighbor) {
                            neighbors_queue.push(new_neighbor);
                        }
                    }
                }
            }

            // If neighbor is not assigned to any cluster, assign it to current cluster
            if labels[neighbor_idx] == -1 {
                labels[neighbor_idx] = cluster_id;
            }
        }

        Ok(())
    }

    /// Calculate distance between two points
    fn calculate_distance(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        match self.distance_metric.as_str() {
            "euclidean" => self.euclidean_distance(a, b),
            "manhattan" => self.manhattan_distance(a, b),
            "cosine" => self.cosine_distance(a, b),
            _ => self.euclidean_distance(a, b), // Default to euclidean
        }
    }

    /// Calculate Euclidean distance
    fn euclidean_distance(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate Manhattan distance
    fn manhattan_distance(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f64>()
    }

    /// Calculate Cosine distance
    fn cosine_distance(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            1.0 // Maximum distance
        } else {
            1.0 - (dot_product / (norm_a * norm_b))
        }
    }

    /// Auto-tune DBSCAN parameters
    async fn auto_tune_parameters(&mut self, data: &Array2<f64>) -> Result<()> {
        log::info!("Auto-tuning DBSCAN parameters...");
        
        // Use k-distance method to find optimal eps
        let k = self.min_samples;
        let optimal_eps = self.find_optimal_eps(data, k)?;
        
        // Use density-based heuristic for min_samples
        let optimal_min_samples = self.find_optimal_min_samples(data, optimal_eps)?;
        
        self.eps = optimal_eps;
        self.min_samples = optimal_min_samples;
        
        log::info!("Auto-tuned parameters: eps={:.3}, min_samples={}", self.eps, self.min_samples);
        Ok(())
    }

    /// Find optimal eps using k-distance method
    fn find_optimal_eps(&self, data: &Array2<f64>, k: usize) -> Result<f64> {
        let (n_samples, _) = data.dim();
        let mut k_distances = Vec::new();

        // Calculate k-distance for each point
        for i in 0..n_samples {
            let mut distances = Vec::new();
            
            for j in 0..n_samples {
                if i != j {
                    let distance = self.calculate_distance(&data.row(i), &data.row(j));
                    distances.push(distance);
                }
            }
            
            // Sort distances and get k-th distance
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if distances.len() >= k {
                k_distances.push(distances[k - 1]);
            }
        }

        // Sort k-distances and find elbow point
        k_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Simple elbow detection: look for the point with maximum acceleration
        let mut best_eps = k_distances[k_distances.len() / 2]; // Default to median
        let mut max_acceleration = 0.0;
        
        for i in 1..k_distances.len() - 1 {
            let acceleration = k_distances[i + 1] - 2.0 * k_distances[i] + k_distances[i - 1];
            if acceleration > max_acceleration {
                max_acceleration = acceleration;
                best_eps = k_distances[i];
            }
        }

        Ok(best_eps)
    }

    /// Find optimal min_samples based on data density
    fn find_optimal_min_samples(&self, data: &Array2<f64>, eps: f64) -> Result<usize> {
        let (n_samples, n_features) = data.dim();
        
        // Calculate average number of neighbors per point
        let mut total_neighbors = 0;
        
        for i in 0..n_samples {
            let neighbors = self.count_neighbors_within_eps(data, i, eps)?;
            total_neighbors += neighbors;
        }
        
        let avg_neighbors = total_neighbors as f64 / n_samples as f64;
        
        // Use rule of thumb: min_samples = max(dimension + 1, avg_neighbors / 2)
        let min_samples_heuristic = (n_features + 1).max((avg_neighbors / 2.0) as usize);
        
        // Ensure minimum value
        Ok(min_samples_heuristic.max(3))
    }

    /// Count neighbors within eps distance
    fn count_neighbors_within_eps(&self, data: &Array2<f64>, point_idx: usize, eps: f64) -> Result<usize> {
        let (n_samples, _) = data.dim();
        let mut count = 0;

        for i in 0..n_samples {
            if i != point_idx {
                let distance = self.calculate_distance(&data.row(point_idx), &data.row(i));
                if distance <= eps {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Calculate cluster quality metrics
    pub fn calculate_cluster_quality(&self, data: &Array2<f64>, labels: &[i32]) -> Result<ClusterQuality> {
        let unique_clusters: std::collections::HashSet<i32> = labels.iter()
            .filter(|&&label| label > 0)
            .copied()
            .collect();
        
        let num_clusters = unique_clusters.len();
        let num_noise = labels.iter().filter(|&&label| label == -1).count();
        let num_points = labels.len();
        
        // Calculate noise ratio
        let noise_ratio = num_noise as f64 / num_points as f64;
        
        // Calculate average cluster size
        let clustered_points = num_points - num_noise;
        let avg_cluster_size = if num_clusters > 0 {
            clustered_points as f64 / num_clusters as f64
        } else {
            0.0
        };
        
        // Calculate cluster density
        let mut total_intra_cluster_distance = 0.0;
        let mut total_pairs = 0;
        
        for cluster_id in unique_clusters {
            let cluster_points: Vec<usize> = labels.iter()
                .enumerate()
                .filter(|(_, &label)| label == cluster_id)
                .map(|(i, _)| i)
                .collect();
            
            if cluster_points.len() > 1 {
                for i in 0..cluster_points.len() {
                    for j in i + 1..cluster_points.len() {
                        let distance = self.calculate_distance(
                            &data.row(cluster_points[i]),
                            &data.row(cluster_points[j])
                        );
                        total_intra_cluster_distance += distance;
                        total_pairs += 1;
                    }
                }
            }
        }
        
        let avg_intra_cluster_distance = if total_pairs > 0 {
            total_intra_cluster_distance / total_pairs as f64
        } else {
            0.0
        };
        
        Ok(ClusterQuality {
            num_clusters,
            num_noise,
            noise_ratio,
            avg_cluster_size,
            avg_intra_cluster_distance,
        })
    }
}

/// Cluster quality metrics for DBSCAN
#[derive(Debug, Clone)]
pub struct ClusterQuality {
    pub num_clusters: usize,
    pub num_noise: usize,
    pub noise_ratio: f64,
    pub avg_cluster_size: f64,
    pub avg_intra_cluster_distance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_dbscan_creation() {
        let clusterer = DBSCANClusterer::new();
        assert_eq!(clusterer.eps, 0.5);
        assert_eq!(clusterer.min_samples, 3);
    }

    #[test]
    fn test_distance_calculations() {
        let clusterer = DBSCANClusterer::new();
        let data = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 3.0, 4.0]).unwrap();
        
        let euclidean_dist = clusterer.euclidean_distance(&data.row(0), &data.row(1));
        assert!((euclidean_dist - 5.0).abs() < 1e-10);
        
        let manhattan_dist = clusterer.manhattan_distance(&data.row(0), &data.row(1));
        assert!((manhattan_dist - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_neighbors() {
        let clusterer = DBSCANClusterer::new();
        let data = Array2::from_shape_vec((4, 2), vec![
            0.0, 0.0,   // Point 0
            0.1, 0.1,   // Point 1 - close to point 0
            5.0, 5.0,   // Point 2 - far from others
            0.2, 0.2,   // Point 3 - close to point 0
        ]).unwrap();

        let neighbors = clusterer.find_neighbors(&data, 0).unwrap();
        
        // Points 1 and 3 should be neighbors of point 0 (within eps=0.5)
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&3));
        assert!(!neighbors.contains(&2)); // Point 2 is too far
    }

    #[tokio::test]
    async fn test_dbscan_clustering() {
        let mut clusterer = DBSCANClusterer::new();
        clusterer.eps = 1.0;
        clusterer.min_samples = 2;
        
        // Create data with 2 clusters and some noise
        let data = Array2::from_shape_vec((7, 2), vec![
            0.0, 0.0,   // Cluster 1
            0.5, 0.5,   // Cluster 1
            1.0, 1.0,   // Cluster 1
            5.0, 5.0,   // Cluster 2
            5.5, 5.5,   // Cluster 2
            6.0, 6.0,   // Cluster 2
            10.0, 10.0, // Noise
        ]).unwrap();

        let config = ClusteringConfig {
            eps: 1.0,
            min_samples: 2,
            auto_tune: false,
            ..Default::default()
        };

        let labels = clusterer.cluster(&data, &config).await.unwrap();
        
        // Should have 2 clusters plus noise
        let unique_labels: std::collections::HashSet<i32> = labels.iter().copied().collect();
        assert!(unique_labels.len() <= 3); // 2 clusters + noise (-1)
        
        // Check that noise point is labeled as -1
        assert_eq!(labels[6], -1);
    }

    #[test]
    fn test_cosine_distance() {
        let clusterer = DBSCANClusterer::new();
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        
        let cosine_dist = clusterer.cosine_distance(&data.row(0), &data.row(1));
        assert!((cosine_dist - 1.0).abs() < 1e-10); // Orthogonal vectors
    }

    #[test]
    fn test_cluster_quality() {
        let clusterer = DBSCANClusterer::new();
        let data = Array2::from_shape_vec((4, 2), vec![
            0.0, 0.0,
            0.1, 0.1,
            5.0, 5.0,
            10.0, 10.0,
        ]).unwrap();
        
        let labels = vec![1, 1, 2, -1]; // 2 clusters, 1 noise point
        let quality = clusterer.calculate_cluster_quality(&data, &labels).unwrap();
        
        assert_eq!(quality.num_clusters, 2);
        assert_eq!(quality.num_noise, 1);
        assert!((quality.noise_ratio - 0.25).abs() < 1e-10);
    }
}