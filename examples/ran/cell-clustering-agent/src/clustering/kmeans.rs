//! K-Means clustering implementation optimized for RAN cell behavior analysis

use crate::ClusteringConfig;
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rayon::prelude::*;

/// K-Means clustering implementation
pub struct KMeansClusterer {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub init_method: String,
}

impl KMeansClusterer {
    /// Create a new K-Means clusterer
    pub fn new() -> Self {
        Self {
            max_iterations: 300,
            convergence_threshold: 1e-4,
            init_method: "k-means++".to_string(),
        }
    }

    /// Perform K-Means clustering
    pub async fn cluster(
        &mut self,
        data: &Array2<f64>,
        config: &ClusteringConfig,
    ) -> Result<Vec<i32>> {
        let (n_samples, n_features) = data.dim();
        let k = if config.auto_tune {
            self.find_optimal_k(data, 2..=10).await?
        } else {
            config.num_clusters
        };

        if k == 0 || k > n_samples {
            return Err(anyhow!("Invalid number of clusters: {}", k));
        }

        log::info!("Starting K-Means clustering with k={}, {} samples", k, n_samples);

        // Initialize centroids
        let mut centroids = self.initialize_centroids(data, k)?;
        let mut assignments = vec![0i32; n_samples];
        let mut prev_centroids = centroids.clone();

        // Main K-Means loop
        for iteration in 0..self.max_iterations {
            // Assignment step
            self.assign_points_to_centroids(data, &centroids, &mut assignments)?;

            // Update step
            self.update_centroids(data, &assignments, &mut centroids, k)?;

            // Check convergence
            let centroid_shift = self.calculate_centroid_shift(&centroids, &prev_centroids);
            if centroid_shift < self.convergence_threshold {
                log::info!("K-Means converged after {} iterations", iteration + 1);
                break;
            }

            prev_centroids = centroids.clone();
        }

        log::info!("K-Means clustering completed with {} clusters", k);
        Ok(assignments)
    }

    /// Initialize centroids using K-Means++ algorithm
    fn initialize_centroids(&self, data: &Array2<f64>, k: usize) -> Result<Array2<f64>> {
        let (n_samples, n_features) = data.dim();
        let mut rng = thread_rng();
        let mut centroids = Array2::zeros((k, n_features));

        if self.init_method == "k-means++" {
            // K-Means++ initialization
            
            // Choose first centroid randomly
            let first_idx = rng.gen_range(0..n_samples);
            centroids.row_mut(0).assign(&data.row(first_idx));

            // Choose remaining centroids with probability proportional to squared distance
            for i in 1..k {
                let mut distances = vec![f64::INFINITY; n_samples];
                
                // Calculate minimum distance to existing centroids
                for j in 0..n_samples {
                    for c in 0..i {
                        let dist = self.euclidean_distance(&data.row(j), &centroids.row(c));
                        distances[j] = distances[j].min(dist * dist);
                    }
                }

                // Choose next centroid with probability proportional to squared distance
                let total_distance: f64 = distances.iter().sum();
                if total_distance == 0.0 {
                    // If all distances are 0, choose randomly
                    let idx = rng.gen_range(0..n_samples);
                    centroids.row_mut(i).assign(&data.row(idx));
                } else {
                    let mut cumulative_probs = vec![0.0; n_samples];
                    cumulative_probs[0] = distances[0] / total_distance;
                    
                    for j in 1..n_samples {
                        cumulative_probs[j] = cumulative_probs[j-1] + distances[j] / total_distance;
                    }

                    let random_val = rng.gen::<f64>();
                    let chosen_idx = cumulative_probs.iter()
                        .position(|&p| p >= random_val)
                        .unwrap_or(n_samples - 1);
                    
                    centroids.row_mut(i).assign(&data.row(chosen_idx));
                }
            }
        } else {
            // Random initialization
            for i in 0..k {
                let idx = rng.gen_range(0..n_samples);
                centroids.row_mut(i).assign(&data.row(idx));
            }
        }

        Ok(centroids)
    }

    /// Assign points to nearest centroids
    fn assign_points_to_centroids(
        &self,
        data: &Array2<f64>,
        centroids: &Array2<f64>,
        assignments: &mut [i32],
    ) -> Result<()> {
        let (n_samples, _) = data.dim();
        let k = centroids.nrows();

        // Parallel assignment for better performance
        assignments.par_iter_mut().enumerate().for_each(|(i, assignment)| {
            let mut min_distance = f64::INFINITY;
            let mut closest_centroid = 0;

            for j in 0..k {
                let distance = self.euclidean_distance(&data.row(i), &centroids.row(j));
                if distance < min_distance {
                    min_distance = distance;
                    closest_centroid = j;
                }
            }

            *assignment = closest_centroid as i32;
        });

        Ok(())
    }

    /// Update centroids based on current assignments
    fn update_centroids(
        &self,
        data: &Array2<f64>,
        assignments: &[i32],
        centroids: &mut Array2<f64>,
        k: usize,
    ) -> Result<()> {
        let (n_samples, n_features) = data.dim();
        let mut counts = vec![0; k];
        
        // Reset centroids
        centroids.fill(0.0);

        // Sum points for each cluster
        for i in 0..n_samples {
            let cluster_id = assignments[i] as usize;
            if cluster_id < k {
                for j in 0..n_features {
                    centroids[[cluster_id, j]] += data[[i, j]];
                }
                counts[cluster_id] += 1;
            }
        }

        // Calculate averages
        for i in 0..k {
            if counts[i] > 0 {
                for j in 0..n_features {
                    centroids[[i, j]] /= counts[i] as f64;
                }
            }
        }

        Ok(())
    }

    /// Calculate shift in centroids between iterations
    fn calculate_centroid_shift(&self, current: &Array2<f64>, previous: &Array2<f64>) -> f64 {
        let mut total_shift = 0.0;
        let k = current.nrows();

        for i in 0..k {
            let shift = self.euclidean_distance(&current.row(i), &previous.row(i));
            total_shift += shift;
        }

        total_shift / k as f64
    }

    /// Calculate Euclidean distance between two points
    fn euclidean_distance(&self, a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Find optimal number of clusters using elbow method
    async fn find_optimal_k(&self, data: &Array2<f64>, k_range: std::ops::RangeInclusive<usize>) -> Result<usize> {
        let mut inertias = Vec::new();
        let mut best_k = *k_range.start();
        let mut best_score = f64::INFINITY;

        for k in k_range {
            let mut temp_clusterer = KMeansClusterer::new();
            temp_clusterer.max_iterations = 100; // Reduced iterations for k-selection
            
            let assignments = temp_clusterer.cluster_fixed_k(data, k).await?;
            let inertia = self.calculate_inertia(data, &assignments, k)?;
            inertias.push(inertia);

            // Simple elbow detection - look for largest decrease in inertia
            if inertias.len() > 1 {
                let prev_inertia = inertias[inertias.len() - 2];
                let improvement = prev_inertia - inertia;
                let improvement_rate = improvement / prev_inertia;

                if improvement_rate > 0.1 && inertia < best_score {
                    best_score = inertia;
                    best_k = k;
                }
            }
        }

        log::info!("Optimal k selected: {}", best_k);
        Ok(best_k)
    }

    /// Perform K-Means clustering with fixed k
    async fn cluster_fixed_k(&mut self, data: &Array2<f64>, k: usize) -> Result<Vec<i32>> {
        let (n_samples, n_features) = data.dim();

        if k == 0 || k > n_samples {
            return Err(anyhow!("Invalid number of clusters: {}", k));
        }

        // Initialize centroids
        let mut centroids = self.initialize_centroids(data, k)?;
        let mut assignments = vec![0i32; n_samples];
        let mut prev_centroids = centroids.clone();

        // Main K-Means loop
        for _ in 0..self.max_iterations {
            // Assignment step
            self.assign_points_to_centroids(data, &centroids, &mut assignments)?;

            // Update step
            self.update_centroids(data, &assignments, &mut centroids, k)?;

            // Check convergence
            let centroid_shift = self.calculate_centroid_shift(&centroids, &prev_centroids);
            if centroid_shift < self.convergence_threshold {
                break;
            }

            prev_centroids = centroids.clone();
        }

        Ok(assignments)
    }

    /// Calculate inertia (within-cluster sum of squares)
    fn calculate_inertia(&self, data: &Array2<f64>, assignments: &[i32], k: usize) -> Result<f64> {
        let (n_samples, n_features) = data.dim();
        let mut centroids = Array2::zeros((k, n_features));
        let mut counts = vec![0; k];
        
        // Calculate centroids
        for i in 0..n_samples {
            let cluster_id = assignments[i] as usize;
            if cluster_id < k {
                for j in 0..n_features {
                    centroids[[cluster_id, j]] += data[[i, j]];
                }
                counts[cluster_id] += 1;
            }
        }

        for i in 0..k {
            if counts[i] > 0 {
                for j in 0..n_features {
                    centroids[[i, j]] /= counts[i] as f64;
                }
            }
        }

        // Calculate inertia
        let mut inertia = 0.0;
        for i in 0..n_samples {
            let cluster_id = assignments[i] as usize;
            if cluster_id < k {
                let distance = self.euclidean_distance(&data.row(i), &centroids.row(cluster_id));
                inertia += distance * distance;
            }
        }

        Ok(inertia)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ClusteringConfig;
    use ndarray::Array2;

    #[test]
    fn test_kmeans_creation() {
        let clusterer = KMeansClusterer::new();
        assert_eq!(clusterer.max_iterations, 300);
        assert_eq!(clusterer.convergence_threshold, 1e-4);
    }

    #[test]
    fn test_euclidean_distance() {
        let clusterer = KMeansClusterer::new();
        let data = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 3.0, 4.0]).unwrap();
        let distance = clusterer.euclidean_distance(&data.row(0), &data.row(1));
        assert!((distance - 5.0).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_kmeans_clustering() {
        let mut clusterer = KMeansClusterer::new();
        
        // Create simple 2D data with 2 obvious clusters
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

        // Check that points are properly clustered
        // Points 0,1,2 should be in one cluster, points 3,4,5 in another
        let cluster_0 = assignments[0];
        let cluster_1 = assignments[3];
        assert_ne!(cluster_0, cluster_1);
        assert_eq!(assignments[1], cluster_0);
        assert_eq!(assignments[2], cluster_0);
        assert_eq!(assignments[4], cluster_1);
        assert_eq!(assignments[5], cluster_1);
    }

    #[test]
    fn test_initialize_centroids() {
        let clusterer = KMeansClusterer::new();
        let data = Array2::from_shape_vec((4, 2), vec![
            1.0, 1.0,
            2.0, 2.0,
            8.0, 8.0,
            9.0, 9.0,
        ]).unwrap();

        let centroids = clusterer.initialize_centroids(&data, 2).unwrap();
        assert_eq!(centroids.shape(), &[2, 2]);
    }

    #[tokio::test]
    async fn test_optimal_k_selection() {
        let clusterer = KMeansClusterer::new();
        
        // Create data with 3 obvious clusters
        let data = Array2::from_shape_vec((9, 2), vec![
            1.0, 1.0, 1.5, 1.5, 2.0, 2.0,   // Cluster 1
            8.0, 8.0, 8.5, 8.5, 9.0, 9.0,   // Cluster 2
            15.0, 15.0, 15.5, 15.5, 16.0, 16.0, // Cluster 3
        ]).unwrap();

        let optimal_k = clusterer.find_optimal_k(&data, 2..=5).await.unwrap();
        assert!(optimal_k >= 2 && optimal_k <= 5);
    }
}