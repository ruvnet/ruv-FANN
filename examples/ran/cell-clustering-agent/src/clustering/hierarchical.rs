//! Hierarchical clustering implementation for RAN cell behavior analysis

use crate::ClusteringConfig;
use anyhow::{anyhow, Result};
use ndarray::{Array2, ArrayView1};
use std::collections::HashMap;

/// Hierarchical clustering implementation
pub struct HierarchicalClusterer {
    pub linkage_method: String,
    pub distance_metric: String,
}

impl HierarchicalClusterer {
    /// Create a new hierarchical clusterer
    pub fn new() -> Self {
        Self {
            linkage_method: "ward".to_string(),
            distance_metric: "euclidean".to_string(),
        }
    }

    /// Perform hierarchical clustering
    pub async fn cluster(
        &mut self,
        data: &Array2<f64>,
        config: &ClusteringConfig,
    ) -> Result<Vec<i32>> {
        let (n_samples, _) = data.dim();
        let target_clusters = if config.auto_tune {
            self.find_optimal_clusters(data, 2..=10).await?
        } else {
            config.num_clusters
        };

        log::info!("Starting hierarchical clustering with {} target clusters", target_clusters);

        // Build distance matrix
        let distance_matrix = self.build_distance_matrix(data)?;
        
        // Perform agglomerative clustering
        let dendrogram = self.agglomerative_clustering(&distance_matrix, n_samples)?;
        
        // Cut dendrogram to get desired number of clusters
        let assignments = self.cut_dendrogram(&dendrogram, target_clusters, n_samples)?;
        
        log::info!("Hierarchical clustering completed with {} clusters", target_clusters);
        Ok(assignments)
    }

    /// Build distance matrix for all pairs of points
    fn build_distance_matrix(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, _) = data.dim();
        let mut distance_matrix = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let distance = self.calculate_distance(&data.row(i), &data.row(j));
                distance_matrix[[i, j]] = distance;
                distance_matrix[[j, i]] = distance;
            }
        }

        Ok(distance_matrix)
    }

    /// Perform agglomerative clustering
    fn agglomerative_clustering(
        &self,
        distance_matrix: &Array2<f64>,
        n_samples: usize,
    ) -> Result<Vec<ClusterMerge>> {
        let mut clusters: Vec<Cluster> = (0..n_samples)
            .map(|i| Cluster {
                id: i,
                members: vec![i],
                size: 1,
            })
            .collect();

        let mut dendrogram = Vec::new();
        let mut next_cluster_id = n_samples;

        while clusters.len() > 1 {
            // Find the closest pair of clusters
            let (min_i, min_j, min_distance) = self.find_closest_clusters(&clusters, distance_matrix)?;

            // Merge the closest clusters
            let cluster_a = clusters.remove(min_i.max(min_j));
            let cluster_b = clusters.remove(min_i.min(min_j));

            let mut new_members = cluster_a.members.clone();
            new_members.extend(cluster_b.members.clone());

            let new_cluster = Cluster {
                id: next_cluster_id,
                members: new_members,
                size: cluster_a.size + cluster_b.size,
            };

            dendrogram.push(ClusterMerge {
                cluster_a: cluster_a.id,
                cluster_b: cluster_b.id,
                distance: min_distance,
                new_cluster: next_cluster_id,
                size: new_cluster.size,
            });

            clusters.push(new_cluster);
            next_cluster_id += 1;
        }

        Ok(dendrogram)
    }

    /// Find the closest pair of clusters
    fn find_closest_clusters(
        &self,
        clusters: &[Cluster],
        distance_matrix: &Array2<f64>,
    ) -> Result<(usize, usize, f64)> {
        let mut min_distance = f64::INFINITY;
        let mut min_i = 0;
        let mut min_j = 0;

        for i in 0..clusters.len() {
            for j in i + 1..clusters.len() {
                let distance = self.calculate_cluster_distance(
                    &clusters[i],
                    &clusters[j],
                    distance_matrix,
                )?;

                if distance < min_distance {
                    min_distance = distance;
                    min_i = i;
                    min_j = j;
                }
            }
        }

        Ok((min_i, min_j, min_distance))
    }

    /// Calculate distance between two clusters based on linkage method
    fn calculate_cluster_distance(
        &self,
        cluster_a: &Cluster,
        cluster_b: &Cluster,
        distance_matrix: &Array2<f64>,
    ) -> Result<f64> {
        match self.linkage_method.as_str() {
            "single" => self.single_linkage(cluster_a, cluster_b, distance_matrix),
            "complete" => self.complete_linkage(cluster_a, cluster_b, distance_matrix),
            "average" => self.average_linkage(cluster_a, cluster_b, distance_matrix),
            "ward" => self.ward_linkage(cluster_a, cluster_b, distance_matrix),
            _ => self.average_linkage(cluster_a, cluster_b, distance_matrix), // Default
        }
    }

    /// Single linkage (minimum distance)
    fn single_linkage(
        &self,
        cluster_a: &Cluster,
        cluster_b: &Cluster,
        distance_matrix: &Array2<f64>,
    ) -> Result<f64> {
        let mut min_distance = f64::INFINITY;

        for &i in &cluster_a.members {
            for &j in &cluster_b.members {
                let distance = distance_matrix[[i, j]];
                min_distance = min_distance.min(distance);
            }
        }

        Ok(min_distance)
    }

    /// Complete linkage (maximum distance)
    fn complete_linkage(
        &self,
        cluster_a: &Cluster,
        cluster_b: &Cluster,
        distance_matrix: &Array2<f64>,
    ) -> Result<f64> {
        let mut max_distance = 0.0;

        for &i in &cluster_a.members {
            for &j in &cluster_b.members {
                let distance = distance_matrix[[i, j]];
                max_distance = max_distance.max(distance);
            }
        }

        Ok(max_distance)
    }

    /// Average linkage (average distance)
    fn average_linkage(
        &self,
        cluster_a: &Cluster,
        cluster_b: &Cluster,
        distance_matrix: &Array2<f64>,
    ) -> Result<f64> {
        let mut total_distance = 0.0;
        let mut count = 0;

        for &i in &cluster_a.members {
            for &j in &cluster_b.members {
                total_distance += distance_matrix[[i, j]];
                count += 1;
            }
        }

        if count > 0 {
            Ok(total_distance / count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Ward linkage (minimize within-cluster sum of squares)
    fn ward_linkage(
        &self,
        cluster_a: &Cluster,
        cluster_b: &Cluster,
        distance_matrix: &Array2<f64>,
    ) -> Result<f64> {
        // Simplified Ward linkage calculation
        // In a full implementation, this would calculate the increase in SSE
        let avg_distance = self.average_linkage(cluster_a, cluster_b, distance_matrix)?;
        let size_factor = (cluster_a.size * cluster_b.size) as f64 / (cluster_a.size + cluster_b.size) as f64;
        
        Ok(avg_distance * size_factor.sqrt())
    }

    /// Cut dendrogram to get desired number of clusters
    fn cut_dendrogram(
        &self,
        dendrogram: &[ClusterMerge],
        target_clusters: usize,
        n_samples: usize,
    ) -> Result<Vec<i32>> {
        if target_clusters > n_samples {
            return Err(anyhow!("Target clusters ({}) cannot exceed number of samples ({})", target_clusters, n_samples));
        }

        // Start with each point as its own cluster
        let mut cluster_assignments = HashMap::new();
        for i in 0..n_samples {
            cluster_assignments.insert(i, i as i32);
        }

        // Apply merges until we have the desired number of clusters
        let merges_to_apply = n_samples - target_clusters;
        
        for merge in dendrogram.iter().take(merges_to_apply) {
            // Find all points that belong to cluster_a or cluster_b
            let mut points_to_merge = Vec::new();
            
            for (&point, &cluster) in &cluster_assignments {
                if self.belongs_to_cluster(point, merge.cluster_a, dendrogram, n_samples)? ||
                   self.belongs_to_cluster(point, merge.cluster_b, dendrogram, n_samples)? {
                    points_to_merge.push(point);
                }
            }

            // Assign all these points to the same cluster
            if let Some(&first_point) = points_to_merge.first() {
                let new_cluster_id = cluster_assignments[&first_point];
                for &point in &points_to_merge {
                    cluster_assignments.insert(point, new_cluster_id);
                }
            }
        }

        // Convert to vector format
        let mut assignments = vec![0i32; n_samples];
        for i in 0..n_samples {
            assignments[i] = cluster_assignments[&i];
        }

        // Renumber clusters to be consecutive starting from 0
        self.renumber_clusters(&mut assignments);

        Ok(assignments)
    }

    /// Check if a point belongs to a cluster (considering the dendrogram)
    fn belongs_to_cluster(
        &self,
        point: usize,
        cluster_id: usize,
        dendrogram: &[ClusterMerge],
        n_samples: usize,
    ) -> Result<bool> {
        if cluster_id < n_samples {
            // Original cluster - check if point matches
            return Ok(point == cluster_id);
        }

        // Find the merge that created this cluster
        for merge in dendrogram {
            if merge.new_cluster == cluster_id {
                return Ok(
                    self.belongs_to_cluster(point, merge.cluster_a, dendrogram, n_samples)? ||
                    self.belongs_to_cluster(point, merge.cluster_b, dendrogram, n_samples)?
                );
            }
        }

        Ok(false)
    }

    /// Renumber clusters to be consecutive starting from 0
    fn renumber_clusters(&self, assignments: &mut [i32]) {
        let mut cluster_map = HashMap::new();
        let mut next_id = 0;

        for assignment in assignments.iter_mut() {
            if !cluster_map.contains_key(assignment) {
                cluster_map.insert(*assignment, next_id);
                next_id += 1;
            }
            *assignment = cluster_map[assignment];
        }
    }

    /// Calculate distance between two points
    fn calculate_distance(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        match self.distance_metric.as_str() {
            "euclidean" => self.euclidean_distance(a, b),
            "manhattan" => self.manhattan_distance(a, b),
            "cosine" => self.cosine_distance(a, b),
            _ => self.euclidean_distance(a, b), // Default
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
            1.0
        } else {
            1.0 - (dot_product / (norm_a * norm_b))
        }
    }

    /// Find optimal number of clusters using silhouette analysis
    async fn find_optimal_clusters(
        &self,
        data: &Array2<f64>,
        k_range: std::ops::RangeInclusive<usize>,
    ) -> Result<usize> {
        let mut best_k = *k_range.start();
        let mut best_silhouette = -1.0;

        for k in k_range {
            let mut temp_clusterer = HierarchicalClusterer::new();
            temp_clusterer.linkage_method = self.linkage_method.clone();
            temp_clusterer.distance_metric = self.distance_metric.clone();
            
            let config = ClusteringConfig {
                num_clusters: k,
                auto_tune: false,
                ..Default::default()
            };

            let assignments = temp_clusterer.cluster(data, &config).await?;
            let silhouette = self.calculate_silhouette_score(data, &assignments)?;

            if silhouette > best_silhouette {
                best_silhouette = silhouette;
                best_k = k;
            }
        }

        log::info!("Optimal number of clusters: {} (silhouette: {:.3})", best_k, best_silhouette);
        Ok(best_k)
    }

    /// Calculate silhouette score for cluster validation
    fn calculate_silhouette_score(&self, data: &Array2<f64>, assignments: &[i32]) -> Result<f64> {
        let n_samples = data.nrows();
        let mut silhouette_scores = Vec::new();

        for i in 0..n_samples {
            let cluster_i = assignments[i];
            
            // Calculate a(i) - average distance to points in same cluster
            let mut same_cluster_distances = Vec::new();
            for j in 0..n_samples {
                if i != j && assignments[j] == cluster_i {
                    let distance = self.calculate_distance(&data.row(i), &data.row(j));
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
                .filter(|&&c| c != cluster_i)
                .copied()
                .collect();

            let mut min_avg_distance = f64::INFINITY;
            for other_cluster in unique_clusters {
                let mut other_cluster_distances = Vec::new();
                for j in 0..n_samples {
                    if assignments[j] == other_cluster {
                        let distance = self.calculate_distance(&data.row(i), &data.row(j));
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

        Ok(silhouette_scores.iter().sum::<f64>() / silhouette_scores.len() as f64)
    }
}

/// Cluster representation for hierarchical clustering
#[derive(Debug, Clone)]
struct Cluster {
    id: usize,
    members: Vec<usize>,
    size: usize,
}

/// Cluster merge information for dendrogram
#[derive(Debug, Clone)]
struct ClusterMerge {
    cluster_a: usize,
    cluster_b: usize,
    distance: f64,
    new_cluster: usize,
    size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_hierarchical_creation() {
        let clusterer = HierarchicalClusterer::new();
        assert_eq!(clusterer.linkage_method, "ward");
        assert_eq!(clusterer.distance_metric, "euclidean");
    }

    #[test]
    fn test_distance_matrix() {
        let clusterer = HierarchicalClusterer::new();
        let data = Array2::from_shape_vec((3, 2), vec![
            0.0, 0.0,
            1.0, 1.0,
            2.0, 2.0,
        ]).unwrap();

        let distance_matrix = clusterer.build_distance_matrix(&data).unwrap();
        assert_eq!(distance_matrix.shape(), &[3, 3]);
        assert_eq!(distance_matrix[[0, 0]], 0.0);
        assert!((distance_matrix[[0, 1]] - (2.0_f64).sqrt()).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_hierarchical_clustering() {
        let mut clusterer = HierarchicalClusterer::new();
        
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

        // Check that we have 2 clusters
        let unique_clusters: std::collections::HashSet<i32> = assignments.iter().copied().collect();
        assert_eq!(unique_clusters.len(), 2);
    }

    #[test]
    fn test_renumber_clusters() {
        let clusterer = HierarchicalClusterer::new();
        let mut assignments = vec![5, 5, 10, 10, 15];
        
        clusterer.renumber_clusters(&mut assignments);
        
        // Should be renumbered to 0, 0, 1, 1, 2
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
        assert_ne!(assignments[2], assignments[4]);
    }

    #[test]
    fn test_linkage_methods() {
        let clusterer = HierarchicalClusterer::new();
        let data = Array2::from_shape_vec((3, 2), vec![
            0.0, 0.0,
            1.0, 1.0,
            2.0, 2.0,
        ]).unwrap();
        
        let distance_matrix = clusterer.build_distance_matrix(&data).unwrap();
        
        let cluster_a = Cluster {
            id: 0,
            members: vec![0],
            size: 1,
        };
        
        let cluster_b = Cluster {
            id: 1,
            members: vec![1, 2],
            size: 2,
        };
        
        let single_distance = clusterer.single_linkage(&cluster_a, &cluster_b, &distance_matrix).unwrap();
        let complete_distance = clusterer.complete_linkage(&cluster_a, &cluster_b, &distance_matrix).unwrap();
        let average_distance = clusterer.average_linkage(&cluster_a, &cluster_b, &distance_matrix).unwrap();
        
        // Single linkage should give minimum distance
        // Complete linkage should give maximum distance
        // Average should be between them
        assert!(single_distance <= average_distance);
        assert!(average_distance <= complete_distance);
    }
}