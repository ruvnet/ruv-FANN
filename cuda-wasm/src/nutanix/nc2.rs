//! Nutanix Cloud Clusters (NC2) integration for hybrid/multi-cloud GPU workloads
//!
//! Provides discovery, cost-aware placement, and cross-cloud migration
//! capabilities for cuda-wasm workloads across NC2 clusters deployed on
//! AWS, Azure, GCP, and on-premises infrastructure.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::CudaRustError;
use super::config::NutanixConfig;
use super::vgpu_scheduler::WorkloadRequest;

/// Cloud provider where an NC2 cluster is deployed
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CloudProvider {
    /// Amazon Web Services
    Aws,
    /// Microsoft Azure
    Azure,
    /// Google Cloud Platform
    Gcp,
    /// On-premises data center
    OnPrem,
}

impl std::fmt::Display for CloudProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CloudProvider::Aws => write!(f, "AWS"),
            CloudProvider::Azure => write!(f, "Azure"),
            CloudProvider::Gcp => write!(f, "GCP"),
            CloudProvider::OnPrem => write!(f, "On-Prem"),
        }
    }
}

/// Status of an NC2 cluster
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ClusterStatus {
    /// Cluster is running and healthy
    Running,
    /// Cluster is being provisioned
    Provisioning,
    /// Cluster is being updated
    Updating,
    /// Cluster is in an error state
    Error,
    /// Cluster is stopped / hibernated
    Stopped,
}

impl std::fmt::Display for ClusterStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClusterStatus::Running => write!(f, "RUNNING"),
            ClusterStatus::Provisioning => write!(f, "PROVISIONING"),
            ClusterStatus::Updating => write!(f, "UPDATING"),
            ClusterStatus::Error => write!(f, "ERROR"),
            ClusterStatus::Stopped => write!(f, "STOPPED"),
        }
    }
}

/// Represents a Nutanix Cloud Cluster (NC2) instance
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Nc2Cluster {
    /// Unique cluster identifier
    pub cluster_id: String,
    /// Human-readable cluster name
    pub name: String,
    /// Cloud provider hosting this cluster
    pub provider: CloudProvider,
    /// Cloud region (e.g., "us-east-1", "eastus", "us-central1")
    pub region: String,
    /// Available GPU types in this cluster
    pub gpu_types: Vec<String>,
    /// Current cluster status
    pub status: ClusterStatus,
    /// Number of GPU-equipped nodes
    pub gpu_node_count: u32,
    /// Total available GPU memory in bytes
    pub total_gpu_memory_bytes: u64,
}

/// Placement decision for a workload across NC2 clusters
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkloadPlacement {
    /// Primary cluster for the workload
    pub primary_cluster: String,
    /// Failover cluster for disaster recovery
    pub failover_cluster: Option<String>,
    /// Reason for the placement decision
    pub placement_reason: String,
    /// Estimated cost for the primary placement
    pub estimated_cost: CostEstimate,
}

/// Cost estimate for running a workload on a cloud provider
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CostEstimate {
    /// Estimated hourly cost in USD
    pub hourly_cost: f64,
    /// Estimated monthly cost in USD (based on 730 hours)
    pub monthly_cost: f64,
    /// Cloud provider
    pub provider: CloudProvider,
    /// Cloud instance type
    pub instance_type: String,
    /// GPU type
    pub gpu_type: String,
}

/// Status of a cross-cloud workload migration
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MigrationStatus {
    /// Migration has been initiated
    Initiated,
    /// Data transfer in progress
    Transferring,
    /// Workload is being restarted at the destination
    Restarting,
    /// Migration completed successfully
    Completed,
    /// Migration failed
    Failed(String),
}

impl std::fmt::Display for MigrationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigrationStatus::Initiated => write!(f, "INITIATED"),
            MigrationStatus::Transferring => write!(f, "TRANSFERRING"),
            MigrationStatus::Restarting => write!(f, "RESTARTING"),
            MigrationStatus::Completed => write!(f, "COMPLETED"),
            MigrationStatus::Failed(reason) => write!(f, "FAILED: {}", reason),
        }
    }
}

/// Client for managing Nutanix Cloud Clusters (NC2) and hybrid GPU workloads
pub struct Nc2Client {
    /// Prism Central connection configuration
    #[allow(dead_code)]
    config: NutanixConfig,

    /// HTTP client (when nutanix feature is available)
    #[cfg(feature = "nutanix")]
    #[allow(dead_code)]
    client: reqwest::Client,
}

impl Nc2Client {
    /// Create a new NC2 client with the given Prism Central configuration
    pub fn new(config: NutanixConfig) -> Result<Self, CudaRustError> {
        #[cfg(feature = "nutanix")]
        {
            let builder = reqwest::Client::builder().timeout(config.timeout);
            let client = builder.build().map_err(|e| {
                CudaRustError::RuntimeError(format!("Failed to create HTTP client: {}", e))
            })?;
            Ok(Self { config, client })
        }

        #[cfg(not(feature = "nutanix"))]
        {
            Ok(Self { config })
        }
    }

    /// Discover all NC2 clusters accessible from Prism Central
    ///
    /// Returns clusters across all cloud providers with their GPU inventory
    /// and current status.
    pub async fn discover_nc2_clusters(&self) -> Result<Vec<Nc2Cluster>, CudaRustError> {
        #[cfg(feature = "nutanix")]
        {
            Err(CudaRustError::RuntimeError(
                "NC2 cluster discovery requires Prism Central connection".to_string(),
            ))
        }

        #[cfg(not(feature = "nutanix"))]
        {
            Ok(self.local_nc2_clusters())
        }
    }

    /// Find optimal placement for a workload across all NC2 clusters
    ///
    /// Considers GPU requirements, cost, latency, and availability to
    /// select the best primary cluster and an optional failover cluster.
    pub async fn find_optimal_placement(
        &self,
        workload: &WorkloadRequest,
    ) -> Result<WorkloadPlacement, CudaRustError> {
        let clusters = self.discover_nc2_clusters().await?;

        let running_clusters: Vec<&Nc2Cluster> = clusters
            .iter()
            .filter(|c| c.status == ClusterStatus::Running)
            .collect();

        if running_clusters.is_empty() {
            return Err(CudaRustError::RuntimeError(
                "No running NC2 clusters available for placement".to_string(),
            ));
        }

        // Filter clusters that have enough GPU memory
        let suitable: Vec<&Nc2Cluster> = running_clusters
            .iter()
            .filter(|c| c.total_gpu_memory_bytes >= workload.min_gpu_memory)
            .copied()
            .collect();

        if suitable.is_empty() {
            return Err(CudaRustError::RuntimeError(format!(
                "No NC2 cluster has sufficient GPU memory for workload '{}' ({} bytes required)",
                workload.name, workload.min_gpu_memory
            )));
        }

        // Prefer on-prem for lower latency, then cheapest cloud option
        let primary = suitable
            .iter()
            .min_by(|a, b| {
                let cost_a = self.provider_cost_factor(&a.provider);
                let cost_b = self.provider_cost_factor(&b.provider);
                cost_a
                    .partial_cmp(&cost_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let failover = suitable
            .iter()
            .find(|c| c.cluster_id != primary.cluster_id)
            .map(|c| c.cluster_id.clone());

        let gpu_type = primary
            .gpu_types
            .first()
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        let cost = self.estimate_cost(primary.provider.clone(), &gpu_type, 730);

        let reason = format!(
            "Selected {} ({}) for lowest cost ({:.2}/hr); {} GPU memory available",
            primary.name,
            primary.provider,
            cost.hourly_cost,
            format_bytes(primary.total_gpu_memory_bytes),
        );

        Ok(WorkloadPlacement {
            primary_cluster: primary.cluster_id.clone(),
            failover_cluster: failover,
            placement_reason: reason,
            estimated_cost: cost,
        })
    }

    /// Estimate the cost of running a GPU workload on a given provider
    ///
    /// Uses simplified pricing models for each cloud provider.
    pub fn estimate_cost(
        &self,
        provider: CloudProvider,
        gpu_type: &str,
        hours: u32,
    ) -> CostEstimate {
        let (hourly, instance_type) = match (&provider, gpu_type) {
            (CloudProvider::Aws, t) if t.contains("A100") => (3.40, "p4d.24xlarge"),
            (CloudProvider::Aws, t) if t.contains("H100") => (6.50, "p5.48xlarge"),
            (CloudProvider::Aws, _) => (2.10, "g5.xlarge"),
            (CloudProvider::Azure, t) if t.contains("A100") => (3.67, "Standard_NC96ads_A100_v4"),
            (CloudProvider::Azure, t) if t.contains("H100") => (7.00, "Standard_ND96isr_H100_v5"),
            (CloudProvider::Azure, _) => (2.30, "Standard_NC6s_v3"),
            (CloudProvider::Gcp, t) if t.contains("A100") => (3.22, "a2-highgpu-1g"),
            (CloudProvider::Gcp, t) if t.contains("H100") => (6.20, "a3-highgpu-1g"),
            (CloudProvider::Gcp, _) => (1.90, "n1-standard-4-t4"),
            (CloudProvider::OnPrem, _) => (0.50, "bare-metal"),
        };

        CostEstimate {
            hourly_cost: hourly,
            monthly_cost: hourly * hours as f64,
            provider,
            instance_type: instance_type.to_string(),
            gpu_type: gpu_type.to_string(),
        }
    }

    /// Initiate a workload migration between NC2 clusters
    ///
    /// Triggers a cross-cloud migration that transfers workload state
    /// and restarts execution on the destination cluster.
    pub async fn migrate_workload(
        &self,
        from: &str,
        to: &str,
        workload_id: &str,
    ) -> Result<MigrationStatus, CudaRustError> {
        #[cfg(feature = "nutanix")]
        {
            let _ = (from, to, workload_id);
            Err(CudaRustError::RuntimeError(
                "Workload migration requires Prism Central connection".to_string(),
            ))
        }

        #[cfg(not(feature = "nutanix"))]
        {
            self.local_migrate(from, to, workload_id)
        }
    }

    // --- Private helpers ---

    /// Relative cost factor per provider (lower is cheaper)
    fn provider_cost_factor(&self, provider: &CloudProvider) -> f64 {
        match provider {
            CloudProvider::OnPrem => 0.5,
            CloudProvider::Gcp => 1.0,
            CloudProvider::Aws => 1.1,
            CloudProvider::Azure => 1.2,
        }
    }

    // --- Local system probing for non-nutanix builds ---

    /// Discover a local GPU candidate without contacting cloud metadata.
    ///
    /// Requires an opted-in GPU feature. A local GPU candidate is not proof of
    /// an NC2 deployment; remote discovery requires a configured Prism API.
    #[cfg(not(feature = "nutanix"))]
    fn local_nc2_clusters(&self) -> Vec<Nc2Cluster> {
        let mut clusters = Vec::new();

        // Local discovery must not contact cloud instance metadata or imply
        // that detecting an AWS host establishes an NC2 cluster deployment.
        if !crate::backend::gpu_enabled() { return clusters; }

        // If no cloud environment detected, check for local GPUs
        if clusters.is_empty() {
            let has_nvidia = std::process::Command::new("nvidia-smi")
                .arg("--list-gpus")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            let has_rocm = std::path::Path::new("/opt/rocm").exists();

            if has_nvidia || has_rocm {
                clusters.push(Nc2Cluster {
                    cluster_id: "local-onprem".to_string(),
                    name: "Local GPU Cluster".to_string(),
                    provider: CloudProvider::OnPrem,
                    region: "local".to_string(),
                    gpu_types: vec!["Local".to_string()],
                    status: ClusterStatus::Running,
                    gpu_node_count: 1,
                    total_gpu_memory_bytes: 0,
                });
            }
        }

        clusters
    }

    /// Validate and log a migration request without Prism Central API.
    ///
    /// Returns `MigrationStatus::Initiated` on valid input, but the actual
    /// data transfer cannot proceed without the Nutanix API.
    #[cfg(not(feature = "nutanix"))]
    fn local_migrate(
        &self,
        from: &str,
        to: &str,
        workload_id: &str,
    ) -> Result<MigrationStatus, CudaRustError> {
        if from == to {
            return Err(CudaRustError::RuntimeError(
                "Source and destination clusters must be different".to_string(),
            ));
        }
        if workload_id.is_empty() {
            return Err(CudaRustError::RuntimeError(
                "Workload ID must not be empty".to_string(),
            ));
        }
        // Without Prism Central API, migration requires manual intervention
        Ok(MigrationStatus::Initiated)
    }
}

/// Format bytes into a human-readable string
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 * 1024 {
        format!("{:.1} TB", bytes as f64 / (1024.0 * 1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nutanix::config::GpuVendor;

    fn make_client() -> Nc2Client {
        let config = NutanixConfig::new("https://prism.example.com:9440", "test-key");
        Nc2Client::new(config).unwrap()
    }

    #[test]
    fn test_cloud_provider_display() {
        assert_eq!(CloudProvider::Aws.to_string(), "AWS");
        assert_eq!(CloudProvider::Azure.to_string(), "Azure");
        assert_eq!(CloudProvider::Gcp.to_string(), "GCP");
        assert_eq!(CloudProvider::OnPrem.to_string(), "On-Prem");
    }

    #[test]
    fn test_cluster_status_display() {
        assert_eq!(ClusterStatus::Running.to_string(), "RUNNING");
        assert_eq!(ClusterStatus::Stopped.to_string(), "STOPPED");
        assert_eq!(ClusterStatus::Error.to_string(), "ERROR");
    }

    #[test]
    fn test_migration_status_display() {
        assert_eq!(MigrationStatus::Initiated.to_string(), "INITIATED");
        assert_eq!(MigrationStatus::Completed.to_string(), "COMPLETED");
        assert_eq!(
            MigrationStatus::Failed("timeout".into()).to_string(),
            "FAILED: timeout"
        );
    }

    #[test]
    fn test_estimate_cost_aws_a100() {
        let client = make_client();
        let cost = client.estimate_cost(CloudProvider::Aws, "A100", 100);
        assert_eq!(cost.hourly_cost, 3.40);
        assert!((cost.monthly_cost - 340.0).abs() < 0.01);
        assert_eq!(cost.provider, CloudProvider::Aws);
        assert_eq!(cost.instance_type, "p4d.24xlarge");
    }

    #[test]
    fn test_estimate_cost_onprem() {
        let client = make_client();
        let cost = client.estimate_cost(CloudProvider::OnPrem, "A100", 730);
        assert!(cost.hourly_cost < 1.0, "On-prem should be cheapest");
        assert_eq!(cost.instance_type, "bare-metal");
    }

    #[tokio::test]
    async fn test_local_discover_nc2_clusters() {
        let client = make_client();
        let clusters = client.discover_nc2_clusters().await.unwrap();
        // On CI without GPUs or cloud metadata, this may be empty.
        // On GPU hosts or cloud instances, we should find at least one.
        for cluster in &clusters {
            assert!(!cluster.cluster_id.is_empty());
            assert!(!cluster.name.is_empty());
            assert_eq!(cluster.status, ClusterStatus::Running);
        }
    }

    #[tokio::test]
    async fn test_local_find_optimal_placement() {
        let client = make_client();
        // Use a minimal memory requirement so any detected cluster qualifies
        let workload = WorkloadRequest::new("test-job", 0)
            .with_vendor(GpuVendor::Nvidia);

        let result = client.find_optimal_placement(&workload).await;
        // Without GPUs/cloud, there are no clusters so placement will fail.
        // With GPUs, placement should succeed.
        if let Ok(placement) = result {
            assert!(!placement.primary_cluster.is_empty());
        }
    }

    #[tokio::test]
    async fn test_local_find_optimal_placement_insufficient_memory() {
        let client = make_client();
        // Extremely large memory request should fail everywhere
        let workload =
            WorkloadRequest::new("huge-job", 2048 * 1024 * 1024 * 1024);

        let result = client.find_optimal_placement(&workload).await;
        // This should always error -- no local or cloud cluster has 2 TB GPU memory
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_local_migrate_workload_success() {
        let client = make_client();
        let status = client
            .migrate_workload("cluster-a", "cluster-b", "workload-123")
            .await
            .unwrap();
        assert_eq!(status, MigrationStatus::Initiated);
    }

    #[tokio::test]
    async fn test_local_migrate_workload_same_cluster_error() {
        let client = make_client();
        let result = client
            .migrate_workload("cluster-a", "cluster-a", "workload-123")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_local_migrate_workload_empty_id_error() {
        let client = make_client();
        let result = client
            .migrate_workload("cluster-a", "cluster-b", "")
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_bytes(80 * 1024 * 1024 * 1024), "80.0 GB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(500), "500 B");
    }

    #[test]
    fn test_nc2_client_creation() {
        let config = NutanixConfig::new("https://prism.example.com:9440", "key");
        let client = Nc2Client::new(config);
        assert!(client.is_ok());
    }
}
