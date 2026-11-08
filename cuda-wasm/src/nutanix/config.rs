//! Configuration types for Nutanix platform integration
//!
//! Provides strongly-typed configuration for Prism Central connections,
//! GPU resource descriptions, and workload deployment settings.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::time::Duration;

/// GPU vendor enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GpuVendor {
    /// NVIDIA GPUs (Tesla, A100, H100, etc.)
    Nvidia,
    /// AMD GPUs (Instinct MI series, etc.)
    Amd,
    /// Intel GPUs (Data Center GPU Max, Arc, etc.)
    Intel,
    /// Unknown or unrecognized vendor
    Unknown(String),
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuVendor::Nvidia => write!(f, "NVIDIA"),
            GpuVendor::Amd => write!(f, "AMD"),
            GpuVendor::Intel => write!(f, "Intel"),
            GpuVendor::Unknown(name) => write!(f, "{}", name),
        }
    }
}

/// Known GPU model identifiers
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GpuModel {
    // NVIDIA models
    /// NVIDIA A100 (40GB or 80GB)
    NvidiaA100,
    /// NVIDIA H100
    NvidiaH100,
    /// NVIDIA L40S
    NvidiaL40S,
    /// NVIDIA T4
    NvidiaT4,
    /// NVIDIA V100
    NvidiaV100,

    // AMD models
    /// AMD Instinct MI250X
    AmdMI250X,
    /// AMD Instinct MI300X
    AmdMI300X,
    /// AMD Instinct MI210
    AmdMI210,

    // Intel models
    /// Intel Data Center GPU Max 1550
    IntelMax1550,

    /// Other / unrecognized model
    Other(String),
}

impl GpuModel {
    /// Parse a GPU model from a device name string.
    ///
    /// Matches known model identifiers (case-insensitive) and returns the
    /// corresponding enum variant, or `GpuModel::Other` if unrecognized.
    pub fn from_name(name: &str) -> Self {
        let upper = name.to_uppercase();
        if upper.contains("A100") {
            GpuModel::NvidiaA100
        } else if upper.contains("H100") {
            GpuModel::NvidiaH100
        } else if upper.contains("L40") {
            GpuModel::NvidiaL40S
        } else if upper.contains("T4") && !upper.contains("RTX") {
            GpuModel::NvidiaT4
        } else if upper.contains("V100") {
            GpuModel::NvidiaV100
        } else if upper.contains("MI250") {
            GpuModel::AmdMI250X
        } else if upper.contains("MI300") {
            GpuModel::AmdMI300X
        } else if upper.contains("MI210") {
            GpuModel::AmdMI210
        } else if upper.contains("MAX") && upper.contains("1550") {
            GpuModel::IntelMax1550
        } else {
            GpuModel::Other(name.to_string())
        }
    }
}

impl std::fmt::Display for GpuModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuModel::NvidiaA100 => write!(f, "NVIDIA A100"),
            GpuModel::NvidiaH100 => write!(f, "NVIDIA H100"),
            GpuModel::NvidiaL40S => write!(f, "NVIDIA L40S"),
            GpuModel::NvidiaT4 => write!(f, "NVIDIA T4"),
            GpuModel::NvidiaV100 => write!(f, "NVIDIA V100"),
            GpuModel::AmdMI250X => write!(f, "AMD Instinct MI250X"),
            GpuModel::AmdMI300X => write!(f, "AMD Instinct MI300X"),
            GpuModel::AmdMI210 => write!(f, "AMD Instinct MI210"),
            GpuModel::IntelMax1550 => write!(f, "Intel Data Center GPU Max 1550"),
            GpuModel::Other(name) => write!(f, "{}", name),
        }
    }
}

/// Nutanix Prism Central connection configuration
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NutanixConfig {
    /// Prism Central base URL (e.g., "https://prism-central.example.com:9440")
    pub base_url: String,

    /// API key or bearer token for authentication
    pub api_key: String,

    /// Optional username for basic auth (used if api_key is empty)
    #[cfg_attr(feature = "serde", serde(default))]
    pub username: Option<String>,

    /// Optional password for basic auth
    #[cfg_attr(feature = "serde", serde(default))]
    pub password: Option<String>,

    /// HTTP request timeout
    #[cfg_attr(feature = "serde", serde(with = "duration_serde", default = "default_timeout"))]
    pub timeout: Duration,

    /// Whether to verify TLS certificates (disable for self-signed certs in labs)
    #[cfg_attr(feature = "serde", serde(default = "default_true"))]
    pub verify_ssl: bool,

    /// Prism Central API version to use (default: "v3")
    #[cfg_attr(feature = "serde", serde(default = "default_api_version"))]
    pub api_version: String,
}

impl std::fmt::Debug for NutanixConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NutanixConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &"[REDACTED]")
            .field("username", &self.username)
            .field("password", &"[REDACTED]")
            .field("timeout", &self.timeout)
            .field("verify_ssl", &self.verify_ssl)
            .field("api_version", &self.api_version)
            .finish()
    }
}

fn default_timeout() -> Duration {
    Duration::from_secs(30)
}

fn default_true() -> bool {
    true
}

fn default_api_version() -> String {
    "v3".to_string()
}

impl Default for NutanixConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: String::new(),
            username: None,
            password: None,
            timeout: default_timeout(),
            verify_ssl: true,
            api_version: default_api_version(),
        }
    }
}

impl NutanixConfig {
    /// Create a new NutanixConfig with the given base URL and API key
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Create a config using basic authentication
    pub fn with_basic_auth(
        base_url: impl Into<String>,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: String::new(),
            username: Some(username.into()),
            password: Some(password.into()),
            ..Default::default()
        }
    }

    /// Set the HTTP timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Disable SSL verification (for development/lab environments)
    pub fn with_insecure_ssl(mut self) -> Self {
        self.verify_ssl = false;
        self
    }

    /// Construct the full API endpoint URL
    pub fn api_url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        format!("{}/api/nutanix/{}/{}", base, self.api_version, path.trim_start_matches('/'))
    }
}

/// Information about a single GPU device on a host
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuInfo {
    /// GPU vendor
    pub vendor: GpuVendor,

    /// GPU model
    pub model: GpuModel,

    /// GPU device ID (PCI bus ID or Nutanix UUID)
    pub device_id: String,

    /// Total GPU memory in bytes
    pub memory_bytes: u64,

    /// Number of compute units / SMs / CUs
    pub compute_units: u32,

    /// Whether the GPU is currently assigned to a VM
    pub assigned: bool,

    /// VM UUID if assigned
    #[cfg_attr(feature = "serde", serde(default))]
    pub assigned_vm: Option<String>,

    /// GPU mode: passthrough or vGPU
    #[cfg_attr(feature = "serde", serde(default = "default_gpu_mode"))]
    pub mode: String,

    /// NUMA node for the GPU
    #[cfg_attr(feature = "serde", serde(default))]
    pub numa_node: Option<u32>,
}

fn default_gpu_mode() -> String {
    "passthrough".to_string()
}

/// Capabilities of a Nutanix host
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HostCapabilities {
    /// Host UUID
    pub host_id: String,

    /// Host name
    pub host_name: String,

    /// CPU architecture (x86_64, aarch64)
    pub cpu_arch: String,

    /// Total CPU cores
    pub cpu_cores: u32,

    /// Total RAM in bytes
    pub ram_bytes: u64,

    /// Whether the host has NVIDIA GPUs
    pub has_nvidia: bool,

    /// Whether the host has AMD GPUs
    pub has_amd: bool,

    /// Whether the host is ARM-based
    pub is_arm: bool,

    /// List of GPUs on this host
    pub gpus: Vec<GpuInfo>,

    /// Hypervisor type (AHV, ESXi)
    pub hypervisor: String,

    /// AOS version
    pub aos_version: String,

    /// Whether the host supports GPU passthrough
    pub gpu_passthrough_supported: bool,

    /// Whether the host supports vGPU
    pub vgpu_supported: bool,

    /// Additional host metadata
    #[cfg_attr(feature = "serde", serde(default))]
    pub metadata: HashMap<String, String>,
}

/// A node in the cluster that has GPU resources
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuNode {
    /// Nutanix host UUID
    pub host_id: String,

    /// Host display name
    pub host_name: String,

    /// Cluster UUID this host belongs to
    pub cluster_id: String,

    /// Cluster name
    pub cluster_name: String,

    /// IP address of the host
    pub ip_address: String,

    /// Available (unassigned) GPUs
    pub available_gpus: Vec<GpuInfo>,

    /// Total GPUs (including assigned)
    pub total_gpus: Vec<GpuInfo>,

    /// Host capabilities
    pub capabilities: HostCapabilities,
}

impl GpuNode {
    /// Count of available GPUs by vendor
    pub fn available_gpu_count(&self, vendor: &GpuVendor) -> usize {
        self.available_gpus
            .iter()
            .filter(|g| &g.vendor == vendor)
            .count()
    }

    /// Total available GPU memory in bytes
    pub fn available_gpu_memory(&self) -> u64 {
        self.available_gpus.iter().map(|g| g.memory_bytes).sum()
    }

    /// Check if this node has at least N available GPUs of the given vendor
    pub fn has_available_gpus(&self, vendor: &GpuVendor, count: usize) -> bool {
        self.available_gpu_count(vendor) >= count
    }
}

/// Aggregated GPU information for an entire cluster
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuClusterSummary {
    /// Cluster UUID
    pub cluster_id: String,

    /// Cluster name
    pub cluster_name: String,

    /// Total number of GPU-equipped hosts
    pub gpu_host_count: u32,

    /// Total GPUs across all hosts
    pub total_gpu_count: u32,

    /// Available (unassigned) GPUs across all hosts
    pub available_gpu_count: u32,

    /// GPU counts per vendor
    pub gpus_by_vendor: HashMap<String, u32>,

    /// GPU counts per model
    pub gpus_by_model: HashMap<String, u32>,

    /// Total GPU memory in bytes across the cluster
    pub total_gpu_memory_bytes: u64,

    /// Available GPU memory in bytes
    pub available_gpu_memory_bytes: u64,

    /// Individual GPU nodes in this cluster
    pub nodes: Vec<GpuNode>,
}

impl GpuClusterSummary {
    /// Get the dominant GPU vendor in this cluster
    pub fn dominant_vendor(&self) -> Option<String> {
        self.gpus_by_vendor
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(vendor, _)| vendor.clone())
    }

    /// Check if the cluster has mixed GPU vendors
    pub fn is_multi_vendor(&self) -> bool {
        self.gpus_by_vendor.len() > 1
    }
}

/// Configuration for deploying cuda-wasm workloads on Nutanix
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DeploymentConfig {
    /// Deployment name
    pub name: String,

    /// Kubernetes namespace
    #[cfg_attr(feature = "serde", serde(default = "default_namespace"))]
    pub namespace: String,

    /// Container image for the cuda-wasm workload
    pub image: String,

    /// Number of replicas
    #[cfg_attr(feature = "serde", serde(default = "default_replicas"))]
    pub replicas: u32,

    /// GPU vendor to target
    pub gpu_vendor: GpuVendor,

    /// Number of GPUs per pod
    #[cfg_attr(feature = "serde", serde(default = "default_gpu_count"))]
    pub gpus_per_pod: u32,

    /// CPU request per pod (millicores, e.g., "1000m" = 1 core)
    #[cfg_attr(feature = "serde", serde(default = "default_cpu_request"))]
    pub cpu_request: String,

    /// CPU limit per pod
    #[cfg_attr(feature = "serde", serde(default = "default_cpu_limit"))]
    pub cpu_limit: String,

    /// Memory request per pod
    #[cfg_attr(feature = "serde", serde(default = "default_mem_request"))]
    pub memory_request: String,

    /// Memory limit per pod
    #[cfg_attr(feature = "serde", serde(default = "default_mem_limit"))]
    pub memory_limit: String,

    /// PVC size for kernel cache storage
    #[cfg_attr(feature = "serde", serde(default = "default_cache_size"))]
    pub kernel_cache_size: String,

    /// Nutanix CSI storage class name
    #[cfg_attr(feature = "serde", serde(default = "default_storage_class"))]
    pub storage_class: String,

    /// Service port for the workload
    #[cfg_attr(feature = "serde", serde(default = "default_service_port"))]
    pub service_port: u16,

    /// Enable horizontal pod autoscaler
    #[cfg_attr(feature = "serde", serde(default))]
    pub enable_hpa: bool,

    /// HPA minimum replicas
    #[cfg_attr(feature = "serde", serde(default = "default_hpa_min"))]
    pub hpa_min_replicas: u32,

    /// HPA maximum replicas
    #[cfg_attr(feature = "serde", serde(default = "default_hpa_max"))]
    pub hpa_max_replicas: u32,

    /// HPA target GPU utilization percentage
    #[cfg_attr(feature = "serde", serde(default = "default_hpa_target"))]
    pub hpa_target_gpu_utilization: u32,

    /// Additional environment variables
    #[cfg_attr(feature = "serde", serde(default))]
    pub env_vars: HashMap<String, String>,

    /// Additional labels for the deployment
    #[cfg_attr(feature = "serde", serde(default))]
    pub labels: HashMap<String, String>,

    /// Additional annotations (e.g., NKE-specific)
    #[cfg_attr(feature = "serde", serde(default))]
    pub annotations: HashMap<String, String>,
}

fn default_namespace() -> String {
    "cuda-wasm".to_string()
}
fn default_replicas() -> u32 {
    1
}
fn default_gpu_count() -> u32 {
    1
}
fn default_cpu_request() -> String {
    "1000m".to_string()
}
fn default_cpu_limit() -> String {
    "4000m".to_string()
}
fn default_mem_request() -> String {
    "4Gi".to_string()
}
fn default_mem_limit() -> String {
    "16Gi".to_string()
}
fn default_cache_size() -> String {
    "10Gi".to_string()
}
fn default_storage_class() -> String {
    "nutanix-volume".to_string()
}
fn default_service_port() -> u16 {
    8080
}
fn default_hpa_min() -> u32 {
    1
}
fn default_hpa_max() -> u32 {
    8
}
fn default_hpa_target() -> u32 {
    70
}

impl Default for DeploymentConfig {
    fn default() -> Self {
        Self {
            name: "cuda-wasm-worker".to_string(),
            namespace: default_namespace(),
            image: "cuda-wasm:latest".to_string(),
            replicas: default_replicas(),
            gpu_vendor: GpuVendor::Nvidia,
            gpus_per_pod: default_gpu_count(),
            cpu_request: default_cpu_request(),
            cpu_limit: default_cpu_limit(),
            memory_request: default_mem_request(),
            memory_limit: default_mem_limit(),
            kernel_cache_size: default_cache_size(),
            storage_class: default_storage_class(),
            service_port: default_service_port(),
            enable_hpa: false,
            hpa_min_replicas: default_hpa_min(),
            hpa_max_replicas: default_hpa_max(),
            hpa_target_gpu_utilization: default_hpa_target(),
            env_vars: HashMap::new(),
            labels: HashMap::new(),
            annotations: HashMap::new(),
        }
    }
}

impl DeploymentConfig {
    /// Create a new DeploymentConfig with the given name and image
    pub fn new(name: impl Into<String>, image: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            image: image.into(),
            ..Default::default()
        }
    }

    /// Set the target GPU vendor
    pub fn with_gpu_vendor(mut self, vendor: GpuVendor) -> Self {
        self.gpu_vendor = vendor;
        self
    }

    /// Set the number of GPUs per pod
    pub fn with_gpus(mut self, count: u32) -> Self {
        self.gpus_per_pod = count;
        self
    }

    /// Enable HPA with the given min/max replicas
    pub fn with_hpa(mut self, min: u32, max: u32, target_utilization: u32) -> Self {
        self.enable_hpa = true;
        self.hpa_min_replicas = min;
        self.hpa_max_replicas = max;
        self.hpa_target_gpu_utilization = target_utilization;
        self
    }

    /// Add an NKE-specific annotation
    pub fn with_nke_annotation(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.annotations.insert(key.into(), value.into());
        self
    }
}

/// Serde helper for Duration serialization
#[cfg(feature = "serde")]
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nutanix_config_new() {
        let config = NutanixConfig::new("https://prism.example.com:9440", "my-api-key");
        assert_eq!(config.base_url, "https://prism.example.com:9440");
        assert_eq!(config.api_key, "my-api-key");
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(config.verify_ssl);
    }

    #[test]
    fn test_nutanix_config_api_url() {
        let config = NutanixConfig::new("https://prism.example.com:9440", "key");
        assert_eq!(
            config.api_url("hosts/list"),
            "https://prism.example.com:9440/api/nutanix/v3/hosts/list"
        );
    }

    #[test]
    fn test_deployment_config_default() {
        let config = DeploymentConfig::default();
        assert_eq!(config.namespace, "cuda-wasm");
        assert_eq!(config.replicas, 1);
        assert_eq!(config.gpus_per_pod, 1);
    }

    #[test]
    fn test_deployment_config_builder() {
        let config = DeploymentConfig::new("my-workload", "my-image:v1")
            .with_gpu_vendor(GpuVendor::Amd)
            .with_gpus(2)
            .with_hpa(1, 4, 80);

        assert_eq!(config.name, "my-workload");
        assert_eq!(config.gpu_vendor, GpuVendor::Amd);
        assert_eq!(config.gpus_per_pod, 2);
        assert!(config.enable_hpa);
        assert_eq!(config.hpa_max_replicas, 4);
    }

    #[test]
    fn test_gpu_vendor_display() {
        assert_eq!(GpuVendor::Nvidia.to_string(), "NVIDIA");
        assert_eq!(GpuVendor::Amd.to_string(), "AMD");
        assert_eq!(GpuVendor::Unknown("Custom".into()).to_string(), "Custom");
    }

    #[test]
    fn test_gpu_node_helpers() {
        let node = GpuNode {
            host_id: "host-1".to_string(),
            host_name: "gpu-host-01".to_string(),
            cluster_id: "cluster-1".to_string(),
            cluster_name: "GPU Cluster".to_string(),
            ip_address: "10.0.0.1".to_string(),
            available_gpus: vec![
                GpuInfo {
                    vendor: GpuVendor::Nvidia,
                    model: GpuModel::NvidiaA100,
                    device_id: "gpu-0".into(),
                    memory_bytes: 80 * 1024 * 1024 * 1024,
                    compute_units: 108,
                    assigned: false,
                    assigned_vm: None,
                    mode: "passthrough".into(),
                    numa_node: Some(0),
                },
                GpuInfo {
                    vendor: GpuVendor::Nvidia,
                    model: GpuModel::NvidiaA100,
                    device_id: "gpu-1".into(),
                    memory_bytes: 80 * 1024 * 1024 * 1024,
                    compute_units: 108,
                    assigned: false,
                    assigned_vm: None,
                    mode: "passthrough".into(),
                    numa_node: Some(1),
                },
            ],
            total_gpus: vec![],
            capabilities: HostCapabilities {
                host_id: "host-1".into(),
                host_name: "gpu-host-01".into(),
                cpu_arch: "x86_64".into(),
                cpu_cores: 64,
                ram_bytes: 512 * 1024 * 1024 * 1024,
                has_nvidia: true,
                has_amd: false,
                is_arm: false,
                gpus: vec![],
                hypervisor: "AHV".into(),
                aos_version: "6.7".into(),
                gpu_passthrough_supported: true,
                vgpu_supported: true,
                metadata: HashMap::new(),
            },
        };

        assert_eq!(node.available_gpu_count(&GpuVendor::Nvidia), 2);
        assert_eq!(node.available_gpu_count(&GpuVendor::Amd), 0);
        assert!(node.has_available_gpus(&GpuVendor::Nvidia, 2));
        assert!(!node.has_available_gpus(&GpuVendor::Nvidia, 3));
        assert_eq!(node.available_gpu_memory(), 2 * 80 * 1024 * 1024 * 1024);
    }
}
