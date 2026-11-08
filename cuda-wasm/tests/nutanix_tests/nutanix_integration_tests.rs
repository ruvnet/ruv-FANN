//! Tests for Nutanix integration
//!
//! Since the Nutanix submodules (config, discovery, deployment) are declared but
//! not yet implemented, these tests exercise the configuration and manifest
//! generation patterns using serde serialization and string-based YAML validation.
//! They serve as specification-level tests for the expected Nutanix integration API.

#[cfg(all(test, feature = "serde"))]
mod tests {
    use serde::{Deserialize, Serialize};

    // ---------------------------------------------------------------
    // Local config types matching the expected Nutanix integration API
    // ---------------------------------------------------------------

    /// Nutanix cluster connection configuration
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct NutanixConfig {
        /// Prism Central endpoint URL
        prism_central_url: String,
        /// Username for authentication
        username: String,
        /// Credential (stored securely in production)
        credential: String,
        /// Target cluster name
        cluster_name: String,
        /// Whether to use HTTPS
        use_tls: bool,
        /// Connection timeout in seconds
        timeout_seconds: u32,
    }

    impl Default for NutanixConfig {
        fn default() -> Self {
            Self {
                prism_central_url: "https://prism.example.com:9440".to_string(),
                username: "admin".to_string(),
                credential: "".to_string(),
                cluster_name: "gpu-cluster-01".to_string(),
                use_tls: true,
                timeout_seconds: 30,
            }
        }
    }

    /// GPU vendor classification
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    enum GpuVendor {
        Nvidia,
        Amd,
        Intel,
        Unknown(String),
    }

    /// GPU model identifier
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct GpuModel {
        vendor: GpuVendor,
        name: String,
        vram_mb: u32,
        compute_capability: Option<String>,
    }

    /// GPU information for a node
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct GpuInfo {
        model: GpuModel,
        count: u32,
        driver_version: String,
    }

    /// GPU node discovered via Nutanix API
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct GpuNode {
        node_name: String,
        ip_address: String,
        gpu_info: Vec<GpuInfo>,
        total_cpu_cores: u32,
        total_memory_mb: u64,
        kubernetes_labels: std::collections::HashMap<String, String>,
    }

    /// GPU cluster summary
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct GpuClusterSummary {
        total_nodes: usize,
        total_gpus: usize,
        gpu_nodes: Vec<GpuNode>,
        vendor_breakdown: std::collections::HashMap<String, usize>,
    }

    /// Deployment configuration for cuda-wasm workloads
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct DeploymentConfig {
        name: String,
        namespace: String,
        replicas: u32,
        gpu_count: u32,
        gpu_vendor_preference: Option<GpuVendor>,
        memory_limit_mb: u64,
        cpu_limit_millicores: u32,
        image: String,
        env_vars: std::collections::HashMap<String, String>,
    }

    impl Default for DeploymentConfig {
        fn default() -> Self {
            Self {
                name: "cuda-wasm-workload".to_string(),
                namespace: "gpu-workloads".to_string(),
                replicas: 1,
                gpu_count: 1,
                gpu_vendor_preference: None,
                memory_limit_mb: 4096,
                cpu_limit_millicores: 2000,
                image: "cuda-wasm:latest".to_string(),
                env_vars: std::collections::HashMap::new(),
            }
        }
    }

    // ---------------------------------------------------------------
    // Helper: Generate Kubernetes deployment YAML
    // ---------------------------------------------------------------
    fn generate_deployment_yaml(config: &DeploymentConfig) -> String {
        let gpu_resource = match &config.gpu_vendor_preference {
            Some(GpuVendor::Nvidia) => "nvidia.com/gpu",
            Some(GpuVendor::Amd) => "amd.com/gpu",
            Some(GpuVendor::Intel) => "gpu.intel.com/i915",
            _ => "nvidia.com/gpu",
        };

        let env_section = config.env_vars.iter()
            .map(|(k, v)| format!("        - name: {}\n          value: \"{}\"", k, v))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"apiVersion: apps/v1
kind: Deployment
metadata:
  name: {}
  namespace: {}
spec:
  replicas: {}
  selector:
    matchLabels:
      app: {}
  template:
    metadata:
      labels:
        app: {}
    spec:
      containers:
      - name: cuda-wasm
        image: {}
        resources:
          limits:
            memory: "{}Mi"
            cpu: "{}m"
            {}: "{}"
          requests:
            memory: "{}Mi"
            cpu: "{}m"
        env:
{}
      nodeSelector:
        gpu: "true"
"#,
            config.name,
            config.namespace,
            config.replicas,
            config.name,
            config.name,
            config.image,
            config.memory_limit_mb,
            config.cpu_limit_millicores,
            gpu_resource,
            config.gpu_count,
            config.memory_limit_mb / 2,
            config.cpu_limit_millicores / 2,
            env_section,
        )
    }

    // ---------------------------------------------------------------
    // Helper: Select GPU nodes by vendor
    // ---------------------------------------------------------------
    fn select_gpu_nodes<'a>(nodes: &'a [GpuNode], vendor: &GpuVendor) -> Vec<&'a GpuNode> {
        nodes.iter().filter(|node| {
            node.gpu_info.iter().any(|gpu| &gpu.model.vendor == vendor)
        }).collect()
    }

    // ---------------------------------------------------------------
    // Test 1: NutanixConfig serialization/deserialization
    // ---------------------------------------------------------------
    #[test]
    fn test_nutanix_config_serde() {
        let config = NutanixConfig {
            prism_central_url: "https://10.0.0.1:9440".to_string(),
            username: "admin".to_string(),
            credential: "secret123".to_string(),
            cluster_name: "prod-gpu-cluster".to_string(),
            use_tls: true,
            timeout_seconds: 60,
        };

        // Serialize
        let json = serde_json::to_string_pretty(&config).expect("Should serialize");
        assert!(json.contains("10.0.0.1:9440"));
        assert!(json.contains("prod-gpu-cluster"));

        // Deserialize
        let deserialized: NutanixConfig = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(config, deserialized);
    }

    // ---------------------------------------------------------------
    // Test 2: NutanixConfig defaults
    // ---------------------------------------------------------------
    #[test]
    fn test_nutanix_config_defaults() {
        let config = NutanixConfig::default();
        assert!(config.use_tls);
        assert_eq!(config.timeout_seconds, 30);
        assert!(!config.prism_central_url.is_empty());
        assert!(!config.cluster_name.is_empty());
    }

    // ---------------------------------------------------------------
    // Test 3: Deployment YAML generation produces valid structure
    // ---------------------------------------------------------------
    #[test]
    fn test_deployment_yaml_generation() {
        let config = DeploymentConfig {
            name: "my-cuda-job".to_string(),
            namespace: "gpu-ns".to_string(),
            replicas: 3,
            gpu_count: 2,
            gpu_vendor_preference: Some(GpuVendor::Nvidia),
            memory_limit_mb: 8192,
            cpu_limit_millicores: 4000,
            image: "registry.example.com/cuda-wasm:v1.0".to_string(),
            env_vars: {
                let mut m = std::collections::HashMap::new();
                m.insert("CUDA_VISIBLE_DEVICES".to_string(), "0,1".to_string());
                m
            },
        };

        let yaml = generate_deployment_yaml(&config);

        // Validate key YAML fields
        assert!(yaml.contains("apiVersion: apps/v1"), "Should have apiVersion");
        assert!(yaml.contains("kind: Deployment"), "Should be a Deployment");
        assert!(yaml.contains("name: my-cuda-job"), "Should have correct name");
        assert!(yaml.contains("namespace: gpu-ns"), "Should have correct namespace");
        assert!(yaml.contains("replicas: 3"), "Should have correct replicas");
        assert!(yaml.contains("nvidia.com/gpu"), "Should reference nvidia GPU resource");
        assert!(yaml.contains("\"2\""), "Should request 2 GPUs");
        assert!(yaml.contains("registry.example.com/cuda-wasm:v1.0"), "Should have correct image");
        assert!(yaml.contains("CUDA_VISIBLE_DEVICES"), "Should include env vars");
        assert!(yaml.contains("nodeSelector"), "Should have nodeSelector");
    }

    // ---------------------------------------------------------------
    // Test 4: Deployment YAML with AMD GPUs
    // ---------------------------------------------------------------
    #[test]
    fn test_deployment_yaml_amd_gpu() {
        let config = DeploymentConfig {
            gpu_vendor_preference: Some(GpuVendor::Amd),
            ..DeploymentConfig::default()
        };

        let yaml = generate_deployment_yaml(&config);
        assert!(
            yaml.contains("amd.com/gpu"),
            "AMD deployment should use amd.com/gpu resource. Got:\n{}",
            yaml
        );
    }

    // ---------------------------------------------------------------
    // Test 5: GPU node discovery response parsing
    // ---------------------------------------------------------------
    #[test]
    fn test_gpu_node_discovery_parsing() {
        let json_response = r#"
        {
            "node_name": "gpu-node-01",
            "ip_address": "10.0.1.10",
            "gpu_info": [
                {
                    "model": {
                        "vendor": "Nvidia",
                        "name": "A100",
                        "vram_mb": 81920,
                        "compute_capability": "8.0"
                    },
                    "count": 4,
                    "driver_version": "535.129.03"
                }
            ],
            "total_cpu_cores": 64,
            "total_memory_mb": 524288,
            "kubernetes_labels": {
                "gpu": "true",
                "gpu-type": "a100"
            }
        }
        "#;

        let node: GpuNode = serde_json::from_str(json_response).expect("Should parse GPU node");

        assert_eq!(node.node_name, "gpu-node-01");
        assert_eq!(node.ip_address, "10.0.1.10");
        assert_eq!(node.gpu_info.len(), 1);
        assert_eq!(node.gpu_info[0].count, 4);
        assert_eq!(node.gpu_info[0].model.vendor, GpuVendor::Nvidia);
        assert_eq!(node.gpu_info[0].model.name, "A100");
        assert_eq!(node.gpu_info[0].model.vram_mb, 81920);
        assert_eq!(node.total_cpu_cores, 64);
        assert_eq!(node.total_memory_mb, 524288);
        assert_eq!(node.kubernetes_labels.get("gpu-type"), Some(&"a100".to_string()));
    }

    // ---------------------------------------------------------------
    // Test 6: Multi-vendor GPU node selection
    // ---------------------------------------------------------------
    #[test]
    fn test_multi_vendor_gpu_node_selection() {
        let nodes = vec![
            GpuNode {
                node_name: "nvidia-node-01".to_string(),
                ip_address: "10.0.1.1".to_string(),
                gpu_info: vec![GpuInfo {
                    model: GpuModel {
                        vendor: GpuVendor::Nvidia,
                        name: "A100".to_string(),
                        vram_mb: 81920,
                        compute_capability: Some("8.0".to_string()),
                    },
                    count: 4,
                    driver_version: "535.129.03".to_string(),
                }],
                total_cpu_cores: 64,
                total_memory_mb: 524288,
                kubernetes_labels: std::collections::HashMap::new(),
            },
            GpuNode {
                node_name: "amd-node-01".to_string(),
                ip_address: "10.0.1.2".to_string(),
                gpu_info: vec![GpuInfo {
                    model: GpuModel {
                        vendor: GpuVendor::Amd,
                        name: "MI250X".to_string(),
                        vram_mb: 131072,
                        compute_capability: None,
                    },
                    count: 2,
                    driver_version: "6.0.5".to_string(),
                }],
                total_cpu_cores: 128,
                total_memory_mb: 1048576,
                kubernetes_labels: std::collections::HashMap::new(),
            },
            GpuNode {
                node_name: "nvidia-node-02".to_string(),
                ip_address: "10.0.1.3".to_string(),
                gpu_info: vec![GpuInfo {
                    model: GpuModel {
                        vendor: GpuVendor::Nvidia,
                        name: "H100".to_string(),
                        vram_mb: 81920,
                        compute_capability: Some("9.0".to_string()),
                    },
                    count: 8,
                    driver_version: "545.23.08".to_string(),
                }],
                total_cpu_cores: 128,
                total_memory_mb: 1048576,
                kubernetes_labels: std::collections::HashMap::new(),
            },
        ];

        // Select Nvidia nodes
        let nvidia_nodes = select_gpu_nodes(&nodes, &GpuVendor::Nvidia);
        assert_eq!(nvidia_nodes.len(), 2, "Should find 2 Nvidia nodes");
        assert!(nvidia_nodes.iter().all(|n| n.node_name.contains("nvidia")));

        // Select AMD nodes
        let amd_nodes = select_gpu_nodes(&nodes, &GpuVendor::Amd);
        assert_eq!(amd_nodes.len(), 1, "Should find 1 AMD node");
        assert_eq!(amd_nodes[0].node_name, "amd-node-01");

        // Select Intel nodes (none)
        let intel_nodes = select_gpu_nodes(&nodes, &GpuVendor::Intel);
        assert_eq!(intel_nodes.len(), 0, "Should find 0 Intel nodes");
    }

    // ---------------------------------------------------------------
    // Test 7: GPU cluster summary
    // ---------------------------------------------------------------
    #[test]
    fn test_gpu_cluster_summary() {
        let nodes = vec![
            GpuNode {
                node_name: "node-1".to_string(),
                ip_address: "10.0.0.1".to_string(),
                gpu_info: vec![GpuInfo {
                    model: GpuModel {
                        vendor: GpuVendor::Nvidia,
                        name: "V100".to_string(),
                        vram_mb: 32768,
                        compute_capability: Some("7.0".to_string()),
                    },
                    count: 4,
                    driver_version: "525.0".to_string(),
                }],
                total_cpu_cores: 32,
                total_memory_mb: 262144,
                kubernetes_labels: std::collections::HashMap::new(),
            },
            GpuNode {
                node_name: "node-2".to_string(),
                ip_address: "10.0.0.2".to_string(),
                gpu_info: vec![GpuInfo {
                    model: GpuModel {
                        vendor: GpuVendor::Nvidia,
                        name: "V100".to_string(),
                        vram_mb: 32768,
                        compute_capability: Some("7.0".to_string()),
                    },
                    count: 4,
                    driver_version: "525.0".to_string(),
                }],
                total_cpu_cores: 32,
                total_memory_mb: 262144,
                kubernetes_labels: std::collections::HashMap::new(),
            },
        ];

        let total_gpus: u32 = nodes.iter()
            .flat_map(|n| &n.gpu_info)
            .map(|g| g.count)
            .sum();

        let summary = GpuClusterSummary {
            total_nodes: nodes.len(),
            total_gpus: total_gpus as usize,
            gpu_nodes: nodes,
            vendor_breakdown: {
                let mut m = std::collections::HashMap::new();
                m.insert("Nvidia".to_string(), 8);
                m
            },
        };

        assert_eq!(summary.total_nodes, 2);
        assert_eq!(summary.total_gpus, 8);
        assert_eq!(summary.gpu_nodes.len(), 2);
        assert_eq!(summary.vendor_breakdown.get("Nvidia"), Some(&8));

        // Verify serialization
        let json = serde_json::to_string(&summary).expect("Should serialize");
        assert!(json.contains("\"total_nodes\":2"));
        assert!(json.contains("\"total_gpus\":8"));
    }

    // ---------------------------------------------------------------
    // Test 8: DeploymentConfig serialization roundtrip
    // ---------------------------------------------------------------
    #[test]
    fn test_deployment_config_serde_roundtrip() {
        let config = DeploymentConfig {
            name: "test-deployment".to_string(),
            namespace: "test-ns".to_string(),
            replicas: 2,
            gpu_count: 4,
            gpu_vendor_preference: Some(GpuVendor::Nvidia),
            memory_limit_mb: 16384,
            cpu_limit_millicores: 8000,
            image: "test:latest".to_string(),
            env_vars: {
                let mut m = std::collections::HashMap::new();
                m.insert("KEY1".to_string(), "value1".to_string());
                m.insert("KEY2".to_string(), "value2".to_string());
                m
            },
        };

        let json = serde_json::to_string(&config).expect("Should serialize");
        let deserialized: DeploymentConfig = serde_json::from_str(&json).expect("Should deserialize");

        assert_eq!(deserialized.name, "test-deployment");
        assert_eq!(deserialized.replicas, 2);
        assert_eq!(deserialized.gpu_count, 4);
        assert_eq!(deserialized.env_vars.len(), 2);
    }

    // ---------------------------------------------------------------
    // Test 9: Intel GPU deployment YAML
    // ---------------------------------------------------------------
    #[test]
    fn test_deployment_yaml_intel_gpu() {
        let config = DeploymentConfig {
            gpu_vendor_preference: Some(GpuVendor::Intel),
            ..DeploymentConfig::default()
        };

        let yaml = generate_deployment_yaml(&config);
        assert!(
            yaml.contains("gpu.intel.com/i915"),
            "Intel deployment should use gpu.intel.com/i915. Got:\n{}",
            yaml
        );
    }

    // ---------------------------------------------------------------
    // Test 10: GpuVendor enum completeness and serde
    // ---------------------------------------------------------------
    #[test]
    fn test_gpu_vendor_serde() {
        let vendors = vec![
            GpuVendor::Nvidia,
            GpuVendor::Amd,
            GpuVendor::Intel,
            GpuVendor::Unknown("CustomGPU".to_string()),
        ];

        for vendor in &vendors {
            let json = serde_json::to_string(vendor).expect("Should serialize vendor");
            let deserialized: GpuVendor = serde_json::from_str(&json).expect("Should deserialize vendor");
            assert_eq!(vendor, &deserialized);
        }
    }

    // ---------------------------------------------------------------
    // Test 11: Kubernetes manifest has required labels
    // ---------------------------------------------------------------
    #[test]
    fn test_kubernetes_manifest_labels() {
        let config = DeploymentConfig::default();
        let yaml = generate_deployment_yaml(&config);

        assert!(yaml.contains("matchLabels"), "Should have matchLabels");
        assert!(yaml.contains("app:"), "Should have app label");
        assert!(yaml.contains("spec:"), "Should have spec section");
        assert!(yaml.contains("containers:"), "Should have containers section");
    }

    // ---------------------------------------------------------------
    // Test 12: Multiple GPU info per node
    // ---------------------------------------------------------------
    #[test]
    fn test_node_with_multiple_gpu_types() {
        let node = GpuNode {
            node_name: "mixed-gpu-node".to_string(),
            ip_address: "10.0.0.5".to_string(),
            gpu_info: vec![
                GpuInfo {
                    model: GpuModel {
                        vendor: GpuVendor::Nvidia,
                        name: "A100".to_string(),
                        vram_mb: 81920,
                        compute_capability: Some("8.0".to_string()),
                    },
                    count: 2,
                    driver_version: "535.0".to_string(),
                },
                GpuInfo {
                    model: GpuModel {
                        vendor: GpuVendor::Nvidia,
                        name: "T4".to_string(),
                        vram_mb: 16384,
                        compute_capability: Some("7.5".to_string()),
                    },
                    count: 4,
                    driver_version: "535.0".to_string(),
                },
            ],
            total_cpu_cores: 96,
            total_memory_mb: 786432,
            kubernetes_labels: std::collections::HashMap::new(),
        };

        assert_eq!(node.gpu_info.len(), 2);
        let total_gpus: u32 = node.gpu_info.iter().map(|g| g.count).sum();
        assert_eq!(total_gpus, 6, "Should have 6 total GPUs");

        // Verify serde roundtrip
        let json = serde_json::to_string(&node).expect("Should serialize");
        let deserialized: GpuNode = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.gpu_info.len(), 2);
    }
}
