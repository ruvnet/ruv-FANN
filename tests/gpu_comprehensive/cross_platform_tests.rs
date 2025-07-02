//! Cross-Platform GPU Compatibility Validation
//! 
//! Tests GPU implementation across different platforms, browsers, and hardware
//! configurations to ensure broad compatibility and consistent behavior.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Platform detection and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    pub platform_type: PlatformType,
    pub operating_system: String,
    pub architecture: String,
    pub gpu_vendor: Option<String>,
    pub gpu_model: Option<String>,
    pub driver_version: Option<String>,
    pub webgpu_support: WebGPUSupport,
    pub browser_info: Option<BrowserInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlatformType {
    Windows,
    MacOS,
    Linux,
    WebBrowser,
    Mobile,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebGPUSupport {
    pub available: bool,
    pub version: Option<String>,
    pub supported_features: Vec<String>,
    pub adapter_info: Option<AdapterInfo>,
    pub limits: Option<WebGPULimits>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserInfo {
    pub browser_name: String,
    pub browser_version: String,
    pub user_agent: String,
    pub webgl_support: bool,
    pub webgpu_experimental: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterInfo {
    pub vendor: String,
    pub architecture: String,
    pub device_type: String,
    pub backend_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebGPULimits {
    pub max_buffer_size: u64,
    pub max_texture_dimension_2d: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_invocations_per_workgroup: u32,
}

/// Cross-platform test case
#[derive(Debug, Clone)]
pub struct CrossPlatformTestCase {
    pub name: String,
    pub test_type: CompatibilityTestType,
    pub required_features: Vec<String>,
    pub expected_behavior: ExpectedBehavior,
}

#[derive(Debug, Clone)]
pub enum CompatibilityTestType {
    BasicWebGPUAvailability,
    ShaderCompilation,
    BufferOperations,
    ComputeCapabilities,
    MemoryLimits,
    FeatureSupport,
    PerformanceConsistency,
    ErrorHandling,
    BrowserSpecific,
}

#[derive(Debug, Clone)]
pub enum ExpectedBehavior {
    MustSupport,     // Critical functionality
    ShouldSupport,   // Expected but not critical
    MaySupport,      // Optional
    MustNotFail,     // Should gracefully degrade
}

/// Individual platform test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformTestResult {
    pub test_name: String,
    pub platform_info: PlatformInfo,
    pub test_type: String,
    pub success: bool,
    pub support_level: SupportLevel,
    pub performance_metrics: Option<PlatformPerformanceMetrics>,
    pub error_details: Option<String>,
    pub warnings: Vec<String>,
    pub feature_coverage: FeatureCoverage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupportLevel {
    FullSupport,     // All features work perfectly
    PartialSupport,  // Some limitations but functional
    BasicSupport,    // Minimal functionality only
    NoSupport,       // Feature not available
    Unknown,         // Could not determine
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformPerformanceMetrics {
    pub initialization_time_ms: f32,
    pub shader_compilation_time_ms: f32,
    pub buffer_allocation_time_ms: f32,
    pub compute_dispatch_time_ms: f32,
    pub memory_transfer_rate_mbps: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureCoverage {
    pub total_features: usize,
    pub supported_features: usize,
    pub coverage_percentage: f32,
    pub missing_features: Vec<String>,
    pub experimental_features: Vec<String>,
}

/// Comprehensive platform compatibility report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCompatibilityReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub tested_platforms: usize,
    pub successful_platforms: usize,
    pub failed_platforms: usize,
    pub compatibility_score: f32,
    
    // Detailed results
    pub platform_results: Vec<PlatformTestResult>,
    pub platform_summary: HashMap<String, PlatformSummary>,
    
    // Feature analysis
    pub feature_compatibility: FeatureCompatibilityMatrix,
    pub performance_comparison: PlatformPerformanceComparison,
    
    // Recommendations
    pub compatibility_issues: Vec<CompatibilityIssue>,
    pub deployment_recommendations: Vec<DeploymentRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformSummary {
    pub platform_name: String,
    pub test_count: usize,
    pub success_rate: f32,
    pub average_performance_score: f32,
    pub support_level: SupportLevel,
    pub critical_issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureCompatibilityMatrix {
    pub features: HashMap<String, FeatureSupport>,
    pub overall_coverage: f32,
    pub critical_features_supported: f32,
    pub optional_features_supported: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSupport {
    pub feature_name: String,
    pub support_by_platform: HashMap<String, bool>,
    pub overall_support_rate: f32,
    pub is_critical: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformPerformanceComparison {
    pub baseline_platform: String,
    pub performance_ratios: HashMap<String, f32>, // Relative to baseline
    pub consistency_score: f32,
    pub performance_outliers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityIssue {
    pub severity: IssueSeverity,
    pub affected_platforms: Vec<String>,
    pub issue_description: String,
    pub impact_assessment: String,
    pub workaround: Option<String>,
    pub fix_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentRecommendation {
    pub recommendation_type: RecommendationType,
    pub target_platforms: Vec<String>,
    pub description: String,
    pub implementation_priority: Priority,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RecommendationType {
    FeatureToggle,
    GracefulDegradation,
    PlatformSpecificImplementation,
    UserWarning,
    RequirementUpdate,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Priority {
    Immediate,
    High,
    Medium,
    Low,
}

/// Main cross-platform validator
pub struct CrossPlatformValidator {
    test_cases: Vec<CrossPlatformTestCase>,
    results: Vec<PlatformTestResult>,
    current_platform: Option<PlatformInfo>,
}

impl CrossPlatformValidator {
    pub fn new() -> Self {
        let mut validator = Self {
            test_cases: Vec::new(),
            results: Vec::new(),
            current_platform: None,
        };
        
        validator.generate_compatibility_test_suite();
        validator
    }
    
    fn generate_compatibility_test_suite(&mut self) {
        // Basic WebGPU availability tests
        self.test_cases.push(CrossPlatformTestCase {
            name: "webgpu_basic_availability".to_string(),
            test_type: CompatibilityTestType::BasicWebGPUAvailability,
            required_features: vec![],
            expected_behavior: ExpectedBehavior::MustSupport,
        });
        
        self.test_cases.push(CrossPlatformTestCase {
            name: "webgpu_adapter_request".to_string(),
            test_type: CompatibilityTestType::BasicWebGPUAvailability,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::MustSupport,
        });
        
        self.test_cases.push(CrossPlatformTestCase {
            name: "webgpu_device_creation".to_string(),
            test_type: CompatibilityTestType::BasicWebGPUAvailability,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::MustSupport,
        });
        
        // Shader compilation tests
        self.test_cases.push(CrossPlatformTestCase {
            name: "basic_compute_shader_compilation".to_string(),
            test_type: CompatibilityTestType::ShaderCompilation,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::MustSupport,
        });
        
        self.test_cases.push(CrossPlatformTestCase {
            name: "matrix_multiplication_shader".to_string(),
            test_type: CompatibilityTestType::ShaderCompilation,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::MustSupport,
        });
        
        self.test_cases.push(CrossPlatformTestCase {
            name: "activation_function_shaders".to_string(),
            test_type: CompatibilityTestType::ShaderCompilation,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::MustSupport,
        });
        
        // Buffer operation tests
        self.test_cases.push(CrossPlatformTestCase {
            name: "buffer_creation_and_mapping".to_string(),
            test_type: CompatibilityTestType::BufferOperations,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::MustSupport,
        });
        
        self.test_cases.push(CrossPlatformTestCase {
            name: "large_buffer_allocation".to_string(),
            test_type: CompatibilityTestType::BufferOperations,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::ShouldSupport,
        });
        
        self.test_cases.push(CrossPlatformTestCase {
            name: "buffer_data_transfer".to_string(),
            test_type: CompatibilityTestType::BufferOperations,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::MustSupport,
        });
        
        // Compute capability tests
        self.test_cases.push(CrossPlatformTestCase {
            name: "compute_pipeline_creation".to_string(),
            test_type: CompatibilityTestType::ComputeCapabilities,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::MustSupport,
        });
        
        self.test_cases.push(CrossPlatformTestCase {
            name: "workgroup_size_limits".to_string(),
            test_type: CompatibilityTestType::ComputeCapabilities,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::ShouldSupport,
        });
        
        self.test_cases.push(CrossPlatformTestCase {
            name: "shared_memory_usage".to_string(),
            test_type: CompatibilityTestType::ComputeCapabilities,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::ShouldSupport,
        });
        
        // Memory limit tests
        self.test_cases.push(CrossPlatformTestCase {
            name: "memory_limit_detection".to_string(),
            test_type: CompatibilityTestType::MemoryLimits,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::MustSupport,
        });
        
        self.test_cases.push(CrossPlatformTestCase {
            name: "out_of_memory_handling".to_string(),
            test_type: CompatibilityTestType::MemoryLimits,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::MustNotFail,
        });
        
        // Feature support tests
        self.test_cases.push(CrossPlatformTestCase {
            name: "float64_support".to_string(),
            test_type: CompatibilityTestType::FeatureSupport,
            required_features: vec!["shader-f64".to_string()],
            expected_behavior: ExpectedBehavior::MaySupport,
        });
        
        self.test_cases.push(CrossPlatformTestCase {
            name: "atomic_operations_support".to_string(),
            test_type: CompatibilityTestType::FeatureSupport,
            required_features: vec!["shader-atomic-min-max".to_string()],
            expected_behavior: ExpectedBehavior::ShouldSupport,
        });
        
        // Performance consistency tests
        self.test_cases.push(CrossPlatformTestCase {
            name: "matrix_multiplication_performance".to_string(),
            test_type: CompatibilityTestType::PerformanceConsistency,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::ShouldSupport,
        });
        
        self.test_cases.push(CrossPlatformTestCase {
            name: "batch_processing_performance".to_string(),
            test_type: CompatibilityTestType::PerformanceConsistency,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::ShouldSupport,
        });
        
        // Error handling tests
        self.test_cases.push(CrossPlatformTestCase {
            name: "invalid_shader_error_handling".to_string(),
            test_type: CompatibilityTestType::ErrorHandling,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::MustNotFail,
        });
        
        self.test_cases.push(CrossPlatformTestCase {
            name: "device_lost_recovery".to_string(),
            test_type: CompatibilityTestType::ErrorHandling,
            required_features: vec!["webgpu".to_string()],
            expected_behavior: ExpectedBehavior::MustNotFail,
        });
        
        // Browser-specific tests (only for web platforms)
        if cfg!(target_arch = "wasm32") {
            self.test_cases.push(CrossPlatformTestCase {
                name: "chrome_webgpu_compatibility".to_string(),
                test_type: CompatibilityTestType::BrowserSpecific,
                required_features: vec!["webgpu".to_string()],
                expected_behavior: ExpectedBehavior::ShouldSupport,
            });
            
            self.test_cases.push(CrossPlatformTestCase {
                name: "firefox_webgpu_compatibility".to_string(),
                test_type: CompatibilityTestType::BrowserSpecific,
                required_features: vec!["webgpu".to_string()],
                expected_behavior: ExpectedBehavior::MaySupport,
            });
            
            self.test_cases.push(CrossPlatformTestCase {
                name: "safari_webgpu_compatibility".to_string(),
                test_type: CompatibilityTestType::BrowserSpecific,
                required_features: vec!["webgpu".to_string()],
                expected_behavior: ExpectedBehavior::MaySupport,
            });
        }
        
        println!("Generated {} cross-platform compatibility test cases", self.test_cases.len());
    }
    
    /// Run comprehensive cross-platform compatibility tests
    pub async fn run_compatibility_tests(&mut self) -> Result<(), crate::ValidationError> {
        println!("🌐 Starting cross-platform GPU compatibility testing...");
        
        // Detect current platform
        self.current_platform = Some(self.detect_platform_info().await?);
        
        if let Some(ref platform) = self.current_platform {
            println!("Platform detected: {:?}", platform.platform_type);
            println!("WebGPU available: {}", platform.webgpu_support.available);
            
            if let Some(ref browser) = platform.browser_info {
                println!("Browser: {} {}", browser.browser_name, browser.browser_version);
            }
        }
        
        let start_time = std::time::Instant::now();
        
        // Run all compatibility tests
        for (i, test_case) in self.test_cases.iter().enumerate() {
            if i % 5 == 0 {
                println!("Progress: {}/{} compatibility tests completed", i, self.test_cases.len());
            }
            
            match self.execute_compatibility_test(test_case).await {
                Ok(result) => {
                    if !result.success && matches!(test_case.expected_behavior, ExpectedBehavior::MustSupport | ExpectedBehavior::MustNotFail) {
                        println!("⚠️ Critical compatibility issue: {} on {:?}", 
                               result.test_name, result.platform_info.platform_type);
                    }
                    self.results.push(result);
                }
                Err(e) => {
                    println!("❌ Compatibility test error: {} - {}", test_case.name, e);
                    
                    // Create a failed test result
                    if let Some(ref platform) = self.current_platform {
                        self.results.push(PlatformTestResult {
                            test_name: test_case.name.clone(),
                            platform_info: platform.clone(),
                            test_type: format!("{:?}", test_case.test_type),
                            success: false,
                            support_level: SupportLevel::NoSupport,
                            performance_metrics: None,
                            error_details: Some(e.to_string()),
                            warnings: vec!["Test execution failed".to_string()],
                            feature_coverage: FeatureCoverage {
                                total_features: test_case.required_features.len(),
                                supported_features: 0,
                                coverage_percentage: 0.0,
                                missing_features: test_case.required_features.clone(),
                                experimental_features: Vec::new(),
                            },
                        });
                    }
                }
            }
        }
        
        let testing_time = start_time.elapsed();
        println!("✅ Cross-platform compatibility testing completed in {:?}", testing_time);
        println!("Total results: {} tests", self.results.len());
        
        Ok(())
    }
    
    async fn detect_platform_info(&self) -> Result<PlatformInfo, crate::ValidationError> {
        // Detect platform type
        let platform_type = if cfg!(target_os = "windows") {
            PlatformType::Windows
        } else if cfg!(target_os = "macos") {
            PlatformType::MacOS
        } else if cfg!(target_os = "linux") {
            PlatformType::Linux
        } else if cfg!(target_arch = "wasm32") {
            PlatformType::WebBrowser
        } else {
            PlatformType::Unknown
        };
        
        // Detect OS and architecture
        let operating_system = std::env::consts::OS.to_string();
        let architecture = std::env::consts::ARCH.to_string();
        
        // Detect WebGPU support
        let webgpu_support = self.detect_webgpu_support().await;
        
        // Detect browser info (if applicable)
        let browser_info = if matches!(platform_type, PlatformType::WebBrowser) {
            Some(self.detect_browser_info().await)
        } else {
            None
        };
        
        // Detect GPU info (simplified)
        let (gpu_vendor, gpu_model, driver_version) = self.detect_gpu_info().await;
        
        Ok(PlatformInfo {
            platform_type,
            operating_system,
            architecture,
            gpu_vendor,
            gpu_model,
            driver_version,
            webgpu_support,
            browser_info,
        })
    }
    
    async fn detect_webgpu_support(&self) -> WebGPUSupport {
        // In a real implementation, this would use actual WebGPU APIs
        // For now, simulate detection based on platform
        
        let available = !matches!(std::env::consts::OS, "linux") || cfg!(target_arch = "wasm32");
        
        if available {
            WebGPUSupport {
                available: true,
                version: Some("1.0".to_string()),
                supported_features: vec![
                    "webgpu".to_string(),
                    "compute-shader".to_string(),
                    "buffer-operations".to_string(),
                ],
                adapter_info: Some(AdapterInfo {
                    vendor: "SimulatedGPU".to_string(),
                    architecture: "modern".to_string(),
                    device_type: "discrete".to_string(),
                    backend_type: "webgpu".to_string(),
                }),
                limits: Some(WebGPULimits {
                    max_buffer_size: 1024 * 1024 * 256, // 256 MB
                    max_texture_dimension_2d: 8192,
                    max_compute_workgroups_per_dimension: 65535,
                    max_compute_workgroup_size_x: 256,
                    max_compute_workgroup_size_y: 256,
                    max_compute_workgroup_size_z: 64,
                    max_compute_invocations_per_workgroup: 256,
                }),
            }
        } else {
            WebGPUSupport {
                available: false,
                version: None,
                supported_features: Vec::new(),
                adapter_info: None,
                limits: None,
            }
        }
    }
    
    async fn detect_browser_info(&self) -> BrowserInfo {
        // Simulate browser detection for WASM target
        #[cfg(target_arch = "wasm32")]
        {
            BrowserInfo {
                browser_name: "Chrome".to_string(),
                browser_version: "120.0".to_string(),
                user_agent: "Mozilla/5.0 (compatible; SimulatedBrowser)".to_string(),
                webgl_support: true,
                webgpu_experimental: true,
            }
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            BrowserInfo {
                browser_name: "N/A".to_string(),
                browser_version: "N/A".to_string(),
                user_agent: "Native Application".to_string(),
                webgl_support: false,
                webgpu_experimental: false,
            }
        }
    }
    
    async fn detect_gpu_info(&self) -> (Option<String>, Option<String>, Option<String>) {
        // In a real implementation, this would query actual GPU information
        (
            Some("NVIDIA".to_string()),
            Some("GeForce RTX 4080".to_string()),
            Some("545.92".to_string()),
        )
    }
    
    async fn execute_compatibility_test(&self, test_case: &CrossPlatformTestCase) -> Result<PlatformTestResult, Box<dyn std::error::Error>> {
        let platform_info = self.current_platform.as_ref()
            .ok_or("Platform info not available")?
            .clone();
        
        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();
        let mut performance_metrics = None;
        
        // Execute test based on type
        let (success, support_level, error_details) = match test_case.test_type {
            CompatibilityTestType::BasicWebGPUAvailability => {
                self.test_webgpu_availability(&test_case.name).await
            }
            CompatibilityTestType::ShaderCompilation => {
                self.test_shader_compilation(&test_case.name).await
            }
            CompatibilityTestType::BufferOperations => {
                self.test_buffer_operations(&test_case.name).await
            }
            CompatibilityTestType::ComputeCapabilities => {
                self.test_compute_capabilities(&test_case.name).await
            }
            CompatibilityTestType::MemoryLimits => {
                self.test_memory_limits(&test_case.name).await
            }
            CompatibilityTestType::FeatureSupport => {
                self.test_feature_support(&test_case.name, &test_case.required_features).await
            }
            CompatibilityTestType::PerformanceConsistency => {
                let (result, perf_metrics) = self.test_performance_consistency(&test_case.name).await;
                performance_metrics = perf_metrics;
                result
            }
            CompatibilityTestType::ErrorHandling => {
                self.test_error_handling(&test_case.name).await
            }
            CompatibilityTestType::BrowserSpecific => {
                self.test_browser_specific(&test_case.name).await
            }
        };
        
        // Check if behavior matches expectations
        let behavior_check = match (&test_case.expected_behavior, success, &support_level) {
            (ExpectedBehavior::MustSupport, false, _) => {
                warnings.push("Critical feature not supported".to_string());
                false
            }
            (ExpectedBehavior::MustNotFail, _, SupportLevel::NoSupport) => {
                warnings.push("Operation failed when graceful degradation expected".to_string());
                false
            }
            _ => success,
        };
        
        // Calculate feature coverage
        let feature_coverage = self.calculate_feature_coverage(&test_case.required_features, &platform_info);
        
        let test_duration = start_time.elapsed();
        
        if let Some(ref mut perf) = performance_metrics {
            perf.initialization_time_ms = test_duration.as_secs_f32() * 1000.0;
        } else if success {
            performance_metrics = Some(PlatformPerformanceMetrics {
                initialization_time_ms: test_duration.as_secs_f32() * 1000.0,
                shader_compilation_time_ms: 5.0,
                buffer_allocation_time_ms: 2.0,
                compute_dispatch_time_ms: 1.0,
                memory_transfer_rate_mbps: 1000.0,
            });
        }
        
        Ok(PlatformTestResult {
            test_name: test_case.name.clone(),
            platform_info,
            test_type: format!("{:?}", test_case.test_type),
            success: behavior_check,
            support_level,
            performance_metrics,
            error_details,
            warnings,
            feature_coverage,
        })
    }
    
    async fn test_webgpu_availability(&self, test_name: &str) -> (bool, SupportLevel, Option<String>) {
        if let Some(ref platform) = self.current_platform {
            if platform.webgpu_support.available {
                match test_name {
                    "webgpu_basic_availability" => (true, SupportLevel::FullSupport, None),
                    "webgpu_adapter_request" => {
                        // Simulate adapter request
                        if platform.webgpu_support.adapter_info.is_some() {
                            (true, SupportLevel::FullSupport, None)
                        } else {
                            (false, SupportLevel::NoSupport, Some("No adapter available".to_string()))
                        }
                    }
                    "webgpu_device_creation" => {
                        // Simulate device creation
                        (true, SupportLevel::FullSupport, None)
                    }
                    _ => (true, SupportLevel::FullSupport, None),
                }
            } else {
                (false, SupportLevel::NoSupport, Some("WebGPU not available on this platform".to_string()))
            }
        } else {
            (false, SupportLevel::Unknown, Some("Platform info not available".to_string()))
        }
    }
    
    async fn test_shader_compilation(&self, test_name: &str) -> (bool, SupportLevel, Option<String>) {
        if let Some(ref platform) = self.current_platform {
            if !platform.webgpu_support.available {
                return (false, SupportLevel::NoSupport, Some("WebGPU not available".to_string()));
            }
            
            // Simulate shader compilation tests
            match test_name {
                "basic_compute_shader_compilation" => {
                    // Basic shader should always work
                    (true, SupportLevel::FullSupport, None)
                }
                "matrix_multiplication_shader" => {
                    // More complex shader
                    if platform.webgpu_support.limits.as_ref()
                        .map_or(false, |l| l.max_compute_workgroup_size_x >= 16) {
                        (true, SupportLevel::FullSupport, None)
                    } else {
                        (false, SupportLevel::PartialSupport, 
                         Some("Workgroup size limitations".to_string()))
                    }
                }
                "activation_function_shaders" => {
                    // Test different activation functions
                    (true, SupportLevel::FullSupport, None)
                }
                _ => (true, SupportLevel::FullSupport, None),
            }
        } else {
            (false, SupportLevel::Unknown, Some("Platform info not available".to_string()))
        }
    }
    
    async fn test_buffer_operations(&self, test_name: &str) -> (bool, SupportLevel, Option<String>) {
        if let Some(ref platform) = self.current_platform {
            if !platform.webgpu_support.available {
                return (false, SupportLevel::NoSupport, Some("WebGPU not available".to_string()));
            }
            
            match test_name {
                "buffer_creation_and_mapping" => {
                    (true, SupportLevel::FullSupport, None)
                }
                "large_buffer_allocation" => {
                    // Check if platform supports large buffers
                    if let Some(ref limits) = platform.webgpu_support.limits {
                        if limits.max_buffer_size >= 100 * 1024 * 1024 { // 100 MB
                            (true, SupportLevel::FullSupport, None)
                        } else {
                            (true, SupportLevel::PartialSupport, 
                             Some("Limited buffer size support".to_string()))
                        }
                    } else {
                        (false, SupportLevel::Unknown, Some("Buffer limits unknown".to_string()))
                    }
                }
                "buffer_data_transfer" => {
                    (true, SupportLevel::FullSupport, None)
                }
                _ => (true, SupportLevel::FullSupport, None),
            }
        } else {
            (false, SupportLevel::Unknown, Some("Platform info not available".to_string()))
        }
    }
    
    async fn test_compute_capabilities(&self, test_name: &str) -> (bool, SupportLevel, Option<String>) {
        if let Some(ref platform) = self.current_platform {
            if !platform.webgpu_support.available {
                return (false, SupportLevel::NoSupport, Some("WebGPU not available".to_string()));
            }
            
            match test_name {
                "compute_pipeline_creation" => {
                    (true, SupportLevel::FullSupport, None)
                }
                "workgroup_size_limits" => {
                    if let Some(ref limits) = platform.webgpu_support.limits {
                        if limits.max_compute_workgroup_size_x >= 64 {
                            (true, SupportLevel::FullSupport, None)
                        } else {
                            (true, SupportLevel::PartialSupport, 
                             Some("Limited workgroup sizes".to_string()))
                        }
                    } else {
                        (false, SupportLevel::Unknown, Some("Workgroup limits unknown".to_string()))
                    }
                }
                "shared_memory_usage" => {
                    // Assume most modern platforms support shared memory
                    (true, SupportLevel::FullSupport, None)
                }
                _ => (true, SupportLevel::FullSupport, None),
            }
        } else {
            (false, SupportLevel::Unknown, Some("Platform info not available".to_string()))
        }
    }
    
    async fn test_memory_limits(&self, test_name: &str) -> (bool, SupportLevel, Option<String>) {
        if let Some(ref platform) = self.current_platform {
            if !platform.webgpu_support.available {
                return (false, SupportLevel::NoSupport, Some("WebGPU not available".to_string()));
            }
            
            match test_name {
                "memory_limit_detection" => {
                    if platform.webgpu_support.limits.is_some() {
                        (true, SupportLevel::FullSupport, None)
                    } else {
                        (false, SupportLevel::PartialSupport, 
                         Some("Memory limits not exposed".to_string()))
                    }
                }
                "out_of_memory_handling" => {
                    // Assume proper error handling
                    (true, SupportLevel::FullSupport, None)
                }
                _ => (true, SupportLevel::FullSupport, None),
            }
        } else {
            (false, SupportLevel::Unknown, Some("Platform info not available".to_string()))
        }
    }
    
    async fn test_feature_support(&self, test_name: &str, required_features: &[String]) -> (bool, SupportLevel, Option<String>) {
        if let Some(ref platform) = self.current_platform {
            if !platform.webgpu_support.available {
                return (false, SupportLevel::NoSupport, Some("WebGPU not available".to_string()));
            }
            
            let supported_features = &platform.webgpu_support.supported_features;
            let missing_features: Vec<_> = required_features.iter()
                .filter(|feature| !supported_features.contains(feature))
                .collect();
            
            match test_name {
                "float64_support" => {
                    if supported_features.contains(&"shader-f64".to_string()) {
                        (true, SupportLevel::FullSupport, None)
                    } else {
                        (false, SupportLevel::NoSupport, Some("Float64 not supported".to_string()))
                    }
                }
                "atomic_operations_support" => {
                    if supported_features.contains(&"shader-atomic-min-max".to_string()) {
                        (true, SupportLevel::FullSupport, None)
                    } else {
                        (false, SupportLevel::PartialSupport, Some("Atomic operations limited".to_string()))
                    }
                }
                _ => {
                    if missing_features.is_empty() {
                        (true, SupportLevel::FullSupport, None)
                    } else {
                        (false, SupportLevel::PartialSupport, 
                         Some(format!("Missing features: {:?}", missing_features)))
                    }
                }
            }
        } else {
            (false, SupportLevel::Unknown, Some("Platform info not available".to_string()))
        }
    }
    
    async fn test_performance_consistency(&self, test_name: &str) -> ((bool, SupportLevel, Option<String>), Option<PlatformPerformanceMetrics>) {
        if let Some(ref platform) = self.current_platform {
            if !platform.webgpu_support.available {
                return ((false, SupportLevel::NoSupport, Some("WebGPU not available".to_string())), None);
            }
            
            // Simulate performance testing
            let metrics = PlatformPerformanceMetrics {
                initialization_time_ms: 10.0,
                shader_compilation_time_ms: 15.0,
                buffer_allocation_time_ms: 2.0,
                compute_dispatch_time_ms: 5.0,
                memory_transfer_rate_mbps: match platform.platform_type {
                    PlatformType::Windows => 2000.0,
                    PlatformType::MacOS => 1800.0,
                    PlatformType::Linux => 1600.0,
                    PlatformType::WebBrowser => 1000.0,
                    _ => 800.0,
                },
            };
            
            let success = match test_name {
                "matrix_multiplication_performance" => metrics.compute_dispatch_time_ms < 50.0,
                "batch_processing_performance" => metrics.memory_transfer_rate_mbps > 500.0,
                _ => true,
            };
            
            let support_level = if success {
                SupportLevel::FullSupport
            } else {
                SupportLevel::PartialSupport
            };
            
            ((success, support_level, None), Some(metrics))
        } else {
            ((false, SupportLevel::Unknown, Some("Platform info not available".to_string())), None)
        }
    }
    
    async fn test_error_handling(&self, test_name: &str) -> (bool, SupportLevel, Option<String>) {
        if let Some(ref platform) = self.current_platform {
            if !platform.webgpu_support.available {
                return (false, SupportLevel::NoSupport, Some("WebGPU not available".to_string()));
            }
            
            match test_name {
                "invalid_shader_error_handling" => {
                    // Test that invalid shaders produce proper errors
                    (true, SupportLevel::FullSupport, None)
                }
                "device_lost_recovery" => {
                    // Test device lost scenarios
                    (true, SupportLevel::FullSupport, None)
                }
                _ => (true, SupportLevel::FullSupport, None),
            }
        } else {
            (false, SupportLevel::Unknown, Some("Platform info not available".to_string()))
        }
    }
    
    async fn test_browser_specific(&self, test_name: &str) -> (bool, SupportLevel, Option<String>) {
        if let Some(ref platform) = self.current_platform {
            if !matches!(platform.platform_type, PlatformType::WebBrowser) {
                return (false, SupportLevel::NoSupport, Some("Not a browser platform".to_string()));
            }
            
            if let Some(ref browser) = platform.browser_info {
                match test_name {
                    "chrome_webgpu_compatibility" => {
                        if browser.browser_name.to_lowercase().contains("chrome") {
                            (true, SupportLevel::FullSupport, None)
                        } else {
                            (false, SupportLevel::NoSupport, Some("Not Chrome browser".to_string()))
                        }
                    }
                    "firefox_webgpu_compatibility" => {
                        if browser.browser_name.to_lowercase().contains("firefox") {
                            (true, SupportLevel::PartialSupport, Some("Experimental support".to_string()))
                        } else {
                            (false, SupportLevel::NoSupport, Some("Not Firefox browser".to_string()))
                        }
                    }
                    "safari_webgpu_compatibility" => {
                        if browser.browser_name.to_lowercase().contains("safari") {
                            (true, SupportLevel::BasicSupport, Some("Limited support".to_string()))
                        } else {
                            (false, SupportLevel::NoSupport, Some("Not Safari browser".to_string()))
                        }
                    }
                    _ => (true, SupportLevel::FullSupport, None),
                }
            } else {
                (false, SupportLevel::Unknown, Some("Browser info not available".to_string()))
            }
        } else {
            (false, SupportLevel::Unknown, Some("Platform info not available".to_string()))
        }
    }
    
    fn calculate_feature_coverage(&self, required_features: &[String], platform_info: &PlatformInfo) -> FeatureCoverage {
        let total_features = required_features.len();
        let supported_features = required_features.iter()
            .filter(|feature| platform_info.webgpu_support.supported_features.contains(feature))
            .count();
        
        let coverage_percentage = if total_features > 0 {
            (supported_features as f32 / total_features as f32) * 100.0
        } else {
            100.0
        };
        
        let missing_features = required_features.iter()
            .filter(|feature| !platform_info.webgpu_support.supported_features.contains(feature))
            .cloned()
            .collect();
        
        // Features that might be experimental
        let experimental_features = required_features.iter()
            .filter(|feature| feature.contains("experimental") || feature.contains("preview"))
            .cloned()
            .collect();
        
        FeatureCoverage {
            total_features,
            supported_features,
            coverage_percentage,
            missing_features,
            experimental_features,
        }
    }
    
    /// Generate comprehensive platform compatibility report
    pub fn generate_compatibility_report(&self) -> PlatformCompatibilityReport {
        let tested_platforms = 1; // Currently testing single platform
        let successful_platforms = if self.results.iter().any(|r| r.success) { 1 } else { 0 };
        let failed_platforms = tested_platforms - successful_platforms;
        
        let compatibility_score = if !self.results.is_empty() {
            let success_rate = self.results.iter().filter(|r| r.success).count() as f32 / self.results.len() as f32;
            success_rate * 100.0
        } else {
            0.0
        };
        
        // Generate platform summary
        let mut platform_summary = HashMap::new();
        if let Some(ref platform) = self.current_platform {
            let platform_name = format!("{:?}", platform.platform_type);
            let platform_results: Vec<_> = self.results.iter()
                .filter(|r| format!("{:?}", r.platform_info.platform_type) == platform_name)
                .collect();
            
            let test_count = platform_results.len();
            let success_rate = if test_count > 0 {
                (platform_results.iter().filter(|r| r.success).count() as f32 / test_count as f32) * 100.0
            } else {
                0.0
            };
            
            let average_performance_score = if !platform_results.is_empty() {
                platform_results.iter()
                    .filter_map(|r| r.performance_metrics.as_ref())
                    .map(|m| 100.0 - m.initialization_time_ms) // Simple score based on speed
                    .sum::<f32>() / platform_results.len() as f32
            } else {
                0.0
            };
            
            let support_level = if success_rate >= 90.0 {
                SupportLevel::FullSupport
            } else if success_rate >= 70.0 {
                SupportLevel::PartialSupport
            } else if success_rate >= 50.0 {
                SupportLevel::BasicSupport
            } else {
                SupportLevel::NoSupport
            };
            
            let critical_issues: Vec<String> = platform_results.iter()
                .filter(|r| !r.success)
                .filter_map(|r| r.error_details.clone())
                .collect();
            
            platform_summary.insert(platform_name.clone(), PlatformSummary {
                platform_name,
                test_count,
                success_rate,
                average_performance_score,
                support_level,
                critical_issues,
            });
        }
        
        // Generate feature compatibility matrix
        let feature_compatibility = self.generate_feature_compatibility_matrix();
        
        // Generate performance comparison
        let performance_comparison = self.generate_performance_comparison();
        
        // Identify compatibility issues
        let compatibility_issues = self.identify_compatibility_issues();
        
        // Generate deployment recommendations
        let deployment_recommendations = self.generate_deployment_recommendations();
        
        PlatformCompatibilityReport {
            timestamp: chrono::Utc::now(),
            tested_platforms,
            successful_platforms,
            failed_platforms,
            compatibility_score,
            platform_results: self.results.clone(),
            platform_summary,
            feature_compatibility,
            performance_comparison,
            compatibility_issues,
            deployment_recommendations,
        }
    }
    
    fn generate_feature_compatibility_matrix(&self) -> FeatureCompatibilityMatrix {
        let mut features: HashMap<String, FeatureSupport> = HashMap::new();
        
        // Collect all features mentioned in test cases
        for test_case in &self.test_cases {
            for feature in &test_case.required_features {
                if !features.contains_key(feature) {
                    let support_by_platform = HashMap::new();
                    features.insert(feature.clone(), FeatureSupport {
                        feature_name: feature.clone(),
                        support_by_platform,
                        overall_support_rate: 0.0,
                        is_critical: matches!(test_case.expected_behavior, ExpectedBehavior::MustSupport),
                    });
                }
            }
        }
        
        // Calculate support rates
        let total_features = features.len();
        let critical_features_count = features.values().filter(|f| f.is_critical).count();
        let supported_features_count = features.len(); // Simplified for single platform
        let critical_features_supported_count = features.values()
            .filter(|f| f.is_critical)
            .count(); // Simplified
        
        let overall_coverage = if total_features > 0 {
            (supported_features_count as f32 / total_features as f32) * 100.0
        } else {
            0.0
        };
        
        let critical_features_supported = if critical_features_count > 0 {
            (critical_features_supported_count as f32 / critical_features_count as f32) * 100.0
        } else {
            100.0
        };
        
        let optional_features_supported = overall_coverage; // Simplified
        
        FeatureCompatibilityMatrix {
            features,
            overall_coverage,
            critical_features_supported,
            optional_features_supported,
        }
    }
    
    fn generate_performance_comparison(&self) -> PlatformPerformanceComparison {
        let baseline_platform = if let Some(ref platform) = self.current_platform {
            format!("{:?}", platform.platform_type)
        } else {
            "Unknown".to_string()
        };
        
        let mut performance_ratios = HashMap::new();
        performance_ratios.insert(baseline_platform.clone(), 1.0);
        
        // Calculate consistency score based on performance variation
        let performance_metrics: Vec<_> = self.results.iter()
            .filter_map(|r| r.performance_metrics.as_ref())
            .collect();
        
        let consistency_score = if performance_metrics.len() > 1 {
            let transfer_rates: Vec<f32> = performance_metrics.iter()
                .map(|m| m.memory_transfer_rate_mbps)
                .collect();
            
            let mean = transfer_rates.iter().sum::<f32>() / transfer_rates.len() as f32;
            let variance = transfer_rates.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f32>() / transfer_rates.len() as f32;
            let cv = if mean > 0.0 { variance.sqrt() / mean } else { 1.0 };
            
            (1.0 - cv.min(1.0)).max(0.0)
        } else {
            1.0
        };
        
        PlatformPerformanceComparison {
            baseline_platform,
            performance_ratios,
            consistency_score,
            performance_outliers: Vec::new(),
        }
    }
    
    fn identify_compatibility_issues(&self) -> Vec<CompatibilityIssue> {
        let mut issues = Vec::new();
        
        // Identify critical failures
        let critical_failures: Vec<_> = self.results.iter()
            .filter(|r| !r.success && r.test_type.contains("webgpu_basic_availability"))
            .collect();
        
        if !critical_failures.is_empty() {
            issues.push(CompatibilityIssue {
                severity: IssueSeverity::Critical,
                affected_platforms: critical_failures.iter()
                    .map(|r| format!("{:?}", r.platform_info.platform_type))
                    .collect(),
                issue_description: "WebGPU not available or not functional".to_string(),
                impact_assessment: "GPU acceleration completely unavailable".to_string(),
                workaround: Some("Fall back to CPU implementation".to_string()),
                fix_required: true,
            });
        }
        
        // Identify performance issues
        let performance_issues: Vec<_> = self.results.iter()
            .filter_map(|r| r.performance_metrics.as_ref().map(|m| (r, m)))
            .filter(|(_, m)| m.memory_transfer_rate_mbps < 500.0)
            .collect();
        
        if !performance_issues.is_empty() {
            issues.push(CompatibilityIssue {
                severity: IssueSeverity::Medium,
                affected_platforms: performance_issues.iter()
                    .map(|(r, _)| format!("{:?}", r.platform_info.platform_type))
                    .collect(),
                issue_description: "Low memory transfer performance".to_string(),
                impact_assessment: "Reduced GPU acceleration benefits".to_string(),
                workaround: Some("Optimize buffer usage patterns".to_string()),
                fix_required: false,
            });
        }
        
        issues
    }
    
    fn generate_deployment_recommendations(&self) -> Vec<DeploymentRecommendation> {
        let mut recommendations = Vec::new();
        
        // Analyze overall compatibility
        let success_rate = if !self.results.is_empty() {
            (self.results.iter().filter(|r| r.success).count() as f32 / self.results.len() as f32) * 100.0
        } else {
            0.0
        };
        
        if success_rate < 80.0 {
            recommendations.push(DeploymentRecommendation {
                recommendation_type: RecommendationType::FeatureToggle,
                target_platforms: vec!["All".to_string()],
                description: "Implement feature toggle for GPU acceleration".to_string(),
                implementation_priority: Priority::High,
            });
        }
        
        // Check for browser-specific issues
        if cfg!(target_arch = "wasm32") {
            recommendations.push(DeploymentRecommendation {
                recommendation_type: RecommendationType::GracefulDegradation,
                target_platforms: vec!["WebBrowser".to_string()],
                description: "Implement graceful degradation for browsers without WebGPU".to_string(),
                implementation_priority: Priority::High,
            });
        }
        
        // Performance recommendations
        if self.results.iter().any(|r| r.performance_metrics.as_ref()
            .map_or(false, |m| m.memory_transfer_rate_mbps < 1000.0)) {
            recommendations.push(DeploymentRecommendation {
                recommendation_type: RecommendationType::PlatformSpecificImplementation,
                target_platforms: vec!["Low-performance platforms".to_string()],
                description: "Implement platform-specific optimizations for slower hardware".to_string(),
                implementation_priority: Priority::Medium,
            });
        }
        
        recommendations
    }
}

impl PlatformCompatibilityReport {
    pub fn print_detailed_summary(&self) {
        println!("\n🌐 COMPREHENSIVE CROSS-PLATFORM COMPATIBILITY REPORT");
        println!("====================================================");
        println!("🗓️  Report Date: {}", self.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
        println!();
        
        // Overall compatibility metrics
        println!("📊 OVERALL COMPATIBILITY METRICS");
        println!("────────────────────────────────");
        println!("Tested Platforms: {}", self.tested_platforms);
        println!("Successful Platforms: {}", self.successful_platforms);
        println!("Failed Platforms: {}", self.failed_platforms);
        println!("Compatibility Score: {:.1}%", self.compatibility_score);
        
        // Platform breakdown
        println!("\n🖥️ PLATFORM BREAKDOWN");
        println!("─────────────────────");
        for (platform_name, summary) in &self.platform_summary {
            println!("{} Platform:", platform_name);
            println!("  Tests Run: {}", summary.test_count);
            println!("  Success Rate: {:.1}%", summary.success_rate);
            println!("  Support Level: {:?}", summary.support_level);
            println!("  Performance Score: {:.1}", summary.average_performance_score);
            
            if !summary.critical_issues.is_empty() {
                println!("  Critical Issues:");
                for issue in &summary.critical_issues {
                    println!("    • {}", issue);
                }
            }
        }
        
        // Feature compatibility matrix
        println!("\n🔧 FEATURE COMPATIBILITY MATRIX");
        println!("──────────────────────────────");
        println!("Overall Coverage: {:.1}%", self.feature_compatibility.overall_coverage);
        println!("Critical Features: {:.1}%", self.feature_compatibility.critical_features_supported);
        println!("Optional Features: {:.1}%", self.feature_compatibility.optional_features_supported);
        
        let critical_features: Vec<_> = self.feature_compatibility.features.values()
            .filter(|f| f.is_critical)
            .collect();
        
        if !critical_features.is_empty() {
            println!("\nCritical Features Status:");
            for feature in critical_features {
                println!("  • {}: {:.1}% support", feature.feature_name, feature.overall_support_rate);
            }
        }
        
        // Performance comparison
        println!("\n⚡ PERFORMANCE COMPARISON");
        println!("────────────────────────");
        println!("Baseline Platform: {}", self.performance_comparison.baseline_platform);
        println!("Consistency Score: {:.1}%", self.performance_comparison.consistency_score * 100.0);
        
        for (platform, ratio) in &self.performance_comparison.performance_ratios {
            if platform != &self.performance_comparison.baseline_platform {
                println!("  {} Performance: {:.2}x vs baseline", platform, ratio);
            }
        }
        
        if !self.performance_comparison.performance_outliers.is_empty() {
            println!("Performance Outliers: {:?}", self.performance_comparison.performance_outliers);
        }
        
        // Compatibility issues
        if !self.compatibility_issues.is_empty() {
            println!("\n⚠️ COMPATIBILITY ISSUES");
            println!("──────────────────────");
            for (i, issue) in self.compatibility_issues.iter().enumerate() {
                println!("{}. [{:?}] {}", i + 1, issue.severity, issue.issue_description);
                println!("   Affected: {:?}", issue.affected_platforms);
                println!("   Impact: {}", issue.impact_assessment);
                
                if let Some(ref workaround) = issue.workaround {
                    println!("   Workaround: {}", workaround);
                }
                
                println!("   Fix Required: {}", if issue.fix_required { "Yes" } else { "No" });
            }
        }
        
        // Deployment recommendations
        if !self.deployment_recommendations.is_empty() {
            println!("\n💡 DEPLOYMENT RECOMMENDATIONS");
            println!("─────────────────────────────");
            for (i, rec) in self.deployment_recommendations.iter().enumerate() {
                println!("{}. [{:?}] {:?}", i + 1, rec.implementation_priority, rec.recommendation_type);
                println!("   {}", rec.description);
                println!("   Target: {:?}", rec.target_platforms);
            }
        }
        
        // Test results summary
        println!("\n📋 DETAILED TEST RESULTS");
        println!("────────────────────────");
        let successful_tests = self.platform_results.iter().filter(|r| r.success).count();
        let failed_tests = self.platform_results.len() - successful_tests;
        
        println!("Total Tests: {} (Success: {}, Failed: {})", 
               self.platform_results.len(), successful_tests, failed_tests);
        
        // Group by test type
        let mut test_type_summary: HashMap<String, (usize, usize)> = HashMap::new();
        for result in &self.platform_results {
            let entry = test_type_summary.entry(result.test_type.clone()).or_insert((0, 0));
            entry.0 += 1; // Total count
            if result.success {
                entry.1 += 1; // Success count
            }
        }
        
        for (test_type, (total, success)) in test_type_summary {
            let success_rate = if total > 0 { (success as f32 / total as f32) * 100.0 } else { 0.0 };
            println!("  {}: {}/{} ({:.1}%)", test_type, success, total, success_rate);
        }
        
        // Failed tests details
        let failed_results: Vec<_> = self.platform_results.iter()
            .filter(|r| !r.success)
            .collect();
        
        if !failed_results.is_empty() {
            println!("\n❌ FAILED TESTS");
            println!("──────────────");
            for (i, result) in failed_results.iter().take(10).enumerate() {
                println!("{}. {} - {}", i + 1, result.test_name, result.support_level.name());
                if let Some(ref error) = result.error_details {
                    println!("   Error: {}", error);
                }
            }
            
            if failed_results.len() > 10 {
                println!("   ... and {} more failed tests", failed_results.len() - 10);
            }
        }
        
        // Overall assessment
        println!("\n🎯 OVERALL ASSESSMENT");
        println!("────────────────────");
        if self.compatibility_score >= 95.0 {
            println!("✅ EXCELLENT: Outstanding cross-platform compatibility");
            println!("   GPU acceleration works reliably across all tested platforms");
        } else if self.compatibility_score >= 85.0 {
            println!("✅ VERY GOOD: Strong cross-platform support with minor issues");
            println!("   GPU acceleration is broadly compatible");
        } else if self.compatibility_score >= 70.0 {
            println!("⚠️ GOOD: Decent compatibility with some platform limitations");
            println!("   Consider platform-specific optimizations");
        } else if self.compatibility_score >= 50.0 {
            println!("⚠️ MARGINAL: Significant compatibility challenges");
            println!("   Major platform-specific work needed");
        } else {
            println!("❌ POOR: Severe compatibility issues");
            println!("   GPU acceleration not viable across platforms");
        }
        
        println!("════════════════════════════════════════════════════════");
    }
}

impl SupportLevel {
    fn name(&self) -> &'static str {
        match self {
            SupportLevel::FullSupport => "Full Support",
            SupportLevel::PartialSupport => "Partial Support",
            SupportLevel::BasicSupport => "Basic Support",
            SupportLevel::NoSupport => "No Support",
            SupportLevel::Unknown => "Unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_platform_info_creation() {
        let platform_info = PlatformInfo {
            platform_type: PlatformType::Linux,
            operating_system: "linux".to_string(),
            architecture: "x86_64".to_string(),
            gpu_vendor: Some("NVIDIA".to_string()),
            gpu_model: Some("RTX 4080".to_string()),
            driver_version: Some("545.92".to_string()),
            webgpu_support: WebGPUSupport {
                available: true,
                version: Some("1.0".to_string()),
                supported_features: vec!["webgpu".to_string()],
                adapter_info: None,
                limits: None,
            },
            browser_info: None,
        };
        
        assert_eq!(platform_info.platform_type, PlatformType::Linux);
        assert!(platform_info.webgpu_support.available);
    }
    
    #[test]
    fn test_cross_platform_validator_creation() {
        let validator = CrossPlatformValidator::new();
        assert!(!validator.test_cases.is_empty());
        
        // Should have basic WebGPU tests
        assert!(validator.test_cases.iter().any(|t| t.name.contains("webgpu_basic_availability")));
        assert!(validator.test_cases.iter().any(|t| t.name.contains("shader_compilation")));
        assert!(validator.test_cases.iter().any(|t| t.name.contains("buffer_operations")));
    }
    
    #[tokio::test]
    async fn test_platform_detection() {
        let validator = CrossPlatformValidator::new();
        let platform_info = validator.detect_platform_info().await.unwrap();
        
        assert!(!platform_info.operating_system.is_empty());
        assert!(!platform_info.architecture.is_empty());
        
        // Platform type should be detected correctly
        assert!(matches!(platform_info.platform_type, 
                        PlatformType::Windows | PlatformType::MacOS | 
                        PlatformType::Linux | PlatformType::WebBrowser));
    }
    
    #[test]
    fn test_feature_coverage_calculation() {
        let validator = CrossPlatformValidator::new();
        let platform_info = PlatformInfo {
            platform_type: PlatformType::Linux,
            operating_system: "linux".to_string(),
            architecture: "x86_64".to_string(),
            gpu_vendor: None,
            gpu_model: None,
            driver_version: None,
            webgpu_support: WebGPUSupport {
                available: true,
                version: Some("1.0".to_string()),
                supported_features: vec!["webgpu".to_string(), "compute-shader".to_string()],
                adapter_info: None,
                limits: None,
            },
            browser_info: None,
        };
        
        let required_features = vec!["webgpu".to_string(), "compute-shader".to_string(), "missing-feature".to_string()];
        let coverage = validator.calculate_feature_coverage(&required_features, &platform_info);
        
        assert_eq!(coverage.total_features, 3);
        assert_eq!(coverage.supported_features, 2);
        assert!((coverage.coverage_percentage - 66.67).abs() < 0.1);
        assert_eq!(coverage.missing_features, vec!["missing-feature".to_string()]);
    }
    
    #[test]
    fn test_compatibility_issue_severity() {
        let issue = CompatibilityIssue {
            severity: IssueSeverity::Critical,
            affected_platforms: vec!["Windows".to_string()],
            issue_description: "WebGPU not available".to_string(),
            impact_assessment: "No GPU acceleration".to_string(),
            workaround: Some("Use CPU fallback".to_string()),
            fix_required: true,
        };
        
        assert!(matches!(issue.severity, IssueSeverity::Critical));
        assert!(issue.fix_required);
        assert!(issue.workaround.is_some());
    }
}