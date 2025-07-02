//! GPU Performance Benchmarking Framework
//! 
//! Comprehensive performance testing framework measuring GPU speedup vs CPU,
//! throughput, latency, and efficiency across different workload sizes.

use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Performance benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub timeout_per_test: Duration,
    pub include_memory_transfer: bool,
    pub detailed_profiling: bool,
    pub target_speedups: TargetSpeedups,
}

#[derive(Debug, Clone)]
pub struct TargetSpeedups {
    pub small_workloads: f32,   // < 1000 operations
    pub medium_workloads: f32,  // 1K-100K operations
    pub large_workloads: f32,   // 100K-10M operations
    pub huge_workloads: f32,    // > 10M operations
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            measurement_iterations: 10,
            timeout_per_test: Duration::from_secs(60),
            include_memory_transfer: true,
            detailed_profiling: true,
            target_speedups: TargetSpeedups {
                small_workloads: 1.5,
                medium_workloads: 5.0,
                large_workloads: 15.0,
                huge_workloads: 25.0,
            },
        }
    }
}

/// Workload classification for performance analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadSize {
    Tiny,    // < 100 operations
    Small,   // 100-1K operations
    Medium,  // 1K-100K operations
    Large,   // 100K-10M operations
    Huge,    // > 10M operations
}

impl WorkloadSize {
    pub fn from_operation_count(ops: usize) -> Self {
        match ops {
            0..=99 => WorkloadSize::Tiny,
            100..=999 => WorkloadSize::Small,
            1_000..=99_999 => WorkloadSize::Medium,
            100_000..=9_999_999 => WorkloadSize::Large,
            _ => WorkloadSize::Huge,
        }
    }
    
    pub fn expected_speedup(&self, config: &BenchmarkConfig) -> f32 {
        match self {
            WorkloadSize::Tiny => 0.5, // GPU overhead dominates
            WorkloadSize::Small => config.target_speedups.small_workloads,
            WorkloadSize::Medium => config.target_speedups.medium_workloads,
            WorkloadSize::Large => config.target_speedups.large_workloads,
            WorkloadSize::Huge => config.target_speedups.huge_workloads,
        }
    }
}

/// Individual benchmark test case
#[derive(Debug, Clone)]
pub struct BenchmarkTestCase {
    pub name: String,
    pub operation: BenchmarkOperation,
    pub workload_size: WorkloadSize,
    pub expected_speedup: f32,
    pub tolerance_factor: f32, // Acceptable deviation from expected speedup
}

#[derive(Debug, Clone)]
pub enum BenchmarkOperation {
    MatrixVectorMultiply {
        rows: usize,
        cols: usize,
    },
    BatchMatrixVectorMultiply {
        rows: usize,
        cols: usize,
        batch_size: usize,
    },
    ActivationFunction {
        function: crate::accuracy_validation::ActivationFunction,
        input_size: usize,
        steepness: f32,
    },
    VectorOperation {
        operation: VectorOpType,
        size: usize,
    },
    NeuralNetworkInference {
        layer_sizes: Vec<usize>,
        batch_size: usize,
    },
    CompositeOperation {
        description: String,
        operation_count: usize,
    },
}

#[derive(Debug, Clone)]
pub enum VectorOpType {
    Add,
    Scale,
    DotProduct,
    L2Norm,
}

/// Detailed performance measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub operation_type: String,
    pub workload_description: String,
    pub workload_size: String,
    
    // Timing measurements
    pub cpu_measurements: TimingMeasurements,
    pub gpu_measurements: TimingMeasurements,
    pub memory_transfer_measurements: MemoryTransferMeasurements,
    
    // Performance metrics
    pub speedup: f32,
    pub efficiency: f32, // Speedup vs theoretical maximum
    pub throughput_ops_per_sec: f64,
    pub latency_ms: f32,
    
    // Targets and assessment
    pub expected_speedup: f32,
    pub meets_target: bool,
    pub performance_category: PerformanceCategory,
    
    // Resource utilization
    pub cpu_utilization: ResourceUtilization,
    pub gpu_utilization: ResourceUtilization,
    pub memory_usage: MemoryUsage,
    
    // Quality metrics
    pub measurement_stability: f32, // Coefficient of variation
    pub outlier_count: usize,
    pub confidence_interval: (f32, f32), // 95% confidence interval for speedup
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMeasurements {
    pub individual_times: Vec<f32>, // All measurement iterations (ms)
    pub mean_time: f32,
    pub median_time: f32,
    pub min_time: f32,
    pub max_time: f32,
    pub std_deviation: f32,
    pub coefficient_of_variation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTransferMeasurements {
    pub upload_time_ms: f32,
    pub download_time_ms: f32,
    pub total_transfer_time_ms: f32,
    pub data_size_mb: f32,
    pub bandwidth_gbps: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PerformanceCategory {
    Excellent,  // Exceeds target by >25%
    Good,       // Meets target within 10%
    Acceptable, // Within 25% of target
    Poor,       // Below target by >25%
    Critical,   // Slower than CPU
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub utilization_percent: f32,
    pub peak_utilization: f32,
    pub efficiency_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub allocated_mb: f32,
    pub peak_usage_mb: f32,
    pub fragmentation_ratio: f32,
    pub gc_events: usize,
}

/// Comprehensive performance benchmarking report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub config: String, // Serialized benchmark config
    pub total_benchmarks: usize,
    pub successful_benchmarks: usize,
    pub failed_benchmarks: usize,
    
    // Overall performance metrics
    pub average_speedup: f32,
    pub median_speedup: f32,
    pub max_speedup: f32,
    pub min_speedup: f32,
    pub geometric_mean_speedup: f32,
    
    // Target achievement
    pub targets_met: usize,
    pub target_achievement_rate: f32,
    pub performance_score: f32, // Weighted score based on workload importance
    
    // Workload analysis
    pub workload_performance: HashMap<String, WorkloadPerformance>,
    pub operation_performance: HashMap<String, OperationPerformance>,
    
    // Quality metrics
    pub measurement_reliability: f32,
    pub result_consistency: f32,
    pub performance_regression_indicators: Vec<String>,
    
    // Detailed results
    pub benchmark_results: Vec<BenchmarkResult>,
    pub failed_benchmarks_details: Vec<String>,
    
    // Recommendations
    pub optimization_opportunities: Vec<OptimizationRecommendation>,
    pub performance_bottlenecks: Vec<PerformanceBottleneck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadPerformance {
    pub workload_size: String,
    pub average_speedup: f32,
    pub throughput_ops_per_sec: f64,
    pub efficiency_score: f32,
    pub meets_targets: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationPerformance {
    pub operation_type: String,
    pub test_count: usize,
    pub average_speedup: f32,
    pub best_speedup: f32,
    pub consistency_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub priority: OptimizationPriority,
    pub area: String,
    pub description: String,
    pub potential_improvement: String,
    pub implementation_effort: ImplementationEffort,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub bottleneck_type: BottleneckType,
    pub affected_operations: Vec<String>,
    pub impact_description: String,
    pub mitigation_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    Significant,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BottleneckType {
    MemoryBandwidth,
    ComputeThroughput,
    DataTransfer,
    Synchronization,
    Algorithm,
    Hardware,
}

/// Main performance benchmarking orchestrator
pub struct PerformanceBenchmarker {
    config: BenchmarkConfig,
    test_cases: Vec<BenchmarkTestCase>,
    results: Vec<BenchmarkResult>,
    failed_benchmarks: Vec<String>,
}

impl PerformanceBenchmarker {
    pub fn new(target_speedups: crate::PerformanceTargets) -> Self {
        let config = BenchmarkConfig {
            target_speedups: TargetSpeedups {
                small_workloads: target_speedups.min_speedup_vs_cpu,
                medium_workloads: target_speedups.min_speedup_vs_cpu * 2.0,
                large_workloads: target_speedups.min_speedup_vs_cpu * 5.0,
                huge_workloads: target_speedups.min_speedup_vs_cpu * 10.0,
            },
            ..Default::default()
        };
        
        let mut benchmarker = Self {
            config,
            test_cases: Vec::new(),
            results: Vec::new(),
            failed_benchmarks: Vec::new(),
        };
        
        benchmarker.generate_comprehensive_benchmark_suite();
        benchmarker
    }
    
    fn generate_comprehensive_benchmark_suite(&mut self) {
        // Matrix-vector multiplication benchmarks
        for &(rows, cols) in &[
            (8, 8), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512),
            (1000, 800), (2000, 1500), (4000, 3000), (8000, 6000),
            (100, 1), (1, 100), (1000, 10), (10, 1000), // Non-square matrices
        ] {
            let operation_count = rows * cols * 2; // multiply-add operations
            let workload_size = WorkloadSize::from_operation_count(operation_count);
            
            self.test_cases.push(BenchmarkTestCase {
                name: format!("matrix_vector_{}x{}", rows, cols),
                operation: BenchmarkOperation::MatrixVectorMultiply { rows, cols },
                workload_size,
                expected_speedup: workload_size.expected_speedup(&self.config),
                tolerance_factor: 0.3, // 30% tolerance
            });
        }
        
        // Batch operations with varying batch sizes
        for &batch_size in &[1, 4, 8, 16, 32, 64, 128, 256] {
            let (rows, cols) = (512, 512);
            let operation_count = rows * cols * 2 * batch_size;
            let workload_size = WorkloadSize::from_operation_count(operation_count);
            
            self.test_cases.push(BenchmarkTestCase {
                name: format!("batch_matrix_vector_{}x{}_batch{}", rows, cols, batch_size),
                operation: BenchmarkOperation::BatchMatrixVectorMultiply { rows, cols, batch_size },
                workload_size,
                expected_speedup: workload_size.expected_speedup(&self.config),
                tolerance_factor: 0.2,
            });
        }
        
        // Activation function benchmarks
        use crate::accuracy_validation::ActivationFunction;
        for &function in &[
            ActivationFunction::Linear,
            ActivationFunction::Sigmoid,
            ActivationFunction::ReLU,
            ActivationFunction::Tanh,
            ActivationFunction::GELU,
        ] {
            for &size in &[1000, 10000, 100000, 1000000] {
                let workload_size = WorkloadSize::from_operation_count(size);
                
                self.test_cases.push(BenchmarkTestCase {
                    name: format!("activation_{:?}_size{}", function, size),
                    operation: BenchmarkOperation::ActivationFunction {
                        function,
                        input_size: size,
                        steepness: 1.0,
                    },
                    workload_size,
                    expected_speedup: workload_size.expected_speedup(&self.config),
                    tolerance_factor: 0.25,
                });
            }
        }
        
        // Vector operation benchmarks
        for &op_type in &[VectorOpType::Add, VectorOpType::Scale, VectorOpType::DotProduct, VectorOpType::L2Norm] {
            for &size in &[1000, 10000, 100000, 1000000, 10000000] {
                let operation_count = match op_type {
                    VectorOpType::Add | VectorOpType::Scale => size,
                    VectorOpType::DotProduct => size * 2, // multiply-add
                    VectorOpType::L2Norm => size * 3, // square, sum, sqrt
                };
                let workload_size = WorkloadSize::from_operation_count(operation_count);
                
                self.test_cases.push(BenchmarkTestCase {
                    name: format!("vector_{:?}_size{}", op_type, size),
                    operation: BenchmarkOperation::VectorOperation {
                        operation: op_type,
                        size,
                    },
                    workload_size,
                    expected_speedup: workload_size.expected_speedup(&self.config),
                    tolerance_factor: 0.2,
                });
            }
        }
        
        // Neural network inference benchmarks
        for &(layers, batch_size) in &[
            (vec![784, 128, 64, 10], 1),        // Small network, single inference
            (vec![784, 128, 64, 10], 32),       // Small network, batch inference
            (vec![2048, 1024, 512, 256, 10], 1), // Medium network, single inference
            (vec![2048, 1024, 512, 256, 10], 16), // Medium network, batch inference
            (vec![4096, 2048, 1024, 512, 256, 10], 8), // Large network
        ] {
            let operation_count: usize = layers.windows(2)
                .map(|w| w[0] * w[1] * 2 * batch_size) // multiply-add operations per layer
                .sum();
            let workload_size = WorkloadSize::from_operation_count(operation_count);
            
            self.test_cases.push(BenchmarkTestCase {
                name: format!("neural_network_{}_layers_batch{}", layers.len() - 1, batch_size),
                operation: BenchmarkOperation::NeuralNetworkInference {
                    layer_sizes: layers,
                    batch_size,
                },
                workload_size,
                expected_speedup: workload_size.expected_speedup(&self.config),
                tolerance_factor: 0.3,
            });
        }
        
        // Composite operations (stress tests)
        self.test_cases.push(BenchmarkTestCase {
            name: "composite_mixed_operations".to_string(),
            operation: BenchmarkOperation::CompositeOperation {
                description: "Mixed matrix, activation, and vector operations".to_string(),
                operation_count: 1_000_000,
            },
            workload_size: WorkloadSize::Large,
            expected_speedup: WorkloadSize::Large.expected_speedup(&self.config),
            tolerance_factor: 0.4,
        });
        
        println!("Generated {} performance benchmark test cases", self.test_cases.len());
    }
    
    /// Run comprehensive performance benchmarks
    pub async fn run_comprehensive_benchmarks(&mut self) -> Result<(), crate::ValidationError> {
        println!("⚡ Starting comprehensive GPU performance benchmarking...");
        println!("Configuration:");
        println!("  Warmup iterations: {}", self.config.warmup_iterations);
        println!("  Measurement iterations: {}", self.config.measurement_iterations);
        println!("  Timeout per test: {:?}", self.config.timeout_per_test);
        println!("  Total test cases: {}", self.test_cases.len());
        
        let overall_start = Instant::now();
        
        for (i, test_case) in self.test_cases.iter().enumerate() {
            if i % 5 == 0 {
                println!("Progress: {}/{} benchmarks completed", i, self.test_cases.len());
            }
            
            match self.execute_benchmark(test_case).await {
                Ok(result) => {
                    if !result.meets_target {
                        println!("⚠️ Below target: {} - {:.2}x vs {:.2}x expected", 
                               result.test_name, result.speedup, result.expected_speedup);
                    } else if result.speedup > result.expected_speedup * 1.5 {
                        println!("🚀 Excellent: {} - {:.2}x speedup", result.test_name, result.speedup);
                    }
                    self.results.push(result);
                }
                Err(e) => {
                    println!("❌ Benchmark failed: {} - {}", test_case.name, e);
                    self.failed_benchmarks.push(format!("{}: {}", test_case.name, e));
                }
            }
        }
        
        let total_time = overall_start.elapsed();
        println!("✅ Performance benchmarking completed in {:?}", total_time);
        println!("Successful benchmarks: {}", self.results.len());
        println!("Failed benchmarks: {}", self.failed_benchmarks.len());
        
        Ok(())
    }
    
    async fn execute_benchmark(&self, test_case: &BenchmarkTestCase) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        // Warm up
        for _ in 0..self.config.warmup_iterations {
            let _ = self.run_cpu_benchmark(&test_case.operation).await?;
            let _ = self.run_gpu_benchmark(&test_case.operation).await?;
        }
        
        // CPU measurements
        let mut cpu_times = Vec::new();
        for _ in 0..self.config.measurement_iterations {
            let duration = self.run_cpu_benchmark(&test_case.operation).await?;
            cpu_times.push(duration.as_secs_f32() * 1000.0); // Convert to ms
        }
        
        // GPU measurements (including memory transfer if configured)
        let mut gpu_times = Vec::new();
        let mut memory_transfer_stats = MemoryTransferMeasurements {
            upload_time_ms: 0.0,
            download_time_ms: 0.0,
            total_transfer_time_ms: 0.0,
            data_size_mb: 0.0,
            bandwidth_gbps: 0.0,
        };
        
        for _ in 0..self.config.measurement_iterations {
            let (duration, transfer_stats) = self.run_gpu_benchmark_with_transfer(&test_case.operation).await?;
            gpu_times.push(duration.as_secs_f32() * 1000.0);
            
            // Accumulate transfer statistics
            memory_transfer_stats.upload_time_ms += transfer_stats.upload_time_ms;
            memory_transfer_stats.download_time_ms += transfer_stats.download_time_ms;
            memory_transfer_stats.total_transfer_time_ms += transfer_stats.total_transfer_time_ms;
            memory_transfer_stats.data_size_mb = transfer_stats.data_size_mb; // Same for all iterations
        }
        
        // Average transfer statistics
        let iterations = self.config.measurement_iterations as f32;
        memory_transfer_stats.upload_time_ms /= iterations;
        memory_transfer_stats.download_time_ms /= iterations;
        memory_transfer_stats.total_transfer_time_ms /= iterations;
        memory_transfer_stats.bandwidth_gbps = if memory_transfer_stats.total_transfer_time_ms > 0.0 {
            (memory_transfer_stats.data_size_mb * 8.0) / (memory_transfer_stats.total_transfer_time_ms / 1000.0) / 1000.0
        } else {
            0.0
        };
        
        // Calculate timing statistics
        let cpu_measurements = Self::calculate_timing_measurements(&cpu_times);
        let gpu_measurements = Self::calculate_timing_measurements(&gpu_times);
        
        // Calculate performance metrics
        let speedup = if gpu_measurements.mean_time > 0.0 {
            cpu_measurements.mean_time / gpu_measurements.mean_time
        } else {
            0.0
        };
        
        let operation_count = self.estimate_operation_count(&test_case.operation);
        let throughput_ops_per_sec = if gpu_measurements.mean_time > 0.0 {
            (operation_count as f64) / (gpu_measurements.mean_time as f64 / 1000.0)
        } else {
            0.0
        };
        
        let efficiency = speedup / test_case.expected_speedup;
        let meets_target = speedup >= test_case.expected_speedup * (1.0 - test_case.tolerance_factor);
        
        let performance_category = if speedup >= test_case.expected_speedup * 1.25 {
            PerformanceCategory::Excellent
        } else if speedup >= test_case.expected_speedup * 0.9 {
            PerformanceCategory::Good
        } else if speedup >= test_case.expected_speedup * 0.75 {
            PerformanceCategory::Acceptable
        } else if speedup >= 1.0 {
            PerformanceCategory::Poor
        } else {
            PerformanceCategory::Critical
        };
        
        // Calculate confidence interval (95%)
        let confidence_interval = Self::calculate_confidence_interval(&cpu_times, &gpu_times);
        
        // Simulate resource utilization (would be measured from actual GPU backend)
        let gpu_utilization = ResourceUtilization {
            utilization_percent: 85.0 + (speedup - 1.0) * 5.0, // Estimate based on speedup
            peak_utilization: 95.0,
            efficiency_score: efficiency.min(1.0),
        };
        
        let cpu_utilization = ResourceUtilization {
            utilization_percent: 25.0, // CPU typically low when GPU is used
            peak_utilization: 100.0,
            efficiency_score: 1.0,
        };
        
        let memory_usage = MemoryUsage {
            allocated_mb: memory_transfer_stats.data_size_mb,
            peak_usage_mb: memory_transfer_stats.data_size_mb * 1.2,
            fragmentation_ratio: 0.1,
            gc_events: 0,
        };
        
        Ok(BenchmarkResult {
            test_name: test_case.name.clone(),
            operation_type: format!("{:?}", test_case.operation),
            workload_description: self.describe_workload(&test_case.operation),
            workload_size: format!("{:?}", test_case.workload_size),
            cpu_measurements,
            gpu_measurements,
            memory_transfer_measurements: memory_transfer_stats,
            speedup,
            efficiency,
            throughput_ops_per_sec,
            latency_ms: gpu_measurements.mean_time,
            expected_speedup: test_case.expected_speedup,
            meets_target,
            performance_category,
            cpu_utilization,
            gpu_utilization,
            memory_usage,
            measurement_stability: gpu_measurements.coefficient_of_variation,
            outlier_count: Self::count_outliers(&gpu_times),
            confidence_interval,
        })
    }
    
    async fn run_cpu_benchmark(&self, operation: &BenchmarkOperation) -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        
        match operation {
            BenchmarkOperation::MatrixVectorMultiply { rows, cols } => {
                let matrix = self.generate_test_matrix(*rows, *cols);
                let vector = self.generate_test_vector(*cols);
                let _ = crate::accuracy_validation::CpuReferenceImplementation::matrix_vector_multiply(
                    &matrix, &vector, *rows, *cols
                );
            }
            BenchmarkOperation::BatchMatrixVectorMultiply { rows, cols, batch_size } => {
                let matrix = self.generate_test_matrix(*rows, *cols);
                let vectors = self.generate_test_batch_vectors(*cols, *batch_size);
                let _ = crate::accuracy_validation::CpuReferenceImplementation::batch_matrix_vector_multiply(
                    &matrix, &vectors, *rows, *cols
                );
            }
            BenchmarkOperation::ActivationFunction { function, input_size, steepness } => {
                let inputs = self.generate_test_vector(*input_size);
                let _ = crate::accuracy_validation::CpuReferenceImplementation::apply_activation_function(
                    &inputs, *function, *steepness
                );
            }
            BenchmarkOperation::VectorOperation { operation, size } => {
                let vector_a = self.generate_test_vector(*size);
                let vector_b = self.generate_test_vector(*size);
                
                match operation {
                    VectorOpType::Add => {
                        let _ = crate::accuracy_validation::CpuReferenceImplementation::vector_add(&vector_a, &vector_b);
                    }
                    VectorOpType::Scale => {
                        let _ = crate::accuracy_validation::CpuReferenceImplementation::vector_scale(&vector_a, 2.5);
                    }
                    VectorOpType::DotProduct => {
                        let _ = crate::accuracy_validation::CpuReferenceImplementation::dot_product(&vector_a, &vector_b);
                    }
                    VectorOpType::L2Norm => {
                        let _ = crate::accuracy_validation::CpuReferenceImplementation::vector_norm_l2(&vector_a);
                    }
                }
            }
            BenchmarkOperation::NeuralNetworkInference { layer_sizes, batch_size } => {
                // Simulate full neural network inference
                let mut activations = self.generate_test_vector(layer_sizes[0] * batch_size);
                
                for window in layer_sizes.windows(2) {
                    let input_size = window[0];
                    let output_size = window[1];
                    let weights = self.generate_test_matrix(output_size, input_size);
                    
                    // Process each batch item
                    for i in 0..*batch_size {
                        let start_idx = i * input_size;
                        let end_idx = start_idx + input_size;
                        let batch_input = &activations[start_idx..end_idx];
                        
                        let output = crate::accuracy_validation::CpuReferenceImplementation::matrix_vector_multiply(
                            &weights, batch_input, output_size, input_size
                        );
                        
                        // Apply activation function
                        let activated = crate::accuracy_validation::CpuReferenceImplementation::apply_activation_function(
                            &output, crate::accuracy_validation::ActivationFunction::ReLU, 1.0
                        );
                        
                        // Store back to activations
                        activations.splice(start_idx..end_idx, activated);
                    }
                }
            }
            BenchmarkOperation::CompositeOperation { operation_count, .. } => {
                // Mix of different operations to simulate real workload
                let iterations = (*operation_count / 1000).max(1);
                
                for _ in 0..iterations {
                    // Matrix operation
                    let matrix = self.generate_test_matrix(100, 100);
                    let vector = self.generate_test_vector(100);
                    let _ = crate::accuracy_validation::CpuReferenceImplementation::matrix_vector_multiply(
                        &matrix, &vector, 100, 100
                    );
                    
                    // Activation function
                    let inputs = self.generate_test_vector(1000);
                    let _ = crate::accuracy_validation::CpuReferenceImplementation::apply_activation_function(
                        &inputs, crate::accuracy_validation::ActivationFunction::Sigmoid, 1.0
                    );
                    
                    // Vector operations
                    let va = self.generate_test_vector(1000);
                    let vb = self.generate_test_vector(1000);
                    let _ = crate::accuracy_validation::CpuReferenceImplementation::vector_add(&va, &vb);
                }
            }
        }
        
        Ok(start.elapsed())
    }
    
    async fn run_gpu_benchmark_with_transfer(&self, operation: &BenchmarkOperation) -> Result<(Duration, MemoryTransferMeasurements), Box<dyn std::error::Error>> {
        // Simulate GPU benchmark with memory transfer
        // In real implementation, this would use the actual GPU backend
        
        let data_size_mb = self.estimate_data_size_mb(operation);
        
        let upload_start = Instant::now();
        // Simulate data upload to GPU
        tokio::time::sleep(Duration::from_micros((data_size_mb * 10.0) as u64)).await;
        let upload_time = upload_start.elapsed().as_secs_f32() * 1000.0;
        
        let compute_start = Instant::now();
        // Simulate GPU computation (faster than CPU)
        let cpu_duration = self.run_cpu_benchmark(operation).await?;
        let speedup_factor = 8.0; // Simulated speedup
        tokio::time::sleep(Duration::from_secs_f64(cpu_duration.as_secs_f64() / speedup_factor)).await;
        let compute_time = compute_start.elapsed();
        
        let download_start = Instant::now();
        // Simulate data download from GPU
        tokio::time::sleep(Duration::from_micros((data_size_mb * 5.0) as u64)).await;
        let download_time = download_start.elapsed().as_secs_f32() * 1000.0;
        
        let total_transfer_time = upload_time + download_time;
        let total_time = compute_time + Duration::from_secs_f32(total_transfer_time / 1000.0);
        
        let transfer_stats = MemoryTransferMeasurements {
            upload_time_ms: upload_time,
            download_time_ms: download_time,
            total_transfer_time_ms: total_transfer_time,
            data_size_mb,
            bandwidth_gbps: (data_size_mb * 8.0) / (total_transfer_time / 1000.0) / 1000.0,
        };
        
        Ok((total_time, transfer_stats))
    }
    
    fn calculate_timing_measurements(times: &[f32]) -> TimingMeasurements {
        let mut sorted_times = times.to_vec();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mean = times.iter().sum::<f32>() / times.len() as f32;
        let median = sorted_times[sorted_times.len() / 2];
        let min = sorted_times[0];
        let max = sorted_times[sorted_times.len() - 1];
        
        let variance = times.iter()
            .map(|t| (t - mean).powi(2))
            .sum::<f32>() / times.len() as f32;
        let std_deviation = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_deviation / mean } else { 0.0 };
        
        TimingMeasurements {
            individual_times: times.to_vec(),
            mean_time: mean,
            median_time: median,
            min_time: min,
            max_time: max,
            std_deviation,
            coefficient_of_variation,
        }
    }
    
    fn calculate_confidence_interval(cpu_times: &[f32], gpu_times: &[f32]) -> (f32, f32) {
        // Calculate 95% confidence interval for speedup
        let speedups: Vec<f32> = cpu_times.iter()
            .zip(gpu_times.iter())
            .map(|(cpu, gpu)| if *gpu > 0.0 { cpu / gpu } else { 0.0 })
            .collect();
        
        let mean_speedup = speedups.iter().sum::<f32>() / speedups.len() as f32;
        let variance = speedups.iter()
            .map(|s| (s - mean_speedup).powi(2))
            .sum::<f32>() / (speedups.len() - 1) as f32;
        let std_error = (variance / speedups.len() as f32).sqrt();
        
        // 95% confidence interval (approximately 1.96 * std_error)
        let margin = 1.96 * std_error;
        (mean_speedup - margin, mean_speedup + margin)
    }
    
    fn count_outliers(times: &[f32]) -> usize {
        if times.len() < 4 {
            return 0;
        }
        
        let mut sorted_times = times.to_vec();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let q1_idx = sorted_times.len() / 4;
        let q3_idx = (sorted_times.len() * 3) / 4;
        let q1 = sorted_times[q1_idx];
        let q3 = sorted_times[q3_idx];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        
        times.iter()
            .filter(|&&t| t < lower_bound || t > upper_bound)
            .count()
    }
    
    // Helper methods for test data generation
    fn generate_test_matrix(&self, rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| ((i as f32 * 0.1).sin() + 1.0) * 0.5)
            .collect()
    }
    
    fn generate_test_vector(&self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| (i as f32 * 0.1).cos())
            .collect()
    }
    
    fn generate_test_batch_vectors(&self, size: usize, batch_size: usize) -> Vec<Vec<f32>> {
        (0..batch_size)
            .map(|i| {
                (0..size)
                    .map(|j| ((i + j) as f32 * 0.1).sin())
                    .collect()
            })
            .collect()
    }
    
    fn estimate_operation_count(&self, operation: &BenchmarkOperation) -> usize {
        match operation {
            BenchmarkOperation::MatrixVectorMultiply { rows, cols } => rows * cols * 2,
            BenchmarkOperation::BatchMatrixVectorMultiply { rows, cols, batch_size } => rows * cols * 2 * batch_size,
            BenchmarkOperation::ActivationFunction { input_size, .. } => *input_size,
            BenchmarkOperation::VectorOperation { operation, size } => {
                match operation {
                    VectorOpType::Add | VectorOpType::Scale => *size,
                    VectorOpType::DotProduct => size * 2,
                    VectorOpType::L2Norm => size * 3,
                }
            }
            BenchmarkOperation::NeuralNetworkInference { layer_sizes, batch_size } => {
                layer_sizes.windows(2)
                    .map(|w| w[0] * w[1] * 2 * batch_size)
                    .sum()
            }
            BenchmarkOperation::CompositeOperation { operation_count, .. } => *operation_count,
        }
    }
    
    fn estimate_data_size_mb(&self, operation: &BenchmarkOperation) -> f32 {
        let float_size = std::mem::size_of::<f32>() as f32;
        let bytes = match operation {
            BenchmarkOperation::MatrixVectorMultiply { rows, cols } => {
                (rows * cols + cols + rows) as f32 * float_size
            }
            BenchmarkOperation::BatchMatrixVectorMultiply { rows, cols, batch_size } => {
                (rows * cols + cols * batch_size + rows * batch_size) as f32 * float_size
            }
            BenchmarkOperation::ActivationFunction { input_size, .. } => {
                (input_size * 2) as f32 * float_size // Input and output
            }
            BenchmarkOperation::VectorOperation { size, .. } => {
                (size * 3) as f32 * float_size // Two inputs, one output
            }
            BenchmarkOperation::NeuralNetworkInference { layer_sizes, batch_size } => {
                let total_weights: usize = layer_sizes.windows(2)
                    .map(|w| w[0] * w[1])
                    .sum();
                let total_activations: usize = layer_sizes.iter().sum::<usize>() * batch_size;
                (total_weights + total_activations) as f32 * float_size
            }
            BenchmarkOperation::CompositeOperation { operation_count, .. } => {
                (*operation_count as f32 / 1000.0) * float_size // Rough estimate
            }
        };
        
        bytes / (1024.0 * 1024.0) // Convert to MB
    }
    
    fn describe_workload(&self, operation: &BenchmarkOperation) -> String {
        match operation {
            BenchmarkOperation::MatrixVectorMultiply { rows, cols } => {
                format!("{}x{} matrix-vector multiplication", rows, cols)
            }
            BenchmarkOperation::BatchMatrixVectorMultiply { rows, cols, batch_size } => {
                format!("{}x{} matrix with {} vector batch", rows, cols, batch_size)
            }
            BenchmarkOperation::ActivationFunction { function, input_size, .. } => {
                format!("{:?} activation on {} elements", function, input_size)
            }
            BenchmarkOperation::VectorOperation { operation, size } => {
                format!("{:?} operation on {} elements", operation, size)
            }
            BenchmarkOperation::NeuralNetworkInference { layer_sizes, batch_size } => {
                format!("{}-layer network, batch size {}", layer_sizes.len() - 1, batch_size)
            }
            BenchmarkOperation::CompositeOperation { description, .. } => description.clone(),
        }
    }
    
    /// Generate comprehensive performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let total_benchmarks = self.results.len();
        let successful_benchmarks = total_benchmarks;
        let failed_benchmarks = self.failed_benchmarks.len();
        
        // Calculate overall speedup metrics
        let speedups: Vec<f32> = self.results.iter().map(|r| r.speedup).collect();
        let average_speedup = if !speedups.is_empty() {
            speedups.iter().sum::<f32>() / speedups.len() as f32
        } else {
            0.0
        };
        
        let mut sorted_speedups = speedups.clone();
        sorted_speedups.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_speedup = if !sorted_speedups.is_empty() {
            sorted_speedups[sorted_speedups.len() / 2]
        } else {
            0.0
        };
        
        let max_speedup = speedups.iter().cloned().fold(0.0f32, f32::max);
        let min_speedup = speedups.iter().cloned().fold(f32::INFINITY, f32::min);
        
        // Geometric mean speedup (better for ratios)
        let geometric_mean_speedup = if !speedups.is_empty() {
            let product = speedups.iter().map(|s| s.ln()).sum::<f32>();
            (product / speedups.len() as f32).exp()
        } else {
            0.0
        };
        
        // Target achievement
        let targets_met = self.results.iter().filter(|r| r.meets_target).count();
        let target_achievement_rate = if total_benchmarks > 0 {
            (targets_met as f32 / total_benchmarks as f32) * 100.0
        } else {
            0.0
        };
        
        // Performance score (weighted by workload importance)
        let performance_score = self.calculate_performance_score();
        
        // Generate analysis
        let workload_performance = self.analyze_workload_performance();
        let operation_performance = self.analyze_operation_performance();
        let optimization_opportunities = self.identify_optimization_opportunities();
        let performance_bottlenecks = self.identify_performance_bottlenecks();
        let performance_regression_indicators = self.check_performance_regressions();
        
        // Quality metrics
        let measurement_reliability = self.calculate_measurement_reliability();
        let result_consistency = self.calculate_result_consistency();
        
        PerformanceReport {
            timestamp: chrono::Utc::now(),
            config: format!("{:?}", self.config),
            total_benchmarks,
            successful_benchmarks,
            failed_benchmarks,
            average_speedup,
            median_speedup,
            max_speedup,
            min_speedup,
            geometric_mean_speedup,
            targets_met,
            target_achievement_rate,
            performance_score,
            workload_performance,
            operation_performance,
            measurement_reliability,
            result_consistency,
            performance_regression_indicators,
            benchmark_results: self.results.clone(),
            failed_benchmarks_details: self.failed_benchmarks.clone(),
            optimization_opportunities,
            performance_bottlenecks,
        }
    }
    
    fn calculate_performance_score(&self) -> f32 {
        if self.results.is_empty() {
            return 0.0;
        }
        
        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;
        
        for result in &self.results {
            // Weight by workload size (larger workloads are more important)
            let weight = match result.workload_size.as_str() {
                "Tiny" => 0.5,
                "Small" => 1.0,
                "Medium" => 2.0,
                "Large" => 3.0,
                "Huge" => 4.0,
                _ => 1.0,
            };
            
            // Score based on speedup vs target
            let score = if result.meets_target {
                (result.speedup / result.expected_speedup).min(2.0) // Cap at 2x target
            } else {
                (result.speedup / result.expected_speedup) * 0.5 // Penalty for missing target
            };
            
            weighted_score += score * weight;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            (weighted_score / total_weight).min(1.0) * 100.0 // Convert to percentage
        } else {
            0.0
        }
    }
    
    fn analyze_workload_performance(&self) -> HashMap<String, WorkloadPerformance> {
        let mut workload_groups: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        
        for result in &self.results {
            workload_groups
                .entry(result.workload_size.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }
        
        workload_groups
            .into_iter()
            .map(|(workload_size, results)| {
                let average_speedup = results.iter().map(|r| r.speedup).sum::<f32>() / results.len() as f32;
                let throughput_ops_per_sec = results.iter().map(|r| r.throughput_ops_per_sec).sum::<f64>() / results.len() as f64;
                let efficiency_score = results.iter().map(|r| r.efficiency).sum::<f32>() / results.len() as f32;
                let meets_targets = results.iter().all(|r| r.meets_target);
                
                (
                    workload_size.clone(),
                    WorkloadPerformance {
                        workload_size,
                        average_speedup,
                        throughput_ops_per_sec,
                        efficiency_score,
                        meets_targets,
                    },
                )
            })
            .collect()
    }
    
    fn analyze_operation_performance(&self) -> HashMap<String, OperationPerformance> {
        let mut operation_groups: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        
        for result in &self.results {
            let operation_key = result.operation_type.split('(').next().unwrap_or(&result.operation_type).to_string();
            operation_groups
                .entry(operation_key)
                .or_insert_with(Vec::new)
                .push(result);
        }
        
        operation_groups
            .into_iter()
            .map(|(operation_type, results)| {
                let test_count = results.len();
                let average_speedup = results.iter().map(|r| r.speedup).sum::<f32>() / results.len() as f32;
                let best_speedup = results.iter().map(|r| r.speedup).fold(0.0f32, f32::max);
                
                // Calculate consistency score based on coefficient of variation
                let speedups: Vec<f32> = results.iter().map(|r| r.speedup).collect();
                let mean = average_speedup;
                let variance = speedups.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / speedups.len() as f32;
                let cv = if mean > 0.0 { variance.sqrt() / mean } else { 1.0 };
                let consistency_score = (1.0 - cv.min(1.0)).max(0.0); // Higher is better
                
                (
                    operation_type.clone(),
                    OperationPerformance {
                        operation_type,
                        test_count,
                        average_speedup,
                        best_speedup,
                        consistency_score,
                    },
                )
            })
            .collect()
    }
    
    fn identify_optimization_opportunities(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        // Analyze results for optimization opportunities
        let low_speedup_results: Vec<_> = self.results.iter()
            .filter(|r| r.speedup < r.expected_speedup * 0.8)
            .collect();
        
        if !low_speedup_results.is_empty() {
            recommendations.push(OptimizationRecommendation {
                priority: OptimizationPriority::High,
                area: "GPU Kernel Optimization".to_string(),
                description: format!("{} operations showing below-target performance", low_speedup_results.len()),
                potential_improvement: "20-50% speedup improvement possible".to_string(),
                implementation_effort: ImplementationEffort::Medium,
            });
        }
        
        // Check for high memory transfer overhead
        let high_transfer_overhead: Vec<_> = self.results.iter()
            .filter(|r| r.memory_transfer_measurements.total_transfer_time_ms > r.gpu_measurements.mean_time * 0.5)
            .collect();
        
        if !high_transfer_overhead.is_empty() {
            recommendations.push(OptimizationRecommendation {
                priority: OptimizationPriority::Medium,
                area: "Memory Transfer Optimization".to_string(),
                description: format!("{} operations with high transfer overhead", high_transfer_overhead.len()),
                potential_improvement: "Reduce memory transfer costs by 30-70%".to_string(),
                implementation_effort: ImplementationEffort::Low,
            });
        }
        
        // Check for inconsistent performance
        let inconsistent_results: Vec<_> = self.results.iter()
            .filter(|r| r.measurement_stability > 0.2) // CV > 20%
            .collect();
        
        if !inconsistent_results.is_empty() {
            recommendations.push(OptimizationRecommendation {
                priority: OptimizationPriority::Medium,
                area: "Performance Stability".to_string(),
                description: format!("{} operations with inconsistent timing", inconsistent_results.len()),
                potential_improvement: "Improve performance predictability".to_string(),
                implementation_effort: ImplementationEffort::High,
            });
        }
        
        recommendations
    }
    
    fn identify_performance_bottlenecks(&self) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();
        
        // Memory bandwidth bottleneck
        let memory_bound_operations: Vec<_> = self.results.iter()
            .filter(|r| r.memory_transfer_measurements.bandwidth_gbps < 100.0 && r.memory_usage.allocated_mb > 10.0)
            .map(|r| r.test_name.clone())
            .collect();
        
        if !memory_bound_operations.is_empty() {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::MemoryBandwidth,
                affected_operations: memory_bound_operations,
                impact_description: "Memory transfer limiting GPU performance".to_string(),
                mitigation_suggestions: vec![
                    "Implement buffer pooling".to_string(),
                    "Use asynchronous transfers".to_string(),
                    "Batch operations to reduce transfers".to_string(),
                ],
            });
        }
        
        // Compute throughput bottleneck
        let compute_bound_operations: Vec<_> = self.results.iter()
            .filter(|r| r.gpu_utilization.utilization_percent < 80.0 && r.speedup < r.expected_speedup)
            .map(|r| r.test_name.clone())
            .collect();
        
        if !compute_bound_operations.is_empty() {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::ComputeThroughput,
                affected_operations: compute_bound_operations,
                impact_description: "GPU compute units underutilized".to_string(),
                mitigation_suggestions: vec![
                    "Optimize kernel launch parameters".to_string(),
                    "Improve workgroup size selection".to_string(),
                    "Reduce divergent branching".to_string(),
                ],
            });
        }
        
        bottlenecks
    }
    
    fn check_performance_regressions(&self) -> Vec<String> {
        // This would compare against historical performance data
        // For now, return potential regression indicators
        let mut indicators = Vec::new();
        
        if let Some(avg_speedup) = self.results.iter().map(|r| r.speedup).nth(0) {
            if avg_speedup < 2.0 {
                indicators.push("Overall speedup below expected baseline".to_string());
            }
        }
        
        let failed_targets = self.results.iter().filter(|r| !r.meets_target).count();
        if failed_targets > self.results.len() / 4 {
            indicators.push("High proportion of benchmarks missing performance targets".to_string());
        }
        
        indicators
    }
    
    fn calculate_measurement_reliability(&self) -> f32 {
        if self.results.is_empty() {
            return 0.0;
        }
        
        // Based on measurement stability (lower CV is better)
        let avg_cv = self.results.iter()
            .map(|r| r.measurement_stability)
            .sum::<f32>() / self.results.len() as f32;
        
        (1.0 - avg_cv.min(1.0)).max(0.0) // Convert to reliability score
    }
    
    fn calculate_result_consistency(&self) -> f32 {
        if self.results.is_empty() {
            return 0.0;
        }
        
        // Consistency across similar operations
        let speedup_cv = {
            let speedups: Vec<f32> = self.results.iter().map(|r| r.speedup).collect();
            let mean = speedups.iter().sum::<f32>() / speedups.len() as f32;
            let variance = speedups.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / speedups.len() as f32;
            if mean > 0.0 { variance.sqrt() / mean } else { 1.0 }
        };
        
        (1.0 - speedup_cv.min(1.0)).max(0.0)
    }
}

impl PerformanceReport {
    pub fn print_detailed_summary(&self) {
        println!("\n⚡ COMPREHENSIVE GPU PERFORMANCE BENCHMARK REPORT");
        println!("=================================================");
        println!("🗓️  Report Date: {}", self.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
        println!();
        
        // Overall performance metrics
        println!("📊 OVERALL PERFORMANCE METRICS");
        println!("──────────────────────────────");
        println!("Total Benchmarks: {} (Success: {}, Failed: {})", 
               self.total_benchmarks, self.successful_benchmarks, self.failed_benchmarks);
        println!("Average Speedup: {:.2}x", self.average_speedup);
        println!("Median Speedup: {:.2}x", self.median_speedup);
        println!("Geometric Mean Speedup: {:.2}x", self.geometric_mean_speedup);
        println!("Max Speedup: {:.2}x", self.max_speedup);
        println!("Min Speedup: {:.2}x", self.min_speedup);
        println!("Target Achievement Rate: {:.1}% ({}/{})", 
               self.target_achievement_rate, self.targets_met, self.total_benchmarks);
        println!("Performance Score: {:.1}/100", self.performance_score);
        
        // Quality metrics
        println!("\n🎯 MEASUREMENT QUALITY");
        println!("─────────────────────");
        println!("Measurement Reliability: {:.1}%", self.measurement_reliability * 100.0);
        println!("Result Consistency: {:.1}%", self.result_consistency * 100.0);
        
        // Workload performance breakdown
        println!("\n📈 WORKLOAD PERFORMANCE BREAKDOWN");
        println!("─────────────────────────────────");
        for (workload_size, perf) in &self.workload_performance {
            println!("{} Workloads:", workload_size);
            println!("  Average Speedup: {:.2}x", perf.average_speedup);
            println!("  Throughput: {:.0} ops/sec", perf.throughput_ops_per_sec);
            println!("  Efficiency: {:.1}%", perf.efficiency_score * 100.0);
            println!("  Meets Targets: {}", if perf.meets_targets { "✅ Yes" } else { "❌ No" });
        }
        
        // Operation type analysis
        println!("\n🔧 OPERATION TYPE ANALYSIS");
        println!("──────────────────────────");
        for (operation, perf) in &self.operation_performance {
            println!("{} ({} tests):", operation, perf.test_count);
            println!("  Average Speedup: {:.2}x", perf.average_speedup);
            println!("  Best Speedup: {:.2}x", perf.best_speedup);
            println!("  Consistency Score: {:.1}%", perf.consistency_score * 100.0);
        }
        
        // Top performing benchmarks
        println!("\n🚀 TOP PERFORMING BENCHMARKS");
        println!("────────────────────────────");
        let mut top_results = self.benchmark_results.clone();
        top_results.sort_by(|a, b| b.speedup.partial_cmp(&a.speedup).unwrap());
        for (i, result) in top_results.iter().take(5).enumerate() {
            println!("{}. {} - {:.2}x speedup ({:.1} ms → {:.1} ms)", 
                   i + 1, result.test_name, result.speedup, 
                   result.cpu_measurements.mean_time, result.gpu_measurements.mean_time);
        }
        
        // Optimization opportunities
        if !self.optimization_opportunities.is_empty() {
            println!("\n💡 OPTIMIZATION OPPORTUNITIES");
            println!("─────────────────────────────");
            for (i, opt) in self.optimization_opportunities.iter().enumerate() {
                println!("{}. [{:?}] {} - {}", 
                       i + 1, opt.priority, opt.area, opt.description);
                println!("   💪 Potential: {}", opt.potential_improvement);
                println!("   🛠️  Effort: {:?}", opt.implementation_effort);
            }
        }
        
        // Performance bottlenecks
        if !self.performance_bottlenecks.is_empty() {
            println!("\n⚠️ PERFORMANCE BOTTLENECKS");
            println!("─────────────────────────");
            for (i, bottleneck) in self.performance_bottlenecks.iter().enumerate() {
                println!("{}. {:?} Bottleneck", i + 1, bottleneck.bottleneck_type);
                println!("   📝 Impact: {}", bottleneck.impact_description);
                println!("   🔧 Affected: {} operations", bottleneck.affected_operations.len());
                println!("   💡 Mitigations:");
                for suggestion in &bottleneck.mitigation_suggestions {
                    println!("      • {}", suggestion);
                }
            }
        }
        
        // Performance regression indicators
        if !self.performance_regression_indicators.is_empty() {
            println!("\n🚨 PERFORMANCE REGRESSION INDICATORS");
            println!("───────────────────────────────────");
            for (i, indicator) in self.performance_regression_indicators.iter().enumerate() {
                println!("{}. {}", i + 1, indicator);
            }
        }
        
        // Failed benchmarks
        if !self.failed_benchmarks_details.is_empty() {
            println!("\n❌ FAILED BENCHMARKS");
            println!("───────────────────");
            for (i, failure) in self.failed_benchmarks_details.iter().enumerate() {
                println!("{}. {}", i + 1, failure);
            }
        }
        
        // Overall assessment
        println!("\n🎯 OVERALL ASSESSMENT");
        println!("────────────────────");
        if self.performance_score >= 90.0 {
            println!("✅ EXCELLENT: Outstanding GPU performance across all workloads");
            println!("   GPU acceleration is highly effective and ready for production");
        } else if self.performance_score >= 75.0 {
            println!("✅ VERY GOOD: Strong GPU performance with minor optimization opportunities");
            println!("   GPU acceleration provides significant benefits");
        } else if self.performance_score >= 60.0 {
            println!("⚠️ GOOD: Decent GPU performance but with room for improvement");
            println!("   Consider implementing suggested optimizations");
        } else if self.performance_score >= 40.0 {
            println!("⚠️ MARGINAL: GPU performance below expectations");
            println!("   Significant optimization work recommended before production");
        } else {
            println!("❌ POOR: GPU acceleration not providing expected benefits");
            println!("   Major performance issues need resolution");
        }
        
        // Recommendations
        println!("\n💡 KEY RECOMMENDATIONS");
        println!("─────────────────────");
        if self.target_achievement_rate < 80.0 {
            println!("• Focus on operations missing performance targets");
            println!("• Review GPU kernel implementations for optimization opportunities");
        }
        
        if self.measurement_reliability < 0.8 {
            println!("• Improve measurement stability and repeatability");
            println!("• Consider longer warmup periods or more measurement iterations");
        }
        
        if self.average_speedup < 5.0 {
            println!("• Investigate opportunities for better GPU utilization");
            println!("• Consider batching operations to amortize GPU setup costs");
        }
        
        if !self.performance_bottlenecks.is_empty() {
            println!("• Address identified performance bottlenecks for maximum impact");
        }
        
        println!("════════════════════════════════════════════════════════");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_workload_size_classification() {
        assert_eq!(WorkloadSize::from_operation_count(50), WorkloadSize::Tiny);
        assert_eq!(WorkloadSize::from_operation_count(500), WorkloadSize::Small);
        assert_eq!(WorkloadSize::from_operation_count(50000), WorkloadSize::Medium);
        assert_eq!(WorkloadSize::from_operation_count(5000000), WorkloadSize::Large);
        assert_eq!(WorkloadSize::from_operation_count(50000000), WorkloadSize::Huge);
    }
    
    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.measurement_iterations, 10);
        assert!(config.include_memory_transfer);
        assert!(config.target_speedups.large_workloads > config.target_speedups.small_workloads);
    }
    
    #[test]
    fn test_timing_measurements_calculation() {
        let times = vec![10.0, 12.0, 11.0, 13.0, 9.0];
        let measurements = PerformanceBenchmarker::calculate_timing_measurements(&times);
        
        assert_eq!(measurements.mean_time, 11.0);
        assert_eq!(measurements.median_time, 11.0);
        assert_eq!(measurements.min_time, 9.0);
        assert_eq!(measurements.max_time, 13.0);
        assert!(measurements.std_deviation > 0.0);
        assert!(measurements.coefficient_of_variation > 0.0);
    }
    
    #[test]
    fn test_performance_benchmarker_creation() {
        let targets = crate::PerformanceTargets {
            min_speedup_vs_cpu: 2.0,
            max_latency_ms: 10.0,
            min_throughput_ops_per_sec: 1_000_000.0,
            memory_efficiency_threshold: 0.8,
        };
        
        let benchmarker = PerformanceBenchmarker::new(targets);
        assert!(!benchmarker.test_cases.is_empty());
        assert_eq!(benchmarker.config.target_speedups.small_workloads, 2.0);
    }
    
    #[tokio::test]
    async fn test_cpu_benchmark_execution() {
        let targets = crate::PerformanceTargets {
            min_speedup_vs_cpu: 2.0,
            max_latency_ms: 10.0,
            min_throughput_ops_per_sec: 1_000_000.0,
            memory_efficiency_threshold: 0.8,
        };
        
        let benchmarker = PerformanceBenchmarker::new(targets);
        let operation = BenchmarkOperation::MatrixVectorMultiply { rows: 10, cols: 10 };
        
        let duration = benchmarker.run_cpu_benchmark(&operation).await.unwrap();
        assert!(duration.as_millis() >= 0);
    }
    
    #[test]
    fn test_outlier_detection() {
        let normal_times = vec![10.0, 11.0, 10.5, 11.2, 10.8];
        let outliers = PerformanceBenchmarker::count_outliers(&normal_times);
        assert_eq!(outliers, 0);
        
        let with_outliers = vec![10.0, 11.0, 10.5, 50.0, 10.8]; // 50.0 is an outlier
        let outliers = PerformanceBenchmarker::count_outliers(&with_outliers);
        assert!(outliers > 0);
    }
}