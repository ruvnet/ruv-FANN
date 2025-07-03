//! GPU Performance Test Suite for ruv-FANN
//!
//! This module contains comprehensive tests for validating GPU acceleration
//! performance across different operations and configurations.

#![cfg(test)]

use ruv_fann::*;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "webgpu")]
use ruv_fann::webgpu::{BackendSelector, BackendType, ComputeBackend, WebGPUBackend};

/// Performance test results structure
#[derive(Debug, Clone)]
struct PerformanceResult {
    operation: String,
    cpu_time_ms: f64,
    gpu_time_ms: f64,
    speedup: f64,
    problem_size: usize,
    accuracy_error: f64,
}

/// Benchmark configuration
struct BenchmarkConfig {
    warmup_iterations: usize,
    test_iterations: usize,
    matrix_sizes: Vec<usize>,
    batch_sizes: Vec<usize>,
    network_architectures: Vec<Vec<usize>>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            test_iterations: 20,
            matrix_sizes: vec![64, 128, 256, 512, 1024, 2048],
            batch_sizes: vec![1, 8, 16, 32, 64, 128],
            network_architectures: vec![
                vec![2, 4, 1],          // XOR problem
                vec![10, 20, 10, 5],    // Medium network
                vec![784, 128, 64, 10], // MNIST-sized network
            ],
        }
    }
}

/// Matrix multiplication performance test
#[cfg(feature = "webgpu")]
#[tokio::test]
async fn test_matrix_multiplication_performance() {
    let config = BenchmarkConfig::default();
    let mut results = Vec::new();

    // Initialize GPU backend
    let gpu_backend = match WebGPUBackend::new().await {
        Ok(backend) => backend,
        Err(e) => {
            eprintln!("GPU not available: {:?}, skipping GPU tests", e);
            return;
        }
    };

    for size in &config.matrix_sizes {
        // Generate test data
        let matrix: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001).collect();
        let vector: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        // CPU baseline
        let cpu_start = Instant::now();
        for _ in 0..config.test_iterations {
            let _result = cpu_matrix_vector_multiply(&matrix, &vector, *size, *size);
        }
        let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0 / config.test_iterations as f64;

        // GPU test
        let gpu_start = Instant::now();
        for _ in 0..config.test_iterations {
            let _result = gpu_backend
                .matrix_vector_multiply(&matrix, &vector, *size, *size)
                .expect("GPU operation failed");
        }
        let gpu_time = gpu_start.elapsed().as_secs_f64() * 1000.0 / config.test_iterations as f64;

        // Verify accuracy
        let cpu_result = cpu_matrix_vector_multiply(&matrix, &vector, *size, *size);
        let gpu_result = gpu_backend
            .matrix_vector_multiply(&matrix, &vector, *size, *size)
            .expect("GPU operation failed");

        let accuracy_error = calculate_max_error(&cpu_result, &gpu_result);

        let result = PerformanceResult {
            operation: format!("matrix_multiply_{}x{}", size, size),
            cpu_time_ms: cpu_time,
            gpu_time_ms: gpu_time,
            speedup: cpu_time / gpu_time,
            problem_size: *size,
            accuracy_error,
        };

        println!("{:#?}", result);
        results.push(result);
    }

    // Validate performance improvements
    for result in &results {
        if result.problem_size >= 256 {
            assert!(
                result.speedup >= 5.0,
                "Expected at least 5x speedup for size {}, got {:.2}x",
                result.problem_size,
                result.speedup
            );
        }
        assert!(
            result.accuracy_error < 1e-5,
            "Accuracy error too high: {}",
            result.accuracy_error
        );
    }
}

/// Activation function performance test
#[cfg(feature = "webgpu")]
#[tokio::test]
async fn test_activation_function_performance() {
    let config = BenchmarkConfig::default();
    let mut results = Vec::new();

    // Initialize GPU backend
    let gpu_backend = match WebGPUBackend::new().await {
        Ok(backend) => backend,
        Err(e) => {
            eprintln!("GPU not available: {:?}, skipping GPU tests", e);
            return;
        }
    };

    let activation_functions = vec![
        (ActivationFunction::ReLU, "ReLU"),
        (ActivationFunction::Sigmoid, "Sigmoid"),
        (ActivationFunction::Tanh, "Tanh"),
        (ActivationFunction::Linear, "Linear"),
    ];

    let test_sizes = vec![1000, 10000, 100000, 1000000];

    for size in &test_sizes {
        let inputs: Vec<f32> = (0..*size).map(|i| ((i as f32) * 0.001 - 0.5)).collect();

        for (activation, name) in &activation_functions {
            // CPU baseline
            let cpu_start = Instant::now();
            for _ in 0..config.test_iterations {
                let _result = cpu_apply_activation(&inputs, *activation, 1.0);
            }
            let cpu_time =
                cpu_start.elapsed().as_secs_f64() * 1000.0 / config.test_iterations as f64;

            // GPU test
            let gpu_start = Instant::now();
            for _ in 0..config.test_iterations {
                let _result = gpu_backend
                    .apply_activation_function(&inputs, *activation, 1.0)
                    .expect("GPU operation failed");
            }
            let gpu_time =
                gpu_start.elapsed().as_secs_f64() * 1000.0 / config.test_iterations as f64;

            let result = PerformanceResult {
                operation: format!("activation_{}_{}", name, size),
                cpu_time_ms: cpu_time,
                gpu_time_ms: gpu_time,
                speedup: cpu_time / gpu_time,
                problem_size: *size,
                accuracy_error: 0.0, // Will be validated separately
            };

            println!("{:#?}", result);
            results.push(result);
        }
    }

    // Validate significant speedup for large problems
    for result in &results {
        if result.problem_size >= 100000 {
            assert!(
                result.speedup >= 10.0,
                "Expected at least 10x speedup for large activation functions, got {:.2}x",
                result.speedup
            );
        }
    }
}

/// Neural network training performance test
#[cfg(feature = "webgpu")]
#[tokio::test]
async fn test_neural_network_training_performance() {
    let config = BenchmarkConfig::default();
    let mut results = Vec::new();

    for architecture in &config.network_architectures {
        // Create training data
        let input_size = architecture[0];
        let output_size = architecture[architecture.len() - 1];
        let training_samples = 100;

        let inputs: Vec<Vec<f32>> = (0..training_samples)
            .map(|_| (0..input_size).map(|_| rand::random::<f32>()).collect())
            .collect();
        let outputs: Vec<Vec<f32>> = (0..training_samples)
            .map(|_| (0..output_size).map(|_| rand::random::<f32>()).collect())
            .collect();

        let training_data = TrainingData::new(inputs.clone(), outputs.clone())
            .expect("Failed to create training data");

        // CPU network
        let mut cpu_network = NetworkBuilder::<f32>::new();
        for (i, &layer_size) in architecture.iter().enumerate() {
            if i == 0 {
                cpu_network = cpu_network.input_layer(layer_size);
            } else if i == architecture.len() - 1 {
                cpu_network = cpu_network.output_layer_with_activation(
                    layer_size,
                    ActivationFunction::Sigmoid,
                    1.0,
                );
            } else {
                cpu_network = cpu_network.hidden_layer_with_activation(
                    layer_size,
                    ActivationFunction::Sigmoid,
                    1.0,
                );
            }
        }
        let mut cpu_network = cpu_network.build();

        // GPU network
        let mut gpu_network = cpu_network.clone();
        #[cfg(feature = "webgpu")]
        {
            if let Ok(selector) = BackendSelector::new().with_gpu().await {
                gpu_network.set_backend_selector(selector);
            }
        }

        // Training performance test
        let epochs = 10;

        // CPU training
        let cpu_start = Instant::now();
        for _ in 0..epochs {
            cpu_network
                .train_epoch(&training_data, 0.1)
                .expect("Training failed");
        }
        let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0;

        // GPU training
        let gpu_start = Instant::now();
        for _ in 0..epochs {
            gpu_network
                .train_epoch(&training_data, 0.1)
                .expect("Training failed");
        }
        let gpu_time = gpu_start.elapsed().as_secs_f64() * 1000.0;

        let result = PerformanceResult {
            operation: format!("nn_training_{:?}", architecture),
            cpu_time_ms: cpu_time,
            gpu_time_ms: gpu_time,
            speedup: cpu_time / gpu_time,
            problem_size: architecture.iter().sum(),
            accuracy_error: 0.0,
        };

        println!("{:#?}", result);
        results.push(result);
    }

    // Validate performance improvements
    for result in &results {
        if result.problem_size >= 100 {
            assert!(
                result.speedup >= 3.0,
                "Expected at least 3x speedup for neural network training, got {:.2}x",
                result.speedup
            );
        }
    }
}

/// Batch processing performance test
#[cfg(feature = "webgpu")]
#[tokio::test]
async fn test_batch_processing_performance() {
    let config = BenchmarkConfig::default();
    let mut results = Vec::new();

    // Test network
    let network = NetworkBuilder::<f32>::new()
        .input_layer(100)
        .hidden_layer_with_activation(50, ActivationFunction::ReLU, 1.0)
        .output_layer_with_activation(10, ActivationFunction::Sigmoid, 1.0)
        .build();

    for batch_size in &config.batch_sizes {
        // Generate batch data
        let batch_inputs: Vec<Vec<f32>> = (0..*batch_size)
            .map(|_| (0..100).map(|_| rand::random::<f32>()).collect())
            .collect();

        // CPU batch processing
        let cpu_start = Instant::now();
        for _ in 0..config.test_iterations {
            for input in &batch_inputs {
                let _output = network.run(input).expect("Network run failed");
            }
        }
        let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0 / config.test_iterations as f64;

        // GPU batch processing (if available)
        #[cfg(feature = "webgpu")]
        {
            let mut gpu_network = network.clone();
            if let Ok(selector) = BackendSelector::new().with_gpu().await {
                gpu_network.set_backend_selector(selector);

                let gpu_start = Instant::now();
                for _ in 0..config.test_iterations {
                    let _outputs = gpu_network
                        .run_batch(&batch_inputs)
                        .expect("Batch processing failed");
                }
                let gpu_time =
                    gpu_start.elapsed().as_secs_f64() * 1000.0 / config.test_iterations as f64;

                let result = PerformanceResult {
                    operation: format!("batch_inference_{}", batch_size),
                    cpu_time_ms: cpu_time,
                    gpu_time_ms: gpu_time,
                    speedup: cpu_time / gpu_time,
                    problem_size: *batch_size,
                    accuracy_error: 0.0,
                };

                println!("{:#?}", result);
                results.push(result);
            }
        }
    }

    // Validate batch processing efficiency
    for result in &results {
        if result.problem_size >= 32 {
            assert!(
                result.speedup >= 5.0,
                "Expected at least 5x speedup for batch processing, got {:.2}x",
                result.speedup
            );
        }
    }
}

/// DAA integration performance test
#[cfg(all(feature = "webgpu", feature = "ruv-swarm"))]
#[tokio::test]
async fn test_daa_gpu_coordination_performance() {
    use ruv_swarm::{AgentType, SwarmBuilder, TaskType};

    // Initialize swarm with GPU-aware agents
    let swarm = SwarmBuilder::new()
        .with_topology(Topology::Hierarchical)
        .with_max_agents(8)
        .build()
        .await
        .expect("Failed to create swarm");

    // Spawn GPU-specialized agents
    let gpu_agent = swarm
        .spawn_agent(AgentType::GpuCompute)
        .await
        .expect("Failed to spawn GPU agent");

    // Test GPU task distribution
    let gpu_task = TaskType::GPU(GPUTask {
        operation: GPUOperation::NeuralTraining {
            network_id: "test_network".to_string(),
            batch_size: 128,
        },
        data_size: 10000,
        compute_requirements: ComputeRequirements::High,
        fallback_strategy: FallbackStrategy::Adaptive,
    });

    let start = Instant::now();
    let result = swarm
        .execute_task(gpu_task)
        .await
        .expect("GPU task execution failed");
    let execution_time = start.elapsed();

    println!("DAA GPU coordination time: {:?}", execution_time);
    assert!(
        execution_time.as_millis() < 1000,
        "GPU coordination took too long"
    );
}

/// Memory transfer overhead test
#[cfg(feature = "webgpu")]
#[tokio::test]
async fn test_memory_transfer_overhead() {
    let gpu_backend = match WebGPUBackend::new().await {
        Ok(backend) => backend,
        Err(_) => return,
    };

    let transfer_sizes = vec![1_000, 10_000, 100_000, 1_000_000, 10_000_000];

    for size in transfer_sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        // Measure upload time
        let upload_start = Instant::now();
        let buffer = gpu_backend
            .memory_manager()
            .allocate_buffer(size * 4)
            .expect("Buffer allocation failed");
        gpu_backend
            .memory_manager()
            .upload_data(buffer, &data)
            .expect("Upload failed");
        let upload_time = upload_start.elapsed();

        // Measure download time
        let download_start = Instant::now();
        let downloaded = gpu_backend
            .memory_manager()
            .download_data(buffer)
            .expect("Download failed");
        let download_time = download_start.elapsed();

        let bandwidth_upload = (size as f64 * 4.0) / upload_time.as_secs_f64() / 1e9;
        let bandwidth_download = (size as f64 * 4.0) / download_time.as_secs_f64() / 1e9;

        println!("Transfer size: {} elements", size);
        println!("  Upload: {:?} ({:.2} GB/s)", upload_time, bandwidth_upload);
        println!(
            "  Download: {:?} ({:.2} GB/s)",
            download_time, bandwidth_download
        );

        // Verify data integrity
        assert_eq!(data.len(), downloaded.len());
        for (i, (a, b)) in data.iter().zip(downloaded.iter()).enumerate().take(100) {
            assert!((a - b).abs() < 1e-6, "Data mismatch at index {}", i);
        }
    }
}

// Helper functions

fn cpu_matrix_vector_multiply(
    matrix: &[f32],
    vector: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut result = vec![0.0; rows];
    for i in 0..rows {
        for j in 0..cols {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
    result
}

fn cpu_apply_activation(inputs: &[f32], function: ActivationFunction, steepness: f32) -> Vec<f32> {
    inputs
        .iter()
        .map(|&x| match function {
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-steepness * x).exp()),
            ActivationFunction::Tanh => (steepness * x).tanh(),
            ActivationFunction::Linear => steepness * x,
            _ => x,
        })
        .collect()
}

fn calculate_max_error(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() as f64)
        .fold(0.0, f64::max)
}
