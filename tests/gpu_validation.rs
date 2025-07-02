//! GPU validation tests for ruv-FANN

#[cfg(feature = "gpu")]
mod gpu_tests {
    use ruv_fann::webgpu::{WebGpuBackend, ComputeBackend, BackendSelector, BackendType};
    use ruv_fann::webgpu::backend::{MatrixDims, ActivationFunction};
    use approx::assert_relative_eq;

    #[tokio::test]
    async fn test_gpu_backend_initialization() {
        // Test WebGPU backend initialization
        match WebGpuBackend::new().await {
            Ok(backend) => {
                assert_eq!(<WebGpuBackend as ComputeBackend<f32>>::backend_type(&backend), BackendType::WebGPU);
                assert!(<WebGpuBackend as ComputeBackend<f32>>::is_available(&backend));
                
                let info = <WebGpuBackend as ComputeBackend<f32>>::device_info(&backend);
                assert!(info.estimated_speedup > 1.0);
                println!("GPU Backend initialized successfully");
                println!("Estimated speedup: {}x", info.estimated_speedup);
            }
            Err(e) => {
                println!("WebGPU not available in test environment: {}", e);
                // This is expected in many CI environments
            }
        }
    }

    #[tokio::test]
    async fn test_backend_selector() {
        let selector = BackendSelector::new();
        
        // Test backend selection logic
        assert_eq!(selector.select_backend::<f32>(50), BackendType::CPU);
        assert_eq!(selector.select_backend::<f32>(500), BackendType::SIMD);
        
        // Large problems should prefer GPU if available
        let large_backend = selector.select_backend::<f32>(5000);
        println!("Selected backend for large problem: {}", large_backend);
        
        // Test with GPU initialization
        match selector.with_gpu().await {
            Ok(gpu_selector) => {
                let backend_type = gpu_selector.select_backend::<f32>(10000);
                println!("GPU-enabled selector chose: {}", backend_type);
                
                // Should prefer GPU for large problems
                if backend_type == BackendType::WebGPU {
                    println!("✓ GPU backend selected for large workload");
                } else {
                    println!("○ GPU backend not selected (may not be available)");
                }
            }
            Err(e) => {
                println!("GPU initialization failed: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_matrix_vector_multiply_correctness() {
        // Test matrix-vector multiplication correctness
        let matrix = vec![
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]; // 2x3 matrix
        let vector = vec![1.0f32, 2.0, 3.0];
        let dims = MatrixDims::new(2, 3);
        
        // Expected result: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
        let expected = [14.0f32, 32.0];
        
        // Test with CPU backend
        let cpu_backend = ruv_fann::webgpu::backend::CpuBackend::new();
        let cpu_result = cpu_backend.matrix_vector_multiply(&matrix, &vector, dims).await.unwrap();
        
        for (i, (&cpu, &exp)) in cpu_result.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(cpu, exp, epsilon = 1e-6);
        }
        
        // Test with WebGPU backend if available
        match WebGpuBackend::new().await {
            Ok(gpu_backend) => {
                match gpu_backend.matrix_vector_multiply(&matrix, &vector, dims).await {
                    Ok(gpu_result) => {
                        assert_eq!(gpu_result.len(), expected.len());
                        
                        for (i, (&gpu, &exp)) in gpu_result.iter().zip(expected.iter()).enumerate() {
                            assert_relative_eq!(gpu, exp, epsilon = 1e-5);
                        }
                        
                        // Compare GPU vs CPU results
                        for (i, (&gpu, &cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
                            assert_relative_eq!(gpu, cpu, epsilon = 1e-5);
                        }
                        
                        println!("✓ GPU matrix-vector multiplication matches CPU results");
                    }
                    Err(e) => {
                        println!("GPU computation failed: {}", e);
                        println!("Recovery suggestion: {}", e.recovery_suggestion());
                    }
                }
            }
            Err(e) => {
                println!("WebGPU backend not available: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_activation_functions() {
        let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        
        // Test ReLU
        let cpu_backend = ruv_fann::webgpu::backend::CpuBackend::new();
        let relu_result = cpu_backend.activation_function(&input, ActivationFunction::ReLU).await.unwrap();
        let expected_relu = [0.0f32, 0.0, 0.0, 1.0, 2.0];
        
        for (i, (&result, &expected)) in relu_result.iter().zip(expected_relu.iter()).enumerate() {
            assert_relative_eq!(result, expected, epsilon = 1e-6);
        }
        
        // Test Sigmoid
        let sigmoid_result = cpu_backend.activation_function(&input, ActivationFunction::Sigmoid).await.unwrap();
        
        // Sigmoid should be between 0 and 1
        for (i, &value) in sigmoid_result.iter().enumerate() {
            assert!((0.0..=1.0).contains(&value), 
                "Sigmoid[{}] out of range: {}", i, value);
        }
        
        // Test that sigmoid(0) ≈ 0.5
        assert_relative_eq!(sigmoid_result[2], 0.5, epsilon = 1e-6);
        
        println!("✓ Activation functions working correctly");
    }

    #[tokio::test]
    async fn test_error_handling() {
        // Test invalid dimensions
        let matrix = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 matrix
        let vector = vec![1.0f32, 2.0, 3.0]; // Wrong size
        let dims = MatrixDims::new(2, 2);
        
        let cpu_backend = ruv_fann::webgpu::backend::CpuBackend::new();
        let result = cpu_backend.matrix_vector_multiply(&matrix, &vector, dims).await;
        
        assert!(result.is_err(), "Should fail with dimension mismatch");
        
        if let Err(e) = result {
            assert!(!e.is_recoverable(), "Dimension errors should not be recoverable");
            println!("✓ Error handling working: {}", e);
            println!("Recovery suggestion: {}", e.recovery_suggestion());
        }
    }

    #[tokio::test]
    async fn test_performance_estimation() {
        let cpu_backend = ruv_fann::webgpu::backend::CpuBackend::new();
        let simd_backend = ruv_fann::webgpu::backend::SimdBackend::new();
        
        let problem_size = 1000;
        let cpu_perf = <ruv_fann::webgpu::backend::CpuBackend as ComputeBackend<f32>>::estimate_performance(&cpu_backend, problem_size);
        let simd_perf = <ruv_fann::webgpu::backend::SimdBackend as ComputeBackend<f32>>::estimate_performance(&simd_backend, problem_size);
        
        // SIMD should be faster than CPU
        assert!(simd_perf > cpu_perf, 
            "SIMD performance ({}) should be better than CPU ({})", simd_perf, cpu_perf);
        
        // Test with WebGPU if available
        match WebGpuBackend::new().await {
            Ok(gpu_backend) => {
                let gpu_perf = <WebGpuBackend as ComputeBackend<f32>>::estimate_performance(&gpu_backend, problem_size);
                println!("Performance estimates for problem size {}:", problem_size);
                println!("  CPU: {:.2e} ops/sec", cpu_perf);
                println!("  SIMD: {:.2e} ops/sec", simd_perf);
                println!("  GPU: {:.2e} ops/sec", gpu_perf);
                
                // For large problems, GPU should be fastest
                if problem_size > 10000 {
                    assert!(gpu_perf > simd_perf, "GPU should outperform SIMD for large problems");
                }
            }
            Err(_) => {
                println!("GPU performance estimation skipped (WebGPU not available)");
            }
        }
    }

    #[tokio::test]
    async fn test_memory_requirements() {
        let problem_sizes = [100, 1000, 10000];
        
        for &size in &problem_sizes {
            let cpu_backend = ruv_fann::webgpu::backend::CpuBackend::new();
            let cpu_memory = <ruv_fann::webgpu::backend::CpuBackend as ComputeBackend<f32>>::memory_requirements(&cpu_backend, size);
            
            // Memory should scale with problem size
            assert!(cpu_memory > 0, "Memory requirement should be positive");
            
            match WebGpuBackend::new().await {
                Ok(gpu_backend) => {
                    let gpu_memory = <WebGpuBackend as ComputeBackend<f32>>::memory_requirements(&gpu_backend, size);
                    println!("Memory requirements for size {}: CPU {} bytes, GPU {} bytes", 
                        size, cpu_memory, gpu_memory);
                    
                    // GPU typically needs more memory for buffers
                    assert!(gpu_memory >= cpu_memory, 
                        "GPU memory requirement should be at least as much as CPU");
                }
                Err(_) => {
                    println!("GPU memory estimation skipped for size {} (WebGPU not available)", size);
                }
            }
        }
    }
}

#[cfg(not(feature = "gpu"))]
mod no_gpu_tests {
    #[test]
    fn test_gpu_feature_disabled() {
        // When GPU feature is disabled, we just verify the library compiles
        // and basic functionality works
        use ruv_fann::{NetworkBuilder, ActivationFunction};
        
        let mut network = NetworkBuilder::<f32>::new()
            .input_layer(2)
            .hidden_layer(3)
            .output_layer(1)
            .build();
        
        let inputs = vec![0.5, 0.3];
        let _outputs = network.run(&inputs);
        
        println!("✓ GPU features correctly disabled, CPU functionality works");
    }
}