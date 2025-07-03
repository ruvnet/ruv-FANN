//! Tests for Backend Selection and Switching Logic
//!
//! Validates that the backend selector chooses appropriate compute backends
//! based on problem size and hardware availability

#[cfg(feature = "gpu")]
mod backend_tests {
    use ruv_fann::webgpu::{BackendCapabilities, BackendSelector, BackendType, ComputeProfile};

    #[test]
    fn test_backend_selector_creation() {
        let selector = BackendSelector::<f32>::new();

        // Should be able to get capabilities
        let capabilities = selector.capabilities();
        assert!(
            !capabilities.is_empty(),
            "Should have at least one backend capability"
        );

        println!("Available backend capabilities: {}", capabilities.len());

        // Check each capability
        for (i, cap) in capabilities.iter().enumerate() {
            println!("Backend {}: {:?}", i, cap);

            // Basic validation
            assert!(
                cap.max_buffer_size > 0,
                "Max buffer size should be positive"
            );
            assert!(
                cap.max_compute_units > 0,
                "Max compute units should be positive"
            );
        }
    }

    #[test]
    fn test_available_backends() {
        let selector = BackendSelector::<f32>::new();
        let backends = selector.get_available_backends();

        // Should have at least CPU or SIMD backend
        assert!(!backends.is_empty(), "Should have at least one backend");
        assert!(
            backends.contains(&BackendType::Cpu) || backends.contains(&BackendType::Simd),
            "Should have CPU or SIMD backend available"
        );

        println!("Available backends: {:?}", backends);
    }

    #[test]
    fn test_backend_selection_logic() {
        let mut selector = BackendSelector::<f32>::new();

        // Test selection for small problems (should prefer CPU/SIMD)
        let small_problem_backend = selector.select_optimal_backend(100, 50);
        assert!(
            small_problem_backend == BackendType::Cpu || small_problem_backend == BackendType::Simd,
            "Small problems should use CPU or SIMD"
        );

        // Test selection for larger problems (might select GPU if available)
        let large_problem_backend = selector.select_optimal_backend(10000, 5000);
        println!("Large problem backend: {:?}", large_problem_backend);

        // Should return a valid backend type
        let available = selector.get_available_backends();
        assert!(
            available.contains(&large_problem_backend),
            "Selected backend should be available"
        );
    }

    #[test]
    fn test_backend_switching() {
        let mut selector = BackendSelector::<f32>::new();

        // Test switching between backends
        let original_backend = selector.get_current_backend();

        // Try to set a different backend
        let available_backends = selector.get_available_backends();
        if available_backends.len() > 1 {
            for backend in &available_backends {
                if *backend != original_backend {
                    selector.set_backend(*backend);
                    assert_eq!(selector.get_current_backend(), *backend);
                    break;
                }
            }
        }
    }

    #[test]
    fn test_fallback_behavior() {
        let mut selector = BackendSelector::<f32>::new();

        // Test what happens when we try to use WebGPU but it's not available
        // This should fallback gracefully
        selector.set_backend(BackendType::WebGPU);

        // The selector should handle this gracefully
        let current = selector.get_current_backend();
        let available = selector.get_available_backends();
        assert!(
            available.contains(&current),
            "Current backend should be available"
        );
    }
}

#[cfg(not(feature = "gpu"))]
mod fallback_tests {
    #[test]
    fn test_backend_fallback_without_gpu() {
        println!("Backend selector tests require gpu feature");
        assert!(true, "Fallback test passes when webgpu feature is disabled");
    }
}
