//! Tests for WebGPU compute backend

#[cfg(test)]
mod webgpu_tests {
    use crate::webgpu::{BackendSelector, ComputeProfile, MatrixSize, OperationType};

    // Helper function to check if we're running in CI
    fn is_ci_environment() -> bool {
        std::env::var("RUV_FANN_CI_TESTING").is_ok()
    }

    #[test]
    fn test_backend_selector_creation() {
        // Skip test in CI environment
        if is_ci_environment() {
            println!("Skipping WebGPU test in CI environment");
            return;
        }

        let selector = BackendSelector::<f32>::new();
        let capabilities = selector.capabilities();

        // Should have at least one backend (CPU fallback)
        assert!(!capabilities.is_empty());

        // All backends should support f32
        for cap in &capabilities {
            assert!(cap.supports_f32);
        }
    }

    #[test]
    fn test_compute_profile_selection() {
        // Skip test in CI environment
        if is_ci_environment() {
            println!("Skipping WebGPU test in CI environment");
            return;
        }

        let selector = BackendSelector::<f32>::new();

        let profiles = vec![
            ComputeProfile {
                matrix_size: MatrixSize::Small,
                batch_size: 1,
                operation_type: OperationType::ForwardPass,
            },
            ComputeProfile {
                matrix_size: MatrixSize::Large,
                batch_size: 32,
                operation_type: OperationType::Inference,
            },
        ];

        for profile in profiles {
            let backend = selector.select_backend(&profile);
            assert!(backend.is_some(), "Should always find a backend");
        }
    }
}
