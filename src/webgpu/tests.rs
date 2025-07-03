//! Tests for WebGPU compute backend

#[cfg(test)]
mod tests {

    #[test]
    fn test_backend_selector_creation() {
        let selector = crate::webgpu::BackendSelector::<f32>::new();
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
        let selector = crate::webgpu::BackendSelector::<f32>::new();

        let profiles = vec![
            crate::webgpu::ComputeProfile {
                matrix_size: crate::webgpu::MatrixSize::Small,
                batch_size: 1,
                operation_type: crate::webgpu::OperationType::ForwardPass,
            },
            crate::webgpu::ComputeProfile {
                matrix_size: crate::webgpu::MatrixSize::Large,
                batch_size: 32,
                operation_type: crate::webgpu::OperationType::Inference,
            },
        ];

        for profile in profiles {
            let backend = selector.select_backend(&profile);
            assert!(backend.is_some(), "Should always find a backend");
        }
    }
}
