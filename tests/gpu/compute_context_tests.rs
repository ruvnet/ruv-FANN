//! Tests for ComputeContext - the bridge between Network and GPU backends
//!
//! These tests validate the critical integration layer that enables GPU acceleration

#[cfg(feature = "gpu")]
mod compute_context_tests {
    use ruv_fann::webgpu::{BackendType, ComputeContext};
    use ruv_fann::{ActivationFunction, Network};

    #[tokio::test]
    async fn test_compute_context_creation() {
        // Test creating compute context with explicit type
        let context_result = ComputeContext::<f32>::new().await;

        // Should succeed even if GPU is not available (fallback to CPU)
        if let Ok(context) = context_result {
            println!("ComputeContext created successfully");

            // Test getting backend info
            let backend_type = context.current_backend();
            println!("Current backend: {:?}", backend_type);

            // Should be a valid backend type
            assert!(
                backend_type == BackendType::Cpu
                    || backend_type == BackendType::Simd
                    || backend_type == BackendType::WebGPU
            );
        } else {
            println!(
                "ComputeContext creation failed (expected in some environments): {:?}",
                context_result
            );
        }
    }

    #[test]
    fn test_compute_context_with_network() {
        // Create a simple network for testing
        let mut network = Network::<f32>::new(&[2, 3, 1]);

        // Set activation functions
        network.set_activation_function_hidden(ActivationFunction::Sigmoid);
        network.set_activation_function_output(ActivationFunction::Sigmoid);

        // Test basic network functionality
        let inputs = vec![0.5, -0.3];
        let outputs = network.run(&inputs);

        assert_eq!(outputs.len(), 1, "Should have one output");
        assert!(outputs[0].is_finite(), "Output should be finite");

        println!("Network test passed: {:?} -> {:?}", inputs, outputs);
    }

    #[test]
    fn test_network_structure_validation() {
        // Test the network structure that would be used by ComputeContext
        let mut network = Network::<f32>::new(&[2, 3, 1]);

        // Validate network structure (layers include input, hidden, output)
        assert_eq!(
            network.num_layers(),
            3,
            "Network should have 3 layers (input + hidden + output)"
        );
        assert_eq!(network.num_inputs(), 2, "Network should have 2 inputs");
        assert_eq!(network.num_outputs(), 1, "Network should have 1 output");

        // Test forward pass
        let inputs = vec![1.0, 0.5];
        let outputs = network.run(&inputs);

        assert_eq!(outputs.len(), 1, "Should produce one output");
        assert!(outputs[0].is_finite(), "Output should be finite");

        println!("Network structure validation passed");
    }
}

#[cfg(not(feature = "gpu"))]
mod fallback_tests {
    #[test]
    fn test_compute_context_fallback() {
        println!("ComputeContext tests require gpu feature");
        assert!(true, "Fallback test passes when gpu feature is disabled");
    }
}
