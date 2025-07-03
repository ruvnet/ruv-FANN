//! Integration test for GPU module

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_module_exists() {
    // Import the webgpu module to ensure it compiles
    use ruv_fann::webgpu::{BackendSelector, BackendType};

    // Create a backend selector
    let selector = BackendSelector::<f32>::new();

    // Get available backends
    let backends = selector.capabilities();

    // Should have at least one backend
    assert!(!backends.is_empty());
    println!("Available backends: {}", backends.len());

    // Check that we have CPU or SIMD backend
    for cap in &backends {
        println!("Backend supports f32: {}", cap.supports_f32);
        assert!(cap.supports_f32);
    }
}

#[cfg(not(feature = "gpu"))]
#[test]
fn test_gpu_module_disabled() {
    println!("GPU module is disabled - gpu feature not enabled");
    // Test passes when GPU features are not available
}
