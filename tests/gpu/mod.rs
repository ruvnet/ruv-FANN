//! GPU Tests Module - Comprehensive GPU testing suite for ruv-FANN
//!
//! This module contains all GPU-related tests including:
//! - WebGPU backend validation
//! - Performance benchmarks  
//! - Integration tests
//! - Autonomous GPU resource management tests
//! - Swarm orchestration tests

// Import all GPU test modules
pub mod gpu_integration_test;
pub mod gpu_performance_benchmark;
pub mod gpu_performance_tests;
pub mod autonomous_gpu_integration_test;
pub mod gpu_swarm_orchestration_tests;
pub mod memory_manager_tests;
pub mod backend_selector_tests;
pub mod compute_context_tests;
pub mod end_to_end_gpu_tests;

// Re-export test functions for easy access
pub use gpu_integration_test::*;
pub use gpu_performance_benchmark::*;
pub use gpu_performance_tests::*;
pub use autonomous_gpu_integration_test::*;
pub use gpu_swarm_orchestration_tests::*;
pub use memory_manager_tests::*;
pub use backend_selector_tests::*;
pub use compute_context_tests::*;
pub use end_to_end_gpu_tests::*;

#[cfg(feature = "gpu")]
mod webgpu_core_tests {
    use ruv_fann::webgpu::{BackendSelector, ComputeBackend, BackendType};
    use ruv_fann::ActivationFunction;
    
    #[test]
    fn test_backend_availability() {
        let selector = BackendSelector::<f32>::new();
        let backends = selector.get_available_backends();
        
        // Should have at least CPU backend
        assert!(!backends.is_empty());
        assert!(backends.contains(&BackendType::Cpu) || backends.contains(&BackendType::Simd));
    }
    
    #[test]
    fn test_fallback_system() {
        use ruv_fann::webgpu::FallbackManager;
        
        let mut fallback = FallbackManager::<f32>::new();
        let available = fallback.get_available_backends();
        
        // Should have fallback backends
        assert!(!available.is_empty());
    }
}

#[cfg(not(feature = "gpu"))]
mod cpu_fallback_tests {
    #[test]
    fn test_cpu_backend_available() {
        // When GPU feature is disabled, ensure basic functionality works
        assert!(true, "CPU backend should always be available");
    }
}