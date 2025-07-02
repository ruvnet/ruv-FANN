//! WebGPU acceleration module for ruv-FANN
//! 
//! This module provides GPU acceleration for neural network computations using WebGPU.
//! It features automatic fallback to CPU when GPU is unavailable and intelligent backend
//! selection based on workload characteristics.

#[cfg(feature = "gpu")]
pub mod backend;
#[cfg(feature = "gpu")]
pub mod memory;
#[cfg(feature = "gpu")]
pub mod error;
#[cfg(feature = "gpu")]
pub mod circuit_breaker;
#[cfg(feature = "gpu")]
pub mod shaders;
#[cfg(feature = "gpu")]
pub mod device;
#[cfg(feature = "gpu")]
pub mod webgpu_backend;
#[cfg(feature = "gpu")]
pub mod compute_context;

#[cfg(feature = "gpu")]
pub use backend::{ComputeBackend, BackendSelector, BackendType};
#[cfg(feature = "gpu")]
pub use memory::{GpuMemoryManager, BufferPool};
#[cfg(feature = "gpu")]
pub use error::{ComputeError, ComputeResult};
#[cfg(feature = "gpu")]
pub use device::{GpuDevice, DeviceInfo};
#[cfg(feature = "gpu")]
pub use webgpu_backend::WebGpuBackend;
#[cfg(feature = "gpu")]
pub use compute_context::{ComputeContext, ComputePerformanceStats};

#[cfg(not(feature = "gpu"))]
pub use crate::network::Network as GpuNetwork;

/// GPU configuration for neural network acceleration
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Enable automatic backend selection
    pub auto_backend: bool,
    /// Memory limit in bytes (0 = unlimited)
    pub memory_limit: u64,
    /// Enable debug logging
    pub debug: bool,
}

#[cfg(feature = "gpu")]
impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            auto_backend: true,
            memory_limit: 0,
            debug: false,
        }
    }
}

#[cfg(feature = "gpu")]
/// Initialize GPU context for ruv-FANN
/// 
/// This should be called once at the start of your application to set up the GPU context.
/// Returns true if GPU acceleration is available, false otherwise.
pub async fn initialize_gpu() -> bool {
    match GpuDevice::new().await {
        Ok(_) => {
            log::info!("GPU acceleration initialized successfully");
            true
        }
        Err(e) => {
            log::warn!("GPU acceleration unavailable: {}", e);
            false
        }
    }
}

#[cfg(not(feature = "gpu"))]
/// GPU initialization stub when GPU feature is disabled
pub async fn initialize_gpu() -> bool {
    false
}

/// Check if GPU acceleration is available
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        GpuDevice::is_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_default() {
        #[cfg(feature = "gpu")]
        {
            let config = GpuConfig::default();
            assert!(config.auto_backend);
            assert_eq!(config.memory_limit, 0);
            assert!(!config.debug);
        }
    }

    #[test]
    fn test_gpu_availability() {
        // This test works both with and without GPU feature
        let available = is_gpu_available();
        #[cfg(feature = "gpu")]
        {
            // GPU availability depends on runtime environment
            println!("GPU available: {}", available);
        }
        #[cfg(not(feature = "gpu"))]
        {
            assert!(!available);
        }
    }
}