//! Backend abstraction layer

pub mod backend_trait;
pub mod native_gpu;
pub mod webgpu;
pub mod webgpu_optimized;
pub mod wasm_runtime;

pub use backend_trait::{BackendTrait, BackendCapabilities};

// Re-export BackendTrait as Backend for backward compatibility
pub use backend_trait::BackendTrait as Backend;

/// Get the current backend implementation
pub fn get_backend() -> Box<dyn Backend> {
    #[cfg(target_arch = "wasm32")]
    {
        Box::new(webgpu::WebGPUBackend::new())
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        #[cfg(feature = "cuda-backend")]
        {
            if native_gpu::is_cuda_available() {
                return Box::new(native_gpu::NativeGPUBackend::new());
            }
        }
        
        // Fallback to CPU backend
        Box::new(wasm_runtime::WasmRuntime::new())
    }
}
/// GPU runtime discovery is opt-in so CPU/parser users do not load drivers.
pub(crate) fn gpu_enabled() -> bool {
    cfg!(any(feature = "gpu", feature = "webgpu", feature = "webgpu-only",
        feature = "native-gpu", feature = "cuda-backend", feature = "vulkan"))
}
