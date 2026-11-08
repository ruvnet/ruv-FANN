//! Neural Network Integration for CUDA-WASM with ruv-FANN
//!
//! This module provides seamless integration between CUDA-WASM transpiler
//! and ruv-FANN neural networks for GPU-accelerated neural computation.
//!
//! Features:
//! - Automatic CUDA-to-WGSL transpilation for neural operations
//! - GPU-accelerated forward/backward propagation
//! - Memory-efficient data transfer between CPU and GPU
//! - Automatic fallback to CPU when GPU unavailable
//! - Performance monitoring and profiling
//! - TypeScript bindings for web usage
//! - 5x+ speedup for neural network operations

pub mod bridge;
pub mod cuda_kernels;
pub mod gpu_neural_ops;
pub mod memory_manager;
pub mod performance_monitor;
pub mod wasm_bindings;
pub mod wasm_types;
pub mod examples;
pub mod benchmarks;

use crate::{CudaRust, Result as CudaResult};
use std::sync::Arc;
use std::marker::PhantomData;
use thiserror::Error;

/// Errors specific to neural integration
#[derive(Error, Debug)]
pub enum NeuralIntegrationError {
    #[error("CUDA transpilation failed: {0}")]
    TranspilationError(String),
    
    #[error("GPU initialization failed: {0}")]
    GpuInitError(String),
    
    #[error("Memory allocation failed: {0}")]
    MemoryError(String),
    
    #[error("Neural operation failed: {0}")]
    OperationError(String),
    
    #[error("Performance degradation detected: {0}")]
    PerformanceError(String),
    
    #[error("Type conversion error: {0}")]
    TypeError(String),
}

pub type NeuralResult<T> = std::result::Result<T, NeuralIntegrationError>;

/// Main integration interface between CUDA-WASM and ruv-FANN
pub struct NeuralBridge {
    cuda_transpiler: CudaRust,
    gpu_backend: Option<Arc<dyn GpuBackendTrait>>,
    memory_manager: Arc<dyn MemoryManagerTrait>,
    performance_monitor: Arc<dyn PerformanceMonitorTrait>,
    config: BridgeConfig,
}

/// Configuration for the neural bridge
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Whether to enable GPU acceleration
    pub enable_gpu: bool,
    /// GPU device preference
    pub gpu_device: GpuDevice,
    /// Memory pool size in MB
    pub memory_pool_size: usize,
    /// Whether to enable performance monitoring
    pub enable_monitoring: bool,
    /// Automatic fallback to CPU if GPU fails
    pub auto_fallback: bool,
    /// Batch size for operations
    pub batch_size: usize,
    /// Precision level
    pub precision: Precision,
}

/// GPU device preference
#[derive(Debug, Clone, Copy)]
pub enum GpuDevice {
    Auto,
    HighPerformance,
    LowPower,
    Discrete,
    Integrated,
}

/// Precision level for computations
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    Float16,
    Float32,
    Float64,
}

/// GPU backends are thread-safe on native targets and local to the browser on WASM.
pub trait GpuBackendTrait: wgpu::WasmNotSendSync {
    fn initialize(&self) -> NeuralResult<()>;
    fn is_available(&self) -> bool;
    fn get_device_info(&self) -> DeviceInfo;
    fn create_buffer(&self, size: usize) -> NeuralResult<BufferHandle>;
    fn execute_kernel(&self, kernel: &CompiledKernel, inputs: &[BufferHandle]) -> NeuralResult<BufferHandle>;
}

/// Memory managers are thread-safe on native targets and local to the browser on WASM.
pub trait MemoryManagerTrait: wgpu::WasmNotSendSync {
    fn allocate(&self, size: usize) -> NeuralResult<MemoryHandle>;
    fn deallocate(&self, handle: MemoryHandle) -> NeuralResult<()>;
    fn transfer_to_gpu(&self, data: &[f32]) -> NeuralResult<BufferHandle>;
    fn transfer_from_gpu(&self, buffer: BufferHandle) -> NeuralResult<Vec<f32>>;
    fn get_memory_stats(&self) -> MemoryStats;
}

/// Trait for performance monitoring
pub trait PerformanceMonitorTrait: Send + Sync {
    fn start_operation(&self, name: &str) -> OperationHandle;
    fn end_operation(&self, handle: OperationHandle) -> NeuralResult<OperationStats>;
    fn get_performance_summary(&self) -> PerformanceStats;
    fn detect_degradation(&self) -> Option<PerformanceDegradation>;
}

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor: String,
    pub device_type: String,
    pub memory_size: usize,
    pub compute_units: u32,
    pub max_workgroup_size: u32,
    pub supports_f16: bool,
    pub supports_f64: bool,
}

/// Handle types
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferHandle(u64);
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct MemoryHandle(u64);
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct OperationHandle(u64);

/// Compiled kernel representation
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub name: String,
    pub wgsl_source: String,
    pub entry_point: String,
    pub workgroup_size: [u32; 3],
    pub bind_group_layout: Vec<BindingType>,
}

/// Binding types for shaders
#[derive(Debug, Clone)]
pub enum BindingType {
    Buffer { read_only: bool },
    UniformBuffer,
    StorageTexture,
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub gpu_allocated: usize,
    pub cpu_allocated: usize,
    pub peak_usage: usize,
    pub allocations: u64,
    pub deallocations: u64,
}

/// Performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub total_operations: u64,
    pub average_execution_time: f64,
    pub gpu_utilization: f32,
    pub memory_bandwidth: f64,
    pub throughput: f64,
}

/// Operation statistics
#[derive(Debug, Clone)]
pub struct OperationStats {
    pub name: String,
    pub execution_time: f64,
    pub gpu_time: f64,
    pub memory_transfer_time: f64,
    pub throughput: f64,
}

/// Performance degradation information
#[derive(Debug, Clone)]
pub struct PerformanceDegradation {
    pub operation: String,
    pub expected_time: f64,
    pub actual_time: f64,
    pub degradation_factor: f64,
    pub suggested_action: String,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            enable_gpu: crate::backend::gpu_enabled(),
            gpu_device: GpuDevice::Auto,
            memory_pool_size: 512, // 512 MB
            enable_monitoring: true,
            auto_fallback: true,
            batch_size: 32,
            precision: Precision::Float32,
        }
    }
}

impl NeuralBridge {
    /// Create a new neural bridge with default configuration
    pub fn new() -> NeuralResult<Self> {
        Self::with_config(BridgeConfig::default())
    }
    
    /// Create a new neural bridge with custom configuration
    pub fn with_config(config: BridgeConfig) -> NeuralResult<Self> {
        let cuda_transpiler = CudaRust::new();
        
        // Initialize GPU backend if enabled
        let gpu_backend = if config.enable_gpu {
            match bridge::WebGpuBackend::new(&config) {
                Ok(backend) => Some(Arc::new(backend) as Arc<dyn GpuBackendTrait>),
                Err(e) => {
                    if config.auto_fallback {
                        log::warn!("GPU initialization failed, falling back to CPU: {e}");
                        None
                    } else {
                        return Err(NeuralIntegrationError::GpuInitError(e.to_string()));
                    }
                }
            }
        } else {
            None
        };
        
        // Initialize memory manager
        let memory_manager = Arc::new(memory_manager::HybridMemoryManager::new(&config)?);
        
        // Initialize performance monitor
        let performance_monitor: Arc<dyn PerformanceMonitorTrait> = if config.enable_monitoring {
            Arc::new(performance_monitor::RealTimeMonitor::new()?)
        } else {
            Arc::new(performance_monitor::NoOpMonitor::new())
        };
        
        Ok(Self {
            cuda_transpiler,
            gpu_backend,
            memory_manager,
            performance_monitor,
            config,
        })
    }
    
    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_backend.as_ref().is_some_and(|b| b.is_available())
    }
    
    /// Get device information
    pub fn get_device_info(&self) -> Option<DeviceInfo> {
        self.gpu_backend.as_ref().map(|b| b.get_device_info())
    }
    
    /// Transpile CUDA kernel to WGSL
    pub fn transpile_cuda_kernel(&self, cuda_source: &str) -> NeuralResult<CompiledKernel> {
        // Use CUDA-WASM transpiler to convert CUDA to Rust/WGSL
        let rust_code = self.cuda_transpiler
            .transpile(cuda_source)
            .map_err(|e| NeuralIntegrationError::TranspilationError(e.to_string()))?;
        
        // Extract WGSL from transpiled code (implementation in bridge module)
        bridge::extract_wgsl_from_rust(&rust_code)
    }
    
    /// Execute a neural operation with automatic optimization
    pub fn execute_neural_operation<T>(
        &self,
        operation: NeuralOperation<T>,
        inputs: &[T],
    ) -> NeuralResult<Vec<T>>
    where
        T: Clone + Send + Sync + 'static + bytemuck::Pod + num_traits::Float,
    {
        let handle = self.performance_monitor.start_operation(&operation.name());
        
        let result = if let Some(ref backend) = self.gpu_backend {
            // Try GPU execution first
            match self.execute_on_gpu(operation.clone(), inputs, backend) {
                Ok(result) => result,
                Err(e) => {
                    if self.config.auto_fallback {
                        log::warn!("GPU execution failed, falling back to CPU: {e}");
                        self.execute_on_cpu(operation, inputs)?
                    } else {
                        return Err(e);
                    }
                }
            }
        } else {
            // CPU execution
            self.execute_on_cpu(operation, inputs)?
        };
        
        let stats = self.performance_monitor.end_operation(handle)?;
        
        // Check for performance degradation
        if let Some(degradation) = self.performance_monitor.detect_degradation() {
            log::warn!("Performance degradation detected: {}", degradation.suggested_action);
        }
        
        log::debug!("Operation {} completed in {:.2}ms", stats.name, stats.execution_time * 1000.0);
        
        Ok(result)
    }
    
    /// Execute operation on GPU
    fn execute_on_gpu<T>(
        &self,
        operation: NeuralOperation<T>,
        inputs: &[T],
        backend: &Arc<dyn GpuBackendTrait>,
    ) -> NeuralResult<Vec<T>>
    where
        T: Clone + Send + Sync + 'static + bytemuck::Pod,
    {
        // Implementation in gpu_neural_ops module
        gpu_neural_ops::execute_operation(operation, inputs, backend, &self.memory_manager)
    }
    
    /// Execute operation on CPU (fallback)
    fn execute_on_cpu<T>(
        &self,
        operation: NeuralOperation<T>,
        inputs: &[T],
    ) -> NeuralResult<Vec<T>>
    where
        T: Clone + Send + Sync + 'static + num_traits::Float,
    {
        // Implementation in bridge module
        bridge::execute_cpu_fallback(operation, inputs)
    }
    
    /// Get memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        self.memory_manager.get_memory_stats()
    }
    
    /// Get performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        self.performance_monitor.get_performance_summary()
    }
    
    /// Create a batch processor for efficient bulk operations
    pub fn create_batch_processor(&self) -> BatchProcessor {
        BatchProcessor::new(
            self.gpu_backend.clone(),
            self.memory_manager.clone(),
            self.config.batch_size,
        )
    }
}

/// Neural operation types
#[derive(Debug, Clone)]
pub enum NeuralOperation<T> {
    MatrixMultiply { a_rows: usize, a_cols: usize, b_cols: usize, _phantom: PhantomData<T> },
    VectorAdd { size: usize, _phantom: PhantomData<T> },
    ActivationFunction { function: ActivationFunction, size: usize, _phantom: PhantomData<T> },
    Convolution { channels: usize, kernel_size: usize, stride: usize, _phantom: PhantomData<T> },
    ForwardPropagation { layer_sizes: Vec<usize>, _phantom: PhantomData<T> },
    BackwardPropagation { layer_sizes: Vec<usize>, _phantom: PhantomData<T> },
    Custom { kernel_source: String, name: String, _phantom: PhantomData<T> },
}

impl<T> NeuralOperation<T> {
    pub fn name(&self) -> String {
        match self {
            Self::MatrixMultiply { .. } => "matrix_multiply".to_string(),
            Self::VectorAdd { .. } => "vector_add".to_string(),
            Self::ActivationFunction { function, .. } => format!("activation_{function:?}"),
            Self::Convolution { .. } => "convolution".to_string(),
            Self::ForwardPropagation { .. } => "forward_propagation".to_string(),
            Self::BackwardPropagation { .. } => "backward_propagation".to_string(),
            Self::Custom { name, .. } => name.clone(),
        }
    }
}

/// Activation function types
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
    LeakyReLU,
    Swish,
    GELU,
}

/// Batch processor for efficient bulk operations
pub struct BatchProcessor {
    gpu_backend: Option<Arc<dyn GpuBackendTrait>>,
    memory_manager: Arc<dyn MemoryManagerTrait>,
    batch_size: usize,
}

impl BatchProcessor {
    pub fn new(
        gpu_backend: Option<Arc<dyn GpuBackendTrait>>,
        memory_manager: Arc<dyn MemoryManagerTrait>,
        batch_size: usize,
    ) -> Self {
        Self {
            gpu_backend,
            memory_manager,
            batch_size,
        }
    }
    
    /// Process a batch of operations efficiently
    pub fn process_batch<T>(&self, operations: Vec<NeuralOperation<T>>, inputs: Vec<Vec<T>>) -> NeuralResult<Vec<Vec<T>>>
    where
        T: Clone + Send + Sync + 'static + bytemuck::Pod + num_traits::Float,
    {
        // Implementation in gpu_neural_ops module
        gpu_neural_ops::process_batch(operations, inputs, &self.gpu_backend, &self.memory_manager, self.batch_size)
    }
}

// Re-export public types
pub use bridge::{WebGpuBackend, extract_wgsl_from_rust, execute_cpu_fallback};
pub use cuda_kernels::*;
pub use gpu_neural_ops::{execute_operation, process_batch};
pub use memory_manager::{HybridMemoryManager};
pub use performance_monitor::{RealTimeMonitor, NoOpMonitor};
pub use wasm_bindings::*;

/// Initialize the neural integration system
pub fn initialize() -> NeuralResult<()> {
    // Initialize logging
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        wasm_logger::init(wasm_logger::Config::default());
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = env_logger::try_init();
    }
    
    log::info!("Neural integration system initialized");
    Ok(())
}

/// Get system capabilities
pub fn get_capabilities() -> SystemCapabilities {
    SystemCapabilities {
        cuda_transpilation: true,
        gpu_acceleration: cfg!(any(feature = "gpu", feature = "webgpu")),
        wasm_support: cfg!(target_arch = "wasm32"),
        performance_monitoring: true,
        memory_pooling: true,
        auto_fallback: true,
        batch_processing: true,
        precision_f16: true,
        precision_f32: true,
        precision_f64: cfg!(not(target_arch = "wasm32")),
    }
}

/// System capabilities
#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    pub cuda_transpilation: bool,
    pub gpu_acceleration: bool,
    pub wasm_support: bool,
    pub performance_monitoring: bool,
    pub memory_pooling: bool,
    pub auto_fallback: bool,
    pub batch_processing: bool,
    pub precision_f16: bool,
    pub precision_f32: bool,
    pub precision_f64: bool,
}

impl Default for NeuralBridge {
    fn default() -> Self {
        Self::new().expect("Failed to create default neural bridge")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bridge_creation() {
        let bridge = NeuralBridge::new();
        assert!(bridge.is_ok());
    }
    
    #[test]
    fn test_capabilities() {
        let capabilities = get_capabilities();
        assert!(capabilities.cuda_transpilation);
        assert!(capabilities.performance_monitoring);
    }
    
    #[test]
    fn test_config_default() {
        let config = BridgeConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.memory_pool_size, 512);
        assert_eq!(config.enable_gpu, crate::backend::gpu_enabled());
        assert!(config.auto_fallback);
    }
}