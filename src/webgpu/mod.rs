//! WebGPU compute backend for ruv-FANN neural networks
//!
//! This module provides GPU acceleration for neural network operations using WebGPU.
//! It includes automatic fallback to SIMD/CPU backends when GPU is unavailable.
//!
//! Advanced features:
//! - Pipeline caching for optimized shader compilation
//! - Kernel optimization for different GPU architectures
//! - Performance monitoring and auto-tuning
//! - Comprehensive gradient operations for training
//! - Advanced 5-tier buffer pooling system with DAA integration
//! - Real-time memory pressure monitoring and autonomous optimization
//! - Circuit breaker protection and predictive analytics

pub mod backend;
pub mod compute_context;
pub mod error;
pub mod fallback;
pub mod memory;
pub mod shaders;

// Enhanced memory management components
pub mod buffer_pool;
pub mod pressure_monitor;

// Advanced shader system components
pub mod kernel_optimizer;
pub mod performance_monitor;
pub mod pipeline_cache;

#[cfg(any(feature = "gpu", feature = "webgpu"))]
pub mod webgpu_backend;

#[cfg(any(feature = "gpu", feature = "webgpu"))]
pub mod device;

// Autonomous GPU resource management system
#[cfg(all(any(feature = "gpu", feature = "webgpu"), not(target_arch = "wasm32")))]
pub mod autonomous_gpu_resource_manager;
// WASM GPU bridge for browser deployment
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub mod wasm_gpu_bridge;

// Re-export main types
pub use backend::{BackendSelector, ComputeProfile};
pub use compute_context::{ComputeContext, ComputePerformanceStats, DaaCoordinationMetrics};
pub use error::ComputeError;
pub use fallback::FallbackManager;
pub use memory::{BufferHandle, MemoryStats};

// Re-export enhanced memory management
pub use buffer_pool::{
    AdvancedBufferPool, BufferCategory, GpuBuffer as AdvancedGpuBuffer, MemoryPressure,
    PoolStatisticsSnapshot,
};
pub use pressure_monitor::{
    AnomalyEvent, MemoryPressureMonitor, MonitorConfig, MonitoringReport, PressurePrediction,
    PressureReading,
};
// Enhanced memory types are now in memory module
pub use memory::{
    EnhancedGpuMemoryManager, EnhancedMemoryStats, GpuMemoryConfig, GpuMemoryManager,
    GpuMemoryManagerEnhanced, OptimizationResult, WebGPUMemoryManager,
};

// Re-export advanced shader system components
pub use kernel_optimizer::{GpuCapabilities, KernelConfig, KernelOptimizer, OptimizationMetrics};
pub use performance_monitor::{
    PerformanceAlert, PerformanceMeasurement, PerformanceMonitor, PerformanceStats,
};
pub use pipeline_cache::{CacheStats, CompilationStats, PipelineCache};

// Re-export traits
pub use backend::{BackendCapabilities, BackendType, MatrixSize, OperationType};
pub use backend::{ComputeBackend, MemoryManager, VectorOps};

// Re-export WebGPU backend when available
#[cfg(any(feature = "gpu", feature = "webgpu"))]
pub use webgpu_backend::WebGPUBackend;

#[cfg(any(feature = "gpu", feature = "webgpu"))]
pub use shaders::*;

#[cfg(any(feature = "gpu", feature = "webgpu"))]
pub use device::GpuDevice;

// Re-export autonomous resource management
#[cfg(all(any(feature = "gpu", feature = "webgpu"), not(target_arch = "wasm32")))]
pub use autonomous_gpu_resource_manager::{
    AgentResourceAllocation,
    AllocationEngine,
    // Error types
    AllocationError,
    AllocationRequest,
    AllocationResult,
    AutonomousGpuResourceManager,
    ConflictError,
    ConflictResolution,
    ConflictResolver,
    OptimizationEngine,
    OptimizationError,
    OptimizationResult as ResourceOptimizationResult,
    PerformanceAnalyzer,
    PerformanceTier,
    PoolType,
    Priority,
    QualityRequirements,
    ResourceAllocation,
    ResourceCapacity,
    ResourceMarket,
    ResourcePolicies,
    ResourcePool,
    // Resource types
    ResourceRequirements,
    ResourceTrade,
    ResourceTradingSystem,
    ResourceType,
    RuvTokenLedger,
    TradeError,
    TradeProposal,
    TradeResult,
    UsagePredictor,
    UtilizationSummary,
};

// Re-export WASM GPU bridge for browser deployment
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub use wasm_gpu_bridge::{
    BrowserCompatibility, CrossOriginManager, CrossTabCoordinator, DaaWebRuntime,
    ServiceWorkerAgent, SharedBuffer, WasmGpuBridge, WasmMemoryManager, WasmPerformanceMonitor,
    WebGpuContext, WebMessageRouter, WebStorageManager, WebWorkerAgent,
};

/// Check if enhanced memory management features are available
pub fn has_enhanced_memory_features() -> bool {
    cfg!(feature = "gpu")
}

/// Get memory management capabilities summary
pub fn get_memory_capabilities() -> MemoryCapabilities {
    MemoryCapabilities {
        webgpu_available: cfg!(feature = "gpu"),
        enhanced_features: has_enhanced_memory_features(),
        daa_support: cfg!(feature = "gpu"),
        pressure_monitoring: cfg!(feature = "gpu"),
        circuit_breaker: cfg!(feature = "gpu"),
        buffer_pooling: true,
        predictive_analytics: cfg!(feature = "gpu"),
        wasm_gpu_bridge: has_wasm_gpu_bridge(),
    }
}

/// Check if WASM GPU bridge is available
#[cfg(target_arch = "wasm32")]
pub fn has_wasm_gpu_bridge() -> bool {
    cfg!(feature = "wasm-gpu")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn has_wasm_gpu_bridge() -> bool {
    false
}

/// Memory management capabilities
#[derive(Debug, Clone)]
pub struct MemoryCapabilities {
    pub webgpu_available: bool,
    pub enhanced_features: bool,
    pub daa_support: bool,
    pub pressure_monitoring: bool,
    pub circuit_breaker: bool,
    pub buffer_pooling: bool,
    pub predictive_analytics: bool,
    pub wasm_gpu_bridge: bool,
}

impl MemoryCapabilities {
    /// Get capabilities summary string
    pub fn summary(&self) -> String {
        let features = [
            ("WebGPU", self.webgpu_available),
            ("Enhanced Features", self.enhanced_features),
            ("DAA Support", self.daa_support),
            ("Pressure Monitoring", self.pressure_monitoring),
            ("Circuit Breaker", self.circuit_breaker),
            ("Buffer Pooling", self.buffer_pooling),
            ("Predictive Analytics", self.predictive_analytics),
            ("WASM GPU Bridge", self.wasm_gpu_bridge),
        ];

        let enabled: Vec<&str> = features
            .iter()
            .filter_map(|(name, enabled)| if *enabled { Some(*name) } else { None })
            .collect();

        format!("Memory Capabilities: {}", enabled.join(", "))
    }
}

// Tests
#[cfg(test)]
mod tests;
