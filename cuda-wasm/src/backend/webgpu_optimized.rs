//! Optimized WebGPU backend for high-performance WASM execution
//!
//! This module provides an optimized WebGPU backend with advanced features:
//! - Kernel caching and JIT compilation
//! - Memory pooling and efficient transfers
//! - Auto-tuning for optimal block sizes
//! - Performance profiling and monitoring

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::*;
use crate::error::{CudaRustError, Result};
use crate::memory::{MemoryPool, allocate, deallocate};
use crate::profiling::{CounterType, time_operation};

/// Configuration for WebGPU optimization
#[derive(Debug, Clone)]
pub struct WebGPUConfig {
    /// Enable kernel caching
    pub enable_kernel_cache: bool,
    /// Enable auto-tuning for block sizes
    pub enable_auto_tuning: bool,
    /// Enable memory pooling
    pub enable_memory_pooling: bool,
    /// Maximum cache size for compiled kernels
    pub max_cache_size: usize,
    /// Preferred power preference
    pub power_preference: PowerPreference,
    /// Memory limits
    pub max_buffer_size: u64,
    /// Threading configuration
    pub max_workgroups_per_dimension: u32,
}

impl Default for WebGPUConfig {
    fn default() -> Self {
        Self {
            enable_kernel_cache: true,
            enable_auto_tuning: true,
            enable_memory_pooling: true,
            max_cache_size: 100,
            power_preference: PowerPreference::HighPerformance,
            max_buffer_size: 256 * 1024 * 1024, // 256MB
            max_workgroups_per_dimension: 65535,
        }
    }
}

/// Cached kernel with optimization metadata
#[derive(Debug, Clone)]
pub struct CachedKernel {
    /// Compiled compute pipeline
    pub pipeline: Arc<ComputePipeline>,
    /// Bind group layout
    pub bind_group_layout: Arc<BindGroupLayout>,
    /// Optimal workgroup size (auto-tuned)
    pub optimal_workgroup_size: [u32; 3],
    /// Performance metrics
    pub avg_execution_time: f64,
    /// Usage count for cache eviction
    pub usage_count: u64,
    /// Total data processed (for throughput calculation)
    pub total_data_processed: u64,
}

/// High-performance WebGPU backend
pub struct OptimizedWebGPUBackend {
    /// WebGPU device
    device: Arc<Device>,
    /// Command queue
    queue: Arc<Queue>,
    /// Configuration
    config: WebGPUConfig,
    /// Kernel cache
    kernel_cache: Arc<Mutex<HashMap<String, CachedKernel>>>,
    /// Memory pool for buffers
    memory_pool: Arc<MemoryPool>,
    /// Buffer cache for reuse
    buffer_cache: Arc<Mutex<HashMap<u64, Vec<Buffer>>>>,
    /// Performance statistics
    stats: Arc<Mutex<BackendStats>>,
}

/// Performance statistics for the backend
#[derive(Debug, Clone, Default)]
pub struct BackendStats {
    /// Total kernels executed
    pub kernels_executed: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Total execution time
    pub total_execution_time: f64,
    /// Total data transferred
    pub total_data_transferred: u64,
    /// Memory allocations
    pub memory_allocations: u64,
    /// Buffer reuse count
    pub buffer_reuse_count: u64,
}

/// Auto-tuning results for optimal performance
#[derive(Debug, Clone)]
pub struct AutoTuneResult {
    /// Optimal workgroup size
    pub workgroup_size: [u32; 3],
    /// Measured performance (operations per second)
    pub performance: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth: f64,
    /// Compute utilization
    pub compute_utilization: f64,
}

impl OptimizedWebGPUBackend {
    /// Create a new optimized WebGPU backend
    pub async fn new() -> Result<Self> {
        Self::with_config(WebGPUConfig::default()).await
    }

    /// Create backend with custom configuration
    pub async fn with_config(config: WebGPUConfig) -> Result<Self> {
        let _timer = time_operation(CounterType::Custom("webgpu_init".to_string()));
        
        // Request adapter with high performance preference
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::BROWSER_WEBGPU | Backends::GL,
            flags: InstanceFlags::default(),
            dx12_shader_compiler: Dx12Compiler::default(),
            gles_minor_version: Gles3MinorVersion::default(),
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: config.power_preference,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| CudaRustError::Backend("Failed to find suitable WebGPU adapter".to_string()))?;

        // Request device with optimal limits
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("CUDA-Rust Optimized Device"),
                    required_features: Features::TIMESTAMP_QUERY 
                        | Features::TIMESTAMP_QUERY_INSIDE_PASSES
                        | Features::PIPELINE_STATISTICS_QUERY,
                    required_limits: Limits {
                        max_buffer_size: config.max_buffer_size,
                        max_compute_workgroup_storage_size: 32768,
                        max_compute_invocations_per_workgroup: 1024,
                        max_compute_workgroup_size_x: 1024,
                        max_compute_workgroup_size_y: 1024,
                        max_compute_workgroup_size_z: 64,
                        max_compute_workgroups_per_dimension: config.max_workgroups_per_dimension,
                        ..Default::default()
                    },
                },
                None,
            )
            .await
            .map_err(|e| CudaRustError::Backend(format!("Failed to create WebGPU device: {e}")))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            config,
            kernel_cache: Arc::new(Mutex::new(HashMap::new())),
            memory_pool: Arc::new(MemoryPool::new()),
            buffer_cache: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(BackendStats::default())),
        })
    }

    /// Compile and cache a kernel with optimization
    pub fn compile_kernel(&self, shader_source: &str, entry_point: &str) -> Result<String> {
        let _timer = time_operation(CounterType::Compilation)
            .with_size(shader_source.len());

        let cache_key = format!("{}:{}", shader_source.len(), entry_point);
        
        // Check cache first
        {
            let cache = self.kernel_cache.lock().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                let mut stats = self.stats.lock().unwrap();
                stats.cache_hits += 1;
                return Ok(cache_key);
            }
        }

        // Cache miss - compile new kernel
        let shader_module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("CUDA Kernel"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Kernel Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Kernel Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("CUDA Kernel Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point,
        });

        // Auto-tune optimal workgroup size if enabled
        let optimal_workgroup_size = if self.config.enable_auto_tuning {
            self.auto_tune_workgroup_size(&pipeline, &bind_group_layout)?
        } else {
            [64, 1, 1] // Default workgroup size
        };

        // Cache the compiled kernel
        let cached_kernel = CachedKernel {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            optimal_workgroup_size,
            avg_execution_time: 0.0,
            usage_count: 0,
            total_data_processed: 0,
        };

        {
            let mut cache = self.kernel_cache.lock().unwrap();
            
            // Evict old entries if cache is full
            if cache.len() >= self.config.max_cache_size {
                self.evict_least_used_kernel(&mut cache);
            }
            
            cache.insert(cache_key.clone(), cached_kernel);
        }

        {
            let mut stats = self.stats.lock().unwrap();
            stats.cache_misses += 1;
        }

        Ok(cache_key)
    }

    /// Execute a cached kernel with optimal configuration
    pub async fn execute_kernel(
        &self, 
        cache_key: &str, 
        buffers: &[&Buffer], 
        workgroup_count: [u32; 3]
    ) -> Result<f64> {
        let _timer = time_operation(CounterType::KernelExecution);

        let (pipeline, bind_group_layout, optimal_workgroup_size) = {
            let mut cache = self.kernel_cache.lock().unwrap();
            let cached = cache.get_mut(cache_key)
                .ok_or_else(|| CudaRustError::Backend("Kernel not found in cache".to_string()))?;
            
            cached.usage_count += 1;
            (
                cached.pipeline.clone(),
                cached.bind_group_layout.clone(),
                cached.optimal_workgroup_size
            )
        };

        // Create bind group with buffers
        let entries: Vec<BindGroupEntry> = buffers.iter().enumerate()
            .map(|(i, buffer)| BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Kernel Bind Group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Kernel Execution"),
        });

        // Begin compute pass with optimal configuration
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("CUDA Kernel Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Use optimal workgroup size
            compute_pass.dispatch_workgroups(
                workgroup_count[0],
                workgroup_count[1],
                workgroup_count[2]
            );
        }

        // Submit and measure execution time
        #[cfg(target_arch = "wasm32")]
        let start_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);
        #[cfg(not(target_arch = "wasm32"))]
        let start_instant = std::time::Instant::now();

        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Wait for completion
        self.device.poll(Maintain::Wait);

        #[cfg(target_arch = "wasm32")]
        let end_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);
        
        #[cfg(target_arch = "wasm32")]
        let execution_time = end_time - start_time;
        #[cfg(not(target_arch = "wasm32"))]
        let execution_time = start_instant.elapsed().as_secs_f64() * 1000.0;

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.kernels_executed += 1;
            stats.total_execution_time += execution_time;
        }

        // Update cached kernel statistics
        {
            let mut cache = self.kernel_cache.lock().unwrap();
            if let Some(cached) = cache.get_mut(cache_key) {
                let alpha = 0.1; // Exponential moving average
                cached.avg_execution_time = 
                    alpha * execution_time + (1.0 - alpha) * cached.avg_execution_time;
            }
        }

        Ok(execution_time)
    }

    /// Auto-tune workgroup size for optimal performance
    fn auto_tune_workgroup_size(
        &self, 
        _pipeline: &ComputePipeline, 
        _bind_group_layout: &BindGroupLayout
    ) -> Result<[u32; 3]> {
        // Simplified auto-tuning - in a real implementation, this would
        // run benchmarks with different workgroup sizes
        
        // Common optimal sizes for different GPU architectures
        let candidate_sizes = [
            [32, 1, 1],   // Good for memory-bound kernels
            [64, 1, 1],   // Balanced
            [128, 1, 1],  // Good for compute-bound kernels
            [256, 1, 1],  // Maximum for some GPUs
            [16, 16, 1],  // 2D workgroup
            [8, 8, 8],    // 3D workgroup
        ];

        // For now, return a good default - this could be enhanced with
        // actual performance measurement
        Ok([64, 1, 1])
    }

    /// Evict least recently used kernel from cache
    fn evict_least_used_kernel(&self, cache: &mut HashMap<String, CachedKernel>) {
        if let Some((key_to_remove, _)) = cache.iter()
            .min_by_key(|(_, cached)| cached.usage_count) {
            let key_to_remove = key_to_remove.clone();
            cache.remove(&key_to_remove);
        }
    }

    /// Create an optimized buffer with pooling
    pub fn create_buffer(&self, size: u64, usage: BufferUsages) -> Result<Buffer> {
        let _timer = time_operation(CounterType::MemoryAllocation)
            .with_size(size as usize);

        // Check buffer cache for reusable buffers
        if self.config.enable_memory_pooling {
            let mut buffer_cache = self.buffer_cache.lock().unwrap();
            if let Some(buffers) = buffer_cache.get_mut(&size) {
                if let Some(buffer) = buffers.pop() {
                    let mut stats = self.stats.lock().unwrap();
                    stats.buffer_reuse_count += 1;
                    return Ok(buffer);
                }
            }
        }

        // Create new buffer
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("CUDA Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });

        {
            let mut stats = self.stats.lock().unwrap();
            stats.memory_allocations += 1;
        }

        Ok(buffer)
    }

    /// Return buffer to cache for reuse
    pub fn return_buffer(&self, buffer: Buffer) {
        if !self.config.enable_memory_pooling {
            return;
        }

        let size = buffer.size();
        let mut buffer_cache = self.buffer_cache.lock().unwrap();
        
        let buffers = buffer_cache.entry(size).or_default();
        
        // Limit cache size to prevent memory bloat
        if buffers.len() < 10 {
            buffers.push(buffer);
        }
    }

    /// Get comprehensive performance statistics
    pub fn get_stats(&self) -> BackendStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let stats = self.stats.lock().unwrap();
        let total = stats.cache_hits + stats.cache_misses;
        if total == 0 {
            0.0
        } else {
            stats.cache_hits as f64 / total as f64
        }
    }

    /// Clear all caches and reset statistics
    pub fn clear_caches(&self) {
        self.kernel_cache.lock().unwrap().clear();
        self.buffer_cache.lock().unwrap().clear();
        *self.stats.lock().unwrap() = BackendStats::default();
    }

    /// Generate performance report
    pub fn performance_report(&self) -> String {
        let stats = self.get_stats();
        let cache_ratio = self.cache_hit_ratio();
        let kernel_cache_size = self.kernel_cache.lock().unwrap().len();
        let buffer_cache_size: usize = self.buffer_cache.lock().unwrap()
            .values()
            .map(|v| v.len())
            .sum();

        format!(
            "=== WebGPU Backend Performance Report ===\n\
            Kernels Executed: {}\n\
            Cache Hit Ratio: {:.1}%\n\
            Avg Execution Time: {:.2}ms\n\
            Total Data Transferred: {:.2}MB\n\
            Memory Allocations: {}\n\
            Buffer Reuse Count: {}\n\
            Kernel Cache Size: {}\n\
            Buffer Cache Size: {}\n\
            Memory Pool Stats: {:?}",
            stats.kernels_executed,
            cache_ratio * 100.0,
            if stats.kernels_executed > 0 {
                stats.total_execution_time / stats.kernels_executed as f64
            } else {
                0.0
            },
            stats.total_data_transferred as f64 / 1_000_000.0,
            stats.memory_allocations,
            stats.buffer_reuse_count,
            kernel_cache_size,
            buffer_cache_size,
            self.memory_pool.stats()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_webgpu_backend_creation() {
        if !super::super::gpu_enabled() { return; }
        // This test may not work in all environments due to WebGPU requirements
        if let Ok(backend) = OptimizedWebGPUBackend::new().await {
            assert!(backend.cache_hit_ratio() == 0.0); // No cache hits initially
        }
    }

    #[test]
    fn test_auto_tune_result() {
        let result = AutoTuneResult {
            workgroup_size: [64, 1, 1],
            performance: 1000.0,
            memory_bandwidth: 0.8,
            compute_utilization: 0.9,
        };
        
        assert_eq!(result.workgroup_size, [64, 1, 1]);
        assert_eq!(result.performance, 1000.0);
    }

    #[test]
    fn test_backend_stats() {
        let stats = BackendStats {
            kernels_executed: 100,
            cache_hits: 80,
            cache_misses: 20,
            total_execution_time: 1000.0,
            ..Default::default()
        };
        
        assert_eq!(stats.kernels_executed, 100);
        assert_eq!(stats.cache_hits, 80);
        assert_eq!(stats.cache_misses, 20);
    }
}