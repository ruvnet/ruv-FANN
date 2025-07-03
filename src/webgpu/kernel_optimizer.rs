//! GPU Kernel Optimization System
//! Advanced workgroup sizing, memory coalescing, and performance optimization

use crate::webgpu::error::ComputeError;
use std::collections::HashMap;

#[cfg(feature = "gpu")]
use crate::webgpu::shaders::webgpu_shaders::ShaderType;

#[cfg(not(feature = "gpu"))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShaderType {
    // Matrix operations - must match the actual enum variants used
    MatrixVectorMultiply,
    BatchMatrixVectorMultiply,

    // Basic fallback variants
    Neural,
    Compute,
    Training,
}

/// GPU device capabilities and optimization parameters
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// Maximum workgroup size in each dimension
    pub max_workgroup_size: [u32; 3],

    /// Maximum number of threads per workgroup
    pub max_threads_per_workgroup: u32,

    /// Preferred workgroup size for this GPU architecture
    pub preferred_workgroup_size: u32,

    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f32,

    /// Number of compute units/streaming multiprocessors
    pub compute_units: u32,

    /// Maximum shared memory per workgroup in bytes
    pub max_shared_memory_bytes: u32,

    /// Whether the GPU supports subgroups/wavefronts
    pub supports_subgroups: bool,

    /// Subgroup/wavefront size (typically 32 or 64)
    pub subgroup_size: u32,
}

/// Optimized kernel configuration for specific operations
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Optimal workgroup size for this kernel
    pub workgroup_size: [u32; 3],

    /// Number of workgroups to dispatch
    pub num_workgroups: [u32; 3],

    /// Whether to use vectorized memory access
    pub use_vectorized_access: bool,

    /// Tile size for tiled algorithms
    pub tile_size: u32,

    /// Number of elements processed per thread
    pub elements_per_thread: u32,

    /// Estimated performance in GFLOPS
    pub estimated_gflops: f32,
}

/// Performance optimization metrics
#[derive(Debug, Clone)]
pub struct OptimizationMetrics {
    /// Memory bandwidth utilization (0.0 to 1.0)
    pub memory_utilization: f32,

    /// Compute utilization (0.0 to 1.0)
    pub compute_utilization: f32,

    /// Occupancy (threads per SM / max threads per SM)
    pub occupancy: f32,

    /// Memory access efficiency
    pub memory_efficiency: f32,

    /// Estimated execution time in microseconds
    pub estimated_execution_time_us: f32,
}

/// Advanced kernel optimizer with GPU-specific optimizations
#[derive(Debug)]
pub struct KernelOptimizer {
    /// GPU capabilities for this device
    gpu_capabilities: GpuCapabilities,

    /// Cached optimal configurations for different operations
    config_cache: HashMap<(ShaderType, usize), KernelConfig>,

    /// Performance history for continuous optimization
    performance_history: HashMap<(ShaderType, usize), Vec<OptimizationMetrics>>,
}

impl KernelOptimizer {
    /// Create a new kernel optimizer with detected GPU capabilities
    pub fn new(gpu_capabilities: GpuCapabilities) -> Self {
        Self {
            gpu_capabilities,
            config_cache: HashMap::new(),
            performance_history: HashMap::new(),
        }
    }

    /// Create with default capabilities (fallback for unknown GPUs)
    pub fn with_default_capabilities() -> Self {
        let default_caps = GpuCapabilities {
            max_workgroup_size: [256, 256, 64],
            max_threads_per_workgroup: 1024,
            preferred_workgroup_size: 256,
            memory_bandwidth_gbps: 500.0, // Conservative estimate
            compute_units: 20,
            max_shared_memory_bytes: 32768,
            supports_subgroups: true,
            subgroup_size: 32,
        };

        Self::new(default_caps)
    }

    /// Optimize kernel configuration for matrix-vector multiplication
    pub fn optimize_matrix_vector_multiply(
        &mut self,
        rows: usize,
        cols: usize,
    ) -> Result<KernelConfig, ComputeError> {
        let cache_key = (ShaderType::MatrixVectorMultiply, rows * cols);

        // Check cache first
        if let Some(cached_config) = self.config_cache.get(&cache_key) {
            return Ok(cached_config.clone());
        }

        // Calculate optimal configuration
        let config = self.calculate_matrix_vector_config(rows, cols)?;

        // Cache the result
        self.config_cache.insert(cache_key, config.clone());

        Ok(config)
    }

    /// Optimize configuration for batch matrix-vector operations
    pub fn optimize_batch_matrix_vector(
        &mut self,
        rows: usize,
        cols: usize,
        batch_size: usize,
    ) -> Result<KernelConfig, ComputeError> {
        let cache_key = (
            ShaderType::BatchMatrixVectorMultiply,
            rows * cols * batch_size,
        );

        if let Some(cached_config) = self.config_cache.get(&cache_key) {
            return Ok(cached_config.clone());
        }

        let config = self.calculate_batch_matrix_vector_config(rows, cols, batch_size)?;
        self.config_cache.insert(cache_key, config.clone());

        Ok(config)
    }

    /// Optimize configuration for activation functions
    pub fn optimize_activation_function(
        &mut self,
        shader_type: ShaderType,
        vector_size: usize,
    ) -> Result<KernelConfig, ComputeError> {
        let cache_key = (shader_type.clone(), vector_size);

        if let Some(cached_config) = self.config_cache.get(&cache_key) {
            return Ok(cached_config.clone());
        }

        let config = self.calculate_activation_config(shader_type.clone(), vector_size)?;
        self.config_cache.insert(cache_key, config.clone());

        Ok(config)
    }

    /// Record performance metrics for continuous optimization
    pub fn record_performance(
        &mut self,
        shader_type: ShaderType,
        data_size: usize,
        metrics: OptimizationMetrics,
    ) {
        let key = (shader_type, data_size);
        let history = self.performance_history.entry(key.clone()).or_default();
        history.push(metrics);

        // Keep only recent history (last 10 measurements)
        if let Some(history) = self.performance_history.get_mut(&key) {
            if history.len() > 10 {
                history.remove(0);
            }
        }
    }

    /// Get performance predictions based on historical data
    pub fn predict_performance(
        &self,
        shader_type: &ShaderType,
        data_size: usize,
    ) -> Option<OptimizationMetrics> {
        let key = (shader_type.clone(), data_size);

        if let Some(history) = self.performance_history.get(&key) {
            if !history.is_empty() {
                // Return average of recent measurements
                let count = history.len() as f32;
                let avg_memory_util =
                    history.iter().map(|m| m.memory_utilization).sum::<f32>() / count;
                let avg_compute_util =
                    history.iter().map(|m| m.compute_utilization).sum::<f32>() / count;
                let avg_occupancy = history.iter().map(|m| m.occupancy).sum::<f32>() / count;
                let avg_memory_eff =
                    history.iter().map(|m| m.memory_efficiency).sum::<f32>() / count;
                let avg_exec_time = history
                    .iter()
                    .map(|m| m.estimated_execution_time_us)
                    .sum::<f32>()
                    / count;

                return Some(OptimizationMetrics {
                    memory_utilization: avg_memory_util,
                    compute_utilization: avg_compute_util,
                    occupancy: avg_occupancy,
                    memory_efficiency: avg_memory_eff,
                    estimated_execution_time_us: avg_exec_time,
                });
            }
        }

        None
    }

    /// Auto-tune workgroup size by testing different configurations
    pub fn auto_tune_workgroup_size(
        &mut self,
        shader_type: ShaderType,
        data_size: usize,
    ) -> Result<[u32; 3], ComputeError> {
        let test_sizes = vec![64, 128, 256, 512, 1024];
        let mut best_config = [256, 1, 1]; // Default
        let mut best_score = 0.0f32;

        for size in test_sizes {
            if size <= self.gpu_capabilities.max_threads_per_workgroup {
                let config = [size, 1, 1];
                let score = self.evaluate_workgroup_config(&shader_type, data_size, config);

                if score > best_score {
                    best_score = score;
                    best_config = config;
                }
            }
        }

        Ok(best_config)
    }

    /// Clear optimization caches (useful for testing or GPU changes)
    pub fn clear_caches(&mut self) {
        self.config_cache.clear();
        self.performance_history.clear();
    }

    /// Get GPU capabilities
    pub fn get_gpu_capabilities(&self) -> &GpuCapabilities {
        &self.gpu_capabilities
    }

    // Private optimization calculation methods

    fn calculate_matrix_vector_config(
        &self,
        rows: usize,
        _cols: usize,
    ) -> Result<KernelConfig, ComputeError> {
        // Optimize for memory bandwidth and occupancy
        let workgroup_size = if rows >= 1024 {
            [256, 1, 1] // Large problems - maximize parallelism
        } else if rows >= 256 {
            [128, 1, 1] // Medium problems - balance parallelism and overhead
        } else {
            [64, 1, 1] // Small problems - avoid underutilization
        };

        let num_workgroups = [
            ((rows as u32 + workgroup_size[0] - 1) / workgroup_size[0]),
            1,
            1,
        ];

        Ok(KernelConfig {
            workgroup_size,
            num_workgroups,
            use_vectorized_access: true,
            tile_size: 16,
            elements_per_thread: 4,
            estimated_gflops: self.estimate_matrix_vector_gflops(rows),
        })
    }

    fn calculate_batch_matrix_vector_config(
        &self,
        rows: usize,
        _cols: usize,
        batch_size: usize,
    ) -> Result<KernelConfig, ComputeError> {
        // 2D workgroup layout for batch processing
        let total_work = rows * batch_size;

        let workgroup_size = if total_work >= 4096 {
            [16, 16, 1] // 256 threads total, good for large batches
        } else if total_work >= 1024 {
            [16, 8, 1] // 128 threads total
        } else {
            [8, 8, 1] // 64 threads total for small batches
        };

        let num_workgroups = [
            ((rows as u32 + workgroup_size[0] - 1) / workgroup_size[0]),
            ((batch_size as u32 + workgroup_size[1] - 1) / workgroup_size[1]),
            1,
        ];

        Ok(KernelConfig {
            workgroup_size,
            num_workgroups,
            use_vectorized_access: true,
            tile_size: 16,
            elements_per_thread: 1,
            estimated_gflops: self.estimate_batch_matrix_vector_gflops(rows, batch_size),
        })
    }

    fn calculate_activation_config(
        &self,
        _shader_type: ShaderType,
        vector_size: usize,
    ) -> Result<KernelConfig, ComputeError> {
        // Activation functions are memory-bound, optimize for memory bandwidth
        let workgroup_size = if vector_size >= 2048 {
            [256, 1, 1] // Maximize memory bandwidth utilization
        } else if vector_size >= 512 {
            [128, 1, 1]
        } else {
            [64, 1, 1]
        };

        let num_workgroups = [
            ((vector_size as u32 + workgroup_size[0] - 1) / workgroup_size[0]),
            1,
            1,
        ];

        Ok(KernelConfig {
            workgroup_size,
            num_workgroups,
            use_vectorized_access: true,
            tile_size: 1, // Not applicable for element-wise operations
            elements_per_thread: 1,
            estimated_gflops: self.estimate_activation_gflops(vector_size),
        })
    }

    fn evaluate_workgroup_config(
        &self,
        _shader_type: &ShaderType,
        data_size: usize,
        workgroup_config: [u32; 3],
    ) -> f32 {
        let threads_per_workgroup = workgroup_config[0] * workgroup_config[1] * workgroup_config[2];
        let num_workgroups = (data_size as u32 + threads_per_workgroup - 1) / threads_per_workgroup;

        // Calculate occupancy
        let max_workgroups_per_sm =
            self.gpu_capabilities.max_threads_per_workgroup / threads_per_workgroup;
        let occupancy = (max_workgroups_per_sm.min(8) as f32) / 8.0; // Assume 8 max workgroups per SM

        // Simple scoring function (higher is better)
        let memory_efficiency = if threads_per_workgroup >= 32 {
            1.0
        } else {
            threads_per_workgroup as f32 / 32.0
        };
        let parallelism_score = (num_workgroups as f32)
            .min(self.gpu_capabilities.compute_units as f32)
            / self.gpu_capabilities.compute_units as f32;

        occupancy * 0.4 + memory_efficiency * 0.3 + parallelism_score * 0.3
    }

    fn estimate_matrix_vector_gflops(&self, rows: usize) -> f32 {
        // Rough estimate based on GPU capabilities
        let base_gflops = self.gpu_capabilities.compute_units as f32 * 100.0; // 100 GFLOPS per CU
        let problem_efficiency = if rows >= 1024 {
            0.8
        } else {
            rows as f32 / 1024.0 * 0.8
        };
        base_gflops * problem_efficiency
    }

    fn estimate_batch_matrix_vector_gflops(&self, rows: usize, batch_size: usize) -> f32 {
        let single_gflops = self.estimate_matrix_vector_gflops(rows);
        let batch_efficiency = (batch_size as f32).min(16.0) / 16.0; // Diminishing returns after 16
        single_gflops * batch_size as f32 * batch_efficiency
    }

    fn estimate_activation_gflops(&self, vector_size: usize) -> f32 {
        // Activation functions are simpler, memory-bound operations
        let memory_bound_gflops = self.gpu_capabilities.memory_bandwidth_gbps * 4.0; // 4 GFLOPS per GB/s
        let problem_efficiency = (vector_size as f32 / 1024.0).min(1.0);
        memory_bound_gflops * problem_efficiency
    }
}

impl Default for KernelOptimizer {
    fn default() -> Self {
        Self::with_default_capabilities()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_optimizer_creation() {
        let optimizer = KernelOptimizer::with_default_capabilities();
        assert_eq!(optimizer.gpu_capabilities.preferred_workgroup_size, 256);
    }

    #[test]
    fn test_matrix_vector_optimization() {
        let mut optimizer = KernelOptimizer::with_default_capabilities();
        let config = optimizer
            .optimize_matrix_vector_multiply(1024, 512)
            .unwrap();

        assert_eq!(config.workgroup_size[0], 256);
        assert!(config.use_vectorized_access);
        assert!(config.estimated_gflops > 0.0);
    }

    #[test]
    fn test_performance_recording() {
        let mut optimizer = KernelOptimizer::with_default_capabilities();
        let metrics = OptimizationMetrics {
            memory_utilization: 0.8,
            compute_utilization: 0.9,
            occupancy: 0.75,
            memory_efficiency: 0.85,
            estimated_execution_time_us: 100.0,
        };

        optimizer.record_performance(ShaderType::MatrixVectorMultiply, 1024, metrics);
        let prediction = optimizer.predict_performance(&ShaderType::MatrixVectorMultiply, 1024);

        assert!(prediction.is_some());
        assert_eq!(prediction.unwrap().memory_utilization, 0.8);
    }
}
