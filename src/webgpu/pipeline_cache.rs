//! Legacy device-less pipeline lookup interface.
//!
//! This helper has no GPU device and cannot compile pipelines or create layouts.
//! Those requests return `UnsupportedOperation`; the real WebGPU backend owns
//! its device-bound pipeline cache separately. Request counts do not imply a
//! successful compilation or cache hit.

use crate::webgpu::error::ComputeError;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[cfg(feature = "gpu")]
use crate::webgpu::shaders::webgpu_shaders::ShaderType;

#[cfg(not(feature = "gpu"))]
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum ShaderType {
    MatrixVectorMultiply,
    BatchMatrixVectorMultiply,
    ActivationSigmoid,
    ActivationReLU,
    ActivationTanh,
    Other,
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct ComputePipeline {
    // This would be the actual wgpu::ComputePipeline when WebGPU is available
    _placeholder: std::marker::PhantomData<()>,
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct BindGroupLayout {
    // This would be the actual wgpu::BindGroupLayout when WebGPU is available
    _placeholder: std::marker::PhantomData<()>,
}

#[cfg(not(feature = "gpu"))]
#[derive(Debug)]
pub struct ComputePipeline {
    _placeholder: std::marker::PhantomData<()>,
}

#[cfg(not(feature = "gpu"))]
#[derive(Debug)]
pub struct BindGroupLayout {
    _placeholder: std::marker::PhantomData<()>,
}

/// Legacy lookup helper without a GPU device. Compilation and layout creation
/// explicitly fail, leaving compiled entries and hit statistics unchanged.
#[derive(Debug)]
pub struct PipelineCache {
    /// Compiled compute pipelines indexed by shader type
    pipelines: Arc<RwLock<HashMap<ShaderType, Arc<ComputePipeline>>>>,

    /// Cached bind group layouts to avoid recompilation
    bind_group_layouts: Arc<RwLock<HashMap<ShaderType, Arc<BindGroupLayout>>>>,

    /// Compilation statistics for performance monitoring
    compilation_stats: Arc<RwLock<CompilationStats>>,

    /// Cache hit/miss statistics
    cache_stats: Arc<RwLock<CacheStats>>,
}

#[derive(Debug, Default, Clone)]
pub struct CompilationStats {
    pub total_compilations: u64,
    pub compilation_time_ns: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_compilation_time_ns: u64,
}

#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub pipeline_requests: u64,
    pub pipeline_hits: u64,
    pub layout_requests: u64,
    pub layout_hits: u64,
    pub memory_usage_bytes: u64,
}

impl PipelineCache {
    /// Create a new pipeline cache with optimized settings
    pub fn new() -> Self {
        Self {
            pipelines: Arc::new(RwLock::new(HashMap::new())),
            bind_group_layouts: Arc::new(RwLock::new(HashMap::new())),
            compilation_stats: Arc::new(RwLock::new(CompilationStats::default())),
            cache_stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Request compilation. A device-less helper cannot produce a real pipeline.
    pub fn get_or_compile_pipeline(
        &self,
        shader_type: &ShaderType,
    ) -> Result<Arc<ComputePipeline>, ComputeError> {
        {
            let mut stats = self.cache_stats.write().unwrap();
            stats.pipeline_requests = stats.pipeline_requests.saturating_add(1);
        }
        self.compile_pipeline(shader_type).map(Arc::new)
    }

    /// Request a layout. Device-bound layout creation is unsupported here.
    pub fn get_or_create_bind_group_layout(
        &self,
        shader_type: &ShaderType,
    ) -> Result<Arc<BindGroupLayout>, ComputeError> {
        {
            let mut stats = self.cache_stats.write().unwrap();
            stats.layout_requests = stats.layout_requests.saturating_add(1);
        }
        self.create_bind_group_layout(shader_type).map(Arc::new)
    }

    /// Attempt warmup; returns the same unsupported error as compilation.
    pub fn warmup_cache(&self) -> Result<(), ComputeError> {
        let common_shaders = vec![
            ShaderType::MatrixVectorMultiply,
            ShaderType::BatchMatrixVectorMultiply,
            ShaderType::ActivationReLU,
            ShaderType::ActivationSigmoid,
            ShaderType::ActivationTanh,
        ];

        for shader_type in common_shaders {
            self.get_or_compile_pipeline(&shader_type)?;
            self.get_or_create_bind_group_layout(&shader_type)?;
        }

        Ok(())
    }

    /// Get comprehensive cache performance statistics
    pub fn get_performance_stats(&self) -> (CompilationStats, CacheStats) {
        let compilation_stats = {
            let stats = self.compilation_stats.read().unwrap();
            stats.clone()
        };
        let cache_stats = {
            let stats = self.cache_stats.read().unwrap();
            stats.clone()
        };
        (compilation_stats, cache_stats)
    }

    /// Clear cache and reset statistics (useful for testing)
    pub fn clear_cache(&self) {
        {
            let mut pipelines = self.pipelines.write().unwrap();
            pipelines.clear();
        }
        {
            let mut layouts = self.bind_group_layouts.write().unwrap();
            layouts.clear();
        }
        {
            let mut stats = self.compilation_stats.write().unwrap();
            *stats = CompilationStats::default();
        }
        {
            let mut stats = self.cache_stats.write().unwrap();
            *stats = CacheStats::default();
        }
    }

    /// Get cache hit ratio for performance monitoring
    pub fn get_cache_hit_ratio(&self) -> f64 {
        let stats = self.cache_stats.read().unwrap();
        if stats.pipeline_requests == 0 {
            return 0.0;
        }
        stats.pipeline_hits as f64 / stats.pipeline_requests as f64
    }

    fn compile_pipeline(&self, _shader_type: &ShaderType) -> Result<ComputePipeline, ComputeError> {
        Err(ComputeError::UnsupportedOperation(
            "Legacy PipelineCache cannot compile without a GPU device; use WebGPUBackend's device-bound cache".into(),
        ))
    }

    fn create_bind_group_layout(
        &self,
        _shader_type: &ShaderType,
    ) -> Result<BindGroupLayout, ComputeError> {
        Err(ComputeError::UnsupportedOperation(
            "Legacy PipelineCache cannot create a bind group layout without a GPU device".into(),
        ))
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_less_requests_fail_without_compilations_or_cached_entries() {
        let cache = PipelineCache::new();
        for _ in 0..3 {
            assert!(matches!(
                cache.get_or_compile_pipeline(&ShaderType::MatrixVectorMultiply),
                Err(ComputeError::UnsupportedOperation(_))
            ));
            assert!(matches!(
                cache.get_or_create_bind_group_layout(&ShaderType::MatrixVectorMultiply),
                Err(ComputeError::UnsupportedOperation(_))
            ));
        }
        let (compiled, requested) = cache.get_performance_stats();
        assert_eq!(requested.pipeline_requests, 3);
        assert_eq!(requested.layout_requests, 3);
        assert_eq!(compiled.total_compilations, 0);
        assert_eq!(compiled.compilation_time_ns, 0);
        assert_eq!(compiled.average_compilation_time_ns, 0);
        assert_eq!(compiled.cache_hits, 0);
        assert_eq!(requested.pipeline_hits, 0);
        assert_eq!(requested.layout_hits, 0);
        assert_eq!(requested.memory_usage_bytes, 0);
        assert_eq!(cache.get_cache_hit_ratio(), 0.0);
        assert!(cache.pipelines.read().unwrap().is_empty());
        assert!(cache.bind_group_layouts.read().unwrap().is_empty());
    }

    #[test]
    fn unsupported_warmup_never_populates_the_cache_and_clear_resets_requests() {
        let cache = PipelineCache::new();
        assert!(matches!(
            cache.warmup_cache(),
            Err(ComputeError::UnsupportedOperation(_))
        ));
        assert_eq!(cache.get_performance_stats().0.total_compilations, 0);
        assert_eq!(cache.get_performance_stats().1.pipeline_hits, 0);
        assert!(cache.pipelines.read().unwrap().is_empty());
        cache.clear_cache();
        assert_eq!(cache.get_performance_stats().1.pipeline_requests, 0);
        assert_eq!(cache.get_performance_stats().1.layout_requests, 0);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn shader_manager_propagates_unsupported_pipeline_and_layout_errors() {
        use crate::webgpu::shaders::webgpu_shaders::ShaderManager;
        let manager = ShaderManager::new().unwrap();
        let shader = ShaderType::MatrixVectorMultiply;
        assert!(manager.get_shader_source(&shader).is_some());
        for _ in 0..2 {
            assert!(matches!(
                manager.get_pipeline(&shader),
                Err(ComputeError::UnsupportedOperation(_))
            ));
            assert!(matches!(
                manager.get_bind_group_layout(&shader),
                Err(ComputeError::UnsupportedOperation(_))
            ));
        }
        assert!(matches!(
            manager.warmup_cache(),
            Err(ComputeError::UnsupportedOperation(_))
        ));
        let (compiled, requested) = manager.get_performance_stats();
        assert_eq!(compiled.total_compilations, 0);
        assert_eq!(requested.pipeline_hits, 0);
        assert_eq!(requested.layout_hits, 0);
        assert_eq!(manager.get_cache_hit_ratio(), 0.0);
    }

    #[test]
    fn test_pipeline_cache_creation() {
        let cache = PipelineCache::new();
        let (compilation_stats, cache_stats) = cache.get_performance_stats();

        assert_eq!(compilation_stats.total_compilations, 0);
        assert_eq!(cache_stats.pipeline_requests, 0);
    }

    #[test]
    fn test_cache_hit_ratio_calculation() {
        let cache = PipelineCache::new();
        assert_eq!(cache.get_cache_hit_ratio(), 0.0);
    }

    #[test]
    fn test_cache_clearing() {
        let cache = PipelineCache::new();
        cache.clear_cache();
        let (compilation_stats, cache_stats) = cache.get_performance_stats();

        assert_eq!(compilation_stats.total_compilations, 0);
        assert_eq!(cache_stats.pipeline_requests, 0);
    }
}
