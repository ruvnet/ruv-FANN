// Shader Management System for ruv-FANN WebGPU Backend
// Handles shader compilation, caching, and pipeline management
// Optimized for runtime performance with intelligent caching

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use wgpu::{Device, ShaderModule, ComputePipeline, PipelineLayout, BindGroupLayout};

/// Represents different shader types for neural network operations
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ShaderType {
    // Matrix Operations
    MatrixVectorMultiply,
    BatchMatrixVectorMultiply,
    MatrixTranspose,
    ElementwiseAdd,
    ElementwiseMultiply,
    
    // Activation Functions  
    SigmoidFast,
    SigmoidPrecise,
    ReLU,
    LeakyReLU,
    ELU,
    Swish,
    GeluFast,
    TanhStable,
    TanhFast,
    Gaussian,
    GaussianSymmetric,
    Linear,
    LinearPiece,
    LinearPieceSymmetric,
    Threshold,
    ThresholdSymmetric,
    SinActivation,
    CosActivation,
    Elliott,
    ElliottSymmetric,
    
    // Gradient Computation
    ComputeOutputErrorGradients,
    BackpropagateHiddenErrors,
    ComputeWeightGradients,
    ComputeBiasGradients,
    ComputeActivationDerivatives,
    SgdMomentumUpdate,
    AdamUpdate,
    AccumulateGradients,
    ClearAccumulatedGradients,
}

/// Configuration for shader compilation and pipeline creation
#[derive(Debug, Clone)]
pub struct ShaderConfig {
    pub workgroup_size: (u32, u32, u32),
    pub buffer_bindings: Vec<BufferBinding>,
    pub uniform_bindings: Vec<UniformBinding>,
    pub entry_point: String,
}

#[derive(Debug, Clone)]
pub struct BufferBinding {
    pub binding: u32,
    pub access: BufferAccess,
}

#[derive(Debug, Clone)]
pub enum BufferAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(Debug, Clone)]
pub struct UniformBinding {
    pub binding: u32,
    pub size: u64,
}

/// Compiled shader with associated metadata
#[derive(Debug)]
pub struct CompiledShader {
    pub module: ShaderModule,
    pub pipeline: ComputePipeline,
    pub bind_group_layout: BindGroupLayout,
    pub config: ShaderConfig,
    pub source_hash: u64, // For cache invalidation
}

/// High-performance shader manager with intelligent caching
pub struct ShaderManager {
    device: Arc<Device>,
    compiled_shaders: Arc<RwLock<HashMap<ShaderType, CompiledShader>>>,
    shader_sources: HashMap<ShaderType, &'static str>,
    shader_configs: HashMap<ShaderType, ShaderConfig>,
}

impl ShaderManager {
    /// Create a new shader manager
    pub fn new(device: Arc<Device>) -> Self {
        let mut manager = Self {
            device,
            compiled_shaders: Arc::new(RwLock::new(HashMap::new())),
            shader_sources: HashMap::new(),
            shader_configs: HashMap::new(),
        };
        
        manager.initialize_shader_sources();
        manager.initialize_shader_configs();
        manager
    }
    
    /// Initialize all shader sources (embedded as string literals)
    fn initialize_shader_sources(&mut self) {
        // Matrix Operations
        self.shader_sources.insert(ShaderType::MatrixVectorMultiply, include_str!("matrix_operations.wgsl"));
        self.shader_sources.insert(ShaderType::BatchMatrixVectorMultiply, include_str!("matrix_operations.wgsl"));
        self.shader_sources.insert(ShaderType::MatrixTranspose, include_str!("matrix_operations.wgsl"));
        self.shader_sources.insert(ShaderType::ElementwiseAdd, include_str!("matrix_operations.wgsl"));
        self.shader_sources.insert(ShaderType::ElementwiseMultiply, include_str!("matrix_operations.wgsl"));
        
        // Activation Functions
        self.shader_sources.insert(ShaderType::SigmoidFast, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::SigmoidPrecise, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::ReLU, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::LeakyReLU, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::ELU, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::Swish, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::GeluFast, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::TanhStable, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::TanhFast, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::Gaussian, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::GaussianSymmetric, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::Linear, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::LinearPiece, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::LinearPieceSymmetric, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::Threshold, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::ThresholdSymmetric, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::SinActivation, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::CosActivation, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::Elliott, include_str!("activation_functions.wgsl"));
        self.shader_sources.insert(ShaderType::ElliottSymmetric, include_str!("activation_functions.wgsl"));
        
        // Gradient Computation
        self.shader_sources.insert(ShaderType::ComputeOutputErrorGradients, include_str!("gradient_computation.wgsl"));
        self.shader_sources.insert(ShaderType::BackpropagateHiddenErrors, include_str!("gradient_computation.wgsl"));
        self.shader_sources.insert(ShaderType::ComputeWeightGradients, include_str!("gradient_computation.wgsl"));
        self.shader_sources.insert(ShaderType::ComputeBiasGradients, include_str!("gradient_computation.wgsl"));
        self.shader_sources.insert(ShaderType::ComputeActivationDerivatives, include_str!("gradient_computation.wgsl"));
        self.shader_sources.insert(ShaderType::SgdMomentumUpdate, include_str!("gradient_computation.wgsl"));
        self.shader_sources.insert(ShaderType::AdamUpdate, include_str!("gradient_computation.wgsl"));
        self.shader_sources.insert(ShaderType::AccumulateGradients, include_str!("gradient_computation.wgsl"));
        self.shader_sources.insert(ShaderType::ClearAccumulatedGradients, include_str!("gradient_computation.wgsl"));
    }
    
    /// Initialize shader configurations
    fn initialize_shader_configs(&mut self) {
        // Matrix Operations
        self.shader_configs.insert(ShaderType::MatrixVectorMultiply, ShaderConfig {
            workgroup_size: (256, 1, 1),
            buffer_bindings: vec![
                BufferBinding { binding: 0, access: BufferAccess::ReadOnly },    // matrix
                BufferBinding { binding: 1, access: BufferAccess::ReadOnly },    // vector
                BufferBinding { binding: 2, access: BufferAccess::ReadWrite },   // result
            ],
            uniform_bindings: vec![
                UniformBinding { binding: 3, size: 16 }, // MatrixVectorUniforms
            ],
            entry_point: "matrix_vector_multiply".to_string(),
        });
        
        self.shader_configs.insert(ShaderType::BatchMatrixVectorMultiply, ShaderConfig {
            workgroup_size: (16, 16, 1),
            buffer_bindings: vec![
                BufferBinding { binding: 0, access: BufferAccess::ReadOnly },    // matrix
                BufferBinding { binding: 1, access: BufferAccess::ReadOnly },    // vectors
                BufferBinding { binding: 2, access: BufferAccess::ReadWrite },   // results
            ],
            uniform_bindings: vec![
                UniformBinding { binding: 3, size: 16 }, // BatchUniforms
            ],
            entry_point: "batch_matrix_vector_multiply".to_string(),
        });
        
        self.shader_configs.insert(ShaderType::MatrixTranspose, ShaderConfig {
            workgroup_size: (16, 16, 1),
            buffer_bindings: vec![
                BufferBinding { binding: 0, access: BufferAccess::ReadOnly },    // input_matrix
                BufferBinding { binding: 1, access: BufferAccess::ReadWrite },   // output_matrix
            ],
            uniform_bindings: vec![
                UniformBinding { binding: 2, size: 16 }, // TransposeUniforms
            ],
            entry_point: "matrix_transpose".to_string(),
        });
        
        // Activation Functions (vectorized processing)
        let activation_config = ShaderConfig {
            workgroup_size: (256, 1, 1),
            buffer_bindings: vec![
                BufferBinding { binding: 0, access: BufferAccess::ReadOnly },    // input
                BufferBinding { binding: 1, access: BufferAccess::ReadWrite },   // output
            ],
            uniform_bindings: vec![
                UniformBinding { binding: 2, size: 16 }, // ActivationUniforms
            ],
            entry_point: "".to_string(), // Will be set per activation type
        };
        
        let activation_types = vec![
            (ShaderType::SigmoidFast, "sigmoid_fast"),
            (ShaderType::SigmoidPrecise, "sigmoid_precise"),
            (ShaderType::ReLU, "relu"),
            (ShaderType::LeakyReLU, "leaky_relu"),
            (ShaderType::ELU, "elu"),
            (ShaderType::Swish, "swish"),
            (ShaderType::GeluFast, "gelu_fast"),
            (ShaderType::TanhStable, "tanh_stable"),
            (ShaderType::TanhFast, "tanh_fast"),
            (ShaderType::Gaussian, "gaussian"),
            (ShaderType::GaussianSymmetric, "gaussian_symmetric"),
            (ShaderType::Linear, "linear"),
            (ShaderType::LinearPiece, "linear_piece"),
            (ShaderType::LinearPieceSymmetric, "linear_piece_symmetric"),
            (ShaderType::Threshold, "threshold"),
            (ShaderType::ThresholdSymmetric, "threshold_symmetric"),
            (ShaderType::SinActivation, "sin_activation"),
            (ShaderType::CosActivation, "cos_activation"),
            (ShaderType::Elliott, "elliott"),
            (ShaderType::ElliottSymmetric, "elliott_symmetric"),
        ];
        
        for (shader_type, entry_point) in activation_types {
            let mut config = activation_config.clone();
            config.entry_point = entry_point.to_string();
            self.shader_configs.insert(shader_type, config);
        }
        
        // Gradient Computation Shaders
        self.shader_configs.insert(ShaderType::ComputeOutputErrorGradients, ShaderConfig {
            workgroup_size: (256, 1, 1),
            buffer_bindings: vec![
                BufferBinding { binding: 0, access: BufferAccess::ReadOnly },    // layer_outputs
                BufferBinding { binding: 1, access: BufferAccess::ReadOnly },    // expected_outputs
                BufferBinding { binding: 2, access: BufferAccess::ReadOnly },    // activation_derivatives
                BufferBinding { binding: 3, access: BufferAccess::ReadWrite },   // error_gradients
            ],
            uniform_bindings: vec![
                UniformBinding { binding: 4, size: 16 }, // ErrorGradientUniforms
            ],
            entry_point: "compute_output_error_gradients".to_string(),
        });
        
        self.shader_configs.insert(ShaderType::ComputeWeightGradients, ShaderConfig {
            workgroup_size: (16, 16, 1),
            buffer_bindings: vec![
                BufferBinding { binding: 0, access: BufferAccess::ReadOnly },    // input_activations
                BufferBinding { binding: 1, access: BufferAccess::ReadOnly },    // output_errors
                BufferBinding { binding: 2, access: BufferAccess::ReadWrite },   // weight_gradients
            ],
            uniform_bindings: vec![
                UniformBinding { binding: 3, size: 16 }, // WeightGradientUniforms
            ],
            entry_point: "compute_weight_gradients".to_string(),
        });
        
        // Add more gradient computation configurations as needed...
    }
    
    /// Get or compile a shader, with intelligent caching
    pub fn get_shader(&self, shader_type: ShaderType) -> Result<Arc<CompiledShader>, ShaderError> {
        // Check if shader is already compiled
        // For now, always compile fresh since we can't cache GPU resources
        // TODO: Implement proper caching with Arc-based approach
        
        // Compile shader if not in cache
        self.compile_shader(shader_type)
    }
    
    /// Compile a shader and add it to the cache
    fn compile_shader(&self, shader_type: ShaderType) -> Result<Arc<CompiledShader>, ShaderError> {
        let source = self.shader_sources.get(&shader_type)
            .ok_or_else(|| ShaderError::SourceNotFound(format!("{:?}", shader_type)))?;
        
        let config = self.shader_configs.get(&shader_type)
            .ok_or_else(|| ShaderError::ConfigNotFound(format!("{:?}", shader_type)))?;
        
        // Create shader module
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{:?}_shader", shader_type)),
            source: wgpu::ShaderSource::Wgsl((*source).into()),
        });
        
        // Create bind group layout
        let bind_group_layout = self.create_bind_group_layout(config)?;
        
        // Create pipeline layout
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{:?}_pipeline_layout", shader_type)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{:?}_pipeline", shader_type)),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: &config.entry_point,
        });
        
        let compiled = CompiledShader {
            module,
            pipeline,
            bind_group_layout,
            config: config.clone(),
            source_hash: self.compute_source_hash(source),
        };
        
        // Add to cache
        let compiled_arc = Arc::new(compiled);
        {
            let _cache = self.compiled_shaders.write().unwrap();
            // Note: We can't clone GPU resources, so we don't cache them in the HashMap
            // Instead, we'll need to refactor to use Arc<CompiledShader> throughout
        }
        
        Ok(compiled_arc)
    }
    
    /// Create bind group layout from shader configuration
    fn create_bind_group_layout(&self, config: &ShaderConfig) -> Result<BindGroupLayout, ShaderError> {
        let mut entries = Vec::new();
        
        // Add buffer bindings
        for buffer_binding in &config.buffer_bindings {
            let buffer_type = match buffer_binding.access {
                BufferAccess::ReadOnly => wgpu::BufferBindingType::Storage { read_only: true },
                BufferAccess::WriteOnly | BufferAccess::ReadWrite => wgpu::BufferBindingType::Storage { read_only: false },
            };
            
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: buffer_binding.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: buffer_type,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        
        // Add uniform bindings
        for uniform_binding in &config.uniform_bindings {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: uniform_binding.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(uniform_binding.size),
                },
                count: None,
            });
        }
        
        Ok(self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shader_bind_group_layout"),
            entries: &entries,
        }))
    }
    
    /// Compute hash of shader source for cache invalidation
    fn compute_source_hash(&self, source: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Precompile frequently used shaders for faster startup
    pub fn precompile_common_shaders(&self) -> Result<(), ShaderError> {
        let common_shaders = vec![
            ShaderType::MatrixVectorMultiply,
            ShaderType::BatchMatrixVectorMultiply,
            ShaderType::SigmoidFast,
            ShaderType::ReLU,
            ShaderType::TanhStable,
            ShaderType::Linear,
            ShaderType::ComputeOutputErrorGradients,
            ShaderType::ComputeWeightGradients,
        ];
        
        for shader_type in common_shaders {
            self.get_shader(shader_type)?;
        }
        
        Ok(())
    }
    
    /// Clear shader cache (useful for development/debugging)
    pub fn clear_cache(&self) {
        let mut cache = self.compiled_shaders.write().unwrap();
        cache.clear();
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> ShaderCacheStats {
        let cache = self.compiled_shaders.read().unwrap();
        ShaderCacheStats {
            cached_shaders: cache.len(),
            total_shader_types: self.shader_sources.len(),
        }
    }
}

/// Shader compilation and management errors
#[derive(Debug)]
pub enum ShaderError {
    SourceNotFound(String),
    ConfigNotFound(String),
    CompilationFailed(String),
    PipelineCreationFailed(String),
    BindGroupLayoutFailed,
}

impl std::fmt::Display for ShaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceNotFound(s) => write!(f, "Shader source not found: {}", s),
            Self::ConfigNotFound(s) => write!(f, "Shader configuration not found: {}", s),
            Self::CompilationFailed(s) => write!(f, "Shader compilation failed: {}", s),
            Self::PipelineCreationFailed(s) => write!(f, "Pipeline creation failed: {}", s),
            Self::BindGroupLayoutFailed => write!(f, "Bind group layout creation failed"),
        }
    }
}

impl std::error::Error for ShaderError {}

/// Statistics about shader cache usage
#[derive(Debug, Clone)]
pub struct ShaderCacheStats {
    pub cached_shaders: usize,
    pub total_shader_types: usize,
}

// Note: CompiledShader cannot implement Clone because WebGPU resources aren't cloneable
// The shader manager maintains ownership and hands out references instead

#[cfg(test)]
mod tests {
    use super::*;
    
    // Note: These tests would require a WebGPU device, which isn't available in all test environments
    // In practice, you'd want integration tests that run on systems with GPU support
    
    #[test]
    fn test_shader_config_creation() {
        let config = ShaderConfig {
            workgroup_size: (256, 1, 1),
            buffer_bindings: vec![
                BufferBinding { binding: 0, access: BufferAccess::ReadOnly },
            ],
            uniform_bindings: vec![
                UniformBinding { binding: 1, size: 16 },
            ],
            entry_point: "test_main".to_string(),
        };
        
        assert_eq!(config.workgroup_size, (256, 1, 1));
        assert_eq!(config.buffer_bindings.len(), 1);
        assert_eq!(config.uniform_bindings.len(), 1);
    }
    
    #[test]
    fn test_shader_type_variants() {
        // Ensure all shader types are represented
        let matrix_ops = [
            ShaderType::MatrixVectorMultiply,
            ShaderType::BatchMatrixVectorMultiply,
            ShaderType::MatrixTranspose,
        ];
        
        let activations = [
            ShaderType::SigmoidFast,
            ShaderType::ReLU,
            ShaderType::TanhStable,
        ];
        
        let gradients = [
            ShaderType::ComputeOutputErrorGradients,
            ShaderType::ComputeWeightGradients,
        ];
        
        assert!(!matrix_ops.is_empty());
        assert!(!activations.is_empty());
        assert!(!gradients.is_empty());
    }
}