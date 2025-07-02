// WebGPU Shader Module for ruv-FANN
// Provides comprehensive GPU compute shaders for neural network operations

pub mod shader_manager;
pub mod shader_pipeline;

pub use shader_manager::{ShaderManager, ShaderType, ShaderConfig, CompiledShader};
pub use shader_pipeline::{ShaderPipeline, ExecutionConfig, ExecutionResults};

// Re-export commonly used types
pub use shader_manager::ShaderError;
pub use shader_pipeline::ShaderPipelineError;

/// Initialize shader system with device
pub fn initialize_shaders(device: std::sync::Arc<wgpu::Device>) -> Result<ShaderManager, ShaderError> {
    let manager = ShaderManager::new(device);
    
    // Precompile commonly used shaders
    manager.precompile_common_shaders()?;
    
    Ok(manager)
}

/// Available shader files embedded at compile time
pub const MATRIX_OPERATIONS_SHADER: &str = include_str!("matrix_operations.wgsl");
pub const ACTIVATION_FUNCTIONS_SHADER: &str = include_str!("activation_functions.wgsl");
pub const GRADIENT_COMPUTATION_SHADER: &str = include_str!("gradient_computation.wgsl");
pub const ADVANCED_OPERATIONS_SHADER: &str = include_str!("advanced_operations.wgsl");