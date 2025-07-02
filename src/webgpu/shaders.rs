//! WebGPU shader definitions and compilation

/// Embedded shader source code
pub mod embedded {
    /// Matrix-vector multiplication shader
    pub const MATRIX_VECTOR_MULTIPLY_SHADER: &str =
        include_str!("shaders/matrix_vector_multiply.wgsl");

    /// Batch matrix-vector multiplication shader
    pub const BATCH_MATRIX_VECTOR_MULTIPLY_SHADER: &str =
        include_str!("shaders/batch_matrix_vector_multiply.wgsl");

    /// Activation functions shader
    pub const ACTIVATION_FUNCTIONS_SHADER: &str = include_str!("shaders/activation_functions.wgsl");

    /// Gradient operations shader for backpropagation
    pub const GRADIENT_OPERATIONS_SHADER: &str = include_str!("shaders/gradient_operations.wgsl");

    /// Advanced neural network operations shader
    pub const ADVANCED_OPERATIONS_SHADER: &str = include_str!("shaders/advanced_operations.wgsl");
}

#[cfg(feature = "gpu")]
pub mod webgpu_shaders {
    use super::embedded;
    use crate::webgpu::error::ComputeError;

    #[derive(Debug, Hash, PartialEq, Eq, Clone)]
    pub enum ShaderType {
        // Matrix operations
        MatrixVectorMultiply,
        BatchMatrixVectorMultiply,

        // Activation functions
        ActivationSigmoid,
        ActivationReLU,
        ActivationLeakyReLU,
        ActivationTanh,
        ActivationLinear,
        ActivationGaussian,
        ActivationGaussianSymmetric,
        ActivationElliott,
        ActivationElliottSymmetric,
        ActivationSin,
        ActivationCos,
        ActivationSinSymmetric,
        ActivationCosSymmetric,
        ActivationLinearPiece,
        ActivationLinearPieceSymmetric,
        ActivationThreshold,
        ActivationThresholdSymmetric,
        ActivationGELU,
        ActivationSwish,

        // Gradient operations
        GradientSigmoid,
        GradientReLU,
        GradientLeakyReLU,
        GradientTanh,
        GradientLinear,
        WeightGradient,
        InputGradient,
        GradientClipping,
        L2Regularization,
        MomentumUpdate,
        AdamUpdate,
        BatchNormGradient,

        // Advanced operations
        Conv2D,
        MaxPool2D,
        AvgPool2D,
        Softmax,
        LayerNorm,
        ScaledDotProductAttention,
        ElementWiseAdd,
        ElementWiseMultiply,

        // Basic vector operations
        VectorAdd,
        VectorScale,
        DotProduct,
    }

    use crate::webgpu::kernel_optimizer::KernelOptimizer;
    use crate::webgpu::pipeline_cache::PipelineCache;
    use std::sync::Arc;

    #[derive(Debug)]
    pub struct ShaderManager {
        /// Pipeline cache for optimized shader compilation and reuse
        pipeline_cache: Arc<PipelineCache>,

        /// Kernel optimizer for workgroup sizing and performance tuning
        kernel_optimizer: Arc<std::sync::Mutex<KernelOptimizer>>,

        /// Performance monitoring enabled
        performance_monitoring: bool,
    }

    impl ShaderManager {
        /// Create a new shader manager with pipeline caching and optimization
        pub fn new() -> Result<Self, ComputeError> {
            let pipeline_cache = Arc::new(PipelineCache::new());
            let kernel_optimizer = Arc::new(std::sync::Mutex::new(
                KernelOptimizer::with_default_capabilities(),
            ));

            Ok(Self {
                pipeline_cache,
                kernel_optimizer,
                performance_monitoring: true,
            })
        }

        /// Create with custom GPU capabilities for optimization
        pub fn with_gpu_capabilities(
            gpu_caps: crate::webgpu::kernel_optimizer::GpuCapabilities,
        ) -> Result<Self, ComputeError> {
            let pipeline_cache = Arc::new(PipelineCache::new());
            let kernel_optimizer = Arc::new(std::sync::Mutex::new(KernelOptimizer::new(gpu_caps)));

            Ok(Self {
                pipeline_cache,
                kernel_optimizer,
                performance_monitoring: true,
            })
        }

        /// Get or compile a compute pipeline with automatic caching
        pub fn get_pipeline(&self, shader_type: &ShaderType) -> Result<Option<()>, ComputeError> {
            // In actual implementation, this would:
            // 1. Use pipeline_cache.get_or_compile_pipeline(shader_type)
            // 2. Return the compiled pipeline for GPU execution
            // For now, simulate successful pipeline retrieval
            match self.pipeline_cache.get_or_compile_pipeline(shader_type) {
                Ok(_) => Ok(Some(())),
                Err(e) => Err(e),
            }
        }

        /// Get bind group layout with caching
        pub fn get_bind_group_layout(
            &self,
            shader_type: &ShaderType,
        ) -> Result<Option<()>, ComputeError> {
            match self
                .pipeline_cache
                .get_or_create_bind_group_layout(shader_type)
            {
                Ok(_) => Ok(Some(())),
                Err(e) => Err(e),
            }
        }

        /// Get optimized kernel configuration for operation
        pub fn get_optimized_config(
            &self,
            shader_type: &ShaderType,
            data_size: usize,
        ) -> Result<crate::webgpu::kernel_optimizer::KernelConfig, ComputeError> {
            let mut optimizer = self.kernel_optimizer.lock().unwrap();

            match shader_type {
                ShaderType::MatrixVectorMultiply => {
                    // Assume square matrix for simplicity
                    let size = (data_size as f64).sqrt() as usize;
                    optimizer.optimize_matrix_vector_multiply(size, size)
                }
                ShaderType::BatchMatrixVectorMultiply => {
                    // Estimate dimensions
                    let batch_size = (data_size / 1024).max(1);
                    let matrix_size = (1024_f64).sqrt() as usize;
                    optimizer.optimize_batch_matrix_vector(matrix_size, matrix_size, batch_size)
                }
                _ => {
                    // For activation functions and other element-wise operations
                    optimizer.optimize_activation_function(shader_type.clone(), data_size)
                }
            }
        }

        /// Get shader type for activation function
        pub fn activation_shader_for_function(
            &self,
            function: crate::ActivationFunction,
        ) -> Option<ShaderType> {
            use crate::ActivationFunction;

            match function {
                ActivationFunction::Sigmoid => Some(ShaderType::ActivationSigmoid),
                ActivationFunction::SigmoidSymmetric => Some(ShaderType::ActivationSigmoid), // Use same shader
                ActivationFunction::ReLU => Some(ShaderType::ActivationReLU),
                ActivationFunction::ReLULeaky => Some(ShaderType::ActivationLeakyReLU),
                ActivationFunction::Tanh => Some(ShaderType::ActivationTanh),
                ActivationFunction::Linear => Some(ShaderType::ActivationLinear),
                ActivationFunction::Gaussian => Some(ShaderType::ActivationGaussian),
                ActivationFunction::GaussianSymmetric => {
                    Some(ShaderType::ActivationGaussianSymmetric)
                }
                ActivationFunction::Elliot => Some(ShaderType::ActivationElliott),
                ActivationFunction::ElliotSymmetric => Some(ShaderType::ActivationElliottSymmetric),
                ActivationFunction::Sin => Some(ShaderType::ActivationSin),
                ActivationFunction::Cos => Some(ShaderType::ActivationCos),
                ActivationFunction::SinSymmetric => Some(ShaderType::ActivationSinSymmetric),
                ActivationFunction::CosSymmetric => Some(ShaderType::ActivationCosSymmetric),
                ActivationFunction::LinearPiece => Some(ShaderType::ActivationLinearPiece),
                ActivationFunction::LinearPieceSymmetric => {
                    Some(ShaderType::ActivationLinearPieceSymmetric)
                }
                ActivationFunction::Threshold => Some(ShaderType::ActivationThreshold),
                ActivationFunction::ThresholdSymmetric => {
                    Some(ShaderType::ActivationThresholdSymmetric)
                }
            }
        }

        /// Get shader source code for a given shader type
        pub fn get_shader_source(&self, shader_type: &ShaderType) -> Option<&'static str> {
            match shader_type {
                // Matrix operations
                ShaderType::MatrixVectorMultiply => Some(embedded::MATRIX_VECTOR_MULTIPLY_SHADER),
                ShaderType::BatchMatrixVectorMultiply => {
                    Some(embedded::BATCH_MATRIX_VECTOR_MULTIPLY_SHADER)
                }

                // Activation functions
                ShaderType::ActivationSigmoid
                | ShaderType::ActivationReLU
                | ShaderType::ActivationLeakyReLU
                | ShaderType::ActivationTanh
                | ShaderType::ActivationLinear
                | ShaderType::ActivationGaussian
                | ShaderType::ActivationGaussianSymmetric
                | ShaderType::ActivationElliott
                | ShaderType::ActivationElliottSymmetric
                | ShaderType::ActivationSin
                | ShaderType::ActivationCos
                | ShaderType::ActivationSinSymmetric
                | ShaderType::ActivationCosSymmetric
                | ShaderType::ActivationLinearPiece
                | ShaderType::ActivationLinearPieceSymmetric
                | ShaderType::ActivationThreshold
                | ShaderType::ActivationThresholdSymmetric => {
                    Some(embedded::ACTIVATION_FUNCTIONS_SHADER)
                }

                // Gradient operations
                ShaderType::GradientSigmoid
                | ShaderType::GradientReLU
                | ShaderType::GradientLeakyReLU
                | ShaderType::GradientTanh
                | ShaderType::GradientLinear
                | ShaderType::WeightGradient
                | ShaderType::InputGradient
                | ShaderType::GradientClipping
                | ShaderType::L2Regularization
                | ShaderType::MomentumUpdate
                | ShaderType::AdamUpdate
                | ShaderType::BatchNormGradient => Some(embedded::GRADIENT_OPERATIONS_SHADER),

                // Advanced operations
                ShaderType::Conv2D
                | ShaderType::MaxPool2D
                | ShaderType::AvgPool2D
                | ShaderType::Softmax
                | ShaderType::LayerNorm
                | ShaderType::ScaledDotProductAttention
                | ShaderType::ElementWiseAdd
                | ShaderType::ElementWiseMultiply
                | ShaderType::ActivationGELU
                | ShaderType::ActivationSwish => Some(embedded::ADVANCED_OPERATIONS_SHADER),

                // Basic vector operations (would need separate shaders)
                _ => None,
            }
        }

        /// Precompile commonly used shaders for optimal performance
        pub fn warmup_cache(&self) -> Result<(), ComputeError> {
            self.pipeline_cache.warmup_cache()
        }

        /// Get performance statistics from caches
        pub fn get_performance_stats(
            &self,
        ) -> (
            crate::webgpu::pipeline_cache::CompilationStats,
            crate::webgpu::pipeline_cache::CacheStats,
        ) {
            self.pipeline_cache.get_performance_stats()
        }

        /// Record performance metrics for optimization
        pub fn record_performance(
            &self,
            shader_type: ShaderType,
            data_size: usize,
            metrics: crate::webgpu::kernel_optimizer::OptimizationMetrics,
        ) {
            let mut optimizer = self.kernel_optimizer.lock().unwrap();
            optimizer.record_performance(shader_type, data_size, metrics);
        }

        /// Get performance predictions
        pub fn predict_performance(
            &self,
            shader_type: &ShaderType,
            data_size: usize,
        ) -> Option<crate::webgpu::kernel_optimizer::OptimizationMetrics> {
            let optimizer = self.kernel_optimizer.lock().unwrap();
            optimizer.predict_performance(shader_type, data_size)
        }

        /// Clear all caches (useful for testing)
        pub fn clear_caches(&self) {
            self.pipeline_cache.clear_cache();
            let mut optimizer = self.kernel_optimizer.lock().unwrap();
            optimizer.clear_caches();
        }

        /// Get cache hit ratio for monitoring
        pub fn get_cache_hit_ratio(&self) -> f64 {
            self.pipeline_cache.get_cache_hit_ratio()
        }
    }
}
