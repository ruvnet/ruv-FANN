//! Compute backend abstraction for multiple acceleration types

use std::sync::Arc;
use num_traits::Float;
use crate::webgpu::error::{ComputeError, ComputeResult};
use crate::webgpu::device::GpuDevice;
use crate::webgpu::memory::GpuMemoryManager;

/// Matrix dimensions for operations
#[derive(Debug, Clone, Copy)]
pub struct MatrixDims {
    pub rows: usize,
    pub cols: usize,
}

impl MatrixDims {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    pub fn size(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_compatible_for_multiply(&self, other: &MatrixDims) -> bool {
        self.cols == other.rows
    }
}

/// Device capabilities and performance characteristics
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub compute_units: u32,
    pub memory_bandwidth_gbps: f32,
    pub peak_compute_throughput: f32,
    pub supports_f16: bool,
    pub supports_f64: bool,
    pub max_workgroup_size: (u32, u32, u32),
    pub estimated_speedup: f32, // Relative to CPU baseline
}

/// Neural network activation functions
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    Linear,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU(f32), // alpha parameter
    ELU(f32),       // alpha parameter
    GELU,
    Swish,
}

/// Backend type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    CPU,
    SIMD,
    WebGPU,
    CUDA,   // Future
    OpenCL, // Future
    Metal,  // Future
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CPU => write!(f, "CPU"),
            Self::SIMD => write!(f, "SIMD"),
            Self::WebGPU => write!(f, "WebGPU"),
            Self::CUDA => write!(f, "CUDA"),
            Self::OpenCL => write!(f, "OpenCL"),
            Self::Metal => write!(f, "Metal"),
        }
    }
}

/// Core compute backend trait for neural network operations
pub trait ComputeBackend<T: Float + Send + Sync>: Send + Sync {
    /// Get backend type
    fn backend_type(&self) -> BackendType;

    /// Get device capabilities
    fn device_info(&self) -> DeviceCapabilities;

    /// Check if this backend is available on current system
    fn is_available(&self) -> bool;

    /// Initialize the backend (async for GPU backends)
    fn initialize(&mut self) -> impl std::future::Future<Output = ComputeResult<()>> + Send;

    /// Matrix-vector multiplication: y = A * x
    fn matrix_vector_multiply(
        &self,
        matrix: &[T],
        vector: &[T],
        matrix_dims: MatrixDims,
    ) -> impl std::future::Future<Output = ComputeResult<Vec<T>>> + Send;

    /// Batch matrix-vector multiplication for multiple inputs
    fn batch_matrix_vector_multiply(
        &self,
        matrix: &[T],
        vectors: &[&[T]],
        matrix_dims: MatrixDims,
    ) -> impl std::future::Future<Output = ComputeResult<Vec<Vec<T>>>> + Send;

    /// Apply activation function to a vector
    fn activation_function(
        &self,
        input: &[T],
        function: ActivationFunction,
    ) -> impl std::future::Future<Output = ComputeResult<Vec<T>>> + Send;

    /// Compute gradients for backpropagation
    fn compute_gradients(
        &self,
        weights: &[T],
        activations: &[T],
        errors: &[T],
        weight_dims: MatrixDims,
    ) -> impl std::future::Future<Output = ComputeResult<Vec<T>>> + Send;

    /// Update weights with gradients
    fn update_weights(
        &self,
        weights: &mut [T],
        gradients: &[T],
        learning_rate: T,
    ) -> impl std::future::Future<Output = ComputeResult<()>> + Send;

    /// Element-wise vector addition
    fn vector_add(
        &self,
        a: &[T],
        b: &[T],
    ) -> impl std::future::Future<Output = ComputeResult<Vec<T>>> + Send;

    /// Scalar-vector multiplication
    fn vector_scale(
        &self,
        vector: &[T],
        scalar: T,
    ) -> impl std::future::Future<Output = ComputeResult<Vec<T>>> + Send;

    /// Estimate performance for a given workload
    fn estimate_performance(&self, problem_size: usize) -> f32;

    /// Get memory requirements for a workload
    fn memory_requirements(&self, problem_size: usize) -> u64;
}

/// CPU backend using existing SIMD optimizations
#[derive(Debug, Clone)]
pub struct CpuBackend {
    capabilities: DeviceCapabilities,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    pub fn new() -> Self {
        let num_cores = num_cpus::get() as u32;
        
        Self {
            capabilities: DeviceCapabilities {
                compute_units: num_cores,
                memory_bandwidth_gbps: 25.0, // Typical DDR4
                peak_compute_throughput: num_cores as f32 * 2.0, // Rough estimate
                supports_f16: false,
                supports_f64: true,
                max_workgroup_size: (1, 1, 1), // No workgroups for CPU
                estimated_speedup: 1.0, // Baseline
            },
        }
    }
}

impl<T: Float + Send + Sync> ComputeBackend<T> for CpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::CPU
    }

    fn device_info(&self) -> DeviceCapabilities {
        self.capabilities.clone()
    }

    fn is_available(&self) -> bool {
        true // CPU always available
    }

    async fn initialize(&mut self) -> ComputeResult<()> {
        Ok(()) // No initialization needed for CPU
    }

    async fn matrix_vector_multiply(
        &self,
        matrix: &[T],
        vector: &[T],
        matrix_dims: MatrixDims,
    ) -> ComputeResult<Vec<T>> {
        if matrix.len() != matrix_dims.size() {
            return Err(ComputeError::invalid_dimensions(
                format!("{}x{}", matrix_dims.rows, matrix_dims.cols),
                format!("{}", matrix.len()),
            ));
        }
        
        if vector.len() != matrix_dims.cols {
            return Err(ComputeError::invalid_dimensions(
                format!("{}", matrix_dims.cols),
                format!("{}", vector.len()),
            ));
        }

        let mut result = vec![T::zero(); matrix_dims.rows];
        
        for i in 0..matrix_dims.rows {
            let mut sum = T::zero();
            for j in 0..matrix_dims.cols {
                let matrix_val = matrix[i * matrix_dims.cols + j];
                let vector_val = vector[j];
                sum = sum + matrix_val * vector_val;
            }
            result[i] = sum;
        }

        Ok(result)
    }

    async fn batch_matrix_vector_multiply(
        &self,
        matrix: &[T],
        vectors: &[&[T]],
        matrix_dims: MatrixDims,
    ) -> ComputeResult<Vec<Vec<T>>> {
        let mut results = Vec::with_capacity(vectors.len());
        
        for vector in vectors {
            let result = self.matrix_vector_multiply(matrix, vector, matrix_dims).await?;
            results.push(result);
        }
        
        Ok(results)
    }

    async fn activation_function(
        &self,
        input: &[T],
        function: ActivationFunction,
    ) -> ComputeResult<Vec<T>> {
        let mut result = Vec::with_capacity(input.len());
        
        for &val in input {
            let activated = match function {
                ActivationFunction::Linear => val,
                ActivationFunction::Sigmoid => {
                    let exp_neg = (-val).exp();
                    T::one() / (T::one() + exp_neg)
                }
                ActivationFunction::Tanh => val.tanh(),
                ActivationFunction::ReLU => {
                    if val > T::zero() { val } else { T::zero() }
                }
                ActivationFunction::LeakyReLU(alpha) => {
                    if val > T::zero() { 
                        val 
                    } else { 
                        val * T::from(alpha).unwrap_or(T::zero())
                    }
                }
                ActivationFunction::ELU(alpha) => {
                    if val > T::zero() {
                        val
                    } else {
                        T::from(alpha).unwrap_or(T::one()) * (val.exp() - T::one())
                    }
                }
                ActivationFunction::GELU => {
                    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                    let sqrt_2_pi = T::from(0.7978845608).unwrap_or(T::one());
                    let coeff = T::from(0.044715).unwrap_or(T::zero());
                    let x3 = val * val * val;
                    let inner = sqrt_2_pi * (val + coeff * x3);
                    T::from(0.5).unwrap_or(T::one()) * val * (T::one() + inner.tanh())
                }
                ActivationFunction::Swish => {
                    // Swish: x * sigmoid(x)
                    let sigmoid = T::one() / (T::one() + (-val).exp());
                    val * sigmoid
                }
            };
            result.push(activated);
        }
        
        Ok(result)
    }

    async fn compute_gradients(
        &self,
        _weights: &[T],
        activations: &[T],
        errors: &[T],
        weight_dims: MatrixDims,
    ) -> ComputeResult<Vec<T>> {
        if activations.len() != weight_dims.cols || errors.len() != weight_dims.rows {
            return Err(ComputeError::invalid_dimensions(
                format!("weights: {}x{}, activations: {}, errors: {}", 
                    weight_dims.rows, weight_dims.cols, activations.len(), errors.len()),
                "mismatched dimensions".to_string(),
            ));
        }

        let mut gradients = vec![T::zero(); weight_dims.size()];
        
        for i in 0..weight_dims.rows {
            for j in 0..weight_dims.cols {
                let idx = i * weight_dims.cols + j;
                gradients[idx] = errors[i] * activations[j];
            }
        }
        
        Ok(gradients)
    }

    async fn update_weights(
        &self,
        weights: &mut [T],
        gradients: &[T],
        learning_rate: T,
    ) -> ComputeResult<()> {
        if weights.len() != gradients.len() {
            return Err(ComputeError::invalid_dimensions(
                format!("{}", weights.len()),
                format!("{}", gradients.len()),
            ));
        }
        
        for (weight, gradient) in weights.iter_mut().zip(gradients.iter()) {
            *weight = *weight - learning_rate * *gradient;
        }
        
        Ok(())
    }

    async fn vector_add(&self, a: &[T], b: &[T]) -> ComputeResult<Vec<T>> {
        if a.len() != b.len() {
            return Err(ComputeError::invalid_dimensions(
                format!("{}", a.len()),
                format!("{}", b.len()),
            ));
        }
        
        let result = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
        Ok(result)
    }

    async fn vector_scale(&self, vector: &[T], scalar: T) -> ComputeResult<Vec<T>> {
        let result = vector.iter().map(|&x| x * scalar).collect();
        Ok(result)
    }

    fn estimate_performance(&self, problem_size: usize) -> f32 {
        // CPU performance is roughly linear with problem size
        // Base performance of 1.0 for baseline
        let base_ops_per_sec = 1e9; // 1 billion ops/sec baseline
        base_ops_per_sec / (problem_size as f32)
    }

    fn memory_requirements(&self, problem_size: usize) -> u64 {
        // Rough estimate: input + output + temporary buffers
        (problem_size * 3 * std::mem::size_of::<T>()) as u64
    }
}

/// SIMD backend using vectorized operations
#[derive(Debug, Clone)]
pub struct SimdBackend {
    capabilities: DeviceCapabilities,
}

impl Default for SimdBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdBackend {
    pub fn new() -> Self {
        let num_cores = num_cpus::get() as u32;
        
        Self {
            capabilities: DeviceCapabilities {
                compute_units: num_cores,
                memory_bandwidth_gbps: 25.0,
                peak_compute_throughput: num_cores as f32 * 4.0, // SIMD boost
                supports_f16: false,
                supports_f64: true,
                max_workgroup_size: (4, 1, 1), // SIMD width
                estimated_speedup: 3.0, // 3x speedup over CPU
            },
        }
    }
}

impl<T: Float + Send + Sync> ComputeBackend<T> for SimdBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::SIMD
    }

    fn device_info(&self) -> DeviceCapabilities {
        self.capabilities.clone()
    }

    fn is_available(&self) -> bool {
        // For now, assume SIMD is available - could check CPU features
        true
    }

    async fn initialize(&mut self) -> ComputeResult<()> {
        Ok(())
    }

    // For now, SIMD backend delegates to CPU backend
    // In a full implementation, this would use vectorized operations
    async fn matrix_vector_multiply(
        &self,
        matrix: &[T],
        vector: &[T],
        matrix_dims: MatrixDims,
    ) -> ComputeResult<Vec<T>> {
        let cpu_backend = CpuBackend::new();
        cpu_backend.matrix_vector_multiply(matrix, vector, matrix_dims).await
    }

    async fn batch_matrix_vector_multiply(
        &self,
        matrix: &[T],
        vectors: &[&[T]],
        matrix_dims: MatrixDims,
    ) -> ComputeResult<Vec<Vec<T>>> {
        let cpu_backend = CpuBackend::new();
        cpu_backend.batch_matrix_vector_multiply(matrix, vectors, matrix_dims).await
    }

    async fn activation_function(
        &self,
        input: &[T],
        function: ActivationFunction,
    ) -> ComputeResult<Vec<T>> {
        let cpu_backend = CpuBackend::new();
        cpu_backend.activation_function(input, function).await
    }

    async fn compute_gradients(
        &self,
        weights: &[T],
        activations: &[T],
        errors: &[T],
        weight_dims: MatrixDims,
    ) -> ComputeResult<Vec<T>> {
        let cpu_backend = CpuBackend::new();
        cpu_backend.compute_gradients(weights, activations, errors, weight_dims).await
    }

    async fn update_weights(
        &self,
        weights: &mut [T],
        gradients: &[T],
        learning_rate: T,
    ) -> ComputeResult<()> {
        let cpu_backend = CpuBackend::new();
        cpu_backend.update_weights(weights, gradients, learning_rate).await
    }

    async fn vector_add(&self, a: &[T], b: &[T]) -> ComputeResult<Vec<T>> {
        let cpu_backend = CpuBackend::new();
        cpu_backend.vector_add(a, b).await
    }

    async fn vector_scale(&self, vector: &[T], scalar: T) -> ComputeResult<Vec<T>> {
        let cpu_backend = CpuBackend::new();
        cpu_backend.vector_scale(vector, scalar).await
    }

    fn estimate_performance(&self, problem_size: usize) -> f32 {
        // SIMD is ~3x faster than CPU for vectorizable operations
        let cpu_backend = CpuBackend::new();
        <CpuBackend as ComputeBackend<T>>::estimate_performance(&cpu_backend, problem_size) * 3.0
    }

    fn memory_requirements(&self, problem_size: usize) -> u64 {
        let cpu_backend = CpuBackend::new();
        <CpuBackend as ComputeBackend<T>>::memory_requirements(&cpu_backend, problem_size)
    }
}

/// Backend selector for intelligent backend choice
#[derive(Debug, Clone)]
pub struct BackendSelector {
    cpu_backend: CpuBackend,
    simd_backend: SimdBackend,
    #[cfg(feature = "gpu")]
    gpu_backend: Option<Arc<crate::webgpu::webgpu_backend::WebGpuBackend>>,
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendSelector {
    pub fn new() -> Self {
        Self {
            cpu_backend: CpuBackend::new(),
            simd_backend: SimdBackend::new(),
            #[cfg(feature = "gpu")]
            gpu_backend: None,
        }
    }

    #[cfg(feature = "gpu")]
    pub async fn with_gpu(mut self) -> ComputeResult<Self> {
        match crate::webgpu::webgpu_backend::WebGpuBackend::new().await {
            Ok(gpu_backend) => {
                self.gpu_backend = Some(Arc::new(gpu_backend));
                Ok(self)
            }
            Err(e) => {
                log::warn!("Failed to initialize GPU backend: {}", e);
                Ok(self) // Continue without GPU
            }
        }
    }

    /// Select the best backend for a given workload
    pub fn select_backend<T: Float + Send + Sync>(&self, problem_size: usize) -> BackendType {
        // Simple heuristic for backend selection
        #[cfg(feature = "gpu")]
        {
            if let Some(_gpu) = &self.gpu_backend {
                if problem_size > 1000 { // Large problems benefit from GPU
                    return BackendType::WebGPU;
                }
            }
        }

        if problem_size > 100 { // Medium problems benefit from SIMD
            BackendType::SIMD
        } else {
            BackendType::CPU
        }
    }

    /// Get a CPU backend reference
    pub fn get_cpu_backend(&self) -> &CpuBackend {
        &self.cpu_backend
    }

    /// Get a SIMD backend reference
    pub fn get_simd_backend(&self) -> &SimdBackend {
        &self.simd_backend
    }

    #[cfg(feature = "gpu")]
    /// Get a GPU backend reference if available
    pub fn get_gpu_backend(&self) -> ComputeResult<&crate::webgpu::webgpu_backend::WebGpuBackend> {
        if let Some(gpu) = &self.gpu_backend {
            Ok(gpu.as_ref())
        } else {
            Err(ComputeError::device_unavailable("GPU backend not available"))
        }
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_dims() {
        let dims = MatrixDims::new(3, 4);
        assert_eq!(dims.rows, 3);
        assert_eq!(dims.cols, 4);
        assert_eq!(dims.size(), 12);

        let other = MatrixDims::new(4, 2);
        assert!(dims.is_compatible_for_multiply(&other));
        
        let incompatible = MatrixDims::new(3, 2);
        assert!(!dims.is_compatible_for_multiply(&incompatible));
    }

    #[test]
    fn test_backend_types() {
        assert_eq!(BackendType::CPU.to_string(), "CPU");
        assert_eq!(BackendType::SIMD.to_string(), "SIMD");
        assert_eq!(BackendType::WebGPU.to_string(), "WebGPU");
    }

    #[tokio::test]
    async fn test_cpu_backend_basic() {
        let backend = CpuBackend::new();
        assert_eq!(<CpuBackend as ComputeBackend<f32>>::backend_type(&backend), BackendType::CPU);
        assert!(<CpuBackend as ComputeBackend<f32>>::is_available(&backend));

        // Test simple matrix-vector multiply
        let matrix = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 matrix
        let vector = vec![1.0f32, 2.0];
        let dims = MatrixDims::new(2, 2);

        let result = backend.matrix_vector_multiply(&matrix, &vector, dims).await.unwrap();
        assert_eq!(result, vec![5.0, 11.0]); // [1*1+2*2, 3*1+4*2] = [5, 11]
    }

    #[test]
    fn test_backend_selector() {
        let selector = BackendSelector::new();
        
        // Small problems should use CPU
        assert_eq!(selector.select_backend::<f32>(50), BackendType::CPU);
        
        // Medium problems should use SIMD
        assert_eq!(selector.select_backend::<f32>(500), BackendType::SIMD);
        
        // Large problems would use GPU if available
        let backend_type = selector.select_backend::<f32>(5000);
        #[cfg(feature = "gpu")]
        {
            // Might be GPU or SIMD depending on availability
            assert!(matches!(backend_type, BackendType::WebGPU | BackendType::SIMD));
        }
        #[cfg(not(feature = "gpu"))]
        {
            assert_eq!(backend_type, BackendType::SIMD);
        }
    }
}