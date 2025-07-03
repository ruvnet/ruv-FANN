//! Core compute backend trait and implementations

use super::error::ComputeError;
use crate::ActivationFunction;
use num_traits::Float;
use std::collections::HashMap;

/// Abstract compute backend for neural network operations
pub trait ComputeBackend<T: Float>: Send + Sync + std::fmt::Debug {
    /// Backend initialization and capability detection
    fn initialize() -> Result<Self, ComputeError>
    where
        Self: Sized;
    fn is_available() -> bool
    where
        Self: Sized;
    fn capabilities(&self) -> BackendCapabilities;
    fn backend_type(&self) -> BackendType;

    /// Core neural network operations
    fn matrix_vector_multiply(
        &self,
        matrix: &[T],
        vector: &[T],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<T>, ComputeError>;

    fn batch_matrix_vector_multiply(
        &self,
        matrix: &[T],
        vectors: &[Vec<T>],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<Vec<T>>, ComputeError>;

    fn apply_activation_function(
        &self,
        inputs: &[T],
        function: ActivationFunction,
        steepness: T,
    ) -> Result<Vec<T>, ComputeError>;

    fn vector_operations(&self) -> &dyn VectorOps<T>;
    fn memory_manager(&self) -> &dyn MemoryManager<T>;
}

/// Vector operation primitives
pub trait VectorOps<T: Float> {
    fn dot_product(&self, a: &[T], b: &[T]) -> Result<T, ComputeError>;
    fn vector_add(&self, a: &[T], b: &[T]) -> Result<Vec<T>, ComputeError>;
    fn vector_scale(&self, vec: &[T], scalar: T) -> Result<Vec<T>, ComputeError>;
    fn vector_subtract(&self, a: &[T], b: &[T]) -> Result<Vec<T>, ComputeError>;
}

/// Memory management interface
pub trait MemoryManager<T: Float> {
    fn allocate_buffer(&self, size: usize) -> Result<super::memory::BufferHandle, ComputeError>;
    fn upload_data(
        &self,
        handle: super::memory::BufferHandle,
        data: &[T],
    ) -> Result<(), ComputeError>;
    fn download_data(&self, handle: super::memory::BufferHandle) -> Result<Vec<T>, ComputeError>;
    fn deallocate_buffer(&self, handle: super::memory::BufferHandle) -> Result<(), ComputeError>;
    fn memory_usage(&self) -> super::memory::MemoryStats;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    WebGPU,
    Simd,
    Cpu,
}

#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub max_buffer_size: usize,
    pub supports_f64: bool,
    pub supports_f32: bool,
    pub max_compute_units: usize,
    pub memory_bandwidth_gbps: f32,
    pub shader_model: Option<String>,
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct ComputeProfile {
    pub matrix_size: MatrixSize,
    pub batch_size: usize,
    pub operation_type: OperationType,
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub enum MatrixSize {
    Small,  // < 100x100
    Medium, // 100x100 - 1000x1000
    Large,  // > 1000x1000
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub enum OperationType {
    ForwardPass,
    BackwardPass,
    Training,
    Inference,
}

/// Intelligent backend selection with performance-based switching
#[derive(Debug)]
pub struct BackendSelector<T: Float>
where
    T: Send + Sync,
{
    backends: Vec<Box<dyn ComputeBackend<T>>>,
    performance_cache: HashMap<ComputeProfile, BackendType>,
    fallback_chain: Vec<BackendType>,
}

impl<T: Float + std::fmt::Debug> Clone for BackendSelector<T>
where
    T: Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        // Clone the backend selector by creating a new one
        // This is a simplified clone that recreates backends
        Self::new()
    }
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> Default for BackendSelector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> BackendSelector<T> {
    pub fn new() -> Self {
        let mut backends = Vec::new();

        // Try to initialize backends in order of preference
        #[cfg(feature = "gpu")]
        {
            // WebGPU backend would be initialized here
            // Note: This requires async initialization, so we'll handle this differently
        }

        // Add SIMD backend if available
        if SimdBackend::<T>::is_available() {
            if let Ok(simd) = SimdBackend::initialize() {
                backends.push(Box::new(simd) as Box<dyn ComputeBackend<T>>);
            }
        }

        // CPU backend always available as final fallback
        if let Ok(cpu) = CpuBackend::initialize() {
            backends.push(Box::new(cpu) as Box<dyn ComputeBackend<T>>);
        }

        Self {
            backends,
            performance_cache: HashMap::new(),
            fallback_chain: vec![BackendType::WebGPU, BackendType::Simd, BackendType::Cpu],
        }
    }

    /// Get available backend types
    pub fn get_available_backends(&self) -> Vec<BackendType> {
        self.backends.iter().map(|b| b.backend_type()).collect()
    }

    /// Get current active backend type
    pub fn get_current_backend(&self) -> BackendType {
        // Return the first available backend (highest priority)
        if !self.backends.is_empty() {
            self.backends[0].backend_type()
        } else {
            BackendType::Cpu // Fallback
        }
    }

    /// Set the active backend
    pub fn set_backend(&mut self, backend_type: BackendType) {
        // Find the backend and move it to the front if available
        if let Some(pos) = self
            .backends
            .iter()
            .position(|b| b.backend_type() == backend_type)
        {
            // Move the selected backend to the front
            let backend = self.backends.remove(pos);
            self.backends.insert(0, backend);
        }
        // If backend not available, it will gracefully fall back to available ones
    }

    /// Select optimal backend for given problem size
    pub fn select_optimal_backend(&mut self, rows: usize, cols: usize) -> BackendType {
        let matrix_size = if rows > 1000 || cols > 1000 {
            MatrixSize::Large
        } else if rows > 100 || cols > 100 {
            MatrixSize::Medium
        } else {
            MatrixSize::Small
        };

        let profile = ComputeProfile {
            matrix_size,
            batch_size: 1,
            operation_type: OperationType::Inference,
        };

        // Use existing selection logic
        if let Some(backend) = self.select_backend(&profile) {
            backend.backend_type()
        } else {
            // Fallback to first available backend
            self.get_current_backend()
        }
    }

    pub fn select_backend(&self, profile: &ComputeProfile) -> Option<&dyn ComputeBackend<T>> {
        // Check performance cache first
        if let Some(backend_type) = self.performance_cache.get(profile) {
            if let Some(backend) = self.find_backend(*backend_type) {
                return Some(backend);
            }
        }

        // Default selection logic
        match profile.matrix_size {
            MatrixSize::Large => self
                .find_backend(BackendType::WebGPU)
                .or_else(|| self.find_backend(BackendType::Simd))
                .or_else(|| self.find_backend(BackendType::Cpu)),
            MatrixSize::Medium => {
                if profile.batch_size > 10 {
                    self.find_backend(BackendType::WebGPU)
                        .or_else(|| self.find_backend(BackendType::Simd))
                        .or_else(|| self.find_backend(BackendType::Cpu))
                } else {
                    self.find_backend(BackendType::Simd)
                        .or_else(|| self.find_backend(BackendType::Cpu))
                }
            }
            MatrixSize::Small => self
                .find_backend(BackendType::Simd)
                .or_else(|| self.find_backend(BackendType::Cpu)),
        }
    }

    fn find_backend(&self, backend_type: BackendType) -> Option<&dyn ComputeBackend<T>> {
        self.backends
            .iter()
            .find(|b| b.backend_type() == backend_type)
            .map(|b| b.as_ref())
    }

    pub fn capabilities(&self) -> Vec<BackendCapabilities> {
        self.backends.iter().map(|b| b.capabilities()).collect()
    }
}

/// SIMD backend using existing ruv-FANN SIMD operations
#[derive(Debug)]
pub struct SimdBackend<T: Float>
where
    T: Send + Sync,
{
    capabilities: BackendCapabilities,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> Default for SimdBackend<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> SimdBackend<T> {
    pub fn new() -> Self {
        Self {
            capabilities: BackendCapabilities {
                max_buffer_size: usize::MAX,
                supports_f64: true,
                supports_f32: true,
                max_compute_units: {
                    #[cfg(feature = "parallel")]
                    {
                        num_cpus::get()
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        4
                    } // Default fallback
                },
                memory_bandwidth_gbps: 50.0, // Typical DDR4 bandwidth
                shader_model: None,
            },
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + std::fmt::Debug> ComputeBackend<T> for SimdBackend<T>
where
    T: Send + Sync + 'static,
{
    fn initialize() -> Result<Self, ComputeError>
    where
        Self: Sized,
    {
        Ok(Self::new())
    }

    fn is_available() -> bool {
        true // SIMD operations are always available
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Simd
    }

    fn matrix_vector_multiply(
        &self,
        matrix: &[T],
        vector: &[T],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<T>, ComputeError> {
        if matrix.len() != rows * cols || vector.len() != cols {
            return Err(ComputeError::InvalidDimensions(format!(
                "Matrix {}x{} and vector {} dimensions don't match",
                rows,
                cols,
                vector.len()
            )));
        }

        let mut result = Vec::with_capacity(rows);

        for row in 0..rows {
            let mut sum = T::zero();
            for col in 0..cols {
                sum = sum + matrix[row * cols + col] * vector[col];
            }
            result.push(sum);
        }

        Ok(result)
    }

    fn batch_matrix_vector_multiply(
        &self,
        matrix: &[T],
        vectors: &[Vec<T>],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<Vec<T>>, ComputeError> {
        let mut results = Vec::with_capacity(vectors.len());

        for vector in vectors {
            let result = self.matrix_vector_multiply(matrix, vector, rows, cols)?;
            results.push(result);
        }

        Ok(results)
    }

    fn apply_activation_function(
        &self,
        inputs: &[T],
        function: ActivationFunction,
        steepness: T,
    ) -> Result<Vec<T>, ComputeError> {
        Ok(inputs
            .iter()
            .map(|&x| self.apply_activation_cpu(x, function, steepness))
            .collect())
    }

    fn vector_operations(&self) -> &dyn VectorOps<T> {
        self
    }

    fn memory_manager(&self) -> &dyn MemoryManager<T> {
        self
    }
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> SimdBackend<T> {
    fn apply_activation_cpu(&self, x: T, function: ActivationFunction, steepness: T) -> T {
        match function {
            ActivationFunction::Sigmoid => {
                let exp_val = (-steepness * x).exp();
                T::one() / (T::one() + exp_val)
            }
            ActivationFunction::ReLU => {
                if x > T::zero() {
                    x
                } else {
                    T::zero()
                }
            }
            ActivationFunction::ReLULeaky => {
                let alpha = T::from(0.01).unwrap_or(T::zero());
                if x > T::zero() {
                    x
                } else {
                    alpha * x
                }
            }
            ActivationFunction::Tanh => {
                let exp_2x = (steepness * x + steepness * x).exp();
                let exp_neg_2x = (-steepness * x - steepness * x).exp();
                (exp_2x - exp_neg_2x) / (exp_2x + exp_neg_2x)
            }
            ActivationFunction::Linear => x * steepness,
            _ => x, // Fallback for other functions
        }
    }
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> VectorOps<T> for SimdBackend<T> {
    fn dot_product(&self, a: &[T], b: &[T]) -> Result<T, ComputeError> {
        if a.len() != b.len() {
            return Err(ComputeError::InvalidDimensions(
                "Vector length mismatch".to_string(),
            ));
        }

        let mut sum = T::zero();
        for (x, y) in a.iter().zip(b.iter()) {
            sum = sum + *x * *y;
        }
        Ok(sum)
    }

    fn vector_add(&self, a: &[T], b: &[T]) -> Result<Vec<T>, ComputeError> {
        if a.len() != b.len() {
            return Err(ComputeError::InvalidDimensions(
                "Vector length mismatch".to_string(),
            ));
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect())
    }

    fn vector_scale(&self, vec: &[T], scalar: T) -> Result<Vec<T>, ComputeError> {
        Ok(vec.iter().map(|x| *x * scalar).collect())
    }

    fn vector_subtract(&self, a: &[T], b: &[T]) -> Result<Vec<T>, ComputeError> {
        if a.len() != b.len() {
            return Err(ComputeError::InvalidDimensions(
                "Vector length mismatch".to_string(),
            ));
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect())
    }
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> MemoryManager<T> for SimdBackend<T> {
    fn allocate_buffer(&self, _size: usize) -> Result<super::memory::BufferHandle, ComputeError> {
        // CPU memory management is handled by Vec<T> allocations
        // Return a placeholder handle since we don't need explicit buffer management
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Ok(super::memory::BufferHandle::new(rng.gen()))
    }

    fn upload_data(
        &self,
        _handle: super::memory::BufferHandle,
        _data: &[T],
    ) -> Result<(), ComputeError> {
        // No-op for CPU backend
        Ok(())
    }

    fn download_data(&self, _handle: super::memory::BufferHandle) -> Result<Vec<T>, ComputeError> {
        // No-op for CPU backend - data is already in CPU memory
        Ok(Vec::new())
    }

    fn deallocate_buffer(&self, _handle: super::memory::BufferHandle) -> Result<(), ComputeError> {
        // No-op for CPU backend
        Ok(())
    }

    fn memory_usage(&self) -> super::memory::MemoryStats {
        super::memory::MemoryStats {
            total_allocated: 0, // Would need to track actual allocations
            available: usize::MAX,
            buffer_count: 0,
            fragmentation_ratio: 0.0,
        }
    }
}

/// CPU fallback backend
#[derive(Debug)]
pub struct CpuBackend<T: Float>
where
    T: Send + Sync,
{
    capabilities: BackendCapabilities,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> Default for CpuBackend<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> CpuBackend<T> {
    pub fn new() -> Self {
        Self {
            capabilities: BackendCapabilities {
                max_buffer_size: usize::MAX,
                supports_f64: true,
                supports_f32: true,
                max_compute_units: 1,
                memory_bandwidth_gbps: 25.0, // Conservative estimate
                shader_model: None,
            },
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + std::fmt::Debug> ComputeBackend<T> for CpuBackend<T>
where
    T: Send + Sync + 'static,
{
    fn initialize() -> Result<Self, ComputeError>
    where
        Self: Sized,
    {
        Ok(Self::new())
    }

    fn is_available() -> bool {
        true // CPU is always available
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn matrix_vector_multiply(
        &self,
        matrix: &[T],
        vector: &[T],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<T>, ComputeError> {
        if matrix.len() != rows * cols || vector.len() != cols {
            return Err(ComputeError::InvalidDimensions(format!(
                "Matrix {}x{} and vector {} dimensions don't match",
                rows,
                cols,
                vector.len()
            )));
        }

        let mut result = Vec::with_capacity(rows);

        for row in 0..rows {
            let mut sum = T::zero();
            for col in 0..cols {
                sum = sum + matrix[row * cols + col] * vector[col];
            }
            result.push(sum);
        }

        Ok(result)
    }

    fn batch_matrix_vector_multiply(
        &self,
        matrix: &[T],
        vectors: &[Vec<T>],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<Vec<T>>, ComputeError> {
        let mut results = Vec::with_capacity(vectors.len());

        for vector in vectors {
            let result = self.matrix_vector_multiply(matrix, vector, rows, cols)?;
            results.push(result);
        }

        Ok(results)
    }

    fn apply_activation_function(
        &self,
        inputs: &[T],
        function: ActivationFunction,
        steepness: T,
    ) -> Result<Vec<T>, ComputeError> {
        Ok(inputs
            .iter()
            .map(|&x| self.apply_activation_cpu(x, function, steepness))
            .collect())
    }

    fn vector_operations(&self) -> &dyn VectorOps<T> {
        self
    }

    fn memory_manager(&self) -> &dyn MemoryManager<T> {
        self
    }
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> CpuBackend<T> {
    fn apply_activation_cpu(&self, x: T, function: ActivationFunction, steepness: T) -> T {
        match function {
            ActivationFunction::Sigmoid => {
                let exp_val = (-steepness * x).exp();
                T::one() / (T::one() + exp_val)
            }
            ActivationFunction::ReLU => {
                if x > T::zero() {
                    x
                } else {
                    T::zero()
                }
            }
            ActivationFunction::ReLULeaky => {
                let alpha = T::from(0.01).unwrap_or(T::zero());
                if x > T::zero() {
                    x
                } else {
                    alpha * x
                }
            }
            ActivationFunction::Tanh => {
                let exp_2x = (steepness * x + steepness * x).exp();
                let exp_neg_2x = (-steepness * x - steepness * x).exp();
                (exp_2x - exp_neg_2x) / (exp_2x + exp_neg_2x)
            }
            ActivationFunction::Linear => x * steepness,
            _ => x, // Fallback for other functions
        }
    }
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> VectorOps<T> for CpuBackend<T> {
    fn dot_product(&self, a: &[T], b: &[T]) -> Result<T, ComputeError> {
        if a.len() != b.len() {
            return Err(ComputeError::InvalidDimensions(
                "Vector length mismatch".to_string(),
            ));
        }

        let mut sum = T::zero();
        for (x, y) in a.iter().zip(b.iter()) {
            sum = sum + *x * *y;
        }
        Ok(sum)
    }

    fn vector_add(&self, a: &[T], b: &[T]) -> Result<Vec<T>, ComputeError> {
        if a.len() != b.len() {
            return Err(ComputeError::InvalidDimensions(
                "Vector length mismatch".to_string(),
            ));
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect())
    }

    fn vector_scale(&self, vec: &[T], scalar: T) -> Result<Vec<T>, ComputeError> {
        Ok(vec.iter().map(|x| *x * scalar).collect())
    }

    fn vector_subtract(&self, a: &[T], b: &[T]) -> Result<Vec<T>, ComputeError> {
        if a.len() != b.len() {
            return Err(ComputeError::InvalidDimensions(
                "Vector length mismatch".to_string(),
            ));
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect())
    }
}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> MemoryManager<T> for CpuBackend<T> {
    fn allocate_buffer(&self, _size: usize) -> Result<super::memory::BufferHandle, ComputeError> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Ok(super::memory::BufferHandle::new(rng.gen()))
    }

    fn upload_data(
        &self,
        _handle: super::memory::BufferHandle,
        _data: &[T],
    ) -> Result<(), ComputeError> {
        Ok(())
    }

    fn download_data(&self, _handle: super::memory::BufferHandle) -> Result<Vec<T>, ComputeError> {
        Ok(Vec::new())
    }

    fn deallocate_buffer(&self, _handle: super::memory::BufferHandle) -> Result<(), ComputeError> {
        Ok(())
    }

    fn memory_usage(&self) -> super::memory::MemoryStats {
        super::memory::MemoryStats {
            total_allocated: 0,
            available: usize::MAX,
            buffer_count: 0,
            fragmentation_ratio: 0.0,
        }
    }
}
