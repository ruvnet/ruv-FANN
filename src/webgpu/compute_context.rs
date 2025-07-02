//! Compute context for managing backend selection and GPU/CPU operations
//! 
//! This module bridges the gap between Network and GPU backend with intelligent
//! backend selection and automatic fallback mechanisms.

use std::sync::Arc;
use num_traits::Float;
use crate::webgpu::{
    backend::{ComputeBackend, BackendSelector, BackendType, MatrixDims, ActivationFunction as GpuActivationFunction},
    error::{ComputeError, ComputeResult},
};
use crate::{ActivationFunction, Layer};

/// Compute context manages backend selection and operation dispatch
pub struct ComputeContext<T: Float + Send + Sync> {
    backend_selector: BackendSelector,
    current_backend: BackendType,
    gpu_enabled: bool,
    // Cache for converted weights to avoid repeated conversions
    weight_cache: std::collections::HashMap<usize, (Vec<T>, MatrixDims)>,
}

impl<T: Float + Send + Sync> ComputeContext<T> {
    /// Create a new compute context with automatic backend detection
    pub async fn new() -> ComputeResult<Self> {
        let backend_selector = BackendSelector::new();
        
        // Try to initialize GPU backend
        #[cfg(feature = "gpu")]
        let (gpu_enabled, backend_selector) = match backend_selector.clone().with_gpu().await {
            Ok(selector) => (true, selector),
            Err(_) => (false, backend_selector),
        };
        
        #[cfg(not(feature = "gpu"))]
        let gpu_enabled = false;
        
        // Select initial backend based on availability - start with SIMD for all cases
        let current_backend = BackendType::SIMD;
        
        Ok(Self {
            backend_selector,
            current_backend,
            gpu_enabled,
            weight_cache: std::collections::HashMap::new(),
        })
    }
    
    /// Create a compute context with CPU-only backend (for testing)
    pub fn cpu_only() -> Self {
        Self {
            backend_selector: BackendSelector::new(),
            current_backend: BackendType::CPU,
            gpu_enabled: false,
            weight_cache: std::collections::HashMap::new(),
        }
    }
    
    /// Select the best backend for a given problem size
    pub fn select_backend(&mut self, problem_size: usize) -> BackendType {
        let selected = self.backend_selector.select_backend::<T>(problem_size);
        
        // Only use GPU if it's actually available
        if selected == BackendType::WebGPU && !self.gpu_enabled {
            self.current_backend = BackendType::SIMD;
        } else {
            self.current_backend = selected;
        }
        
        self.current_backend
    }
    
    /// Get the current backend type
    pub fn current_backend(&self) -> BackendType {
        self.current_backend
    }
    
    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_enabled
    }
    
    /// Convert ActivationFunction to GPU ActivationFunction using built-in conversion
    fn convert_activation_function(activation: ActivationFunction) -> GpuActivationFunction {
        // Use the built-in conversion method from ActivationFunction
        activation.to_gpu_activation().unwrap_or(GpuActivationFunction::Sigmoid)
    }
    
    /// Execute a layer computation using the appropriate backend
    pub async fn compute_layer(
        &mut self,
        layer: &Layer<T>,
        layer_id: usize,
        inputs: &[T],
        problem_size: usize,
    ) -> ComputeResult<Vec<T>>
    where
        T: bytemuck::Pod + Clone + 'static,
    {
        // Select backend based on problem size
        let backend_type = self.select_backend(problem_size);
        
        // Convert layer to matrix format
        let (weights, dims) = self.get_layer_weights(layer, layer_id)?;
        
        // Validate input size
        if inputs.len() != dims.cols {
            return Err(ComputeError::invalid_dimensions(
                format!("{}", dims.cols),
                format!("{}", inputs.len()),
            ));
        }
        
        // Execute computation based on selected backend
        let result = match backend_type {
            #[cfg(feature = "gpu")]
            BackendType::WebGPU => {
                match self.backend_selector.get_gpu_backend() {
                    Ok(gpu_backend) => {
                        // Try GPU computation
                        match gpu_backend.matrix_vector_multiply(&weights, inputs, dims).await {
                            Ok(outputs) => {
                                // Apply activation function from first neuron
                                let activation = layer.neurons.first()
                                    .map(|n| n.activation_function)
                                    .unwrap_or(ActivationFunction::Sigmoid);
                                let gpu_activation = Self::convert_activation_function(activation);
                                gpu_backend.activation_function(&outputs, gpu_activation).await
                            }
                            Err(e) => {
                                log::warn!("GPU computation failed, falling back to CPU: {}", e);
                                // Fallback to CPU
                                self.current_backend = BackendType::CPU;
                                self.compute_layer_cpu(layer, inputs).await
                            }
                        }
                    }
                    Err(_) => {
                        // GPU not available, fallback to CPU
                        self.current_backend = BackendType::CPU;
                        self.compute_layer_cpu(layer, inputs).await
                    }
                }
            }
            BackendType::SIMD => {
                let simd_backend = self.backend_selector.get_simd_backend();
                let outputs = simd_backend.matrix_vector_multiply(&weights, inputs, dims).await?;
                let activation = layer.neurons.first()
                    .map(|n| n.activation_function)
                    .unwrap_or(ActivationFunction::Sigmoid);
                let gpu_activation = Self::convert_activation_function(activation);
                simd_backend.activation_function(&outputs, gpu_activation).await
            }
            BackendType::CPU => {
                self.compute_layer_cpu(layer, inputs).await
            }
            _ => {
                // Unsupported backend, fallback to CPU
                self.compute_layer_cpu(layer, inputs).await
            }
        }?;
        
        Ok(result)
    }
    
    /// Convert a layer to weight matrix format and cache it
    fn get_layer_weights(&mut self, layer: &Layer<T>, layer_id: usize) -> ComputeResult<(Vec<T>, MatrixDims)> {
        if let Some(cached) = self.weight_cache.get(&layer_id) {
            return Ok(cached.clone());
        }
        
        // Convert layer connections to matrix format
        let input_size = layer.neurons.first()
            .map(|n| n.connections.len())
            .unwrap_or(0);
        let output_size = layer.neurons.len();
        
        if input_size == 0 || output_size == 0 {
            return Err(ComputeError::invalid_dimensions(
                "0x0".to_string(),
                format!("{}x{}", output_size, input_size),
            ));
        }
        
        let mut weights = Vec::with_capacity(output_size * input_size);
        
        // Build weight matrix row by row (each row = one output neuron's weights)
        for neuron in &layer.neurons {
            if neuron.is_bias {
                continue; // Skip bias neurons for now
            }
            
            for connection in &neuron.connections {
                weights.push(connection.weight);
            }
            
            // Pad if necessary to match expected input size
            while weights.len() % input_size != 0 {
                weights.push(T::zero());
            }
        }
        
        let dims = MatrixDims::new(output_size, input_size);
        let result = (weights, dims);
        
        // Cache the result
        self.weight_cache.insert(layer_id, result.clone());
        
        Ok(result)
    }
    
    /// CPU fallback computation for a layer
    async fn compute_layer_cpu(&self, layer: &Layer<T>, inputs: &[T]) -> ComputeResult<Vec<T>> {
        let cpu_backend = self.backend_selector.get_cpu_backend();
        
        // Convert layer to matrix format
        let input_size = inputs.len();
        let output_size = layer.neurons.iter().filter(|n| !n.is_bias).count();
        
        if output_size == 0 {
            return Ok(Vec::new());
        }
        
        let mut weights = Vec::with_capacity(output_size * input_size);
        
        // Build weight matrix
        for neuron in &layer.neurons {
            if neuron.is_bias {
                continue;
            }
            
            for (i, connection) in neuron.connections.iter().enumerate() {
                if i < input_size {
                    weights.push(connection.weight);
                }
            }
            
            // Pad if necessary
            while weights.len() % input_size != 0 {
                weights.push(T::zero());
            }
        }
        
        let dims = MatrixDims::new(output_size, input_size);
        
        // Matrix-vector multiplication
        let outputs = cpu_backend.matrix_vector_multiply(&weights, inputs, dims).await?;
        
        // Apply activation function
        let activation = layer.neurons.first()
            .map(|n| n.activation_function)
            .unwrap_or(ActivationFunction::Sigmoid);
        let gpu_activation = Self::convert_activation_function(activation);
        cpu_backend.activation_function(&outputs, gpu_activation).await
    }
    
    /// Clear the weight cache (call when network weights change)
    pub fn clear_cache(&mut self) {
        self.weight_cache.clear();
    }
}

/// Performance statistics for compute context
#[derive(Debug, Clone)]
pub struct ComputePerformanceStats {
    pub current_backend: BackendType,
    pub gpu_available: bool,
    pub cache_size: usize,
}