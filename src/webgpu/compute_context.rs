//! ComputeContext bridge for `Network<T>` integration with advanced WebGPU backend
//!
//! This module provides a bridge between the existing `Network<T>` structure and the
//! advanced WebGPU backend, enabling seamless GPU acceleration with DAA compatibility.

use num_traits::Float;
use std::collections::HashMap;
use std::sync::Arc;

use crate::webgpu::{
    backend::{BackendSelector, BackendType, ComputeBackend},
    error::{ComputeError, ComputeResult},
};
use crate::{ActivationFunction, Layer, Network};

#[cfg(feature = "gpu")]
use crate::webgpu::webgpu_backend::WebGPUBackend;

// These types are used across the module regardless of webgpu feature
#[derive(Clone, Copy, Debug)]
pub struct MatrixDims {
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct DeviceCapabilities {
    pub max_buffer_size: u64,
    pub max_workgroup_size: usize,
    pub shared_memory_size: u64,
}

#[derive(Clone, Debug, Default)]
pub struct PerformanceStats {
    pub kernel_time_ms: f64,
    pub memory_transfer_ms: f64,
    pub total_time_ms: f64,
}

/// ComputeContext manages backend selection and operation dispatch for `Network<T>`
///
/// This bridge provides seamless integration between `Network<T>` and the advanced
/// WebGPU backend while maintaining full compatibility with existing code.
#[derive(Debug)]
pub struct ComputeContext<T: Float + std::fmt::Debug + Send + Sync + 'static> {
    /// Backend selector for intelligent backend switching
    backend_selector: BackendSelector<T>,
    /// Current backend type being used
    current_backend: BackendType,
    /// WebGPU backend instance (when available)
    #[cfg(feature = "gpu")]
    webgpu_backend: Option<Arc<WebGPUBackend<T>>>,
    #[cfg(not(feature = "gpu"))]
    webgpu_backend: Option<()>,
    /// GPU acceleration enabled flag
    gpu_enabled: bool,
    /// Performance tracking and optimization
    performance_tracker: Arc<std::sync::Mutex<PerformanceTracker>>,
    /// Cache for converted weights to avoid repeated conversions
    weight_cache: std::collections::HashMap<usize, (Vec<T>, MatrixDims)>,
}

/// Performance tracking for optimization decisions
#[derive(Debug)]
struct PerformanceTracker {
    operation_counts: HashMap<String, u64>,
    execution_times: HashMap<String, Vec<f64>>,
    backend_switches: HashMap<BackendType, u64>,
    optimization_events: Vec<OptimizationEvent>,
}

#[derive(Debug, Clone)]
struct OptimizationEvent {
    timestamp: std::time::Instant,
    event_type: String,
    backend_from: BackendType,
    backend_to: BackendType,
    performance_gain: f64,
}

impl<T: Float + Send + Sync + std::fmt::Debug + 'static> ComputeContext<T> {
    /// Create a new compute context with automatic backend detection
    pub async fn new() -> ComputeResult<Self> {
        let backend_selector = BackendSelector::new();

        // Try to initialize WebGPU backend
        #[cfg(feature = "gpu")]
        let (webgpu_backend, gpu_enabled) = match WebGPUBackend::<T>::initialize().await {
            Ok(backend) => (Some(Arc::new(backend)), true),
            Err(_) => (None, false),
        };

        #[cfg(not(feature = "gpu"))]
        let (webgpu_backend, gpu_enabled) = (None, false);

        // Select initial backend based on availability
        let current_backend = if gpu_enabled {
            BackendType::WebGPU
        } else {
            BackendType::Simd
        };

        Ok(Self {
            backend_selector,
            current_backend,
            webgpu_backend,
            gpu_enabled,
            performance_tracker: Arc::new(std::sync::Mutex::new(PerformanceTracker::new())),
            weight_cache: HashMap::new(),
        })
    }

    /// Create a compute context with CPU-only backend (for testing/fallback)
    pub fn cpu_only() -> Self {
        Self {
            backend_selector: BackendSelector::new(),
            current_backend: BackendType::Cpu,
            webgpu_backend: None,
            gpu_enabled: false,
            performance_tracker: Arc::new(std::sync::Mutex::new(PerformanceTracker::new())),
            weight_cache: HashMap::new(),
        }
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_enabled && self.webgpu_backend.is_some()
    }

    /// Get current backend type
    pub fn current_backend(&self) -> BackendType {
        self.current_backend
    }

    /// Select optimal backend for given problem size
    pub fn select_backend(&mut self, problem_size: usize) -> BackendType {
        let profile = crate::webgpu::backend::ComputeProfile {
            matrix_size: match problem_size {
                0..=10000 => crate::webgpu::backend::MatrixSize::Small,
                10001..=1000000 => crate::webgpu::backend::MatrixSize::Medium,
                _ => crate::webgpu::backend::MatrixSize::Large,
            },
            batch_size: 1,
            operation_type: crate::webgpu::backend::OperationType::Inference,
        };

        let selected = self
            .backend_selector
            .select_backend(&profile)
            .map(|backend| backend.backend_type())
            .unwrap_or(BackendType::Cpu);

        // Only use GPU if it's actually available
        if selected == BackendType::WebGPU && !self.is_gpu_available() {
            self.current_backend = BackendType::Simd;
        } else {
            self.current_backend = selected;
        }

        self.current_backend
    }

    /// Convert Network layer to matrix format with caching
    fn get_layer_weights(
        &mut self,
        layer: &Layer<T>,
        layer_id: usize,
    ) -> ComputeResult<(Vec<T>, MatrixDims)> {
        // Check cache first
        if let Some(cached) = self.weight_cache.get(&layer_id) {
            return Ok(cached.clone());
        }

        // Debug layer information
        println!("Converting layer {} to matrix format", layer_id);
        println!("  Layer has {} neurons", layer.neurons.len());

        // In FANN networks, bias neurons are included in the layer
        // We need to find non-bias neurons for the output
        let non_bias_neurons: Vec<&crate::Neuron<T>> =
            layer.neurons.iter().filter(|n| !n.is_bias).collect();

        println!("  Layer has {} non-bias neurons", non_bias_neurons.len());

        // Convert layer connections to matrix format
        // In a FANN network, the input size is the number of connections on each neuron
        // (all neurons should have the same number of connections)
        let input_size = if let Some(neuron) = non_bias_neurons.first() {
            println!(
                "  First neuron has {} connections",
                neuron.connections.len()
            );
            neuron.connections.len()
        } else {
            println!("  No non-bias neurons found!");
            return Err(ComputeError::InvalidDimensions(format!(
                "Layer {} has no non-bias neurons",
                layer_id
            )));
        };

        let output_size = non_bias_neurons.len();

        println!(
            "  Matrix dimensions: {}x{} (output_size x input_size)",
            output_size, input_size
        );

        if input_size == 0 || output_size == 0 {
            return Err(ComputeError::InvalidDimensions(format!(
                "Invalid layer dimensions: {}x{}",
                output_size, input_size
            )));
        }

        let mut weights = Vec::with_capacity(output_size * input_size);

        // Build weight matrix row by row (each row = one output neuron's weights)
        for neuron in &non_bias_neurons {
            // Ensure we have enough connections
            if neuron.connections.len() != input_size {
                return Err(ComputeError::InvalidDimensions(format!(
                    "Neuron has {} connections, expected {}",
                    neuron.connections.len(),
                    input_size
                )));
            }

            // Add weights for this neuron to the matrix
            for i in 0..input_size {
                weights.push(neuron.connections[i].weight);
            }
        }

        if weights.len() != output_size * input_size {
            return Err(ComputeError::InvalidDimensions(format!(
                "Weight matrix size mismatch: got {}, expected {}",
                weights.len(),
                output_size * input_size
            )));
        }

        let dims = MatrixDims {
            rows: output_size,
            cols: input_size,
        };
        let result = (weights, dims);

        // Cache the result
        self.weight_cache.insert(layer_id, result.clone());

        Ok(result)
    }

    /// Execute forward pass for a layer with optimal backend selection
    pub async fn compute_layer_forward(
        &mut self,
        layer: &Layer<T>,
        layer_id: usize,
        inputs: &[T],
    ) -> ComputeResult<Vec<T>>
    where
        T: Clone + num_traits::ToPrimitive + 'static,
    {
        let start_time = std::time::Instant::now();

        // Get layer weights
        let (weights, dims) = self.get_layer_weights(layer, layer_id)?;

        // Check if we need to append a bias input (value 1.0)
        let mut input_with_bias = inputs.to_vec();
        if dims.cols == inputs.len() + 1 {
            // The extra column is likely for the bias input (common in FANN architecture)
            println!("  Adding bias input to match expected dimensions");
            input_with_bias.push(T::one()); // Add bias input with value 1.0
        } else if inputs.len() != dims.cols {
            return Err(ComputeError::InvalidDimensions(format!(
                "Input size {} doesn't match expected {} and doesn't match bias pattern",
                inputs.len(),
                dims.cols
            )));
        }

        // Select optimal backend for this problem size
        let problem_size = dims.rows * dims.cols;
        let backend_type = self.select_backend(problem_size);

        // Execute computation based on selected backend
        let result = match backend_type {
            BackendType::WebGPU if self.is_gpu_available() => {
                self.compute_layer_gpu(layer, &weights, &input_with_bias, dims)
                    .await
            }
            BackendType::Simd => {
                self.compute_layer_simd(layer, &weights, &input_with_bias, dims)
                    .await
            }
            _ => {
                self.compute_layer_cpu(layer, &weights, &input_with_bias, dims)
                    .await
            }
        };

        // Record performance metrics
        let duration = start_time.elapsed().as_secs_f64();
        if let Ok(mut tracker) = self.performance_tracker.lock() {
            tracker.record_operation("layer_forward", duration, backend_type);
        }

        result
    }

    /// GPU-accelerated layer computation
    async fn compute_layer_gpu(
        &self,
        layer: &Layer<T>,
        weights: &[T],
        inputs: &[T],
        dims: MatrixDims,
    ) -> ComputeResult<Vec<T>>
    where
        T: Clone + num_traits::ToPrimitive + 'static,
    {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref gpu_backend) = self.webgpu_backend {
                // Matrix-vector multiplication
                let outputs =
                    gpu_backend.matrix_vector_multiply(weights, inputs, dims.rows, dims.cols)?;

                // Apply activation function
                // Get activation function from first non-bias neuron
                let activation_function = layer
                    .neurons
                    .iter()
                    .find(|n| !n.is_bias)
                    .map(|n| n.activation_function)
                    .unwrap_or(ActivationFunction::Linear);
                let steepness = T::one();
                gpu_backend.apply_activation_function(&outputs, activation_function, steepness)
            } else {
                Err(ComputeError::GpuUnavailable)
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            Err(ComputeError::GpuUnavailable)
        }
    }

    /// SIMD-optimized layer computation
    async fn compute_layer_simd(
        &self,
        layer: &Layer<T>,
        weights: &[T],
        inputs: &[T],
        dims: MatrixDims,
    ) -> ComputeResult<Vec<T>>
    where
        T: Clone + 'static,
    {
        // Use backend selector to get SIMD backend
        let profile = crate::webgpu::backend::ComputeProfile {
            matrix_size: crate::webgpu::backend::MatrixSize::Medium,
            batch_size: 1,
            operation_type: crate::webgpu::backend::OperationType::Inference,
        };

        if let Some(backend) = self.backend_selector.select_backend(&profile) {
            let outputs = backend.matrix_vector_multiply(weights, inputs, dims.rows, dims.cols)?;
            // Get activation function from first non-bias neuron
            let activation_function = layer
                .neurons
                .iter()
                .find(|n| !n.is_bias)
                .map(|n| n.activation_function)
                .unwrap_or(ActivationFunction::Linear);
            let steepness = T::one();
            backend.apply_activation_function(&outputs, activation_function, steepness)
        } else {
            self.compute_layer_cpu(layer, weights, inputs, dims).await
        }
    }

    /// CPU fallback layer computation
    async fn compute_layer_cpu(
        &self,
        layer: &Layer<T>,
        weights: &[T],
        inputs: &[T],
        dims: MatrixDims,
    ) -> ComputeResult<Vec<T>> {
        let mut outputs = Vec::with_capacity(dims.rows);

        // Manual matrix-vector multiplication
        for row in 0..dims.rows {
            let mut sum = T::zero();
            for col in 0..dims.cols {
                sum = sum + weights[row * dims.cols + col] * inputs[col];
            }
            outputs.push(sum);
        }

        // Apply activation function
        // Get activation function from first non-bias neuron
        let activation_function = layer
            .neurons
            .iter()
            .find(|n| !n.is_bias)
            .map(|n| n.activation_function)
            .unwrap_or(ActivationFunction::Linear);
        let result: Vec<T> = outputs
            .into_iter()
            .map(|x| apply_activation_cpu(x, activation_function, T::one()))
            .collect();

        Ok(result)
    }

    /// Execute complete network forward pass with optimal backend coordination
    pub async fn compute_network_forward(
        &mut self,
        network: &Network<T>,
        inputs: &[T],
    ) -> ComputeResult<Vec<T>>
    where
        T: Clone + num_traits::ToPrimitive + 'static,
    {
        // Validate network has layers
        if network.layers.is_empty() {
            return Err(ComputeError::InvalidDimensions(
                "Network has no layers".to_string(),
            ));
        }

        // Validate input size matches input layer (excluding bias neuron)
        if !network.layers.is_empty() && inputs.len() != network.num_inputs() {
            return Err(ComputeError::InvalidDimensions(format!(
                "Input size {} doesn't match network input size {}",
                inputs.len(),
                network.num_inputs()
            )));
        }

        let mut current_inputs = inputs.to_vec();

        // Process each layer, starting from the first hidden layer (index 1)
        // The input layer (index 0) is just for passing inputs
        for (layer_id, layer) in network.layers.iter().enumerate().skip(1) {
            // Skip input layer (index 0)
            current_inputs = match self
                .compute_layer_forward(layer, layer_id, &current_inputs)
                .await
            {
                Ok(outputs) => outputs,
                Err(e) => {
                    eprintln!("Error in layer {}: {:?}", layer_id, e);
                    return Err(e);
                }
            };
        }

        Ok(current_inputs)
    }

    /// Clear weight cache (call when network weights change)
    pub fn clear_cache(&mut self) {
        self.weight_cache.clear();
    }

    /// Get comprehensive performance statistics
    pub fn get_performance_stats(&self) -> ComputePerformanceStats {
        let tracker_stats = if let Ok(tracker) = self.performance_tracker.lock() {
            Some(tracker.get_stats())
        } else {
            None
        };

        #[cfg(feature = "gpu")]
        let gpu_stats = self.webgpu_backend.as_ref().map(|_gpu_backend| {
            // TODO: Implement get_performance_stats in WebGPUBackend
            PerformanceStats::default()
        });

        #[cfg(not(feature = "gpu"))]
        let gpu_stats = None;

        ComputePerformanceStats {
            current_backend: self.current_backend,
            gpu_available: self.is_gpu_available(),
            cache_size: self.weight_cache.len(),
            tracker_stats,
            gpu_stats,
        }
    }

    /// Get DAA coordination metrics
    pub fn get_daa_metrics(&self) -> DaaCoordinationMetrics {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref _gpu_backend) = self.webgpu_backend {
                DaaCoordinationMetrics {
                    gpu_utilization: 0.0, // TODO: Implement get_daa_metrics in WebGPUBackend
                    memory_efficiency: 1.0,
                    coordination_overhead: 0.0,
                    backend_switches: self.get_backend_switch_count(),
                    optimization_score: self.calculate_optimization_score(),
                }
            } else {
                DaaCoordinationMetrics::default()
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            DaaCoordinationMetrics::default()
        }
    }

    /// Get backend switch count for DAA coordination
    fn get_backend_switch_count(&self) -> u64 {
        if let Ok(tracker) = self.performance_tracker.lock() {
            tracker.backend_switches.values().sum()
        } else {
            0
        }
    }

    /// Calculate optimization score for DAA coordination
    fn calculate_optimization_score(&self) -> f32 {
        if let Ok(tracker) = self.performance_tracker.lock() {
            let total_operations = tracker.operation_counts.values().sum::<u64>();
            if total_operations > 0 {
                let optimization_events = tracker.optimization_events.len() as f32;
                let efficiency = optimization_events / total_operations as f32;
                efficiency.min(1.0)
            } else {
                1.0
            }
        } else {
            0.0
        }
    }
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            operation_counts: HashMap::new(),
            execution_times: HashMap::new(),
            backend_switches: HashMap::new(),
            optimization_events: Vec::new(),
        }
    }

    fn record_operation(&mut self, operation: &str, duration: f64, backend: BackendType) {
        *self
            .operation_counts
            .entry(operation.to_string())
            .or_insert(0) += 1;
        self.execution_times
            .entry(operation.to_string())
            .or_default()
            .push(duration);
        *self.backend_switches.entry(backend).or_insert(0) += 1;
    }

    fn get_stats(&self) -> TrackerStats {
        TrackerStats {
            total_operations: self.operation_counts.values().sum(),
            average_duration: self
                .execution_times
                .values()
                .flat_map(|times| times.iter())
                .sum::<f64>()
                / self
                    .execution_times
                    .values()
                    .map(|times| times.len())
                    .sum::<usize>() as f64,
            backend_distribution: self.backend_switches.clone(),
            optimization_events: self.optimization_events.len(),
        }
    }
}

/// CPU activation function implementation
fn apply_activation_cpu<T: Float>(x: T, function: ActivationFunction, steepness: T) -> T {
    match function {
        ActivationFunction::Linear => x * steepness,
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
        ActivationFunction::Tanh => (steepness * x).tanh(),
        _ => x, // Fallback for other functions
    }
}

/// Comprehensive performance statistics
#[derive(Debug, Clone)]
pub struct ComputePerformanceStats {
    pub current_backend: BackendType,
    pub gpu_available: bool,
    pub cache_size: usize,
    pub tracker_stats: Option<TrackerStats>,
    pub gpu_stats: Option<PerformanceStats>,
}

/// Performance tracker statistics
#[derive(Debug, Clone)]
pub struct TrackerStats {
    pub total_operations: u64,
    pub average_duration: f64,
    pub backend_distribution: HashMap<BackendType, u64>,
    pub optimization_events: usize,
}

/// DAA coordination metrics
#[derive(Debug, Clone)]
pub struct DaaCoordinationMetrics {
    pub gpu_utilization: f32,
    pub memory_efficiency: f32,
    pub coordination_overhead: f32,
    pub backend_switches: u64,
    pub optimization_score: f32,
}

impl Default for DaaCoordinationMetrics {
    fn default() -> Self {
        Self {
            gpu_utilization: 0.0,
            memory_efficiency: 1.0,
            coordination_overhead: 0.0,
            backend_switches: 0,
            optimization_score: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NetworkBuilder;

    #[tokio::test]
    async fn test_compute_context_creation() {
        let context = ComputeContext::<f32>::cpu_only();
        assert!(!context.is_gpu_available());
        assert_eq!(context.current_backend(), BackendType::Cpu);
    }

    #[tokio::test]
    async fn test_backend_selection() {
        let mut context = ComputeContext::<f32>::cpu_only();

        // Small problems should prefer CPU/SIMD
        let backend = context.select_backend(100);
        assert!(matches!(backend, BackendType::Cpu | BackendType::Simd));

        // Large problems would prefer GPU if available
        let backend = context.select_backend(1000000);
        // Since GPU is not available in test, should fallback to SIMD/CPU
        assert!(matches!(backend, BackendType::Cpu | BackendType::Simd));
    }

    #[tokio::test]
    async fn test_network_forward_pass() {
        let mut context = ComputeContext::<f32>::cpu_only();

        // Create a simple test network
        let network = NetworkBuilder::<f32>::new()
            .input_layer(2)
            .hidden_layer(3)
            .output_layer(1)
            .build();

        let inputs = vec![0.5f32, 0.7f32];

        // Debug network structure
        println!("Network structure:");
        println!("  Layers: {}", network.layers.len());
        for (i, layer) in network.layers.iter().enumerate() {
            println!("  Layer {}: {} neurons", i, layer.neurons.len());

            // Debug first neuron in each layer
            if let Some(neuron) = layer.neurons.first() {
                println!(
                    "    First neuron has {} connections, is_bias: {}",
                    neuron.connections.len(),
                    neuron.is_bias
                );
            }
        }

        println!("Starting forward pass with {} inputs", inputs.len());
        let result = context.compute_network_forward(&network, &inputs).await;

        match &result {
            Ok(outputs) => println!("Forward pass succeeded with {} outputs", outputs.len()),
            Err(e) => println!("Forward pass failed: {:?}", e),
        }

        assert!(result.is_ok(), "Forward pass failed");

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1, "Output should have 1 value");
    }

    #[tokio::test]
    async fn test_performance_tracking() {
        let context = ComputeContext::<f32>::cpu_only();
        let stats = context.get_performance_stats();

        assert_eq!(stats.current_backend, BackendType::Cpu);
        assert!(!stats.gpu_available);
        assert_eq!(stats.cache_size, 0);
    }
}
