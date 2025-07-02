//! GPU-accelerated network operations

use crate::network::Network;
use num_traits::{Float, FromPrimitive};

#[cfg(feature = "gpu")]
use crate::webgpu::{ComputeBackend, ComputeResult};
#[cfg(feature = "gpu")]
use crate::webgpu::backend::{MatrixDims, ActivationFunction as GpuActivationFunction};

#[cfg(feature = "gpu")]
impl<T: Float + Send + Sync + Default + bytemuck::Pod + FromPrimitive> Network<T> {
    /// Run the network with GPU acceleration
    pub async fn run_gpu(&mut self, inputs: &[T]) -> ComputeResult<Vec<T>> {
        // Check if we have a backend selector
        let backend_selector = self.backend_selector.as_ref()
            .ok_or(crate::webgpu::error::ComputeError::backend_error("Backend selector not initialized"))?;
        
        // Get the backend based on the selected type
        let backend_type = backend_selector.select_backend::<T>(inputs.len());
        
        // For now, we'll directly use the GPU backend if available
        let gpu_backend = match backend_type {
            crate::webgpu::backend::BackendType::WebGPU => {
                backend_selector.get_gpu_backend()?
            }
            _ => {
                return Err(crate::webgpu::error::ComputeError::backend_error(
                    "GPU backend not available, falling back to CPU would require sync implementation"
                ));
            }
        };
        
        // Convert network to GPU-friendly format
        let mut current_activations = inputs.to_vec();
        
        // Process each layer
        for layer_idx in 1..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let prev_layer = &self.layers[layer_idx - 1];
            
            // Extract weights as a flat matrix
            let mut weights = Vec::new();
            let input_size = prev_layer.neurons.len();
            let output_size = layer.neurons.iter().filter(|n| !n.is_bias).count();
            
            // Build weight matrix (row-major order)
            for neuron in layer.neurons.iter().filter(|n| !n.is_bias) {
                for conn in &neuron.connections {
                    weights.push(conn.weight);
                }
            }
            
            // Matrix dimensions
            let dims = MatrixDims::new(output_size, input_size);
            
            // Perform matrix-vector multiplication on GPU
            let mut layer_outputs = gpu_backend.matrix_vector_multiply(&weights, &current_activations, dims).await?;
            
            // Add bias terms
            let mut bias_idx = 0;
            for neuron in layer.neurons.iter().filter(|n| !n.is_bias) {
                if bias_idx < layer_outputs.len() {
                    // Find bias connection (last connection typically)
                    if let Some(bias_conn) = neuron.connections.last() {
                        layer_outputs[bias_idx] = layer_outputs[bias_idx] + bias_conn.weight;
                    }
                    bias_idx += 1;
                }
            }
            
            // Apply activation function (use the first non-bias neuron's activation)
            let activation = if let Some(neuron) = layer.neurons.iter().find(|n| !n.is_bias) {
                convert_activation_function(neuron.activation_function)
            } else {
                GpuActivationFunction::Linear
            };
            layer_outputs = gpu_backend.activation_function(&layer_outputs, activation).await?;
            
            // Update for next layer
            current_activations = layer_outputs;
        }
        
        Ok(current_activations)
    }
    
    /// Train one epoch with GPU acceleration
    pub async fn train_epoch_gpu(&mut self, data: &crate::training::TrainingData<T>, learning_rate: T) -> ComputeResult<T> 
    where
        T: std::iter::Sum + std::ops::Div<Output = T> + FromPrimitive,
    {
        // Check if we have a backend selector
        let backend_selector = self.backend_selector.as_ref()
            .ok_or(crate::webgpu::error::ComputeError::backend_error("Backend selector not initialized"))?;
        
        let mut total_error = T::zero();
        let num_samples = T::from_usize(data.inputs.len()).unwrap_or_else(T::one);
        
        for (input, target) in data.inputs.iter().zip(data.outputs.iter()) {
            // Forward pass
            let output = self.run_gpu(input).await?;
            
            // Calculate error
            let errors: Vec<T> = output.iter()
                .zip(target.iter())
                .map(|(o, t)| *t - *o)
                .collect();
            
            // Sum squared errors
            let sample_error: T = errors.iter()
                .map(|e| *e * *e)
                .sum();
            total_error = total_error + sample_error;
            
            // Backward pass (simplified - full GPU implementation needed)
            // For now, fall back to CPU for weight updates
            // TODO: Implement full GPU backward propagation
            
            // Update weights on CPU for now
            self.update_weights_cpu(&errors, input, learning_rate);
        }
        
        Ok(total_error / num_samples)
    }
    
    /// CPU fallback for weight updates (temporary)
    fn update_weights_cpu(&mut self, errors: &[T], inputs: &[T], learning_rate: T) {
        // This is a simplified weight update - full backpropagation needed
        // Just to make the training loop functional for now
        
        // Set inputs
        if self.layers[0].set_inputs(inputs).is_err() {
            return;
        }
        
        // Forward propagate to get all activations
        for i in 1..self.layers.len() {
            let prev_outputs = self.layers[i - 1].get_outputs();
            self.layers[i].calculate(&prev_outputs);
        }
        
        // Simple gradient descent on output layer (temporary)
        if let Some(output_layer) = self.layers.last_mut() {
            let mut error_idx = 0;
            for neuron in output_layer.neurons.iter_mut().filter(|n| !n.is_bias) {
                if error_idx < errors.len() {
                    // Update weights based on error
                    let error = errors[error_idx];
                    for conn in neuron.connections.iter_mut() {
                        // Simple delta rule: Δw = η * error * input
                        // Simple delta rule: Δw = η * error * input
                        // For now, just use a simple gradient
                        let delta = learning_rate * error;
                        conn.weight = conn.weight + delta;
                    }
                    error_idx += 1;
                }
            }
        }
    }
}

#[cfg(feature = "gpu")]
fn convert_activation_function(fann_activation: crate::ActivationFunction) -> GpuActivationFunction {
    use crate::ActivationFunction as FannActivation;
    
    match fann_activation {
        FannActivation::ReLU => GpuActivationFunction::ReLU,
        FannActivation::Sigmoid => GpuActivationFunction::Sigmoid,
        FannActivation::Tanh => GpuActivationFunction::Tanh,
        FannActivation::Linear => GpuActivationFunction::Linear,
        // FannActivation::LeakyReLU(alpha) => GpuActivationFunction::LeakyReLU(alpha as f32),
        // FannActivation::ELU(alpha) => GpuActivationFunction::ELU(alpha as f32),
        _ => GpuActivationFunction::Linear, // Default fallback
    }
}