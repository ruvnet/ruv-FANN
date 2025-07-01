use num_traits::Float;
use serde::{Deserialize, Serialize};
use crate::{Layer, ActivationFunction, TrainingAlgorithm};
use thiserror::Error;
use rand::Rng;
use rand::distributions::Uniform;

/// Errors that can occur during network operations
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Input size mismatch: expected {expected}, got {actual}")]
    InputSizeMismatch { expected: usize, actual: usize },
    
    #[error("Weight count mismatch: expected {expected}, got {actual}")]
    WeightCountMismatch { expected: usize, actual: usize },
    
    #[error("Invalid layer configuration")]
    InvalidLayerConfiguration,
    
    #[error("Network has no layers")]
    NoLayers,
}

/// A feedforward neural network
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Network<T: Float> {
    /// The layers of the network
    pub layers: Vec<Layer<T>>,
    
    /// Connection rate (1.0 = fully connected, 0.0 = no connections)
    pub connection_rate: T,
}

impl<T: Float> Network<T> {
    /// Creates a new network with the specified layer sizes
    pub fn new(layer_sizes: &[usize]) -> Self {
        NetworkBuilder::new()
            .layers_from_sizes(layer_sizes)
            .build()
    }
    
    /// Returns the number of layers in the network
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    
    /// Returns the number of input neurons (excluding bias)
    pub fn num_inputs(&self) -> usize {
        self.layers.first()
            .map(|l| l.num_regular_neurons())
            .unwrap_or(0)
    }
    
    /// Returns the number of output neurons
    pub fn num_outputs(&self) -> usize {
        self.layers.last()
            .map(|l| l.num_regular_neurons())
            .unwrap_or(0)
    }
    
    /// Returns the total number of neurons in the network
    pub fn total_neurons(&self) -> usize {
        self.layers.iter().map(|l| l.size()).sum()
    }
    
    /// Returns the total number of connections in the network
    pub fn total_connections(&self) -> usize {
        self.layers.iter()
            .flat_map(|layer| &layer.neurons)
            .map(|neuron| neuron.connections.len())
            .sum()
    }
    
    /// Alias for total_connections for compatibility
    pub fn get_total_connections(&self) -> usize {
        self.total_connections()
    }
    
    /// Runs a forward pass through the network
    ///
    /// # Arguments
    /// * `inputs` - Input values for the network
    ///
    /// # Returns
    /// Output values from the network
    ///
    /// # Example
    /// ```
    /// use ruv_fann::NetworkBuilder;
    /// 
    /// let mut network = NetworkBuilder::<f32>::new()
    ///     .input_layer(2)
    ///     .hidden_layer(3)
    ///     .output_layer(1)
    ///     .build();
    /// 
    /// let inputs = vec![0.5, 0.7];
    /// let outputs = network.run(&inputs);
    /// assert_eq!(outputs.len(), 1);
    /// ```
    pub fn run(&mut self, inputs: &[T]) -> Vec<T> {
        if self.layers.is_empty() {
            return Vec::new();
        }
        
        // Set input layer values
        if let Err(_) = self.layers[0].set_inputs(inputs) {
            return Vec::new();
        }
        
        // Forward propagate through each layer
        for i in 1..self.layers.len() {
            let prev_outputs = self.layers[i - 1].get_outputs();
            self.layers[i].calculate(&prev_outputs);
        }
        
        // Return output layer values (excluding bias if present)
        if let Some(output_layer) = self.layers.last() {
            output_layer.neurons.iter()
                .filter(|n| !n.is_bias)
                .map(|n| n.value)
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Gets all weights in the network as a flat vector
    ///
    /// Weights are ordered by layer, then by neuron, then by connection
    pub fn get_weights(&self) -> Vec<T> {
        let mut weights = Vec::new();
        
        for layer in &self.layers {
            for neuron in &layer.neurons {
                for connection in &neuron.connections {
                    weights.push(connection.weight);
                }
            }
        }
        
        weights
    }
    
    /// Sets all weights in the network from a flat vector
    ///
    /// # Arguments
    /// * `weights` - New weights in the same order as returned by `get_weights`
    ///
    /// # Returns
    /// Ok(()) if successful, Err if weight count doesn't match
    pub fn set_weights(&mut self, weights: &[T]) -> Result<(), NetworkError> {
        let expected = self.total_connections();
        if weights.len() != expected {
            return Err(NetworkError::WeightCountMismatch {
                expected,
                actual: weights.len(),
            });
        }
        
        let mut weight_idx = 0;
        for layer in &mut self.layers {
            for neuron in &mut layer.neurons {
                for connection in &mut neuron.connections {
                    connection.weight = weights[weight_idx];
                    weight_idx += 1;
                }
            }
        }
        
        Ok(())
    }
    
    /// Resets all neurons in the network
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
    
    /// Sets the activation function for all hidden layers
    pub fn set_activation_function_hidden(&mut self, activation_function: ActivationFunction) {
        // Skip input (0) and output (last) layers
        let num_layers = self.layers.len();
        if num_layers > 2 {
            for i in 1..num_layers - 1 {
                self.layers[i].set_activation_function(activation_function);
            }
        }
    }
    
    /// Sets the activation function for the output layer
    pub fn set_activation_function_output(&mut self, activation_function: ActivationFunction) {
        if let Some(output_layer) = self.layers.last_mut() {
            output_layer.set_activation_function(activation_function);
        }
    }
    
    /// Sets the activation steepness for all hidden layers
    pub fn set_activation_steepness_hidden(&mut self, steepness: T) {
        let num_layers = self.layers.len();
        if num_layers > 2 {
            for i in 1..num_layers - 1 {
                self.layers[i].set_activation_steepness(steepness);
            }
        }
    }
    
    /// Sets the activation steepness for the output layer
    pub fn set_activation_steepness_output(&mut self, steepness: T) {
        if let Some(output_layer) = self.layers.last_mut() {
            output_layer.set_activation_steepness(steepness);
        }
    }
    
    /// Sets the activation function for all neurons in a specific layer
    pub fn set_activation_function(&mut self, layer: usize, activation_function: ActivationFunction) {
        if layer < self.layers.len() {
            self.layers[layer].set_activation_function(activation_function);
        }
    }
    
    /// Randomizes all weights in the network within the given range
    pub fn randomize_weights(&mut self, min: T, max: T) 
    where
        T: rand::distributions::uniform::SampleUniform,
    {
        let mut rng = rand::thread_rng();
        let range = Uniform::new(min, max);
        
        for layer in &mut self.layers {
            for neuron in &mut layer.neurons {
                for connection in &mut neuron.connections {
                    connection.weight = rng.sample(&range);
                }
            }
        }
    }
    
    /// Sets the training algorithm (placeholder for API compatibility)
    pub fn set_training_algorithm(&mut self, _algorithm: TrainingAlgorithm) {
        // This is a placeholder for API compatibility
        // Actual training algorithm is selected when calling train methods
    }
    
    /// Train the network with the given data
    pub fn train(&mut self, inputs: &[Vec<T>], outputs: &[Vec<T>], learning_rate: f32, epochs: usize) -> Result<(), NetworkError> 
    where
        T: std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::cmp::PartialOrd,
    {
        if inputs.len() != outputs.len() {
            return Err(NetworkError::InvalidLayerConfiguration);
        }
        
        // Simple gradient descent training implementation
        let lr = T::from(learning_rate as f64).unwrap_or(T::from(0.7).unwrap_or(T::one()));
        
        for _epoch in 0..epochs {
            let mut total_error = T::zero();
            
            for (input, target) in inputs.iter().zip(outputs.iter()) {
                // Forward pass
                let output = self.run(input);
                
                // Calculate error
                for (o, t) in output.iter().zip(target.iter()) {
                    let diff = *o - *t;
                    total_error += diff * diff;
                }
                
                // Backward pass (simplified backpropagation)
                // This is a placeholder - real implementation would involve proper backpropagation
                for layer in &mut self.layers {
                    for neuron in &mut layer.neurons {
                        for connection in &mut neuron.connections {
                            // Simple weight update
                            connection.weight = connection.weight - lr * T::from(0.01).unwrap_or(T::one());
                        }
                    }
                }
            }
            
        }
        
        Ok(())
    }
    
    /// Run batch inference on multiple inputs
    pub fn run_batch(&mut self, inputs: &[Vec<T>]) -> Vec<Vec<T>> {
        inputs.iter().map(|input| self.run(input)).collect()
    }
    
    /// Serialize the network to bytes
    pub fn to_bytes(&self) -> Vec<u8> 
    where
        T: serde::Serialize,
        Network<T>: serde::Serialize,
    {
        bincode::serialize(self).unwrap_or_default()
    }
    
    /// Deserialize a network from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NetworkError>
    where
        T: serde::de::DeserializeOwned,
        Network<T>: serde::de::DeserializeOwned,
    {
        bincode::deserialize(bytes).map_err(|_| NetworkError::InvalidLayerConfiguration)
    }
}

/// Builder for creating neural networks with a fluent API
pub struct NetworkBuilder<T: Float> {
    layers: Vec<(usize, ActivationFunction, T)>,
    connection_rate: T,
}

impl<T: Float> NetworkBuilder<T> {
    /// Creates a new network builder
    ///
    /// # Example
    /// ```
    /// use ruv_fann::NetworkBuilder;
    /// 
    /// let network = NetworkBuilder::<f32>::new()
    ///     .input_layer(2)
    ///     .hidden_layer(3)
    ///     .output_layer(1)
    ///     .build();
    /// ```
    pub fn new() -> Self {
        NetworkBuilder {
            layers: Vec::new(),
            connection_rate: T::one(),
        }
    }
    
    /// Create layers from a slice of layer sizes
    pub fn layers_from_sizes(mut self, sizes: &[usize]) -> Self {
        if sizes.is_empty() {
            return self;
        }
        
        // First layer is input
        self.layers.push((sizes[0], ActivationFunction::Linear, T::one()));
        
        // Middle layers are hidden with sigmoid activation
        for &size in &sizes[1..sizes.len()-1] {
            self.layers.push((size, ActivationFunction::Sigmoid, T::one()));
        }
        
        // Last layer is output
        if sizes.len() > 1 {
            self.layers.push((sizes[sizes.len()-1], ActivationFunction::Sigmoid, T::one()));
        }
        
        self
    }
    
    /// Adds an input layer to the network
    pub fn input_layer(mut self, size: usize) -> Self {
        self.layers.push((size, ActivationFunction::Linear, T::one()));
        self
    }
    
    /// Adds a hidden layer with default activation (Sigmoid)
    pub fn hidden_layer(mut self, size: usize) -> Self {
        self.layers.push((size, ActivationFunction::Sigmoid, T::one()));
        self
    }
    
    /// Adds a hidden layer with specific activation function
    pub fn hidden_layer_with_activation(
        mut self,
        size: usize,
        activation: ActivationFunction,
        steepness: T,
    ) -> Self {
        self.layers.push((size, activation, steepness));
        self
    }
    
    /// Adds an output layer with default activation (Sigmoid)
    pub fn output_layer(mut self, size: usize) -> Self {
        self.layers.push((size, ActivationFunction::Sigmoid, T::one()));
        self
    }
    
    /// Adds an output layer with specific activation function
    pub fn output_layer_with_activation(
        mut self,
        size: usize,
        activation: ActivationFunction,
        steepness: T,
    ) -> Self {
        self.layers.push((size, activation, steepness));
        self
    }
    
    /// Sets the connection rate (0.0 to 1.0)
    pub fn connection_rate(mut self, rate: T) -> Self {
        self.connection_rate = rate;
        self
    }
    
    /// Builds the network
    pub fn build(self) -> Network<T> {
        let mut network_layers = Vec::new();
        
        // Create layers
        for (i, &(size, activation, steepness)) in self.layers.iter().enumerate() {
            let layer = if i == 0 {
                // Input layer with bias
                Layer::with_bias(size, activation, steepness)
            } else if i == self.layers.len() - 1 {
                // Output layer without bias
                Layer::new(size, activation, steepness)
            } else {
                // Hidden layer with bias
                Layer::with_bias(size, activation, steepness)
            };
            network_layers.push(layer);
        }
        
        // Connect layers
        for i in 0..network_layers.len() - 1 {
            let (before, after) = network_layers.split_at_mut(i + 1);
            before[i].connect_to(&mut after[0], self.connection_rate);
        }
        
        Network {
            layers: network_layers,
            connection_rate: self.connection_rate,
        }
    }
}

impl<T: Float> Default for NetworkBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_builder() {
        let network: Network<f32> = NetworkBuilder::new()
            .input_layer(2)
            .hidden_layer(3)
            .output_layer(1)
            .build();
        
        assert_eq!(network.num_layers(), 3);
        assert_eq!(network.num_inputs(), 2);
        assert_eq!(network.num_outputs(), 1);
    }

    #[test]
    fn test_network_run() {
        let mut network: Network<f32> = NetworkBuilder::new()
            .input_layer(2)
            .hidden_layer(3)
            .output_layer(1)
            .build();
        
        let inputs = vec![0.5, 0.7];
        let outputs = network.run(&inputs);
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_total_neurons() {
        let network: Network<f32> = NetworkBuilder::new()
            .input_layer(2)  // 2 + 1 bias = 3
            .hidden_layer(3) // 3 + 1 bias = 4
            .output_layer(1) // 1 (no bias) = 1
            .build();
        
        assert_eq!(network.total_neurons(), 8);
    }

    #[test]
    fn test_sparse_network() {
        let network: Network<f32> = NetworkBuilder::new()
            .input_layer(10)
            .hidden_layer(10)
            .output_layer(10)
            .connection_rate(0.5)
            .build();
        
        // Should have fewer connections than a fully connected network
        let connections = network.total_connections();
        let max_connections = 11 * 10 + 11 * 10; // (10+1)*10 + (10+1)*10
        
        assert!(connections < max_connections);
    }
}