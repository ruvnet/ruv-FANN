use crate::{ActivationFunction, Layer, TrainingAlgorithm};
use num_traits::Float;
use rand::distr::Uniform;
use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use thiserror::Error;

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

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Data size mismatch: expected {expected} samples for inputs and outputs, got {actual}")]
    DataSizeMismatch { expected: usize, actual: usize },
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
        NetworkBuilder::new().layers_from_sizes(layer_sizes).build()
    }

    /// Returns the number of layers in the network
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the number of input neurons (excluding bias)
    pub fn num_inputs(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.num_regular_neurons())
            .unwrap_or(0)
    }

    /// Returns the number of output neurons
    pub fn num_outputs(&self) -> usize {
        self.layers
            .last()
            .map(|l| l.num_regular_neurons())
            .unwrap_or(0)
    }

    /// Returns the total number of neurons in the network
    pub fn total_neurons(&self) -> usize {
        self.layers.iter().map(|l| l.size()).sum()
    }

    /// Returns the total number of connections in the network
    pub fn total_connections(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|layer| &layer.neurons)
            .map(|neuron| neuron.connections.len())
            .sum()
    }

    /// Alias for total_connections for compatibility
    pub fn get_total_connections(&self) -> usize {
        self.total_connections()
    }

    /// Runs a forward pass through the network.
    ///
    /// # Arguments
    /// * `inputs` - Input values for the network
    ///
    /// # Returns
    /// `Ok(Vec<T>)` with the output values, or a `NetworkError` on failure.
    ///
    /// # Errors
    /// * [`NetworkError::NoLayers`] -- the network has no layers.
    /// * [`NetworkError::InputSizeMismatch`] -- `inputs.len()` does not match the input layer.
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
    /// let outputs = network.run(&inputs).unwrap();
    /// assert_eq!(outputs.len(), 1);
    /// ```
    pub fn run(&mut self, inputs: &[T]) -> Result<Vec<T>, NetworkError> {
        if self.layers.is_empty() {
            return Err(NetworkError::NoLayers);
        }

        // Set input layer values
        let expected = self.layers[0].num_regular_neurons();
        if self.layers[0].set_inputs(inputs).is_err() {
            return Err(NetworkError::InputSizeMismatch {
                expected,
                actual: inputs.len(),
            });
        }

        // Forward propagate through each layer, reusing one scratch buffer
        // for previous-layer outputs instead of allocating a Vec per layer.
        let max_neurons = self.layers.iter().map(|l| l.neurons.len()).max().unwrap_or(0);
        let mut prev_outputs: Vec<T> = Vec::with_capacity(max_neurons);
        for i in 1..self.layers.len() {
            prev_outputs.clear();
            prev_outputs.extend(self.layers[i - 1].neurons.iter().map(|n| n.value));
            self.layers[i].calculate(&prev_outputs);
        }

        // Return output layer values (excluding bias if present)
        Ok(self
            .layers
            .last()
            .map(|output_layer| {
                output_layer
                    .neurons
                    .iter()
                    .filter(|n| !n.is_bias)
                    .map(|n| n.value)
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Runs a forward pass without returning errors.
    ///
    /// This is a convenience wrapper around [`run`](Self::run) that silently
    /// returns an empty `Vec` on failure, preserving the pre-1.0 behaviour.
    pub fn run_unchecked(&mut self, inputs: &[T]) -> Vec<T> {
        self.run(inputs).unwrap_or_default()
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
    pub fn set_activation_function(
        &mut self,
        layer: usize,
        activation_function: ActivationFunction,
    ) {
        if layer < self.layers.len() {
            self.layers[layer].set_activation_function(activation_function);
        }
    }

    /// Randomizes all weights in the network within the given range
    pub fn randomize_weights(&mut self, min: T, max: T)
    where
        T: rand::distr::uniform::SampleUniform,
    {
        let mut rng = rand::rng();
        let range = Uniform::new(min, max).expect("randomize_weights: invalid weight range");

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

    /// Train the network with the given data using backpropagation
    pub fn train(
        &mut self,
        inputs: &[Vec<T>],
        outputs: &[Vec<T>],
        learning_rate: f32,
        epochs: usize,
    ) -> Result<(), NetworkError>
    where
        T: std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::cmp::PartialOrd,
    {
        if inputs.len() != outputs.len() {
            return Err(NetworkError::DataSizeMismatch {
                expected: inputs.len(),
                actual: outputs.len(),
            });
        }

        let lr = T::from(learning_rate as f64).unwrap_or(T::from(0.1).unwrap_or(T::one()));

        for _epoch in 0..epochs {
            for (input, target) in inputs.iter().zip(outputs.iter()) {
                // Forward pass - store all layer outputs for backpropagation
                let layer_outputs = self.forward_pass_with_storage(input);

                // Backward pass - calculate gradients and update weights
                self.backward_pass(&layer_outputs, target, lr);
            }
        }

        Ok(())
    }

    /// Forward pass that stores all layer outputs for backpropagation
    fn forward_pass_with_storage(&mut self, input: &[T]) -> Vec<Vec<T>> {
        let mut layer_outputs = Vec::with_capacity(self.layers.len());

        // Set input layer
        if !self.layers.is_empty() {
            let _ = self.layers[0].set_inputs(input);
            layer_outputs.push(self.layers[0].get_outputs());
        }

        // Forward propagate through each layer
        for i in 1..self.layers.len() {
            let prev_outputs = layer_outputs[i - 1].clone();
            self.layers[i].calculate(&prev_outputs);
            layer_outputs.push(self.layers[i].get_outputs());
        }

        layer_outputs
    }

    /// Backward pass - calculate gradients and update weights
    fn backward_pass(&mut self, layer_outputs: &[Vec<T>], target: &[T], learning_rate: T) {
        if self.layers.is_empty() {
            return;
        }

        let num_layers = self.layers.len();
        let mut layer_errors = vec![Vec::new(); num_layers];

        // Calculate output layer errors
        if let Some(output_layer) = self.layers.last() {
            let output_idx = num_layers - 1;
            let outputs = &layer_outputs[output_idx];

            // Calculate output errors (target - output)
            for (i, neuron) in output_layer.neurons.iter().enumerate() {
                if !neuron.is_bias && i < target.len() && i < outputs.len() {
                    let error = target[i] - outputs[i];
                    let delta = error * neuron.activation_derivative();
                    layer_errors[output_idx].push(delta);
                } else {
                    layer_errors[output_idx].push(T::zero());
                }
            }
        }

        // Calculate hidden layer errors (backpropagate)
        for layer_idx in (1..num_layers - 1).rev() {
            let current_layer = &self.layers[layer_idx];
            let next_layer = &self.layers[layer_idx + 1];
            let next_errors = layer_errors[layer_idx + 1].clone(); // Clone to avoid borrowing issues

            let mut current_errors = Vec::new();

            for (i, neuron) in current_layer.neurons.iter().enumerate() {
                if neuron.is_bias {
                    current_errors.push(T::zero());
                    continue;
                }

                let mut error_sum = T::zero();

                // Sum errors from next layer weighted by connections
                for (j, next_neuron) in next_layer.neurons.iter().enumerate() {
                    if !next_neuron.is_bias && j < next_errors.len() {
                        // Find connection from current neuron to next neuron
                        for connection in &next_neuron.connections {
                            if connection.from_neuron == i {
                                error_sum = error_sum + next_errors[j] * connection.weight;
                                break;
                            }
                        }
                    }
                }

                let delta = error_sum * neuron.activation_derivative();
                current_errors.push(delta);
            }

            layer_errors[layer_idx] = current_errors;
        }

        // Update weights
        for layer_idx in 1..num_layers {
            let prev_outputs = if layer_idx == 1 {
                // For first hidden layer, use input layer outputs
                &layer_outputs[0]
            } else {
                &layer_outputs[layer_idx - 1]
            };

            let current_errors = &layer_errors[layer_idx];
            let current_layer = &mut self.layers[layer_idx];

            for (neuron_idx, neuron) in current_layer.neurons.iter_mut().enumerate() {
                if neuron.is_bias || neuron_idx >= current_errors.len() {
                    continue;
                }

                let error = current_errors[neuron_idx];

                // Update weights for this neuron
                for connection in &mut neuron.connections {
                    if connection.from_neuron < prev_outputs.len() {
                        let input_value = prev_outputs[connection.from_neuron];
                        let weight_delta = learning_rate * error * input_value;
                        connection.weight = connection.weight + weight_delta;
                    }
                }
            }
        }
    }

    /// Run batch inference on multiple inputs.
    ///
    /// Uses [`run_unchecked`](Self::run_unchecked) internally so that an
    /// invalid single input does not abort the entire batch.
    pub fn run_batch(&mut self, inputs: &[Vec<T>]) -> Vec<Vec<T>> {
        inputs
            .iter()
            .map(|input| self.run_unchecked(input))
            .collect()
    }

    /// Serialize the network to bytes
    #[cfg(all(feature = "binary", feature = "serde"))]
    pub fn to_bytes(&self) -> Result<Vec<u8>, NetworkError>
    where
        T: serde::Serialize,
        Network<T>: serde::Serialize,
    {
        bincode::serialize(self).map_err(|e| {
            NetworkError::InvalidLayerConfiguration
        })
    }

    #[cfg(feature = "binary")]
    #[cfg(not(feature = "serde"))]
    pub fn to_bytes(&self) -> Vec<u8> {
        // Fallback implementation when serde is not available
        Vec::new()
    }

    /// Default maximum size for deserialization (256 MB)
    const FROM_BYTES_MAX: u64 = 256 * 1024 * 1024;

    /// Deserialize a network from bytes with bounded deserialization and
    /// post-deserialization validation.
    #[cfg(all(feature = "binary", feature = "serde"))]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NetworkError>
    where
        T: serde::de::DeserializeOwned,
        Network<T>: serde::de::DeserializeOwned,
    {
        use bincode::Options;

        if bytes.len() as u64 > Self::FROM_BYTES_MAX {
            return Err(NetworkError::InvalidLayerConfiguration);
        }

        // Use options compatible with bincode::serialize() defaults
        let options = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_little_endian()
            .allow_trailing_bytes()
            .with_limit(Self::FROM_BYTES_MAX);
        let network: Self = options
            .deserialize(bytes)
            .map_err(|_| NetworkError::InvalidLayerConfiguration)?;

        // Validate network structure post-deserialize
        if network.layers.is_empty() {
            return Err(NetworkError::NoLayers);
        }
        for layer in &network.layers {
            if layer.size() == 0 {
                return Err(NetworkError::InvalidLayerConfiguration);
            }
            // Validate connections reference valid neuron indices
            for neuron in &layer.neurons {
                for conn in &neuron.connections {
                    if conn.from_neuron >= layer.size() + 1024 {
                        // Allow some headroom but catch clearly invalid indices
                        return Err(NetworkError::InvalidLayerConfiguration);
                    }
                }
            }
        }

        Ok(network)
    }

    #[cfg(feature = "binary")]
    #[cfg(not(feature = "serde"))]
    pub fn from_bytes(_bytes: &[u8]) -> Result<Self, NetworkError> {
        // Fallback implementation when serde is not available
        Err(NetworkError::InvalidLayerConfiguration)
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
        self.layers
            .push((sizes[0], ActivationFunction::Linear, T::one()));

        // Middle layers are hidden with sigmoid activation
        for &size in &sizes[1..sizes.len() - 1] {
            self.layers
                .push((size, ActivationFunction::Sigmoid, T::one()));
        }

        // Last layer is output
        if sizes.len() > 1 {
            self.layers.push((
                sizes[sizes.len() - 1],
                ActivationFunction::Sigmoid,
                T::one(),
            ));
        }

        self
    }

    /// Adds an input layer to the network
    pub fn input_layer(mut self, size: usize) -> Self {
        self.layers
            .push((size, ActivationFunction::Linear, T::one()));
        self
    }

    /// Adds a hidden layer with default activation (Sigmoid)
    pub fn hidden_layer(mut self, size: usize) -> Self {
        self.layers
            .push((size, ActivationFunction::Sigmoid, T::one()));
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
        self.layers
            .push((size, ActivationFunction::Sigmoid, T::one()));
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
        let outputs = network.run(&inputs).unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_total_neurons() {
        let network: Network<f32> = NetworkBuilder::new()
            .input_layer(2) // 2 + 1 bias = 3
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
