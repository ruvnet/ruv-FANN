use crate::neuron::apply_activation;
use crate::{ActivationFunction, Neuron};
use num_traits::Float;
use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a layer of neurons in the neural network
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Layer<T: Float> {
    /// The neurons in this layer
    pub neurons: Vec<Neuron<T>>,
}

impl<T: Float> Layer<T> {
    /// Creates a new layer with the specified number of neurons
    ///
    /// # Arguments
    /// * `num_neurons` - Number of neurons in the layer
    /// * `activation_function` - Activation function for all neurons
    /// * `activation_steepness` - Steepness parameter for the activation function
    ///
    /// # Example
    /// ```
    /// use ruv_fann::{Layer, ActivationFunction};
    ///
    /// let layer = Layer::<f32>::new(3, ActivationFunction::Sigmoid, 1.0);
    /// assert_eq!(layer.neurons.len(), 3);
    /// ```
    pub fn new(
        num_neurons: usize,
        activation_function: ActivationFunction,
        activation_steepness: T,
    ) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| Neuron::new(activation_function, activation_steepness))
            .collect();

        Layer { neurons }
    }

    /// Creates a new layer with a bias neuron
    ///
    /// # Arguments
    /// * `num_neurons` - Number of regular neurons (bias will be added)
    /// * `activation_function` - Activation function for regular neurons
    /// * `activation_steepness` - Steepness parameter for the activation function
    pub fn with_bias(
        num_neurons: usize,
        activation_function: ActivationFunction,
        activation_steepness: T,
    ) -> Self {
        let mut neurons = Vec::with_capacity(num_neurons + 1);

        // Add regular neurons
        for _ in 0..num_neurons {
            neurons.push(Neuron::new(activation_function, activation_steepness));
        }

        // Add bias neuron
        neurons.push(Neuron::new_bias());

        Layer { neurons }
    }

    /// Returns the number of neurons in the layer (including bias if present)
    pub fn size(&self) -> usize {
        self.neurons.len()
    }

    /// Returns the number of regular neurons (excluding bias)
    pub fn num_regular_neurons(&self) -> usize {
        if self.has_bias() {
            self.neurons.len() - 1
        } else {
            self.neurons.len()
        }
    }

    /// Checks if the layer has a bias neuron
    pub fn has_bias(&self) -> bool {
        self.neurons.last().map(|n| n.is_bias).unwrap_or(false)
    }

    /// Gets a reference to the bias neuron if it exists
    pub fn bias_neuron(&self) -> Option<&Neuron<T>> {
        if self.has_bias() {
            self.neurons.last()
        } else {
            None
        }
    }

    /// Gets a mutable reference to the bias neuron if it exists
    pub fn bias_neuron_mut(&mut self) -> Option<&mut Neuron<T>> {
        if self.has_bias() {
            self.neurons.last_mut()
        } else {
            None
        }
    }

    /// Resets all neurons in the layer
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
    }

    /// Sets the activation function for all neurons in the layer
    /// (except bias neurons)
    pub fn set_activation_function(&mut self, activation_function: ActivationFunction) {
        for neuron in &mut self.neurons {
            if !neuron.is_bias {
                neuron.activation_function = activation_function;
            }
        }
    }

    /// Sets the activation steepness for all neurons in the layer
    /// (except bias neurons)
    pub fn set_activation_steepness(&mut self, steepness: T) {
        for neuron in &mut self.neurons {
            if !neuron.is_bias {
                neuron.activation_steepness = steepness;
            }
        }
    }

    /// Connects all neurons in this layer to all neurons in the next layer
    /// with random weights
    pub fn connect_to(&self, next_layer: &mut Layer<T>, connection_rate: T) {
        let one = T::one();
        let should_connect = connection_rate >= one;
        let mut rng = rand::rng();

        // For each neuron in the next layer (except bias)
        let next_layer_size = next_layer.num_regular_neurons();
        for i in 0..next_layer_size {
            let next_neuron = &mut next_layer.neurons[i];

            // Connect from each neuron in this layer
            for (j, _) in self.neurons.iter().enumerate() {
                let random_val = T::from(rng.random::<f64>()).unwrap();
                if should_connect || random_val < connection_rate {
                    // Random weight between -0.1 and 0.1
                    let weight_val: f64 = rng.random::<f64>() * 0.2 - 0.1;
                    let weight = T::from(weight_val).unwrap();
                    next_neuron.add_connection(j, weight);
                }
            }
        }
    }

    /// Gets the output values of all neurons in the layer
    pub fn get_outputs(&self) -> Vec<T> {
        self.neurons.iter().map(|n| n.value).collect()
    }

    /// Sets the values of neurons in the layer (used for input layer)
    pub fn set_inputs(&mut self, inputs: &[T]) -> Result<(), &'static str> {
        let regular_neurons = self.num_regular_neurons();
        if inputs.len() != regular_neurons {
            return Err("Input size does not match layer size");
        }

        for (i, &input) in inputs.iter().enumerate() {
            self.neurons[i].set_value(input);
        }

        Ok(())
    }

    /// Calculates outputs for all neurons in the layer based on previous layer outputs
    pub fn calculate(&mut self, prev_outputs: &[T]) {
        // Resolve the activation function once per layer when all non-bias
        // neurons share it (always true for builder-made layers) so each
        // specialized loop below folds the per-neuron dispatch away.
        let mut funcs = self
            .neurons
            .iter()
            .filter(|n| !n.is_bias)
            .map(|n| n.activation_function);
        let first = funcs.next();
        match first.filter(|&f| funcs.all(|g| g == f)) {
            Some(func) => calculate_uniform(&mut self.neurons, prev_outputs, func),
            None => {
                for neuron in &mut self.neurons {
                    if !neuron.is_bias {
                        neuron.calculate(prev_outputs);
                    }
                }
            }
        }
    }
}

/// Forward pass for a layer whose non-bias neurons share one activation
/// function: dispatch on the variant once, then run a loop monomorphized for
/// that variant's closure.
fn calculate_uniform<T: Float>(neurons: &mut [Neuron<T>], prev: &[T], func: ActivationFunction) {
    use ActivationFunction as AF;
    match func {
        AF::Linear => calc_loop(neurons, prev, |k, x| apply_activation(AF::Linear, k, x)),
        AF::Sigmoid => calc_loop(neurons, prev, |k, x| apply_activation(AF::Sigmoid, k, x)),
        AF::ReLU => calc_loop(neurons, prev, |k, x| apply_activation(AF::ReLU, k, x)),
        AF::ReLULeaky => calc_loop(neurons, prev, |k, x| apply_activation(AF::ReLULeaky, k, x)),
        AF::Tanh | AF::SigmoidSymmetric => {
            calc_loop(neurons, prev, |k, x| apply_activation(AF::Tanh, k, x))
        }
        AF::Gaussian => calc_loop(neurons, prev, |k, x| apply_activation(AF::Gaussian, k, x)),
        other => calc_loop(neurons, prev, move |k, x| apply_activation(other, k, x)),
    }
}

#[inline]
fn calc_loop<T: Float>(neurons: &mut [Neuron<T>], prev: &[T], f: impl Fn(T, T) -> T) {
    for neuron in neurons.iter_mut() {
        if !neuron.is_bias {
            let sum = neuron.weighted_sum(prev);
            neuron.sum = sum;
            neuron.value = f(neuron.activation_steepness, sum);
        }
    }
}

impl<T: Float> PartialEq for Layer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.neurons == other.neurons
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = Layer::<f32>::new(3, ActivationFunction::Sigmoid, 1.0);
        assert_eq!(layer.neurons.len(), 3);
        assert!(!layer.has_bias());

        for neuron in &layer.neurons {
            assert_eq!(neuron.activation_function, ActivationFunction::Sigmoid);
            assert_eq!(neuron.activation_steepness, 1.0);
        }
    }

    #[test]
    fn test_layer_with_bias() {
        let layer = Layer::<f32>::with_bias(3, ActivationFunction::ReLU, 1.0);
        assert_eq!(layer.neurons.len(), 4);
        assert_eq!(layer.num_regular_neurons(), 3);
        assert!(layer.has_bias());

        // Check bias neuron
        let bias = layer.bias_neuron().unwrap();
        assert!(bias.is_bias);
        assert_eq!(bias.value, 1.0);
    }

    #[test]
    fn test_set_activation_function() {
        let mut layer = Layer::<f32>::with_bias(2, ActivationFunction::Sigmoid, 1.0);
        layer.set_activation_function(ActivationFunction::ReLU);

        // Regular neurons should have new activation function
        assert_eq!(
            layer.neurons[0].activation_function,
            ActivationFunction::ReLU
        );
        assert_eq!(
            layer.neurons[1].activation_function,
            ActivationFunction::ReLU
        );

        // Bias neuron should remain unchanged
        assert_eq!(
            layer.neurons[2].activation_function,
            ActivationFunction::Linear
        );
    }

    #[test]
    fn test_set_inputs() {
        let mut layer = Layer::<f32>::with_bias(3, ActivationFunction::Linear, 1.0);
        let inputs = vec![1.0, 2.0, 3.0];

        assert!(layer.set_inputs(&inputs).is_ok());
        assert_eq!(layer.neurons[0].value, 1.0);
        assert_eq!(layer.neurons[1].value, 2.0);
        assert_eq!(layer.neurons[2].value, 3.0);
        assert_eq!(layer.neurons[3].value, 1.0); // bias
    }

    #[test]
    fn test_set_inputs_wrong_size() {
        let mut layer = Layer::<f32>::new(3, ActivationFunction::Linear, 1.0);
        let inputs = vec![1.0, 2.0]; // Too few

        assert!(layer.set_inputs(&inputs).is_err());
    }

    #[test]
    fn test_get_outputs() {
        let mut layer = Layer::<f32>::with_bias(2, ActivationFunction::Linear, 1.0);
        layer.neurons[0].value = 0.5;
        layer.neurons[1].value = 0.7;

        let outputs = layer.get_outputs();
        assert_eq!(outputs, vec![0.5, 0.7, 1.0]); // Including bias
    }

    #[test]
    fn test_connect_layers() {
        let layer1 = Layer::<f32>::with_bias(2, ActivationFunction::Sigmoid, 1.0);
        let mut layer2 = Layer::<f32>::new(2, ActivationFunction::Sigmoid, 1.0);

        layer1.connect_to(&mut layer2, 1.0);

        // Each neuron in layer2 should have 3 connections (2 regular + 1 bias from layer1)
        assert_eq!(layer2.neurons[0].connections.len(), 3);
        assert_eq!(layer2.neurons[1].connections.len(), 3);
    }
}
