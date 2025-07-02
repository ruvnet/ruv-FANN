use crate::{ActivationFunction, Connection};
use num_traits::Float;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a single neuron in the neural network
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Neuron<T: Float> {
    /// The sum of inputs multiplied by weights
    pub sum: T,

    /// The output value after applying the activation function
    pub value: T,

    /// The steepness parameter for the activation function
    pub activation_steepness: T,

    /// The activation function to use
    pub activation_function: ActivationFunction,

    /// Incoming connections to this neuron
    pub connections: Vec<Connection<T>>,

    /// Whether this is a bias neuron
    pub is_bias: bool,
}

impl<T: Float> Neuron<T> {
    /// Creates a new neuron with the specified activation function and steepness
    ///
    /// # Arguments
    /// * `activation_function` - The activation function to use
    /// * `activation_steepness` - The steepness parameter for the activation function
    ///
    /// # Example
    /// ```
    /// use ruv_fann::{Neuron, ActivationFunction};
    ///
    /// let neuron = Neuron::<f32>::new(ActivationFunction::Sigmoid, 1.0);
    /// assert_eq!(neuron.activation_function, ActivationFunction::Sigmoid);
    /// ```
    pub fn new(activation_function: ActivationFunction, activation_steepness: T) -> Self {
        Neuron {
            sum: T::zero(),
            value: T::zero(),
            activation_steepness,
            activation_function,
            connections: Vec::new(),
            is_bias: false,
        }
    }

    /// Creates a new bias neuron with a constant output value of 1.0
    pub fn new_bias() -> Self {
        let one = T::one();
        Neuron {
            sum: one,
            value: one,
            activation_steepness: one,
            activation_function: ActivationFunction::Linear,
            connections: Vec::new(),
            is_bias: true,
        }
    }

    /// Adds a connection from another neuron to this neuron
    ///
    /// # Arguments
    /// * `from_neuron` - Index of the source neuron
    /// * `weight` - Initial weight of the connection
    pub fn add_connection(&mut self, from_neuron: usize, weight: T) {
        let neuron_index = self.connections.len();
        self.connections
            .push(Connection::new(from_neuron, neuron_index, weight));
    }

    /// Clears all connections
    pub fn clear_connections(&mut self) {
        self.connections.clear();
    }

    /// Resets the neuron's sum and value to zero
    /// (except for bias neurons which maintain value = 1.0)
    pub fn reset(&mut self) {
        if self.is_bias {
            self.sum = T::one();
            self.value = T::one();
        } else {
            self.sum = T::zero();
            self.value = T::zero();
        }
    }

    /// Calculates the neuron's output based on inputs and weights
    ///
    /// # Arguments
    /// * `inputs` - Values from neurons in the previous layer
    pub fn calculate(&mut self, inputs: &[T]) {
        if self.is_bias {
            // Bias neurons always output 1.0
            return;
        }

        // Calculate weighted sum
        self.sum = T::zero();
        for connection in &self.connections {
            if connection.from_neuron < inputs.len() {
                self.sum = self.sum + inputs[connection.from_neuron] * connection.weight;
            }
        }

        // Apply activation function
        self.value = self.apply_activation_function(self.sum);
    }

    /// Sets the neuron's output value directly (used for input neurons)
    pub fn set_value(&mut self, value: T) {
        if !self.is_bias {
            self.value = value;
            self.sum = value;
        }
    }

    /// Gets the weight of a specific connection by index
    pub fn get_connection_weight(&self, index: usize) -> Option<T> {
        self.connections.get(index).map(|c| c.weight)
    }

    /// Sets the weight of a specific connection by index
    pub fn set_connection_weight(&mut self, index: usize, weight: T) -> Result<(), &'static str> {
        if let Some(connection) = self.connections.get_mut(index) {
            connection.set_weight(weight);
            Ok(())
        } else {
            Err("Connection index out of bounds")
        }
    }
    
    /// Apply the activation function to a value
    fn apply_activation_function(&self, x: T) -> T {
        self.activation_function.activate(x, self.activation_steepness)
    }

    /// Compute the derivative of the activation function for backpropagation
    ///
    /// # Returns
    /// The derivative of the activation function at the current neuron state
    pub fn activation_derivative(&self) -> T {
        self.activation_function.derivative(self.sum, self.value, self.activation_steepness)
    }
}

impl<T: Float> PartialEq for Neuron<T> {
    fn eq(&self, other: &Self) -> bool {
        self.activation_function == other.activation_function
            && self.activation_steepness == other.activation_steepness
            && self.is_bias == other.is_bias
            && self.connections == other.connections
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_creation() {
        let neuron = Neuron::<f32>::new(ActivationFunction::Sigmoid, 1.0);
        assert_eq!(neuron.activation_function, ActivationFunction::Sigmoid);
        assert_eq!(neuron.activation_steepness, 1.0);
        assert_eq!(neuron.sum, 0.0);
        assert_eq!(neuron.value, 0.0);
        assert!(!neuron.is_bias);
        assert!(neuron.connections.is_empty());
    }

    #[test]
    fn test_bias_neuron() {
        let bias = Neuron::<f32>::new_bias();
        assert!(bias.is_bias);
        assert_eq!(bias.value, 1.0);
        assert_eq!(bias.sum, 1.0);
    }

    #[test]
    fn test_add_connection() {
        let mut neuron = Neuron::<f32>::new(ActivationFunction::ReLU, 1.0);
        neuron.add_connection(0, 0.5);
        neuron.add_connection(1, -0.3);

        assert_eq!(neuron.connections.len(), 2);
        assert_eq!(neuron.connections[0].from_neuron, 0);
        assert_eq!(neuron.connections[0].weight, 0.5);
        assert_eq!(neuron.connections[1].from_neuron, 1);
        assert_eq!(neuron.connections[1].weight, -0.3);
    }

    #[test]
    fn test_reset_neuron() {
        let mut neuron = Neuron::<f32>::new(ActivationFunction::Sigmoid, 1.0);
        neuron.sum = 5.0;
        neuron.value = 2.5;

        neuron.reset();
        assert_eq!(neuron.sum, 0.0);
        assert_eq!(neuron.value, 0.0);
    }

    #[test]
    fn test_reset_bias_neuron() {
        let mut bias = Neuron::<f32>::new_bias();
        bias.sum = 5.0;
        bias.value = 2.5;

        bias.reset();
        assert_eq!(bias.sum, 1.0);
        assert_eq!(bias.value, 1.0);
    }

    #[test]
    fn test_set_value() {
        let mut neuron = Neuron::<f32>::new(ActivationFunction::Linear, 1.0);
        // Use std::f32::consts::PI instead of hardcoded value
        neuron.set_value(std::f32::consts::PI);
        assert_eq!(neuron.value, std::f32::consts::PI);
        assert_eq!(neuron.sum, std::f32::consts::PI);
    }

    #[test]
    fn test_calculate() {
        let mut neuron = Neuron::<f32>::new(ActivationFunction::Linear, 1.0);
        neuron.add_connection(0, 0.5);
        neuron.add_connection(1, -0.3);
        neuron.add_connection(2, 0.2);

        let inputs = vec![1.0, 2.0, -1.0];
        neuron.calculate(&inputs);

        // 1.0 * 0.5 + 2.0 * -0.3 + -1.0 * 0.2 = 0.5 - 0.6 - 0.2 = -0.3
        assert_eq!(neuron.sum, -0.3);
        // For linear activation with steepness 1.0: f(x) = x * steepness = -0.3 * 1.0 = -0.3
        assert_eq!(neuron.value, -0.3);
    }

    #[test]
    fn test_activation_strategy_pattern() {
        // Test that neuron delegation to activation functions works correctly
        let mut sigmoid_neuron = Neuron::<f32>::new(ActivationFunction::Sigmoid, 1.0);
        sigmoid_neuron.sum = 0.0; // Set sum directly for testing activation
        sigmoid_neuron.value = sigmoid_neuron.apply_activation_function(sigmoid_neuron.sum);
        
        // Should be approximately 0.5 for sigmoid(0)
        assert!((sigmoid_neuron.value - 0.5).abs() < 1e-6);
        
        let mut relu_neuron = Neuron::<f32>::new(ActivationFunction::ReLU, 1.0);
        relu_neuron.sum = -1.0;
        relu_neuron.value = relu_neuron.apply_activation_function(relu_neuron.sum);
        assert_eq!(relu_neuron.value, 0.0);
        
        relu_neuron.sum = 2.0;
        relu_neuron.value = relu_neuron.apply_activation_function(relu_neuron.sum);
        assert_eq!(relu_neuron.value, 2.0);
    }

    #[test]
    fn test_activation_derivative() {
        // Test derivative computation for different activation functions
        let mut sigmoid_neuron = Neuron::<f32>::new(ActivationFunction::Sigmoid, 1.0);
        sigmoid_neuron.sum = 0.0;
        sigmoid_neuron.value = sigmoid_neuron.apply_activation_function(sigmoid_neuron.sum);
        
        let derivative = sigmoid_neuron.activation_derivative();
        // For sigmoid at x=0: derivative should be approximately 0.5
        assert!((derivative - 0.5).abs() < 1e-6);
        
        let mut linear_neuron = Neuron::<f32>::new(ActivationFunction::Linear, 2.0);
        linear_neuron.sum = 5.0; // Doesn't matter for linear
        linear_neuron.value = linear_neuron.apply_activation_function(linear_neuron.sum);
        
        let derivative = linear_neuron.activation_derivative();
        // For linear: derivative should equal steepness
        assert_eq!(derivative, 2.0);
        
        let mut relu_neuron = Neuron::<f32>::new(ActivationFunction::ReLU, 1.0);
        relu_neuron.sum = 1.0; // Positive input
        relu_neuron.value = relu_neuron.apply_activation_function(relu_neuron.sum);
        
        let derivative = relu_neuron.activation_derivative();
        // For ReLU with positive input: derivative should be 1.0
        assert_eq!(derivative, 1.0);
        
        relu_neuron.sum = -1.0; // Negative input
        relu_neuron.value = relu_neuron.apply_activation_function(relu_neuron.sum);
        
        let derivative = relu_neuron.activation_derivative();
        // For ReLU with negative input: derivative should be 0.0
        assert_eq!(derivative, 0.0);
    }
}
