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

        let sum = self.weighted_sum(inputs);
        self.sum = sum;

        // Apply activation function
        self.value = self.apply_activation_function(sum);
    }

    /// Computes the weighted input sum without applying the activation.
    #[inline]
    pub(crate) fn weighted_sum(&self, inputs: &[T]) -> T {
        // Accumulate locally so the compiler can keep the sum in a register
        // instead of writing through `self` on every step.
        let mut sum = T::zero();
        for connection in &self.connections {
            if let Some(&input) = inputs.get(connection.from_neuron) {
                sum = sum + input * connection.weight;
            }
        }
        sum
    }

    /// Apply the activation function to the given input
    fn apply_activation_function(&self, x: T) -> T {
        apply_activation(self.activation_function, self.activation_steepness, x)
    }

    /// Calculate the derivative of the activation function at the current value
    pub fn activation_derivative(&self) -> T {
        match self.activation_function {
            ActivationFunction::Linear => self.activation_steepness,
            ActivationFunction::Sigmoid => {
                // For sigmoid: f'(x) = f(x) * (1 - f(x)) * steepness
                self.value * (T::one() - self.value) * self.activation_steepness
            }
            ActivationFunction::ReLU => {
                if self.sum > T::zero() {
                    T::one()
                } else {
                    T::zero()
                }
            }
            ActivationFunction::ReLULeaky => {
                let alpha = T::from(0.01).unwrap_or(T::zero());
                if self.sum > T::zero() {
                    T::one()
                } else {
                    alpha
                }
            }
            ActivationFunction::Tanh | ActivationFunction::SigmoidSymmetric => {
                // For tanh: f'(x) = (1 - f(x)²) * steepness
                (T::one() - self.value * self.value) * self.activation_steepness
            }
            ActivationFunction::Gaussian => {
                // For gaussian: f'(x) = -2 * steepness² * x * f(x)
                let x_scaled = self.sum * self.activation_steepness;
                let neg_two = T::from(-2.0).unwrap_or(T::zero());
                neg_two * self.activation_steepness * x_scaled * self.value
            }
            _ => T::one(), // Fallback
        }
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
}

/// Applies `function` with `steepness` to the raw input `x`.
///
/// Free function so `Layer::calculate` can resolve the activation once per
/// layer; inlined into a loop specialized for one variant, the match folds
/// away entirely.
#[inline]
pub(crate) fn apply_activation<T: Float>(function: ActivationFunction, steepness: T, x: T) -> T {
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
        ActivationFunction::Tanh | ActivationFunction::SigmoidSymmetric => (steepness * x).tanh(),
        ActivationFunction::Gaussian => {
            let x_scaled = x * steepness;
            (-x_scaled * x_scaled).exp()
        }
        _ => x, // Fallback for other functions
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
    }
}
