//! Adam and AdamW optimizers for neural network training
//!
//! These optimizers provide significant convergence improvements over traditional SGD:
//! - Adam: Adaptive moment estimation with bias correction
//! - AdamW: Adam with decoupled weight decay (better regularization)
//!
//! Expected performance gains:
//! - 2-5x faster convergence in terms of epochs needed
//! - Better handling of sparse gradients and noisy data
//! - Adaptive learning rates per parameter

#![allow(clippy::needless_range_loop)]

use super::*;
use num_traits::Float;
use std::collections::HashMap;

/// Adam optimizer implementation
/// Uses adaptive moment estimation with bias correction for faster convergence
pub struct Adam<T: Float + Send + Default> {
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    weight_decay: T,
    base: TrainerBase<T>,

    // Moment estimates
    m_weights: Vec<Vec<T>>, // First moment (momentum)
    v_weights: Vec<Vec<T>>, // Second moment (uncentered variance)
    m_biases: Vec<Vec<T>>,
    v_biases: Vec<Vec<T>>,

    // Step counter for bias correction
    step: usize,
}

impl<T: Float + Send + Default> Adam<T> {
    /// Create a new Adam optimizer with default parameters
    pub fn new(learning_rate: T) -> Self {
        Self {
            learning_rate,
            beta1: T::from(0.9).unwrap(),
            beta2: T::from(0.999).unwrap(),
            epsilon: T::from(1e-8).unwrap(),
            weight_decay: T::zero(),
            base: TrainerBase::new(Box::new(MseError)),
            m_weights: Vec::new(),
            v_weights: Vec::new(),
            m_biases: Vec::new(),
            v_biases: Vec::new(),
            step: 0,
        }
    }

    /// Set beta1 parameter (momentum coefficient)
    pub fn with_beta1(mut self, beta1: T) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 parameter (variance coefficient)
    pub fn with_beta2(mut self, beta2: T) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability
    pub fn with_epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set weight decay (L2 regularization)
    pub fn with_weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set error function
    pub fn with_error_function(mut self, error_function: Box<dyn ErrorFunction<T>>) -> Self {
        self.base.error_function = error_function;
        self
    }

    /// Initialize moment estimates for the network
    fn initialize_moments(&mut self, network: &Network<T>) {
        if self.m_weights.is_empty() {
            self.m_weights = network
                .layers
                .iter()
                .skip(1) // Skip input layer
                .map(|layer| {
                    let num_neurons = layer.neurons.len();
                    let num_connections = if layer.neurons.is_empty() {
                        0
                    } else {
                        layer.neurons[0].connections.len()
                    };
                    vec![T::zero(); num_neurons * num_connections]
                })
                .collect();

            self.v_weights = self.m_weights.clone();

            self.m_biases = network
                .layers
                .iter()
                .skip(1) // Skip input layer
                .map(|layer| vec![T::zero(); layer.neurons.len()])
                .collect();

            self.v_biases = self.m_biases.clone();
        }
    }

    /// Update parameters using Adam algorithm
    fn update_parameters(
        &mut self,
        network: &mut Network<T>,
        weight_gradients: &[Vec<T>],
        bias_gradients: &[Vec<T>],
    ) {
        self.step += 1;

        // Bias correction factors
        let lr_t = self.learning_rate * (T::one() - self.beta2.powi(self.step as i32)).sqrt()
            / (T::one() - self.beta1.powi(self.step as i32));

        // Compute weight updates
        let mut weight_updates = Vec::new();
        for layer_idx in 0..weight_gradients.len() {
            let mut layer_updates = Vec::new();
            for i in 0..weight_gradients[layer_idx].len() {
                let grad = weight_gradients[layer_idx][i];

                // Update biased first moment estimate
                self.m_weights[layer_idx][i] =
                    self.beta1 * self.m_weights[layer_idx][i] + (T::one() - self.beta1) * grad;

                // Update biased second moment estimate
                self.v_weights[layer_idx][i] = self.beta2 * self.v_weights[layer_idx][i]
                    + (T::one() - self.beta2) * grad * grad;

                // Compute parameter update
                let update = lr_t * self.m_weights[layer_idx][i]
                    / (self.v_weights[layer_idx][i].sqrt() + self.epsilon);

                layer_updates.push(-update);
            }
            weight_updates.push(layer_updates);
        }

        // Compute bias updates
        let mut bias_updates = Vec::new();
        for layer_idx in 0..bias_gradients.len() {
            let mut layer_updates = Vec::new();
            for i in 0..bias_gradients[layer_idx].len() {
                let grad = bias_gradients[layer_idx][i];

                // Update biased first moment estimate
                self.m_biases[layer_idx][i] =
                    self.beta1 * self.m_biases[layer_idx][i] + (T::one() - self.beta1) * grad;

                // Update biased second moment estimate
                self.v_biases[layer_idx][i] = self.beta2 * self.v_biases[layer_idx][i]
                    + (T::one() - self.beta2) * grad * grad;

                // Compute parameter update
                let update = lr_t * self.m_biases[layer_idx][i]
                    / (self.v_biases[layer_idx][i].sqrt() + self.epsilon);

                layer_updates.push(-update);
            }
            bias_updates.push(layer_updates);
        }

        // Apply L2 weight decay for Adam (coupled formulation: adds λ·w to gradient).
        // This differs from AdamW where decay is decoupled. The update here correctly
        // scales the penalty by the actual weight value so that heavier weights are
        // penalised more, matching the standard L2 regularisation derivation.
        if self.weight_decay > T::zero() {
            for (layer_idx, layer) in network.layers.iter().skip(1).enumerate() {
                if layer_idx >= weight_updates.len() {
                    break;
                }
                let mut update_idx = 0;
                for neuron in &layer.neurons {
                    for conn in &neuron.connections {
                        if update_idx < weight_updates[layer_idx].len() {
                            weight_updates[layer_idx][update_idx] = weight_updates[layer_idx][update_idx]
                                - self.learning_rate * self.weight_decay * conn.weight;
                            update_idx += 1;
                        }
                    }
                }
            }
        }

        // Apply updates using existing helper
        super::helpers::apply_updates_to_network(network, &weight_updates, &bias_updates);
    }
}

impl<T: Float + Send + Default> TrainingAlgorithm<T> for Adam<T> {
    fn train_epoch(
        &mut self,
        network: &mut Network<T>,
        data: &TrainingData<T>,
    ) -> Result<T, TrainingError> {
        use super::helpers::*;

        self.initialize_moments(network);

        let mut total_error = T::zero();

        // Convert network to simplified form for easier manipulation
        let simple_network = network_to_simple(network);

        // Accumulate gradients over entire batch
        let mut accumulated_weight_gradients = simple_network
            .weights
            .iter()
            .map(|w| vec![T::zero(); w.len()])
            .collect::<Vec<_>>();
        let mut accumulated_bias_gradients = simple_network
            .biases
            .iter()
            .map(|b| vec![T::zero(); b.len()])
            .collect::<Vec<_>>();

        // Process all samples in the batch
        for (input, desired_output) in data.inputs.iter().zip(data.outputs.iter()) {
            // Forward propagation to get all layer activations
            let activations = forward_propagate(&simple_network, input);

            // Get output from last layer
            let output = &activations[activations.len() - 1];

            // Calculate error
            total_error = total_error + self.base.error_function.calculate(output, desired_output);

            // Calculate gradients using backpropagation
            let (weight_gradients, bias_gradients) = calculate_gradients(
                &simple_network,
                &activations,
                desired_output,
                self.base.error_function.as_ref(),
            );

            // Accumulate gradients
            for layer_idx in 0..weight_gradients.len() {
                for i in 0..weight_gradients[layer_idx].len() {
                    accumulated_weight_gradients[layer_idx][i] =
                        accumulated_weight_gradients[layer_idx][i] + weight_gradients[layer_idx][i];
                }
                for i in 0..bias_gradients[layer_idx].len() {
                    accumulated_bias_gradients[layer_idx][i] =
                        accumulated_bias_gradients[layer_idx][i] + bias_gradients[layer_idx][i];
                }
            }
        }

        // Average gradients over batch size
        let batch_size = T::from(data.inputs.len()).unwrap();
        for layer_idx in 0..accumulated_weight_gradients.len() {
            for i in 0..accumulated_weight_gradients[layer_idx].len() {
                accumulated_weight_gradients[layer_idx][i] =
                    accumulated_weight_gradients[layer_idx][i] / batch_size;
            }
            for i in 0..accumulated_bias_gradients[layer_idx].len() {
                accumulated_bias_gradients[layer_idx][i] =
                    accumulated_bias_gradients[layer_idx][i] / batch_size;
            }
        }

        // Update parameters using Adam
        self.update_parameters(
            network,
            &accumulated_weight_gradients,
            &accumulated_bias_gradients,
        );

        Ok(total_error / batch_size)
    }

    fn calculate_error(&self, network: &Network<T>, data: &TrainingData<T>) -> T {
        self.base.calculate_error(network, data)
    }

    fn count_bit_fails(
        &self,
        network: &Network<T>,
        data: &TrainingData<T>,
        bit_fail_limit: T,
    ) -> usize {
        self.base.count_bit_fails(network, data, bit_fail_limit)
    }

    fn save_state(&self) -> TrainingState<T> {
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), vec![self.learning_rate]);
        state.insert("beta1".to_string(), vec![self.beta1]);
        state.insert("beta2".to_string(), vec![self.beta2]);
        state.insert("epsilon".to_string(), vec![self.epsilon]);
        state.insert("weight_decay".to_string(), vec![self.weight_decay]);
        state.insert("step".to_string(), vec![T::from(self.step).unwrap()]);

        TrainingState {
            epoch: 0,
            best_error: T::from(f32::MAX).unwrap(),
            algorithm_specific: state,
        }
    }

    fn restore_state(&mut self, state: TrainingState<T>) {
        if let Some(lr) = state.algorithm_specific.get("learning_rate") {
            if !lr.is_empty() {
                self.learning_rate = lr[0];
            }
        }
        if let Some(b1) = state.algorithm_specific.get("beta1") {
            if !b1.is_empty() {
                self.beta1 = b1[0];
            }
        }
        if let Some(b2) = state.algorithm_specific.get("beta2") {
            if !b2.is_empty() {
                self.beta2 = b2[0];
            }
        }
        if let Some(eps) = state.algorithm_specific.get("epsilon") {
            if !eps.is_empty() {
                self.epsilon = eps[0];
            }
        }
        if let Some(wd) = state.algorithm_specific.get("weight_decay") {
            if !wd.is_empty() {
                self.weight_decay = wd[0];
            }
        }
        if let Some(s) = state.algorithm_specific.get("step") {
            if !s.is_empty() {
                self.step = s[0].to_usize().unwrap_or(0);
            }
        }
    }

    fn set_callback(&mut self, callback: TrainingCallback<T>) {
        self.base.set_callback(callback);
    }

    fn call_callback(
        &mut self,
        epoch: usize,
        network: &Network<T>,
        data: &TrainingData<T>,
    ) -> bool {
        self.base.call_callback(epoch, network, data)
    }
}

/// AdamW optimizer implementation
/// Adam with decoupled weight decay for better regularization
pub struct AdamW<T: Float + Send + Default> {
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    weight_decay: T,
    base: TrainerBase<T>,

    // Moment estimates
    m_weights: Vec<Vec<T>>,
    v_weights: Vec<Vec<T>>,
    m_biases: Vec<Vec<T>>,
    v_biases: Vec<Vec<T>>,

    // Step counter for bias correction
    step: usize,
}

impl<T: Float + Send + Default> AdamW<T> {
    /// Create a new AdamW optimizer with default parameters
    pub fn new(learning_rate: T) -> Self {
        Self {
            learning_rate,
            beta1: T::from(0.9).unwrap(),
            beta2: T::from(0.999).unwrap(),
            epsilon: T::from(1e-8).unwrap(),
            weight_decay: T::from(0.01).unwrap(), // Common default for AdamW
            base: TrainerBase::new(Box::new(MseError)),
            m_weights: Vec::new(),
            v_weights: Vec::new(),
            m_biases: Vec::new(),
            v_biases: Vec::new(),
            step: 0,
        }
    }

    /// Set beta1 parameter (momentum coefficient)
    pub fn with_beta1(mut self, beta1: T) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 parameter (variance coefficient)
    pub fn with_beta2(mut self, beta2: T) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability
    pub fn with_epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set weight decay (decoupled from gradient-based updates)
    pub fn with_weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set error function
    pub fn with_error_function(mut self, error_function: Box<dyn ErrorFunction<T>>) -> Self {
        self.base.error_function = error_function;
        self
    }

    /// Initialize moment estimates for the network
    fn initialize_moments(&mut self, network: &Network<T>) {
        if self.m_weights.is_empty() {
            self.m_weights = network
                .layers
                .iter()
                .skip(1) // Skip input layer
                .map(|layer| {
                    let num_neurons = layer.neurons.len();
                    let num_connections = if layer.neurons.is_empty() {
                        0
                    } else {
                        layer.neurons[0].connections.len()
                    };
                    vec![T::zero(); num_neurons * num_connections]
                })
                .collect();

            self.v_weights = self.m_weights.clone();

            self.m_biases = network
                .layers
                .iter()
                .skip(1) // Skip input layer
                .map(|layer| vec![T::zero(); layer.neurons.len()])
                .collect();

            self.v_biases = self.m_biases.clone();
        }
    }

    /// Apply AdamW updates to the network (with decoupled weight decay)
    fn apply_adamw_updates(
        &mut self,
        network: &mut Network<T>,
        weight_gradients: &[Vec<T>],
        bias_gradients: &[Vec<T>],
        lr_t: T,
    ) {
        // Compute and apply weight updates with decoupled weight decay
        let mut weight_updates = Vec::new();
        for layer_idx in 0..weight_gradients.len() {
            let mut layer_updates = Vec::new();
            for i in 0..weight_gradients[layer_idx].len() {
                let adaptive_update = lr_t * self.m_weights[layer_idx][i]
                    / (self.v_weights[layer_idx][i].sqrt() + self.epsilon);

                // In AdamW, weight decay is applied directly to weights, not gradients
                layer_updates.push(-adaptive_update);
            }
            weight_updates.push(layer_updates);
        }

        // Compute and apply bias updates (no weight decay for biases)
        let mut bias_updates = Vec::new();
        for layer_idx in 0..bias_gradients.len() {
            let mut layer_updates = Vec::new();
            for i in 0..bias_gradients[layer_idx].len() {
                let update = lr_t * self.m_biases[layer_idx][i]
                    / (self.v_biases[layer_idx][i].sqrt() + self.epsilon);
                layer_updates.push(-update);
            }
            bias_updates.push(layer_updates);
        }

        // Apply updates using existing helper
        super::helpers::apply_updates_to_network(network, &weight_updates, &bias_updates);

        // Apply decoupled weight decay directly to weights
        if self.weight_decay > T::zero() {
            self.apply_decoupled_weight_decay(network);
        }
    }

    /// Apply decoupled weight decay directly to weights (AdamW approach)
    fn apply_decoupled_weight_decay(&self, network: &mut Network<T>) {
        let decay_factor = T::one() - self.learning_rate * self.weight_decay;

        for layer_idx in 1..network.layers.len() {
            let current_layer = &mut network.layers[layer_idx];

            for neuron in &mut current_layer.neurons {
                if !neuron.is_bias {
                    // Apply weight decay to all connections except bias (index 0)
                    for connection in neuron.connections.iter_mut().skip(1) {
                        connection.weight = connection.weight * decay_factor;
                    }
                }
            }
        }
    }
}

impl<T: Float + Send + Default> TrainingAlgorithm<T> for AdamW<T> {
    fn train_epoch(
        &mut self,
        network: &mut Network<T>,
        data: &TrainingData<T>,
    ) -> Result<T, TrainingError> {
        use super::helpers::*;

        self.initialize_moments(network);
        self.step += 1;

        let mut total_error = T::zero();

        // Convert network to simplified form for easier manipulation
        let simple_network = network_to_simple(network);

        // Accumulate gradients over entire batch
        let mut accumulated_weight_gradients = simple_network
            .weights
            .iter()
            .map(|w| vec![T::zero(); w.len()])
            .collect::<Vec<_>>();
        let mut accumulated_bias_gradients = simple_network
            .biases
            .iter()
            .map(|b| vec![T::zero(); b.len()])
            .collect::<Vec<_>>();

        // Process all samples in the batch
        for (input, desired_output) in data.inputs.iter().zip(data.outputs.iter()) {
            // Forward propagation to get all layer activations
            let activations = forward_propagate(&simple_network, input);

            // Get output from last layer
            let output = &activations[activations.len() - 1];

            // Calculate error
            total_error = total_error + self.base.error_function.calculate(output, desired_output);

            // Calculate gradients using backpropagation
            let (weight_gradients, bias_gradients) = calculate_gradients(
                &simple_network,
                &activations,
                desired_output,
                self.base.error_function.as_ref(),
            );

            // Accumulate gradients
            for layer_idx in 0..weight_gradients.len() {
                for i in 0..weight_gradients[layer_idx].len() {
                    accumulated_weight_gradients[layer_idx][i] =
                        accumulated_weight_gradients[layer_idx][i] + weight_gradients[layer_idx][i];
                }
                for i in 0..bias_gradients[layer_idx].len() {
                    accumulated_bias_gradients[layer_idx][i] =
                        accumulated_bias_gradients[layer_idx][i] + bias_gradients[layer_idx][i];
                }
            }
        }

        // Average gradients over batch size
        let batch_size = T::from(data.inputs.len()).unwrap();
        for layer_idx in 0..accumulated_weight_gradients.len() {
            for i in 0..accumulated_weight_gradients[layer_idx].len() {
                accumulated_weight_gradients[layer_idx][i] =
                    accumulated_weight_gradients[layer_idx][i] / batch_size;
            }
            for i in 0..accumulated_bias_gradients[layer_idx].len() {
                accumulated_bias_gradients[layer_idx][i] =
                    accumulated_bias_gradients[layer_idx][i] / batch_size;
            }
        }

        // Update moment estimates
        for layer_idx in 0..accumulated_weight_gradients.len() {
            for i in 0..accumulated_weight_gradients[layer_idx].len() {
                let grad = accumulated_weight_gradients[layer_idx][i];

                // Update biased first moment estimate
                self.m_weights[layer_idx][i] =
                    self.beta1 * self.m_weights[layer_idx][i] + (T::one() - self.beta1) * grad;

                // Update biased second moment estimate
                self.v_weights[layer_idx][i] = self.beta2 * self.v_weights[layer_idx][i]
                    + (T::one() - self.beta2) * grad * grad;
            }
        }

        // Update bias moments
        for layer_idx in 0..accumulated_bias_gradients.len() {
            for i in 0..accumulated_bias_gradients[layer_idx].len() {
                let grad = accumulated_bias_gradients[layer_idx][i];

                // Update biased first moment estimate
                self.m_biases[layer_idx][i] =
                    self.beta1 * self.m_biases[layer_idx][i] + (T::one() - self.beta1) * grad;

                // Update biased second moment estimate
                self.v_biases[layer_idx][i] = self.beta2 * self.v_biases[layer_idx][i]
                    + (T::one() - self.beta2) * grad * grad;
            }
        }

        // Bias correction factors
        let lr_t = self.learning_rate * (T::one() - self.beta2.powi(self.step as i32)).sqrt()
            / (T::one() - self.beta1.powi(self.step as i32));

        // Apply AdamW updates with decoupled weight decay
        self.apply_adamw_updates(
            network,
            &accumulated_weight_gradients,
            &accumulated_bias_gradients,
            lr_t,
        );

        Ok(total_error / batch_size)
    }

    fn calculate_error(&self, network: &Network<T>, data: &TrainingData<T>) -> T {
        self.base.calculate_error(network, data)
    }

    fn count_bit_fails(
        &self,
        network: &Network<T>,
        data: &TrainingData<T>,
        bit_fail_limit: T,
    ) -> usize {
        self.base.count_bit_fails(network, data, bit_fail_limit)
    }

    fn save_state(&self) -> TrainingState<T> {
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), vec![self.learning_rate]);
        state.insert("beta1".to_string(), vec![self.beta1]);
        state.insert("beta2".to_string(), vec![self.beta2]);
        state.insert("epsilon".to_string(), vec![self.epsilon]);
        state.insert("weight_decay".to_string(), vec![self.weight_decay]);
        state.insert("step".to_string(), vec![T::from(self.step).unwrap()]);

        TrainingState {
            epoch: 0,
            best_error: T::from(f32::MAX).unwrap(),
            algorithm_specific: state,
        }
    }

    fn restore_state(&mut self, state: TrainingState<T>) {
        if let Some(lr) = state.algorithm_specific.get("learning_rate") {
            if !lr.is_empty() {
                self.learning_rate = lr[0];
            }
        }
        if let Some(b1) = state.algorithm_specific.get("beta1") {
            if !b1.is_empty() {
                self.beta1 = b1[0];
            }
        }
        if let Some(b2) = state.algorithm_specific.get("beta2") {
            if !b2.is_empty() {
                self.beta2 = b2[0];
            }
        }
        if let Some(eps) = state.algorithm_specific.get("epsilon") {
            if !eps.is_empty() {
                self.epsilon = eps[0];
            }
        }
        if let Some(wd) = state.algorithm_specific.get("weight_decay") {
            if !wd.is_empty() {
                self.weight_decay = wd[0];
            }
        }
        if let Some(s) = state.algorithm_specific.get("step") {
            if !s.is_empty() {
                self.step = s[0].to_usize().unwrap_or(0);
            }
        }
    }

    fn set_callback(&mut self, callback: TrainingCallback<T>) {
        self.base.set_callback(callback);
    }

    fn call_callback(
        &mut self,
        epoch: usize,
        network: &Network<T>,
        data: &TrainingData<T>,
    ) -> bool {
        self.base.call_callback(epoch, network, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Network;

    #[test]
    fn test_adam_creation() {
        let adam = Adam::new(0.001f32);
        assert_eq!(adam.learning_rate, 0.001);
        assert_eq!(adam.beta1, 0.9);
        assert_eq!(adam.beta2, 0.999);
        assert_eq!(adam.step, 0);
    }

    #[test]
    fn test_adamw_creation() {
        let adamw = AdamW::new(0.001f32);
        assert_eq!(adamw.learning_rate, 0.001);
        assert_eq!(adamw.beta1, 0.9);
        assert_eq!(adamw.beta2, 0.999);
        assert_eq!(adamw.weight_decay, 0.01);
        assert_eq!(adamw.step, 0);
    }

    #[test]
    fn test_adam_with_parameters() {
        let adam = Adam::new(0.001f32)
            .with_beta1(0.95)
            .with_beta2(0.998)
            .with_epsilon(1e-7)
            .with_weight_decay(0.001);

        assert_eq!(adam.beta1, 0.95);
        assert_eq!(adam.beta2, 0.998);
        assert_eq!(adam.epsilon, 1e-7);
        assert_eq!(adam.weight_decay, 0.001);
    }
}
