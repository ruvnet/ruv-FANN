//! Shared recurrent layer components
//!
//! This module provides common building blocks for recurrent neural networks,
//! including state management, gate mechanisms, and sequence processing utilities.

use crate::errors::{NeuroDivergentError, NeuroDivergentResult};
use crate::foundation::{RecurrentState, SequenceProcessor};
use crate::utils::math;
use num_traits::Float;
use ruv_fann::{ActivationFunction, Network, NetworkBuilder};
use std::collections::HashMap;

/// Generic recurrent layer trait that all recurrent layers must implement
pub trait RecurrentLayer<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>:
    Send + Sync
{
    /// Process a single time step
    fn forward_step(&mut self, input: &[T]) -> NeuroDivergentResult<Vec<T>>;

    /// Process a full sequence
    fn forward_sequence(&mut self, sequence: &[Vec<T>]) -> NeuroDivergentResult<Vec<Vec<T>>>;

    /// Reset the internal state
    fn reset_state(&mut self);

    /// Get the current hidden state
    fn get_state(&self) -> Vec<T>;

    /// Set the hidden state
    fn set_state(&mut self, state: Vec<T>) -> NeuroDivergentResult<()>;

    /// Get the hidden size
    fn hidden_size(&self) -> usize;

    /// Get the input size
    fn input_size(&self) -> usize;

    /// Clone the layer
    fn clone_layer(&self) -> Box<dyn RecurrentLayer<T> + Send + Sync>;
}

/// Basic recurrent cell implementation
#[derive(Debug, Clone)]
pub struct BasicRecurrentCell<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    pub(super) input_size: usize,
    pub(super) hidden_size: usize,

    // Weight matrices
    pub(super) input_weights: Vec<Vec<T>>, // W_ih: input to hidden
    pub(super) hidden_weights: Vec<Vec<T>>, // W_hh: hidden to hidden
    pub(super) bias: Vec<T>,               // bias vector

    // Current state
    pub(super) hidden_state: Vec<T>,

    // Activation function
    activation: ActivationFunction,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> BasicRecurrentCell<T> {
    /// Create a new basic recurrent cell
    pub fn new(input_size: usize, hidden_size: usize, activation: ActivationFunction) -> Self {
        Self::with_rng(input_size, hidden_size, activation, &mut rand::thread_rng())
    }

    pub(super) fn with_rng(
        input_size: usize,
        hidden_size: usize,
        activation: ActivationFunction,
        rng: &mut impl rand::Rng,
    ) -> Self {
        // Initialize weights with Xavier initialization
        let xavier_input = T::from(1.0 / (input_size as f64).sqrt()).unwrap();
        let xavier_hidden = T::from(1.0 / (hidden_size as f64).sqrt()).unwrap();

        let input_weights = (0..hidden_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| T::from(rng.gen::<f64>() * 2.0 - 1.0).unwrap() * xavier_input)
                    .collect()
            })
            .collect();

        let hidden_weights = (0..hidden_size)
            .map(|_| {
                (0..hidden_size)
                    .map(|_| T::from(rng.gen::<f64>() * 2.0 - 1.0).unwrap() * xavier_hidden)
                    .collect()
            })
            .collect();

        let bias = vec![T::zero(); hidden_size];
        let hidden_state = vec![T::zero(); hidden_size];

        Self {
            input_size,
            hidden_size,
            input_weights,
            hidden_weights,
            bias,
            hidden_state,
            activation,
        }
    }

    /// Apply activation function
    pub(super) fn apply_activation(&self, value: T) -> T {
        match self.activation {
            ActivationFunction::Tanh | ActivationFunction::SigmoidSymmetric => value.tanh(),
            ActivationFunction::Sigmoid => {
                if value >= T::zero() {
                    T::one() / (T::one() + (-value).exp())
                } else {
                    let exp_val = value.exp();
                    exp_val / (T::one() + exp_val)
                }
            }
            ActivationFunction::ReLU => value.max(T::zero()),
            ActivationFunction::ReLULeaky | ActivationFunction::LeakyRelu => {
                if value > T::zero() {
                    value
                } else {
                    value * T::from(0.01).unwrap()
                }
            }
            ActivationFunction::Linear => value,
            _ => value.tanh(), // Default to tanh
        }
    }

    pub(super) fn activation_derivative(&self, output: T) -> T {
        match self.activation {
            ActivationFunction::Linear => T::one(),
            ActivationFunction::Sigmoid => output * (T::one() - output),
            ActivationFunction::ReLU => {
                if output > T::zero() {
                    T::one()
                } else {
                    T::zero()
                }
            }
            ActivationFunction::ReLULeaky | ActivationFunction::LeakyRelu => {
                if output > T::zero() {
                    T::one()
                } else {
                    T::from(0.01).unwrap()
                }
            }
            _ => T::one() - output * output,
        }
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> RecurrentLayer<T>
    for BasicRecurrentCell<T>
{
    fn forward_step(&mut self, input: &[T]) -> NeuroDivergentResult<Vec<T>> {
        if self.input_size == 0
            || self.hidden_size == 0
            || !matches!(
                self.activation,
                ActivationFunction::Tanh
                    | ActivationFunction::SigmoidSymmetric
                    | ActivationFunction::Sigmoid
                    | ActivationFunction::ReLU
                    | ActivationFunction::Linear
                    | ActivationFunction::ReLULeaky
                    | ActivationFunction::LeakyRelu
            )
        {
            return Err(NeuroDivergentError::config(
                "Unsupported basic recurrent cell dimensions or activation",
            ));
        }
        if input.len() != self.input_size {
            return Err(NeuroDivergentError::dimension_mismatch(
                self.input_size,
                input.len(),
            ));
        }
        if input
            .iter()
            .chain(&self.hidden_state)
            .any(|v| !v.is_finite())
        {
            return Err(NeuroDivergentError::data(
                "Recurrent inputs and states must be finite",
            ));
        }

        let mut new_hidden = Vec::with_capacity(self.hidden_size);

        for i in 0..self.hidden_size {
            let mut sum = self.bias[i];

            // Input contribution: W_ih * x
            for (j, &x) in input.iter().enumerate() {
                sum = sum + self.input_weights[i][j] * x;
            }

            // Hidden state contribution: W_hh * h_{t-1}
            for (j, &h) in self.hidden_state.iter().enumerate() {
                sum = sum + self.hidden_weights[i][j] * h;
            }

            // Apply activation function
            new_hidden.push(self.apply_activation(sum));
        }

        if new_hidden.iter().any(|v| !v.is_finite()) {
            return Err(NeuroDivergentError::training(
                "Basic recurrent state overflowed",
            ));
        }
        self.hidden_state = new_hidden.clone();
        Ok(new_hidden)
    }

    fn forward_sequence(&mut self, sequence: &[Vec<T>]) -> NeuroDivergentResult<Vec<Vec<T>>> {
        let mut outputs = Vec::with_capacity(sequence.len());

        for input in sequence {
            let output = self.forward_step(input)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    fn reset_state(&mut self) {
        self.hidden_state.fill(T::zero());
    }

    fn get_state(&self) -> Vec<T> {
        self.hidden_state.clone()
    }

    fn set_state(&mut self, state: Vec<T>) -> NeuroDivergentResult<()> {
        if state.len() != self.hidden_size {
            return Err(NeuroDivergentError::dimension_mismatch(
                self.hidden_size,
                state.len(),
            ));
        }
        self.hidden_state = state;
        Ok(())
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn clone_layer(&self) -> Box<dyn RecurrentLayer<T> + Send + Sync> {
        Box::new(self.clone())
    }
}

/// LSTM cell implementation
#[derive(Debug, Clone)]
pub struct LSTMCell<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    input_size: usize,
    hidden_size: usize,

    // Gate networks using ruv-FANN
    forget_gate: Network<T>,
    input_gate: Network<T>,
    candidate_gate: Network<T>,
    output_gate: Network<T>,

    // LSTM states
    cell_state: Vec<T>,
    hidden_state: Vec<T>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> LSTMCell<T> {
    /// Create a new LSTM cell
    pub fn new(input_size: usize, hidden_size: usize) -> NeuroDivergentResult<Self> {
        let combined_input_size = input_size + hidden_size;

        // Create gate networks
        let forget_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .output_layer_with_activation(hidden_size, ActivationFunction::Sigmoid, T::one())
            .build();

        let input_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .output_layer_with_activation(hidden_size, ActivationFunction::Sigmoid, T::one())
            .build();

        let candidate_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .output_layer_with_activation(hidden_size, ActivationFunction::Tanh, T::one())
            .build();

        let output_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .output_layer_with_activation(hidden_size, ActivationFunction::Sigmoid, T::one())
            .build();

        let cell_state = vec![T::zero(); hidden_size];
        let hidden_state = vec![T::zero(); hidden_size];

        Ok(Self {
            input_size,
            hidden_size,
            forget_gate,
            input_gate,
            candidate_gate,
            output_gate,
            cell_state,
            hidden_state,
        })
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> RecurrentLayer<T>
    for LSTMCell<T>
{
    fn forward_step(&mut self, input: &[T]) -> NeuroDivergentResult<Vec<T>> {
        if input.len() != self.input_size {
            return Err(NeuroDivergentError::dimension_mismatch(
                self.input_size,
                input.len(),
            ));
        }

        // Combine input and previous hidden state
        let mut combined_input = input.to_vec();
        combined_input.extend_from_slice(&self.hidden_state);

        // Compute gates
        let forget_values = self.forget_gate.run(&combined_input)?;
        let input_values = self.input_gate.run(&combined_input)?;
        let candidate_values = self.candidate_gate.run(&combined_input)?;
        let output_values = self.output_gate.run(&combined_input)?;

        // Update cell state: C_t = f_t * C_{t-1} + i_t * C̃_t
        for i in 0..self.hidden_size {
            self.cell_state[i] =
                forget_values[i] * self.cell_state[i] + input_values[i] * candidate_values[i];
        }

        // Update hidden state: h_t = o_t * tanh(C_t)
        for i in 0..self.hidden_size {
            self.hidden_state[i] = output_values[i] * self.cell_state[i].tanh();
        }

        Ok(self.hidden_state.clone())
    }

    fn forward_sequence(&mut self, sequence: &[Vec<T>]) -> NeuroDivergentResult<Vec<Vec<T>>> {
        let mut outputs = Vec::with_capacity(sequence.len());

        for input in sequence {
            let output = self.forward_step(input)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    fn reset_state(&mut self) {
        self.cell_state.fill(T::zero());
        self.hidden_state.fill(T::zero());
    }

    fn get_state(&self) -> Vec<T> {
        // Return both cell state and hidden state concatenated
        let mut state = self.cell_state.clone();
        state.extend_from_slice(&self.hidden_state);
        state
    }

    fn set_state(&mut self, state: Vec<T>) -> NeuroDivergentResult<()> {
        if state.len() != 2 * self.hidden_size {
            return Err(NeuroDivergentError::dimension_mismatch(
                2 * self.hidden_size,
                state.len(),
            ));
        }

        self.cell_state = state[..self.hidden_size].to_vec();
        self.hidden_state = state[self.hidden_size..].to_vec();
        Ok(())
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn clone_layer(&self) -> Box<dyn RecurrentLayer<T> + Send + Sync> {
        Box::new(self.clone())
    }
}

/// GRU cell implementation
#[derive(Debug, Clone)]
pub struct GRUCell<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    input_size: usize,
    hidden_size: usize,

    // Gate networks
    reset_gate: Network<T>,
    update_gate: Network<T>,
    new_gate: Network<T>,

    // GRU state
    hidden_state: Vec<T>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> GRUCell<T> {
    /// Create a new GRU cell
    pub fn new(input_size: usize, hidden_size: usize) -> NeuroDivergentResult<Self> {
        let combined_input_size = input_size + hidden_size;

        // Create gate networks
        let reset_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .output_layer_with_activation(hidden_size, ActivationFunction::Sigmoid, T::one())
            .build();

        let update_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .output_layer_with_activation(hidden_size, ActivationFunction::Sigmoid, T::one())
            .build();

        let new_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .output_layer_with_activation(hidden_size, ActivationFunction::Tanh, T::one())
            .build();

        let hidden_state = vec![T::zero(); hidden_size];

        Ok(Self {
            input_size,
            hidden_size,
            reset_gate,
            update_gate,
            new_gate,
            hidden_state,
        })
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> RecurrentLayer<T>
    for GRUCell<T>
{
    fn forward_step(&mut self, input: &[T]) -> NeuroDivergentResult<Vec<T>> {
        if input.len() != self.input_size {
            return Err(NeuroDivergentError::dimension_mismatch(
                self.input_size,
                input.len(),
            ));
        }

        // Combine input and hidden state for reset and update gates
        let mut combined_input = input.to_vec();
        combined_input.extend_from_slice(&self.hidden_state);

        let reset_values = self.reset_gate.run(&combined_input)?;
        let update_values = self.update_gate.run(&combined_input)?;

        // Create reset hidden state for new gate
        let reset_hidden: Vec<T> = reset_values
            .iter()
            .zip(self.hidden_state.iter())
            .map(|(&r, &h)| r * h)
            .collect();

        let mut new_input = input.to_vec();
        new_input.extend_from_slice(&reset_hidden);

        let new_values = self.new_gate.run(&new_input)?;

        // Update hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        for i in 0..self.hidden_size {
            self.hidden_state[i] = (T::one() - update_values[i]) * new_values[i]
                + update_values[i] * self.hidden_state[i];
        }

        Ok(self.hidden_state.clone())
    }

    fn forward_sequence(&mut self, sequence: &[Vec<T>]) -> NeuroDivergentResult<Vec<Vec<T>>> {
        let mut outputs = Vec::with_capacity(sequence.len());

        for input in sequence {
            let output = self.forward_step(input)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    fn reset_state(&mut self) {
        self.hidden_state.fill(T::zero());
    }

    fn get_state(&self) -> Vec<T> {
        self.hidden_state.clone()
    }

    fn set_state(&mut self, state: Vec<T>) -> NeuroDivergentResult<()> {
        if state.len() != self.hidden_size {
            return Err(NeuroDivergentError::dimension_mismatch(
                self.hidden_size,
                state.len(),
            ));
        }
        self.hidden_state = state;
        Ok(())
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn clone_layer(&self) -> Box<dyn RecurrentLayer<T> + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Multi-layer recurrent network
pub struct MultiLayerRecurrent<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>
{
    layers: Vec<Box<dyn RecurrentLayer<T> + Send + Sync>>,
    dropout_rate: T,
    training_mode: bool,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> Clone
    for MultiLayerRecurrent<T>
{
    fn clone(&self) -> Self {
        let cloned_layers = self
            .layers
            .iter()
            .map(|layer| layer.clone_layer())
            .collect();

        Self {
            layers: cloned_layers,
            dropout_rate: self.dropout_rate,
            training_mode: self.training_mode,
        }
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> MultiLayerRecurrent<T> {
    /// Create a new multi-layer recurrent network
    pub fn new(layers: Vec<Box<dyn RecurrentLayer<T> + Send + Sync>>, dropout_rate: T) -> Self {
        Self {
            layers,
            dropout_rate,
            training_mode: true,
        }
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training_mode = training;
    }

    /// Apply dropout to a vector (simple implementation)
    fn apply_dropout(&self, input: &[T]) -> Vec<T> {
        if !self.training_mode || self.dropout_rate == T::zero() {
            return input.to_vec();
        }

        input
            .iter()
            .map(|&x| {
                if rand::random::<f64>() < self.dropout_rate.to_f64().unwrap_or(0.0) {
                    T::zero()
                } else {
                    x / (T::one() - self.dropout_rate)
                }
            })
            .collect()
    }

    /// Forward pass through all layers
    pub fn forward(&mut self, input: &[T]) -> NeuroDivergentResult<Vec<T>> {
        let mut current_input = input.to_vec();
        let num_layers = self.layers.len();

        for i in 0..num_layers {
            current_input = self.layers[i].forward_step(&current_input)?;

            // Apply dropout between layers (except last layer)
            if i < num_layers - 1 {
                current_input = self.apply_dropout(&current_input);
            }
        }

        Ok(current_input)
    }

    /// Forward pass through sequence
    pub fn forward_sequence(&mut self, sequence: &[Vec<T>]) -> NeuroDivergentResult<Vec<Vec<T>>> {
        let mut outputs = Vec::with_capacity(sequence.len());

        for input in sequence {
            let output = self.forward(input)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Reset all layer states
    pub fn reset_states(&mut self) {
        for layer in &mut self.layers {
            layer.reset_state();
        }
    }

    /// Get states from all layers
    pub fn get_states(&self) -> Vec<Vec<T>> {
        self.layers.iter().map(|layer| layer.get_state()).collect()
    }

    /// Set states for all layers
    pub fn set_states(&mut self, states: Vec<Vec<T>>) -> NeuroDivergentResult<()> {
        if states.len() != self.layers.len() {
            return Err(NeuroDivergentError::dimension_mismatch(
                self.layers.len(),
                states.len(),
            ));
        }

        for (layer, state) in self.layers.iter_mut().zip(states.iter()) {
            layer.set_state(state.clone())?;
        }

        Ok(())
    }

    /// Get the final hidden size
    pub fn output_size(&self) -> usize {
        self.layers
            .last()
            .map(|layer| layer.hidden_size())
            .unwrap_or(0)
    }
}

/// Bidirectional recurrent wrapper
pub struct BidirectionalRecurrent<
    T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static,
> {
    forward_layer: Box<dyn RecurrentLayer<T>>,
    backward_layer: Box<dyn RecurrentLayer<T>>,
    merge_mode: BidirectionalMergeMode,
}

#[derive(Debug, Clone, Copy)]
pub enum BidirectionalMergeMode {
    Concat,
    Sum,
    Average,
    Multiply,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>
    BidirectionalRecurrent<T>
{
    /// Create a new bidirectional recurrent layer
    pub fn new(
        forward_layer: Box<dyn RecurrentLayer<T>>,
        backward_layer: Box<dyn RecurrentLayer<T>>,
        merge_mode: BidirectionalMergeMode,
    ) -> Self {
        Self {
            forward_layer,
            backward_layer,
            merge_mode,
        }
    }

    /// Process sequence in both directions
    pub fn forward_sequence(&mut self, sequence: &[Vec<T>]) -> NeuroDivergentResult<Vec<Vec<T>>> {
        // Forward direction
        let forward_outputs = self.forward_layer.forward_sequence(sequence)?;

        // Backward direction (reverse sequence)
        let mut reversed_sequence = sequence.to_vec();
        reversed_sequence.reverse();
        let mut backward_outputs = self.backward_layer.forward_sequence(&reversed_sequence)?;
        backward_outputs.reverse(); // Reverse back to match forward order

        // Merge outputs
        let mut merged_outputs = Vec::with_capacity(sequence.len());
        for (forward, backward) in forward_outputs.iter().zip(backward_outputs.iter()) {
            let merged = self.merge_outputs(forward, backward)?;
            merged_outputs.push(merged);
        }

        Ok(merged_outputs)
    }

    /// Merge forward and backward outputs
    fn merge_outputs(&self, forward: &[T], backward: &[T]) -> NeuroDivergentResult<Vec<T>> {
        match self.merge_mode {
            BidirectionalMergeMode::Concat => {
                let mut result = forward.to_vec();
                result.extend_from_slice(backward);
                Ok(result)
            }
            BidirectionalMergeMode::Sum => {
                if forward.len() != backward.len() {
                    return Err(NeuroDivergentError::dimension_mismatch(
                        forward.len(),
                        backward.len(),
                    ));
                }
                Ok(forward
                    .iter()
                    .zip(backward.iter())
                    .map(|(&f, &b)| f + b)
                    .collect())
            }
            BidirectionalMergeMode::Average => {
                if forward.len() != backward.len() {
                    return Err(NeuroDivergentError::dimension_mismatch(
                        forward.len(),
                        backward.len(),
                    ));
                }
                let two = T::from(2.0).unwrap();
                Ok(forward
                    .iter()
                    .zip(backward.iter())
                    .map(|(&f, &b)| (f + b) / two)
                    .collect())
            }
            BidirectionalMergeMode::Multiply => {
                if forward.len() != backward.len() {
                    return Err(NeuroDivergentError::dimension_mismatch(
                        forward.len(),
                        backward.len(),
                    ));
                }
                Ok(forward
                    .iter()
                    .zip(backward.iter())
                    .map(|(&f, &b)| f * b)
                    .collect())
            }
        }
    }

    /// Reset both layers
    pub fn reset_states(&mut self) {
        self.forward_layer.reset_state();
        self.backward_layer.reset_state();
    }

    /// Get output size based on merge mode
    pub fn output_size(&self) -> usize {
        match self.merge_mode {
            BidirectionalMergeMode::Concat => {
                self.forward_layer.hidden_size() + self.backward_layer.hidden_size()
            }
            _ => self.forward_layer.hidden_size(), // Assuming same size for both layers
        }
    }
}

#[cfg(test)]
mod fann_integration_tests {
    use super::*;

    #[test]
    fn lstm_and_gru_cells_execute_with_checked_fann_gates() {
        let mut lstm = LSTMCell::<f64>::new(1, 3).unwrap();
        let mut gru = GRUCell::<f64>::new(1, 3).unwrap();
        for input in [0.25, 0.5, -0.25] {
            let output = lstm.forward_step(&[input]).unwrap();
            assert_eq!(output.len(), 3);
            assert!(output.iter().all(|value| value.is_finite()));
            let output = gru.forward_step(&[input]).unwrap();
            assert_eq!(output.len(), 3);
            assert!(output.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn gate_errors_are_returned_before_indexing_outputs_or_updating_state() {
        let mut lstm = LSTMCell::<f64>::new(1, 2).unwrap();
        lstm.forget_gate.layers.clear();
        let before = lstm.get_state();
        assert!(matches!(
            lstm.forward_step(&[0.5]),
            Err(NeuroDivergentError::NetworkError(
                ruv_fann::NetworkError::NoLayers
            ))
        ));
        assert_eq!(lstm.get_state(), before);

        let mut gru = GRUCell::<f64>::new(1, 2).unwrap();
        gru.update_gate.layers.clear();
        let before = gru.get_state();
        assert!(matches!(
            gru.forward_step(&[0.5]),
            Err(NeuroDivergentError::NetworkError(
                ruv_fann::NetworkError::NoLayers
            ))
        ));
        assert_eq!(gru.get_state(), before);
    }
}
