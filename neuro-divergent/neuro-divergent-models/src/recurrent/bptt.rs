//! Differentiable stacked vanilla RNN with a linear FANN projection.

use super::layers::{BasicRecurrentCell, RecurrentLayer};
use crate::config::RNNConfig;
use crate::errors::{NeuroDivergentError as Error, NeuroDivergentResult as Result};
use num_traits::Float;
use rand::Rng;
use ruv_fann::{ActivationFunction, Network};

#[derive(Clone)]
pub(super) struct RecurrentNetwork<
    T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static,
> {
    pub layers: Vec<BasicRecurrentCell<T>>,
    pub dropout: T,
}

pub(super) struct Tape<T> {
    inputs: Vec<Vec<Vec<T>>>,
    pub(super) states: Vec<Vec<Vec<T>>>, // initial state, then one state per time step
    masks: Vec<Vec<Vec<T>>>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> RecurrentNetwork<T> {
    pub fn new(config: &RNNConfig<T>, rng: &mut impl Rng) -> Self {
        let layers = (0..config.num_layers)
            .map(|layer| {
                BasicRecurrentCell::with_rng(
                    if layer == 0 {
                        config.input_size
                    } else {
                        config.hidden_size
                    },
                    config.hidden_size,
                    config.activation,
                    rng,
                )
            })
            .collect();
        Self {
            layers,
            dropout: config.dropout,
        }
    }

    /// Samples are independent: every pass starts at the configured initial states.
    /// Dropout is applied only between layers, with its exact mask retained for BPTT.
    pub fn forward(&self, sequence: &[Vec<T>], mut rng: Option<&mut impl Rng>) -> Result<Tape<T>> {
        if sequence.is_empty() {
            return Err(Error::data("RNN sequence cannot be empty"));
        }
        let mut tape = Tape {
            inputs: Vec::new(),
            states: Vec::new(),
            masks: Vec::new(),
        };
        let mut inputs = sequence.to_vec();
        for (index, layer) in self.layers.iter().enumerate() {
            let mut cell = layer.clone();
            let mut states = vec![cell.get_state()];
            let mut masks = Vec::with_capacity(sequence.len());
            let mut next = Vec::with_capacity(sequence.len());
            for input in &inputs {
                if input.iter().any(|v| !v.is_finite()) {
                    return Err(Error::data("Non-finite RNN input"));
                }
                let output = cell.forward_step(input)?;
                if output.iter().any(|v| !v.is_finite()) {
                    return Err(Error::training("RNN state overflowed"));
                }
                let mut mask = vec![T::one(); layer.hidden_size];
                if index + 1 < self.layers.len() && self.dropout > T::zero() {
                    if let Some(rng) = rng.as_mut() {
                        for value in &mut mask {
                            *value = if rng.gen::<f64>() < self.dropout.to_f64().unwrap() {
                                T::zero()
                            } else {
                                T::one() / (T::one() - self.dropout)
                            };
                        }
                    }
                }
                next.push(output.iter().zip(&mask).map(|(&h, &m)| h * m).collect());
                states.push(output);
                masks.push(mask);
            }
            tape.inputs.push(inputs);
            tape.states.push(states);
            tape.masks.push(masks);
            inputs = next;
        }
        Ok(tape)
    }

    pub fn final_hidden<'a>(&self, tape: &'a Tape<T>) -> &'a [T] {
        tape.states.last().unwrap().last().unwrap()
    }

    pub fn parameters(&self, projection: &Network<T>) -> Vec<Vec<T>> {
        let mut parameters: Vec<Vec<T>> = self
            .layers
            .iter()
            .map(|layer| {
                layer
                    .input_weights
                    .iter()
                    .flatten()
                    .chain(layer.hidden_weights.iter().flatten())
                    .chain(&layer.bias)
                    .copied()
                    .collect()
            })
            .collect();
        parameters.push(projection.get_weights());
        parameters
    }

    fn parameter_shapes(&self, projection: &Network<T>) -> Vec<usize> {
        let mut sizes: Vec<_> = self
            .layers
            .iter()
            .map(|layer| layer.hidden_size * (layer.input_size + layer.hidden_size + 1))
            .collect();
        sizes.push(projection.total_connections());
        sizes
    }

    pub fn set_parameters(
        &mut self,
        projection: &mut Network<T>,
        parameters: &[Vec<T>],
    ) -> Result<()> {
        let expected = self.parameter_shapes(projection);
        if expected.len() != parameters.len()
            || expected
                .iter()
                .zip(parameters)
                .any(|(a, b)| *a != b.len() || b.iter().any(|v| !v.is_finite()))
        {
            return Err(Error::state(
                "Invalid RNN parameter shapes or non-finite values",
            ));
        }
        for (layer, values) in self.layers.iter_mut().zip(parameters) {
            let mut values = values.iter().copied();
            for value in layer
                .input_weights
                .iter_mut()
                .flatten()
                .chain(layer.hidden_weights.iter_mut().flatten())
                .chain(&mut layer.bias)
            {
                *value = values.next().unwrap();
            }
        }
        projection.set_weights(parameters.last().unwrap())?;
        Ok(())
    }

    /// Exact MSE vector-Jacobian product. A truncation of K detaches *all* layer
    /// states preceding the final K time steps; None differentiates the full window.
    pub fn gradients(
        &self,
        tape: &Tape<T>,
        projection: &Network<T>,
        prediction: &[T],
        target: &[T],
        truncation: Option<usize>,
    ) -> Result<(T, Vec<Vec<T>>)> {
        validate_projection(
            projection,
            self.layers.last().unwrap().hidden_size,
            target.len(),
        )?;
        if prediction.len() != target.len() || target.is_empty() {
            return Err(Error::data("RNN target/prediction horizon mismatch"));
        }
        let mut gradients: Vec<Vec<T>> = self
            .parameter_shapes(projection)
            .iter()
            .map(|&size| vec![T::zero(); size])
            .collect();
        let steps = tape.inputs[0].len();
        let hidden = self.layers.last().unwrap().hidden_size;
        let mut upstream = vec![vec![T::zero(); hidden]; steps];
        let final_hidden = self.final_hidden(tape);
        let mut loss = T::zero();
        let divisor = T::from(target.len()).unwrap();
        for (output, (&prediction, &target)) in prediction.iter().zip(target).enumerate() {
            let error = prediction - target;
            loss = loss + error * error / divisor;
            let delta = (T::one() + T::one()) * error / divisor;
            let row = &projection.layers[1].neurons[output].connections;
            for input in 0..hidden {
                gradients.last_mut().unwrap()[output * (hidden + 1) + input] =
                    delta * final_hidden[input];
                upstream[steps - 1][input] = upstream[steps - 1][input] + delta * row[input].weight;
            }
            gradients.last_mut().unwrap()[output * (hidden + 1) + hidden] = delta;
        }
        let start = truncation.map_or(0, |limit| steps.saturating_sub(limit));
        for index in (0..self.layers.len()).rev() {
            let layer = &self.layers[index];
            let h = layer.hidden_size;
            let width = layer.input_size;
            let mut lower = vec![vec![T::zero(); width]; steps];
            let mut future = vec![T::zero(); h];
            for step in (start..steps).rev() {
                let mut previous = vec![T::zero(); h];
                for unit in 0..h {
                    let delta = (upstream[step][unit] + future[unit])
                        * layer.activation_derivative(tape.states[index][step + 1][unit]);
                    for input in 0..width {
                        let slot = unit * width + input;
                        gradients[index][slot] =
                            gradients[index][slot] + delta * tape.inputs[index][step][input];
                        lower[step][input] =
                            lower[step][input] + delta * layer.input_weights[unit][input];
                    }
                    for input in 0..h {
                        let slot = h * width + unit * h + input;
                        gradients[index][slot] =
                            gradients[index][slot] + delta * tape.states[index][step][input];
                        previous[input] =
                            previous[input] + delta * layer.hidden_weights[unit][input];
                    }
                    let slot = h * width + h * h + unit;
                    gradients[index][slot] = gradients[index][slot] + delta;
                }
                future = previous;
            }
            if index > 0 {
                for (row, mask) in lower.iter_mut().zip(&tape.masks[index - 1]) {
                    for (value, &mask) in row.iter_mut().zip(mask) {
                        *value = *value * mask;
                    }
                }
            }
            upstream = lower;
        }
        if !loss.is_finite() || gradients.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::training(
                "RNN loss or gradient overflowed; scale the data or lower the learning rate",
            ));
        }
        Ok((loss, gradients))
    }
}

/// External mutation through NetworkAdapter must not silently invalidate the VJP.
pub(super) fn validate_projection<T: Float>(
    network: &Network<T>,
    hidden: usize,
    horizon: usize,
) -> Result<()> {
    network.validate()?;
    if network.layers.len() != 2
        || network.num_inputs() != hidden
        || network.num_outputs() != horizon
    {
        return Err(Error::state(
            "RNN requires a two-layer hidden-to-horizon projection",
        ));
    }
    for neuron in network.layers[1]
        .neurons
        .iter()
        .filter(|neuron| !neuron.is_bias)
    {
        if neuron.activation_function != ActivationFunction::Linear
            || neuron.activation_steepness != T::one()
            || neuron.connections.len() != hidden + 1
            || neuron
                .connections
                .iter()
                .enumerate()
                .any(|(index, connection)| connection.from_neuron != index)
        {
            return Err(Error::state(
                "RNN requires an ordered, dense, unit-slope linear output projection",
            ));
        }
    }
    Ok(())
}
