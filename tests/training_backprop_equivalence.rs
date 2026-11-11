//! Compare public training with the retained pre-optimization implementation.
//! The reference deliberately keeps connection searches and copied buffers.
use num_traits::Float;
use ruv_fann::{ActivationFunction as A, Network};
use std::fmt::Debug;
use std::ops::{AddAssign, MulAssign, SubAssign};

/// Forward pass that stores all layer outputs for backpropagation
fn reference_forward<T: Float>(network: &mut Network<T>, input: &[T]) -> Vec<Vec<T>> {
    let mut layer_outputs = Vec::with_capacity(network.layers.len());

    // Set input layer
    if !network.layers.is_empty() {
        let _ = network.layers[0].set_inputs(input);
        layer_outputs.push(network.layers[0].get_outputs());
    }

    // Forward propagate through each layer
    for i in 1..network.layers.len() {
        let prev_outputs = layer_outputs[i - 1].clone();
        network.layers[i].calculate(&prev_outputs);
        layer_outputs.push(network.layers[i].get_outputs());
    }

    layer_outputs
}

/// Backward pass - calculate gradients and update weights
fn reference_backward<T: Float>(
    network: &mut Network<T>,
    layer_outputs: &[Vec<T>],
    target: &[T],
    learning_rate: T,
) {
    if network.layers.is_empty() {
        return;
    }

    let num_layers = network.layers.len();
    let mut layer_errors = vec![Vec::new(); num_layers];

    // Calculate output layer errors
    if let Some(output_layer) = network.layers.last() {
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
        let current_layer = &network.layers[layer_idx];
        let next_layer = &network.layers[layer_idx + 1];
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
        let current_layer = &mut network.layers[layer_idx];

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

fn reference_train<T: Float>(
    network: &mut Network<T>,
    inputs: &[Vec<T>],
    targets: &[Vec<T>],
    learning_rate: f32,
    epochs: usize,
) {
    let rate = T::from(learning_rate as f64).unwrap();
    for _ in 0..epochs {
        for (input, target) in inputs.iter().zip(targets) {
            let values = reference_forward(network, input);
            reference_backward(network, &values, target, rate);
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Topology {
    Dense,
    Sparse,
    Reordered,
    Duplicates,
    EmptyRows,
    Biasless,
    Mixed,
}

fn fixture<T: Float>(shape: &[usize], topology: Topology) -> Network<T> {
    let mut network = Network::new(shape);
    let weights = (0..network.total_connections())
        .map(|index| T::from(((index * 17 % 43) as f64 - 21.0) / 128.0).unwrap())
        .collect::<Vec<_>>();
    network.set_weights(&weights).unwrap();
    let activations = [A::Sigmoid, A::Tanh, A::Linear, A::Elliot, A::Sin, A::Swish];
    for (layer_index, layer) in network.layers.iter_mut().enumerate().skip(1) {
        for (index, neuron) in layer.neurons.iter_mut().enumerate() {
            if !neuron.is_bias {
                neuron.activation_function = activations[(index + layer_index) % activations.len()];
                neuron.activation_steepness =
                    T::from(if index % 2 == 0 { 0.75 } else { -0.5 }).unwrap();
            }
            // Cached state must follow the same final forward pass too.
            neuron.sum = T::from(0.125).unwrap();
            neuron.value = T::from(-0.25).unwrap();
        }
    }
    if matches!(topology, Topology::Biasless) {
        for layer in &mut network.layers {
            layer.neurons.retain(|neuron| !neuron.is_bias);
        }
        for index in 1..network.layers.len() {
            let width = network.layers[index - 1].neurons.len();
            for neuron in &mut network.layers[index].neurons {
                neuron
                    .connections
                    .retain(|connection| connection.from_neuron < width);
            }
        }
    }
    for layer in network.layers.iter_mut().skip(1) {
        for (index, neuron) in layer.neurons.iter_mut().enumerate() {
            if neuron.is_bias {
                continue;
            }
            let edit = match topology {
                Topology::Mixed => [
                    Topology::Sparse,
                    Topology::Reordered,
                    Topology::Duplicates,
                    Topology::EmptyRows,
                    Topology::Dense,
                ][index % 5],
                other => other,
            };
            match edit {
                Topology::Sparse => {
                    neuron
                        .connections
                        .retain(|connection| (connection.from_neuron + index) % 3 != 1);
                }
                Topology::Reordered => neuron.connections.reverse(),
                Topology::Duplicates => {
                    // A different first duplicate makes using the later edge,
                    // summing duplicate weights, or unchecked indexing visible.
                    let mut duplicate = neuron.connections[0].clone();
                    duplicate.weight = T::from(0.375).unwrap();
                    neuron.connections.insert(0, duplicate);
                }
                Topology::EmptyRows if index % 2 == 0 => neuron.connections.clear(),
                _ => {}
            }
        }
    }
    network
}

fn assert_bits<T: Float + Debug>(actual: T, expected: T, context: &str) {
    // f32 to f64 is exact, including the sign of zero.
    assert_eq!(
        actual.to_f64().unwrap().to_bits(),
        expected.to_f64().unwrap().to_bits(),
        "{context}: {actual:?} != {expected:?}"
    );
}

fn assert_network_bits<T: Float + Debug>(actual: &Network<T>, expected: &Network<T>) {
    assert_bits(
        actual.connection_rate,
        expected.connection_rate,
        "connection rate",
    );
    assert_eq!(actual.layers.len(), expected.layers.len());
    for (layer_index, (actual, expected)) in actual.layers.iter().zip(&expected.layers).enumerate()
    {
        assert_eq!(actual.neurons.len(), expected.neurons.len());
        for (index, (actual, expected)) in actual.neurons.iter().zip(&expected.neurons).enumerate()
        {
            let context = format!("layer {layer_index}, neuron {index}");
            assert_eq!(actual.is_bias, expected.is_bias, "{context}");
            assert_eq!(
                actual.activation_function, expected.activation_function,
                "{context}"
            );
            assert_bits(
                actual.activation_steepness,
                expected.activation_steepness,
                &context,
            );
            assert_bits(actual.sum, expected.sum, &context);
            assert_bits(actual.value, expected.value, &context);
            assert_eq!(
                actual.connections.len(),
                expected.connections.len(),
                "{context}"
            );
            for (actual, expected) in actual.connections.iter().zip(&expected.connections) {
                assert_eq!(actual.from_neuron, expected.from_neuron, "{context}");
                assert_eq!(actual.to_neuron, expected.to_neuron, "{context}");
                assert_bits(actual.weight, expected.weight, &context);
            }
        }
    }
}

fn equivalence<T: Float + Debug + AddAssign + SubAssign + MulAssign>(topology: Topology) {
    // No hidden layer, multiple hidden layers, odd widths, and single-unit rows.
    for shape in [&[3, 2][..], &[5, 9, 7, 3][..], &[1, 1, 1, 1][..]] {
        let inputs = (0..5)
            .map(|sample| {
                (0..shape[0])
                    .map(|feature| {
                        T::from(((sample * 7 + feature * 3) % 17) as f64 / 16.0 - 0.5).unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let targets = (0..5)
            .map(|sample| {
                (0..*shape.last().unwrap())
                    .map(|output| {
                        T::from(((sample * 3 + output) % 11) as f64 / 16.0 - 0.25).unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for epochs in [0, 1, 4] {
            let mut actual = fixture::<T>(shape, topology);
            let mut expected = actual.clone();
            reference_train(&mut expected, &inputs, &targets, 0.0625, epochs);
            actual.train(&inputs, &targets, 0.0625, epochs).unwrap();
            assert_network_bits(&actual, &expected);
        }
    }
}

#[test]
fn dense_training_matches_original_f32() {
    equivalence::<f32>(Topology::Dense);
}
#[test]
fn dense_training_matches_original_f64() {
    equivalence::<f64>(Topology::Dense);
}
#[test]
fn sparse_training_matches_original_f32() {
    equivalence::<f32>(Topology::Sparse);
}
#[test]
fn sparse_training_matches_original_f64() {
    equivalence::<f64>(Topology::Sparse);
}
#[test]
fn reordered_training_matches_original_f32() {
    equivalence::<f32>(Topology::Reordered);
}
#[test]
fn reordered_training_matches_original_f64() {
    equivalence::<f64>(Topology::Reordered);
}
#[test]
fn duplicate_training_matches_original_f32() {
    equivalence::<f32>(Topology::Duplicates);
}
#[test]
fn duplicate_training_matches_original_f64() {
    equivalence::<f64>(Topology::Duplicates);
}
#[test]
fn empty_row_training_matches_original_f32() {
    equivalence::<f32>(Topology::EmptyRows);
}
#[test]
fn empty_row_training_matches_original_f64() {
    equivalence::<f64>(Topology::EmptyRows);
}
#[test]
fn biasless_training_matches_original_f32() {
    equivalence::<f32>(Topology::Biasless);
}
#[test]
fn biasless_training_matches_original_f64() {
    equivalence::<f64>(Topology::Biasless);
}
#[test]
fn mixed_training_matches_original_f32() {
    equivalence::<f32>(Topology::Mixed);
}
#[test]
fn mixed_training_matches_original_f64() {
    equivalence::<f64>(Topology::Mixed);
}

fn edited_topology<T: Float + Debug + AddAssign + SubAssign + MulAssign>() {
    let mut actual = fixture::<T>(&[3, 5, 3, 2], Topology::Dense);
    let mut expected = actual.clone();
    let inputs = vec![vec![T::from(0.25).unwrap(); 3]];
    let targets = vec![vec![T::from(0.5).unwrap(); 2]];
    for phase in 0..2 {
        reference_train(&mut expected, &inputs, &targets, 0.0625, 2);
        actual.train(&inputs, &targets, 0.0625, 2).unwrap();
        assert_network_bits(&actual, &expected);
        if phase == 0 {
            // A caller can invalidate a previous dense ordering between calls.
            // Use the same edit in each independently evolving network.
            for network in [&mut actual, &mut expected] {
                network.layers[2].neurons[0].connections.reverse();
            }
        }
    }
}

#[test]
fn topology_edited_between_training_calls_f32() {
    edited_topology::<f32>();
}

#[test]
fn topology_edited_between_training_calls_f64() {
    edited_topology::<f64>();
}
