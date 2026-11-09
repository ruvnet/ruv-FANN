//! Public-API regressions for dense output/sample tiles and fallback boundaries.
use num_traits::Float;
use ruv_fann::{prepared::BatchFloat, ActivationFunction as A, Connection, Network, PreparedError};
use std::fmt::Debug;

const ACTIVATIONS: [A; 21] = [
    A::Linear,
    A::Threshold,
    A::ThresholdSymmetric,
    A::Sigmoid,
    A::SigmoidSymmetric,
    A::Tanh,
    A::Gaussian,
    A::GaussianSymmetric,
    A::Elliot,
    A::ElliotSymmetric,
    A::LinearPiece,
    A::LinearPieceSymmetric,
    A::ReLU,
    A::ReLULeaky,
    A::Sin,
    A::Cos,
    A::SinSymmetric,
    A::CosSymmetric,
    A::Gelu,
    A::Swish,
    A::LeakyRelu,
];

fn fixed<T: Float>(shape: &[usize]) -> Network<T> {
    let mut network = Network::new(shape);
    let weights = (0..network.total_connections())
        .map(|i| T::from(((i * 17 % 31) as f64 - 15.0) / 64.0).unwrap())
        .collect::<Vec<_>>();
    network.set_weights(&weights).unwrap();
    for layer in network.layers.iter_mut().skip(1) {
        for (index, neuron) in layer.neurons.iter_mut().filter(|n| !n.is_bias).enumerate() {
            neuron.activation_function = ACTIVATIONS[index % ACTIVATIONS.len()];
            neuron.activation_steepness =
                T::from(if index % 2 == 0 { 0.75 } else { -1.25 }).unwrap();
        }
    }
    network
}

fn assert_batch_bits<T: BatchFloat + Debug>(network: Network<T>) {
    let prepared = network.prepare().unwrap();
    // Larger capacity makes the scratch stride differ from active row counts.
    let bytes = prepared.batch_workspace_bytes(129).unwrap();
    let mut workspace = prepared.batch_workspace(129, bytes).unwrap();
    let mut single_workspace = prepared.workspace().unwrap();
    // Include both sides of the WASM (8/9) and native (32/33) dispatch cutoffs
    // plus larger batches, whose original streaming fallback must remain exact.
    for rows in [
        1, 2, 3, 4, 5, 7, 8, 9, 17, 19, 31, 32, 33, 64, 65, 128, 129, 4, 1,
    ] {
        let inputs = (0..rows * prepared.num_inputs())
            .map(|i| T::from(((i * 7 % 13) as f64 - 6.0) / 8.0).unwrap())
            .collect::<Vec<_>>();
        let mut expected = vec![T::nan(); rows * prepared.num_outputs()];
        for (input, output) in inputs
            .chunks_exact(prepared.num_inputs())
            .zip(expected.chunks_exact_mut(prepared.num_outputs()))
        {
            prepared
                .run_into(input, output, &mut single_workspace)
                .unwrap();
        }
        let mut actual = vec![T::nan(); expected.len()];
        prepared
            .run_batch_into(&inputs, &mut actual, rows, &mut workspace)
            .unwrap();
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            // f32 -> f64 is exact and retains the sign of zero.
            assert_eq!(
                actual.to_f64().unwrap().to_bits(),
                expected.to_f64().unwrap().to_bits(),
                "rows={rows}, index={index}, actual={actual:?}, expected={expected:?}"
            );
        }
    }
}

fn dense_equivalence<T: BatchFloat + Debug>() {
    // Include full tiles, output tails, odd feature widths, hidden/input biases,
    // and every activation inside groups that share the same source samples.
    for outputs in [4, 5, 7, 8, 9, 25] {
        assert_batch_bits(fixed::<T>(&[6, 9, outputs]));
    }
    assert_batch_bits(fixed::<T>(&[1, 4]));
    let mut biasless = fixed::<T>(&[5, 8, 7]);
    for layer in biasless.layers.iter_mut().take(2) {
        layer.neurons.pop();
    }
    for index in 1..biasless.layers.len() {
        let width = biasless.layers[index - 1].size();
        for neuron in &mut biasless.layers[index].neurons {
            neuron
                .connections
                .retain(|connection| connection.from_neuron < width);
        }
    }
    assert_batch_bits(biasless);
}

#[test]
fn dense_tiles_all_activations_biases_and_odd_tails_f32() {
    dense_equivalence::<f32>();
}

#[test]
fn dense_tiles_all_activations_biases_and_odd_tails_f64() {
    dense_equivalence::<f64>();
}

fn mixed_equivalence<T: BatchFloat + Debug>() {
    let mut network = fixed::<T>(&[5, 17, 9]);
    for layer in network.layers.iter_mut().skip(1) {
        for (index, neuron) in layer.neurons.iter_mut().filter(|n| !n.is_bias).enumerate() {
            match index {
                0 => {
                    neuron.connections.clear();
                }
                1 => {
                    neuron.connections.reverse();
                }
                2 => {
                    neuron.connections.push(neuron.connections[0].clone());
                }
                8 => {
                    neuron.connections.remove(0);
                }
                _ => {}
            }
        }
    }
    assert_batch_bits(network);
}

#[test]
fn mixed_dense_sparse_shuffled_duplicate_rows_f32() {
    mixed_equivalence::<f32>();
}

#[test]
fn mixed_dense_sparse_shuffled_duplicate_rows_f64() {
    mixed_equivalence::<f64>();
}

fn cancellation_and_zero<T: BatchFloat + Debug>() {
    let mut network = fixed::<T>(&[3, 9]);
    for (index, neuron) in network.layers[1].neurons.iter_mut().enumerate() {
        neuron.activation_function = A::Linear;
        neuron.activation_steepness = if index % 2 == 0 { T::one() } else { -T::one() };
        let huge = T::from(1e20).unwrap();
        neuron.connections = [huge, T::one(), -huge, -T::zero()]
            .into_iter()
            .enumerate()
            .map(|(source, weight)| Connection::new(source, index, weight))
            .collect();
    }
    let prepared = network.prepare().unwrap();
    let mut workspace = prepared.batch_workspace(9, 8192).unwrap();
    let mut output = [T::one(); 81];
    for rows in [7, 9] {
        prepared
            .run_batch_into(
                &[T::one(); 27][..rows * 3],
                &mut output[..rows * 9],
                rows,
                &mut workspace,
            )
            .unwrap();
        for row in output[..rows * 9].chunks_exact(9) {
            for (index, &value) in row.iter().enumerate() {
                let expected = if index % 2 == 0 { 0.0_f64 } else { -0.0_f64 };
                assert_eq!(value.to_f64().unwrap().to_bits(), expected.to_bits());
            }
        }
    }

    // (-1) + (1 + delta) * (1 - delta) rounds to zero with separate
    // multiplication/addition; an FMA would instead retain -delta squared.
    let exponent = if T::epsilon().to_f64().unwrap() < 1e-10 {
        -27
    } else {
        -13
    };
    let delta = T::from(2.0_f64.powi(exponent)).unwrap();
    let a = T::one() + delta;
    let b = T::one() - delta;
    assert!(a.mul_add(b, -T::one()) != T::zero());
    let mut network = fixed::<T>(&[2, 8]);
    for neuron in &mut network.layers[1].neurons {
        neuron.activation_function = A::Linear;
        neuron.activation_steepness = T::one();
        for (connection, weight) in neuron.connections.iter_mut().zip([-T::one(), b, T::zero()]) {
            connection.weight = weight;
        }
    }
    let prepared = network.prepare().unwrap();
    let mut workspace = prepared.batch_workspace(7, 8192).unwrap();
    let inputs = [T::one(), a].repeat(7);
    let mut output = [T::one(); 56];
    prepared
        .run_batch_into(&inputs, &mut output, 7, &mut workspace)
        .unwrap();
    assert!(output
        .iter()
        .all(|value| value.is_zero() && !value.is_sign_negative()));
}

#[test]
fn dense_tiles_preserve_cancellation_and_signed_zero_f32() {
    cancellation_and_zero::<f32>();
}

#[test]
fn dense_tiles_preserve_cancellation_and_signed_zero_f64() {
    cancellation_and_zero::<f64>();
}

fn error_priority<T: BatchFloat + Debug>() {
    let mut network = fixed::<T>(&[2, 8]);
    for neuron in &mut network.layers[1].neurons {
        neuron.activation_function = A::Linear;
        neuron.activation_steepness = T::one();
        for connection in &mut neuron.connections {
            connection.weight = T::zero();
        }
    }
    network.layers[1].neurons[0].connections[0].weight = T::one();
    network.layers[1].neurons[0].activation_steepness = T::max_value();
    network.layers[1].neurons[1].connections[1].weight = T::max_value();
    let prepared = network.prepare().unwrap();
    let mut workspace = prepared.batch_workspace(7, 8192).unwrap();
    let mut inputs = [T::zero(); 14];
    // Neuron 1 overflows at the first sample. Neuron 0's activation only
    // overflows in the last tail sample, but must remain the reported failure.
    inputs[1] = T::from(2).unwrap();
    inputs[12] = T::from(2).unwrap();
    let mut output = [-T::zero(); 56];
    assert_eq!(
        prepared.run_batch_into(&inputs, &mut output, 7, &mut workspace),
        Err(PreparedError::NumericOverflow {
            layer: 1,
            neuron: 0
        })
    );
    assert!(output
        .iter()
        .all(|value| value.is_zero() && value.is_sign_negative()));
    inputs.fill(T::zero());
    prepared
        .run_batch_into(&inputs, &mut output, 7, &mut workspace)
        .unwrap();
    assert!(output
        .iter()
        .all(|value| value.is_zero() && !value.is_sign_negative()));
}

#[test]
fn dense_tile_errors_keep_neuron_priority_atomicity_and_recover_f32() {
    error_priority::<f32>();
}

#[test]
fn dense_tile_errors_keep_neuron_priority_atomicity_and_recover_f64() {
    error_priority::<f64>();
}
