//! Single-call dense tiles retain the mutable network's scalar arithmetic.
use num_traits::Float;
use ruv_fann::{ActivationFunction as A, Network, PreparedError};
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

fn equivalence<T: Float + Debug>() {
    for shape in [
        &[17, 4][..],
        &[33, 9, 5],
        &[63, 25, 7],
        &[64, 25, 7],
        &[65, 65, 7],
        &[67, 25, 7],
    ] {
        for topology in 0..4 {
            let mut network = Network::<T>::new(shape);
            let weights = (0..network.total_connections())
                .map(|i| T::from(((i * 17 % 31) as f64 - 15.0) / 64.0).unwrap())
                .collect::<Vec<_>>();
            network.set_weights(&weights).unwrap();
            if topology == 2 {
                for layer in network.layers.iter_mut().take(shape.len() - 1) {
                    assert!(layer.neurons.last().unwrap().is_bias);
                    layer.neurons.pop();
                }
                for index in 1..network.layers.len() {
                    let width = network.layers[index - 1].size();
                    for neuron in &mut network.layers[index].neurons {
                        neuron.connections.retain(|edge| edge.from_neuron < width);
                    }
                }
            }
            for (layer_index, layer) in network.layers.iter_mut().enumerate().skip(1) {
                for (index, neuron) in layer.neurons.iter_mut().filter(|n| !n.is_bias).enumerate() {
                    neuron.activation_function = ACTIVATIONS[index % ACTIVATIONS.len()];
                    neuron.activation_steepness =
                        T::from(if index % 2 == 0 { 0.75 } else { -1.25 }).unwrap();
                    if topology == 1 || (topology == 3 && layer_index == 1) {
                        match index {
                            0 => neuron.connections.clear(),
                            1 => neuron.connections.reverse(),
                            2 => neuron.connections.push(neuron.connections[0].clone()),
                            8 => {
                                neuron.connections.remove(0);
                            }
                            _ => (),
                        }
                    }
                }
            }
            let prepared = network.prepare().unwrap();
            let mut workspace = prepared.workspace().unwrap();
            let mut actual = vec![T::nan(); prepared.num_outputs()];
            for sample in 0..31 {
                let input = (0..shape[0])
                    .map(|i| T::from((((i + sample) * 7 % 17) as f64 - 8.0) / 8.0).unwrap())
                    .collect::<Vec<_>>();
                let expected = network.run(&input).unwrap();
                prepared
                    .run_into(&input, &mut actual, &mut workspace)
                    .unwrap();
                for (&a, &b) in actual.iter().zip(&expected) {
                    assert_eq!(
                        a.to_f64().unwrap().to_bits(),
                        b.to_f64().unwrap().to_bits(),
                        "shape={shape:?}, topology={topology}, sample={sample}"
                    );
                }
            }
        }
    }
}

#[test]
fn single_tiles_match_scalar_dense_mixed_and_biasless_f32() {
    equivalence::<f32>();
}
#[test]
fn single_tiles_match_scalar_dense_mixed_and_biasless_f64() {
    equivalence::<f64>();
}

fn overflow_priority<T: Float + Debug>() {
    let mut network = Network::<T>::new(&[65, 4]);
    network
        .set_weights(&vec![T::zero(); network.total_connections()])
        .unwrap();
    network.set_activation_function_output(A::Linear);
    network.set_activation_steepness_output(T::one());
    network.layers[1].neurons[0].connections[0].weight = T::one();
    network.layers[1].neurons[0].activation_steepness = T::max_value();
    network.layers[1].neurons[1].connections[1].weight = T::max_value();
    let prepared = network.prepare().unwrap();
    let mut workspace = prepared.workspace().unwrap();
    let mut input = vec![T::zero(); 65];
    input[0] = T::from(2).unwrap();
    input[1] = T::from(2).unwrap();
    let marker = T::from(42).unwrap();
    let mut output = [marker; 4];
    assert_eq!(
        prepared.run_into(&input, &mut output, &mut workspace),
        Err(PreparedError::NumericOverflow {
            layer: 1,
            neuron: 0
        })
    );
    assert_eq!(output, [marker; 4]);
    input.fill(T::zero());
    prepared
        .run_into(&input, &mut output, &mut workspace)
        .unwrap();
    assert_eq!(output, [T::zero(); 4]);
}

#[test]
fn single_tiles_keep_error_priority_and_recover_f32() {
    overflow_priority::<f32>();
}
#[test]
fn single_tiles_keep_error_priority_and_recover_f64() {
    overflow_priority::<f64>();
}

fn ordered_cancellation<T: Float + Debug>() {
    let mut network = Network::<T>::new(&[65, 4]);
    network.set_activation_function_output(A::Linear);
    network.set_activation_steepness_output(-T::one());
    network
        .set_weights(&vec![T::zero(); network.total_connections()])
        .unwrap();
    for neuron in &mut network.layers[1].neurons {
        for edge in neuron.connections.iter_mut().take(3) {
            edge.weight = T::one();
        }
    }
    let mut input = [T::zero(); 65];
    input[0] = T::from(1e20).unwrap();
    input[1] = T::one();
    input[2] = -input[0];
    let expected = network.run(&input).unwrap();
    assert_eq!(expected[0].to_f64().unwrap().to_bits(), (-0.0f64).to_bits());
    let prepared = network.prepare().unwrap();
    let mut workspace = prepared.workspace().unwrap();
    let mut output = [T::one(); 4];
    prepared
        .run_into(&input, &mut output, &mut workspace)
        .unwrap();
    for (&a, &b) in output.iter().zip(&expected) {
        assert_eq!(a.to_f64().unwrap().to_bits(), b.to_f64().unwrap().to_bits());
    }
}

#[test]
fn single_tiles_preserve_cancellation_and_signed_zero_f32() {
    ordered_cancellation::<f32>();
}
#[test]
fn single_tiles_preserve_cancellation_and_signed_zero_f64() {
    ordered_cancellation::<f64>();
}
