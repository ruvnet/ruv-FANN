use num_traits::Float;
use ruv_fann::{
    prepared::BatchFloat, ActivationFunction as A, Connection, Network, Neuron, PreparedError,
};
use std::{fmt::Debug, sync::Arc};

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
        .map(|i| T::from(((i * 7 % 19) as f64 - 9.0) / 23.0).unwrap())
        .collect::<Vec<_>>();
    network.set_weights(&weights).unwrap();
    network
}

fn equivalence<T: BatchFloat + Debug>() {
    for activation in ACTIVATIONS {
        for sparse in [false, true] {
            let mut network = fixed::<T>(&[5, 7, 3]);
            network.set_activation_function_hidden(activation);
            network.set_activation_function_output(activation);
            network.set_activation_steepness_hidden(T::from(0.7).unwrap());
            network.set_activation_steepness_output(T::from(1.3).unwrap());
            if sparse {
                for layer in network.layers.iter_mut().skip(1) {
                    for neuron in layer.neurons.iter_mut().filter(|n| !n.is_bias) {
                        neuron.connections.reverse();
                        neuron.connections.remove(1);
                        neuron.connections.push(neuron.connections[0].clone());
                    }
                }
                network.layers[1].neurons[2].connections.clear();
            }
            let prepared = network.prepare().unwrap();
            let needed = prepared.batch_workspace_bytes(17).unwrap();
            let mut workspace = prepared.batch_workspace(17, needed).unwrap();
            assert_eq!(workspace.max_rows(), 17);
            assert_eq!(workspace.estimated_bytes(), needed);
            assert_eq!(prepared.connection_count(), network.total_connections());
            assert!(prepared.snapshot_bytes() < prepared.estimated_bytes());
            for rows in [1, 2, 3, 4, 5, 7, 16, 17, 3, 1] {
                let inputs = (0..rows * 5)
                    .map(|i| T::from(((i * 11 % 29) as f64 - 14.0) / 17.0).unwrap())
                    .collect::<Vec<_>>();
                let mut expected = Vec::new();
                for input in inputs.chunks_exact(5) {
                    expected.extend(network.run(input).unwrap());
                }
                let mut output = vec![T::nan(); rows * 3];
                prepared
                    .run_batch_into(&inputs, &mut output, rows, &mut workspace)
                    .unwrap();
                assert_eq!(
                    output, expected,
                    "{activation:?}, sparse={sparse}, rows={rows}"
                );
            }
        }
    }
}

#[test]
fn all_activations_dense_sparse_odd_batches_f32() {
    equivalence::<f32>();
}
#[test]
fn all_activations_dense_sparse_odd_batches_f64() {
    equivalence::<f64>();
}

#[test]
fn duplicate_order_cancellation_is_not_reassociated_across_samples() {
    let mut network = fixed::<f64>(&[2, 1]);
    let neuron = &mut network.layers[1].neurons[0];
    neuron.activation_function = A::Linear;
    neuron.activation_steepness = 1.0;
    neuron.connections = vec![
        Connection::new(0, 0, 1e20),
        Connection::new(1, 0, 1.0),
        Connection::new(0, 0, -1e20),
    ];
    let prepared = network.prepare().unwrap();
    let mut workspace = prepared.batch_workspace(7, 4096).unwrap();
    let mut outputs = [9.0; 7];
    prepared
        .run_batch_into(&[1.0; 14], &mut outputs, 7, &mut workspace)
        .unwrap();
    assert_eq!(outputs, [0.0; 7]);
}

#[test]
fn biasless_output_bias_and_input_only_identity_preserve_layout() {
    let mut network = fixed::<f64>(&[3, 5, 2]);
    for layer in network.layers.iter_mut().take(2) {
        layer.neurons.pop();
    }
    for index in 1..network.layers.len() {
        let width = network.layers[index - 1].size();
        for neuron in &mut network.layers[index].neurons {
            neuron.connections.retain(|c| c.from_neuron < width);
        }
    }
    network
        .layers
        .last_mut()
        .unwrap()
        .neurons
        .push(Neuron::new_bias());
    let prepared = network.prepare().unwrap();
    let inputs = [0.3, -0.2, 0.8, -0.7, 0.1, 0.9, 0.0, 0.2, 0.4];
    let mut workspace = prepared.batch_workspace(5, 4096).unwrap();
    let mut outputs = [0.0; 6];
    prepared
        .run_batch_into(&inputs, &mut outputs, 3, &mut workspace)
        .unwrap();
    for (input, output) in inputs.chunks_exact(3).zip(outputs.chunks_exact(2)) {
        assert_eq!(output, network.run(input).unwrap());
    }
    network.layers.truncate(1);
    let identity = network.prepare().unwrap();
    let mut outputs = [0.0; 9];
    identity
        .run_batch_into(&inputs, &mut outputs, 3, &mut workspace)
        .unwrap();
    assert_eq!(outputs, inputs);
}

#[test]
fn late_bad_rows_preserve_the_entire_output_and_workspace_recovers() {
    let mut network = fixed::<f32>(&[2, 2]);
    network.set_activation_function_output(A::Linear);
    network.set_activation_steepness_output(1.0);
    network
        .set_weights(&[1.0, 0.0, 0.0, f32::MAX, f32::MAX, 0.0])
        .unwrap();
    let prepared = network.prepare().unwrap();
    let mut workspace = prepared.batch_workspace(5, 4096).unwrap();
    let mut outputs = [17.0; 6];
    for bad in [f32::NAN, f32::INFINITY, 2.0] {
        let inputs = [0.0, 0.0, 0.0, 0.0, bad, 2.0];
        assert!(prepared
            .run_batch_into(&inputs, &mut outputs, 3, &mut workspace)
            .is_err());
        assert_eq!(outputs, [17.0; 6]);
    }
    prepared
        .run_batch_into(&[0.0; 6], &mut outputs, 3, &mut workspace)
        .unwrap();
    assert_eq!(outputs, [0.0; 6]);
    network.layers[1].neurons[0].activation_steepness = f32::MAX;
    let prepared = network.prepare().unwrap();
    assert!(prepared
        .run_batch_into(
            &[0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            &mut outputs,
            3,
            &mut workspace
        )
        .is_err());
    assert_eq!(outputs, [0.0; 6]);
}

#[test]
fn checked_budgets_lengths_capacity_and_workspace_compatibility() {
    let prepared = fixed::<f64>(&[5, 9, 3]).prepare().unwrap();
    assert!(prepared.batch_workspace_bytes(0).is_err());
    assert_eq!(
        prepared.batch_workspace_bytes(usize::MAX),
        Err(PreparedError::SizeOverflow)
    );
    let bytes = prepared.batch_workspace_bytes(7).unwrap();
    assert!(prepared.batch_workspace(7, bytes - 1).is_err());
    let mut workspace = prepared.batch_workspace(7, bytes).unwrap();
    let mut output = [23.0; 9];
    assert!(prepared
        .run_batch_into(&[0.0; 14], &mut output, 3, &mut workspace)
        .is_err());
    assert_eq!(output, [23.0; 9]);
    assert!(prepared
        .run_batch_into(&[0.0; 15], &mut output[..8], 3, &mut workspace)
        .is_err());
    assert_eq!(output, [23.0; 9]);
    assert!(prepared
        .run_batch_into(&[0.0; 40], &mut [0.0; 24], 8, &mut workspace)
        .is_err());
    let small = fixed::<f64>(&[1, 1]).prepare().unwrap();
    let mut small_workspace = small.batch_workspace(7, 4096).unwrap();
    assert!(matches!(
        prepared.run_batch_into(&[0.0; 15], &mut output, 3, &mut small_workspace),
        Err(PreparedError::WorkspaceTooSmall { .. })
    ));
    assert_eq!(output, [23.0; 9]);
    small
        .run_batch_into(&[0.1, 0.2, 0.3], &mut [0.0; 3], 3, &mut workspace)
        .unwrap();
    prepared
        .run_batch_into(&[], &mut [], 0, &mut small_workspace)
        .unwrap();
    assert!(prepared
        .run_batch_into(&[0.0], &mut [], 0, &mut workspace)
        .is_err());
}

#[test]
fn frozen_snapshot_is_shared_with_independent_thread_batch_workspaces() {
    let mut network = fixed::<f64>(&[3, 7, 2]);
    let prepared = Arc::new(network.prepare().unwrap());
    network.layers.clear();
    let threads = (0..4)
        .map(|worker| {
            let prepared = Arc::clone(&prepared);
            std::thread::spawn(move || {
                let mut reference = fixed::<f64>(&[3, 7, 2]);
                let mut workspace = prepared.batch_workspace(7, 4096).unwrap();
                for rows in [7, 1, 3, 5] {
                    let inputs = (0..rows * 3)
                        .map(|i| (i + worker) as f64 / 31.0)
                        .collect::<Vec<_>>();
                    let mut output = vec![0.0; rows * 2];
                    prepared
                        .run_batch_into(&inputs, &mut output, rows, &mut workspace)
                        .unwrap();
                    for (input, output) in inputs.chunks_exact(3).zip(output.chunks_exact(2)) {
                        assert_eq!(output, reference.run(input).unwrap());
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        thread.join().unwrap();
    }
}
