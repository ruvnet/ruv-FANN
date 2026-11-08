use num_traits::Float;
use ruv_fann::{
    ActivationFunction as A, Connection, Network, Neuron, PreparationLimits, PreparedError,
    PreparedNetwork,
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

fn equivalence<T: Float + Debug>() {
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
                // A disconnected row still evaluates its activation at zero.
                network.layers[1].neurons[2].connections.clear();
            }
            let prepared = network.prepare().unwrap();
            let mut workspace = prepared.workspace().unwrap();
            for sign in [-1.0, 1.0] {
                let input = (0..5)
                    .map(|i| T::from(sign * (i as f64 - 2.0) / 3.0).unwrap())
                    .collect::<Vec<_>>();
                let expected = network.run(&input).unwrap();
                let mut output = vec![T::nan(); 3];
                prepared
                    .run_into(&input, &mut output, &mut workspace)
                    .unwrap();
                assert_eq!(output, expected, "{activation:?}, sparse={sparse}");
            }
        }
    }
}

#[test]
fn all_activations_dense_sparse_odd_dimensions_f32() {
    equivalence::<f32>();
}

#[test]
fn all_activations_dense_sparse_odd_dimensions_f64() {
    equivalence::<f64>();
}

#[test]
fn duplicate_connections_keep_original_accumulation_order() {
    let mut network = fixed::<f32>(&[2, 1]);
    let neuron = &mut network.layers[1].neurons[0];
    neuron.activation_function = A::Linear;
    neuron.activation_steepness = 1.0;
    neuron.connections = vec![
        Connection::new(0, 0, 1e20),
        Connection::new(1, 0, 1.0),
        Connection::new(0, 0, -1e20),
    ];
    let prepared = network.prepare().unwrap();
    let mut output = [9.0];
    prepared
        .run_into(&[1.0, 1.0], &mut output, &mut prepared.workspace().unwrap())
        .unwrap();
    assert_eq!(output.as_slice(), network.run(&[1.0, 1.0]).unwrap());
    assert_eq!(output, [0.0]); // Combining duplicate weights would instead give 1.
}

#[test]
fn supports_biasless_layers_output_bias_and_single_layer_identity() {
    let mut network = fixed::<f64>(&[3, 5, 2]);
    for layer in network.layers.iter_mut().take(2) {
        layer.neurons.pop();
    }
    for layer_index in 1..network.layers.len() {
        let width = network.layers[layer_index - 1].size();
        for neuron in &mut network.layers[layer_index].neurons {
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
    let mut output = [0.0; 2];
    prepared
        .run_into(
            &[0.3, -0.2, 0.8],
            &mut output,
            &mut prepared.workspace().unwrap(),
        )
        .unwrap();
    assert_eq!(output.as_slice(), network.run(&[0.3, -0.2, 0.8]).unwrap());
    let mut identity = fixed::<f64>(&[3, 1]);
    identity.layers.truncate(1);
    let prepared = identity.prepare().unwrap();
    let mut output = [0.0; 3];
    prepared
        .run_into(
            &[0.2, -0.7, 0.9],
            &mut output,
            &mut prepared.workspace().unwrap(),
        )
        .unwrap();
    assert_eq!(output, [0.2, -0.7, 0.9]);
}

#[test]
fn snapshot_owns_parameters_and_ignores_transient_regular_caches() {
    let mut network = fixed::<f32>(&[3, 5, 2]);
    let expected = network.run(&[0.2, -0.7, 0.9]).unwrap();
    for layer in &mut network.layers {
        for neuron in layer.neurons.iter_mut().filter(|n| !n.is_bias) {
            neuron.sum = f32::NAN;
            neuron.value = f32::INFINITY;
        }
    }
    let prepared = network.prepare().unwrap();
    network.layers.clear();
    let mut output = [0.0; 2];
    prepared
        .run_into(
            &[0.2, -0.7, 0.9],
            &mut output,
            &mut prepared.workspace().unwrap(),
        )
        .unwrap();
    assert_eq!(output.as_slice(), expected);
}

#[test]
fn shares_snapshot_across_threads_with_independent_workspaces() {
    let prepared = Arc::new(fixed::<f64>(&[3, 7, 2]).prepare().unwrap());
    let threads = (0..4)
        .map(|worker| {
            let prepared = Arc::clone(&prepared);
            std::thread::spawn(move || {
                let mut reference = fixed::<f64>(&[3, 7, 2]);
                let mut workspace = prepared.workspace().unwrap();
                for step in 0..64 {
                    let input = [worker as f64 / 7.0, step as f64 / 100.0, -0.5];
                    let expected = reference.run(&input).unwrap();
                    let mut output = [0.0; 2];
                    prepared
                        .run_into(&input, &mut output, &mut workspace)
                        .unwrap();
                    assert_eq!(output.as_slice(), expected);
                }
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        thread.join().unwrap();
    }
}

#[test]
fn rejects_bad_topology_indices_parameters_and_noncanonical_bias() {
    type Mutation = fn(&mut Network<f32>);
    let mutations: &[Mutation] = &[
        |n| n.layers.clear(),
        |n| n.layers[1].neurons.clear(),
        |n| n.layers[0].neurons.last_mut().unwrap().value = 0.5,
        |n| n.layers[1].neurons.last_mut().unwrap().value = 2.0,
        |n| n.layers[0].neurons.swap(0, 2),
        |n| {
            n.layers[1]
                .neurons
                .last_mut()
                .unwrap()
                .connections
                .push(Connection::new(0, 0, 0.0))
        },
        |n| n.layers[1].neurons[0].connections[0].from_neuron = usize::MAX,
        |n| n.layers[1].neurons[0].connections[0].from_neuron = n.layers[0].size(),
        |n| n.layers[1].neurons[0].connections[0].weight = f32::NAN,
        |n| n.layers[1].neurons[0].activation_steepness = f32::INFINITY,
        |n| n.connection_rate = f32::NAN,
        |n| {
            n.layers[0].neurons[0]
                .connections
                .push(Connection::new(0, 0, 1.0))
        },
    ];
    for mutation in mutations {
        let mut network = fixed::<f32>(&[2, 3, 1]);
        mutation(&mut network);
        assert!(matches!(
            network.prepare(),
            Err(PreparedError::InvalidNetwork { .. })
        ));
    }
}

#[test]
fn enforces_storage_neuron_and_connection_budgets_before_packing() {
    let network = fixed::<f64>(&[3, 7, 2]);
    let required = network.prepare().unwrap().estimated_bytes();
    let mut limits = PreparationLimits {
        max_bytes: required - 1,
        ..PreparationLimits::default()
    };
    assert!(matches!(
        PreparedNetwork::from_network_with_limits(&network, limits),
        Err(PreparedError::BudgetExceeded {
            resource: "snapshot and workspace bytes",
            ..
        })
    ));
    limits.max_bytes = required;
    assert!(PreparedNetwork::from_network_with_limits(&network, limits).is_ok());
    limits.max_neurons = 1;
    assert!(matches!(
        PreparedNetwork::from_network_with_limits(&network, limits),
        Err(PreparedError::BudgetExceeded {
            resource: "neurons",
            ..
        })
    ));
    limits.max_neurons = usize::MAX;
    limits.max_connections = 0;
    assert!(matches!(
        PreparedNetwork::from_network_with_limits(&network, limits),
        Err(PreparedError::BudgetExceeded {
            resource: "connections",
            ..
        })
    ));
}

#[test]
fn errors_preserve_outputs_and_workspace_can_be_reused_after_overflow() {
    let mut network = fixed::<f32>(&[2, 2]);
    network.set_activation_function_output(A::Linear);
    network.set_activation_steepness_output(1.0);
    network
        .set_weights(&[1.0, 0.0, 0.0, f32::MAX, f32::MAX, 0.0])
        .unwrap();
    let prepared = network.prepare().unwrap();
    let mut workspace = prepared.workspace().unwrap();
    let mut output = [17.0, 19.0];
    for input in [
        &[1.0][..],
        &[f32::NAN, 0.0],
        &[0.0, f32::INFINITY],
        &[2.0, 2.0],
    ] {
        assert!(prepared
            .run_into(input, &mut output, &mut workspace)
            .is_err());
        assert_eq!(output, [17.0, 19.0]);
    }
    let mut wrong_output = [23.0];
    assert!(matches!(
        prepared.run_into(&[0.0, 0.0], &mut wrong_output, &mut workspace),
        Err(PreparedError::OutputSizeMismatch { .. })
    ));
    assert_eq!(wrong_output, [23.0]);
    prepared
        .run_into(&[0.0, 0.0], &mut output, &mut workspace)
        .unwrap();
    assert_eq!(output, [0.0, 0.0]);
    network.layers[1].neurons[0].activation_steepness = f32::MAX;
    let prepared = network.prepare().unwrap();
    assert!(matches!(
        prepared.run_into(&[2.0, 0.0], &mut output, &mut workspace),
        Err(PreparedError::NumericOverflow { neuron: 0, .. })
    ));
    assert_eq!(output, [0.0, 0.0]);
}

#[test]
fn accepts_larger_workspace_and_rejects_smaller_without_changing_output() {
    let large = fixed::<f32>(&[5, 9, 3]).prepare().unwrap();
    let small = fixed::<f32>(&[1, 1]).prepare().unwrap();
    let mut small_workspace = small.workspace().unwrap();
    let mut output = [7.0; 3];
    assert!(matches!(
        large.run_into(&[0.5; 5], &mut output, &mut small_workspace),
        Err(PreparedError::WorkspaceTooSmall { .. })
    ));
    assert_eq!(output, [7.0; 3]);
    let mut large_workspace = large.workspace().unwrap();
    large
        .run_into(&[0.5; 5], &mut output, &mut large_workspace)
        .unwrap();
    small
        .run_into(&[0.3], &mut [0.0], &mut large_workspace)
        .unwrap();
    let mut repeated = [0.0; 3];
    large
        .run_into(&[0.5; 5], &mut repeated, &mut large_workspace)
        .unwrap();
    assert_eq!(repeated, output);
}
