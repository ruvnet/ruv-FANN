use ruv_fann::{
    ActivationFunction as A, Connection, DenseLayer, Layer, Network, Neuron, PreparationLimits,
    PreparedError, PreparedNetwork,
};

fn layer<'a>(
    inputs: usize,
    outputs: usize,
    weights: &'a [f64],
    bias: Option<&'a [f64]>,
) -> DenseLayer<'a, f64> {
    DenseLayer {
        inputs,
        outputs,
        weights,
        bias,
        activation: A::Linear,
        steepness: 1.0,
    }
}

fn legacy(layers: &[DenseLayer<'_, f64>]) -> Network<f64> {
    let mut graph = Vec::new();
    for index in 0..=layers.len() {
        let width = if index == 0 {
            layers[0].inputs
        } else {
            layers[index - 1].outputs
        };
        let bias = layers.get(index).is_some_and(|next| next.bias.is_some());
        let mut nodes = Vec::new();
        for output in 0..width {
            let mut neuron = Neuron::new(A::Linear, 1.0);
            if index > 0 {
                let source = &layers[index - 1];
                neuron.activation_function = source.activation;
                neuron.activation_steepness = source.steepness;
                for input in 0..source.inputs {
                    neuron.connections.push(Connection::new(
                        input,
                        output,
                        source.weights[output * source.inputs + input],
                    ));
                }
                if let Some(values) = source.bias {
                    neuron
                        .connections
                        .push(Connection::new(source.inputs, output, values[output]));
                }
            }
            nodes.push(neuron);
        }
        if bias {
            nodes.push(Neuron::new_bias());
        }
        graph.push(Layer { neurons: nodes });
    }
    Network {
        layers: graph,
        connection_rate: 1.0,
    }
}

fn run(prepared: &PreparedNetwork<f64>, input: &[f64]) -> Vec<f64> {
    let mut output = vec![0.; prepared.num_outputs()];
    prepared
        .run_into(input, &mut output, &mut prepared.workspace().unwrap())
        .unwrap();
    output
}

#[test]
fn candle_out_in_orientation_and_optional_bias() {
    let no_bias =
        PreparedNetwork::from_dense_layers(&[layer(2, 3, &[1., 2., 3., 4., 5., 6.], None)])
            .unwrap();
    assert_eq!(run(&no_bias, &[10., 100.]), vec![210., 430., 650.]);
    let biased = PreparedNetwork::from_dense_layers(&[layer(
        2,
        3,
        &[1., 2., 3., 4., 5., 6.],
        Some(&[1., -2., 3.]),
    )])
    .unwrap();
    assert_eq!(run(&biased, &[10., 100.]), vec![211., 428., 653.]);
}

#[test]
fn stacked_mixed_biases_keep_legacy_order_and_partial_batches() {
    let w0 = (0..21).map(|i| (i as f64 - 10.) / 16.).collect::<Vec<_>>();
    let w1 = (0..14).map(|i| (i as f64 - 7.) / 8.).collect::<Vec<_>>();
    for first in [None, Some(&[0.2, -0.0, 0.1, -0.3, 0., 0.4, -0.2][..])] {
        for last in [None, Some(&[-0.0, 0.5][..])] {
            let mut layers = [layer(3, 7, &w0, first), layer(7, 2, &w1, last)];
            layers[0].activation = A::Tanh;
            layers[0].steepness = 0.75;
            layers[1].activation = A::Sigmoid;
            let packed = PreparedNetwork::from_dense_layers(&layers).unwrap();
            let reference = legacy(&layers).prepare().unwrap();
            assert_eq!(packed.connection_count(), reference.connection_count());
            assert_eq!(packed.estimated_bytes(), reference.estimated_bytes());
            let mut batch = packed.batch_workspace(9, 65536).unwrap();
            for rows in [1, 2, 3, 7, 9, 0, 5] {
                let inputs = (0..rows * 3)
                    .map(|i| (i as f64 - 4.) / 8.)
                    .collect::<Vec<_>>();
                let expected = inputs
                    .chunks_exact(3)
                    .flat_map(|input| run(&reference, input))
                    .collect::<Vec<_>>();
                let mut output = vec![777.; rows * 2];
                packed
                    .run_batch_into(&inputs, &mut output, rows, &mut batch)
                    .unwrap();
                assert_eq!(
                    output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn direct_pack_owns_parameters_after_the_source_is_dropped() {
    let packed = {
        let weights = vec![2f32, -1., 0.5, 3.];
        let biases = vec![1f32, -2.];
        PreparedNetwork::from_dense_layers(&[DenseLayer {
            inputs: 2,
            outputs: 2,
            weights: &weights,
            bias: Some(&biases),
            activation: A::ReLU,
            steepness: 1.,
        }])
        .unwrap()
    };
    let mut output = [0f32; 2];
    packed
        .run_into(&[3., 4.], &mut output, &mut packed.workspace().unwrap())
        .unwrap();
    assert_eq!(output, [3., 11.5]);
}

#[test]
fn bad_shapes_adjacency_values_and_overflows_reject() {
    assert!(PreparedNetwork::<f64>::from_dense_layers(&[]).is_err());
    for value in [
        layer(0, 1, &[], None),
        layer(2, 2, &[0.], None),
        layer(1, 1, &[1.], Some(&[])),
        layer(usize::MAX, 2, &[], None),
        layer(1, 1, &[f64::NAN], None),
        layer(1, 1, &[0.], Some(&[f64::INFINITY])),
    ] {
        assert!(PreparedNetwork::from_dense_layers(&[value]).is_err());
    }
    assert!(PreparedNetwork::from_dense_layers(&[
        layer(1, 2, &[1., 2.], None),
        layer(3, 1, &[1., 2., 3.], None)
    ])
    .is_err());
    let mut invalid = layer(1, 1, &[1.], None);
    invalid.steepness = f64::NAN;
    assert!(PreparedNetwork::from_dense_layers(&[invalid]).is_err());
}

#[test]
fn exact_requested_storage_and_topology_limits_are_enforced() {
    let layers = [layer(2, 2, &[1., 2., 3., 4.], Some(&[0., 1.]))];
    let packed = PreparedNetwork::from_dense_layers(&layers).unwrap();
    let mut limits = PreparationLimits::default();
    limits.max_bytes = packed.estimated_bytes();
    assert!(PreparedNetwork::from_dense_layers_with_limits(&layers, limits).is_ok());
    limits.max_bytes -= 1;
    assert!(matches!(
        PreparedNetwork::from_dense_layers_with_limits(&layers, limits),
        Err(PreparedError::BudgetExceeded { .. })
    ));
    let mut limits = PreparationLimits::default();
    limits.max_connections = 5;
    assert!(PreparedNetwork::from_dense_layers_with_limits(&layers, limits).is_err());
    limits = PreparationLimits::default();
    limits.max_neurons = 4;
    assert!(PreparedNetwork::from_dense_layers_with_limits(&layers, limits).is_err());
}

#[cfg(feature = "serde")]
#[test]
fn dense_json_schema_shape_and_float64_round_trip() {
    let json = r#"{"schema":"ruv-fann.dense-layers/1","layers":[{"inputs":2,"outputs":1,"weights":[2.0,-1.0],"bias":[0.25],"activation":"Linear","steepness":1.0}]}"#;
    let packed = PreparedNetwork::<f64>::from_dense_layers_json(json).unwrap();
    assert_eq!(run(&packed, &[3., 4.]), [2.25]);
    for bad in [
        json.replace("dense-layers/1", "dense-layers/9"),
        json.replace("\"inputs\":2", "\"inputs\":3"),
        json.replace("\"steepness\":1.0", "\"steepness\":1.0,\"unexpected\":true"),
        "{".to_owned(),
        " ".repeat(4 * 1024 * 1024 + 1),
    ] {
        assert!(PreparedNetwork::<f64>::from_dense_layers_json(&bad).is_err());
    }
    let value = 0.8455124082255701f64;
    let text = format!(
        r#"{{"schema":"ruv-fann.dense-layers/1","layers":[{{"inputs":1,"outputs":1,"weights":[{value:?}],"bias":null,"activation":"Linear","steepness":1.0}}]}}"#
    );
    assert_eq!(
        run(
            &PreparedNetwork::<f64>::from_dense_layers_json(&text).unwrap(),
            &[1.]
        )[0]
        .to_bits(),
        value.to_bits()
    );
}

#[test]
fn dense_constructor_caches_single_tiles_across_width_boundary() {
    for width in [63, 64, 65] {
        for biased in [false, true] {
            let first = (0..width * 65)
                .map(|i| ((i * 17 % 31) as f64 - 15.0) / 128.0)
                .collect::<Vec<_>>();
            let second = (0..65 * 5)
                .map(|i| ((i * 11 % 23) as f64 - 11.0) / 128.0)
                .collect::<Vec<_>>();
            let first_bias = vec![0.125; 65];
            let second_bias = vec![-0.25; 5];
            let layers = [
                layer(width, 65, &first, biased.then_some(first_bias.as_slice())),
                layer(65, 5, &second, biased.then_some(second_bias.as_slice())),
            ];
            let prepared = PreparedNetwork::from_dense_layers(&layers).unwrap();
            let mut scalar = legacy(&layers);
            for sample in 0..8 {
                let input = (0..width)
                    .map(|i| (((i + sample) * 7 % 17) as f64 - 8.0) / 8.0)
                    .collect::<Vec<_>>();
                let expected = scalar.run(&input).unwrap();
                let actual = run(&prepared, &input);
                for (a, b) in actual.iter().zip(expected) {
                    assert_eq!(a.to_bits(), b.to_bits(), "width={width}, biased={biased}");
                }
            }
        }
    }
}
