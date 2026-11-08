use super::*;

fn forward(
    model: &RNN<f64>,
    sequence: &[Vec<f64>],
    seed: Option<u64>,
) -> super::super::bptt::Tape<f64> {
    let mut rng = seed.map(ChaCha8Rng::seed_from_u64);
    model.recurrent.forward(sequence, rng.as_mut()).unwrap()
}

fn numerical_check(mut model: RNN<f64>, sequence: &[Vec<f64>], target: &[f64], seed: Option<u64>) {
    let tape = forward(&model, sequence, seed);
    let predicted = model
        .output_layer
        .clone()
        .run(model.recurrent.final_hidden(&tape))
        .unwrap();
    let (_, analytical) = model
        .recurrent
        .gradients(&tape, &model.output_layer, &predicted, target, None)
        .unwrap();
    let parameters = model.recurrent.parameters(&model.output_layer);
    for layer in 0..parameters.len() {
        for index in 0..parameters[layer].len() {
            let mut values = parameters.clone();
            let mut losses = Vec::new();
            for offset in [1e-6, -1e-6] {
                values[layer][index] = parameters[layer][index] + offset;
                model
                    .recurrent
                    .set_parameters(&mut model.output_layer, &values)
                    .unwrap();
                let tape = forward(&model, sequence, seed);
                let predicted = model
                    .output_layer
                    .clone()
                    .run(model.recurrent.final_hidden(&tape))
                    .unwrap();
                losses.push(math::mse(&predicted, target).unwrap());
            }
            let numerical = (losses[0] - losses[1]) / 2e-6;
            assert!(
                (analytical[layer][index] - numerical).abs() < 2e-8 + numerical.abs() * 1e-5,
                "layer={layer} index={index} analytical={} numerical={numerical}",
                analytical[layer][index]
            );
        }
    }
}

#[test]
fn bptt_dropout_masks_are_differentiated_with_the_same_forward_mask() {
    let mut config = RNNConfig::<f64>::default_with_horizon(2);
    config.hidden_size = 3;
    config.num_layers = 2;
    config.dropout = 0.35;
    let model = RNN::new_seeded(config, 404).unwrap();
    numerical_check(
        model,
        &[vec![0.3], vec![-0.5], vec![0.2]],
        &[0.4, -0.3],
        Some(21),
    );
}

#[test]
fn bptt_supported_activations_match_numerical_gradients() {
    for activation in [
        ActivationFunction::Linear,
        ActivationFunction::Sigmoid,
        ActivationFunction::SigmoidSymmetric,
        ActivationFunction::ReLU,
        ActivationFunction::ReLULeaky,
        ActivationFunction::LeakyRelu,
    ] {
        let mut config = RNNConfig::<f64>::default_with_horizon(2);
        config.hidden_size = 2;
        config.activation = activation;
        let model = RNN::new_seeded(config, 94).unwrap();
        numerical_check(
            model,
            &[vec![0.3], vec![-0.2], vec![0.4]],
            &[0.2, -0.7],
            None,
        );
    }
}

#[test]
fn truncated_stacked_bptt_matches_finite_differences_at_a_detached_boundary() {
    let mut config = RNNConfig::<f64>::default_with_horizon(3);
    config.hidden_size = 2;
    config.num_layers = 2;
    let model = RNN::new_seeded(config, 78).unwrap();
    let sequence = vec![vec![0.3], vec![-0.5], vec![0.2], vec![0.1]];
    let tape = forward(&model, &sequence, None);
    let predicted = model
        .output_layer
        .clone()
        .run(model.recurrent.final_hidden(&tape))
        .unwrap();
    let (_, truncated) = model
        .recurrent
        .gradients(
            &tape,
            &model.output_layer,
            &predicted,
            &[0.1, 0.2, -0.3],
            Some(2),
        )
        .unwrap();
    let mut detached = model.clone();
    for (layer, states) in detached.recurrent.layers.iter_mut().zip(&tape.states) {
        layer.set_state(states[2].clone()).unwrap();
    }
    let suffix = &sequence[2..];
    let suffix_tape = forward(&detached, suffix, None);
    let suffix_prediction = detached
        .output_layer
        .clone()
        .run(detached.recurrent.final_hidden(&suffix_tape))
        .unwrap();
    assert_eq!(predicted, suffix_prediction);
    let (_, gradients) = detached
        .recurrent
        .gradients(
            &suffix_tape,
            &detached.output_layer,
            &suffix_prediction,
            &[0.1, 0.2, -0.3],
            None,
        )
        .unwrap();
    assert_eq!(gradients, truncated);
    numerical_check(detached, suffix, &[0.1, 0.2, -0.3], None);
}
