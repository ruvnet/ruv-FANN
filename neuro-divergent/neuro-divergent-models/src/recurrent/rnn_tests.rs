use super::*;
use crate::foundation::{DatasetMetadata, TimeSeriesSample};

fn dataset(inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>) -> TimeSeriesDataset<f64> {
    TimeSeriesDataset {
        samples: inputs
            .into_iter()
            .zip(targets)
            .map(|(input, target)| TimeSeriesSample {
                input: TimeSeriesInput::new(input),
                target,
                weight: None,
            })
            .collect(),
        metadata: DatasetMetadata {
            name: None,
            description: None,
            num_series: 1,
            total_samples: 0,
            feature_names: Vec::new(),
            target_names: Vec::new(),
        },
    }
}

fn loss(model: &RNN<f64>, sequence: &[Vec<f64>], target: &[f64]) -> f64 {
    let hidden = model.encode(sequence).unwrap();
    math::mse(&model.output_layer.clone().run(&hidden).unwrap(), target).unwrap()
}

#[test]
fn full_bptt_all_parameters_match_finite_differences_for_stacked_multistep_rnn() {
    let mut config = RNNConfig::<f64>::default_with_horizon(3);
    config.input_size = 2;
    config.hidden_size = 2;
    config.num_layers = 2;
    let mut model = RNN::new_seeded(config, 42).unwrap();
    RecurrentState::set_state(&mut model, vec![0.13, -0.17, 0.09, -0.03]).unwrap();
    let sequence = vec![vec![0.2, -0.1], vec![-0.4, 0.3], vec![0.1, 0.5]];
    let target = vec![0.4, -0.2, 0.7];
    let tape = model
        .recurrent
        .forward(&sequence, None::<&mut ChaCha8Rng>)
        .unwrap();
    let prediction = model
        .output_layer
        .clone()
        .run(model.recurrent.final_hidden(&tape))
        .unwrap();
    let (_, analytical) = model
        .recurrent
        .gradients(&tape, &model.output_layer, &prediction, &target, None)
        .unwrap();
    let parameters = model.recurrent.parameters(&model.output_layer);
    let epsilon = 1e-6;
    for layer in 0..parameters.len() {
        for parameter in 0..parameters[layer].len() {
            let mut perturbed = parameters.clone();
            perturbed[layer][parameter] += epsilon;
            model
                .recurrent
                .set_parameters(&mut model.output_layer, &perturbed)
                .unwrap();
            let plus = loss(&model, &sequence, &target);
            perturbed[layer][parameter] -= 2.0 * epsilon;
            model
                .recurrent
                .set_parameters(&mut model.output_layer, &perturbed)
                .unwrap();
            let minus = loss(&model, &sequence, &target);
            let numerical = (plus - minus) / (2.0 * epsilon);
            assert!(
                (analytical[layer][parameter] - numerical).abs() < 2e-8 + numerical.abs() * 1e-5,
                "layer={layer} parameter={parameter} analytical={} numerical={numerical}",
                analytical[layer][parameter]
            );
        }
    }
}

#[test]
fn truncated_bptt_detaches_the_prefix_without_changing_the_forward_prediction() {
    let mut config = RNNConfig::<f64>::default_with_horizon(1);
    config.hidden_size = 1;
    config.activation = ActivationFunction::Linear;
    let mut model = RNN::new_seeded(config, 4).unwrap();
    model
        .recurrent
        .set_parameters(
            &mut model.output_layer,
            &[vec![0.7, 0.5, 0.1], vec![0.9, 0.2]],
        )
        .unwrap();
    let sequence = vec![vec![0.2], vec![0.4], vec![0.6]];
    let tape = model
        .recurrent
        .forward(&sequence, None::<&mut ChaCha8Rng>)
        .unwrap();
    let prediction = model
        .output_layer
        .clone()
        .run(model.recurrent.final_hidden(&tape))
        .unwrap();
    assert!((prediction[0] - 0.893).abs() < 1e-12);
    let (_, truncated) = model
        .recurrent
        .gradients(&tape, &model.output_layer, &prediction, &[0.1], Some(1))
        .unwrap();
    let delta = 2.0 * (0.893 - 0.1) * 0.9;
    for (actual, expected) in truncated[0].iter().zip([delta * 0.6, delta * 0.5, delta]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    let (_, full) = model
        .recurrent
        .gradients(&tape, &model.output_layer, &prediction, &[0.1], None)
        .unwrap();
    let (_, untruncated) = model
        .recurrent
        .gradients(&tape, &model.output_layer, &prediction, &[0.1], Some(3))
        .unwrap();
    assert_eq!(full, untruncated);
    assert_ne!(full[0], truncated[0]);
    assert_eq!(full[1], truncated[1]);
    assert!(model.set_bptt_truncation(Some(0)).is_err());
}

#[test]
fn rnn_learns_multistep_forecasts_with_independent_hidden_and_horizon_sizes() {
    let mut config = RNNConfig::<f64>::default_with_horizon(3);
    config.hidden_size = 5;
    config.max_steps = 400;
    config.learning_rate = 0.02;
    config.gradient_clip_val = Some(1.0);
    let amplitudes: Vec<f64> = (-4..=4).map(|i| i as f64 / 5.0).collect();
    let data = dataset(
        amplitudes
            .iter()
            .map(|a| (0..3).map(|t| a * 0.8_f64.powi(t)).collect())
            .collect(),
        amplitudes
            .iter()
            .map(|a| (3..6).map(|t| a * 0.8_f64.powi(t)).collect())
            .collect(),
    );
    let mut model = RNN::new_seeded(config, 17).unwrap();
    let before = model.recurrent.parameters(&model.output_layer);
    let metrics = model.fit(&data).unwrap();
    eprintln!(
        "RNN AR(0.8), hidden=5 horizon=3: initial MSE={:.8}, trained MSE={:.8}",
        model.model_metrics()["initial_loss"],
        metrics.final_loss
    );
    assert!(metrics.final_loss < 1e-4, "loss={}", metrics.final_loss);
    assert!(metrics.final_loss < model.model_metrics()["initial_loss"] * 0.02);
    assert_eq!(metrics.epochs_completed, 400);
    assert!(model
        .training_history()
        .unwrap()
        .epoch_losses
        .iter()
        .all(|v| v.is_finite()));
    let after = model.recurrent.parameters(&model.output_layer);
    let cell = &model.recurrent.layers[0];
    for range in [
        0..cell.hidden_size * cell.input_size,
        cell.hidden_size * cell.input_size..cell.hidden_size * (cell.input_size + cell.hidden_size),
    ] {
        assert!(range
            .into_iter()
            .any(|i| (before[0][i] - after[0][i]).abs() > 1e-6));
    }
    assert_ne!(before[1], after[1]);
    let input =
        TimeSeriesInput::new((0..3).map(|t| 0.37 * 0.8_f64.powi(t)).collect()).with_id("held-out");
    let output = model.predict(&input).unwrap();
    assert_eq!(output.forecasts.len(), 3);
    assert_eq!(output.unique_id.as_deref(), Some("held-out"));
    for (index, &value) in output.forecasts.iter().enumerate() {
        assert!((value - 0.37 * 0.8_f64.powi(index as i32 + 3)).abs() < 0.02);
    }
    let validation = model.validate(&data).unwrap();
    assert!((validation.mse - metrics.final_loss).abs() < 1e-14);
}

#[test]
fn invalid_training_data_preserves_a_fitted_model() {
    let mut config = RNNConfig::<f64>::default_with_horizon(2);
    config.hidden_size = 3;
    config.max_steps = 2;
    let mut model = RNN::new_seeded(config, 10).unwrap();
    let valid = dataset(vec![vec![0.1, 0.2]], vec![vec![0.3, 0.4]]);
    model.fit(&valid).unwrap();
    let saved = model.save_state().unwrap();
    let mut invalid = valid.clone();
    invalid.samples[0].target.pop();
    assert!(model.fit(&invalid).is_err());
    invalid = valid.clone();
    invalid.samples[0].input.historical_targets[0] = f64::NAN;
    assert!(model.fit(&invalid).is_err());
    invalid = valid.clone();
    invalid.samples[0].weight = Some(0.0);
    assert!(model.fit(&invalid).is_err());
    invalid = valid.clone();
    invalid.samples[0].weight = Some(-0.5);
    assert!(model.fit(&invalid).is_err());
    invalid = valid.clone();
    invalid.samples.clear();
    assert!(model.fit(&invalid).is_err());
    assert_eq!(model.save_state().unwrap(), saved);
}

#[test]
fn rnn_f32_multilayer_training_and_truncation_remain_finite_and_learn() {
    let mut config = RNNConfig::<f32>::default_with_horizon(2);
    config.hidden_size = 4;
    config.num_layers = 2;
    config.max_steps = 300;
    config.learning_rate = 0.02;
    config.gradient_clip_val = Some(1.0);
    let samples = (-4..=4)
        .map(|i| {
            let a = i as f32 / 5.0;
            TimeSeriesSample {
                input: TimeSeriesInput::new(vec![a, a * 0.8, a * 0.64]),
                target: vec![a * 0.512, a * 0.4096],
                weight: None,
            }
        })
        .collect();
    let data = TimeSeriesDataset {
        samples,
        metadata: dataset(Vec::new(), Vec::new()).metadata,
    };
    let mut model = RNN::new_seeded(config, 51).unwrap();
    model.set_bptt_truncation(Some(2)).unwrap();
    let metrics = model.fit(&data).unwrap();
    assert!(metrics.final_loss < 1e-3, "f32 loss={}", metrics.final_loss);
    assert!((metrics.final_loss as f64) < model.model_metrics()["initial_loss"] * 0.05);
    assert!(model
        .predict(&data.samples[0].input)
        .unwrap()
        .forecasts
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn sample_weights_are_used_and_zero_weight_examples_do_not_change_the_fit() {
    let mut config = RNNConfig::<f64>::default_with_horizon(2);
    config.hidden_size = 3;
    config.max_steps = 5;
    let original = RNN::new_seeded(config, 99).unwrap();
    let valid = dataset(vec![vec![0.1, 0.2]], vec![vec![0.3, -0.4]]);
    let mut weighted = valid.clone();
    weighted.samples.push(TimeSeriesSample {
        input: TimeSeriesInput::new(vec![9.0, -8.0]),
        target: vec![100.0, -999.0],
        weight: Some(0.0),
    });
    let mut first = original.clone();
    let mut second = original;
    first.fit(&valid).unwrap();
    second.fit(&weighted).unwrap();
    assert_eq!(
        first.recurrent.parameters(&first.output_layer),
        second.recurrent.parameters(&second.output_layer)
    );
}

#[test]
fn dropout_is_disabled_for_prediction_and_seeded_training_is_reproducible() {
    let mut config = RNNConfig::<f64>::default_with_horizon(2);
    config.hidden_size = 3;
    config.num_layers = 2;
    config.dropout = 0.4;
    config.max_steps = 5;
    let mut first = RNN::new_seeded(config.clone(), 72).unwrap();
    let mut second = RNN::new_seeded(config, 72).unwrap();
    let data = dataset(vec![vec![0.3, -0.2, 0.4]], vec![vec![-0.1, 0.2]]);
    first.fit(&data).unwrap();
    second.fit(&data).unwrap();
    assert_eq!(
        first.recurrent.parameters(&first.output_layer),
        second.recurrent.parameters(&second.output_layer)
    );
    let expected = first.predict(&data.samples[0].input).unwrap().forecasts;
    for _ in 0..8 {
        assert_eq!(
            first.predict(&data.samples[0].input).unwrap().forecasts,
            expected
        );
    }
}

#[test]
fn checkpoints_round_trip_parameters_configuration_initial_states_and_forecasts() {
    let mut config = RNNConfig::<f64>::default_with_horizon(2);
    config.input_size = 3;
    config.hidden_size = 4;
    config.num_layers = 2;
    config.max_steps = 3;
    config.static_features = Some(vec!["static".into()]);
    config.hist_exog_features = Some(vec!["known-history".into()]);
    let mut model = RNN::new_seeded(config, 123).unwrap();
    model.set_bptt_truncation(Some(2)).unwrap();
    RecurrentState::set_state(&mut model, vec![0.1; 8]).unwrap();
    let mut data = dataset(vec![vec![0.1, 0.2, -0.3]], vec![vec![0.4, 0.1]]);
    data.samples[0].input.static_features = Some(vec![0.7]);
    data.samples[0].input.historical_features = Some(vec![vec![0.4], vec![0.3], vec![0.2]]);
    model.fit(&data).unwrap();
    let saved = model.save_state().unwrap();
    let restored = RNN::<f64>::from_state(&saved).unwrap();
    assert_eq!(restored.save_state().unwrap(), saved);
    assert_eq!(
        restored.predict(&data.samples[0].input).unwrap().forecasts,
        model.predict(&data.samples[0].input).unwrap().forecasts
    );
    assert_eq!(restored.bptt_truncation(), Some(2));
    assert_eq!(
        restored.config_owned().static_features,
        model.config_owned().static_features
    );
    assert_eq!(
        restored.clone_state().get_state(),
        RecurrentState::get_state(&model)
    );
    assert!(restored.is_trained());
    for bytes in [&saved[..7], &saved[..saved.len() - 1], b"not a checkpoint"] {
        assert!(model.load_state(bytes).is_err());
        assert_eq!(model.save_state().unwrap(), saved);
    }
    let mut trailing = saved.clone();
    trailing.push(1);
    assert!(RNN::<f64>::from_state(&trailing).is_err());
    let mut huge = saved.clone();
    huge[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(RNN::<f64>::from_state(&huge).is_err());
}

#[test]
fn checkpoints_support_f32_and_reject_overflowing_scalar_conversion() {
    let mut config = RNNConfig::<f32>::default_with_horizon(2);
    config.hidden_size = 3;
    config.max_steps = 1;
    let model = RNN::new_seeded(config, 11).unwrap();
    let saved = model.save_state().unwrap();
    assert_eq!(
        RNN::<f32>::from_state(&saved)
            .unwrap()
            .save_state()
            .unwrap(),
        saved
    );
    let mut wide = RNN::<f64>::from_state(&saved).unwrap();
    let mut weights = wide.output_layer.get_weights();
    weights[0] = f64::MAX;
    wide.output_layer.set_weights(&weights).unwrap();
    assert!(RNN::<f32>::from_state(&wide.save_state().unwrap()).is_err());
}

#[test]
fn gated_model_names_and_bidirectional_mode_explicitly_report_unsupported_training() {
    let lstm =
        crate::recurrent::LSTM::<f64>::new(crate::config::LSTMConfig::default_with_horizon(2))
            .unwrap_err();
    let gru = crate::recurrent::GRU::<f64>::new(crate::config::GRUConfig::default_with_horizon(2))
        .unwrap_err();
    assert!(lstm
        .to_string()
        .contains("LSTM forecasting training is unsupported"));
    assert!(gru
        .to_string()
        .contains("GRU forecasting training is unsupported"));
    let mut model = RNN::<f64>::new_seeded(RNNConfig::default_with_horizon(2), 7).unwrap();
    assert!(model
        .enable_bidirectional(BidirectionalMergeMode::Concat)
        .is_err());
}

#[test]
fn corrupt_projection_is_rejected_before_prediction_or_training() {
    let mut config = RNNConfig::<f64>::default_with_horizon(1);
    config.hidden_size = 2;
    config.max_steps = 2;
    let mut model = RNN::new_seeded(config, 12).unwrap();
    let data = dataset(vec![vec![0.1, 0.2, 0.3]], vec![vec![0.4]]);
    model.fit(&data).unwrap();
    model.output_layer.layers.clear();
    assert!(matches!(
        model.predict(&data.samples[0].input),
        Err(Error::NetworkError(ruv_fann::NetworkError::NoLayers))
    ));
    assert!(model.fit(&data).is_err());
}

#[test]
fn rnn_configuration_rejects_invalid_training_controls_and_excessive_allocations() {
    for mutation in 0..8 {
        let mut config = RNNConfig::<f64>::default_with_horizon(2);
        match mutation {
            0 => config.input_size = 0,
            1 => config.learning_rate = f64::NAN,
            2 => config.learning_rate = 0.0,
            3 => config.weight_decay = -1.0,
            4 => config.dropout = 1.0,
            5 => config.gradient_clip_val = Some(f64::INFINITY),
            6 => config.hidden_size = usize::MAX,
            _ => config.activation = ActivationFunction::Gaussian,
        }
        assert!(RNN::new_seeded(config, 0).is_err());
    }
}
