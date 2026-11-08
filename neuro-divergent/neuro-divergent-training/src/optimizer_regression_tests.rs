use super::{Adam, AdamW, ForecastingAdam, Optimizer, RMSprop, SGD};

#[cfg(feature = "serde")]
#[test]
fn json_optimizer_state_restores_real_amsgrad_continuation() {
    let mut optimizer = Adam::new(0.01_f64, 0.9, 0.999).with_amsgrad(true);
    let mut parameters = vec![vec![1.0, 2.0]];
    let gradients = vec![vec![0.1, -0.2]];
    optimizer.step(&mut parameters, &gradients).unwrap();
    let encoded = serde_json::to_vec(&optimizer.state()).unwrap();
    let state: super::OptimizerState<f64> = serde_json::from_slice(&encoded).unwrap();
    let mut restored = Adam::new(0.01, 0.9, 0.999).with_amsgrad(true);
    restored.restore_state(state).unwrap();
    let mut resumed_parameters = parameters.clone();
    restored.step(&mut resumed_parameters, &gradients).unwrap();
    optimizer.step(&mut parameters, &gradients).unwrap();
    for (actual, expected) in resumed_parameters
        .iter()
        .flatten()
        .zip(parameters.iter().flatten())
    {
        assert!((actual - expected).abs() < 1e-14);
    }
}

#[test]
fn every_optimizer_rejects_invalid_nested_shapes_and_values_without_partial_changes() {
    let mut optimizers: Vec<Box<dyn Optimizer<f64>>> = vec![
        Box::new(Adam::new(0.01, 0.9, 0.999)),
        Box::new(AdamW::new(0.01, 0.9, 0.999, 0.1)),
        Box::new(SGD::new(0.01).with_momentum(0.9)),
        Box::new(RMSprop::new(0.01).with_momentum(0.5).with_centered(true)),
        Box::new(ForecastingAdam::new(0.01, 0.9, 0.999)),
    ];
    for optimizer in &mut optimizers {
        let mut parameters = vec![vec![1.0, 2.0], vec![3.0]];
        optimizer
            .step(&mut parameters, &[vec![0.1, -0.2], vec![0.3]])
            .unwrap();
        let original = parameters.clone();
        let state = optimizer.state();
        for gradients in [
            vec![vec![1.0, 1.0], vec![]],
            vec![vec![0.1, 0.2], vec![f64::NAN]],
        ] {
            assert!(
                optimizer.step(&mut parameters, &gradients).is_err(),
                "{}",
                optimizer.name()
            );
            assert_eq!(parameters, original);
            assert_eq!(optimizer.state().step_count, state.step_count);
            assert_eq!(optimizer.state().momentum_buffers, state.momentum_buffers);
            assert_eq!(optimizer.state().variance_buffers, state.variance_buffers);
        }
        optimizer.set_learning_rate(f64::INFINITY);
        assert!(optimizer
            .step(&mut parameters, &[vec![0.1, 0.2], vec![0.3]])
            .is_err());
        assert_eq!(parameters, original);
    }
}

#[test]
fn sgd_and_rmsprop_reset_allow_a_new_parameter_shape() {
    for mut optimizer in [
        Box::new(SGD::new(0.1).with_momentum(0.9)) as Box<dyn Optimizer<f64>>,
        Box::new(RMSprop::new(0.1).with_centered(true)),
    ] {
        optimizer
            .step(&mut [vec![1.0, 2.0]], &[vec![0.1, 0.2]])
            .unwrap();
        assert!(optimizer.step(&mut [vec![1.0]], &[vec![0.1]]).is_err());
        optimizer.reset();
        optimizer.step(&mut [vec![1.0]], &[vec![0.1]]).unwrap();
    }
}

#[test]
fn adam_bias_correction_keeps_epsilon_in_the_reference_position() {
    let mut optimizer = Adam::new(0.01_f64, 0.9, 0.999).with_epsilon(1e-8);
    let mut parameters = vec![vec![1.0]];
    let mut reference = 1.0;
    let mut momentum = 0.0;
    let mut variance = 0.0;
    for (index, gradient) in [1e-12_f64, -2e-11, 0.0, 1e-10].into_iter().enumerate() {
        momentum = 0.9 * momentum + 0.1 * gradient;
        variance = 0.999 * variance + 0.001 * gradient * gradient;
        let step = (index + 1) as i32;
        reference -= 0.01 * (momentum / (1.0 - 0.9_f64.powi(step)))
            / ((variance / (1.0 - 0.999_f64.powi(step))).sqrt() + 1e-8);
        optimizer.step(&mut parameters, &[vec![gradient]]).unwrap();
        assert!((parameters[0][0] - reference).abs() < 1e-14);
    }
}

#[test]
fn adam_rejects_invalid_controls_and_overflow_without_partial_updates() {
    for mut optimizer in [
        Adam::new(f64::NAN, 0.9, 0.999),
        Adam::new(0.01, 1.0, 0.999),
        Adam::new(0.01, 0.9, -0.1),
        Adam::new(0.01, 0.9, 0.999).with_epsilon(0.0),
    ] {
        let mut parameters = vec![vec![1.0, 2.0]];
        assert!(optimizer.step(&mut parameters, &[vec![0.1, 0.2]]).is_err());
        assert_eq!(parameters, vec![vec![1.0, 2.0]]);
        assert_eq!(optimizer.state().step_count, 0);
    }
    let mut optimizer = Adam::new(0.01_f64, 0.9, 0.999);
    let mut parameters = vec![vec![1.0, 2.0]];
    optimizer.step(&mut parameters, &[vec![0.1, 0.2]]).unwrap();
    let original = parameters.clone();
    let state = optimizer.state();
    assert!(optimizer
        .step(&mut parameters, &[vec![0.1, f64::MAX]])
        .is_err());
    assert_eq!(parameters, original);
    assert_eq!(optimizer.state().step_count, state.step_count);
    assert_eq!(optimizer.state().momentum_buffers, state.momentum_buffers);
    assert_eq!(optimizer.state().variance_buffers, state.variance_buffers);
}

#[test]
fn adamw_validates_before_applying_weight_decay() {
    let mut optimizer = AdamW::new(0.01_f64, 0.9, 0.999, 0.1);
    let mut parameters = vec![vec![1.0, 2.0]];
    assert!(optimizer.step(&mut parameters, &[vec![0.1]]).is_err());
    assert_eq!(parameters, vec![vec![1.0, 2.0]]);
    assert_eq!(optimizer.state().step_count, 0);
}

#[test]
fn enabling_amsgrad_after_updates_initializes_maxima() {
    let mut optimizer = Adam::new(0.01_f64, 0.9, 0.999);
    let mut parameters = vec![vec![1.0]];
    optimizer.step(&mut parameters, &[vec![0.1]]).unwrap();
    let mut optimizer = optimizer.with_amsgrad(true);
    optimizer.step(&mut parameters, &[vec![0.2]]).unwrap();
    assert!(parameters[0][0].is_finite());
    assert_eq!(optimizer.state().additional_state["max_variance"].len(), 1);
}

#[test]
fn amsgrad_restores_into_a_fresh_optimizer_and_replays_the_next_step() {
    let mut original = Adam::new(0.01_f64, 0.9, 0.999).with_amsgrad(true);
    let mut parameters = vec![vec![1.0, 2.0], vec![3.0]];
    let gradients = vec![vec![0.1, -0.2], vec![0.3]];
    original.step(&mut parameters, &gradients).unwrap();
    let state = original.state();
    let mut resumed_parameters = parameters.clone();
    original.step(&mut parameters, &gradients).unwrap();
    let mut resumed = Adam::new(0.01, 0.9, 0.999).with_amsgrad(true);
    resumed.restore_state(state).unwrap();
    resumed.step(&mut resumed_parameters, &gradients).unwrap();
    assert_eq!(resumed_parameters, parameters);
    assert_eq!(
        resumed.state().additional_state,
        original.state().additional_state
    );
}

#[test]
fn adam_rejects_ragged_or_nonfinite_gradients_without_updating_parameters() {
    let mut optimizer = Adam::new(0.01_f64, 0.9, 0.999);
    let initial = vec![vec![1.0, 2.0]];
    for gradients in [vec![vec![0.1]], vec![vec![0.1, f64::NAN]]] {
        let mut parameters = initial.clone();
        assert!(optimizer.step(&mut parameters, &gradients).is_err());
        assert_eq!(parameters, initial);
        assert_eq!(optimizer.state().step_count, 0);
    }
}

#[test]
fn malformed_checkpoint_is_rejected_without_losing_a_valid_state() {
    let mut optimizer = Adam::new(0.01_f64, 0.9, 0.999);
    optimizer.step(&mut [vec![1.0]], &[vec![0.2]]).unwrap();
    let valid = optimizer.state();
    let mut invalid = valid.clone();
    invalid.variance_buffers[0].clear();
    assert!(optimizer.restore_state(invalid).is_err());
    assert_eq!(optimizer.state().momentum_buffers, valid.momentum_buffers);
    assert_eq!(optimizer.state().step_count, valid.step_count);
}
