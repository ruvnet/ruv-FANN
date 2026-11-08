//! Optimizer updates, genuine checkpoint state and numerical loss gradients.
use neuro_divergent_training::{
    loss::{HuberLoss, LossFunction, MAELoss, MSELoss, PinballLoss},
    optimizer::{Adam, Optimizer, SGD},
};

#[test]
fn sgd_updates_parameters_with_the_actual_loss_gradient() {
    let loss = MSELoss::new();
    let mut parameter = vec![vec![2.0_f64]];
    let mut optimizer = SGD::new(0.1);
    for _ in 0..20 {
        let gradient = loss.backward(&parameter[0], &[0.]).unwrap();
        optimizer.step(&mut parameter, &[gradient]).unwrap();
    }
    assert!(loss.forward(&parameter[0], &[0.]).unwrap() < 0.001);
}

#[test]
fn adam_checkpoint_restoration_produces_identical_next_update() {
    let mut original = Adam::new(0.01_f64, 0.9, 0.999);
    let mut parameters = vec![vec![1., -2.]];
    let gradients = vec![vec![0.25, -0.5]];
    original.step(&mut parameters, &gradients).unwrap();
    let state = original.state();
    let mut restored = Adam::new(1., 0.9, 0.999);
    restored.restore_state(state).unwrap();
    let mut other = parameters.clone();
    original.step(&mut parameters, &gradients).unwrap();
    restored.step(&mut other, &gradients).unwrap();
    assert_eq!(parameters, other);
    assert_eq!(original.state().step_count, 2);
}

#[test]
fn invalid_optimizer_shape_does_not_partially_update_parameters() {
    let mut optimizer = SGD::new(0.1);
    let mut parameters = vec![vec![1., 2.], vec![3.]];
    let before = parameters.clone();
    assert!(optimizer
        .step(&mut parameters, &[vec![1., 1.], vec![]])
        .is_err());
    assert_eq!(parameters, before);
}

#[test]
fn supported_losses_match_independent_finite_difference_gradients() {
    let losses: Vec<Box<dyn LossFunction<f64>>> = vec![
        Box::new(MSELoss::new()),
        Box::new(MAELoss::new()),
        Box::new(HuberLoss::new(1.)),
        Box::new(PinballLoss::new(0.7)),
    ];
    let predictions = [0.2, 2.5, -1.5];
    let targets = [0.8, 1.2, -1.0];
    let eps = 1e-6;
    for loss in losses {
        let analytic = loss.backward(&predictions, &targets).unwrap();
        for i in 0..predictions.len() {
            let mut plus = predictions;
            let mut minus = predictions;
            plus[i] += eps;
            minus[i] -= eps;
            let expected = (loss.forward(&plus, &targets).unwrap()
                - loss.forward(&minus, &targets).unwrap())
                / (2. * eps);
            assert!(
                (analytic[i] - expected).abs() < 1e-8,
                "{} gradient {i}",
                loss.name()
            );
        }
    }
}

#[test]
fn losses_reject_empty_mismatched_and_nonfinite_inputs() {
    let loss = MSELoss::<f64>::new();
    assert!(loss.forward(&[], &[]).is_err());
    assert!(loss.backward(&[1.], &[]).is_err());
    assert!(loss.forward(&[f64::NAN], &[1.]).is_err());
}

#[test]
fn clipping_preserves_direction_and_limits_actual_gradient_norm() {
    let mut gradients = vec![vec![3e200_f64, 4e200]];
    let norm = neuro_divergent_training::utils::clip_gradients_by_norm(&mut gradients, 5.);
    assert!((norm / 5e200 - 1.).abs() < 1e-12);
    assert!((gradients[0][0] - 3.).abs() < 1e-12);
    assert!((gradients[0][1] - 4.).abs() < 1e-12);
}
