//! Real probabilistic scoring primitives. No DeepAR/TCN architecture is implied.
use neuro_divergent::models::utils::math;
use neuro_divergent_training::loss::{HuberLoss, LossFunction, MAELoss, PinballLoss, QuantileLoss};

#[test]
fn quantile_forecast_scores_penalize_underprediction_asymmetrically() {
    let loss = PinballLoss::new(0.9_f64);
    assert!((loss.forward(&[0.], &[1.]).unwrap() - 0.9).abs() < 1e-12);
    assert!((loss.forward(&[2.], &[1.]).unwrap() - 0.1).abs() < 1e-12);
    assert_eq!(loss.backward(&[0.], &[1.]).unwrap(), vec![-0.9]);
}

#[test]
fn multiple_quantile_scores_average_the_actual_requested_quantiles() {
    let predictions = [0., 1., 2.];
    let targets = [1.];
    let loss = QuantileLoss::new(vec![0.1_f64, 0.5, 0.9]);
    let score = loss.forward(&predictions, &targets).unwrap();
    assert!((score - (0.1 + 0. + 0.1) / 3.).abs() < 1e-12);
    assert_eq!(loss.backward(&predictions, &targets).unwrap().len(), 3);
}

#[test]
fn huber_loss_limits_outlier_influence_without_erasing_the_error() {
    let huber = HuberLoss::new(1.0_f64);
    assert_eq!(huber.forward(&[100.], &[0.]).unwrap(), 99.5);
    assert_eq!(huber.backward(&[100.], &[0.]).unwrap(), vec![1.]);
    assert_eq!(MAELoss::new().forward(&[100.], &[0.]).unwrap(), 100.);
}

#[test]
fn actual_forecast_scoring_matches_independent_reference_values() {
    let actual = [1_f64, 3., 5.];
    let forecasts = [2., 1., 5.];
    assert_eq!(math::mae(&actual, &forecasts).unwrap(), 1.);
    assert!((math::mse(&actual, &forecasts).unwrap() - 5. / 3.).abs() < 1e-12);
    assert!((math::rmse(&actual, &forecasts).unwrap() - (5_f64 / 3.).sqrt()).abs() < 1e-12);
    assert!(math::mse(&actual, &[1., 2.]).is_err());
}
