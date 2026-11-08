use crate::utils::{clip_gradients_by_norm, gradient_norm};

#[test]
fn loss_adapter_does_not_turn_checked_errors_into_zero_loss() {
    let adapter = crate::LossAdapter::new(Box::new(crate::MSELoss::<f64>::new()));
    assert!(adapter.calculate_loss(&[1.0, 2.0], &[1.0]).is_err());
    assert!(ruv_fann::training::ErrorFunction::calculate(&adapter, &[1.0, 2.0], &[1.0]).is_nan());
    assert_eq!(
        adapter.calculate_gradient(&[1.0], &[0.5]).unwrap(),
        vec![1.0]
    );
}

#[test]
fn seeded_rng_replays_draws_without_claiming_to_reseed_thread_rng() {
    use rand::Rng;
    let mut first = crate::utils::seeded_rng(73);
    let mut second = crate::utils::seeded_rng(73);
    for _ in 0..10 {
        assert_eq!(first.gen::<u64>(), second.gen::<u64>());
    }
}

#[test]
fn large_finite_gradients_do_not_overflow_when_squared() {
    let mut gradients = vec![vec![3e200_f64, 4e200_f64]];
    let original = clip_gradients_by_norm(&mut gradients, 5.0);
    assert!((original / 5e200 - 1.0).abs() < 1e-14);
    assert!((gradient_norm(&gradients) - 5.0).abs() < 1e-12);
}

#[test]
fn clipping_an_unrepresentable_norm_preserves_the_direction() {
    let mut gradients = vec![vec![1e308_f64; 4]];
    assert!(clip_gradients_by_norm(&mut gradients, 2.0).is_infinite());
    assert!(gradients
        .iter()
        .flatten()
        .all(|value| (*value - 1.0).abs() < 1e-14));
}

#[test]
fn invalid_clipping_threshold_does_not_invert_gradients() {
    for threshold in [-1.0_f64, f64::NAN] {
        let mut gradients = vec![vec![3.0, 4.0]];
        clip_gradients_by_norm(&mut gradients, threshold);
        assert_eq!(gradients, vec![vec![3.0, 4.0]]);
    }
}
