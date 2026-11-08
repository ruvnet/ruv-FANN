use super::{CosineScheduler, LearningRateScheduler};

#[cfg(feature = "serde")]
#[test]
fn json_scheduler_state_restores_future_learning_rates() {
    let mut original = CosineScheduler::new(0.1_f64, 5).with_restarts(2.0);
    original.get_learning_rate(17);
    let encoded = serde_json::to_vec(&original.state()).unwrap();
    let state: super::SchedulerState<f64> = serde_json::from_slice(&encoded).unwrap();
    let mut restored = CosineScheduler::new(0.5, 1);
    restored.restore_state(state).unwrap();
    for step in [0, 5, 17, 25, 50] {
        assert!(
            (restored.get_learning_rate(step) - original.get_learning_rate(step)).abs() < 1e-14
        );
    }
}

#[test]
fn cosine_without_restarts_reaches_and_keeps_the_floor() {
    let mut scheduler = CosineScheduler::new(0.1_f64, 10).with_min_lr(0.01);
    assert!((scheduler.get_learning_rate(10) - 0.01).abs() < 1e-12);
    assert!((scheduler.get_learning_rate(1000) - 0.01).abs() < 1e-12);
}

#[test]
fn restart_queries_are_idempotent_and_order_independent() {
    let mut scheduler = CosineScheduler::new(0.1_f64, 10).with_restarts(2.0);
    let expected = scheduler.get_learning_rate(23);
    assert_eq!(scheduler.get_learning_rate(23), expected);
    scheduler.get_learning_rate(100);
    assert_eq!(scheduler.get_learning_rate(23), expected);
    assert!((scheduler.get_learning_rate(10) - 0.1).abs() < 1e-12);
    assert!((scheduler.get_learning_rate(30) - 0.1).abs() < 1e-12);
}

#[test]
fn zero_cycle_and_large_constant_cycles_terminate() {
    assert!(CosineScheduler::new(0.1_f64, 0)
        .get_learning_rate(1)
        .is_finite());
    let mut scheduler = CosineScheduler::new(0.1_f64, 10).with_restarts(1.0);
    assert!((scheduler.get_learning_rate((usize::MAX / 10) * 10) - 0.1).abs() < 1e-12);
}

#[test]
fn restart_state_preserves_future_rates() {
    let mut original = CosineScheduler::new(0.1_f64, 5).with_restarts(2.0);
    original.get_learning_rate(17);
    let mut restored = CosineScheduler::new(0.5, 1);
    restored.restore_state(original.state()).unwrap();
    for step in [0, 5, 17, 25, 50] {
        assert_eq!(
            restored.get_learning_rate(step),
            original.get_learning_rate(step)
        );
    }
}

#[test]
#[should_panic(expected = "Restart factor must be finite and at least one")]
fn invalid_restart_factor_is_rejected_before_a_loop() {
    CosineScheduler::new(0.1_f64, 10).with_restarts(0.0);
}
