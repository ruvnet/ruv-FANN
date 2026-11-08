//! Available decomposition, windowing and basis-feature primitives.
//! No NBEATS/NHITS architecture execution is claimed by these tests.
use crate::support::*;
use chrono::{Duration, TimeZone, Utc};
use neuro_divergent::models::utils::{math, timeseries};
use neuro_divergent_data::{features::FeatureEngine, TimeSeriesDatasetBuilder};

#[test]
fn sliding_windows_obey_lookback_stride_and_chronological_order() {
    let windows = timeseries::create_sliding_windows(&[1., 2., 3., 4., 5., 6.], 3, 2);
    assert_eq!(windows, vec![vec![1., 2., 3.], vec![3., 4., 5.]]);
    assert!(timeseries::create_sliding_windows(&[1., 2.], 3, 1).is_empty());
    assert!(timeseries::create_sliding_windows(&[1., 2.], usize::MAX, 1).is_empty());
}

#[test]
fn validation_split_keeps_future_observations_out_of_training() -> TestResult {
    let (train, validation) = timeseries::train_validation_split(&[1., 2., 3., 1000.], 0.25)?;
    assert_eq!(train, vec![1., 2., 3.]);
    assert_eq!(validation, vec![1000.]);
    assert!(timeseries::train_validation_split(&[1., 2.], f64::NAN).is_err());
    Ok(())
}

#[test]
fn fourier_feature_basis_is_periodic_at_matching_timestamps() -> TestResult {
    let start = Utc.timestamp_millis_opt(START).unwrap();
    let series = TimeSeriesDatasetBuilder::new("periodic".into())
        .with_frequency("D".into())
        .with_values(vec![1.; 20])
        .with_timestamps((0..20).map(|i| start + Duration::days(i)).collect())
        .build()?;
    let basis = FeatureEngine::<f64>::new()
        .with_lag_features(vec![])
        .with_rolling_features(vec![])
        .with_temporal_features(false)
        .with_fourier_features(vec![7], vec![1])
        .generate_features(&series)?;
    assert!(basis.n_features() >= 2);
    assert!(basis.n_samples() > 7);
    for column in 0..basis.n_features() {
        assert!((basis.features[[0, column]] - basis.features[[7, column]]).abs() < 1e-10);
    }
    Ok(())
}

#[test]
fn matrix_basis_projection_matches_a_hand_computed_result() -> TestResult {
    let result = math::matrix_multiply(
        &[vec![1., 0., 1.], vec![1., 1., 1.]],
        &[vec![2.], vec![3.], vec![4.]],
    )?;
    assert_eq!(result, vec![vec![6.], vec![9.]]);
    assert!(math::matrix_multiply(&[vec![1., 2.], vec![1.]], &[vec![1.], vec![2.]]).is_err());
    Ok(())
}
