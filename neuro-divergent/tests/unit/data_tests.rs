//! Real data storage, transforms and causal windows.
use crate::support::*;
use chrono::{Duration, TimeZone, Utc};
use neuro_divergent_data::{
    preprocessing::{
        DifferencingTransformer, LogTransformer, MinMaxScaler, Scaler, StandardScaler,
    },
    TimeSeriesDatasetBuilder,
};
use proptest::prelude::*;

#[test]
fn typed_series_slice_keeps_timestamps_targets_and_id_aligned() -> TestResult {
    let start = Utc.timestamp_millis_opt(START).unwrap();
    let data = TimeSeriesDatasetBuilder::new("real".into())
        .with_frequency("D".into())
        .with_values(vec![1., 2., 3., 4.])
        .with_timestamps((0..4).map(|i| start + Duration::days(i)).collect())
        .build()?;
    let slice = data.slice(1, 3)?;
    assert_eq!(slice.values(), vec![2., 3.]);
    assert_eq!(
        slice.timestamps(),
        vec![start + Duration::days(1), start + Duration::days(2)]
    );
    assert_eq!(slice.series_id, "real");
    assert!(data.slice(2, 5).is_err());
    Ok(())
}

#[test]
fn builder_rejects_mismatched_timestamp_and_target_lengths() {
    let start = Utc.timestamp_millis_opt(START).unwrap();
    assert!(TimeSeriesDatasetBuilder::new("invalid".into())
        .with_values(vec![1., 2.])
        .with_timestamps(vec![start])
        .build()
        .is_err());
}

#[test]
fn differencing_and_integration_are_inverse_for_observed_sequences() -> TestResult {
    let differencer = DifferencingTransformer::new(1);
    let original = [1., 3., 6., 10.];
    let differences = differencer.difference(&original);
    assert_eq!(differences, vec![2., 3., 4.]);
    // Integration returns the restored suffix; the seed observation is supplied separately.
    assert_eq!(differencer.integrate(&differences, &[1.])?, original[1..]);
    Ok(())
}

#[test]
fn logarithmic_transform_roundtrips_positive_targets() -> TestResult {
    let transformer = LogTransformer::<f64>::new();
    let values = [1., 2., 4., 8.];
    let logged = transformer.transform_values(&values)?;
    let restored = transformer.inverse_transform_values(&logged)?;
    for (a, b) in restored.iter().zip(values) {
        assert!((a - b).abs() < 1e-12);
    }
    Ok(())
}

proptest! {
    #[test]
    fn standard_scaling_roundtrips_finite_nonconstant_samples(values in prop::collection::vec(-1000f64..1000.,2..100)) {
        let mut scaler=StandardScaler::new();
        let scaled=scaler.fit_transform(&values).unwrap();
        let restored=scaler.inverse_transform(&scaled).unwrap();
        for (a,b) in restored.iter().zip(values) {prop_assert!((a-b).abs()<1e-8);}
    }
    #[test]
    fn minmax_scaling_is_bounded_on_its_training_observations(values in prop::collection::vec(-1000f64..1000.,2..100)) {
        let mut scaler=MinMaxScaler::new();
        let scaled=scaler.fit_transform(&values).unwrap();
        prop_assert!(scaled.iter().all(|v|*v>=-1e-12 && *v<=1.+1e-12));
    }
}

#[test]
fn higher_order_and_seasonal_differences_restore_the_observed_suffix() -> TestResult {
    let original = [1_f64, 4., 9., 16., 25., 36., 49., 64., 81., 100.];
    for order in [1, 2, 3] {
        for period in [1, 2] {
            let transform = if period == 1 {
                DifferencingTransformer::new(order)
            } else {
                DifferencingTransformer::seasonal(order, period)
            };
            let differences = transform.difference(&original);
            let prefix = order * period;
            let restored = transform.integrate(&differences, &original[..prefix])?;
            assert_eq!(
                restored,
                original[prefix..],
                "order={order}, period={period}"
            );
        }
    }
    Ok(())
}
