use super::{DataTransform, DifferencingTransformer};

#[test]
fn ordinary_and_seasonal_differences_restore_polynomial_suffixes_at_every_order() {
    let data = (1..=20)
        .map(|index| (index as f64).powi(3) + (index % 2) as f64)
        .collect::<Vec<_>>();
    for order in 0..=3 {
        for period in 1..=3 {
            let transformer = DifferencingTransformer::seasonal(order, period);
            let differences = transformer.try_difference(&data).unwrap();
            let extent = order * period;
            assert_eq!(differences.len(), data.len() - extent);
            let restored = transformer
                .integrate(&differences, &data[..extent])
                .unwrap();
            assert_eq!(restored, data[extent..], "order={order}, period={period}");
        }
    }
    let transformer = DifferencingTransformer::new(2);
    assert_eq!(
        transformer.integrate(&[2.0, 2.0], &[1.0, 4.0]).unwrap(),
        [9.0, 16.0]
    );
}

#[test]
fn differencing_parameters_and_arithmetic_fail_without_overflow_or_partial_results() {
    for transformer in [
        DifferencingTransformer::seasonal(1, 0),
        DifferencingTransformer::seasonal(usize::MAX, 2),
    ] {
        assert!(transformer.try_difference(&[1.0, 2.0]).is_err());
        assert!(transformer.difference(&[1.0, 2.0]).is_empty());
        assert!(transformer.integrate(&[1.0], &[1.0]).is_err());
    }
    let transformer = DifferencingTransformer::new(2);
    assert!(transformer.integrate(&[1.0], &[1.0]).is_err());
    assert!(transformer.try_difference(&[1.0, f64::NAN, 2.0]).is_err());
    assert!(DifferencingTransformer::new(1)
        .try_difference(&[-f64::MAX, f64::MAX])
        .is_err());
    assert!(DifferencingTransformer::new(1)
        .integrate(&[f64::MAX], &[f64::MAX])
        .is_err());
}

#[test]
fn transformed_timestamps_follow_the_full_seasonal_warmup() {
    use chrono::{Duration, TimeZone, Utc};
    let times = (0..10)
        .map(|index| Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap() + Duration::days(index))
        .collect::<Vec<_>>();
    let data = crate::TimeSeriesDatasetBuilder::new("a".into())
        .with_frequency("D".into())
        .with_timestamps(times.clone())
        .with_values((0..10).map(|index| (index as f64).powi(2)).collect())
        .build()
        .unwrap();
    let transformed = DifferencingTransformer::seasonal(2, 2)
        .transform(&data)
        .unwrap();
    assert_eq!(transformed.timestamps(), times[4..]);
    assert_eq!(transformed.values(), vec![8.0; 6]);
}
