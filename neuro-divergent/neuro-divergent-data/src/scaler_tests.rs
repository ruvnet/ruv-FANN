use super::{Scaler, StandardScaler};

#[test]
fn scale_only_uses_standard_deviation_and_preserves_offset() {
    let mut scaler = StandardScaler::<f64>::new_scale_only();
    let values = [10.0, 12.0, 14.0];
    scaler.fit(&values).unwrap();
    let expected = (8.0_f64 / 3.0).sqrt();
    assert!((scaler.std().unwrap() - expected).abs() < 1e-12);
    assert!(scaler.mean().is_none());
    let scaled = scaler.transform(&values).unwrap();
    assert!((scaled[0] - 10.0 / expected).abs() < 1e-12);
    let recovered = scaler.inverse_transform(&scaled).unwrap();
    for (actual, expected) in recovered.iter().zip(values.iter()) {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn online_statistics_remain_stable_at_large_offsets() {
    let mut scaler = StandardScaler::<f64>::new();
    let values = [1e12, 1e12 + 1.0, 1e12 + 2.0];
    scaler.fit(&values).unwrap();
    assert_eq!(scaler.mean(), Some(1e12 + 1.0));
    assert!((scaler.std().unwrap() - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
}

#[test]
fn invalid_refit_keeps_previous_statistics() {
    let mut scaler = StandardScaler::<f64>::new();
    scaler.fit(&[1.0, 2.0, 3.0]).unwrap();
    let mean = scaler.mean();
    let standard_deviation = scaler.std();
    for values in [&[4.0, f64::NAN][..], &[f64::INFINITY][..], &[][..]] {
        assert!(scaler.fit(values).is_err());
        assert_eq!(scaler.mean(), mean);
        assert_eq!(scaler.std(), standard_deviation);
    }
}

#[test]
fn constant_series_and_center_only_mode_round_trip() {
    let mut scaler = StandardScaler::<f64>::new();
    assert_eq!(scaler.fit_transform(&[5.0, 5.0]).unwrap(), vec![0.0, 0.0]);
    assert_eq!(
        scaler.inverse_transform(&[0.0, 0.0]).unwrap(),
        vec![5.0, 5.0]
    );
    let mut center = StandardScaler::<f64>::new_center_only();
    assert_eq!(
        center.fit_transform(&[10.0, 12.0]).unwrap(),
        vec![-1.0, 1.0]
    );
    assert_eq!(center.std(), None);
}
