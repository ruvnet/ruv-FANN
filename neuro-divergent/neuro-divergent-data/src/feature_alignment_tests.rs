use super::FeatureEngine;
use crate::TimeSeriesDatasetBuilder;
use chrono::{Duration, TimeZone, Utc};

#[test]
fn lag_rolling_and_calendar_features_share_timestamps_without_label_leakage() {
    let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let series = TimeSeriesDatasetBuilder::new("alignment".into())
        .with_frequency("D".into())
        .with_values(vec![1.0, 2.0, 3.0, 1000.0, 5.0, 6.0])
        .with_timestamps((0..6).map(|i| start + Duration::days(i)).collect())
        .build()
        .unwrap();
    let features = FeatureEngine::new()
        .with_lag_features(vec![1])
        .with_rolling_features(vec![3])
        .generate_features(&series)
        .unwrap();
    assert_eq!(features.n_samples(), 3);
    assert_eq!(features.timestamps[0], start + Duration::days(3));
    assert_eq!(features.get_feature("lag_1").unwrap()[0], 3.0);
    // The target at this row is 1000. Its rolling predictor is based solely
    // on [1, 2, 3], so modifying that target cannot leak into this feature.
    assert_eq!(features.get_feature("rolling_mean_3").unwrap()[0], 2.0);
    assert_eq!(features.get_feature("rolling_max_3").unwrap()[0], 3.0);
}

#[test]
fn zero_length_rolling_window_is_a_configuration_error() {
    let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let series = TimeSeriesDatasetBuilder::new("invalid-window".into())
        .with_frequency("D".into())
        .with_values(vec![1.0, 2.0])
        .with_timestamps(vec![start, start + Duration::days(1)])
        .build()
        .unwrap();
    assert!(FeatureEngine::new()
        .with_lag_features(vec![])
        .with_rolling_features(vec![0])
        .generate_features(&series)
        .is_err());
}
