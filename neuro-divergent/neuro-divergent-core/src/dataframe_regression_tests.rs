use super::*;
use std::io::Write;

fn schema() -> TimeSeriesSchema {
    TimeSeriesSchema::new("unique_id", "ds", "y")
}

#[test]
fn csv_reader_loads_real_dates_and_numeric_targets() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    writeln!(file, "unique_id,ds,y\na,2026-01-01,1.5\na,2026-01-02,2.5").unwrap();
    let dataframe = TimeSeriesDataFrame::<f64>::from_csv(file.path(), schema()).unwrap();
    let dataset = dataframe.to_dataset().unwrap();
    let series = &dataset.series_data["a"];
    assert_eq!(series.target_values, vec![1.5, 2.5]);
    assert_eq!(
        series.timestamps[0],
        Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()
    );
}

#[test]
fn csv_reader_returns_errors_for_missing_file_and_required_columns() {
    let directory = tempfile::tempdir().unwrap();
    assert!(
        TimeSeriesDataFrame::<f64>::from_csv(directory.path().join("absent.csv"), schema())
            .is_err()
    );
    let mut file = tempfile::NamedTempFile::new().unwrap();
    writeln!(file, "unique_id,ds\na,2026-01-01").unwrap();
    assert!(TimeSeriesDataFrame::<f64>::from_csv(file.path(), schema()).is_err());
}

#[test]
fn dataframe_conversion_preserves_all_timestamp_units() {
    let expected = DateTime::<Utc>::from_timestamp_nanos(1_767_225_600_123_456_789);
    for unit in [
        TimeUnit::Milliseconds,
        TimeUnit::Microseconds,
        TimeUnit::Nanoseconds,
    ] {
        let raw = match unit {
            TimeUnit::Milliseconds => expected.timestamp_millis(),
            TimeUnit::Microseconds => expected.timestamp_micros(),
            TimeUnit::Nanoseconds => expected.timestamp_nanos_opt().unwrap(),
        };
        let timestamp = Series::new("ds".into(), [raw])
            .cast(&DataType::Datetime(unit, None))
            .unwrap();
        let frame = DataFrame::new(vec![
            Series::new("unique_id".into(), ["a"]).into(),
            timestamp.into(),
            Series::new("y".into(), [7_i32]).into(),
        ])
        .unwrap();
        let data = TimeSeriesDataFrame::<f64>::from_polars(frame, schema())
            .unwrap()
            .to_dataset()
            .unwrap();
        let actual = &data.series_data["a"];
        assert_eq!(actual.target_values, [7.0]);
        let expected = match unit {
            TimeUnit::Milliseconds => DateTime::<Utc>::from_timestamp_millis(raw).unwrap(),
            TimeUnit::Microseconds => DateTime::<Utc>::from_timestamp_micros(raw).unwrap(),
            TimeUnit::Nanoseconds => expected,
        };
        assert_eq!(actual.timestamps, [expected]);
    }
}

#[test]
fn dataframe_conversion_rejects_nulls_instead_of_misaligning_rows() {
    let timestamp = Series::new("ds".into(), [Some(0_i64), None])
        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
        .unwrap();
    let frame = DataFrame::new(vec![
        Series::new("unique_id".into(), ["a", "a"]).into(),
        timestamp.into(),
        Series::new("y".into(), [None, Some(2.0)]).into(),
    ])
    .unwrap();
    let data = TimeSeriesDataFrame::<f64>::from_polars(frame, schema()).unwrap();
    assert!(data.to_dataset().is_err());
}

#[test]
fn conversion_sorts_and_preserves_numeric_exogenous_static_and_missing_metadata() {
    let frame = df! {
        "unique_id" => ["a", "a", "a"],
        "ds" => [172_800_000_i64, 0, 86_400_000],
        "y" => [3.0, 1.0, 2.0],
        "historical" => [30_i32, 10, 20],
        "known" => [0.3_f32, 0.1, 0.2],
        "static" => [5_f32, 5.0, 5.0],
        "extra" => [Some("value"), None, Some("value")],
    }
    .unwrap();
    let schema = schema()
        .with_static_features(vec!["static".into()])
        .with_historical_exogenous(vec!["historical".into()])
        .with_future_exogenous(vec!["known".into()]);
    let frame = TimeSeriesDataFrame::<f64>::from_polars(frame, schema).unwrap();
    assert_eq!(frame.time_range().unwrap().0.timestamp_millis(), 0);
    assert_eq!(
        frame.time_range().unwrap().1.timestamp_millis(),
        172_800_000
    );
    let dataset = frame.to_dataset().unwrap();
    assert_eq!(dataset.series_data["a"].target_values, [1.0, 2.0, 3.0]);
    assert_eq!(
        dataset.series_data["a"]
            .historical_exogenous
            .as_ref()
            .unwrap()
            .column(0)
            .to_vec(),
        [10.0, 20.0, 30.0]
    );
    assert_eq!(dataset.static_features.as_ref().unwrap()["a"], [5.0]);
    assert_eq!(dataset.metadata.frequency.as_deref(), Some("D"));
    assert_eq!(dataset.metadata.missing_stats.total_missing, 1);
    assert_eq!(dataset.metadata.missing_stats.series_with_missing, ["a"]);
}

#[test]
fn static_feature_changes_and_duplicate_times_are_rejected() {
    let frame = df! { "unique_id" => ["a", "a"], "ds" => [0_i64, 86_400_000], "y" => [1.0, 2.0], "static" => [1.0, 2.0] }.unwrap();
    let schema = schema().with_static_features(vec!["static".into()]);
    assert!(TimeSeriesDataFrame::<f64>::from_polars(frame, schema)
        .unwrap()
        .to_dataset()
        .is_err());
    let frame = df! { "unique_id" => ["a", "a"], "ds" => [0_i64, 0], "y" => [1.0, 2.0] }.unwrap();
    assert!(TimeSeriesDataFrame::<f64>::from_polars(
        frame,
        super::TimeSeriesSchema::new("unique_id", "ds", "y")
    )
    .unwrap()
    .to_dataset()
    .is_err());
}
