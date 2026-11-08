use super::*;
use std::io::Write;

fn datetime_frame(unit: TimeUnit) -> TimeSeriesDataFrame<f64> {
    let original = DateTime::<Utc>::from_timestamp_nanos(1_767_225_600_123_456_789);
    let raw = match unit {
        TimeUnit::Milliseconds => original.timestamp_millis(),
        TimeUnit::Microseconds => original.timestamp_micros(),
        TimeUnit::Nanoseconds => original.timestamp_nanos_opt().unwrap(),
    };
    let timestamps = Series::new("ds".into(), [raw, raw + 1])
        .cast(&DataType::Datetime(unit, None))
        .unwrap();
    let frame = DataFrame::new(vec![
        Series::new("unique_id".into(), ["a", "a"]).into(),
        timestamps.into(),
        Series::new("y".into(), [1.0, 2.0]).into(),
    ])
    .unwrap();
    TimeSeriesDataFrame::from_polars(frame, TimeSeriesSchema::default(), None).unwrap()
}

#[test]
fn time_range_respects_datetime_units_and_precision() {
    for unit in [
        TimeUnit::Milliseconds,
        TimeUnit::Microseconds,
        TimeUnit::Nanoseconds,
    ] {
        let data = datetime_frame(unit);
        let (start, end) = data.time_range().unwrap();
        let expected = DateTime::<Utc>::from_timestamp_nanos(1_767_225_600_123_456_789);
        match unit {
            TimeUnit::Milliseconds => {
                assert_eq!(start.timestamp_millis(), expected.timestamp_millis())
            }
            TimeUnit::Microseconds => {
                assert_eq!(start.timestamp_micros(), expected.timestamp_micros())
            }
            TimeUnit::Nanoseconds => assert_eq!(start, expected),
        }
        let delta = match unit {
            TimeUnit::Milliseconds => chrono::Duration::milliseconds(1),
            TimeUnit::Microseconds => chrono::Duration::microseconds(1),
            TimeUnit::Nanoseconds => chrono::Duration::nanoseconds(1),
        };
        assert_eq!(end - start, delta);
        assert_eq!(data.filter_date_range(start, start).unwrap().shape().0, 1);
    }
}

#[test]
fn time_range_keeps_integer_millisecond_compatibility() {
    let frame = df! {
        "unique_id" => ["a", "a"],
        "ds" => [0_i64, 86_400_000_i64],
        "y" => [1.0, 2.0],
    }
    .unwrap();
    let data =
        TimeSeriesDataFrame::<f64>::from_polars(frame, TimeSeriesSchema::default(), None).unwrap();
    let (start, end) = data.time_range().unwrap();
    assert_eq!(start.timestamp(), 0);
    assert_eq!(end.timestamp(), 86_400);
    assert_eq!(data.filter_date_range(start, start).unwrap().shape().0, 1);
}

#[test]
fn csv_dates_and_required_columns_use_real_reader() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    writeln!(file, "unique_id,ds,y\na,2026-01-01,1.5\na,2026-01-02,2.5").unwrap();
    let data = TimeSeriesDataFrame::<f64>::from_csv(file.path(), TimeSeriesSchema::default(), None)
        .unwrap();
    assert_eq!(data.unique_ids().unwrap(), ["a"]);
    let (start, end) = data.time_range().unwrap();
    assert_eq!(end - start, chrono::Duration::days(1));
    let missing_schema = TimeSeriesSchema::new("absent", "ds", "y");
    assert!(TimeSeriesDataFrame::<f64>::from_csv(file.path(), missing_schema, None).is_err());
}
