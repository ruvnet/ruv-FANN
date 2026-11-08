//! Explicit Python-style column/workflow contracts; no Python execution claim.
use super::support::*;
use chrono::{TimeZone, Utc};
use neuro_divergent::{
    CrossValidationConfig, Frequency, NaiveForecaster, NeuralForecast, TimeSeriesDataFrame,
    TimeSeriesSchema,
};
use polars::prelude::*;

fn utc_millis(year: i32, month: u32, day: u32, hour: u32) -> i64 {
    Utc.with_ymd_and_hms(year, month, day, hour, 0, 0)
        .unwrap()
        .timestamp_millis()
}

fn new_york_history(
    frequency: Frequency,
    instants: &[i64],
) -> Result<TimeSeriesDataFrame<f64>, Box<dyn std::error::Error>> {
    // Attach the declared zone directly: Polars 0.45's cast may drop it.
    let timestamp = Int64Chunked::from_slice("ds".into(), instants)
        .into_datetime(TimeUnit::Milliseconds, Some("America/New_York".into()))
        .into_series();
    let data = DataFrame::new(vec![
        Series::new("unique_id".into(), ["store", "store"]).into(),
        timestamp.into(),
        Series::new("y".into(), [10.0, 20.0]).into(),
    ])?;
    Ok(TimeSeriesDataFrame::from_polars(
        data,
        TimeSeriesSchema::default(),
        Some(frequency),
    )?)
}

#[test]
fn python_style_id_ds_y_input_has_wide_forecast_output() -> TestResult {
    let mut nf = baseline(2);
    let forecast = nf.fit_predict(frame(2, 10))?;
    assert_eq!(
        forecast.data.get_column_names_str(),
        vec!["unique_id", "ds", "Naive"]
    );
    assert_eq!(
        forecast.data.column("unique_id")?.dtype(),
        &DataType::String
    );
    assert_eq!(forecast.data.column("Naive")?.dtype(), &DataType::Float64);
    assert_finite_forecast(&forecast, 2, 2);
    Ok(())
}

#[test]
fn custom_input_column_names_survive_fit_predict_and_cv() -> TestResult {
    let mut raw = frame(2, 10).data;
    raw.rename("unique_id", "store".into())?;
    raw.rename("ds", "date".into())?;
    raw.rename("y", "sales".into())?;
    let data = TimeSeriesDataFrame::from_polars(
        raw,
        TimeSeriesSchema::new("store", "date", "sales"),
        Some(Frequency::Daily),
    )?;
    let mut nf = baseline(2);
    nf.fit(data.clone())?;
    let prediction = nf.predict()?;
    assert_eq!(
        prediction.data.get_column_names_str(),
        vec!["store", "date", "Naive"]
    );
    assert_eq!(prediction.schema.y_col, "sales");
    let cv = nf.cross_validation(data, CrossValidationConfig::new(2, 2))?;
    for name in ["store", "date", "cutoff", "sales", "Naive"] {
        assert!(cv.data.column(name).is_ok());
    }
    Ok(())
}

#[test]
fn datetime_inputs_preserve_time_unit_and_horizon() -> TestResult {
    let mut data = frame(1, 8);
    let datetime = data
        .data
        .column("ds")?
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    // Convert physical milliseconds into microseconds before the datetime cast.
    let microseconds: Vec<_> = times(&data.data, "ds")
        .into_iter()
        .map(|x| x * 1000)
        .collect();
    let physical = Series::new("ds".into(), microseconds).cast(datetime.dtype())?;
    data.data.with_column(physical)?;
    let mut nf = baseline(2);
    nf.fit(data)?;
    let result = nf.predict()?;
    assert_eq!(
        result.data.column("ds")?.dtype(),
        &DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(
        times(&result.data, "ds"),
        vec![(START + 8 * DAY) * 1000, (START + 9 * DAY) * 1000]
    );
    Ok(())
}

#[test]
fn csv_and_parquet_roundtrips_preserve_forecasting_inputs() -> TestResult {
    let directory = tempfile::tempdir()?;
    let data = frame(2, 10);
    let csv = directory.path().join("series.csv");
    let parquet = directory.path().join("series.parquet");
    data.to_csv(&csv)?;
    data.to_parquet(&parquet)?;
    for restored in [
        TimeSeriesDataFrame::<f64>::from_csv(
            csv,
            TimeSeriesSchema::default(),
            Some(Frequency::Daily),
        )?,
        TimeSeriesDataFrame::<f64>::from_parquet(
            parquet,
            TimeSeriesSchema::default(),
            Some(Frequency::Daily),
        )?,
    ] {
        assert_eq!(restored.shape(), data.shape());
        let mut nf = baseline(2);
        nf.fit(restored)?;
        assert_finite_forecast(&nf.predict()?, 2, 2);
    }
    Ok(())
}

#[test]
fn exported_forecast_csv_has_one_row_per_series_and_horizon_step() -> TestResult {
    let forecast = baseline(3).fit_predict(frame(2, 8))?;
    let directory = tempfile::tempdir()?;
    let csv = directory.path().join("forecasts.csv");
    forecast.to_csv(&csv)?;
    let text = std::fs::read_to_string(csv)?;
    assert_eq!(text.lines().next(), Some("unique_id,ds,Naive"));
    assert_eq!(text.lines().count(), 7);
    Ok(())
}

#[test]
fn frequency_parsing_roundtrips_supported_pandas_codes() -> TestResult {
    for code in ["D", "H", "W", "M", "Q", "Y"] {
        let frequency = Frequency::from_str(code)?;
        assert_eq!(Frequency::from_str(frequency.to_pandas_str())?, frequency);
    }
    assert_eq!(
        Frequency::from_str("invented")?,
        Frequency::Custom("invented".into())
    );
    Ok(())
}

#[test]
fn probabilistic_configuration_is_an_explicit_unsupported_boundary() -> TestResult {
    let mut nf = baseline(2);
    nf.fit(frame(1, 8))?;
    let error = nf.predict_with_config(neuro_divergent::PredictionConfig::new().with_intervals());
    assert!(error.is_err());
    // Ordinary point forecasting still executes after the rejected request.
    assert_finite_forecast(&nf.predict()?, 1, 2);
    assert!(!neuro_divergent::info::has_gpu_support());
    Ok(())
}

#[test]
fn date_input_forecasts_have_datetime_milliseconds_with_calendar_dates() -> TestResult {
    let mut history = frame(1, 8);
    let dates = history
        .data
        .column("ds")?
        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?
        .cast(&DataType::Date)?;
    history.data.with_column(dates)?;
    let mut nf = baseline(2);
    nf.fit(history)?;
    let forecast = nf.predict()?;
    assert_eq!(
        forecast.data.column("ds")?.dtype(),
        &DataType::Datetime(TimeUnit::Milliseconds, None)
    );
    assert_eq!(
        times(&forecast.data, "ds"),
        vec![START + 8 * DAY, START + 9 * DAY]
    );
    Ok(())
}

#[test]
fn zoned_month_end_forecasts_preserve_local_midnight_across_dst() -> TestResult {
    // Both observed month ends are New York midnight (05:00 UTC).
    // March onward is daylight time, so local midnight is 04:00 UTC.
    let history = new_york_history(
        Frequency::Monthly,
        &[utc_millis(2023, 1, 31, 5), utc_millis(2023, 2, 28, 5)],
    )?;
    let dtype = DataType::Datetime(TimeUnit::Milliseconds, Some("America/New_York".into()));
    assert_eq!(history.data.column("ds")?.dtype(), &dtype);
    let mut pipeline = NeuralForecast::new(
        vec![Box::new(NaiveForecaster::<f64>::new("Naive", 3)?)],
        Frequency::Monthly,
    )?;
    pipeline.fit(history)?;
    let forecast = pipeline.predict()?;
    assert_eq!(forecast.data.column("ds")?.dtype(), &dtype);
    assert_eq!(
        times(&forecast.data, "ds"),
        vec![
            utc_millis(2023, 3, 31, 4),
            utc_millis(2023, 4, 30, 4),
            utc_millis(2023, 5, 31, 4),
        ]
    );
    assert_eq!(values(&forecast.data, "Naive"), vec![20.0; 3]);
    assert_finite_forecast(&forecast, 1, 3);
    Ok(())
}

#[test]
fn zoned_daily_checkpoint_preserves_dst_adjusted_forecast_instants() -> TestResult {
    // New York enters daylight time during March 8, 2026. March 9 midnight
    // is 04:00 UTC, only 23 elapsed hours after March 8 midnight.
    let history = new_york_history(
        Frequency::Daily,
        &[utc_millis(2026, 3, 7, 5), utc_millis(2026, 3, 8, 5)],
    )?;
    let dtype = DataType::Datetime(TimeUnit::Milliseconds, Some("America/New_York".into()));
    let mut pipeline = baseline(2);
    pipeline.fit(history)?;
    let expected = vec![utc_millis(2026, 3, 9, 4), utc_millis(2026, 3, 10, 4)];
    let forecast = pipeline.predict()?;
    assert_eq!(forecast.data.column("ds")?.dtype(), &dtype);
    assert_eq!(times(&forecast.data, "ds"), expected);
    assert_eq!(values(&forecast.data, "Naive"), vec![20.0; 2]);

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("zoned-baseline.json");
    pipeline.save(&path)?;
    let restored = NeuralForecast::<f64>::load(&path)?;
    let restored_forecast = restored.predict()?;
    assert_eq!(restored_forecast.data.column("ds")?.dtype(), &dtype);
    assert_eq!(times(&restored_forecast.data, "ds"), expected);
    assert_eq!(
        rows(&restored_forecast.data, "Naive"),
        rows(&forecast.data, "Naive")
    );
    Ok(())
}
