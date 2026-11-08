use super::*;
use chrono::{DateTime, TimeZone, Utc};
use polars::prelude::*;

fn history(values: &[f64]) -> TimeSeriesDataFrame<f64> {
    let frame = df! {
        "unique_id" => vec!["a"; values.len()],
        "ds" => (0..values.len()).map(|index| index as i64 * 86_400_000).collect::<Vec<_>>(),
        "y" => values,
    }
    .unwrap();
    TimeSeriesDataFrame::from_polars(frame, TimeSeriesSchema::default(), Some(Frequency::Daily))
        .unwrap()
}

fn naive(horizon: usize) -> NeuralForecast<f64> {
    NeuralForecast::new(
        vec![Box::new(NaiveForecaster::new("Naive", horizon).unwrap())],
        Frequency::Daily,
    )
    .unwrap()
}

#[test]
fn past_only_scalers_and_refit_policy_match_validation_cutoffs() {
    let data = history(&[0.0, 1.0, 2.0, 3.0, 100.0, 200.0, 300.0, 400.0]);
    let mut pipeline = NeuralForecast::builder()
        .with_model(Box::new(NaiveForecaster::new("Naive", 2).unwrap()))
        .with_frequency(Frequency::Daily)
        .with_local_scaler(ScalerType::StandardScaler)
        .build()
        .unwrap();
    let result = pipeline
        .cross_validation(
            data.clone(),
            CrossValidationConfig::new(2, 2).with_refit(false),
        )
        .unwrap();
    assert_eq!(pipeline.scales["a"].location, 1.5);
    let predictions = result
        .data
        .column("Naive")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect::<Vec<_>>();
    assert_eq!(predictions, [3.0, 3.0, 200.0, 200.0]);
    assert_eq!(
        result
            .data
            .column("y")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        [100.0, 200.0, 300.0, 400.0]
    );
    assert_eq!(
        result
            .cutoffs
            .iter()
            .map(|time| time.timestamp_millis())
            .collect::<Vec<_>>(),
        [3 * 86_400_000, 5 * 86_400_000]
    );
    assert!((result.metrics_by_model().unwrap()["Naive"].mae - 148.5).abs() < 1e-12);
    let result = pipeline
        .cross_validation(data, CrossValidationConfig::new(2, 2))
        .unwrap();
    assert!((pipeline.scales["a"].location - 51.0).abs() < 1e-12);
    assert_eq!(result.summary_stats().unwrap().height(), 1);
}

#[test]
fn month_end_and_business_dates_are_calendar_offsets() {
    let dates = [
        Utc.with_ymd_and_hms(2026, 1, 31, 0, 0, 0).unwrap(),
        Utc.with_ymd_and_hms(2026, 2, 28, 0, 0, 0).unwrap(),
        Utc.with_ymd_and_hms(2026, 3, 31, 0, 0, 0).unwrap(),
    ];
    let frame = df! { "unique_id" => ["a", "a", "a"], "ds" => dates.map(|time| time.timestamp_millis()), "y" => [1.0, 2.0, 3.0] }.unwrap();
    let data = TimeSeriesDataFrame::<f64>::from_polars(
        frame,
        TimeSeriesSchema::default(),
        Some(Frequency::Monthly),
    )
    .unwrap();
    let mut pipeline = NeuralForecast::new(
        vec![Box::new(NaiveForecaster::new("Naive", 2).unwrap())],
        Frequency::Monthly,
    )
    .unwrap();
    pipeline.fit(data).unwrap();
    let future = pipeline.predict().unwrap();
    let times = crate::timeline::timestamps(future.data.column("ds").unwrap()).unwrap();
    assert_eq!(
        times[0],
        Utc.with_ymd_and_hms(2026, 4, 30, 0, 0, 0).unwrap()
    );
    assert_eq!(
        times[1],
        Utc.with_ymd_and_hms(2026, 5, 31, 0, 0, 0).unwrap()
    );
    let friday = Utc.with_ymd_and_hms(2026, 1, 30, 12, 0, 0).unwrap();
    assert_eq!(
        crate::timeline::advance(friday, &Frequency::BusinessDaily, 1).unwrap(),
        Utc.with_ymd_and_hms(2026, 2, 2, 12, 0, 0).unwrap()
    );
    assert!(crate::timeline::advance(friday, &Frequency::Custom("invalid".into()), 0).is_err());
}

#[test]
fn nanosecond_forecasts_preserve_precision_and_reject_lossy_output_types() {
    let base = 1_767_225_600_123_456_789_i64;
    let timestamp = Series::new("ds".into(), [base, base + 1])
        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        .unwrap();
    let frame = DataFrame::new(vec![
        Series::new("unique_id".into(), ["a", "a"]).into(),
        timestamp.into(),
        Series::new("y".into(), [1.0, 2.0]).into(),
    ])
    .unwrap();
    let data = TimeSeriesDataFrame::<f64>::from_polars(
        frame,
        TimeSeriesSchema::default(),
        Some(Frequency::Nanosecond),
    )
    .unwrap();
    let mut pipeline = NeuralForecast::new(
        vec![Box::new(NaiveForecaster::new("Naive", 2).unwrap())],
        Frequency::Nanosecond,
    )
    .unwrap();
    pipeline.fit(data).unwrap();
    let times = crate::timeline::timestamps(pipeline.predict().unwrap().data.column("ds").unwrap())
        .unwrap();
    assert_eq!(
        times,
        [
            DateTime::from_timestamp_nanos(base + 2),
            DateTime::from_timestamp_nanos(base + 3)
        ]
    );
    let frame = df! { "unique_id" => ["a"], "ds" => [0_i64], "y" => [1.0] }.unwrap();
    pipeline
        .fit(
            TimeSeriesDataFrame::from_polars(
                frame,
                TimeSeriesSchema::default(),
                Some(Frequency::Nanosecond),
            )
            .unwrap(),
        )
        .unwrap();
    assert!(pipeline
        .predict()
        .unwrap_err()
        .to_string()
        .contains("precision"));
}

#[derive(Debug)]
struct FailingRefit {
    inner: NaiveForecaster<f64>,
    fits: usize,
}
impl BaseModel<f64> for FailingRefit {
    fn fit(&mut self, data: &TimeSeriesDataFrame<f64>) -> NeuroDivergentResult<()> {
        self.fits += 1;
        if self.fits > 1 {
            self.inner.reset()?;
            return Err(NeuroDivergentError::training("deliberate refit failure"));
        }
        self.inner.fit(data)
    }
    fn predict(
        &self,
        data: &TimeSeriesDataFrame<f64>,
    ) -> NeuroDivergentResult<ForecastDataFrame<f64>> {
        self.inner.predict(data)
    }
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn horizon(&self) -> usize {
        self.inner.horizon()
    }
    fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }
    fn reset(&mut self) -> NeuroDivergentResult<()> {
        self.inner.reset()
    }
}

#[test]
fn failed_refits_and_mutable_model_access_invalidate_pipeline_state() {
    let model = FailingRefit {
        inner: NaiveForecaster::new("Naive", 1).unwrap(),
        fits: 0,
    };
    let mut pipeline = NeuralForecast::new(vec![Box::new(model)], Frequency::Daily).unwrap();
    pipeline.fit(history(&[1.0, 2.0])).unwrap();
    assert!(pipeline.predict().is_ok());
    assert!(pipeline.fit(history(&[2.0, 3.0])).is_err());
    assert!(!pipeline.is_fitted());
    assert!(pipeline.predict().is_err());
    let mut pipeline = naive(1);
    pipeline.fit(history(&[1.0, 2.0])).unwrap();
    assert!(pipeline.get_model_mut("Naive").is_some());
    assert!(!pipeline.is_fitted());
    assert!(pipeline.predict().is_err());
}

#[test]
fn checkpoints_roundtrip_scalers_and_reject_unknown_precision_versions_and_sizes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.json");
    let mut pipeline = NeuralForecast::builder()
        .with_model(Box::new(
            NaiveForecaster::seasonal("Seasonal", 3, 2).unwrap(),
        ))
        .with_frequency(Frequency::Daily)
        .with_local_scaler(ScalerType::RobustScaler)
        .build()
        .unwrap();
    pipeline.fit(history(&[1.0, 3.0, 5.0, 7.0])).unwrap();
    pipeline.save(&path).unwrap();
    let loaded = NeuralForecast::<f64>::load(&path).unwrap();
    assert_eq!(
        loaded.predict().unwrap().data,
        pipeline.predict().unwrap().data
    );
    assert!(NeuralForecast::<f32>::load(&path).is_err());
    let mut state: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    state["version"] = serde_json::json!(999);
    std::fs::write(&path, serde_json::to_vec(&state).unwrap()).unwrap();
    assert!(NeuralForecast::<f64>::load(&path).is_err());
    std::fs::File::create(&path)
        .unwrap()
        .set_len(64 * 1024 * 1024 + 1)
        .unwrap();
    assert!(NeuralForecast::<f64>::load(&path)
        .unwrap_err()
        .to_string()
        .contains("64 MiB"));
}

#[test]
fn custom_checkpoint_failure_preserves_existing_artifact() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("existing");
    std::fs::write(&path, b"preserve").unwrap();
    let model = FailingRefit {
        inner: NaiveForecaster::new("Naive", 1).unwrap(),
        fits: 0,
    };
    let mut pipeline = NeuralForecast::new(vec![Box::new(model)], Frequency::Daily).unwrap();
    pipeline.fit(history(&[1.0, 2.0])).unwrap();
    assert!(pipeline.save(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"preserve");
}

#[test]
fn schema_aliases_nonfinite_intervals_and_unsupported_runtime_options_are_errors() {
    assert!(
        PredictionIntervals::new(vec![f64::NAN], crate::config::IntervalMethod::Quantile).is_err()
    );
    let frame = df! { "unique_id" => ["a"], "ds" => [0_i64], "y" => [1.0] }.unwrap();
    assert!(TimeSeriesDataFrame::<f64>::from_polars(
        frame,
        TimeSeriesSchema::new("unique_id", "ds", "ds"),
        None
    )
    .is_err());
    assert!(NeuralForecast::builder()
        .with_model(Box::new(NaiveForecaster::<f64>::new("Naive", 1).unwrap()))
        .with_frequency(Frequency::Daily)
        .with_num_threads(0)
        .build()
        .is_err());
    let mut pipeline = naive(1);
    assert!(pipeline
        .fit_with_validation(
            history(&[1.0, 2.0]),
            ValidationConfig::new().with_shuffle(true)
        )
        .is_err());
}
