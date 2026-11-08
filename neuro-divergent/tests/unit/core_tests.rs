//! Schema, result and error contracts of the exported facade.
use crate::support::*;
use chrono::{TimeZone, Utc};
use neuro_divergent::{AccuracyMetrics, TimeSeriesDataFrame, TimeSeriesSchema};
use polars::prelude::*;

#[test]
fn schema_required_and_feature_columns_are_validated() -> TestResult {
    let schema = TimeSeriesSchema::default().with_static_features(vec!["store_size".into()]);
    assert!(schema.validate_dataframe(&frame(1, 5).data).is_err());
    assert_eq!(schema.required_columns().len(), 3);
    assert_eq!(schema.all_columns().len(), 4);
    let incomplete = df!("unique_id"=>["a"],"y"=>[1.])?;
    assert!(
        TimeSeriesDataFrame::<f64>::from_polars(incomplete, TimeSeriesSchema::default(), None)
            .is_err()
    );
    Ok(())
}

#[test]
fn filtering_retains_schema_ids_and_inclusive_date_bounds() -> TestResult {
    let data = frame(3, 10);
    let filtered = data.filter_by_id("series_1")?;
    assert_eq!(filtered.n_series(), 1);
    assert_eq!(filtered.shape(), (10, 3));
    let dates = filtered.filter_date_range(
        Utc.timestamp_millis_opt(START + 2 * DAY).unwrap(),
        Utc.timestamp_millis_opt(START + 4 * DAY).unwrap(),
    )?;
    assert_eq!(values(&dates.data, "y"), vec![102., 103., 104.]);
    assert_eq!(data.time_range()?.0.timestamp_millis(), START);
    assert_eq!(data.time_range()?.1.timestamp_millis(), START + 9 * DAY);
    Ok(())
}

#[test]
fn accuracy_metrics_preserve_optional_and_custom_values() {
    let metrics = AccuracyMetrics::new(1., 4., 5., 6.)
        .with_mase(0.5)
        .with_custom_metric("bias".into(), -0.25);
    assert_eq!(metrics.get_metric("rmse"), Some(2.));
    assert_eq!(metrics.get_metric("mase"), Some(0.5));
    assert_eq!(metrics.get_metric("bias"), Some(-0.25));
    assert_eq!(metrics.get_metric("missing"), None);
    assert!(metrics.metric_names().contains(&"mae".into()));
}

#[test]
fn point_forecast_extraction_preserves_all_rows_and_key_columns() -> TestResult {
    let forecast = baseline(2).fit_predict(frame(3, 8))?;
    let points = forecast.to_point_forecasts();
    assert_eq!(rows(&points, "Naive"), rows(&forecast.data, "Naive"));
    assert!(forecast.get_prediction_intervals("Naive").is_none());
    Ok(())
}

#[test]
fn structured_errors_include_the_failure_context() {
    let error = neuro_divergent::NeuroDivergentError::data("target column y is missing");
    assert!(error.to_string().contains("target column y"));
    let failure = baseline(2).predict().unwrap_err();
    assert!(!failure.to_string().is_empty());
}
