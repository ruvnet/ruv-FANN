//! Expanding-window backtests using observed per-series cutoffs and past-only fits.
use crate::timeline::{timestamp_column, Timeline};
use crate::{
    AccuracyMetrics, CrossValidationConfig, CrossValidationDataFrame, NeuralForecast,
    NeuroDivergentError as Error, NeuroDivergentResult as Result, TimeSeriesDataFrame,
};
use num_traits::Float;
use polars::prelude::*;
use std::collections::BTreeSet;

pub(crate) fn cross_validate<T: Float + Send + Sync>(
    pipeline: &mut NeuralForecast<T>,
    data: TimeSeriesDataFrame<T>,
    config: CrossValidationConfig,
) -> Result<CrossValidationDataFrame<T>> {
    config.validate()?;
    if data
        .schema
        .all_columns()
        .iter()
        .any(|name| name.as_str() == "cutoff")
    {
        return Err(Error::config(
            "cutoff is reserved for cross-validation timestamps",
        ));
    }
    if pipeline
        .models
        .iter()
        .any(|model| model.horizon() != config.h)
    {
        return Err(Error::config(
            "Cross-validation horizon must match every model",
        ));
    }
    let step = config.step_size.unwrap_or(config.h);
    let span = config
        .n_windows
        .checked_sub(1)
        .and_then(|count| count.checked_mul(step))
        .and_then(|count| count.checked_add(config.h))
        .ok_or_else(|| Error::config("Cross-validation window arithmetic overflow"))?;
    if config.test_size.is_some_and(|test_size| test_size != span) {
        return Err(Error::config(
            "test_size must equal h + (n_windows - 1) * step_size",
        ));
    }
    if config.season_length == Some(0) {
        return Err(Error::config("season_length must be positive"));
    }
    let timeline = Timeline::read(&data, &pipeline.frequency)?;
    if timeline
        .series
        .values()
        .any(|series| series.values.len() <= span)
    {
        return Err(Error::data(
            "Every series needs a nonempty training prefix before all requested validation windows",
        ));
    }
    let mut combined: Option<DataFrame> = None;
    let mut cutoffs = BTreeSet::new();
    let mut errors = Vec::new();
    let mut mase = Vec::new();
    for window in 0..config.n_windows {
        let mut indices = Vec::new();
        let mut actual = Vec::new();
        let mut window_cutoffs = Vec::new();
        let mut seasonal_scales = Vec::new();
        for series in timeline.series.values() {
            let split = series.values.len() - span + window * step;
            indices.extend_from_slice(&series.indices[..split]);
            actual.extend_from_slice(&series.values[split..split + config.h]);
            let cutoff = series.times[split - 1];
            cutoffs.insert(cutoff);
            window_cutoffs.extend(std::iter::repeat_n(cutoff, config.h));
            let denominator = config.season_length.and_then(|season| {
                if split <= season {
                    return None;
                }
                let mut mean = 0.0;
                for (index, point) in series.values[season..split].iter().enumerate() {
                    let delta = (point - series.values[index]).abs();
                    mean += (delta - mean) / (index + 1) as f64;
                }
                (mean.is_finite() && mean > 0.0).then_some(mean)
            });
            seasonal_scales.extend(std::iter::repeat_n(denominator, config.h));
        }
        let frame = data
            .data
            .take(&IdxCa::from_vec("index".into(), indices))
            .map_err(|e| Error::data(e.to_string()))?;
        let history = TimeSeriesDataFrame::from_polars(
            frame,
            data.schema.clone(),
            Some(pipeline.frequency.clone()),
        )?;
        // Always reset/refit the first prefix, even if the caller previously fitted
        // the complete dataset. Reusing that state would leak the validation future.
        if window == 0 || config.refit {
            pipeline.reset()?;
            pipeline.fit(history.clone())?;
        }
        let mut forecast = pipeline.predict_on(history)?;
        for name in &forecast.models {
            let predictions = forecast
                .data
                .column(name)
                .and_then(|c| c.f64())
                .map_err(|e| Error::prediction(e.to_string()))?;
            for (row, value) in predictions.into_no_null_iter().enumerate() {
                errors.push((actual[row], value));
                if let Some(scale) = seasonal_scales[row] {
                    mase.push((actual[row] - value).abs() / scale);
                }
            }
        }
        forecast
            .data
            .with_column(timestamp_column(
                "cutoff",
                &window_cutoffs,
                &timeline.timestamp_type,
            )?)
            .map_err(|e| Error::data(e.to_string()))?;
        forecast
            .data
            .with_column(Series::new(data.schema.y_col.clone().into(), actual))
            .map_err(|e| Error::data(e.to_string()))?;
        if let Some(combined) = &mut combined {
            combined
                .vstack_mut(&forecast.data)
                .map_err(|e| Error::data(e.to_string()))?;
        } else {
            combined = Some(forecast.data);
        }
    }
    let mut metrics = score::<T>(&errors)?;
    if !mase.is_empty() {
        let mean = mase.iter().enumerate().fold(0.0, |mean, (index, value)| {
            mean + (value - mean) / (index + 1) as f64
        });
        metrics.mase = T::from(mean).filter(|value| value.is_finite());
    }
    let results = combined.ok_or_else(|| Error::data("No validation windows were generated"))?;
    Ok(CrossValidationDataFrame::new(
        results,
        cutoffs.into_iter().collect(),
        pipeline.model_names(),
        data.schema,
    )
    .with_metrics(metrics))
}

pub(crate) fn score<T: Float>(observations: &[(f64, f64)]) -> Result<AccuracyMetrics<T>> {
    if observations.is_empty() {
        return Err(Error::data("Cannot score an empty forecast"));
    }
    let (mut mae, mut mse, mut smape, mut mape, mut count) = (0.0, 0.0, 0.0, 0.0, 0_usize);
    for (index, &(actual, predicted)) in observations.iter().enumerate() {
        let error = (actual - predicted).abs();
        let squared = error * error;
        if !error.is_finite() || !squared.is_finite() {
            return Err(Error::math("Forecast error overflowed"));
        }
        mae += (error - mae) / (index + 1) as f64;
        mse += (squared - mse) / (index + 1) as f64;
        let denominator = actual.abs() + predicted.abs();
        let symmetric = if denominator == 0.0 {
            0.0
        } else {
            200.0 * (error / denominator)
        };
        smape += (symmetric - smape) / (index + 1) as f64;
        if actual != 0.0 {
            count += 1;
            mape += (100.0 * (error / actual.abs()) - mape) / count as f64;
        }
    }
    let convert = |value| {
        T::from(value)
            .filter(|value| value.is_finite())
            .ok_or_else(|| Error::math("Metric cannot be represented by the model numeric type"))
    };
    Ok(AccuracyMetrics::new(
        convert(mae)?,
        convert(mse)?,
        convert(mape)?,
        convert(smape)?,
    )
    .with_custom_metric("mape_observations".into(), convert(count as f64)?))
}
