//! Forecast keys, finite-value checks, and model output assembly.
use crate::neural_forecast::BaseModel;
use crate::timeline::{timestamp_column, timestamps, Timeline};
use crate::{
    ForecastDataFrame, Frequency, NeuroDivergentError as Error, NeuroDivergentResult as Result,
    TimeSeriesDataFrame,
};
use num_traits::Float;
use polars::prelude::*;
use std::collections::{BTreeMap, HashMap};

impl<T: Float> ForecastDataFrame<T> {
    /// Create a checked univariate forecast with one value per future series/time key.
    /// Prediction vectors must exist for every history series and have exactly `horizon` values.
    pub fn from_series_predictions(
        history: &TimeSeriesDataFrame<T>,
        horizon: usize,
        model_name: impl Into<String>,
        predictions: HashMap<String, Vec<T>>,
        frequency: &Frequency,
    ) -> Result<Self> {
        let timeline = Timeline::read(history, frequency)?;
        let keys = timeline.future(frequency, horizon)?;
        let name = model_name.into();
        if predictions.len() != timeline.series.len() {
            return Err(Error::prediction("Prediction series do not match history"));
        }
        let mut values = Vec::with_capacity(keys.len());
        for id in timeline.series.keys() {
            let series = predictions
                .get(id)
                .ok_or_else(|| Error::prediction(format!("Missing predictions for '{id}'")))?;
            if series.len() != horizon {
                return Err(Error::prediction("Prediction horizon mismatch"));
            }
            for value in series {
                values.push(
                    value
                        .to_f64()
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| Error::prediction("Predictions must be finite"))?,
                );
            }
        }
        let ids = keys.iter().map(|key| key.0.clone()).collect::<Vec<_>>();
        let times = keys.iter().map(|key| key.1).collect::<Vec<_>>();
        let frame = DataFrame::new(vec![
            Series::new(history.schema.unique_id_col.clone().into(), ids).into(),
            timestamp_column(&history.schema.ds_col, &times, &timeline.timestamp_type)?,
            Series::new(name.clone().into(), values).into(),
        ])
        .map_err(|e| Error::prediction(e.to_string()))?;
        Ok(Self::new(
            frame,
            vec![name],
            horizon,
            None,
            history.schema.clone(),
        ))
    }
}

pub(crate) fn assemble<T: Float + Send + Sync>(
    models: &[Box<dyn BaseModel<T>>],
    history: &TimeSeriesDataFrame<T>,
    frequency: &Frequency,
) -> Result<ForecastDataFrame<T>> {
    let horizon = models
        .first()
        .ok_or_else(|| Error::config("At least one model is required"))?
        .horizon();
    let timeline = Timeline::read(history, frequency)?;
    let keys = timeline.future(frequency, horizon)?;
    let expected = keys
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect::<BTreeMap<_, _>>();
    let ids = keys.iter().map(|key| key.0.clone()).collect::<Vec<_>>();
    let times = keys.iter().map(|key| key.1).collect::<Vec<_>>();
    let mut columns = vec![
        Series::new(history.schema.unique_id_col.clone().into(), ids).into(),
        timestamp_column(&history.schema.ds_col, &times, &timeline.timestamp_type)?,
    ];
    let mut names = Vec::with_capacity(models.len());
    for model in models {
        if !model.is_fitted() {
            return Err(Error::prediction(format!(
                "Model '{}' is not fitted",
                model.name()
            )));
        }
        if model.horizon() != horizon {
            return Err(Error::prediction("All models must use the same horizon"));
        }
        let output = model.predict(history)?;
        if output.forecast_horizon != horizon || output.data.height() != keys.len() {
            return Err(Error::prediction(format!(
                "Model '{}' returned an incorrect forecast size",
                model.name()
            )));
        }
        if output.schema != history.schema || output.models != [model.name()] {
            return Err(Error::prediction(format!(
                "Model '{}' returned an incompatible forecast schema",
                model.name()
            )));
        }
        if output.confidence_levels.is_some() {
            return Err(Error::prediction(
                "Interval aggregation requires an explicit supported interval contract",
            ));
        }
        let actual_ids = output
            .data
            .column(&history.schema.unique_id_col)
            .and_then(|column| column.str())
            .map_err(|_| Error::prediction("Forecast identifiers must be strings"))?;
        let actual_times = timestamps(
            output
                .data
                .column(&history.schema.ds_col)
                .map_err(|e| Error::prediction(e.to_string()))?,
        )?;
        let column = output
            .data
            .column(model.name())
            .and_then(|column| column.cast(&DataType::Float64))
            .map_err(|_| {
                Error::prediction(format!(
                    "Missing numeric forecast column '{}'",
                    model.name()
                ))
            })?;
        let actual_values = column.f64().map_err(|e| Error::prediction(e.to_string()))?;
        let mut values = vec![None; keys.len()];
        for (row, timestamp) in actual_times.into_iter().enumerate() {
            let id = actual_ids
                .get(row)
                .ok_or_else(|| Error::prediction("Null forecast identifier"))?;
            let index = expected.get(&(id.to_owned(), timestamp)).ok_or_else(|| {
                Error::prediction("Forecast contains an unexpected series or future timestamp")
            })?;
            let value = actual_values
                .get(row)
                .filter(|value| value.is_finite())
                .ok_or_else(|| Error::prediction("Null or non-finite forecast"))?;
            if T::from(value).is_none_or(|value| !value.is_finite()) {
                return Err(Error::prediction(
                    "Forecast cannot be represented by the model numeric type",
                ));
            }
            if values[*index].replace(value).is_some() {
                return Err(Error::prediction("Duplicate forecast series/timestamp"));
            }
        }
        let values = values
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::prediction("Model omitted forecast rows"))?;
        columns.push(Series::new(model.name().into(), values).into());
        names.push(model.name().to_owned());
    }
    let frame = DataFrame::new(columns).map_err(|e| Error::prediction(e.to_string()))?;
    Ok(ForecastDataFrame::new(
        frame,
        names,
        horizon,
        None,
        history.schema.clone(),
    ))
}
