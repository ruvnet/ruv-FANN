//! Per-series affine target scaling fitted only on the training prefix.
use crate::timeline::Timeline;
use crate::{
    ForecastDataFrame, NeuroDivergentError as Error, NeuroDivergentResult as Result, ScalerType,
    TimeSeriesDataFrame,
};
use num_traits::Float;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AffineScale {
    pub location: f64,
    pub scale: f64,
}
pub(crate) type Scales = BTreeMap<String, AffineScale>;

pub(crate) fn fit(timeline: &Timeline, kind: Option<ScalerType>) -> Result<Scales> {
    let mut scales = Scales::new();
    for (id, history) in &timeline.series {
        let values = &history.values;
        let (location, scale) = match kind.unwrap_or(ScalerType::IdentityScaler) {
            ScalerType::IdentityScaler => (0.0, 1.0),
            ScalerType::StandardScaler => {
                let mut mean = 0.0;
                let mut sum = 0.0;
                for (index, value) in values.iter().copied().enumerate() {
                    let delta = value - mean;
                    mean += delta / (index + 1) as f64;
                    sum += delta * (value - mean);
                }
                (mean, (sum / values.len() as f64).max(0.0).sqrt())
            }
            ScalerType::MinMaxScaler => {
                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (min, max - min)
            }
            ScalerType::RobustScaler => {
                let mut sorted = values.clone();
                sorted.sort_by(f64::total_cmp);
                let quantile = |p: f64| {
                    let index = p * (sorted.len() - 1) as f64;
                    let lower = index.floor() as usize;
                    let upper = index.ceil() as usize;
                    sorted[lower] + (sorted[upper] - sorted[lower]) * index.fract()
                };
                (quantile(0.5), quantile(0.75) - quantile(0.25))
            }
        };
        if !location.is_finite() || !scale.is_finite() {
            return Err(Error::data("Target scaling statistics overflowed"));
        }
        scales.insert(
            id.clone(),
            AffineScale {
                location,
                scale: if scale > 0.0 { scale } else { 1.0 },
            },
        );
    }
    Ok(scales)
}

pub(crate) fn transform<T: Float>(
    data: &TimeSeriesDataFrame<T>,
    scales: &Scales,
) -> Result<TimeSeriesDataFrame<T>> {
    let ids = data
        .data
        .column(&data.schema.unique_id_col)
        .and_then(|c| c.str())
        .map_err(|e| Error::data(e.to_string()))?;
    let targets = data
        .data
        .column(&data.schema.y_col)
        .and_then(|c| c.cast(&DataType::Float64))
        .map_err(|e| Error::data(e.to_string()))?;
    let values = targets.f64().map_err(|e| Error::data(e.to_string()))?;
    let transformed = ids
        .into_iter()
        .zip(values)
        .map(|(id, value)| {
            let scale = id
                .and_then(|id| scales.get(id))
                .ok_or_else(|| Error::data("Prediction contains an unknown series"))?;
            let value =
                (value.ok_or_else(|| Error::data("Null target"))? - scale.location) / scale.scale;
            if !value.is_finite() || T::from(value).is_none_or(|value| !value.is_finite()) {
                return Err(Error::data(
                    "Scaled target cannot be represented by the model",
                ));
            }
            Ok(value)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut frame = data.data.clone();
    frame
        .with_column(Series::new(data.schema.y_col.clone().into(), transformed))
        .map_err(|e| Error::data(e.to_string()))?;
    TimeSeriesDataFrame::from_polars(frame, data.schema.clone(), data.frequency.clone())
}

pub(crate) fn inverse<T: Float>(
    forecast: &mut ForecastDataFrame<T>,
    scales: &Scales,
) -> Result<()> {
    for name in &forecast.models {
        let ids = forecast
            .data
            .column(&forecast.schema.unique_id_col)
            .and_then(|c| c.str())
            .map_err(|e| Error::prediction(e.to_string()))?;
        let values = forecast
            .data
            .column(name)
            .and_then(|c| c.f64())
            .map_err(|e| Error::prediction(e.to_string()))?;
        let restored = ids
            .into_iter()
            .zip(values)
            .map(|(id, value)| {
                let scale = id
                    .and_then(|id| scales.get(id))
                    .ok_or_else(|| Error::prediction("Unknown forecast series"))?;
                let value = value.ok_or_else(|| Error::prediction("Null forecast"))? * scale.scale
                    + scale.location;
                if !value.is_finite() || T::from(value).is_none_or(|value| !value.is_finite()) {
                    return Err(Error::prediction(
                        "Restored forecast cannot be represented by the model",
                    ));
                }
                Ok(value)
            })
            .collect::<Result<Vec<_>>>()?;
        forecast
            .data
            .with_column(Series::new(name.clone().into(), restored))
            .map_err(|e| Error::prediction(e.to_string()))?;
    }
    Ok(())
}
