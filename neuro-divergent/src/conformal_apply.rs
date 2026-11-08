//! Apply immutable calibration to strictly future, complete point forecasts.
use super::{data, ConformalCalibrator, ConformalForecastDataFrame, Error, Result};
use crate::{timeline::timestamp_column, ForecastDataFrame};
use chrono::{DateTime, Utc};
use num_traits::Float;
use polars::prelude::*;
use std::collections::BTreeMap;

impl ConformalCalibrator {
    /// Build symmetric long-form intervals without mutating the point forecast.
    /// `forecast_origins` must name exactly the forecast's series. Each origin
    /// must be at or after the calibration `as_of`, and all targets strictly
    /// after their origin. Only calibrated series/models/horizons are accepted.
    /// Model retraining is not detectable from a DataFrame: the caller must
    /// recalibrate if its residual distribution or model specification changes.
    pub fn apply<T: Float>(
        &self,
        forecast: &ForecastDataFrame<T>,
        forecast_origins: &BTreeMap<String, DateTime<Utc>>,
    ) -> Result<ConformalForecastDataFrame> {
        self.config.validate()?;
        if forecast.schema != self.schema
            || forecast.forecast_horizon != self.config.horizon
            || forecast.confidence_levels.is_some()
        {
            return Err(Error::data(
                "Conformal forecast schema/horizon differs or already contains intervals",
            ));
        }
        let frame = data::read(
            &forecast.data,
            &forecast.schema,
            &forecast.models,
            &self.config.limits,
        )?;
        if frame.models != self.models || frame.timestamp_type != self.timestamp_type {
            return Err(Error::data(
                "Conformal forecast models/timestamp representation differ from calibration",
            ));
        }
        let count = frame
            .ids
            .len()
            .checked_mul(frame.models.len())
            .and_then(|n| n.checked_mul(self.config.coverages.len()))
            .filter(|n| *n <= self.config.limits.max_output_rows)
            .ok_or_else(|| Error::config("Conformal output row budget exceeded"))?;
        let mut series: BTreeMap<String, Vec<_>> = BTreeMap::new();
        for (row, id) in frame.ids.iter().enumerate() {
            series
                .entry(id.clone())
                .or_default()
                .push((frame.times[row], row));
        }
        if forecast_origins.len() != series.len() {
            return Err(Error::data(
                "Conformal origins must name exactly the forecast series",
            ));
        }
        let mut ids = Vec::with_capacity(count);
        let mut times = Vec::with_capacity(count);
        let mut models = Vec::with_capacity(count);
        let mut horizons = Vec::with_capacity(count);
        let mut levels = Vec::with_capacity(count);
        let mut point = Vec::with_capacity(count);
        let mut lower = Vec::with_capacity(count);
        let mut upper = Vec::with_capacity(count);
        let mut samples = Vec::with_capacity(count);
        let mut origins = Vec::with_capacity(count);
        for (id, mut rows) in series {
            data::sort_window(&mut rows, self.config.horizon)?;
            let origin = *forecast_origins
                .get(&id)
                .ok_or_else(|| Error::data("Missing conformal forecast origin"))?;
            if origin < self.config.as_of {
                return Err(Error::data(
                    "Forecast origin precedes calibration as_of (future leakage)",
                ));
            }
            for (step, (time, row)) in rows.into_iter().enumerate() {
                if time <= origin {
                    return Err(Error::data(
                        "Conformal forecast targets must follow their origin",
                    ));
                }
                for (model_index, model) in frame.models.iter().enumerate() {
                    let group = self
                        .groups
                        .get(&(id.clone(), model.clone(), step + 1))
                        .ok_or_else(|| {
                            Error::data("Unknown conformal series/model/horizon group")
                        })?;
                    let prediction = frame.predictions[model_index][row];
                    for (coverage, radius) in self.config.coverages.iter().zip(&group.radii) {
                        let lo = prediction - radius;
                        let hi = prediction + radius;
                        if !lo.is_finite() || !hi.is_finite() {
                            return Err(Error::math("Conformal interval endpoint overflow"));
                        }
                        ids.push(id.clone());
                        times.push(time);
                        models.push(model.clone());
                        horizons.push((step + 1) as u32);
                        levels.push(*coverage);
                        point.push(prediction);
                        lower.push(lo);
                        upper.push(hi);
                        samples.push(group.samples as u64);
                        origins.push(origin);
                    }
                }
            }
        }
        let mut result = df! {
            &self.schema.unique_id_col => ids, "model" => models, "horizon" => horizons,
            "coverage" => levels, "point" => point, "lower" => lower, "upper" => upper,
            "calibration_samples" => samples,
        }
        .map_err(|e| Error::data(e.to_string()))?;
        result
            .with_column(timestamp_column(
                &self.schema.ds_col,
                &times,
                &self.timestamp_type,
            )?)
            .map_err(|e| Error::data(e.to_string()))?;
        // Provenance instants use nanosecond UTC columns, preserving sub-millisecond
        // as_of/origin precision independently of the target's source representation.
        let metadata_type = DataType::Datetime(TimeUnit::Nanoseconds, Some("UTC".into()));
        result
            .with_column(timestamp_column(
                "forecast_origin",
                &origins,
                &metadata_type,
            )?)
            .map_err(|e| Error::data(e.to_string()))?;
        result
            .with_column(timestamp_column(
                "calibration_as_of",
                &vec![self.config.as_of; count],
                &metadata_type,
            )?)
            .map_err(|e| Error::data(e.to_string()))?;
        Ok(ConformalForecastDataFrame {
            data: result,
            calibration_as_of: self.config.as_of,
        })
    }
}
