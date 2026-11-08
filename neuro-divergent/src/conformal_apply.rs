//! DataFrame application adapter over the shared portable residual engine.
use super::{data, ConformalCalibrator, ConformalForecastDataFrame, Error, Result};
use crate::{timeline::timestamp_column, ForecastDataFrame};
use chrono::{DateTime, Utc};
use num_traits::Float;
use polars::prelude::*;
use ruv_fann::calibration::ForecastRecord;
use std::collections::{BTreeMap, BTreeSet};

impl ConformalCalibrator {
    /// Build symmetric long-form intervals without mutating the point forecast.
    /// Origins must name exactly the forecast series, be >= calibration as_of,
    /// and precede every target. Refit after changing the model/distribution;
    /// DataFrames cannot authenticate the provenance of supplied predictions.
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
        if frame.ids.iter().collect::<BTreeSet<_>>().len() != forecast_origins.len() {
            return Err(Error::data(
                "Conformal origins must name exactly the forecast series",
            ));
        }
        let mut records = Vec::with_capacity(frame.ids.len() * frame.models.len());
        for row in 0..frame.ids.len() {
            let origin = *forecast_origins
                .get(&frame.ids[row])
                .ok_or_else(|| Error::data("Missing conformal forecast origin"))?;
            for (model, predictions) in frame.models.iter().zip(&frame.predictions) {
                records.push(ForecastRecord {
                    series: frame.ids[row].clone(),
                    model: model.clone(),
                    origin,
                    timestamp: frame.times[row],
                    prediction: predictions[row],
                });
            }
        }
        let intervals = self
            .engine
            .apply(&records)
            .map_err(|e| Error::data(e.to_string()))?;
        let count = intervals.len();
        let mut result = df! {
            &self.schema.unique_id_col => intervals.iter().map(|r|r.series.as_str()).collect::<Vec<_>>(),
            "model" => intervals.iter().map(|r|r.model.as_str()).collect::<Vec<_>>(),
            "horizon" => intervals.iter().map(|r|r.horizon as u32).collect::<Vec<_>>(),
            "coverage" => intervals.iter().map(|r|r.coverage).collect::<Vec<_>>(),
            "point" => intervals.iter().map(|r|r.point).collect::<Vec<_>>(),
            "lower" => intervals.iter().map(|r|r.lower).collect::<Vec<_>>(),
            "upper" => intervals.iter().map(|r|r.upper).collect::<Vec<_>>(),
            "calibration_samples" => intervals.iter().map(|r|r.calibration_samples as u64).collect::<Vec<_>>(),
        }.map_err(|e|Error::data(e.to_string()))?;
        let times = intervals.iter().map(|r| r.timestamp).collect::<Vec<_>>();
        result
            .with_column(timestamp_column(
                &self.schema.ds_col,
                &times,
                &self.timestamp_type,
            )?)
            .map_err(|e| Error::data(e.to_string()))?;
        let metadata_type = DataType::Datetime(TimeUnit::Nanoseconds, Some("UTC".into()));
        let origins = intervals
            .iter()
            .map(|r| r.forecast_origin)
            .collect::<Vec<_>>();
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
