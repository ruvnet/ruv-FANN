//! Bounded frame validation shared by calibration and interval application.
use super::{ConformalLimits, Error, Result};
use crate::TimeSeriesSchema;
use chrono::{DateTime, Utc};
use polars::prelude::*;
use std::collections::BTreeSet;

pub(super) struct Frame {
    pub ids: Vec<String>,
    pub times: Vec<DateTime<Utc>>,
    pub models: Vec<String>,
    pub predictions: Vec<Vec<f64>>,
    pub timestamp_type: DataType,
}

pub(super) fn column<'a>(data: &'a DataFrame, name: &str) -> Result<&'a Column> {
    data.column(name).map_err(|e| Error::data(e.to_string()))
}
fn name(value: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > 256 {
        return Err(Error::data(
            "Conformal names must contain 1..=256 UTF-8 bytes",
        ));
    }
    Ok(())
}
pub(super) fn numbers(data: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let value = column(data, name)?;
    if !matches!(
        value.dtype(),
        DataType::Float32
            | DataType::Float64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    ) {
        return Err(Error::data(
            "Conformal values must have a primitive numeric dtype",
        ));
    }
    // Reject information loss before subtraction. Casting back through 128-bit
    // integers avoids the saturating i64/u64 casts accepting rounded MAX values.
    let lossy_integer = match value.dtype() {
        DataType::Int64 => value
            .i64()
            .map_err(|e| Error::data(e.to_string()))?
            .into_iter()
            .flatten()
            .any(|x| (x as f64) as i128 != i128::from(x)),
        DataType::UInt64 => value
            .u64()
            .map_err(|e| Error::data(e.to_string()))?
            .into_iter()
            .flatten()
            .any(|x| (x as f64) as u128 != u128::from(x)),
        _ => false,
    };
    if lossy_integer {
        return Err(Error::data(
            "Conformal integer values must be exactly representable as Float64",
        ));
    }
    let value = value
        .cast(&DataType::Float64)
        .map_err(|e| Error::data(e.to_string()))?;
    value
        .f64()
        .map_err(|e| Error::data(e.to_string()))?
        .into_iter()
        .map(|x| {
            x.filter(|x| x.is_finite())
                .ok_or_else(|| Error::data("Null/non-finite conformal value"))
        })
        .collect()
}
pub(super) fn read(
    data: &DataFrame,
    schema: &TimeSeriesSchema,
    models: &[String],
    limits: &ConformalLimits,
) -> Result<Frame> {
    if data.height() == 0
        || models.is_empty()
        || models.len() > 128
        || data
            .height()
            .checked_mul(models.len())
            .is_none_or(|n| n > limits.max_records)
    {
        return Err(Error::config(
            "Conformal record/model budget exceeded or input empty",
        ));
    }
    for value in [&schema.unique_id_col, &schema.ds_col, &schema.y_col] {
        name(value)?;
    }
    if schema.static_features.len() > 128 || schema.exogenous_features.len() > 128 {
        return Err(Error::config(
            "Conformal schema feature-name budget exceeded",
        ));
    }
    for value in schema
        .static_features
        .iter()
        .chain(&schema.exogenous_features)
    {
        name(value)?;
    }
    let reserved = [
        "model",
        "horizon",
        "coverage",
        "point",
        "lower",
        "upper",
        "calibration_samples",
        "forecast_origin",
        "calibration_as_of",
        "cutoff",
    ];
    if schema.unique_id_col == schema.ds_col
        || schema.unique_id_col == schema.y_col
        || schema.ds_col == schema.y_col
        || schema.y_col == "cutoff"
        || [&schema.unique_id_col, &schema.ds_col]
            .iter()
            .any(|s| reserved.contains(&s.as_str()))
    {
        return Err(Error::data(
            "Conformal schema names collide with required columns",
        ));
    }
    let mut sorted = BTreeSet::new();
    for model in models {
        name(model)?;
        if [
            schema.unique_id_col.as_str(),
            schema.ds_col.as_str(),
            schema.y_col.as_str(),
            "cutoff",
        ]
        .contains(&model.as_str())
            || !sorted.insert(model.clone())
        {
            return Err(Error::data(
                "Conformal model names must be unique and separate from schema/cutoff columns",
            ));
        }
    }
    let models: Vec<_> = sorted.into_iter().collect();
    let ids = column(data, &schema.unique_id_col)?
        .str()
        .map_err(|_| Error::data("Conformal IDs must be strings"))?
        .into_iter()
        .map(|id| {
            let id = id.ok_or_else(|| Error::data("Null conformal series ID"))?;
            name(id)?;
            Ok(id.to_owned())
        })
        .collect::<Result<Vec<_>>>()?;
    let timestamp = column(data, &schema.ds_col)?;
    let times = crate::timeline::timestamps(timestamp)?;
    let predictions = models
        .iter()
        .map(|model| numbers(data, model))
        .collect::<Result<_>>()?;
    Ok(Frame {
        ids,
        times,
        models,
        predictions,
        timestamp_type: timestamp.dtype().clone(),
    })
}
