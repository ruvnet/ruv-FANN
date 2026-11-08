//! Historical absolute-error intervals with a finite-sample corrected rank.
//!
//! This is a split-conformal-style residual baseline, not AcMCP, ACI or PID.
//! Rolling CV errors may be correlated and may come from differently fitted
//! models: the rank correction alone does not guarantee time-series coverage.
//! Callers must supply genuine out-of-sample predictions and refit calibration
//! after changing the forecasting model or error distribution. Calibration is
//! immutable, separate from pipeline state, and has no persistence/update API.
//! Horizon means ordinal position within each sorted complete cutoff window;
//! timestamp gaps are not interpreted as calendar steps.
//! IDs, model names and schema names are limited to 256 UTF-8 bytes; at most
//! 128 models and 128 names in each optional schema feature list are accepted.
//! Numeric columns are evaluated in Float64; integers that cannot be represented
//! exactly are rejected before conversion, rather than silently rounding errors.
use crate::{
    CrossValidationDataFrame, NeuroDivergentError as Error, NeuroDivergentResult as Result,
    TimeSeriesSchema,
};
use chrono::{DateTime, Utc};
use num_traits::Float;
use polars::prelude::*;
use ruv_fann::calibration::{
    CalibrationConfig as PortableConfig, CalibrationLimits as PortableLimits, CalibrationRecord,
    ResidualCalibrator,
};
use std::collections::BTreeSet;

#[path = "conformal_apply.rs"]
mod apply;
#[path = "conformal_data.rs"]
mod data;

/// Resource limits; callers may lower but never exceed the documented maxima.
#[derive(Debug, Clone)]
pub struct ConformalLimits {
    /// Maximum calibration/forecast rows times model count (at most 100,000).
    pub max_records: usize,
    /// Maximum distinct (series, model, horizon) groups (at most 10,000).
    pub max_groups: usize,
    /// Maximum distinct coverage levels (at most 16).
    pub max_levels: usize,
    /// Maximum long-form interval output rows (at most 1,000,000).
    pub max_output_rows: usize,
}
impl Default for ConformalLimits {
    fn default() -> Self {
        Self {
            max_records: 100_000,
            max_groups: 10_000,
            max_levels: 16,
            max_output_rows: 1_000_000,
        }
    }
}
impl ConformalLimits {
    fn validate(&self) -> Result<()> {
        let maxima = Self::default();
        for (value, maximum) in [
            (self.max_records, maxima.max_records),
            (self.max_groups, maxima.max_groups),
            (self.max_levels, maxima.max_levels),
            (self.max_output_rows, maxima.max_output_rows),
        ] {
            if value == 0 || value > maximum {
                return Err(Error::config(
                    "Conformal limits must be positive and within hard maxima",
                ));
            }
        }
        Ok(())
    }
}

/// Immutable, validated calibration settings. Coverage uses fractions, not percentages.
#[derive(Debug, Clone)]
pub struct ConformalConfig {
    as_of: DateTime<Utc>,
    horizon: usize,
    coverages: Vec<f64>,
    limits: ConformalLimits,
}
impl ConformalConfig {
    /// Require all calibration outcomes to be observed by `as_of` (inclusive).
    /// Each (series, cutoff) window must contain exactly `horizon` observations.
    pub fn new(as_of: DateTime<Utc>, horizon: usize, mut coverages: Vec<f64>) -> Result<Self> {
        if coverages.len() > 16 || coverages.is_empty() {
            return Err(Error::config(
                "Conformal coverage levels must number 1..=16",
            ));
        }
        if coverages
            .iter()
            .any(|c| !c.is_finite() || *c <= 0.0 || *c >= 1.0)
        {
            return Err(Error::config(
                "Conformal coverage must be finite and strictly between zero and one",
            ));
        }
        coverages.sort_by(f64::total_cmp);
        if coverages.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::config("Conformal coverage levels must be distinct"));
        }
        let config = Self {
            as_of,
            horizon,
            coverages,
            limits: ConformalLimits::default(),
        };
        config.validate()?;
        Ok(config)
    }
    /// Apply lower resource budgets, checked both here and at fit/apply boundaries.
    pub fn with_limits(mut self, limits: ConformalLimits) -> Result<Self> {
        self.limits = limits;
        self.validate()?;
        Ok(self)
    }
    fn validate(&self) -> Result<()> {
        self.limits.validate()?;
        if self.as_of.timestamp_nanos_opt().is_none() {
            return Err(Error::config(
                "Conformal as_of exceeds nanosecond provenance range",
            ));
        }
        if self.horizon == 0
            || self.horizon > self.limits.max_groups
            || self.coverages.len() > self.limits.max_levels
        {
            return Err(Error::config(
                "Conformal horizon/level count exceeds its budget",
            ));
        }
        Ok(())
    }
    /// Latest instant whose held-out outcomes may enter calibration.
    pub fn as_of(&self) -> DateTime<Utc> {
        self.as_of
    }
    /// Sorted requested marginal coverage fractions.
    pub fn coverages(&self) -> &[f64] {
        &self.coverages
    }
    /// Number of ordinal forecast positions per complete window.
    pub fn horizon(&self) -> usize {
        self.horizon
    }
}

/// Read-only summary from the shared portable calibration engine.
pub type CalibrationGroup = ruv_fann::calibration::CalibrationGroup<DateTime<Utc>>;

/// Checked long-form intervals. The original point forecast is never mutated.
#[derive(Debug, Clone)]
pub struct ConformalForecastDataFrame {
    /// Identifier/timestamp columns retain their schema names. Other columns are
    /// `model` (String), `horizon` (UInt32), `coverage`, `point`, `lower`, `upper`
    /// (Float64), `calibration_samples` (UInt64), `forecast_origin`, and
    /// `calibration_as_of` (Datetime). Rows sort by series/time/model/coverage.
    pub data: DataFrame,
    /// Historical calibration boundary, included in every output row as well.
    pub calibration_as_of: DateTime<Utc>,
}

/// Fitted finite-sample-corrected absolute-error calibrator.
#[derive(Debug, Clone)]
pub struct ConformalCalibrator {
    config: ConformalConfig,
    schema: TimeSeriesSchema,
    timestamp_type: DataType,
    models: Vec<String>,
    engine: ResidualCalibrator<DateTime<Utc>>,
}
impl ConformalCalibrator {
    /// Fit independent (series, model, ordinal horizon) absolute-error quantiles.
    /// Rank is `ceil((n + 1) * coverage)`, one-based. A rank greater than `n`
    /// returns an error instead of silently clamping to the largest residual.
    /// Every row must satisfy `cutoff < target_time <= config.as_of()`.
    pub fn fit<T: Float>(
        cv: &CrossValidationDataFrame<T>,
        config: ConformalConfig,
    ) -> Result<Self> {
        config.validate()?;
        let frame = data::read(&cv.data, &cv.schema, &cv.models, &config.limits)?;
        let actual = data::numbers(&cv.data, &cv.schema.y_col)?;
        let cutoffs = crate::timeline::timestamps(data::column(&cv.data, "cutoff")?)?;
        if cv.cutoffs.len() > cv.data.height() {
            return Err(Error::data("Conformal cutoff metadata exceeds row count"));
        }
        let declared: BTreeSet<_> = cv.cutoffs.iter().copied().collect();
        if declared.len() != cv.cutoffs.len() || declared != cutoffs.iter().copied().collect() {
            return Err(Error::data(
                "Conformal cutoff metadata must match distinct row cutoffs",
            ));
        }
        let mut records = Vec::with_capacity(frame.ids.len() * frame.models.len());
        for row in 0..frame.ids.len() {
            for (model, predictions) in frame.models.iter().zip(&frame.predictions) {
                records.push(CalibrationRecord {
                    series: frame.ids[row].clone(),
                    model: model.clone(),
                    cutoff: cutoffs[row],
                    timestamp: frame.times[row],
                    actual: actual[row],
                    prediction: predictions[row],
                });
            }
        }
        let engine = ResidualCalibrator::fit(
            PortableConfig {
                as_of: config.as_of,
                horizon: config.horizon,
                coverages: config.coverages.clone(),
                limits: PortableLimits {
                    max_records: config.limits.max_records,
                    max_groups: config.limits.max_groups,
                    max_levels: config.limits.max_levels,
                    max_output_rows: config.limits.max_output_rows,
                    ..Default::default()
                },
            },
            &records,
        )
        .map_err(|e| Error::data(e.to_string()))?;
        Ok(Self {
            config,
            schema: cv.schema.clone(),
            timestamp_type: frame.timestamp_type,
            models: frame.models,
            engine,
        })
    }
    /// Configuration used for the immutable fit.
    pub fn config(&self) -> &ConformalConfig {
        &self.config
    }
    /// Deterministic summaries, without exposing mutable fitted calibration state.
    pub fn groups(&self) -> impl Iterator<Item = &CalibrationGroup> {
        self.engine.groups()
    }
}
