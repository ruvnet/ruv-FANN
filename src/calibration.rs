//! Portable historical residual calibration, without DataFrame or WASM dependencies.
//!
//! Errors are separated by series/model/ordinal forecast position. Corrected
//! empirical ranks alone do not guarantee coverage for dependent time series,
//! overlapping windows or differently fitted models. Callers must supply genuine
//! past out-of-sample errors and recalibrate after changing their model.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
#[path = "calibration_apply.rs"]
mod apply;
#[path = "calibration_quantiles.rs"]
mod quantiles;

/// Checked calibration/forecast contract violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationError(pub String);
impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for CalibrationError {}
type Result<T> = std::result::Result<T, CalibrationError>;
fn error(message: impl Into<String>) -> CalibrationError {
    CalibrationError(message.into())
}

/// Resource budgets. Values may be reduced but cannot exceed hard defaults.
#[derive(Debug, Clone)]
pub struct CalibrationLimits {
    /// Maximum long-form input records, at most 100,000.
    pub max_records: usize,
    /// Maximum series/model/horizon groups, at most 10,000.
    pub max_groups: usize,
    /// Maximum distinct coverage levels, at most 16.
    pub max_levels: usize,
    /// Maximum interval output rows, at most 1,000,000.
    pub max_output_rows: usize,
    /// Maximum global model count, at most 128.
    pub max_models: usize,
    /// Maximum UTF-8 bytes per series/model name, at most 256.
    pub max_name_bytes: usize,
}
impl Default for CalibrationLimits {
    fn default() -> Self {
        Self {
            max_records: 100_000,
            max_groups: 10_000,
            max_levels: 16,
            max_output_rows: 1_000_000,
            max_models: 128,
            max_name_bytes: 256,
        }
    }
}
impl CalibrationLimits {
    /// Validate positive budgets against the fixed hard maxima.
    pub fn validate(&self) -> Result<()> {
        let m = Self::default();
        for (n, max) in [
            (self.max_records, m.max_records),
            (self.max_groups, m.max_groups),
            (self.max_levels, m.max_levels),
            (self.max_output_rows, m.max_output_rows),
            (self.max_models, m.max_models),
            (self.max_name_bytes, m.max_name_bytes),
        ] {
            if n == 0 || n > max {
                return Err(error("Calibration budget exceeds hard bounds"));
            }
        }
        Ok(())
    }
    fn name(&self, name: &str) -> Result<()> {
        if name.trim().is_empty() || name.len() > self.max_name_bytes {
            return Err(error(
                "Calibration names are empty or exceed UTF-8 byte budget",
            ));
        }
        Ok(())
    }
}
/// Explicit historical boundary and ordinal-horizon calibration settings.
#[derive(Debug, Clone)]
pub struct CalibrationConfig<T> {
    /// Latest observed calibration target timestamp, inclusive.
    pub as_of: T,
    /// Exactly this many sorted distinct targets are required per model/window.
    pub horizon: usize,
    /// Marginal coverage fractions, sorted and validated when fitting.
    pub coverages: Vec<f64>,
    /// Resource budgets checked on fit and application.
    pub limits: CalibrationLimits,
}
impl<T> CalibrationConfig<T> {
    fn validate(&mut self) -> Result<()> {
        self.limits.validate()?;
        if self.horizon == 0
            || self.horizon > self.limits.max_groups
            || self.coverages.is_empty()
            || self.coverages.len() > self.limits.max_levels
            || self
                .coverages
                .iter()
                .any(|c| !c.is_finite() || *c <= 0. || *c >= 1.)
        {
            return Err(error("Invalid calibration horizon or coverage levels"));
        }
        self.coverages.sort_by(f64::total_cmp);
        if self.coverages.windows(2).any(|p| p[0] == p[1]) {
            return Err(error("Duplicate calibration coverage levels"));
        }
        Ok(())
    }
}
/// One observed out-of-sample residual input; models share actuals within a window.
#[derive(Debug, Clone)]
pub struct CalibrationRecord<T> {
    /// Series identifier.
    pub series: String,
    /// Point model name.
    pub model: String,
    /// Forecast training/history origin, strictly before target timestamp.
    pub cutoff: T,
    /// Observed target timestamp, at or before config as_of.
    pub timestamp: T,
    /// Finite observed target.
    pub actual: f64,
    /// Finite out-of-sample point prediction.
    pub prediction: f64,
}
/// One point prediction to receive historical residual intervals.
#[derive(Debug, Clone)]
pub struct ForecastRecord<T> {
    /// Calibrated series identifier.
    pub series: String,
    /// Calibrated model name.
    pub model: String,
    /// Single history origin per series, at or after calibration as_of.
    pub origin: T,
    /// Future target timestamp, strictly after origin.
    pub timestamp: T,
    /// Finite point prediction.
    pub prediction: f64,
}
/// Metadata and corrected absolute-error half-widths for one independent group.
#[derive(Debug, Clone)]
pub struct CalibrationGroup<T> {
    /// Series identifier.
    pub series: String,
    /// Model name.
    pub model: String,
    /// One-based ordinal forecast position, not an inferred calendar step.
    pub horizon: usize,
    /// Number of calibration errors.
    pub samples: usize,
    /// Earliest observed calibration target.
    pub first_outcome: T,
    /// Latest observed calibration target.
    pub last_outcome: T,
    /// Half-widths aligned to sorted config coverage levels.
    pub radii: Vec<f64>,
}
/// One symmetric calibrated interval with explicit historical provenance.
#[derive(Debug, Clone)]
pub struct IntervalRecord<T> {
    /// Series identifier.
    pub series: String,
    /// Model name.
    pub model: String,
    /// Target timestamp.
    pub timestamp: T,
    /// One-based ordinal forecast position.
    pub horizon: usize,
    /// Requested marginal coverage fraction.
    pub coverage: f64,
    /// Point prediction.
    pub point: f64,
    /// Finite lower endpoint.
    pub lower: f64,
    /// Finite upper endpoint.
    pub upper: f64,
    /// Calibration error count for this series/model/horizon.
    pub calibration_samples: usize,
    /// Origin of the point forecast.
    pub forecast_origin: T,
    /// Historical calibration boundary.
    pub calibration_as_of: T,
}
/// Immutable portable residual calibrator. Timestamps need only ordering and copying.
#[derive(Debug, Clone)]
pub struct ResidualCalibrator<T> {
    config: CalibrationConfig<T>,
    models: BTreeSet<String>,
    groups: BTreeMap<String, BTreeMap<String, Vec<CalibrationGroup<T>>>>,
}
impl<T: Ord + Copy> ResidualCalibrator<T> {
    /// Fit corrected ranks ceil((n+1)*coverage); insufficient samples are errors.
    /// Every series/cutoff window must have the same global model set, exactly H
    /// unique sorted timestamps/model, and aligned timestamps/actuals across models.
    pub fn fit(mut config: CalibrationConfig<T>, records: &[CalibrationRecord<T>]) -> Result<Self> {
        config.validate()?;
        if records.is_empty() || records.len() > config.limits.max_records {
            return Err(error("Calibration record budget exceeded or input empty"));
        }
        let mut models = BTreeSet::new();
        let mut ordered = Vec::with_capacity(records.len());
        for r in records {
            config.limits.name(&r.series)?;
            config.limits.name(&r.model)?;
            if !r.actual.is_finite() || !r.prediction.is_finite() {
                return Err(error("Non-finite calibration observation"));
            }
            if r.cutoff >= r.timestamp || r.timestamp > config.as_of {
                return Err(error("Calibration requires cutoff < timestamp <= as_of"));
            }
            models.insert(r.model.as_str());
            if models.len() > config.limits.max_models {
                return Err(error("Calibration model budget exceeded"));
            }
            ordered.push(r);
        }
        // One borrowed-reference array replaces per-window maps, row vectors,
        // temporary model sets and copied label vectors. No input is mutated.
        ordered.sort_unstable_by(|a, b| {
            a.cutoff
                .cmp(&b.cutoff)
                .then_with(|| a.series.cmp(&b.series))
                .then_with(|| a.model.cmp(&b.model))
                .then_with(|| a.timestamp.cmp(&b.timestamp))
        });
        let mut series = BTreeSet::new();
        for window in ordered.chunk_by(|a, b| a.cutoff == b.cutoff && a.series == b.series) {
            series.insert(window[0].series.as_str());
        }
        if series
            .len()
            .checked_mul(models.len())
            .and_then(|n| n.checked_mul(config.horizon))
            .map_or(true, |n| n > config.limits.max_groups)
        {
            return Err(error("Calibration group budget exceeded"));
        }
        let window_size = models
            .len()
            .checked_mul(config.horizon)
            .ok_or_else(|| error("Calibration window size overflow"))?;
        let mut all_scores: BTreeMap<&str, BTreeMap<&str, Vec<(T, T, Vec<f64>)>>> = BTreeMap::new();
        for window in ordered.chunk_by(|a, b| a.cutoff == b.cutoff && a.series == b.series) {
            if window.len() != window_size {
                return Err(error(
                    "Calibration windows require every model and exactly H targets",
                ));
            }
            let id = window[0].series.as_str();
            let scores = all_scores.entry(id).or_default();
            let reference = &window[..config.horizon];
            for (model, rows) in models.iter().zip(window.chunks_exact(config.horizon)) {
                if rows.iter().any(|r| r.model != *model) {
                    return Err(error("Calibration model set differs between windows"));
                }
                if rows.windows(2).any(|p| p[0].timestamp == p[1].timestamp) {
                    return Err(error(
                        "Calibration windows require exactly H distinct target timestamps",
                    ));
                }
                if !rows
                    .iter()
                    .zip(reference)
                    .all(|(r, first)| r.timestamp == first.timestamp && r.actual == first.actual)
                {
                    return Err(error(
                        "Calibration models must align target timestamps and actuals",
                    ));
                }
                let states = scores.entry(*model).or_insert_with(|| {
                    rows.iter()
                        .map(|r| (r.timestamp, r.timestamp, Vec::new()))
                        .collect()
                });
                for (r, state) in rows.iter().zip(states) {
                    let residual = (r.actual - r.prediction).abs();
                    if !residual.is_finite() {
                        return Err(error("Calibration residual overflow"));
                    }
                    state.0 = state.0.min(r.timestamp);
                    state.1 = state.1.max(r.timestamp);
                    state.2.push(residual);
                }
            }
        }
        let mut groups = BTreeMap::new();
        for (id, scores) in all_scores {
            let mut by_model = BTreeMap::new();
            for (model, states) in scores {
                let mut steps = Vec::with_capacity(config.horizon);
                for (step, (first_outcome, last_outcome, mut errors)) in
                    states.into_iter().enumerate()
                {
                    let radii = quantiles::radii(&mut errors, &config.coverages)?;
                    steps.push(CalibrationGroup {
                        series: id.to_owned(),
                        model: model.to_owned(),
                        horizon: step + 1,
                        samples: errors.len(),
                        first_outcome,
                        last_outcome,
                        radii,
                    });
                }
                by_model.insert(model.to_owned(), steps);
            }
            groups.insert(id.to_owned(), by_model);
        }
        Ok(Self {
            config,
            models: models.into_iter().map(str::to_owned).collect(),
            groups,
        })
    }
    /// Immutable fitted settings (coverage levels are sorted).
    pub fn config(&self) -> &CalibrationConfig<T> {
        &self.config
    }
    /// Deterministically sorted independent group summaries.
    pub fn groups(&self) -> impl Iterator<Item = &CalibrationGroup<T>> {
        self.groups
            .values()
            .flat_map(|models| models.values().flat_map(|groups| groups.iter()))
    }
}
#[cfg(test)]
#[path = "calibration_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "calibration_reference_tests.rs"]
mod reference_tests;
