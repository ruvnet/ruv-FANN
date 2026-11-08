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
    groups: BTreeMap<(String, String, usize), CalibrationGroup<T>>,
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
        let mut series = BTreeSet::new();
        let mut windows: BTreeMap<(&str, T), BTreeMap<&str, Vec<&CalibrationRecord<T>>>> =
            BTreeMap::new();
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
            series.insert(r.series.as_str());
            if models.len() > config.limits.max_models {
                return Err(error("Calibration model budget exceeded"));
            }
            windows
                .entry((&r.series, r.cutoff))
                .or_default()
                .entry(&r.model)
                .or_default()
                .push(r);
        }
        if series
            .len()
            .checked_mul(models.len())
            .and_then(|n| n.checked_mul(config.horizon))
            .map_or(true, |n| n > config.limits.max_groups)
        {
            return Err(error("Calibration group budget exceeded"));
        }
        let mut scores: BTreeMap<(String, String, usize), Vec<f64>> = BTreeMap::new();
        let mut groups: BTreeMap<(String, String, usize), CalibrationGroup<T>> = BTreeMap::new();
        for ((id, _), mut window) in windows {
            if window.keys().copied().collect::<BTreeSet<_>>() != models {
                return Err(error("Calibration model set differs between windows"));
            }
            let mut reference: Option<Vec<(T, f64)>> = None;
            for (model, rows) in &mut window {
                rows.sort_by_key(|r| r.timestamp);
                if rows.len() != config.horizon
                    || rows.windows(2).any(|p| p[0].timestamp == p[1].timestamp)
                {
                    return Err(error(
                        "Calibration windows require exactly H distinct target timestamps",
                    ));
                }
                let labels: Vec<_> = rows.iter().map(|r| (r.timestamp, r.actual)).collect();
                if reference.as_ref().is_some_and(|old| *old != labels) {
                    return Err(error(
                        "Calibration models must align target timestamps and actuals",
                    ));
                }
                reference = Some(labels);
                for (step, r) in rows.iter().enumerate() {
                    let residual = (r.actual - r.prediction).abs();
                    if !residual.is_finite() {
                        return Err(error("Calibration residual overflow"));
                    }
                    let key = (id.to_owned(), (*model).to_owned(), step + 1);
                    let group = groups
                        .entry(key.clone())
                        .or_insert_with(|| CalibrationGroup {
                            series: id.to_owned(),
                            model: (*model).to_owned(),
                            horizon: step + 1,
                            samples: 0,
                            first_outcome: r.timestamp,
                            last_outcome: r.timestamp,
                            radii: Vec::new(),
                        });
                    group.samples += 1;
                    group.first_outcome = group.first_outcome.min(r.timestamp);
                    group.last_outcome = group.last_outcome.max(r.timestamp);
                    scores.entry(key).or_default().push(residual);
                }
            }
        }
        for (key, mut errors) in scores {
            errors.sort_by(f64::total_cmp);
            let group = groups
                .get_mut(&key)
                .ok_or_else(|| error("Missing calibration group"))?;
            for coverage in &config.coverages {
                let rank = ((errors.len() + 1) as f64 * coverage).ceil() as usize;
                if rank == 0 || rank > errors.len() {
                    return Err(error(format!("Insufficient calibration samples: n={}, coverage={coverage}, required rank={rank}",errors.len())));
                }
                group.radii.push(errors[rank - 1]);
            }
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
        self.groups.values()
    }
}
#[cfg(test)]
#[path = "calibration_tests.rs"]
mod tests;
