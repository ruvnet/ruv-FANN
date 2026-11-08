use super::*;
impl<T: Ord + Copy> ResidualCalibrator<T> {
    /// Apply to complete aligned forecasts. Known series subsets are allowed;
    /// every included series must contain the entire fitted global model set.
    /// Output sorts by series/timestamp/model/coverage. Inputs are never mutated.
    pub fn apply(&self, records: &[ForecastRecord<T>]) -> Result<Vec<IntervalRecord<T>>> {
        if records.is_empty() || records.len() > self.config.limits.max_records {
            return Err(error("Forecast record budget exceeded or input empty"));
        }
        let count = records
            .len()
            .checked_mul(self.config.coverages.len())
            .filter(|n| *n <= self.config.limits.max_output_rows)
            .ok_or_else(|| error("Interval output row budget exceeded"))?;
        let mut windows: BTreeMap<&str, BTreeMap<&str, Vec<&ForecastRecord<T>>>> = BTreeMap::new();
        let mut origins = BTreeMap::new();
        for r in records {
            self.config.limits.name(&r.series)?;
            self.config.limits.name(&r.model)?;
            if !r.prediction.is_finite() {
                return Err(error("Non-finite forecast prediction"));
            }
            if r.origin < self.config.as_of || r.timestamp <= r.origin {
                return Err(error(
                    "Forecast requires origin >= as_of and timestamp > origin",
                ));
            }
            if origins
                .insert(r.series.as_str(), r.origin)
                .is_some_and(|old| old != r.origin)
            {
                return Err(error("Forecast series must have one origin"));
            }
            windows
                .entry(&r.series)
                .or_default()
                .entry(&r.model)
                .or_default()
                .push(r);
        }
        let mut output = Vec::with_capacity(count);
        for (id, mut window) in windows {
            if window.keys().copied().collect::<BTreeSet<_>>()
                != self.models.iter().map(String::as_str).collect()
            {
                return Err(error("Forecast model set differs from calibration"));
            }
            let mut reference: Option<Vec<T>> = None;
            for (model, rows) in &mut window {
                rows.sort_by_key(|r| r.timestamp);
                if rows.len() != self.config.horizon
                    || rows.windows(2).any(|p| p[0].timestamp == p[1].timestamp)
                {
                    return Err(error(
                        "Forecast windows require exactly H distinct target timestamps",
                    ));
                }
                let labels: Vec<_> = rows.iter().map(|r| r.timestamp).collect();
                if reference.as_ref().is_some_and(|old| *old != labels) {
                    return Err(error("Forecast models must align target timestamps"));
                }
                reference = Some(labels);
                for (step, r) in rows.iter().enumerate() {
                    let group = self
                        .groups
                        .get(&(id.to_owned(), (*model).to_owned(), step + 1))
                        .ok_or_else(|| error("Unknown calibrated series/model/horizon"))?;
                    for (coverage, radius) in self.config.coverages.iter().zip(&group.radii) {
                        let lower = r.prediction - radius;
                        let upper = r.prediction + radius;
                        if !lower.is_finite() || !upper.is_finite() {
                            return Err(error("Interval endpoint overflow"));
                        }
                        output.push(IntervalRecord {
                            series: id.to_owned(),
                            model: (*model).to_owned(),
                            timestamp: r.timestamp,
                            horizon: step + 1,
                            coverage: *coverage,
                            point: r.prediction,
                            lower,
                            upper,
                            calibration_samples: group.samples,
                            forecast_origin: r.origin,
                            calibration_as_of: self.config.as_of,
                        });
                    }
                }
            }
        }
        output.sort_by(|a, b| {
            a.series
                .cmp(&b.series)
                .then(a.timestamp.cmp(&b.timestamp))
                .then(a.model.cmp(&b.model))
                .then(a.coverage.total_cmp(&b.coverage))
        });
        Ok(output)
    }
}
