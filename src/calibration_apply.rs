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
            if !window
                .keys()
                .copied()
                .eq(self.models.iter().map(String::as_str))
            {
                return Err(error("Forecast model set differs from calibration"));
            }
            for rows in window.values_mut() {
                rows.sort_by_key(|r| r.timestamp);
                if rows.len() != self.config.horizon
                    || rows.windows(2).any(|p| p[0].timestamp == p[1].timestamp)
                {
                    return Err(error(
                        "Forecast windows require exactly H distinct target timestamps",
                    ));
                }
            }
            let reference = window
                .values()
                .next()
                .ok_or_else(|| error("Empty forecast window"))?;
            for rows in window.values().skip(1) {
                if !rows
                    .iter()
                    .zip(reference)
                    .all(|(r, first)| r.timestamp == first.timestamp)
                {
                    return Err(error("Forecast models must align target timestamps"));
                }
            }
            let calibrated = self
                .groups
                .get(id)
                .ok_or_else(|| error("Unknown calibrated series/model/horizon"))?;
            // Both maps are already in model order. Borrowed lookups need no
            // temporary owned tuple keys, and horizon storage is contiguous.
            let aligned = window
                .iter()
                .map(|(model, rows)| {
                    calibrated
                        .get(*model)
                        .map(|groups| (*model, rows, groups))
                        .ok_or_else(|| error("Unknown calibrated series/model/horizon"))
                })
                .collect::<Result<Vec<_>>>()?;
            // Aligned target times let us emit final order directly, avoiding a
            // second O(output_rows log output_rows) sort and its temporary buffer.
            for step in 0..self.config.horizon {
                for (model, rows, groups) in &aligned {
                    let r = rows[step];
                    let group = groups
                        .get(step)
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
        Ok(output)
    }
}
