//! Independent full-sort reference for optimized grouping and interval assembly.
use super::*;
type Reference = BTreeMap<(String, String, usize), (usize, i64, i64, Vec<f64>)>;
fn input(
    seed: &mut u64,
    series: usize,
    models: usize,
    horizon: usize,
    n: usize,
) -> Vec<CalibrationRecord<i64>> {
    let mut rows = Vec::new();
    for s in 0..series {
        for m in 0..models {
            for w in 0..n {
                for h in 1..=horizon {
                    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    // Many ties plus binary fractions expose rank boundaries without changing
                    // arithmetic between the independent reference and production paths.
                    let residual = ((*seed >> 32) % 23) as f64 / 16.;
                    rows.push(CalibrationRecord {
                        series: format!("series_{}_雪", s),
                        model: format!("model_{m}"),
                        cutoff: (w * horizon) as i64,
                        timestamp: (w * horizon + h) as i64,
                        actual: (s * 17 + w + h) as f64,
                        prediction: (s * 17 + w + h) as f64 - residual,
                    });
                }
            }
        }
    }
    for i in (1..rows.len()).rev() {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        rows.swap(i, (*seed as usize) % (i + 1));
    }
    rows
}
fn reference(rows: &[CalibrationRecord<i64>], levels: &[f64]) -> Reference {
    let mut windows: BTreeMap<(&str, &str, i64), Vec<&CalibrationRecord<i64>>> = BTreeMap::new();
    for r in rows {
        windows
            .entry((&r.series, &r.model, r.cutoff))
            .or_default()
            .push(r);
    }
    let mut scores: BTreeMap<(String, String, usize), Vec<(i64, f64)>> = BTreeMap::new();
    for ((s, m, _), mut rows) in windows {
        rows.sort_by_key(|r| r.timestamp);
        for (h, r) in rows.iter().enumerate() {
            scores
                .entry((s.into(), m.into(), h + 1))
                .or_default()
                .push((r.timestamp, (r.actual - r.prediction).abs()));
        }
    }
    scores
        .into_iter()
        .map(|(key, entries)| {
            let first = entries.iter().map(|x| x.0).min().unwrap();
            let last = entries.iter().map(|x| x.0).max().unwrap();
            let mut errors = entries.iter().map(|x| x.1).collect::<Vec<_>>();
            errors.sort_by(f64::total_cmp);
            let radii = levels
                .iter()
                .map(|c| errors[((errors.len() + 1) as f64 * c).ceil() as usize - 1])
                .collect();
            (key, (errors.len(), first, last, radii))
        })
        .collect()
}
fn bits(row: &IntervalRecord<i64>) -> (&str, &str, i64, usize, [u64; 4], usize, i64, i64) {
    (
        &row.series,
        &row.model,
        row.timestamp,
        row.horizon,
        [
            row.coverage.to_bits(),
            row.point.to_bits(),
            row.lower.to_bits(),
            row.upper.to_bits(),
        ],
        row.calibration_samples,
        row.forecast_origin,
        row.calibration_as_of,
    )
}
#[test]
fn deterministic_shuffled_fixtures_match_full_sort_and_old_interval_order() {
    for trial in 0..24 {
        let mut seed = trial + 1;
        let horizon = 1 + (trial as usize % 4);
        let series = 1 + (trial as usize % 3);
        let models = 1 + (trial as usize % 3);
        let levels = match trial % 3 {
            0 => vec![0.8],
            1 => vec![0.5, 0.8, 0.9, 0.95],
            _ => (1..=16).map(|i| 0.95 * i as f64 / 16.).collect(),
        };
        let rows = input(&mut seed, series, models, horizon, 24);
        let expected = reference(&rows, &levels);
        let config = CalibrationConfig {
            as_of: 200,
            horizon,
            coverages: levels.clone(),
            limits: CalibrationLimits::default(),
        };
        let engine = ResidualCalibrator::fit(config, &rows).unwrap();
        let observed: Vec<_> = engine.groups().collect();
        assert_eq!(observed.len(), expected.len());
        for (g, ((s, m, h), (n, first, last, radii))) in observed.iter().zip(&expected) {
            assert_eq!(
                (
                    &g.series,
                    &g.model,
                    g.horizon,
                    g.samples,
                    g.first_outcome,
                    g.last_outcome
                ),
                (s, m, *h, *n, *first, *last)
            );
            assert_eq!(
                g.radii.iter().map(|r| r.to_bits()).collect::<Vec<_>>(),
                radii.iter().map(|r| r.to_bits()).collect::<Vec<_>>()
            );
        }
        let mut forecast = Vec::new();
        let mut reference_rows = Vec::new();
        for ((s, m, h), (n, _, _, radii)) in &expected {
            let timestamp = 200 + *h as i64 * 3;
            let point = 30. + *h as f64 / 8.;
            forecast.push(ForecastRecord {
                series: s.clone(),
                model: m.clone(),
                origin: 200,
                timestamp,
                prediction: point,
            });
            for (level, radius) in levels.iter().zip(radii) {
                reference_rows.push(IntervalRecord {
                    series: s.clone(),
                    model: m.clone(),
                    timestamp,
                    horizon: *h,
                    coverage: *level,
                    point,
                    lower: point - radius,
                    upper: point + radius,
                    calibration_samples: *n,
                    forecast_origin: 200,
                    calibration_as_of: 200,
                });
            }
        }
        forecast.reverse();
        reference_rows.sort_by(|a, b| {
            a.series
                .cmp(&b.series)
                .then(a.timestamp.cmp(&b.timestamp))
                .then(a.model.cmp(&b.model))
                .then(a.coverage.total_cmp(&b.coverage))
        });
        let actual = engine.apply(&forecast).unwrap();
        assert_eq!(
            actual.iter().map(bits).collect::<Vec<_>>(),
            reference_rows.iter().map(bits).collect::<Vec<_>>()
        );
    }
}
#[test]
fn fitted_names_and_groups_outlive_original_record_storage() {
    let engine = {
        let rows = input(&mut 42, 2, 3, 4, 24);
        ResidualCalibrator::fit(
            CalibrationConfig {
                as_of: 200,
                horizon: 4,
                coverages: vec![0.8],
                limits: CalibrationLimits::default(),
            },
            &rows,
        )
        .unwrap()
    };
    assert_eq!(engine.groups().count(), 24);
    assert!(engine
        .groups()
        .all(|g| g.series.contains('雪') && g.samples == 24));
}
