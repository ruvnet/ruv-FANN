//! Reproducible calibration fit/apply microbenchmarks, without data generation in timings.
//! Run a release binary on an otherwise idle machine. CSV medians are workload-
//! specific observations, not predictive-accuracy or universal speed guarantees.
//! Fit/apply timings include dropping each returned calibrator/interval vector.
//! `--quantiles [samples]` compares quantile kernels with scratch reset excluded.
use ruv_fann::calibration::{
    CalibrationConfig, CalibrationLimits, CalibrationRecord, ForecastRecord, ResidualCalibrator,
};
use std::{collections::BTreeMap, hint::black_box, time::Instant};

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    series: usize,
    models: usize,
    horizon: usize,
    windows: usize,
}
fn fixtures(case: Case) -> (Vec<CalibrationRecord<i64>>, Vec<ForecastRecord<i64>>, i64) {
    let mut records = Vec::new();
    let mut forecast = Vec::new();
    let mut seed = 0x1234_5678_u64;
    let as_of = (case.windows * case.horizon + case.horizon) as i64;
    for w in 0..case.windows {
        for s in 0..case.series {
            for m in (0..case.models).rev() {
                for h in (1..=case.horizon).rev() {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let actual = ((w + h) * 3 % 1000 + s * 17) as f64;
                    let error = ((seed >> 32) & 65535) as f64 / 16. * h as f64 * (m + 1) as f64;
                    records.push(CalibrationRecord {
                        series: format!("series_{s:04}"),
                        model: format!("model_{m:03}"),
                        cutoff: (w * case.horizon) as i64,
                        timestamp: (w * case.horizon + h) as i64,
                        actual,
                        prediction: actual - error,
                    });
                }
            }
        }
    }
    records.reverse();
    for s in (0..case.series).rev() {
        for m in (0..case.models).rev() {
            for h in (1..=case.horizon).rev() {
                forecast.push(ForecastRecord {
                    series: format!("series_{s:04}"),
                    model: format!("model_{m:03}"),
                    origin: as_of,
                    timestamp: as_of + h as i64,
                    prediction: (10000 + s * 17 + m * 3 + h) as f64,
                });
            }
        }
    }
    (records, forecast, as_of)
}
fn config(as_of: i64, horizon: usize, levels: usize) -> CalibrationConfig<i64> {
    CalibrationConfig {
        as_of,
        horizon,
        coverages: (1..=levels)
            .map(|i| 0.9 * i as f64 / levels as f64)
            .collect(),
        limits: CalibrationLimits::default(),
    }
}
fn mix(state: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *state = (*state ^ u64::from(*byte)).wrapping_mul(1099511628211);
    }
}
fn group_hash(engine: &ResidualCalibrator<i64>) -> u64 {
    let mut hash = 14695981039346656037;
    for g in engine.groups() {
        mix(&mut hash, g.series.as_bytes());
        mix(&mut hash, g.model.as_bytes());
        for n in [
            g.horizon as u64,
            g.samples as u64,
            g.first_outcome as u64,
            g.last_outcome as u64,
        ] {
            mix(&mut hash, &n.to_le_bytes());
        }
        for radius in &g.radii {
            mix(&mut hash, &radius.to_bits().to_le_bytes());
        }
    }
    hash
}
fn interval_hash(rows: &[ruv_fann::calibration::IntervalRecord<i64>]) -> u64 {
    let mut hash = 14695981039346656037;
    for r in rows {
        mix(&mut hash, r.series.as_bytes());
        mix(&mut hash, r.model.as_bytes());
        for n in [
            r.timestamp as u64,
            r.horizon as u64,
            r.calibration_samples as u64,
            r.forecast_origin as u64,
            r.calibration_as_of as u64,
        ] {
            mix(&mut hash, &n.to_le_bytes());
        }
        for n in [r.coverage, r.point, r.lower, r.upper] {
            mix(&mut hash, &n.to_bits().to_le_bytes());
        }
    }
    hash
}
// Independent full-sort order-statistic reference, outside benchmark timings.
fn verify_full_sort(records: &[CalibrationRecord<i64>], engine: &ResidualCalibrator<i64>) {
    let mut windows: BTreeMap<(&str, i64, &str), Vec<&CalibrationRecord<i64>>> = BTreeMap::new();
    for r in records {
        windows
            .entry((&r.series, r.cutoff, &r.model))
            .or_default()
            .push(r);
    }
    let mut scores: BTreeMap<(&str, &str, usize), Vec<f64>> = BTreeMap::new();
    for ((series, _, model), mut rows) in windows {
        rows.sort_by_key(|r| r.timestamp);
        for (h, r) in rows.iter().enumerate() {
            scores
                .entry((series, model, h + 1))
                .or_default()
                .push((r.actual - r.prediction).abs());
        }
    }
    for g in engine.groups() {
        let errors = scores
            .get_mut(&(g.series.as_str(), g.model.as_str(), g.horizon))
            .unwrap();
        errors.sort_by(f64::total_cmp);
        for (coverage, radius) in engine.config().coverages.iter().zip(&g.radii) {
            let rank = ((errors.len() + 1) as f64 * coverage).ceil() as usize;
            assert_eq!(radius.to_bits(), errors[rank - 1].to_bits());
        }
    }
}
fn verify_intervals(
    records: &[ForecastRecord<i64>],
    engine: &ResidualCalibrator<i64>,
    intervals: &[ruv_fann::calibration::IntervalRecord<i64>],
) {
    let groups = engine
        .groups()
        .map(|g| ((g.series.as_str(), g.model.as_str(), g.horizon), g))
        .collect::<BTreeMap<_, _>>();
    let mut ordered = records.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        a.series
            .cmp(&b.series)
            .then(a.timestamp.cmp(&b.timestamp))
            .then(a.model.cmp(&b.model))
    });
    let levels = &engine.config().coverages;
    assert_eq!(intervals.len(), ordered.len() * levels.len());
    for (record, rows) in ordered.iter().zip(intervals.chunks(levels.len())) {
        // This fixture advances one integer unit per ordinal forecast position.
        let h = (record.timestamp - record.origin) as usize;
        let group = groups[&(record.series.as_str(), record.model.as_str(), h)];
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(
                (&row.series, &row.model, row.timestamp, row.horizon),
                (&record.series, &record.model, record.timestamp, h)
            );
            assert_eq!(
                [
                    row.coverage.to_bits(),
                    row.point.to_bits(),
                    row.lower.to_bits(),
                    row.upper.to_bits()
                ],
                [
                    levels[i].to_bits(),
                    record.prediction.to_bits(),
                    (record.prediction - group.radii[i]).to_bits(),
                    (record.prediction + group.radii[i]).to_bits()
                ]
            );
            assert_eq!(
                (
                    row.calibration_samples,
                    row.forecast_origin,
                    row.calibration_as_of
                ),
                (group.samples, record.origin, engine.config().as_of)
            );
        }
    }
}
fn samples(mut f: impl FnMut(), repeats: usize) -> Vec<f64> {
    f();
    let mut values = Vec::new();
    for _ in 0..repeats {
        let start = Instant::now();
        f();
        values.push(start.elapsed().as_secs_f64() * 1e6);
    }
    values.sort_by(f64::total_cmp);
    values
}
fn print_row(
    case: Case,
    levels: usize,
    phase: &str,
    n: usize,
    groups: usize,
    output: usize,
    times: &[f64],
    fingerprint: u64,
) {
    println!(
        "{},{levels},{phase},{n},{groups},{output},{},{:.3},{:.3},{:.3},{fingerprint:016x}",
        case.name,
        times.len(),
        times[times.len() / 2],
        times[times.len() / 10],
        times[(times.len() * 9 / 10).min(times.len() - 1)]
    );
}
fn select_indices(values: &mut [f64], offset: usize, ranks: &[usize]) {
    if ranks.is_empty() {
        return;
    }
    let middle = ranks.len() / 2;
    let index = ranks[middle];
    let (left, _, right) = values.select_nth_unstable_by(index - offset, f64::total_cmp);
    select_indices(left, offset, &ranks[..middle]);
    select_indices(right, index + 1, &ranks[middle + 1..]);
}
fn sorted(values: &mut [f64], ranks: &[usize]) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    ranks.iter().map(|i| values[*i]).collect()
}
fn unstable(values: &mut [f64], ranks: &[usize]) -> Vec<f64> {
    values.sort_unstable_by(f64::total_cmp);
    ranks.iter().map(|i| values[*i]).collect()
}
fn selected(values: &mut [f64], ranks: &[usize]) -> Vec<f64> {
    let mut unique = ranks.to_vec();
    unique.dedup();
    select_indices(values, 0, &unique);
    ranks.iter().map(|i| values[*i]).collect()
}
fn quantile_benchmark(repeats: usize) {
    println!("records,levels,method,samples,median_us,p10_us,p90_us");
    let mut seed = 123456_u64;
    for n in [10, 50, 128, 256, 512, 1024, 3000, 100000] {
        for levels in [1, 4, 16] {
            let source = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((seed >> 32) & 65535) as f64 / 16.
                })
                .collect::<Vec<_>>();
            let ranks = (1..=levels)
                .map(|i| (((n + 1) as f64 * (0.9 * i as f64 / levels as f64)).ceil() as usize) - 1)
                .collect::<Vec<_>>();
            let expected = sorted(&mut source.clone(), &ranks);
            let methods: [(&str, fn(&mut [f64], &[usize]) -> Vec<f64>); 3] = [
                ("sort", sorted),
                ("sort_unstable", unstable),
                ("select", selected),
            ];
            for (name, method) in methods {
                let mut scratch = source.clone();
                assert_eq!(method(&mut scratch, &ranks), expected);
                let mut timings = Vec::new();
                for _ in 0..repeats {
                    scratch.copy_from_slice(&source);
                    let start = Instant::now();
                    let output = method(black_box(&mut scratch), black_box(&ranks));
                    timings.push(start.elapsed().as_secs_f64() * 1e6);
                    black_box(output);
                }
                timings.sort_by(f64::total_cmp);
                println!(
                    "{n},{levels},{name},{repeats},{:.3},{:.3},{:.3}",
                    timings[repeats / 2],
                    timings[repeats / 10],
                    timings[(repeats * 9 / 10).min(repeats - 1)]
                );
            }
        }
    }
}
fn main() {
    if std::env::args().nth(1).as_deref() == Some("--quantiles") {
        let count = std::env::args()
            .nth(2)
            .map(|n| n.parse::<usize>().unwrap())
            .unwrap_or(31);
        assert!((1..=101).contains(&count));
        quantile_benchmark(count);
        return;
    }
    let repeats = std::env::args()
        .nth(1)
        .map(|s| s.parse::<usize>().expect("positive sample count"))
        .unwrap_or(9);
    assert!((1..=101).contains(&repeats));
    let cases = [
        Case {
            name: "single_long",
            series: 1,
            models: 1,
            horizon: 1,
            windows: 100000,
        },
        Case {
            name: "few_long",
            series: 4,
            models: 2,
            horizon: 4,
            windows: 3000,
        },
        Case {
            name: "many_small",
            series: 100,
            models: 2,
            horizon: 5,
            windows: 50,
        },
        Case {
            name: "max_groups",
            series: 500,
            models: 2,
            horizon: 10,
            windows: 10,
        },
    ];
    println!(
        "case,levels,phase,records,groups,output_rows,samples,median_us,p10_us,p90_us,fingerprint"
    );
    for case in cases {
        let (records, forecast, as_of) = fixtures(case);
        for levels in [1, 4, 16] {
            let cfg = config(as_of, case.horizon, levels);
            let fitted = ResidualCalibrator::fit(cfg.clone(), &records).unwrap();
            verify_full_sort(&records, &fitted);
            let output = fitted.apply(&forecast).unwrap();
            verify_intervals(&forecast, &fitted, &output);
            let groups = fitted.groups().count();
            let fit_times = samples(
                || {
                    black_box(
                        ResidualCalibrator::fit(black_box(cfg.clone()), black_box(&records))
                            .unwrap(),
                    );
                },
                repeats,
            );
            let apply_times = samples(
                || {
                    black_box(fitted.apply(black_box(&forecast)).unwrap());
                },
                repeats,
            );
            print_row(
                case,
                levels,
                "fit",
                records.len(),
                groups,
                output.len(),
                &fit_times,
                group_hash(&fitted),
            );
            print_row(
                case,
                levels,
                "apply",
                forecast.len(),
                groups,
                output.len(),
                &apply_times,
                interval_hash(&output),
            );
        }
    }
}
