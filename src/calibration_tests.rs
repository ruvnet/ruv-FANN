use super::*;
fn config() -> CalibrationConfig<i64> {
    CalibrationConfig {
        as_of: 18,
        horizon: 2,
        coverages: vec![0.9, 0.5, 0.8],
        limits: CalibrationLimits::default(),
    }
}
fn records() -> Vec<CalibrationRecord<i64>> {
    let mut output = Vec::new();
    for (i, e) in [7., 1., 5., 9., 3., 2., 8., 6., 4.].iter().enumerate() {
        for (series, scale) in [("a", 1.), ("b", 10.)] {
            for (model, multiplier) in [("A", 1.), ("B", 2.)] {
                for h in [2, 1] {
                    output.push(CalibrationRecord {
                        series: series.into(),
                        model: model.into(),
                        cutoff: 2 * i as i64,
                        timestamp: 2 * i as i64 + h,
                        actual: 0.,
                        prediction: -e * scale * multiplier * if h == 2 { 3. } else { 1. },
                    });
                }
            }
        }
    }
    output
}
fn future() -> Vec<ForecastRecord<i64>> {
    let mut out = Vec::new();
    for series in ["b", "a"] {
        for model in ["B", "A"] {
            for time in [20, 19] {
                out.push(ForecastRecord {
                    series: series.into(),
                    model: model.into(),
                    origin: 18,
                    timestamp: time,
                    prediction: 100.,
                });
            }
        }
    }
    out
}
#[test]
fn reference_ranks_and_deterministic_sorted_intervals() {
    let engine = ResidualCalibrator::fit(config(), &records()).unwrap();
    assert_eq!(engine.config().coverages, [0.5, 0.8, 0.9]);
    assert_eq!(engine.groups().count(), 8);
    let first = engine.groups().next().unwrap();
    assert_eq!(first.radii, [5., 8., 9.]);
    let output = engine.apply(&future()).unwrap();
    assert_eq!(output.len(), 24);
    assert_eq!(
        (&*output[0].series, &*output[0].model, output[0].horizon),
        ("a", "A", 1)
    );
    assert_eq!(
        (
            output[0].lower,
            output[0].upper,
            output[0].calibration_samples
        ),
        (95., 105., 9)
    );
    assert_eq!(
        (output[0].forecast_origin, output[0].calibration_as_of),
        (18, 18)
    );
    assert!(output
        .chunks(3)
        .all(|r| r[0].lower >= r[1].lower && r[1].upper <= r[2].upper));
}
#[test]
fn no_clamped_quantile_and_no_invalid_configuration() {
    let mut cfg = config();
    cfg.coverages = vec![0.95];
    assert!(ResidualCalibrator::fit(cfg, &records())
        .unwrap_err()
        .to_string()
        .contains("required rank=10"));
    for levels in [
        vec![],
        vec![0.],
        vec![1.],
        vec![f64::NAN],
        vec![0.5, 0.5],
        vec![0.5; 17],
    ] {
        let mut cfg = config();
        cfg.coverages = levels;
        assert!(ResidualCalibrator::fit(cfg, &records()).is_err());
    }
    let mut cfg = config();
    cfg.horizon = usize::MAX;
    assert!(ResidualCalibrator::fit(cfg, &records()).is_err());
}
#[test]
fn fit_rejects_bad_chronology_duplicates_and_inconsistent_models() {
    for mutation in 0..6 {
        let mut input = records();
        match mutation {
            0 => input[0].timestamp = 19,
            1 => input[0].cutoff = input[0].timestamp,
            2 => input[0].timestamp = input[1].timestamp,
            3 => {
                input.pop();
            }
            4 => input[2].actual = 1.,
            _ => input[2].timestamp += 1,
        }
        assert!(
            ResidualCalibrator::fit(config(), &input).is_err(),
            "mutation {mutation}"
        );
    }
    let mut input = records();
    input.retain(|r| !(r.series == "a" && r.cutoff == 0 && r.model == "B"));
    assert!(ResidualCalibrator::fit(config(), &input).is_err());
}
#[test]
fn finite_arithmetic_and_text_budgets_fail_closed() {
    for value in [f64::NAN, f64::INFINITY] {
        let mut input = records();
        input[0].prediction = value;
        assert!(ResidualCalibrator::fit(config(), &input).is_err());
    }
    let mut input = records();
    input[0].actual = f64::MAX;
    input[0].prediction = -f64::MAX;
    assert!(ResidualCalibrator::fit(config(), &input).is_err());
    for name in [String::new(), "x".repeat(257)] {
        let mut input = records();
        input[0].series = name;
        assert!(ResidualCalibrator::fit(config(), &input).is_err());
    }
    let mut input = records();
    for r in &mut input {
        r.actual = 0.;
        r.prediction = -f64::MAX;
    }
    let engine = ResidualCalibrator::fit(config(), &input).unwrap();
    let mut f = future();
    for r in &mut f {
        r.prediction = f64::MAX;
    }
    assert!(engine.apply(&f).is_err());
}
#[test]
fn apply_rejects_leakage_alignment_incomplete_and_unknown_groups() {
    let engine = ResidualCalibrator::fit(config(), &records()).unwrap();
    for mutation in 0..7 {
        let mut input = future();
        match mutation {
            0 => input[0].origin = 17,
            1 => input[0].timestamp = 18,
            2 => input[0].timestamp = input[1].timestamp,
            3 => {
                input.pop();
            }
            4 => input[0].series = "unknown".into(),
            5 => input[0].timestamp = 21,
            _ => input[0].origin = 19,
        }
        assert!(engine.apply(&input).is_err(), "mutation {mutation}");
    }
    let mut subset = future();
    subset.retain(|r| r.series == "a");
    assert_eq!(engine.apply(&subset).unwrap().len(), 12);
}
#[test]
fn resource_budgets_checked_before_densification() {
    let mut cfg = config();
    cfg.limits.max_records = 1;
    assert!(ResidualCalibrator::fit(cfg, &records()).is_err());
    let mut cfg = config();
    cfg.limits.max_groups = 7;
    assert!(ResidualCalibrator::fit(cfg, &records()).is_err());
    let mut cfg = config();
    cfg.limits.max_models = 1;
    assert!(ResidualCalibrator::fit(cfg, &records()).is_err());
    let mut cfg = config();
    cfg.limits.max_levels = 0;
    assert!(ResidualCalibrator::fit(cfg, &records()).is_err());
    let mut cfg = config();
    cfg.limits.max_output_rows = 23;
    let engine = ResidualCalibrator::fit(cfg, &records()).unwrap();
    assert!(engine.apply(&future()).is_err());
}
