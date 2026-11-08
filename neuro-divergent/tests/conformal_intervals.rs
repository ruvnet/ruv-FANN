//! Independent fixtures for historical absolute-error interval calibration.
use chrono::{DateTime, Duration, TimeZone, Utc};
use neuro_divergent::{
    ConformalCalibrator, ConformalConfig, ConformalLimits, CrossValidationDataFrame,
    ForecastDataFrame, TimeSeriesSchema,
};
use polars::prelude::*;
use std::collections::BTreeMap;

type TestResult = Result<(), Box<dyn std::error::Error>>;
fn day(n: i64) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + Duration::days(n)
}
fn fixture() -> CrossValidationDataFrame<f64> {
    let mut ids = Vec::new();
    let mut times = Vec::new();
    let mut cutoffs = Vec::new();
    let mut a = Vec::new();
    let mut b = Vec::new();
    // Deliberately unsorted residuals; reference sorted errors are 1,...,9.
    for (i, error) in [7., 1., 5., 9., 3., 2., 8., 6., 4.].iter().enumerate() {
        for (id, scale) in [("a", 1.), ("b", 10.)] {
            for h in [2, 1] {
                // Deliberately reverse ordinal timestamps.
                ids.push(id);
                times.push(day(2 * i as i64 + h).timestamp_millis());
                cutoffs.push(day(2 * i as i64).timestamp_millis());
                let error = error * scale * if h == 2 { 3. } else { 1. };
                a.push(-error);
                b.push(-2. * error);
            }
        }
    }
    let data = df!("unique_id" => ids, "ds" => times, "cutoff" => cutoffs,
        "y" => vec![0.; a.len()], "A" => a, "B" => b)
    .unwrap();
    CrossValidationDataFrame::new(
        data,
        (0..9).map(|i| day(2 * i)).collect(),
        vec!["B".into(), "A".into()],
        TimeSeriesSchema::default(),
    )
}
fn config() -> ConformalConfig {
    ConformalConfig::new(day(18), 2, vec![0.9, 0.5, 0.8]).unwrap()
}
fn forecast() -> ForecastDataFrame<f64> {
    ForecastDataFrame::new(df!("unique_id" => ["b", "a", "b", "a"],
        "ds" => [day(20).timestamp_millis(), day(19).timestamp_millis(), day(19).timestamp_millis(), day(20).timestamp_millis()],
        "A" => [100., 100., 100., 100.], "B" => [200., 200., 200., 200.]).unwrap(),
        vec!["A".into(), "B".into()], 2, None, TimeSeriesSchema::default())
}
fn origins() -> BTreeMap<String, DateTime<Utc>> {
    [("a".into(), day(18)), ("b".into(), day(18))]
        .into_iter()
        .collect()
}

#[test]
fn corrected_rank_matches_external_r_order_statistic_fixture() -> TestResult {
    // R reference: sort(c(7,1,5,9,3,2,8,6,4))[ceiling((9+1)*c(.5,.8,.9))]
    // gives c(5,8,9); this is an order statistic, not interpolated quantile(type=7).
    let fitted = ConformalCalibrator::fit(&fixture(), config())?;
    assert_eq!(fitted.config().coverages(), &[0.5, 0.8, 0.9]);
    assert_eq!(fitted.groups().count(), 8);
    for group in fitted.groups() {
        let factor = if group.series == "b" { 10. } else { 1. }
            * if group.model == "B" { 2. } else { 1. }
            * if group.horizon == 2 { 3. } else { 1. };
        assert_eq!(group.radii, [5., 8., 9.].map(|r| r * factor));
        assert_eq!(group.samples, 9);
        assert_eq!(group.last_outcome, day(16 + group.horizon as i64));
    }
    Ok(())
}

#[test]
fn apply_is_sorted_nested_group_specific_and_preserves_input() -> TestResult {
    let cv = fixture();
    let f = forecast();
    let original_cv = cv.data.clone();
    let original_f = f.data.clone();
    let fitted = ConformalCalibrator::fit(&cv, config())?;
    let result = fitted.apply(&f, &origins())?;
    assert_eq!(result.data.height(), 24);
    assert!(cv.data.equals_missing(&original_cv));
    assert!(f.data.equals_missing(&original_f));
    let ids = result.data.column("unique_id")?.str()?;
    let models = result.data.column("model")?.str()?;
    let hs = result.data.column("horizon")?.u32()?;
    let lo = result.data.column("lower")?.f64()?;
    let hi = result.data.column("upper")?.f64()?;
    let p = result.data.column("point")?.f64()?;
    assert_eq!(
        (ids.get(0), models.get(0), hs.get(0)),
        (Some("a"), Some("A"), Some(1))
    );
    for offset in (0..24).step_by(3) {
        assert!(lo.get(offset) >= lo.get(offset + 1) && lo.get(offset + 1) >= lo.get(offset + 2));
        assert!(hi.get(offset) <= hi.get(offset + 1) && hi.get(offset + 1) <= hi.get(offset + 2));
        let scale = if ids.get(offset) == Some("b") {
            10.
        } else {
            1.
        } * if models.get(offset) == Some("B") {
            2.
        } else {
            1.
        } * if hs.get(offset) == Some(2) { 3. } else { 1. };
        assert_eq!(hi.get(offset).unwrap() - p.get(offset).unwrap(), 5. * scale);
    }
    assert_eq!(result.calibration_as_of, day(18));
    assert_eq!(
        result.data.column("calibration_samples")?.u64()?.get(0),
        Some(9)
    );
    Ok(())
}

#[test]
fn high_coverage_refuses_infinite_corrected_quantile() {
    let cfg = ConformalConfig::new(day(18), 2, vec![0.95]).unwrap();
    let error = ConformalCalibrator::fit(&fixture(), cfg)
        .unwrap_err()
        .to_string();
    assert!(error.contains("Insufficient") && error.contains("required rank=10"));
}

#[test]
fn malformed_coverage_and_limits_are_rejected() {
    for levels in [
        vec![],
        vec![0.],
        vec![1.],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![0.8, 0.8],
        vec![0.5; 17],
    ] {
        assert!(ConformalConfig::new(day(18), 2, levels).is_err());
    }
    assert!(ConformalConfig::new(day(18), 0, vec![0.5]).is_err());
    for limits in [
        ConformalLimits {
            max_records: 0,
            ..Default::default()
        },
        ConformalLimits {
            max_records: 100_001,
            ..Default::default()
        },
        ConformalLimits {
            max_groups: 10_001,
            ..Default::default()
        },
        ConformalLimits {
            max_levels: 17,
            ..Default::default()
        },
        ConformalLimits {
            max_output_rows: 1_000_001,
            ..Default::default()
        },
    ] {
        assert!(config().with_limits(limits).is_err());
    }
}

#[test]
fn record_group_level_and_output_budgets_are_checked() -> TestResult {
    for limits in [
        ConformalLimits {
            max_records: 71,
            ..Default::default()
        },
        ConformalLimits {
            max_groups: 7,
            ..Default::default()
        },
    ] {
        assert!(ConformalCalibrator::fit(&fixture(), config().with_limits(limits)?).is_err());
    }
    assert!(config()
        .with_limits(ConformalLimits {
            max_levels: 2,
            ..Default::default()
        })
        .is_err());
    let cfg = config().with_limits(ConformalLimits {
        max_output_rows: 23,
        ..Default::default()
    })?;
    assert!(ConformalCalibrator::fit(&fixture(), cfg)?
        .apply(&forecast(), &origins())
        .is_err());
    Ok(())
}

#[test]
fn calibration_rejects_future_targets_noncausal_cutoffs_and_incomplete_windows() -> TestResult {
    assert!(
        ConformalCalibrator::fit(&fixture(), ConformalConfig::new(day(17), 2, vec![0.5])?).is_err()
    );
    let mut cv = fixture();
    cv.data
        .with_column(cv.data.column("ds")?.clone().with_name("cutoff".into()))?;
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    let mut cv = fixture();
    cv.data = cv.data.slice(1, cv.data.height() - 1);
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    Ok(())
}

#[test]
fn cutoff_metadata_duplicates_and_window_duplicates_are_rejected() -> TestResult {
    let mut cv = fixture();
    cv.cutoffs.push(cv.cutoffs[0]);
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    let mut cv = fixture();
    cv.cutoffs.pop();
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    let mut cv = fixture();
    let mut times = cv
        .data
        .column("ds")?
        .i64()?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    times[0] = times[1];
    cv.data.with_column(Series::new("ds".into(), times))?;
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    Ok(())
}

#[test]
fn calibration_rejects_null_nan_infinity_and_residual_overflow() -> TestResult {
    for value in [None, Some(f64::NAN), Some(f64::INFINITY)] {
        for column in ["y", "A"] {
            let mut cv = fixture();
            let mut v = vec![Some(0.); cv.data.height()];
            v[0] = value;
            cv.data.with_column(Series::new(column.into(), v))?;
            assert!(ConformalCalibrator::fit(&cv, config()).is_err());
        }
    }
    let mut cv = fixture();
    cv.data
        .with_column(Series::new("y".into(), vec![f64::MAX; cv.data.height()]))?;
    cv.data
        .with_column(Series::new("A".into(), vec![-f64::MAX; cv.data.height()]))?;
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    let mut cv = fixture();
    let mut times = vec![Some(day(1).timestamp_millis()); cv.data.height()];
    times[0] = None;
    cv.data.with_column(Series::new("ds".into(), times))?;
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    Ok(())
}

#[test]
fn null_oversized_ids_models_and_schema_collisions_fail_before_cloning() -> TestResult {
    for id in [None, Some("".into()), Some("x".repeat(257))] {
        let mut cv = fixture();
        cv.data
            .with_column(Series::new("unique_id".into(), vec![id; cv.data.height()]))?;
        assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    }
    let mut cv = fixture();
    cv.models = vec!["x".repeat(257)];
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    let mut cv = fixture();
    cv.models = vec!["A".into(), "A".into()];
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    let mut cv = fixture();
    cv.schema.unique_id_col = "model".into();
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    Ok(())
}

#[test]
fn apply_rejects_leakage_missing_origins_unknown_groups_and_wrong_contract() -> TestResult {
    let fitted = ConformalCalibrator::fit(&fixture(), config())?;
    let mut old = origins();
    old.insert("a".into(), day(17));
    assert!(fitted.apply(&forecast(), &old).is_err());
    let mut missing = origins();
    missing.remove("a");
    assert!(fitted.apply(&forecast(), &missing).is_err());
    let mut extra = origins();
    extra.insert("other".into(), day(18));
    assert!(fitted.apply(&forecast(), &extra).is_err());
    let mut f = forecast();
    f.forecast_horizon = 1;
    assert!(fitted.apply(&f, &origins()).is_err());
    let mut f = forecast();
    f.models.pop();
    assert!(fitted.apply(&f, &origins()).is_err());
    let mut f = forecast();
    f.confidence_levels = Some(vec![0.9]);
    assert!(fitted.apply(&f, &origins()).is_err());
    let mut f = forecast();
    f.data
        .with_column(Series::new("unique_id".into(), ["c", "a", "c", "a"]))?;
    let unknown = [("a".into(), day(18)), ("c".into(), day(18))]
        .into_iter()
        .collect();
    assert!(fitted.apply(&f, &unknown).is_err());
    let mut f = forecast();
    f.data.with_column(Series::new(
        "ds".into(),
        vec![day(18).timestamp_millis(); 4],
    ))?;
    assert!(fitted.apply(&f, &origins()).is_err());
    Ok(())
}

#[test]
fn endpoint_overflow_and_invalid_point_values_do_not_mutate_forecasts() -> TestResult {
    let mut cv = fixture();
    cv.data
        .with_column(Series::new("A".into(), vec![-f64::MAX; cv.data.height()]))?;
    let fitted = ConformalCalibrator::fit(&cv, config())?;
    let mut f = forecast();
    f.data
        .with_column(Series::new("A".into(), vec![f64::MAX; 4]))?;
    let before = f.data.clone();
    assert!(fitted.apply(&f, &origins()).is_err());
    assert!(before.equals_missing(&f.data));
    for value in [None, Some(f64::NAN), Some(f64::INFINITY)] {
        let mut f = forecast();
        f.data
            .with_column(Series::new("A".into(), vec![value; 4]))?;
        assert!(fitted.apply(&f, &origins()).is_err());
    }
    Ok(())
}

#[test]
fn ordinal_horizon_does_not_infer_frequency_from_timestamp_gaps() -> TestResult {
    let fitted = ConformalCalibrator::fit(&fixture(), config())?;
    let mut f = forecast();
    f.data.with_column(Series::new(
        "ds".into(),
        [
            day(99).timestamp_millis(),
            day(19).timestamp_millis(),
            day(19).timestamp_millis(),
            day(99).timestamp_millis(),
        ],
    ))?;
    let result = fitted.apply(&f, &origins())?;
    assert_eq!(result.data.column("horizon")?.u32()?.get(6), Some(2));
    Ok(())
}

#[test]
fn timestamp_units_and_timezone_metadata_are_preserved() -> TestResult {
    for unit in [
        TimeUnit::Milliseconds,
        TimeUnit::Microseconds,
        TimeUnit::Nanoseconds,
    ] {
        let convert = |column: &Column, name: &str, unit| -> Column {
            let factor = match unit {
                TimeUnit::Milliseconds => 1,
                TimeUnit::Microseconds => 1000,
                TimeUnit::Nanoseconds => 1_000_000,
            };
            let values = column
                .i64()
                .unwrap()
                .into_no_null_iter()
                .map(|v| v * factor)
                .collect();
            Int64Chunked::from_vec(name.into(), values)
                .into_datetime(unit, Some("America/New_York".into()))
                .into_series()
                .into()
        };
        let mut cv = fixture();
        cv.data
            .with_column(convert(cv.data.column("ds")?, "ds", unit))?;
        cv.data.with_column(convert(
            cv.data.column("cutoff")?,
            "cutoff",
            TimeUnit::Nanoseconds,
        ))?;
        let mut f = forecast();
        f.data
            .with_column(convert(f.data.column("ds")?, "ds", unit))?;
        let fitted = ConformalCalibrator::fit(&cv, config())?;
        let result = fitted.apply(&f, &origins())?;
        assert_eq!(
            result.data.column("ds")?.dtype(),
            &DataType::Datetime(unit, Some("America/New_York".into()))
        );
        assert_eq!(
            result.data.column("calibration_as_of")?.dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, Some("UTC".into()))
        );
        // A differently represented forecast must be converted explicitly by its caller.
        assert!(fitted.apply(&forecast(), &origins()).is_err());
    }
    Ok(())
}

#[test]
fn custom_schema_is_retained_and_invalid_column_types_are_rejected() -> TestResult {
    let mut collision = fixture();
    collision.schema.y_col = "cutoff".into();
    assert!(ConformalCalibrator::fit(&collision, config()).is_err());
    let mut cv = fixture();
    let mut f = forecast();
    for frame in [&mut cv.data, &mut f.data] {
        frame.rename("unique_id", "item".into())?;
        frame.rename("ds", "date".into())?;
    }
    cv.data.rename("y", "observed".into())?;
    cv.schema = TimeSeriesSchema::new("item", "date", "observed");
    f.schema = cv.schema.clone();
    let result = ConformalCalibrator::fit(&cv, config())?.apply(&f, &origins())?;
    assert!(result.data.column("item").is_ok() && result.data.column("date").is_ok());
    for name in ["y", "A", "ds", "unique_id", "cutoff"] {
        let mut cv = fixture();
        cv.data.with_column(Series::new(
            name.into(),
            vec!["wrong-type"; cv.data.height()],
        ))?;
        assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    }
    Ok(())
}

#[test]
fn missing_columns_null_cutoffs_and_incomplete_future_are_rejected() -> TestResult {
    for name in ["unique_id", "ds", "cutoff", "y", "A"] {
        let mut cv = fixture();
        cv.data.drop_in_place(name)?;
        assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    }
    let mut cv = fixture();
    cv.data.with_column(Series::new(
        "cutoff".into(),
        vec![None::<i64>; cv.data.height()],
    ))?;
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    let fitted = ConformalCalibrator::fit(&fixture(), config())?;
    let mut f = forecast();
    f.data = f.data.slice(1, 3);
    assert!(fitted.apply(&f, &origins()).is_err());
    let mut f = forecast();
    f.data
        .with_column(Series::new("ds".into(), vec![None::<i64>; 4]))?;
    assert!(fitted.apply(&f, &origins()).is_err());
    Ok(())
}

#[test]
fn oversized_metadata_and_out_of_range_provenance_are_rejected() -> TestResult {
    let mut cv = fixture();
    cv.schema.static_features = vec!["x".into(); 129];
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    let mut cv = fixture();
    cv.schema.exogenous_features = vec!["x".repeat(257)];
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    let mut cv = fixture();
    cv.models = (0..129).map(|i| format!("model{i}")).collect();
    assert!(ConformalCalibrator::fit(&cv, config()).is_err());
    let beyond_ns = Utc.with_ymd_and_hms(2500, 1, 1, 0, 0, 0).unwrap();
    assert!(ConformalConfig::new(beyond_ns, 2, vec![0.8]).is_err());
    Ok(())
}
