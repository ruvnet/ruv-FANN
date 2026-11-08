//! Exact integer conversion must precede floating-point residual subtraction.
use chrono::{DateTime, Utc};
use neuro_divergent::{
    ConformalCalibrator, ConformalConfig, CrossValidationDataFrame, TimeSeriesSchema,
};
use polars::prelude::*;
fn fit(data: DataFrame) -> neuro_divergent::NeuroDivergentResult<ConformalCalibrator> {
    let cv = CrossValidationDataFrame::<f64>::new(
        data,
        vec![DateTime::<Utc>::from_timestamp_millis(0).unwrap()],
        vec!["A".into()],
        TimeSeriesSchema::default(),
    );
    ConformalCalibrator::fit(
        &cv,
        ConformalConfig::new(DateTime::from_timestamp_millis(1).unwrap(), 1, vec![0.5])?,
    )
}
#[test]
fn lossy_i64_and_u64_targets_or_predictions_are_rejected() {
    for (actual, predicted) in [
        (9_007_199_254_740_993_i64, 9_007_199_254_740_992_i64),
        (0, i64::MAX),
        (i64::MAX, 0),
    ] {
        let data =
            df!("unique_id"=>["a"],"ds"=>[1_i64],"cutoff"=>[0_i64],"y"=>[actual],"A"=>[predicted])
                .unwrap();
        assert!(fit(data)
            .unwrap_err()
            .to_string()
            .contains("exactly representable"));
    }
    for (actual, predicted) in [(u64::MAX, 0), (0, u64::MAX)] {
        let data =
            df!("unique_id"=>["a"],"ds"=>[1_i64],"cutoff"=>[0_i64],"y"=>[actual],"A"=>[predicted])
                .unwrap();
        assert!(fit(data)
            .unwrap_err()
            .to_string()
            .contains("exactly representable"));
    }
}
#[test]
fn exactly_representable_large_integers_retain_their_residual() {
    let x = 1_i64 << 60;
    let data =
        df!("unique_id"=>["a"],"ds"=>[1_i64],"cutoff"=>[0_i64],"y"=>[x+256],"A"=>[x]).unwrap();
    assert_eq!(fit(data).unwrap().groups().next().unwrap().radii, [256.]);
    let data =
        df!("unique_id"=>["a"],"ds"=>[1_i64],"cutoff"=>[0_i64],"y"=>[i64::MIN],"A"=>[i64::MIN])
            .unwrap();
    assert_eq!(fit(data).unwrap().groups().next().unwrap().radii, [0.]);
}
