//! Real fit/predict/reset workflows, including the RNN adapter.
use super::support::*;
use neuro_divergent::{Frequency, NaiveForecaster, NeuralForecast};
use polars::prelude::*;

#[test]
fn basic_single_model_workflow_has_exact_future_dates_and_values() -> TestResult {
    let mut nf = baseline(3);
    assert!(nf.predict().is_err());
    nf.fit(frame(2, 12))?;
    assert!(nf.is_fitted());
    let forecast = nf.predict()?;
    assert_finite_forecast(&forecast, 2, 3);
    let actual = rows(&forecast.data, "Naive");
    for id in 0..2 {
        for step in 0..3 {
            assert_eq!(
                actual[&(format!("series_{id}"), START + (12 + step) * DAY)],
                id as f64 * 100.0 + 11.0
            );
        }
    }
    Ok(())
}

#[test]
fn fit_predict_matches_the_two_step_workflow() -> TestResult {
    let mut direct = baseline(2);
    let mut separate = baseline(2);
    let expected = direct.fit_predict(frame(3, 10))?;
    separate.fit(frame(3, 10))?;
    assert_eq!(
        rows(&expected.data, "Naive"),
        rows(&separate.predict()?.data, "Naive")
    );
    Ok(())
}

#[test]
fn predict_on_uses_updated_history_without_future_targets() -> TestResult {
    let mut nf = baseline(2);
    nf.fit(frame(1, 10))?;
    let updated = nf.predict_on(frame(1, 12))?;
    assert_eq!(
        times(&updated.data, "ds"),
        vec![START + 12 * DAY, START + 13 * DAY]
    );
    assert_eq!(values(&updated.data, "Naive"), vec![11.0; 2]);
    // Calling predict_on must not replace the retained training history.
    assert_eq!(values(&nf.predict()?.data, "Naive"), vec![9.0; 2]);
    Ok(())
}

#[test]
fn reset_discards_fit_state_and_allows_refitting() -> TestResult {
    let mut nf = baseline(2);
    nf.fit(frame(1, 10))?;
    nf.reset()?;
    assert!(!nf.is_fitted());
    assert!(!nf.get_model("Naive").unwrap().is_fitted());
    assert!(nf.predict().is_err());
    nf.fit(frame(1, 15))?;
    assert_eq!(values(&nf.predict()?.data, "Naive"), vec![14.0; 2]);
    Ok(())
}

#[test]
fn workflow_validates_empty_models_duplicates_and_unknown_series() -> TestResult {
    assert!(NeuralForecast::<f64>::new(vec![], Frequency::Daily).is_err());
    let duplicate = vec![
        Box::new(NaiveForecaster::<f64>::new("same", 2)?)
            as Box<dyn neuro_divergent::BaseModel<f64>>,
        Box::new(NaiveForecaster::<f64>::new("same", 2)?),
    ];
    assert!(NeuralForecast::new(duplicate, Frequency::Daily).is_err());
    let mut nf = baseline(2);
    nf.fit(frame(1, 10))?;
    assert!(nf.predict_on(frame(2, 10)).is_err());
    assert!(nf.get_model("missing").is_none());
    Ok(())
}

#[test]
fn interleaved_unsorted_history_preserves_series_identity() -> TestResult {
    let mut data = frame(3, 8);
    let order: Vec<IdxSize> = (0..data.data.height())
        .rev()
        .map(|x| x as IdxSize)
        .collect();
    data.data = data.data.take(&IdxCa::from_vec("".into(), order))?;
    let mut nf = baseline(2);
    nf.fit(data)?;
    let actual = rows(&nf.predict()?.data, "Naive");
    for id in 0..3 {
        assert_eq!(
            actual[&(format!("series_{id}"), START + 8 * DAY)],
            id as f64 * 100.0 + 7.0
        );
    }
    Ok(())
}

#[test]
fn large_panel_workflow_is_executed_in_the_regular_suite() -> TestResult {
    let mut nf = baseline(7);
    nf.fit(frame(128, 128))?;
    assert_finite_forecast(&nf.predict()?, 128, 7);
    Ok(())
}

#[cfg(feature = "models")]
#[test]
fn real_rnn_fit_predict_supports_hidden_size_different_from_horizon() -> TestResult {
    let mut nf = neural(3); // Hidden width 5, horizon 3, lookback 4.
    let data = frame(1, 18);
    nf.fit(data)?;
    let forecast = nf.predict()?;
    assert_finite_forecast(&forecast, 1, 3);
    assert_eq!(
        times(&forecast.data, "ds"),
        vec![START + 18 * DAY, START + 19 * DAY, START + 20 * DAY]
    );
    assert!(nf.get_model("RNN").unwrap().is_fitted());
    Ok(())
}

#[test]
fn seasonal_naive_repeats_the_latest_complete_season_for_any_horizon() -> TestResult {
    let model = NaiveForecaster::<f64>::seasonal("Seasonal", 8, 3)?;
    let mut nf = NeuralForecast::new(vec![Box::new(model)], Frequency::Daily)?;
    nf.fit(frame(1, 10))?;
    let forecast = nf.predict()?;
    assert_eq!(
        values(&forecast.data, "Seasonal"),
        vec![7., 8., 9., 7., 8., 9., 7., 8.]
    );
    assert_finite_forecast(&forecast, 1, 8);
    Ok(())
}

#[test]
fn invalid_observations_fail_before_models_enter_a_fitted_state() -> TestResult {
    for values in [
        vec![Some(1.), None, Some(3.)],
        vec![Some(1.), Some(f64::NAN), Some(3.)],
        vec![Some(1.), Some(f64::INFINITY), Some(3.)],
    ] {
        let mut history = frame(1, 3);
        history.data.with_column(Series::new("y".into(), values))?;
        let mut nf = baseline(2);
        assert!(nf.fit(history).is_err());
        assert!(!nf.is_fitted());
        assert!(nf.predict().is_err());
    }
    let mut duplicate = frame(1, 3);
    duplicate.data.with_column(Series::new(
        "ds".into(),
        vec![START, START, START + 2 * DAY],
    ))?;
    assert!(baseline(2).fit(duplicate).is_err());
    let mut irregular = frame(1, 3);
    irregular.data.with_column(Series::new(
        "ds".into(),
        vec![START, START + DAY, START + 3 * DAY],
    ))?;
    assert!(baseline(2).fit(irregular).is_err());
    Ok(())
}
