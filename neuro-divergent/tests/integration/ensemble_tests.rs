//! Multiple models share keys; application-level weighted forecasts are explicit.
use super::support::*;
use neuro_divergent::{BaseModel, Frequency, NaiveForecaster, NeuralForecast};

fn ensemble(h: usize) -> NeuralForecast<f64> {
    let mut drift = DriftForecaster::new("Drift", h);
    drift.reverse_rows = true;
    NeuralForecast::new(
        vec![
            Box::new(NaiveForecaster::new("Naive", h).unwrap()),
            Box::new(drift),
        ],
        Frequency::Daily,
    )
    .unwrap()
}

#[test]
fn heterogeneous_predictions_are_joined_by_id_and_timestamp() -> TestResult {
    let mut nf = ensemble(3);
    nf.fit(frame(3, 10))?;
    let result = nf.predict()?;
    assert_finite_forecast(&result, 3, 3);
    assert_eq!(result.models, vec!["Naive", "Drift"]);
    let naive = rows(&result.data, "Naive");
    let drift = rows(&result.data, "Drift");
    for ((id, ds), y) in naive {
        let step = (ds - START) / DAY - 9;
        assert_eq!(drift[&(id, ds)], y + step as f64);
    }
    Ok(())
}

#[test]
fn weighted_ensemble_is_computed_from_real_model_columns() -> TestResult {
    let mut nf = ensemble(2);
    nf.fit(frame(1, 10))?;
    let result = nf.predict()?;
    let combined: Vec<_> = values(&result.data, "Naive")
        .iter()
        .zip(values(&result.data, "Drift"))
        .map(|(a, b)| 0.25 * a + 0.75 * b)
        .collect();
    assert_eq!(combined, vec![9.75, 10.5]);
    Ok(())
}

#[test]
fn extracting_one_model_retains_only_its_own_forecasts_and_keys() -> TestResult {
    let mut nf = ensemble(2);
    nf.fit(frame(2, 8))?;
    let result = nf.predict()?;
    let selected = result.get_model_forecasts("Drift")?;
    assert_eq!(selected.height(), 4);
    assert!(selected.column("Drift").is_ok());
    assert!(selected.column("Naive").is_err());
    assert_eq!(rows(&selected, "Drift"), rows(&result.data, "Drift"));
    assert!(result.get_model_forecasts("missing").is_err());
    Ok(())
}

#[test]
fn independent_instances_do_not_share_fitted_state() -> TestResult {
    let mut a = ensemble(2);
    let mut b = ensemble(2);
    a.fit(frame(1, 10))?;
    b.fit(frame(1, 20))?;
    assert_eq!(values(&a.predict()?.data, "Naive"), vec![9.0; 2]);
    assert_eq!(values(&b.predict()?.data, "Naive"), vec![19.0; 2]);
    a.reset()?;
    assert!(b.predict().is_ok());
    Ok(())
}

#[test]
fn invalid_horizons_are_rejected_before_ensemble_fit() -> TestResult {
    assert!(NaiveForecaster::<f64>::new("Naive", 0).is_err());
    let models: Vec<Box<dyn BaseModel<f64>>> = vec![
        Box::new(NaiveForecaster::new("a", 2)?),
        Box::new(NaiveForecaster::new("b", 3)?),
    ];
    assert!(NeuralForecast::new(models, Frequency::Daily).is_err());
    Ok(())
}

#[test]
fn repeated_predictions_are_equivalent_without_mutating_model_state() -> TestResult {
    let mut nf = ensemble(4);
    nf.fit(frame(16, 32))?;
    let expected = nf.predict()?;
    for _ in 0..16 {
        let actual = nf.predict()?;
        for model in &expected.models {
            assert_eq!(rows(&actual.data, model), rows(&expected.data, model));
        }
    }
    Ok(())
}

#[test]
fn chronological_evaluation_can_select_the_more_accurate_model() -> TestResult {
    let mut nf = ensemble(2);
    let cv = nf.cross_validation(
        frame(2, 12),
        neuro_divergent::CrossValidationConfig::new(3, 2),
    )?;
    let y = values(&cv.data, "y");
    let mse = |model: &str| {
        values(&cv.data, model)
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / y.len() as f64
    };
    assert_eq!(mse("Drift"), 0.0);
    assert!(mse("Naive") > mse("Drift"));
    Ok(())
}

#[cfg(feature = "models")]
#[test]
fn actual_rnn_and_naive_can_share_one_forecast_table() -> TestResult {
    let adapter = neuro_divergent::ModelAdapter::new("RNN", rnn(2, 5, 20), 4, Frequency::Daily)?;
    let mut nf = NeuralForecast::new(
        vec![
            Box::new(adapter),
            Box::new(NaiveForecaster::new("Naive", 2)?),
        ],
        Frequency::Daily,
    )?;
    nf.fit(frame(2, 14))?;
    assert_finite_forecast(&nf.predict()?, 2, 2);
    Ok(())
}
