//! Run with `cargo run --example complete_workflow --locked`.
//! Fits an actual recurrent model (with `models` enabled), forecasts future dates,
//! evaluates a naive reference with past-only CV, then verifies state restoration.
mod common;
use anyhow::Result;
use neuro_divergent::{CrossValidationConfig, Frequency, NeuralForecast};

fn main() -> Result<()> {
    let history = common::data(2, 32);
    #[cfg(feature = "models")]
    let mut forecast = NeuralForecast::new(
        vec![Box::new(common::adapter("RNN", 3, 8))],
        Frequency::Daily,
    )?;
    #[cfg(not(feature = "models"))]
    let mut forecast = common::baseline(3);
    forecast.fit(history.clone())?;
    let predictions = forecast.predict()?;
    assert_eq!(predictions.data.height(), 6);
    println!(
        "Future forecasts (one row per item and horizon step):\n{}",
        predictions.data
    );

    let mut reference = common::baseline(3);
    let cv =
        reference.cross_validation(history, CrossValidationConfig::new(3, 3).with_step_size(3))?;
    let observed = cv.data.column("y")?.f64()?;
    let predicted = cv.data.column("Naive")?.f64()?;
    let mae = observed
        .into_no_null_iter()
        .zip(predicted.into_no_null_iter())
        .map(|(y, p)| (y - p).abs())
        .sum::<f64>()
        / cv.data.height() as f64;
    println!("Naive reference, 3 chronological windows, MAE: {mae:.6}");

    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("forecast.json");
    forecast.save(&path)?;
    let restored = NeuralForecast::<f64>::load(path)?;
    let actual = restored.predict()?;
    assert_eq!(actual.data.shape(), predictions.data.shape());
    println!(
        "Restored {} fitted model(s); future target values were not supplied.",
        restored.num_models()
    );
    Ok(())
}
