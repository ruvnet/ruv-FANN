//! Explicit preprocessing and a bounded model-selection workflow.
//! Quantile intervals, exogenous model inputs and GPU training are not enabled.
mod common;
#[path = "migration/python_to_rust_examples.rs"]
mod python_to_rust;
use anyhow::Result;
use chrono::{Duration, TimeZone, Utc};
use neuro_divergent::{CrossValidationConfig, Frequency, NeuralForecast};
use neuro_divergent_data::{
    features::FeatureEngine,
    preprocessing::{Scaler, StandardScaler},
    TimeSeriesDatasetBuilder,
};

fn main() -> Result<()> {
    let start = Utc.timestamp_millis_opt(common::START).unwrap();
    let values: Vec<f64> = (0..32).map(|i| 0.2 * (i as f64 * 0.2).sin()).collect();
    let series = TimeSeriesDatasetBuilder::new("item_0".into())
        .with_frequency("D".into())
        .with_values(values.clone())
        .with_timestamps((0..32).map(|i| start + Duration::days(i)).collect())
        .build()?;
    let features = FeatureEngine::new()
        .with_lag_features(vec![1, 3])
        .with_rolling_features(vec![7])
        .generate_features(&series)?;
    println!(
        "{} causal feature rows, {} columns; rolling windows exclude the current target.",
        features.n_samples(),
        features.n_features()
    );
    let mut scaler = StandardScaler::new();
    scaler.fit(&values[..24])?; // Holdout observations do not enter fitted parameters.
    let holdout_scaled = scaler.transform(&values[24..])?;
    let recovered = scaler.inverse_transform(&holdout_scaled)?;
    assert!(recovered
        .iter()
        .zip(&values[24..])
        .all(|(a, b)| (a - b).abs() < 1e-12));

    #[cfg(feature = "models")]
    for hidden in [4, 8] {
        let model = common::adapter("RNN", 2, hidden);
        let mut nf = NeuralForecast::new(vec![Box::new(model)], Frequency::Daily)?;
        let cv = nf.cross_validation(
            common::data(1, 24),
            CrossValidationConfig::new(2, 2).with_step_size(2),
        )?;
        let y = cv.data.column("y")?.f64()?;
        let p = cv.data.column("RNN")?.f64()?;
        let mse = y
            .into_no_null_iter()
            .zip(p.into_no_null_iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / cv.data.height() as f64;
        println!("RNN hidden width {hidden}, chronological validation MSE {mse:.6}");
    }
    python_to_rust::run()?;
    Ok(())
}
