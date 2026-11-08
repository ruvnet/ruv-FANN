//! Real model columns and an explicit application-level weighted ensemble.
mod common;
use anyhow::Result;
use neuro_divergent::{
    BaseModel, CrossValidationConfig, Frequency, NaiveForecaster, NeuralForecast,
};

fn main() -> Result<()> {
    let history = common::data(2, 32);
    let mut models: Vec<Box<dyn BaseModel<f64>>> =
        vec![Box::new(NaiveForecaster::new("Naive", 3)?)];
    #[cfg(feature = "models")]
    models.push(Box::new(common::adapter("RNN", 3, 8)));
    #[cfg(not(feature = "models"))]
    models.push(Box::new(NaiveForecaster::new("Reference", 3)?));
    let mut nf = NeuralForecast::new(models, Frequency::Daily)?;
    // Weights are computed from held-out observations, not invented benchmarks.
    let cv = nf.cross_validation(
        history.clone(),
        CrossValidationConfig::new(2, 3).with_step_size(3),
    )?;
    let y = cv.data.column("y")?.f64()?;
    let mut inverse_errors = Vec::new();
    for name in nf.model_names() {
        let predictions = cv.data.column(&name)?.f64()?;
        let mse = y
            .into_no_null_iter()
            .zip(predictions.into_no_null_iter())
            .map(|(y, p)| (y - p).powi(2))
            .sum::<f64>()
            / cv.data.height() as f64;
        inverse_errors.push((name, 1.0 / (mse + 1e-8)));
    }
    let total = inverse_errors.iter().map(|(_, score)| score).sum::<f64>();
    nf.fit(history)?;
    let forecasts = nf.predict()?;
    let mut weighted = vec![0.; forecasts.data.height()];
    for (name, score) in inverse_errors {
        let weight = score / total;
        println!("{name} weight from chronological CV: {weight:.3}");
        for (value, prediction) in weighted
            .iter_mut()
            .zip(forecasts.data.column(&name)?.f64()?.into_no_null_iter())
        {
            *value += weight * prediction;
        }
    }
    assert!(weighted.iter().all(|x| x.is_finite()));
    println!("Aligned model forecasts:\n{}", forecasts.data);
    println!("Weighted point forecasts in the same row order: {weighted:?}");
    Ok(())
}
