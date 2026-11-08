//! Small deterministic inputs used by these runnable examples.
#![allow(dead_code)]
use neuro_divergent::{
    Frequency, NaiveForecaster, NeuralForecast, TimeSeriesDataFrame, TimeSeriesSchema,
};
use polars::prelude::*;
pub const START: i64 = 1_704_067_200_000;
pub const DAY: i64 = 86_400_000;
pub fn data(series: usize, length: usize) -> TimeSeriesDataFrame<f64> {
    let ids: Vec<_> = (0..series)
        .flat_map(|id| std::iter::repeat(format!("item_{id}")).take(length))
        .collect();
    let timestamps: Vec<_> = (0..series)
        .flat_map(|_| (0..length).map(|t| START + t as i64 * DAY))
        .collect();
    let targets: Vec<_> = (0..series)
        .flat_map(|id| (0..length).map(move |t| 0.1 * id as f64 + 0.2 * (t as f64 * 0.2).sin()))
        .collect();
    TimeSeriesDataFrame::from_polars(
        df!("unique_id"=>ids,"ds"=>timestamps,"y"=>targets).unwrap(),
        TimeSeriesSchema::default(),
        Some(Frequency::Daily),
    )
    .unwrap()
}
pub fn baseline(horizon: usize) -> NeuralForecast<f64> {
    NeuralForecast::new(
        vec![Box::new(NaiveForecaster::new("Naive", horizon).unwrap())],
        Frequency::Daily,
    )
    .unwrap()
}
#[cfg(feature = "models")]
pub fn adapter(
    name: &str,
    horizon: usize,
    hidden: usize,
) -> neuro_divergent::ModelAdapter<neuro_divergent::models::models::RNN<f64>, f64> {
    let mut config = neuro_divergent::models::RNNConfig::default_with_horizon(horizon);
    config.hidden_size = hidden;
    config.max_steps = 40;
    config.learning_rate = 0.02;
    let model = neuro_divergent::models::models::RNN::new_seeded(config, 42).unwrap();
    neuro_divergent::ModelAdapter::new(name, model, 6, Frequency::Daily).unwrap()
}
