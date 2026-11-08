//! Deterministic fixtures shared by the registered tests and benchmarks.
#![allow(dead_code)]

use neuro_divergent::{
    BaseModel, ForecastDataFrame, Frequency, NaiveForecaster, NeuralForecast, NeuroDivergentError,
    NeuroDivergentResult, TimeSeriesDataFrame, TimeSeriesSchema,
};
use polars::prelude::*;
use std::collections::BTreeMap;

pub const START: i64 = 1_704_067_200_000;
pub const DAY: i64 = 86_400_000;
pub type TestResult = Result<(), Box<dyn std::error::Error>>;

pub fn frame(series: usize, length: usize) -> TimeSeriesDataFrame<f64> {
    let mut ids = Vec::new();
    let mut times = Vec::new();
    let mut values = Vec::new();
    // Reverse series order deliberately: assembly must align by keys.
    for id in (0..series).rev() {
        for t in 0..length {
            ids.push(format!("series_{id}"));
            times.push(START + t as i64 * DAY);
            values.push(id as f64 * 100.0 + t as f64);
        }
    }
    TimeSeriesDataFrame::from_polars(
        df!("unique_id" => ids, "ds" => times, "y" => values).unwrap(),
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

pub fn values(df: &DataFrame, column: &str) -> Vec<f64> {
    df.column(column)
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

pub fn times(df: &DataFrame, column: &str) -> Vec<i64> {
    df.column(column)
        .unwrap()
        .cast(&DataType::Int64)
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

pub fn rows(df: &DataFrame, model: &str) -> BTreeMap<(String, i64), f64> {
    let ids = df.column("unique_id").unwrap().str().unwrap();
    let ts = times(df, "ds");
    let forecasts = values(df, model);
    (0..df.height())
        .map(|i| ((ids.get(i).unwrap().to_owned(), ts[i]), forecasts[i]))
        .collect()
}

pub fn assert_finite_forecast(forecast: &ForecastDataFrame<f64>, n: usize, h: usize) {
    assert_eq!(forecast.data.height(), n * h);
    assert_eq!(forecast.forecast_horizon, h);
    for name in &forecast.models {
        let v = values(&forecast.data, name);
        assert_eq!(v.len(), n * h);
        assert!(v.iter().all(|x| x.is_finite()));
    }
}

#[cfg(feature = "models")]
pub fn rnn(
    horizon: usize,
    hidden: usize,
    steps: usize,
) -> neuro_divergent::models::models::RNN<f64> {
    let mut config = neuro_divergent::models::RNNConfig::default_with_horizon(horizon);
    config.hidden_size = hidden;
    config.max_steps = steps;
    config.learning_rate = 0.02;
    neuro_divergent::models::models::RNN::new_seeded(config, 42).unwrap()
}

#[cfg(feature = "models")]
pub fn neural(horizon: usize) -> NeuralForecast<f64> {
    let adapter =
        neuro_divergent::ModelAdapter::new("RNN", rnn(horizon, 5, 40), 4, Frequency::Daily)
            .unwrap();
    NeuralForecast::new(vec![Box::new(adapter)], Frequency::Daily).unwrap()
}

/// A real last-difference extrapolator for testing custom-model integration.
/// It estimates each series' slope using training observations only, then
/// anchors forecasts at the supplied history's final observed target.
#[derive(Debug)]
pub struct DriftForecaster {
    pub name: String,
    pub horizon: usize,
    pub slopes: BTreeMap<String, f64>,
    pub reverse_rows: bool,
}

impl DriftForecaster {
    pub fn new(name: &str, horizon: usize) -> Self {
        Self {
            name: name.into(),
            horizon,
            slopes: BTreeMap::new(),
            reverse_rows: false,
        }
    }
}

fn observations(
    data: &TimeSeriesDataFrame<f64>,
) -> NeuroDivergentResult<BTreeMap<String, Vec<(i64, f64)>>> {
    let ids = data
        .data
        .column(&data.schema.unique_id_col)
        .map_err(|e| NeuroDivergentError::data(e.to_string()))?
        .str()
        .map_err(|e| NeuroDivergentError::data(e.to_string()))?;
    let ds = data
        .data
        .column(&data.schema.ds_col)
        .map_err(|e| NeuroDivergentError::data(e.to_string()))?
        .i64()
        .map_err(|e| NeuroDivergentError::data(e.to_string()))?;
    let targets = data
        .data
        .column(&data.schema.y_col)
        .map_err(|e| NeuroDivergentError::data(e.to_string()))?
        .f64()
        .map_err(|e| NeuroDivergentError::data(e.to_string()))?;
    let mut out: BTreeMap<String, Vec<(i64, f64)>> = BTreeMap::new();
    for ((id, t), y) in ids.into_iter().zip(ds).zip(targets) {
        let (id, t, y) = match (id, t, y) {
            (Some(id), Some(t), Some(y)) if y.is_finite() => (id, t, y),
            _ => {
                return Err(NeuroDivergentError::data(
                    "Missing or nonfinite observation",
                ))
            }
        };
        out.entry(id.into()).or_default().push((t, y));
    }
    for series in out.values_mut() {
        series.sort_by_key(|(t, _)| *t);
    }
    Ok(out)
}

impl BaseModel<f64> for DriftForecaster {
    fn fit(&mut self, data: &TimeSeriesDataFrame<f64>) -> NeuroDivergentResult<()> {
        self.slopes.clear();
        for (id, series) in observations(data)? {
            if series.len() < 2 {
                return Err(NeuroDivergentError::data("Drift needs two observations"));
            }
            let n = series.len();
            let slope = series[n - 1].1 - series[n - 2].1;
            self.slopes.insert(id, slope);
        }
        Ok(())
    }
    fn predict(
        &self,
        data: &TimeSeriesDataFrame<f64>,
    ) -> NeuroDivergentResult<ForecastDataFrame<f64>> {
        let mut output = Vec::new();
        for (id, series) in observations(data)? {
            let slope = self
                .slopes
                .get(&id)
                .ok_or_else(|| NeuroDivergentError::prediction("Unknown series"))?;
            let &(last_t, last_y) = series
                .last()
                .ok_or_else(|| NeuroDivergentError::data("Empty history"))?;
            for h in 1..=self.horizon {
                output.push((
                    id.clone(),
                    last_t + h as i64 * DAY,
                    last_y + h as f64 * slope,
                ));
            }
        }
        if self.reverse_rows {
            output.reverse();
        }
        let ids: Vec<_> = output.iter().map(|x| x.0.clone()).collect();
        let times: Vec<_> = output.iter().map(|x| x.1).collect();
        let forecasts: Vec<_> = output.iter().map(|x| x.2).collect();
        let df = DataFrame::new(vec![
            Series::new(data.schema.unique_id_col.clone().into(), ids).into(),
            Series::new(data.schema.ds_col.clone().into(), times).into(),
            Series::new(self.name.clone().into(), forecasts).into(),
        ])
        .map_err(|e| NeuroDivergentError::prediction(e.to_string()))?;
        Ok(ForecastDataFrame::new(
            df,
            vec![self.name.clone()],
            self.horizon,
            None,
            data.schema.clone(),
        ))
    }
    fn horizon(&self) -> usize {
        self.horizon
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn is_fitted(&self) -> bool {
        !self.slopes.is_empty()
    }
    fn reset(&mut self) -> NeuroDivergentResult<()> {
        self.slopes.clear();
        Ok(())
    }
}

#[cfg(feature = "models")]
pub fn model_dataset(
    samples: usize,
    lookback: usize,
    horizon: usize,
) -> neuro_divergent::models::foundation::TimeSeriesDataset<f64> {
    use neuro_divergent::models::foundation::{
        DatasetMetadata, TimeSeriesDataset, TimeSeriesInput, TimeSeriesSample,
    };
    TimeSeriesDataset {
        samples: (0..samples)
            .map(|i| TimeSeriesSample {
                input: TimeSeriesInput::new(
                    (0..lookback)
                        .map(|j| ((i + j) as f64 * 0.1).sin() * 0.2)
                        .collect(),
                ),
                target: vec![0.25; horizon],
                weight: None,
            })
            .collect(),
        metadata: DatasetMetadata {
            name: Some("deterministic-constant-target".into()),
            description: None,
            num_series: 1,
            total_samples: samples,
            feature_names: vec![],
            target_names: vec!["y".into()],
        },
    }
}
