//! Genuine last-value and seasonal-naive forecasting baselines.
use crate::neural_forecast::BaseModel;
use crate::timeline::Timeline;
use crate::{
    ForecastDataFrame, NeuroDivergentError as Error, NeuroDivergentResult as Result,
    TimeSeriesDataFrame,
};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Repeat the latest value, or the latest complete season, into the future.
#[derive(Debug, Clone)]
pub struct NaiveForecaster<T: Float + Send + Sync + std::fmt::Debug> {
    name: String,
    horizon: usize,
    season: usize,
    fitted: bool,
    marker: PhantomData<T>,
}

#[derive(Serialize, Deserialize)]
struct State {
    version: u32,
    name: String,
    horizon: usize,
    season: usize,
    fitted: bool,
}

impl<T: Float + Send + Sync + std::fmt::Debug> NaiveForecaster<T> {
    /// Construct a last-value baseline with an explicit name and horizon.
    pub fn new(name: impl Into<String>, horizon: usize) -> Result<Self> {
        Self::seasonal(name, horizon, 1)
    }

    /// Repeat the last `season` observations cyclically.
    pub fn seasonal(name: impl Into<String>, horizon: usize, season: usize) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() || horizon == 0 || horizon > 1_000_000 || season == 0 {
            return Err(Error::config(
                "A nonempty name, positive season, and horizon in 1..=1000000 are required",
            ));
        }
        Ok(Self {
            name,
            horizon,
            season,
            fitted: false,
            marker: PhantomData,
        })
    }

    pub(crate) fn from_state(bytes: &[u8]) -> Result<Self> {
        let state: State =
            serde_json::from_slice(bytes).map_err(|e| Error::serialization(e.to_string()))?;
        if state.version != 1 {
            return Err(Error::serialization("Unsupported naive checkpoint version"));
        }
        let mut model = Self::seasonal(state.name, state.horizon, state.season)?;
        model.fitted = state.fitted;
        Ok(model)
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug> BaseModel<T> for NaiveForecaster<T> {
    fn fit(&mut self, data: &TimeSeriesDataFrame<T>) -> Result<()> {
        self.fitted = false;
        let frequency = data
            .frequency
            .as_ref()
            .ok_or_else(|| Error::config("A frequency is required"))?;
        let timeline = Timeline::read(data, frequency)?;
        if timeline
            .series
            .values()
            .any(|series| series.values.len() < self.season)
        {
            return Err(Error::training(
                "Every series must contain a complete season",
            ));
        }
        self.fitted = true;
        Ok(())
    }
    fn predict(&self, data: &TimeSeriesDataFrame<T>) -> Result<ForecastDataFrame<T>> {
        if !self.fitted {
            return Err(Error::prediction("Naive model is not fitted"));
        }
        let frequency = data
            .frequency
            .as_ref()
            .ok_or_else(|| Error::config("A frequency is required"))?;
        let timeline = Timeline::read(data, frequency)?;
        let mut predictions = HashMap::new();
        for (id, series) in &timeline.series {
            if series.values.len() < self.season {
                return Err(Error::prediction(
                    "Prediction history is shorter than a season",
                ));
            }
            let start = series.values.len() - self.season;
            predictions.insert(
                id.clone(),
                (0..self.horizon)
                    .map(|step| {
                        T::from(series.values[start + step % self.season])
                            .ok_or_else(|| Error::prediction("Value conversion failed"))
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        ForecastDataFrame::from_series_predictions(
            data,
            self.horizon,
            &self.name,
            predictions,
            frequency,
        )
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn horizon(&self) -> usize {
        self.horizon
    }
    fn is_fitted(&self) -> bool {
        self.fitted
    }
    fn reset(&mut self) -> Result<()> {
        self.fitted = false;
        Ok(())
    }
    fn model_type(&self) -> &str {
        "naive"
    }
    fn save_state(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&State {
            version: 1,
            name: self.name.clone(),
            horizon: self.horizon,
            season: self.season,
            fitted: self.fitted,
        })
        .map_err(|e| Error::serialization(e.to_string()))
    }
}
