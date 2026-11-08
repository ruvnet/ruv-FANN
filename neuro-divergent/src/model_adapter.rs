//! Connect the model crate's real univariate models to the DataFrame facade.
use crate::neural_forecast::BaseModel;
use crate::timeline::Timeline;
use crate::{
    ForecastDataFrame, Frequency, NeuroDivergentError as Error, NeuroDivergentResult as Result,
    TimeSeriesDataFrame,
};
use neuro_divergent_models::foundation::{
    BaseModel as Model, DatasetMetadata, TimeSeriesDataset, TimeSeriesInput, TimeSeriesSample,
};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt, marker::PhantomData};

/// Adapt a univariate model using actual sliding training windows and latest history.
pub struct ModelAdapter<M, T> {
    model: M,
    name: String,
    lookback: usize,
    frequency: Frequency,
    marker: PhantomData<T>,
}

impl<M, T> fmt::Debug for ModelAdapter<M, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelAdapter")
            .field("name", &self.name)
            .field("lookback", &self.lookback)
            .finish_non_exhaustive()
    }
}

#[derive(Serialize, Deserialize)]
struct Header {
    version: u32,
    name: String,
    lookback: usize,
    frequency: Frequency,
}

impl<M, T> ModelAdapter<M, T>
where
    T: Float + Send + Sync + fmt::Debug + std::iter::Sum + 'static,
    M: Model<T>,
{
    /// `lookback` is the number of historical observations in each training sample.
    /// Models must accept one target feature per time step; exogenous schemas are rejected.
    pub fn new(
        name: impl Into<String>,
        model: M,
        lookback: usize,
        frequency: Frequency,
    ) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty()
            || lookback == 0
            || model.horizon() == 0
            || model.input_size() != 1
        {
            return Err(Error::config(
                "Univariate adapters require a name, positive lookback/horizon, and input_size=1",
            ));
        }
        Ok(Self {
            model,
            name,
            lookback,
            frequency,
            marker: PhantomData,
        })
    }
    /// Access the underlying model.
    pub fn inner(&self) -> &M {
        &self.model
    }
    /// Access the underlying model mutably. Refit the facade after changing model configuration.
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.model
    }

    fn timeline(&self, data: &TimeSeriesDataFrame<T>) -> Result<Timeline> {
        if !data.schema.static_features.is_empty() || !data.schema.exogenous_features.is_empty() {
            return Err(Error::config("This adapter supports univariate targets; exogenous/static model mapping is not implemented"));
        }
        Timeline::read(data, &self.frequency)
    }
}

impl<M, T> BaseModel<T> for ModelAdapter<M, T>
where
    T: Float + Send + Sync + fmt::Debug + std::iter::Sum + 'static,
    M: Model<T>,
{
    fn fit(&mut self, data: &TimeSeriesDataFrame<T>) -> Result<()> {
        self.model.reset();
        let timeline = self.timeline(data)?;
        let needed = self
            .lookback
            .checked_add(self.model.horizon())
            .ok_or_else(|| Error::config("Window overflow"))?;
        let mut samples = Vec::new();
        for (id, series) in &timeline.series {
            if series.values.len() < needed {
                return Err(Error::training(format!(
                    "Series '{id}' needs at least {needed} observations"
                )));
            }
            for start in 0..=series.values.len() - needed {
                let split = start + self.lookback;
                let historical = series.values[start..split]
                    .iter()
                    .map(|value| {
                        T::from(*value).ok_or_else(|| Error::data("Target conversion failed"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let target = series.values[split..split + self.model.horizon()]
                    .iter()
                    .map(|value| {
                        T::from(*value).ok_or_else(|| Error::data("Target conversion failed"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                samples.push(TimeSeriesSample {
                    input: TimeSeriesInput::new(historical)
                        .with_id(id)
                        .with_timestamp(series.times[split - 1]),
                    target,
                    weight: None,
                });
            }
        }
        let count = samples.len();
        let dataset = TimeSeriesDataset {
            samples,
            metadata: DatasetMetadata {
                name: Some("facade sliding windows".into()),
                description: None,
                num_series: timeline.series.len(),
                total_samples: count,
                feature_names: vec![data.schema.y_col.clone()],
                target_names: vec![data.schema.y_col.clone()],
            },
        };
        self.model
            .fit(&dataset)
            .map_err(|e| Error::training(e.to_string()))?;
        if !self.model.is_trained() {
            return Err(Error::training("Model did not enter a trained state"));
        }
        Ok(())
    }
    fn predict(&self, data: &TimeSeriesDataFrame<T>) -> Result<ForecastDataFrame<T>> {
        let timeline = self.timeline(data)?;
        if !self.model.is_trained() {
            return Err(Error::prediction("Model is not trained"));
        }
        let mut predictions = HashMap::new();
        for (id, series) in &timeline.series {
            if series.values.len() < self.lookback {
                return Err(Error::prediction(
                    "Prediction history is shorter than lookback",
                ));
            }
            let values = series.values[series.values.len() - self.lookback..]
                .iter()
                .map(|value| T::from(*value).ok_or_else(|| Error::data("Target conversion failed")))
                .collect::<Result<Vec<_>>>()?;
            let input = TimeSeriesInput::new(values)
                .with_id(id)
                .with_timestamp(*series.times.last().unwrap());
            let output = self
                .model
                .predict(&input)
                .map_err(|e| Error::prediction(e.to_string()))?;
            if output.forecasts.len() != self.model.horizon() {
                return Err(Error::prediction("Model output horizon mismatch"));
            }
            predictions.insert(id.clone(), output.forecasts);
        }
        ForecastDataFrame::from_series_predictions(
            data,
            self.horizon(),
            &self.name,
            predictions,
            &self.frequency,
        )
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn horizon(&self) -> usize {
        self.model.horizon()
    }
    fn is_fitted(&self) -> bool {
        self.model.is_trained()
    }
    fn reset(&mut self) -> Result<()> {
        self.model.reset();
        Ok(())
    }
    fn model_type(&self) -> &str {
        self.model.config().model_type()
    }
    fn save_state(&self) -> Result<Vec<u8>> {
        let header = serde_json::to_vec(&Header {
            version: 1,
            name: self.name.clone(),
            lookback: self.lookback,
            frequency: self.frequency.clone(),
        })
        .map_err(|e| Error::serialization(e.to_string()))?;
        let model = self
            .model
            .save_state()
            .map_err(|e| Error::serialization(e.to_string()))?;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(model);
        Ok(bytes)
    }
}

impl<T> ModelAdapter<neuro_divergent_models::models::RNN<T>, T>
where
    T: Float + Send + Sync + fmt::Debug + std::iter::Sum + 'static,
{
    pub(crate) fn from_state(bytes: &[u8]) -> Result<Self> {
        let length = u64::from_le_bytes(
            bytes
                .get(..8)
                .ok_or_else(|| Error::serialization("Truncated adapter state"))?
                .try_into()
                .unwrap(),
        );
        let length = usize::try_from(length)
            .ok()
            .and_then(|length| length.checked_add(8))
            .filter(|length| *length <= bytes.len())
            .ok_or_else(|| Error::serialization("Invalid adapter state length"))?;
        let header: Header = serde_json::from_slice(&bytes[8..length])
            .map_err(|e| Error::serialization(e.to_string()))?;
        if header.version != 1 {
            return Err(Error::serialization("Unsupported adapter state version"));
        }
        let model = neuro_divergent_models::models::RNN::from_state(&bytes[length..])
            .map_err(|e| Error::serialization(e.to_string()))?;
        Self::new(header.name, model, header.lookback, header.frequency)
    }
}
