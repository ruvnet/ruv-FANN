//! Fitted forecasting pipelines with checked chronological data and model outputs.

use num_traits::Float;
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::Path;

use crate::config::{CrossValidationConfig, Device, Frequency, PredictionIntervals, ScalerType};
use crate::errors::{NeuroDivergentError, NeuroDivergentResult};
pub use crate::pipeline_config::{NeuralForecastBuilder, PredictionConfig, ValidationConfig};
use crate::results::{
    CrossValidationDataFrame, ForecastDataFrame, TimeSeriesDataFrame, TimeSeriesSchema,
};

// Forward declarations for core traits (will be implemented in core module)
pub trait BaseModel<T: Float + Send + Sync>: Send + Sync + std::fmt::Debug {
    /// Fit the model to training data
    fn fit(&mut self, data: &TimeSeriesDataFrame<T>) -> NeuroDivergentResult<()>;

    /// Generate predictions
    fn predict(&self, data: &TimeSeriesDataFrame<T>) -> NeuroDivergentResult<ForecastDataFrame<T>>;

    /// Get model name
    fn name(&self) -> &str;

    /// Number of future observations returned for each series.
    fn horizon(&self) -> usize;

    /// Check if model is fitted
    fn is_fitted(&self) -> bool;

    /// Reset model to unfitted state
    fn reset(&mut self) -> NeuroDivergentResult<()>;

    /// Stable implementation identifier used by supported checkpoint loaders.
    fn model_type(&self) -> &str {
        "custom"
    }

    /// Export a real implementation-specific checkpoint, or return an error.
    fn save_state(&self) -> NeuroDivergentResult<Vec<u8>> {
        Err(NeuroDivergentError::serialization(
            "Persistence is unsupported for this custom model",
        ))
    }
}

/// Facade for user-supplied implementations of this module's [`BaseModel`] trait.
///
/// The models crate's trait is separate and requires an explicit adapter.
/// Predictions are aligned by series/time keys; backtests train on chronological prefixes.
///
/// # Examples
///
/// A model is required before this facade can be built:
/// ```rust
/// use neuro_divergent::NeuralForecast;
/// assert!(NeuralForecast::<f64>::builder().build().is_err());
/// ```
#[derive(Debug)]
pub struct NeuralForecast<T: Float + Send + Sync> {
    /// Collection of forecasting models
    pub(crate) models: Vec<Box<dyn BaseModel<T>>>,
    /// Time series frequency
    pub(crate) frequency: Frequency,
    /// Local scaler type for preprocessing
    pub(crate) local_scaler_type: Option<ScalerType>,
    /// Number of threads for parallel processing
    pub(crate) num_threads: Option<usize>,
    /// Prediction intervals configuration
    pub(crate) prediction_intervals: Option<PredictionIntervals>,
    /// Device for computation
    pub(crate) device: Device,
    /// Whether models have been fitted
    pub(crate) is_fitted: bool,
    /// Training data schema
    pub(crate) training_schema: Option<TimeSeriesSchema>,
    /// Model metadata
    model_metadata: HashMap<String, ModelInfo>,
    /// Retained raw training context, committed only after a successful fit.
    pub(crate) training_data: Option<TimeSeriesDataFrame<T>>,
    pub(crate) scales: crate::scaling::Scales,
    /// Phantom data for type parameter
    phantom: PhantomData<T>,
}

/// Model information and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelInfo {
    /// Model name
    name: String,
    /// Whether model is fitted
    fitted: bool,
    /// Training history if available
    training_history: Option<TrainingHistory<f64>>,
    /// Model configuration parameters
    config_summary: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingHistory<T: Float> {
    /// Training loss history
    train_loss: Vec<T>,
    /// Validation loss history
    val_loss: Option<Vec<T>>,
    /// Number of epochs trained
    epochs: usize,
    /// Best validation loss
    best_val_loss: Option<T>,
}

impl<T: Float + Send + Sync> NeuralForecast<T> {
    /// Create a new NeuralForecast builder
    ///
    /// # Example
    /// ```rust
    /// use neuro_divergent::{NeuralForecast, Frequency};
    ///
    /// let nf_builder = NeuralForecast::<f32>::builder();
    /// ```
    pub fn builder() -> NeuralForecastBuilder<T> {
        NeuralForecastBuilder::new()
    }

    /// Create NeuralForecast with models and frequency (direct constructor)
    ///
    /// # Arguments
    /// * `models` - Vector of boxed models implementing BaseModel trait
    /// * `frequency` - Time series frequency
    ///
    /// # Example
    /// ```rust
    /// use neuro_divergent::{Frequency, NeuralForecast, NeuroDivergentResult};
    /// use neuro_divergent::neural_forecast::BaseModel;
    ///
    /// fn configure(model: Box<dyn BaseModel<f64>>) -> NeuroDivergentResult<NeuralForecast<f64>> {
    ///     NeuralForecast::new(vec![model], Frequency::Daily)
    /// }
    /// ```
    pub fn new(
        models: Vec<Box<dyn BaseModel<T>>>,
        frequency: Frequency,
    ) -> NeuroDivergentResult<Self> {
        if models.is_empty() {
            return Err(NeuroDivergentError::config(
                "At least one model is required",
            ));
        }

        // Check for duplicate model names
        let mut model_names = std::collections::HashSet::new();
        let horizon = models[0].horizon();
        if horizon == 0 || horizon > 1_000_000 {
            return Err(NeuroDivergentError::config(
                "Horizon must be in 1..=1000000",
            ));
        }
        for model in &models {
            if model.name().trim().is_empty()
                || model.name() == "cutoff"
                || model.horizon() != horizon
            {
                return Err(NeuroDivergentError::config("Models need nonempty unique names and identical positive horizons; cutoff is reserved"));
            }
            if !model_names.insert(model.name()) {
                return Err(NeuroDivergentError::config(format!(
                    "Duplicate model name: {}",
                    model.name()
                )));
            }
        }

        let model_metadata = models
            .iter()
            .map(|model| {
                let info = ModelInfo {
                    name: model.name().to_string(),
                    fitted: model.is_fitted(),
                    training_history: None,
                    config_summary: HashMap::new(),
                };
                (model.name().to_string(), info)
            })
            .collect();

        Ok(Self {
            models,
            frequency,
            local_scaler_type: None,
            num_threads: None,
            prediction_intervals: None,
            device: Device::default(),
            is_fitted: false,
            training_schema: None,
            training_data: None,
            scales: Default::default(),
            model_metadata,
            phantom: PhantomData,
        })
    }

    /// Fit each user-supplied model to the provided time series data.
    ///
    /// # Example
    /// ```rust
    /// use neuro_divergent::{NeuralForecast, TimeSeriesDataFrame, NeuroDivergentResult};
    ///
    /// fn fit_existing_pipeline(
    ///     pipeline: &mut NeuralForecast<f64>, data: TimeSeriesDataFrame<f64>,
    /// ) -> NeuroDivergentResult<()> {
    ///     pipeline.fit(data)
    /// }
    /// ```
    pub fn fit(&mut self, data: TimeSeriesDataFrame<T>) -> NeuroDivergentResult<()> {
        self.fit_with_validation(data, ValidationConfig::default())
    }

    /// Fit with custom validation configuration
    pub fn fit_with_validation(
        &mut self,
        data: TimeSeriesDataFrame<T>,
        validation_config: ValidationConfig,
    ) -> NeuroDivergentResult<()> {
        if validation_config.validation_split.is_some()
            || validation_config.shuffle
            || validation_config.random_seed.is_some()
        {
            return Err(NeuroDivergentError::config("Use chronological cross_validation for validation splits; facade shuffling and random seeding are unsupported"));
        }
        if !matches!(self.device, Device::CPU) {
            return Err(NeuroDivergentError::config(
                "GPU forecasting is not implemented",
            ));
        }
        if self.prediction_intervals.is_some() {
            return Err(NeuroDivergentError::config(
                "Prediction interval aggregation is not implemented",
            ));
        }
        let timeline = crate::timeline::Timeline::read(&data, &self.frequency)?;
        for model in &self.models {
            if data
                .schema
                .all_columns()
                .iter()
                .any(|column| column.as_str() == model.name())
            {
                return Err(NeuroDivergentError::config(
                    "Model names cannot collide with data columns",
                ));
            }
        }
        let normalized = timeline.normalized(&data, &self.frequency)?;
        let scales = crate::scaling::fit(&timeline, self.local_scaler_type)?;
        let transformed = crate::scaling::transform(&normalized, &scales)?;
        self.is_fitted = false;
        self.training_data = None;
        self.training_schema = None;
        self.scales.clear();
        let threads = self.num_threads.unwrap_or(1);
        if threads == 0 || threads > 256 {
            return Err(NeuroDivergentError::config(
                "Thread count must be in 1..=256",
            ));
        }
        let fit = |models: &mut [Box<dyn BaseModel<T>>]| {
            models
                .par_iter_mut()
                .map(|model| model.fit(&transformed))
                .collect::<Vec<_>>()
        };
        let results = if threads > 1 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| NeuroDivergentError::config(e.to_string()))?
                .install(|| fit(&mut self.models))
        } else {
            self.models
                .iter_mut()
                .map(|model| model.fit(&transformed))
                .collect()
        };
        let mut errors = Vec::new();
        for (model, result) in self.models.iter().zip(results) {
            if let Some(info) = self.model_metadata.get_mut(model.name()) {
                info.fitted = model.is_fitted();
            }
            if let Err(error) = result {
                errors.push(format!("{}: {error}", model.name()));
            } else if !model.is_fitted() {
                errors.push(format!("{} did not enter a fitted state", model.name()));
            }
        }
        if !errors.is_empty() {
            return Err(NeuroDivergentError::training(errors.join("; ")));
        }
        self.training_schema = Some(normalized.schema.clone());
        self.training_data = Some(normalized);
        self.scales = scales;
        self.is_fitted = true;
        Ok(())
    }

    /// Forecast after the last observation of each retained training series.
    ///
    /// # Example
    /// ```rust
    /// use neuro_divergent::{ForecastDataFrame, NeuralForecast, NeuroDivergentResult};
    ///
    /// fn request_forecast(pipeline: &NeuralForecast<f64>) -> NeuroDivergentResult<ForecastDataFrame<f64>> {
    ///     pipeline.predict()
    /// }
    /// ```
    pub fn predict(&self) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        self.predict_with_config(PredictionConfig::default())
    }

    /// Generate forecasts with custom configuration
    pub fn predict_with_config(
        &self,
        config: PredictionConfig,
    ) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        if config.include_intervals || config.num_samples.is_some() || config.temperature.is_some()
        {
            return Err(NeuroDivergentError::config(
                "Probabilistic sampling and interval aggregation are not implemented",
            ));
        }
        let history = self
            .training_data
            .as_ref()
            .ok_or_else(|| NeuroDivergentError::prediction("Models have not been fitted yet"))?;
        self.predict_on(history.clone())
    }

    /// Predict on new data
    ///
    /// Equivalent to Python: `nf.predict(df=new_data)`
    pub fn predict_on(
        &self,
        data: TimeSeriesDataFrame<T>,
    ) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        if !self.is_fitted() {
            return Err(NeuroDivergentError::prediction(
                "Models have not been fitted yet",
            ));
        }
        if self.training_schema.as_ref() != Some(&data.schema) {
            return Err(NeuroDivergentError::data(
                "Data schema does not match training data schema",
            ));
        }
        let timeline = crate::timeline::Timeline::read(&data, &self.frequency)?;
        let training = self
            .training_data
            .as_ref()
            .ok_or_else(|| NeuroDivergentError::prediction("Missing training context"))?;
        let training = crate::timeline::Timeline::read(training, &self.frequency)?;
        for (id, series) in &timeline.series {
            let previous = training.series.get(id).ok_or_else(|| {
                NeuroDivergentError::data("Prediction contains an unknown series")
            })?;
            if series.times.last() < previous.times.last() {
                return Err(NeuroDivergentError::data(
                    "Prediction context ends before this model's training cutoff",
                ));
            }
        }
        let normalized = timeline.normalized(&data, &self.frequency)?;
        let transformed = crate::scaling::transform(&normalized, &self.scales)?;
        let mut forecast = crate::forecast::assemble(&self.models, &transformed, &self.frequency)?;
        crate::scaling::inverse(&mut forecast, &self.scales)?;
        Ok(forecast)
    }

    /// Cross-validation for model evaluation
    ///
    /// Equivalent to Python: `nf.cross_validation(df, n_windows=3)`
    pub fn cross_validation(
        &mut self,
        data: TimeSeriesDataFrame<T>,
        config: CrossValidationConfig,
    ) -> NeuroDivergentResult<CrossValidationDataFrame<T>> {
        crate::backtest::cross_validate(self, data, config)
    }

    /// Fit and predict in one step
    pub fn fit_predict(
        &mut self,
        train_data: TimeSeriesDataFrame<T>,
    ) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        self.fit(train_data)?;
        self.predict()
    }

    /// Get model by name
    pub fn get_model(&self, name: &str) -> Option<&dyn BaseModel<T>> {
        self.models
            .iter()
            .find(|model| model.name() == name)
            .map(|model| model.as_ref())
    }

    /// Get mutable model by name
    pub fn get_model_mut(&mut self, name: &str) -> Option<&mut dyn BaseModel<T>> {
        let index = self.models.iter().position(|model| model.name() == name)?;
        self.is_fitted = false;
        self.training_data = None;
        self.training_schema = None;
        self.scales.clear();
        Some(self.models[index].as_mut())
    }

    /// List all model names
    pub fn model_names(&self) -> Vec<String> {
        self.models.iter().map(|m| m.name().to_string()).collect()
    }

    /// Check if models are fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted && self.models.iter().all(|model| model.is_fitted())
    }

    /// Common forecast horizon enforced when the pipeline is constructed.
    pub fn horizon(&self) -> usize {
        self.models[0].horizon()
    }

    /// Get frequency
    pub fn frequency(&self) -> Frequency {
        self.frequency.clone()
    }

    /// Get number of models
    pub fn num_models(&self) -> usize {
        self.models.len()
    }

    /// Reset all models to unfitted state
    pub fn reset(&mut self) -> NeuroDivergentResult<()> {
        self.is_fitted = false;
        self.training_schema = None;
        self.training_data = None;
        self.scales.clear();
        let results = self
            .models
            .iter_mut()
            .map(|model| model.reset())
            .collect::<Vec<_>>();
        for info in self.model_metadata.values_mut() {
            info.fitted = false;
            info.training_history = None;
        }
        for result in results {
            result?;
        }
        Ok(())
    }

    /// Atomically save supported fitted models, training history, and fitted scaler state.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> NeuroDivergentResult<()> {
        crate::persistence::save(self, path.as_ref())
    }

    /// Restore a versioned checkpoint of supported baseline/RNN models.
    pub fn load<P: AsRef<Path>>(path: P) -> NeuroDivergentResult<Self>
    where
        T: std::fmt::Debug + std::iter::Sum + 'static,
    {
        crate::persistence::load(path.as_ref())
    }
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
