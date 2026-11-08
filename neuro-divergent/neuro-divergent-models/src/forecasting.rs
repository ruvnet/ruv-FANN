//! Main forecasting interface for neuro-divergent models
//!
//! This module provides the main NeuralForecast class that manages
//! multiple models and provides a unified interface for training and prediction.

use crate::config::{CrossValidationConfig, PredictionConfig, TrainingConfig};
use crate::data::{ForecastDataFrame, TimeSeriesDataFrame};
use crate::errors::{NeuroDivergentError, NeuroDivergentResult};
use crate::foundation::{BaseModel, TrainingMetrics, ValidationMetrics};
use num_traits::Float;
use std::collections::HashMap;

/// Main entry point for neural forecasting
pub struct NeuralForecast<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    models: Vec<Box<dyn BaseModel<T>>>,
    frequency: Option<String>,
    local_scaler_type: Option<String>,
    num_threads: Option<usize>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> NeuralForecast<T> {
    /// Create new NeuralForecast instance builder
    pub fn new() -> NeuralForecastBuilder<T> {
        NeuralForecastBuilder::new()
    }

    /// Fit models to the provided time series data
    pub fn fit(&mut self, data: TimeSeriesDataFrame<T>) -> NeuroDivergentResult<()> {
        self.fit_with_config(data, TrainingConfig::default())
    }

    /// Fit with custom training configuration
    pub fn fit_with_config(
        &mut self,
        data: TimeSeriesDataFrame<T>,
        config: TrainingConfig<T>,
    ) -> NeuroDivergentResult<()> {
        let dataset = data.to_dataset()?;

        for model in &mut self.models {
            let metrics = model.fit(&dataset)?;
            log::info!(
                "Model {} trained with final loss: {:?}",
                model.name(),
                metrics.final_loss
            );
        }

        Ok(())
    }

    /// Generate forecasts for all fitted models
    pub fn predict(&self) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        self.predict_with_config(PredictionConfig::default())
    }

    /// Generate forecasts with custom configuration
    pub fn predict_with_config(
        &self,
        _config: PredictionConfig,
    ) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        Err(NeuroDivergentError::prediction("Legacy models NeuralForecast prediction is unsupported; use the primary facade NeuralForecast with ModelAdapter and explicit history"))
    }

    /// Cross-validation for model evaluation
    pub fn cross_validation(
        &mut self,
        data: TimeSeriesDataFrame<T>,
        config: CrossValidationConfig,
    ) -> NeuroDivergentResult<HashMap<String, ValidationMetrics<T>>> {
        let mut results = HashMap::new();
        let dataset = data.to_dataset()?;

        for model in &mut self.models {
            let metrics = model.validate(&dataset)?;
            results.insert(model.name().to_string(), metrics);
        }

        Ok(results)
    }

    /// Get model by name
    pub fn get_model(&self, name: &str) -> Option<&dyn BaseModel<T>> {
        self.models
            .iter()
            .find(|m| m.name() == name)
            .map(|m| m.as_ref())
    }

    /// List all model names
    pub fn model_names(&self) -> Vec<String> {
        self.models.iter().map(|m| m.name().to_string()).collect()
    }
}

#[cfg(test)]
mod legacy_boundary_tests {
    use super::*;

    #[test]
    fn legacy_forecasting_does_not_return_an_empty_hardcoded_forecast() {
        let model = crate::recurrent::RNN::<f64>::new_seeded(
            crate::config::RNNConfig::default_with_horizon(3),
            1,
        )
        .unwrap();
        let forecasting = NeuralForecast::new()
            .with_model(Box::new(model))
            .build()
            .unwrap();
        assert!(forecasting
            .predict()
            .unwrap_err()
            .to_string()
            .contains("unsupported"));
    }
}

/// Builder for constructing NeuralForecast instances
pub struct NeuralForecastBuilder<
    T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static,
> {
    models: Vec<Box<dyn BaseModel<T>>>,
    frequency: Option<String>,
    local_scaler_type: Option<String>,
    num_threads: Option<usize>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> NeuralForecastBuilder<T> {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            frequency: None,
            local_scaler_type: None,
            num_threads: None,
        }
    }

    /// Add models to the forecasting ensemble
    pub fn with_models(mut self, models: Vec<Box<dyn BaseModel<T>>>) -> Self {
        self.models = models;
        self
    }

    /// Add a single model
    pub fn with_model(mut self, model: Box<dyn BaseModel<T>>) -> Self {
        self.models.push(model);
        self
    }

    /// Set the frequency of the time series
    pub fn with_frequency<S: Into<String>>(mut self, frequency: S) -> Self {
        self.frequency = Some(frequency.into());
        self
    }

    /// Set local scaler type for data preprocessing
    pub fn with_local_scaler<S: Into<String>>(mut self, scaler_type: S) -> Self {
        self.local_scaler_type = Some(scaler_type.into());
        self
    }

    /// Set number of threads for parallel processing
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Build the NeuralForecast instance
    pub fn build(self) -> NeuroDivergentResult<NeuralForecast<T>> {
        if self.models.is_empty() {
            return Err(NeuroDivergentError::config(
                "At least one model is required",
            ));
        }

        Ok(NeuralForecast {
            models: self.models,
            frequency: self.frequency,
            local_scaler_type: self.local_scaler_type,
            num_threads: self.num_threads,
        })
    }
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> Default
    for NeuralForecastBuilder<T>
{
    fn default() -> Self {
        Self::new()
    }
}
