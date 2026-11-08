//! Accuracy calculations from held-out observations.
use crate::{
    AccuracyMetrics, CrossValidationDataFrame, NeuroDivergentError, NeuroDivergentResult,
    TimeSeriesSchema,
};
use chrono::{DateTime, Utc};
use num_traits::Float;
use polars::prelude::*;
use std::{collections::HashMap, marker::PhantomData, path::Path};

impl<T: Float> CrossValidationDataFrame<T> {
    /// Create new cross-validation dataframe
    pub fn new(
        data: DataFrame,
        cutoffs: Vec<DateTime<Utc>>,
        models: Vec<String>,
        schema: TimeSeriesSchema,
    ) -> Self {
        Self {
            data,
            cutoffs,
            models,
            metrics: None,
            schema,
            phantom: PhantomData,
        }
    }

    /// Set accuracy metrics
    pub fn with_metrics(mut self, metrics: AccuracyMetrics<T>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Get results for specific cutoff
    pub fn get_cutoff_results(&self, cutoff: DateTime<Utc>) -> NeuroDivergentResult<DataFrame> {
        let cutoff_column = self
            .data
            .column("cutoff")
            .map_err(|e| NeuroDivergentError::data(e.to_string()))?;
        let mask = crate::timeline::timestamps(cutoff_column)?
            .into_iter()
            .map(|time| time == cutoff)
            .collect::<Vec<_>>();
        let filtered = self
            .data
            .filter(&BooleanChunked::from_slice("cutoff".into(), &mask))
            .map_err(|e| NeuroDivergentError::data(e.to_string()))?;

        Ok(filtered)
    }

    /// Get results for specific model
    pub fn get_model_results(&self, model_name: &str) -> NeuroDivergentResult<DataFrame> {
        if !self.models.iter().any(|name| name == model_name) {
            return Err(NeuroDivergentError::data("Unknown cross-validation model"));
        }
        self.data
            .select([
                self.schema.unique_id_col.as_str(),
                self.schema.ds_col.as_str(),
                "cutoff",
                self.schema.y_col.as_str(),
                model_name,
            ])
            .map_err(|e| NeuroDivergentError::data(e.to_string()))
    }

    /// Calculate pooled accuracy separately for each model from actual held-out targets.
    pub fn metrics_by_model(&self) -> NeuroDivergentResult<HashMap<String, AccuracyMetrics<T>>> {
        let target = self
            .data
            .column(&self.schema.y_col)
            .and_then(|c| c.cast(&DataType::Float64))
            .map_err(|e| NeuroDivergentError::data(e.to_string()))?;
        let target = target
            .f64()
            .map_err(|e| NeuroDivergentError::data(e.to_string()))?;
        let mut metrics = HashMap::new();
        for name in &self.models {
            let prediction = self
                .data
                .column(name)
                .and_then(|c| c.cast(&DataType::Float64))
                .map_err(|e| NeuroDivergentError::data(e.to_string()))?;
            let pairs = target
                .into_iter()
                .zip(
                    prediction
                        .f64()
                        .map_err(|e| NeuroDivergentError::data(e.to_string()))?,
                )
                .map(|(actual, predicted)| {
                    actual
                        .zip(predicted)
                        .ok_or_else(|| NeuroDivergentError::data("Null validation observation"))
                })
                .collect::<NeuroDivergentResult<Vec<_>>>()?;
            metrics.insert(name.clone(), crate::backtest::score(&pairs)?);
        }
        Ok(metrics)
    }

    /// Calculate summary statistics across CV windows
    pub fn summary_stats(&self) -> NeuroDivergentResult<DataFrame> {
        let mut names = Vec::new();
        let mut maes = Vec::new();
        let mut mses = Vec::new();
        let mut rmses = Vec::new();
        let mut mapes = Vec::new();
        let mut smapes = Vec::new();
        let metrics = self.metrics_by_model()?;
        for name in &self.models {
            let metric = &metrics[name];
            names.push(name.clone());
            maes.push(metric.mae.to_f64());
            mses.push(metric.mse.to_f64());
            rmses.push(metric.rmse.to_f64());
            mapes.push(metric.mape.to_f64());
            smapes.push(metric.smape.to_f64());
        }
        df! { "model" => names, "mae" => maes, "mse" => mses, "rmse" => rmses, "mape" => mapes, "smape" => smapes }
            .map_err(|e| NeuroDivergentError::data(e.to_string()))
    }

    /// Export to CSV
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> NeuroDivergentResult<()> {
        let mut file = std::fs::File::create(path)?;
        CsvWriter::new(&mut file)
            .finish(&mut self.data.clone())
            .map_err(|e| NeuroDivergentError::data(format!("CSV write error: {}", e)))?;
        Ok(())
    }
}
