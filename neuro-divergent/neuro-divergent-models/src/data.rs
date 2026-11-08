//! Data structures for time series handling
//!
//! This module provides data structures for handling time series data,
//! including DataFrame-like structures for time series operations.

use crate::errors::{NeuroDivergentError, NeuroDivergentResult};
use crate::foundation::{TimeSeriesDataset, TimeSeriesInput, TimeSeriesSample};
use chrono::{DateTime, Utc};
use num_traits::Float;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Schema definition for time series data
#[derive(Debug, Clone)]
pub struct TimeSeriesSchema {
    pub unique_id_col: String,
    pub ds_col: String, // Date/time column
    pub y_col: String,  // Target variable column
    pub static_features: Vec<String>,
    pub exogenous_features: Vec<String>,
}

impl TimeSeriesSchema {
    /// Create schema with required columns
    pub fn new(
        unique_id_col: impl Into<String>,
        ds_col: impl Into<String>,
        y_col: impl Into<String>,
    ) -> Self {
        Self {
            unique_id_col: unique_id_col.into(),
            ds_col: ds_col.into(),
            y_col: y_col.into(),
            static_features: Vec::new(),
            exogenous_features: Vec::new(),
        }
    }

    /// Add static features
    pub fn with_static_features(mut self, features: Vec<String>) -> Self {
        self.static_features = features;
        self
    }

    /// Add exogenous features
    pub fn with_exogenous_features(mut self, features: Vec<String>) -> Self {
        self.exogenous_features = features;
        self
    }
}

/// Main data structure for time series data
#[derive(Debug, Clone)]
pub struct TimeSeriesDataFrame<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>
{
    pub data: Vec<HashMap<String, DataValue<T>>>,
    pub schema: TimeSeriesSchema,
    phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> TimeSeriesDataFrame<T> {
    /// Create a new TimeSeriesDataFrame
    pub fn new(schema: TimeSeriesSchema) -> Self {
        Self {
            data: Vec::new(),
            schema,
            phantom: PhantomData,
        }
    }

    /// Legacy conversion without a lookback/horizon contract is unsupported.
    /// Use explicit TimeSeriesDataset windows or the primary facade's ModelAdapter.
    pub fn to_dataset(&self) -> NeuroDivergentResult<TimeSeriesDataset<T>> {
        Err(NeuroDivergentError::data("Legacy models DataFrame conversion is unsupported; provide explicit lookback/horizon windows or use the primary facade ModelAdapter"))
    }
}

#[cfg(test)]
mod legacy_boundary_tests {
    use super::*;

    #[test]
    fn legacy_dataframe_conversion_does_not_return_an_empty_success() {
        let mut frame = TimeSeriesDataFrame::<f64>::new(TimeSeriesSchema::new("id", "ds", "y"));
        assert!(frame
            .to_dataset()
            .unwrap_err()
            .to_string()
            .contains("unsupported"));
        frame
            .data
            .push(HashMap::from([("y".into(), DataValue::Float(1.0))]));
        assert!(frame.to_dataset().is_err());
    }
}

/// Results from forecasting operations
#[derive(Debug, Clone)]
pub struct ForecastDataFrame<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    pub data: Vec<HashMap<String, DataValue<T>>>,
    pub models: Vec<String>,
    pub forecast_horizon: usize,
    pub confidence_levels: Option<Vec<f64>>,
    phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> ForecastDataFrame<T> {
    /// Create a new ForecastDataFrame
    pub fn new(models: Vec<String>, forecast_horizon: usize) -> Self {
        Self {
            data: Vec::new(),
            models,
            forecast_horizon,
            confidence_levels: None,
            phantom: PhantomData,
        }
    }
}

/// Generic data value for DataFrame operations
#[derive(Debug, Clone)]
pub enum DataValue<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    Float(T),
    Int(i64),
    String(String),
    DateTime(DateTime<Utc>),
    Bool(bool),
    Null,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> DataValue<T> {
    pub fn as_float(&self) -> Option<T> {
        match self {
            DataValue::Float(f) => Some(*f),
            DataValue::Int(i) => T::from(*i),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            DataValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_datetime(&self) -> Option<DateTime<Utc>> {
        match self {
            DataValue::DateTime(dt) => Some(*dt),
            _ => None,
        }
    }
}
