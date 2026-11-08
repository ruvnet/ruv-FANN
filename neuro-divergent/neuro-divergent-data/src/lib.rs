#![doc = include_str!("../README.md")]

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

use num_traits::Float;
use std::fmt;
use std::marker::PhantomData;

#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

pub mod preprocessing;
pub mod features;
pub mod validation;

// Re-export commonly used types
pub use chrono::{DateTime, Utc};
pub use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// A comprehensive prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        TimeSeriesDataset, TimeSeriesData, DataPoint,
        preprocessing::*,
        features::*,
        validation::*,
    };
    pub use chrono::{DateTime, Utc};
    pub use ndarray::{Array1, Array2};
}

/// Errors that can occur during data processing operations
#[derive(thiserror::Error, Debug)]
pub enum DataPipelineError {
    /// Invalid data format or structure
    #[error("Invalid data format: {message}")]
    InvalidFormat {
        /// Description of the failure.
        message: String,
    },
    
    /// Missing required data
    #[error("Missing data: {field}")]
    MissingData {
        /// Field or column involved in the validation result.
        field: String,
    },
    
    /// Data validation failed
    #[error("Validation failed: {reason}")]
    ValidationFailed {
        /// Reason validation failed.
        reason: String,
    },
    
    /// Input/Output error
    #[error("I/O error: {source}")]
    IoError {
        #[from]
        /// Underlying error returned by the operation.
        source: std::io::Error,
    },
    
    /// Serialization error
    #[cfg(feature = "serde_support")]
    #[error("Serialization error: {source}")]
    SerializationError {
        #[from]
        /// Underlying error returned by the operation.
        source: serde_json::Error,
    },
    
    /// Computation error
    #[error("Computation error: {message}")]
    ComputationError {
        /// Description of the failure.
        message: String,
    },
    
    /// Index out of bounds
    #[error("Index out of bounds: {index} >= {length}")]
    IndexOutOfBounds {
        /// Requested index.
        index: usize,
        /// Number of available elements.
        length: usize,
    },
    
    /// Incompatible dimensions
    #[error("Incompatible dimensions: expected {expected}, got {actual}")]
    IncompatibleDimensions {
        /// Expected shape or type.
        expected: String,
        /// Observed shape or type.
        actual: String,
    },
    
    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError {
        /// Description of the failure.
        message: String,
    },
}

/// Specialized Result type for data pipeline operations
pub type Result<T> = std::result::Result<T, DataPipelineError>;

/// Represents a single data point in a time series
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct DataPoint<T: Float> {
    /// Timestamp of the data point
    pub timestamp: DateTime<Utc>,
    /// Value at this timestamp
    pub value: T,
    /// Optional exogenous variables
    pub exogenous: Option<Vec<T>>,
}

impl<T: Float> DataPoint<T> {
    /// Create a new data point
    pub fn new(timestamp: DateTime<Utc>, value: T) -> Self {
        Self {
            timestamp,
            value,
            exogenous: None,
        }
    }
    
    /// Create a data point with exogenous variables
    pub fn with_exogenous(timestamp: DateTime<Utc>, value: T, exogenous: Vec<T>) -> Self {
        Self {
            timestamp,
            value,
            exogenous: Some(exogenous),
        }
    }
}

/// Core time series data structure
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct TimeSeriesData<T: Float> {
    /// Unique identifier for this time series
    pub series_id: String,
    /// Vector of data points
    pub data_points: Vec<DataPoint<T>>,
    /// Static features that don't change over time
    pub static_features: Option<Vec<T>>,
    /// Frequency of the time series (e.g., "D" for daily, "H" for hourly)
    pub frequency: String,
    /// Optional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl<T: Float> TimeSeriesData<T> {
    /// Create a new time series
    pub fn new(series_id: String, frequency: String) -> Self {
        Self {
            series_id,
            data_points: Vec::new(),
            static_features: None,
            frequency,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Add a data point to the time series
    pub fn add_point(&mut self, point: DataPoint<T>) {
        self.data_points.push(point);
    }
    
    /// Get values as a vector
    pub fn values(&self) -> Vec<T> {
        self.data_points.iter().map(|p| p.value).collect()
    }
    
    /// Get timestamps as a vector
    pub fn timestamps(&self) -> Vec<DateTime<Utc>> {
        self.data_points.iter().map(|p| p.timestamp).collect()
    }
    
    /// Get the length of the time series
    pub fn len(&self) -> usize {
        self.data_points.len()
    }
    
    /// Check if the time series is empty
    pub fn is_empty(&self) -> bool {
        self.data_points.is_empty()
    }
    
    /// Sort data points by timestamp
    pub fn sort_by_time(&mut self) {
        self.data_points.sort_by_key(|p| p.timestamp);
    }
    
    /// Get a slice of the time series
    pub fn slice(&self, start: usize, end: usize) -> Result<Self> {
        if start >= self.len() || end > self.len() || start >= end {
            return Err(DataPipelineError::IndexOutOfBounds {
                index: if start >= self.len() { start } else { end },
                length: self.len(),
            });
        }
        
        let mut sliced = Self::new(self.series_id.clone(), self.frequency.clone());
        sliced.data_points = self.data_points[start..end].to_vec();
        sliced.static_features = self.static_features.clone();
        sliced.metadata = self.metadata.clone();
        
        Ok(sliced)
    }
    
    /// Get the first n data points
    pub fn head(&self, n: usize) -> Result<Self> {
        let end = std::cmp::min(n, self.len());
        self.slice(0, end)
    }
    
    /// Get the last n data points
    pub fn tail(&self, n: usize) -> Result<Self> {
        let start = if self.len() > n { self.len() - n } else { 0 };
        self.slice(start, self.len())
    }
}

impl<T: Float + fmt::Display> fmt::Display for TimeSeriesData<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TimeSeriesData(id={}, length={}, frequency={})", 
               self.series_id, self.len(), self.frequency)
    }
}

/// Builder for creating time series datasets
#[derive(Debug)]
pub struct TimeSeriesDatasetBuilder<T: Float> {
    series_id: String,
    frequency: Option<String>,
    values: Option<Vec<T>>,
    timestamps: Option<Vec<DateTime<Utc>>>,
    static_features: Option<Vec<T>>,
    exogenous: Option<Vec<Vec<T>>>,
    metadata: std::collections::HashMap<String, String>,
}

impl<T: Float> TimeSeriesDatasetBuilder<T> {
    /// Create a new builder
    pub fn new(series_id: String) -> Self {
        Self {
            series_id,
            frequency: None,
            values: None,
            timestamps: None,
            static_features: None,
            exogenous: None,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Set the frequency
    pub fn with_frequency(mut self, frequency: String) -> Self {
        self.frequency = Some(frequency);
        self
    }
    
    /// Set the values
    pub fn with_values(mut self, values: Vec<T>) -> Self {
        self.values = Some(values);
        self
    }
    
    /// Set the timestamps
    pub fn with_timestamps(mut self, timestamps: Vec<DateTime<Utc>>) -> Self {
        self.timestamps = Some(timestamps);
        self
    }
    
    /// Set static features
    pub fn with_static_features(mut self, features: Vec<T>) -> Self {
        self.static_features = Some(features);
        self
    }
    
    /// Set exogenous variables
    pub fn with_exogenous(mut self, exogenous: Vec<Vec<T>>) -> Self {
        self.exogenous = Some(exogenous);
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Build the time series dataset
    pub fn build(self) -> Result<TimeSeriesData<T>> {
        let frequency = self.frequency.ok_or_else(|| DataPipelineError::MissingData {
            field: "frequency".to_string(),
        })?;
        
        let values = self.values.ok_or_else(|| DataPipelineError::MissingData {
            field: "values".to_string(),
        })?;
        
        let timestamps = self.timestamps.ok_or_else(|| DataPipelineError::MissingData {
            field: "timestamps".to_string(),
        })?;
        
        if values.len() != timestamps.len() {
            return Err(DataPipelineError::IncompatibleDimensions {
                expected: format!("values.len()={}", values.len()),
                actual: format!("timestamps.len()={}", timestamps.len()),
            });
        }
        
        let mut time_series = TimeSeriesData::new(self.series_id, frequency);
        time_series.static_features = self.static_features;
        time_series.metadata = self.metadata;
        
        // Create data points
        for (timestamp, value) in timestamps.into_iter().zip(values.into_iter()) {
            let exogenous = if let Some(ref exog) = self.exogenous {
                exog.get(time_series.data_points.len()).cloned()
            } else {
                None
            };
            
            let point = if let Some(exog) = exogenous {
                DataPoint::with_exogenous(timestamp, value, exog)
            } else {
                DataPoint::new(timestamp, value)
            };
            
            time_series.add_point(point);
        }
        
        // Sort by timestamp to ensure temporal order
        time_series.sort_by_time();
        
        Ok(time_series)
    }
}

/// Collection of time series datasets for batch processing
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct TimeSeriesDataset<T: Float> {
    /// Collection of time series
    pub series: Vec<TimeSeriesData<T>>,
    /// Dataset metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl<T: Float> TimeSeriesDataset<T> {
    /// Create a new empty dataset
    pub fn new() -> Self {
        Self {
            series: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Create a dataset from a single time series
    pub fn from_series(series: TimeSeriesData<T>) -> Self {
        Self {
            series: vec![series],
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Add a time series to the dataset
    pub fn add_series(&mut self, series: TimeSeriesData<T>) {
        self.series.push(series);
    }
    
    /// Get the number of time series in the dataset
    pub fn len(&self) -> usize {
        self.series.len()
    }
    
    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }
    
    /// Get a time series by index
    pub fn get(&self, index: usize) -> Option<&TimeSeriesData<T>> {
        self.series.get(index)
    }
    
    /// Get a mutable reference to a time series by index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut TimeSeriesData<T>> {
        self.series.get_mut(index)
    }
    
    /// Get a time series by series ID
    pub fn get_by_id(&self, series_id: &str) -> Option<&TimeSeriesData<T>> {
        self.series.iter().find(|s| s.series_id == series_id)
    }
    
    /// Get total number of data points across all series
    pub fn total_points(&self) -> usize {
        self.series.iter().map(|s| s.len()).sum()
    }
    
    /// Add metadata to the dataset
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
    
    /// Apply a function to all series in the dataset
    pub fn map_series<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(&mut TimeSeriesData<T>) -> Result<()>,
    {
        for series in &mut self.series {
            f(series)?;
        }
        Ok(())
    }
    
    /// Filter series based on a predicate
    pub fn filter_series<F>(&mut self, predicate: F)
    where
        F: Fn(&TimeSeriesData<T>) -> bool,
    {
        self.series.retain(predicate);
    }
    
    /// Create an iterator over the series
    pub fn iter(&self) -> std::slice::Iter<TimeSeriesData<T>> {
        self.series.iter()
    }
    
    /// Create a mutable iterator over the series
    pub fn iter_mut(&mut self) -> std::slice::IterMut<TimeSeriesData<T>> {
        self.series.iter_mut()
    }
}

impl<T: Float> Default for TimeSeriesDataset<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> IntoIterator for TimeSeriesDataset<T> {
    type Item = TimeSeriesData<T>;
    type IntoIter = std::vec::IntoIter<TimeSeriesData<T>>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.series.into_iter()
    }
}

impl<T: Float> FromIterator<TimeSeriesData<T>> for TimeSeriesDataset<T> {
    fn from_iter<I: IntoIterator<Item = TimeSeriesData<T>>>(iter: I) -> Self {
        Self {
            series: iter.into_iter().collect(),
            metadata: std::collections::HashMap::new(),
        }
    }
}

impl<T: Float + fmt::Display> fmt::Display for TimeSeriesDataset<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TimeSeriesDataset(series_count={}, total_points={})", 
               self.len(), self.total_points())
    }
}

/// Trait for data transformations that can be applied to time series
pub trait DataTransform<T: Float> {
    /// Apply the transformation to a time series
    fn transform(&self, data: &TimeSeriesData<T>) -> Result<TimeSeriesData<T>>;
    
    /// Apply the transformation in-place
    fn transform_inplace(&self, data: &mut TimeSeriesData<T>) -> Result<()> {
        let transformed = self.transform(data)?;
        *data = transformed;
        Ok(())
    }
}

/// Trait for transformations that need to be fitted before application
pub trait FittableTransform<T: Float>: DataTransform<T> {
    /// Fit the transformation to data
    fn fit(&mut self, data: &TimeSeriesData<T>) -> Result<()>;
    
    /// Fit and transform in one step
    fn fit_transform(&mut self, data: &TimeSeriesData<T>) -> Result<TimeSeriesData<T>> {
        self.fit(data)?;
        self.transform(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    
    #[test]
    fn test_data_point_creation() {
        let timestamp = Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
        let point = DataPoint::new(timestamp, 42.0f64);
        
        assert_eq!(point.timestamp, timestamp);
        assert_eq!(point.value, 42.0);
        assert!(point.exogenous.is_none());
    }
    
    #[test]
    fn test_data_point_with_exogenous() {
        let timestamp = Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
        let exogenous = vec![1.0, 2.0, 3.0];
        let point = DataPoint::with_exogenous(timestamp, 42.0f64, exogenous.clone());
        
        assert_eq!(point.timestamp, timestamp);
        assert_eq!(point.value, 42.0);
        assert_eq!(point.exogenous, Some(exogenous));
    }
    
    #[test]
    fn test_time_series_builder() {
        let timestamps = vec![
            Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            Utc.ymd_opt(2023, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            Utc.ymd_opt(2023, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0];
        
        let series = TimeSeriesDatasetBuilder::new("test_series".to_string())
            .with_frequency("D".to_string())
            .with_values(values.clone())
            .with_timestamps(timestamps.clone())
            .build()
            .unwrap();
        
        assert_eq!(series.series_id, "test_series");
        assert_eq!(series.frequency, "D");
        assert_eq!(series.len(), 3);
        assert_eq!(series.values(), values);
        assert_eq!(series.timestamps(), timestamps);
    }
    
    #[test]
    fn test_time_series_slice() {
        let timestamps = vec![
            Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            Utc.ymd_opt(2023, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            Utc.ymd_opt(2023, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0];
        
        let series = TimeSeriesDatasetBuilder::new("test_series".to_string())
            .with_frequency("D".to_string())
            .with_values(values)
            .with_timestamps(timestamps)
            .build()
            .unwrap();
        
        let sliced = series.slice(1, 3).unwrap();
        assert_eq!(sliced.len(), 2);
        assert_eq!(sliced.values(), vec![2.0, 3.0]);
    }
    
    #[test]
    fn test_dataset_operations() {
        let series1 = TimeSeriesDatasetBuilder::new("series1".to_string())
            .with_frequency("D".to_string())
            .with_values(vec![1.0, 2.0])
            .with_timestamps(vec![
                Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
                Utc.ymd_opt(2023, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            ])
            .build()
            .unwrap();
        
        let series2 = TimeSeriesDatasetBuilder::new("series2".to_string())
            .with_frequency("D".to_string())
            .with_values(vec![3.0, 4.0])
            .with_timestamps(vec![
                Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
                Utc.ymd_opt(2023, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            ])
            .build()
            .unwrap();
        
        let mut dataset = TimeSeriesDataset::new();
        dataset.add_series(series1);
        dataset.add_series(series2);
        
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.total_points(), 4);
        assert!(dataset.get_by_id("series1").is_some());
        assert!(dataset.get_by_id("series2").is_some());
        assert!(dataset.get_by_id("nonexistent").is_none());
    }
}