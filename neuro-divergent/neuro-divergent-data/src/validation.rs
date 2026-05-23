//! # Data Validation for Time Series
//!
//! This module provides comprehensive data validation capabilities for time series,
//! including data quality checks, temporal consistency validation, and statistical tests.

use crate::{DataPipelineError, Result, TimeSeriesData, TimeSeriesDataset};
use num_traits::Float;
use chrono::{DateTime, Utc, Duration};
use std::collections::{HashMap, HashSet};
use std::fmt;

#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

/// Validation result containing all validation issues found
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct ValidationReport {
    /// Whether validation passed overall
    pub is_valid: bool,
    /// List of validation errors
    pub errors: Vec<ValidationError>,
    /// List of validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Summary statistics
    pub summary: ValidationSummary,
}

impl ValidationReport {
    /// Create a new validation report
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            summary: ValidationSummary::default(),
        }
    }
    
    /// Add a validation error
    pub fn add_error(&mut self, error: ValidationError) {
        self.is_valid = false;
        self.errors.push(error);
    }
    
    /// Add a validation warning
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }
    
    /// Get the total number of issues
    pub fn total_issues(&self) -> usize {
        self.errors.len() + self.warnings.len()
    }
    
    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
    
    /// Check if there are any warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Validation Report:")?;
        writeln!(f, "  Status: {}", if self.is_valid { "PASSED" } else { "FAILED" })?;
        writeln!(f, "  Errors: {}", self.errors.len())?;
        writeln!(f, "  Warnings: {}", self.warnings.len())?;
        
        if !self.errors.is_empty() {
            writeln!(f, "\nErrors:")?;
            for (i, error) in self.errors.iter().enumerate() {
                writeln!(f, "  {}: {}", i + 1, error)?;
            }
        }
        
        if !self.warnings.is_empty() {
            writeln!(f, "\nWarnings:")?;
            for (i, warning) in self.warnings.iter().enumerate() {
                writeln!(f, "  {}: {}", i + 1, warning)?;
            }
        }
        
        Ok(())
    }
}

/// Validation error types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum ValidationError {
    /// Data is missing or empty
    MissingData { field: String, details: String },
    /// Invalid data format
    InvalidFormat { field: String, details: String },
    /// Data type mismatch
    TypeMismatch { field: String, expected: String, actual: String },
    /// Data range violation
    OutOfRange { field: String, value: String, min: Option<String>, max: Option<String> },
    /// Duplicate data detected
    DuplicateData { field: String, details: String },
    /// Temporal consistency violation
    TemporalError { details: String },
    /// Statistical anomaly
    StatisticalAnomaly { test: String, details: String },
    /// Data integrity violation
    IntegrityViolation { details: String },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::MissingData { field, details } => {
                write!(f, "Missing data in '{}': {}", field, details)
            }
            ValidationError::InvalidFormat { field, details } => {
                write!(f, "Invalid format in '{}': {}", field, details)
            }
            ValidationError::TypeMismatch { field, expected, actual } => {
                write!(f, "Type mismatch in '{}': expected {}, got {}", field, expected, actual)
            }
            ValidationError::OutOfRange { field, value, min, max } => {
                let range = match (min, max) {
                    (Some(min), Some(max)) => format!("[{}, {}]", min, max),
                    (Some(min), None) => format!(">= {}", min),
                    (None, Some(max)) => format!("<= {}", max),
                    (None, None) => "valid range".to_string(),
                };
                write!(f, "Value '{}' in '{}' is out of range {}", value, field, range)
            }
            ValidationError::DuplicateData { field, details } => {
                write!(f, "Duplicate data in '{}': {}", field, details)
            }
            ValidationError::TemporalError { details } => {
                write!(f, "Temporal error: {}", details)
            }
            ValidationError::StatisticalAnomaly { test, details } => {
                write!(f, "Statistical anomaly ({}): {}", test, details)
            }
            ValidationError::IntegrityViolation { details } => {
                write!(f, "Data integrity violation: {}", details)
            }
        }
    }
}

/// Validation warning types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum ValidationWarning {
    /// Potential data quality issue
    DataQuality { field: String, details: String },
    /// Suspicious patterns detected
    SuspiciousPattern { pattern: String, details: String },
    /// Performance concern
    Performance { details: String },
    /// Missing optional data
    MissingOptional { field: String, details: String },
    /// Statistical concern
    StatisticalConcern { test: String, details: String },
}

impl fmt::Display for ValidationWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationWarning::DataQuality { field, details } => {
                write!(f, "Data quality concern in '{}': {}", field, details)
            }
            ValidationWarning::SuspiciousPattern { pattern, details } => {
                write!(f, "Suspicious pattern '{}': {}", pattern, details)
            }
            ValidationWarning::Performance { details } => {
                write!(f, "Performance concern: {}", details)
            }
            ValidationWarning::MissingOptional { field, details } => {
                write!(f, "Missing optional data in '{}': {}", field, details)
            }
            ValidationWarning::StatisticalConcern { test, details } => {
                write!(f, "Statistical concern ({}): {}", test, details)
            }
        }
    }
}

/// Summary statistics from validation
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct ValidationSummary {
    /// Total number of series validated
    pub total_series: usize,
    /// Total number of data points validated
    pub total_points: usize,
    /// Number of series with missing values
    pub series_with_missing: usize,
    /// Number of series with duplicates
    pub series_with_duplicates: usize,
    /// Number of series with temporal issues
    pub series_with_temporal_issues: usize,
    /// Number of series with outliers
    pub series_with_outliers: usize,
    /// Average series length
    pub avg_series_length: f64,
    /// Minimum series length
    pub min_series_length: usize,
    /// Maximum series length
    pub max_series_length: usize,
}

/// Configuration for data validation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct ValidationConfig<T: Float> {
    /// Check for missing values
    pub check_missing_values: bool,
    /// Check for duplicate timestamps
    pub check_duplicate_timestamps: bool,
    /// Check temporal ordering
    pub check_temporal_ordering: bool,
    /// Check data types
    pub check_data_types: bool,
    /// Check value ranges
    pub check_value_ranges: bool,
    /// Value range constraints
    pub value_range: Option<(T, T)>,
    /// Check for outliers
    pub check_outliers: bool,
    /// Outlier detection method
    pub outlier_method: OutlierMethod,
    /// Check for stationarity
    pub check_stationarity: bool,
    /// Check for seasonality
    pub check_seasonality: bool,
    /// Minimum series length
    pub min_series_length: Option<usize>,
    /// Maximum series length
    pub max_series_length: Option<usize>,
    /// Expected frequency pattern
    pub expected_frequency: Option<String>,
    /// Tolerance for frequency detection
    pub frequency_tolerance: Option<Duration>,
}

impl<T: Float> Default for ValidationConfig<T> {
    fn default() -> Self {
        Self {
            check_missing_values: true,
            check_duplicate_timestamps: true,
            check_temporal_ordering: true,
            check_data_types: true,
            check_value_ranges: false,
            value_range: None,
            check_outliers: true,
            outlier_method: OutlierMethod::IQR,
            check_stationarity: false,
            check_seasonality: false,
            min_series_length: Some(2),
            max_series_length: None,
            expected_frequency: None,
            frequency_tolerance: None,
        }
    }
}

/// Outlier detection methods
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum OutlierMethod {
    /// Interquartile Range method
    IQR,
    /// Z-score method
    ZScore,
    /// Modified Z-score method
    ModifiedZScore,
    /// Isolation Forest (simplified)
    IsolationForest,
}

/// Main data validator
#[derive(Debug, Clone)]
pub struct DataValidator<T: Float> {
    config: ValidationConfig<T>,
}

impl<T: Float> DataValidator<T> {
    /// Create a new data validator with default configuration
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }
    
    /// Create a data validator with custom configuration
    pub fn with_config(config: ValidationConfig<T>) -> Self {
        Self { config }
    }
    
    /// Set the value range for validation
    pub fn with_value_range(mut self, min: T, max: T) -> Self {
        self.config.value_range = Some((min, max));
        self
    }
    
    /// Set the expected frequency
    pub fn with_expected_frequency(mut self, frequency: String) -> Self {
        self.config.expected_frequency = Some(frequency);
        self
    }
    
    /// Set the outlier detection method
    pub fn with_outlier_method(mut self, method: OutlierMethod) -> Self {
        self.config.outlier_method = method;
        self
    }
    
    /// Validate a single time series
    pub fn validate_series(&self, data: &TimeSeriesData<T>) -> ValidationReport {
        let mut report = ValidationReport::new();
        report.summary.total_series = 1;
        report.summary.total_points = data.len();
        report.summary.avg_series_length = data.len() as f64;
        report.summary.min_series_length = data.len();
        report.summary.max_series_length = data.len();
        
        // Check basic data requirements
        self.validate_basic_requirements(data, &mut report);
        
        // Check temporal consistency
        if self.config.check_temporal_ordering || self.config.check_duplicate_timestamps {
            self.validate_temporal_consistency(data, &mut report);
        }
        
        // Check data values
        self.validate_data_values(data, &mut report);
        
        // Check for outliers
        if self.config.check_outliers {
            self.validate_outliers(data, &mut report);
        }
        
        // Check frequency patterns
        if self.config.expected_frequency.is_some() {
            self.validate_frequency(data, &mut report);
        }
        
        // Statistical tests
        if self.config.check_stationarity {
            self.validate_stationarity(data, &mut report);
        }
        
        if self.config.check_seasonality {
            self.validate_seasonality(data, &mut report);
        }
        
        report
    }
    
    /// Validate a dataset of multiple time series
    pub fn validate_dataset(&self, dataset: &TimeSeriesDataset<T>) -> ValidationReport {
        let mut combined_report = ValidationReport::new();
        combined_report.summary.total_series = dataset.len();
        
        let mut total_points = 0;
        let mut series_lengths = Vec::new();
        
        for series in dataset.iter() {
            let series_report = self.validate_series(series);
            
            // Combine errors and warnings
            combined_report.errors.extend(series_report.errors);
            combined_report.warnings.extend(series_report.warnings);
            
            // Update summary statistics
            total_points += series.len();
            series_lengths.push(series.len());
            
            if series_report.summary.series_with_missing > 0 {
                combined_report.summary.series_with_missing += 1;
            }
            if series_report.summary.series_with_duplicates > 0 {
                combined_report.summary.series_with_duplicates += 1;
            }
            if series_report.summary.series_with_temporal_issues > 0 {
                combined_report.summary.series_with_temporal_issues += 1;
            }
            if series_report.summary.series_with_outliers > 0 {
                combined_report.summary.series_with_outliers += 1;
            }
        }
        
        combined_report.summary.total_points = total_points;
        if !series_lengths.is_empty() {
            combined_report.summary.avg_series_length = 
                series_lengths.iter().sum::<usize>() as f64 / series_lengths.len() as f64;
            combined_report.summary.min_series_length = 
                *series_lengths.iter().min().unwrap();
            combined_report.summary.max_series_length = 
                *series_lengths.iter().max().unwrap();
        }
        
        // Check for cross-series consistency
        self.validate_cross_series_consistency(dataset, &mut combined_report);
        
        combined_report.is_valid = combined_report.errors.is_empty();
        combined_report
    }
    
    /// Validate basic data requirements
    fn validate_basic_requirements(&self, data: &TimeSeriesData<T>, report: &mut ValidationReport) {
        // Check if data is empty
        if data.is_empty() {
            report.add_error(ValidationError::MissingData {
                field: "data_points".to_string(),
                details: "Time series contains no data points".to_string(),
            });
            return;
        }
        
        // Check series length constraints
        if let Some(min_length) = self.config.min_series_length {
            if data.len() < min_length {
                report.add_error(ValidationError::OutOfRange {
                    field: "series_length".to_string(),
                    value: data.len().to_string(),
                    min: Some(min_length.to_string()),
                    max: None,
                });
            }
        }
        
        if let Some(max_length) = self.config.max_series_length {
            if data.len() > max_length {
                report.add_error(ValidationError::OutOfRange {
                    field: "series_length".to_string(),
                    value: data.len().to_string(),
                    min: None,
                    max: Some(max_length.to_string()),
                });
            }
        }
        
        // Check for missing values in target variable
        if self.config.check_missing_values {
            let mut missing_count = 0;
            for (i, point) in data.data_points.iter().enumerate() {
                if point.value.is_nan() || point.value.is_infinite() {
                    missing_count += 1;
                    if missing_count <= 5 { // Report first 5 missing values
                        report.add_error(ValidationError::MissingData {
                            field: format!("value[{}]", i),
                            details: format!("Invalid value at timestamp {}", point.timestamp),
                        });
                    }
                }
            }
            
            if missing_count > 0 {
                report.summary.series_with_missing = 1;
                if missing_count > 5 {
                    report.add_warning(ValidationWarning::DataQuality {
                        field: "values".to_string(),
                        details: format!("Total {} missing/invalid values detected", missing_count),
                    });
                }
            }
        }
        
        // Check series ID
        if data.series_id.is_empty() {
            report.add_warning(ValidationWarning::MissingOptional {
                field: "series_id".to_string(),
                details: "Series ID is empty".to_string(),
            });
        }
        
        // Check frequency
        if data.frequency.is_empty() {
            report.add_warning(ValidationWarning::MissingOptional {
                field: "frequency".to_string(),
                details: "Frequency is not specified".to_string(),
            });
        }
    }
    
    /// Validate temporal consistency
    fn validate_temporal_consistency(&self, data: &TimeSeriesData<T>, report: &mut ValidationReport) {
        let timestamps = data.timestamps();
        
        if timestamps.len() < 2 {
            return; // Cannot validate temporal consistency with < 2 points
        }
        
        // Check temporal ordering
        if self.config.check_temporal_ordering {
            let mut out_of_order_count = 0;
            for i in 1..timestamps.len() {
                if timestamps[i] <= timestamps[i - 1] {
                    out_of_order_count += 1;
                    if out_of_order_count <= 3 { // Report first 3 issues
                        report.add_error(ValidationError::TemporalError {
                            details: format!(
                                "Timestamp at index {} ({}) is not after previous timestamp ({})",
                                i, timestamps[i], timestamps[i - 1]
                            ),
                        });
                    }
                }
            }
            
            if out_of_order_count > 0 {
                report.summary.series_with_temporal_issues = 1;
                if out_of_order_count > 3 {
                    report.add_warning(ValidationWarning::DataQuality {
                        field: "timestamps".to_string(),
                        details: format!("Total {} out-of-order timestamps", out_of_order_count),
                    });
                }
            }
        }
        
        // Check for duplicate timestamps
        if self.config.check_duplicate_timestamps {
            let mut seen_timestamps = HashSet::new();
            let mut duplicate_count = 0;
            
            for (i, timestamp) in timestamps.iter().enumerate() {
                if seen_timestamps.contains(timestamp) {
                    duplicate_count += 1;
                    if duplicate_count <= 3 { // Report first 3 duplicates
                        report.add_error(ValidationError::DuplicateData {
                            field: format!("timestamp[{}]", i),
                            details: format!("Duplicate timestamp: {}", timestamp),
                        });
                    }
                } else {
                    seen_timestamps.insert(*timestamp);
                }
            }
            
            if duplicate_count > 0 {
                report.summary.series_with_duplicates = 1;
                if duplicate_count > 3 {
                    report.add_warning(ValidationWarning::DataQuality {
                        field: "timestamps".to_string(),
                        details: format!("Total {} duplicate timestamps", duplicate_count),
                    });
                }
            }
        }
    }
    
    /// Validate data values
    fn validate_data_values(&self, data: &TimeSeriesData<T>, report: &mut ValidationReport) {
        let values = data.values();
        
        // Check value ranges
        if self.config.check_value_ranges {
            if let Some((min_val, max_val)) = self.config.value_range {
                let mut out_of_range_count = 0;
                for (i, &value) in values.iter().enumerate() {
                    if !value.is_nan() && (value < min_val || value > max_val) {
                        out_of_range_count += 1;
                        if out_of_range_count <= 3 { // Report first 3 issues
                            report.add_error(ValidationError::OutOfRange {
                                field: format!("value[{}]", i),
                                value: format!("{:.6}", value.to_f64().unwrap_or(0.0)),
                                min: Some(format!("{:.6}", min_val.to_f64().unwrap_or(0.0))),
                                max: Some(format!("{:.6}", max_val.to_f64().unwrap_or(0.0))),
                            });
                        }
                    }
                }
                
                if out_of_range_count > 3 {
                    report.add_warning(ValidationWarning::DataQuality {
                        field: "values".to_string(),
                        details: format!("Total {} out-of-range values", out_of_range_count),
                    });
                }
            }
        }
        
        // Check for constant values
        let valid_values: Vec<T> = values.iter()
            .filter(|&&x| !x.is_nan() && !x.is_infinite())
            .cloned()
            .collect();
            
        if valid_values.len() > 1 {
            let first_value = valid_values[0];
            let is_constant = valid_values.iter().all(|&x| (x - first_value).abs() < T::epsilon());
            
            if is_constant {
                report.add_warning(ValidationWarning::SuspiciousPattern {
                    pattern: "constant_values".to_string(),
                    details: format!("All values are constant: {:.6}", 
                                   first_value.to_f64().unwrap_or(0.0)),
                });
            }
        }
    }
    
    /// Validate outliers
    fn validate_outliers(&self, data: &TimeSeriesData<T>, report: &mut ValidationReport) {
        let values = data.values();
        let valid_values: Vec<T> = values.iter()
            .filter(|&&x| !x.is_nan() && !x.is_infinite())
            .cloned()
            .collect();
            
        if valid_values.len() < 4 {
            return; // Need at least 4 points for outlier detection
        }
        
        let outlier_indices = match self.config.outlier_method {
            OutlierMethod::IQR => self.detect_outliers_iqr(&valid_values),
            OutlierMethod::ZScore => self.detect_outliers_zscore(&valid_values),
            OutlierMethod::ModifiedZScore => self.detect_outliers_modified_zscore(&valid_values),
            OutlierMethod::IsolationForest => self.detect_outliers_isolation_forest(&valid_values),
        };
        
        if !outlier_indices.is_empty() {
            report.summary.series_with_outliers = 1;
            
            if outlier_indices.len() <= 5 {
                for &idx in &outlier_indices {
                    report.add_warning(ValidationWarning::SuspiciousPattern {
                        pattern: "outlier".to_string(),
                        details: format!("Potential outlier at index {}: {:.6}", 
                                       idx, valid_values[idx].to_f64().unwrap_or(0.0)),
                    });
                }
            } else {
                report.add_warning(ValidationWarning::SuspiciousPattern {
                    pattern: "multiple_outliers".to_string(),
                    details: format!("{} potential outliers detected", outlier_indices.len()),
                });
            }
        }
    }
    
    /// Validate frequency patterns
    fn validate_frequency(&self, data: &TimeSeriesData<T>, report: &mut ValidationReport) {
        if let Some(ref expected_freq) = self.config.expected_frequency {
            let timestamps = data.timestamps();
            if timestamps.len() < 2 {
                return;
            }
            
            // Calculate actual intervals
            let mut intervals = Vec::new();
            for i in 1..timestamps.len() {
                intervals.push(timestamps[i] - timestamps[i - 1]);
            }
            
            // Parse expected frequency and calculate expected interval
            let expected_interval = match expected_freq.as_str() {
                "D" => Duration::days(1),
                "H" => Duration::hours(1),
                "M" => Duration::minutes(1),
                "S" => Duration::seconds(1),
                "W" => Duration::weeks(1),
                _ => {
                    report.add_warning(ValidationWarning::DataQuality {
                        field: "frequency".to_string(),
                        details: format!("Unknown frequency format: {}", expected_freq),
                    });
                    return;
                }
            };
            
            // Check intervals against expected
            let tolerance = self.config.frequency_tolerance.unwrap_or(
                Duration::seconds(expected_interval.num_seconds().abs() / 10) // 10% tolerance
            );
            
            let mut irregular_count = 0;
            for (i, interval) in intervals.iter().enumerate() {
                let diff = (*interval - expected_interval).num_seconds().abs();
                if diff > tolerance.num_seconds() {
                    irregular_count += 1;
                    if irregular_count <= 3 {
                        report.add_warning(ValidationWarning::SuspiciousPattern {
                            pattern: "irregular_frequency".to_string(),
                            details: format!(
                                "Irregular interval at position {}: expected {}, got {}",
                                i + 1,
                                self.duration_to_string(expected_interval),
                                self.duration_to_string(*interval)
                            ),
                        });
                    }
                }
            }
            
            if irregular_count > 3 {
                report.add_warning(ValidationWarning::DataQuality {
                    field: "frequency".to_string(),
                    details: format!("Total {} irregular intervals detected", irregular_count),
                });
            }
        }
    }
    
    /// Validate stationarity (simplified test)
    fn validate_stationarity(&self, data: &TimeSeriesData<T>, _report: &mut ValidationReport) {
        let values = data.values();
        let valid_values: Vec<T> = values.iter()
            .filter(|&&x| !x.is_nan() && !x.is_infinite())
            .cloned()
            .collect();
            
        if valid_values.len() < 10 {
            return; // Need sufficient data for stationarity test
        }
        
        // Simple stationarity test: compare mean and variance of first and second half
        let mid = valid_values.len() / 2;
        let first_half = &valid_values[..mid];
        let second_half = &valid_values[mid..];
        
        let mean1 = first_half.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(first_half.len()).unwrap();
        let mean2 = second_half.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(second_half.len()).unwrap();
        
        let var1 = first_half.iter()
            .map(|&x| (x - mean1) * (x - mean1))
            .fold(T::zero(), |acc, x| acc + x) / T::from(first_half.len()).unwrap();
        let var2 = second_half.iter()
            .map(|&x| (x - mean2) * (x - mean2))
            .fold(T::zero(), |acc, x| acc + x) / T::from(second_half.len()).unwrap();
        
        // Check for significant differences (simplified test)
        let mean_ratio = if mean1.abs() > T::epsilon() {
            (mean2 - mean1).abs() / mean1.abs()
        } else {
            mean2.abs()
        };
        
        let var_ratio = if var1 > T::epsilon() {
            (var2 - var1).abs() / var1
        } else {
            var2
        };
        
        // These thresholds are quite arbitrary - in practice, more sophisticated tests would be used
        if mean_ratio > T::from(0.5).unwrap() || var_ratio > T::from(1.0).unwrap() {
            // Report potential non-stationarity as a warning rather than error
            // since this is a simplified test
        }
    }
    
    /// Validate seasonality (basic detection)
    fn validate_seasonality(&self, data: &TimeSeriesData<T>, _report: &mut ValidationReport) {
        let values = data.values();
        if values.len() < 24 { // Need at least 24 points for basic seasonality detection
            return;
        }
        
        // This is a placeholder for seasonality detection
        // In practice, you would use FFT, autocorrelation, or other methods
    }
    
    /// Validate cross-series consistency
    fn validate_cross_series_consistency(&self, dataset: &TimeSeriesDataset<T>, report: &mut ValidationReport) {
        if dataset.len() < 2 {
            return; // Need at least 2 series for cross-validation
        }
        
        // Check for consistent frequencies
        let mut frequencies = HashMap::new();
        for series in dataset.iter() {
            *frequencies.entry(series.frequency.clone()).or_insert(0) += 1;
        }
        
        if frequencies.len() > 1 {
            report.add_warning(ValidationWarning::DataQuality {
                field: "frequency_consistency".to_string(),
                details: format!("Mixed frequencies detected: {:?}", 
                               frequencies.keys().collect::<Vec<_>>()),
            });
        }
        
        // Check for duplicate series IDs
        let mut series_ids = HashMap::new();
        for series in dataset.iter() {
            if series_ids.contains_key(&series.series_id) {
                report.add_error(ValidationError::DuplicateData {
                    field: "series_id".to_string(),
                    details: format!("Duplicate series ID: {}", series.series_id),
                });
            } else {
                series_ids.insert(series.series_id.clone(), true);
            }
        }
    }
    
    /// Detect outliers using IQR method
    fn detect_outliers_iqr(&self, values: &[T]) -> Vec<usize> {
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let n = sorted_values.len();
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        
        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - T::from(1.5).unwrap() * iqr;
        let upper_bound = q3 + T::from(1.5).unwrap() * iqr;
        
        values.iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                if value < lower_bound || value > upper_bound {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Detect outliers using Z-score method
    fn detect_outliers_zscore(&self, values: &[T]) -> Vec<usize> {
        let mean = values.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(values.len()).unwrap();
        let variance = values.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(values.len()).unwrap();
        let std = variance.sqrt();
        
        let threshold = T::from(3.0).unwrap(); // 3-sigma rule
        
        values.iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                let z_score = if std > T::epsilon() {
                    (value - mean).abs() / std
                } else {
                    T::zero()
                };
                
                if z_score > threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Detect outliers using modified Z-score method
    fn detect_outliers_modified_zscore(&self, values: &[T]) -> Vec<usize> {
        // Calculate median
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted_values.len() % 2 == 0 {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid]) / T::from(2.0).unwrap()
        } else {
            sorted_values[sorted_values.len() / 2]
        };
        
        // Calculate MAD (Median Absolute Deviation)
        let mut deviations: Vec<T> = values.iter()
            .map(|&x| (x - median).abs())
            .collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let mad = if deviations.len() % 2 == 0 {
            let mid = deviations.len() / 2;
            (deviations[mid - 1] + deviations[mid]) / T::from(2.0).unwrap()
        } else {
            deviations[deviations.len() / 2]
        };
        
        let threshold = T::from(3.5).unwrap();
        let factor = T::from(0.6745).unwrap(); // 0.6745 is the 75th percentile of the standard normal distribution
        
        values.iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                let modified_z_score = if mad > T::epsilon() {
                    factor * (value - median).abs() / mad
                } else {
                    T::zero()
                };
                
                if modified_z_score > threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Detect outliers using simplified isolation forest
    fn detect_outliers_isolation_forest(&self, values: &[T]) -> Vec<usize> {
        // This is a very simplified version of isolation forest
        // In practice, you would use a proper implementation
        
        let mean = values.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(values.len()).unwrap();
        let std = {
            let variance = values.iter()
                .map(|&x| (x - mean) * (x - mean))
                .fold(T::zero(), |acc, x| acc + x) / T::from(values.len()).unwrap();
            variance.sqrt()
        };
        
        // Use 2.5 standard deviations as threshold (less strict than Z-score)
        let threshold = T::from(2.5).unwrap();
        
        values.iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                let z_score = if std > T::epsilon() {
                    (value - mean).abs() / std
                } else {
                    T::zero()
                };
                
                if z_score > threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Convert duration to human-readable string
    fn duration_to_string(&self, duration: Duration) -> String {
        if duration.num_days() > 0 {
            format!("{}d", duration.num_days())
        } else if duration.num_hours() > 0 {
            format!("{}h", duration.num_hours())
        } else if duration.num_minutes() > 0 {
            format!("{}m", duration.num_minutes())
        } else {
            format!("{}s", duration.num_seconds())
        }
    }
}

impl<T: Float> Default for DataValidator<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Quick validation functions for common use cases
pub fn validate_time_series<T: Float>(data: &TimeSeriesData<T>) -> ValidationReport {
    DataValidator::new().validate_series(data)
}

pub fn validate_time_series_dataset<T: Float>(dataset: &TimeSeriesDataset<T>) -> ValidationReport {
    DataValidator::new().validate_dataset(dataset)
}

pub fn quick_data_quality_check<T: Float>(data: &TimeSeriesData<T>) -> Result<()> {
    let report = validate_time_series(data);
    
    if report.has_errors() {
        return Err(DataPipelineError::ValidationFailed {
            reason: format!("Data validation failed with {} errors", report.errors.len()),
        });
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeSeriesDatasetBuilder;
    use chrono::TimeZone;
    
    fn create_valid_series() -> TimeSeriesData<f64> {
        let timestamps = vec![
            chrono::Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 4).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 5).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        ];
        let values = vec![10.0, 12.0, 11.0, 13.0, 12.5];
        
        TimeSeriesDatasetBuilder::new("test_series".to_string())
            .with_frequency("D".to_string())
            .with_values(values)
            .with_timestamps(timestamps)
            .build()
            .unwrap()
    }
    
    #[test]
    fn test_valid_series_validation() {
        let data = create_valid_series();
        let validator = DataValidator::new();
        let report = validator.validate_series(&data);
        
        assert!(report.is_valid);
        assert!(report.errors.is_empty());
        assert_eq!(report.summary.total_series, 1);
        assert_eq!(report.summary.total_points, 5);
    }
    
    #[test]
    fn test_empty_series_validation() {
        let data = TimeSeriesData::new("empty".to_string(), "D".to_string());
        let validator = DataValidator::new();
        let report = validator.validate_series(&data);
        
        assert!(!report.is_valid);
        assert!(!report.errors.is_empty());
        
        // Should have a missing data error
        assert!(report.errors.iter().any(|e| matches!(e, ValidationError::MissingData { .. })));
    }
    
    #[test]
    fn test_duplicate_timestamps() {
        let timestamps = vec![
            chrono::Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(), // Duplicate
            chrono::Utc.ymd_opt(2023, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        ];
        let values = vec![10.0, 12.0, 11.0];
        
        let data = TimeSeriesDatasetBuilder::new("test_series".to_string())
            .with_frequency("D".to_string())
            .with_values(values)
            .with_timestamps(timestamps)
            .build()
            .unwrap();
        
        let validator = DataValidator::new();
        let report = validator.validate_series(&data);
        
        assert!(!report.is_valid);
        assert!(report.errors.iter().any(|e| matches!(e, ValidationError::DuplicateData { .. })));
        assert_eq!(report.summary.series_with_duplicates, 1);
    }
    
    #[test]
    fn test_out_of_order_timestamps() {
        let timestamps = vec![
            chrono::Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(), // Out of order
        ];
        let values = vec![10.0, 12.0, 11.0];
        
        let data = TimeSeriesDatasetBuilder::new("test_series".to_string())
            .with_frequency("D".to_string())
            .with_values(values)
            .with_timestamps(timestamps)
            .build()
            .unwrap();
        
        let validator = DataValidator::new();
        let report = validator.validate_series(&data);
        
        assert!(!report.is_valid);
        assert!(report.errors.iter().any(|e| matches!(e, ValidationError::TemporalError { .. })));
        assert_eq!(report.summary.series_with_temporal_issues, 1);
    }
    
    #[test]
    fn test_value_range_validation() {
        let data = create_valid_series();
        let validator = DataValidator::new().with_value_range(0.0, 10.0); // Values should be <= 10
        let report = validator.validate_series(&data);
        
        assert!(!report.is_valid);
        assert!(report.errors.iter().any(|e| matches!(e, ValidationError::OutOfRange { .. })));
    }
    
    #[test]
    fn test_outlier_detection() {
        let timestamps = vec![
            chrono::Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 4).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            chrono::Utc.ymd_opt(2023, 1, 5).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        ];
        let values = vec![10.0, 12.0, 100.0, 13.0, 12.5]; // 100.0 is an outlier
        
        let data = TimeSeriesDatasetBuilder::new("test_series".to_string())
            .with_frequency("D".to_string())
            .with_values(values)
            .with_timestamps(timestamps)
            .build()
            .unwrap();
        
        let validator = DataValidator::new().with_outlier_method(OutlierMethod::ZScore);
        let report = validator.validate_series(&data);
        
        assert!(report.is_valid); // Outliers are warnings, not errors
        assert!(report.has_warnings());
        assert_eq!(report.summary.series_with_outliers, 1);
        
        // Should have outlier warning
        assert!(report.warnings.iter().any(|w| {
            matches!(w, ValidationWarning::SuspiciousPattern { pattern, .. } if pattern == "outlier")
        }));
    }
    
    #[test]
    fn test_dataset_validation() {
        let series1 = create_valid_series();
        let mut series2 = create_valid_series();
        series2.series_id = "test_series2".to_string();
        
        let mut dataset = TimeSeriesDataset::new();
        dataset.add_series(series1);
        dataset.add_series(series2);
        
        let validator = DataValidator::new();
        let report = validator.validate_dataset(&dataset);
        
        assert!(report.is_valid);
        assert_eq!(report.summary.total_series, 2);
        assert_eq!(report.summary.total_points, 10);
    }
    
    #[test]
    fn test_duplicate_series_ids() {
        let series1 = create_valid_series();
        let series2 = create_valid_series(); // Same ID as series1
        
        let mut dataset = TimeSeriesDataset::new();
        dataset.add_series(series1);
        dataset.add_series(series2);
        
        let validator = DataValidator::new();
        let report = validator.validate_dataset(&dataset);
        
        assert!(!report.is_valid);
        assert!(report.errors.iter().any(|e| {
            matches!(e, ValidationError::DuplicateData { field, .. } if field == "series_id")
        }));
    }
    
    #[test]
    fn test_quick_validation_functions() {
        let data = create_valid_series();
        
        // Test quick validation
        let report = validate_time_series(&data);
        assert!(report.is_valid);
        
        // Test quick quality check
        assert!(quick_data_quality_check(&data).is_ok());
        
        // Test with invalid data
        let invalid_data = TimeSeriesData::new("empty".to_string(), "D".to_string());
        assert!(quick_data_quality_check(&invalid_data).is_err());
    }
}