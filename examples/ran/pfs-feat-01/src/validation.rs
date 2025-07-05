use crate::config::*;
use crate::error::*;
use crate::stats::*;

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, instrument, warn};

/// Feature validation framework
#[derive(Debug, Clone)]
pub struct FeatureValidator {
    config: FeatureEngineConfig,
    validation_rules: Vec<ValidationRule>,
}

impl FeatureValidator {
    /// Create a new feature validator
    pub fn new(config: FeatureEngineConfig) -> Self {
        let validation_rules = Self::create_default_validation_rules();
        
        Self {
            config,
            validation_rules,
        }
    }

    /// Create default validation rules
    fn create_default_validation_rules() -> Vec<ValidationRule> {
        vec![
            ValidationRule::RequiredColumns(vec![
                "timestamp".to_string(),
                "kpi_value".to_string(),
            ]),
            ValidationRule::NonEmptyDataFrame,
            ValidationRule::NoAllNullColumns,
            ValidationRule::TimestampFormat,
            ValidationRule::NumericKpiValues,
            ValidationRule::FeatureCountIncrease,
            ValidationRule::RowCountPreservation,
            ValidationRule::NoInfiniteValues,
            ValidationRule::ReasonableValueRanges(HashMap::from([
                ("kpi_value".to_string(), (-1000.0, 10000.0)),
                ("prb_utilization_dl".to_string(), (0.0, 100.0)),
                ("prb_utilization_ul".to_string(), (0.0, 100.0)),
                ("rsrp_avg".to_string(), (-150.0, -30.0)),
                ("sinr_avg".to_string(), (-30.0, 50.0)),
            ])),
        ]
    }

    /// Validate feature generation output
    #[instrument(skip(self, input_df, output_df))]
    pub async fn validate_features(
        &self,
        input_df: &DataFrame,
        output_df: &DataFrame,
        expected_features: &[String],
    ) -> FeatureEngineResult<ValidationResult> {
        info!("Starting feature validation");
        
        let mut validation_errors = Vec::new();
        let mut validation_warnings = Vec::new();
        let mut validation_stats = ValidationStats::new();

        // Run all validation rules
        for rule in &self.validation_rules {
            match self.apply_validation_rule(rule, input_df, output_df).await {
                Ok(result) => {
                    validation_stats.add_check_result(&result);
                    if !result.is_valid {
                        validation_errors.extend(result.errors);
                        validation_warnings.extend(result.warnings);
                    }
                }
                Err(e) => {
                    validation_errors.push(format!("Validation rule error: {}", e));
                }
            }
        }

        // Validate expected features
        let expected_feature_result = self.validate_expected_features(output_df, expected_features).await;
        validation_stats.add_check_result(&expected_feature_result);
        if !expected_feature_result.is_valid {
            validation_errors.extend(expected_feature_result.errors);
            validation_warnings.extend(expected_feature_result.warnings);
        }

        // Validate feature quality
        let quality_result = self.validate_feature_quality(output_df).await;
        validation_stats.add_check_result(&quality_result);
        if !quality_result.is_valid {
            validation_errors.extend(quality_result.errors);
            validation_warnings.extend(quality_result.warnings);
        }

        let is_valid = validation_errors.is_empty();
        
        info!(
            "Feature validation completed: {} errors, {} warnings",
            validation_errors.len(),
            validation_warnings.len()
        );

        Ok(ValidationResult {
            is_valid,
            errors: validation_errors,
            warnings: validation_warnings,
            validation_stats,
            feature_quality_scores: self.calculate_feature_quality_scores(output_df).await,
        })
    }

    /// Apply a single validation rule
    async fn apply_validation_rule(
        &self,
        rule: &ValidationRule,
        input_df: &DataFrame,
        output_df: &DataFrame,
    ) -> FeatureEngineResult<ValidationCheckResult> {
        match rule {
            ValidationRule::RequiredColumns(columns) => {
                self.validate_required_columns(output_df, columns).await
            }
            ValidationRule::NonEmptyDataFrame => {
                self.validate_non_empty_dataframe(output_df).await
            }
            ValidationRule::NoAllNullColumns => {
                self.validate_no_all_null_columns(output_df).await
            }
            ValidationRule::TimestampFormat => {
                self.validate_timestamp_format(output_df).await
            }
            ValidationRule::NumericKpiValues => {
                self.validate_numeric_kpi_values(output_df).await
            }
            ValidationRule::FeatureCountIncrease => {
                self.validate_feature_count_increase(input_df, output_df).await
            }
            ValidationRule::RowCountPreservation => {
                self.validate_row_count_preservation(input_df, output_df).await
            }
            ValidationRule::NoInfiniteValues => {
                self.validate_no_infinite_values(output_df).await
            }
            ValidationRule::ReasonableValueRanges(ranges) => {
                self.validate_reasonable_value_ranges(output_df, ranges).await
            }
        }
    }

    /// Validate required columns
    async fn validate_required_columns(
        &self,
        df: &DataFrame,
        required_columns: &[String],
    ) -> FeatureEngineResult<ValidationCheckResult> {
        let mut errors = Vec::new();
        let column_names = df.get_column_names();
        
        for required_col in required_columns {
            if !column_names.contains(&required_col.as_str()) {
                errors.push(format!("Required column '{}' is missing", required_col));
            }
        }

        Ok(ValidationCheckResult {
            rule_name: "RequiredColumns".to_string(),
            is_valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
        })
    }

    /// Validate non-empty DataFrame
    async fn validate_non_empty_dataframe(
        &self,
        df: &DataFrame,
    ) -> FeatureEngineResult<ValidationCheckResult> {
        let is_valid = df.height() > 0;
        let errors = if !is_valid {
            vec!["DataFrame is empty".to_string()]
        } else {
            Vec::new()
        };

        Ok(ValidationCheckResult {
            rule_name: "NonEmptyDataFrame".to_string(),
            is_valid,
            errors,
            warnings: Vec::new(),
        })
    }

    /// Validate no all-null columns
    async fn validate_no_all_null_columns(
        &self,
        df: &DataFrame,
    ) -> FeatureEngineResult<ValidationCheckResult> {
        let mut errors = Vec::new();
        
        for column_name in df.get_column_names() {
            if let Ok(column) = df.column(column_name) {
                if column.null_count() == column.len() {
                    errors.push(format!("Column '{}' contains only null values", column_name));
                }
            }
        }

        Ok(ValidationCheckResult {
            rule_name: "NoAllNullColumns".to_string(),
            is_valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
        })
    }

    /// Validate timestamp format
    async fn validate_timestamp_format(
        &self,
        df: &DataFrame,
    ) -> FeatureEngineResult<ValidationCheckResult> {
        let mut errors = Vec::new();
        
        if let Ok(timestamp_col) = df.column("timestamp") {
            if !matches!(timestamp_col.dtype(), DataType::Datetime(_, _)) {
                errors.push("Timestamp column must be of datetime type".to_string());
            }
        }

        Ok(ValidationCheckResult {
            rule_name: "TimestampFormat".to_string(),
            is_valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
        })
    }

    /// Validate numeric KPI values
    async fn validate_numeric_kpi_values(
        &self,
        df: &DataFrame,
    ) -> FeatureEngineResult<ValidationCheckResult> {
        let mut errors = Vec::new();
        
        if let Ok(kpi_col) = df.column("kpi_value") {
            if !kpi_col.dtype().is_numeric() {
                errors.push("KPI value column must be numeric".to_string());
            }
        }

        Ok(ValidationCheckResult {
            rule_name: "NumericKpiValues".to_string(),
            is_valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
        })
    }

    /// Validate feature count increase
    async fn validate_feature_count_increase(
        &self,
        input_df: &DataFrame,
        output_df: &DataFrame,
    ) -> FeatureEngineResult<ValidationCheckResult> {
        let input_width = input_df.width();
        let output_width = output_df.width();
        
        let is_valid = output_width > input_width;
        let errors = if !is_valid {
            vec![format!(
                "Feature count did not increase: input {} -> output {}",
                input_width, output_width
            )]
        } else {
            Vec::new()
        };

        Ok(ValidationCheckResult {
            rule_name: "FeatureCountIncrease".to_string(),
            is_valid,
            errors,
            warnings: Vec::new(),
        })
    }

    /// Validate row count preservation
    async fn validate_row_count_preservation(
        &self,
        input_df: &DataFrame,
        output_df: &DataFrame,
    ) -> FeatureEngineResult<ValidationCheckResult> {
        let input_height = input_df.height();
        let output_height = output_df.height();
        
        let is_valid = input_height == output_height;
        let errors = if !is_valid {
            vec![format!(
                "Row count changed: input {} -> output {}",
                input_height, output_height
            )]
        } else {
            Vec::new()
        };

        Ok(ValidationCheckResult {
            rule_name: "RowCountPreservation".to_string(),
            is_valid,
            errors,
            warnings: Vec::new(),
        })
    }

    /// Validate no infinite values
    async fn validate_no_infinite_values(
        &self,
        df: &DataFrame,
    ) -> FeatureEngineResult<ValidationCheckResult> {
        let mut errors = Vec::new();
        
        for column_name in df.get_column_names() {
            if let Ok(column) = df.column(column_name) {
                if column.dtype().is_numeric() {
                    // Check for infinite values in numeric columns
                    if let Ok(series) = column.cast(&DataType::Float64) {
                        let has_infinite = series.f64()
                            .map(|ca| ca.iter().any(|opt| opt.map_or(false, |v| v.is_infinite())))
                            .unwrap_or(false);
                        
                        if has_infinite {
                            errors.push(format!("Column '{}' contains infinite values", column_name));
                        }
                    }
                }
            }
        }

        Ok(ValidationCheckResult {
            rule_name: "NoInfiniteValues".to_string(),
            is_valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
        })
    }

    /// Validate reasonable value ranges
    async fn validate_reasonable_value_ranges(
        &self,
        df: &DataFrame,
        ranges: &HashMap<String, (f64, f64)>,
    ) -> FeatureEngineResult<ValidationCheckResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        for (column_name, (min_val, max_val)) in ranges {
            if let Ok(column) = df.column(column_name) {
                if column.dtype().is_numeric() {
                    if let Ok(series) = column.cast(&DataType::Float64) {
                        if let Ok(float_series) = series.f64() {
                            for opt_val in float_series.iter() {
                                if let Some(val) = opt_val {
                                    if val < *min_val || val > *max_val {
                                        warnings.push(format!(
                                            "Column '{}' has value {} outside reasonable range [{}, {}]",
                                            column_name, val, min_val, max_val
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(ValidationCheckResult {
            rule_name: "ReasonableValueRanges".to_string(),
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }

    /// Validate expected features
    async fn validate_expected_features(
        &self,
        df: &DataFrame,
        expected_features: &[String],
    ) -> ValidationCheckResult {
        let mut errors = Vec::new();
        let column_names = df.get_column_names();
        
        for expected_feature in expected_features {
            if !column_names.contains(&expected_feature.as_str()) {
                errors.push(format!("Expected feature '{}' is missing", expected_feature));
            }
        }

        ValidationCheckResult {
            rule_name: "ExpectedFeatures".to_string(),
            is_valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
        }
    }

    /// Validate feature quality
    async fn validate_feature_quality(&self, df: &DataFrame) -> ValidationCheckResult {
        let mut warnings = Vec::new();
        
        for column_name in df.get_column_names() {
            if let Ok(column) = df.column(column_name) {
                let null_percentage = (column.null_count() as f64 / column.len() as f64) * 100.0;
                
                if null_percentage > 50.0 {
                    warnings.push(format!(
                        "Feature '{}' has high null percentage: {:.1}%",
                        column_name, null_percentage
                    ));
                }
                
                // Check for constant features
                if column.dtype().is_numeric() {
                    if let Ok(unique_count) = column.n_unique() {
                        if unique_count <= 1 {
                            warnings.push(format!(
                                "Feature '{}' appears to be constant (unique values: {})",
                                column_name, unique_count
                            ));
                        }
                    }
                }
            }
        }

        ValidationCheckResult {
            rule_name: "FeatureQuality".to_string(),
            is_valid: true, // Warnings don't make validation invalid
            errors: Vec::new(),
            warnings,
        }
    }

    /// Calculate feature quality scores
    async fn calculate_feature_quality_scores(&self, df: &DataFrame) -> HashMap<String, f64> {
        let mut quality_scores = HashMap::new();
        
        for column_name in df.get_column_names() {
            if let Ok(column) = df.column(column_name) {
                let mut score = 1.0;
                
                // Penalize for null values
                let null_percentage = column.null_count() as f64 / column.len() as f64;
                score -= null_percentage * 0.5; // Max penalty of 0.5 for 100% nulls
                
                // Reward for variety (if numeric)
                if column.dtype().is_numeric() {
                    if let Ok(unique_count) = column.n_unique() {
                        let variety_score = (unique_count as f64 / column.len() as f64).min(1.0);
                        score = score * (0.5 + 0.5 * variety_score); // Scale by variety
                    }
                }
                
                quality_scores.insert(column_name.to_string(), score.max(0.0));
            }
        }
        
        quality_scores
    }

    /// Validate 1000 time-series sample (acceptance criteria)
    #[instrument(skip(self, output_directory))]
    pub async fn validate_sample_batch(
        &self,
        output_directory: &Path,
        expected_series_count: usize,
    ) -> FeatureEngineResult<SampleValidationResult> {
        info!("Validating sample batch with {} expected series", expected_series_count);
        
        // List all parquet files in output directory
        let mut parquet_files = Vec::new();
        if output_directory.exists() {
            let entries = std::fs::read_dir(output_directory)
                .map_err(|e| FeatureEngineError::io(e))?;
            
            for entry in entries {
                let entry = entry.map_err(|e| FeatureEngineError::io(e))?;
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "parquet") {
                    parquet_files.push(path);
                }
            }
        }
        
        let actual_series_count = parquet_files.len();
        let mut validation_errors = Vec::new();
        let mut processed_series = 0;
        let mut total_features_generated = 0;
        
        // Validate each file
        for file_path in &parquet_files {
            match self.validate_single_file(file_path).await {
                Ok(file_result) => {
                    processed_series += 1;
                    total_features_generated += file_result.feature_count;
                    if !file_result.is_valid {
                        validation_errors.extend(file_result.errors);
                    }
                }
                Err(e) => {
                    validation_errors.push(format!("Failed to validate file {:?}: {}", file_path, e));
                }
            }
        }
        
        // Check if we have the expected number of series
        if actual_series_count != expected_series_count {
            validation_errors.push(format!(
                "Expected {} series, found {}",
                expected_series_count, actual_series_count
            ));
        }
        
        let is_valid = validation_errors.is_empty();
        
        info!(
            "Sample batch validation completed: {} series processed, {} features total, {} errors",
            processed_series, total_features_generated, validation_errors.len()
        );
        
        Ok(SampleValidationResult {
            is_valid,
            expected_series_count,
            actual_series_count,
            processed_series,
            total_features_generated,
            errors: validation_errors,
        })
    }

    /// Validate a single file
    async fn validate_single_file(&self, file_path: &Path) -> FeatureEngineResult<SingleFileValidationResult> {
        let df = LazyFrame::scan_parquet(file_path, ScanArgsParquet::default())
            .map_err(|e| FeatureEngineError::data_processing(format!("Failed to read parquet file: {}", e)))?
            .collect()
            .map_err(|e| FeatureEngineError::data_processing(format!("Failed to collect dataframe: {}", e)))?;
        
        let mut errors = Vec::new();
        
        // Basic validations
        if df.height() == 0 {
            errors.push("File contains no data".to_string());
        }
        
        if df.width() == 0 {
            errors.push("File contains no columns".to_string());
        }
        
        // Check for required columns
        let required_columns = ["timestamp", "kpi_value"];
        for col in required_columns {
            if !df.get_column_names().contains(&col) {
                errors.push(format!("Missing required column: {}", col));
            }
        }
        
        Ok(SingleFileValidationResult {
            file_path: file_path.to_path_buf(),
            is_valid: errors.is_empty(),
            feature_count: df.width(),
            row_count: df.height(),
            errors,
        })
    }

    /// Add custom validation rule
    pub fn add_validation_rule(&mut self, rule: ValidationRule) {
        self.validation_rules.push(rule);
    }

    /// Remove validation rule
    pub fn remove_validation_rule(&mut self, rule_name: &str) {
        self.validation_rules.retain(|rule| rule.name() != rule_name);
    }
}

/// Validation rules
#[derive(Debug, Clone)]
pub enum ValidationRule {
    RequiredColumns(Vec<String>),
    NonEmptyDataFrame,
    NoAllNullColumns,
    TimestampFormat,
    NumericKpiValues,
    FeatureCountIncrease,
    RowCountPreservation,
    NoInfiniteValues,
    ReasonableValueRanges(HashMap<String, (f64, f64)>),
}

impl ValidationRule {
    pub fn name(&self) -> &str {
        match self {
            ValidationRule::RequiredColumns(_) => "RequiredColumns",
            ValidationRule::NonEmptyDataFrame => "NonEmptyDataFrame",
            ValidationRule::NoAllNullColumns => "NoAllNullColumns",
            ValidationRule::TimestampFormat => "TimestampFormat",
            ValidationRule::NumericKpiValues => "NumericKpiValues",
            ValidationRule::FeatureCountIncrease => "FeatureCountIncrease",
            ValidationRule::RowCountPreservation => "RowCountPreservation",
            ValidationRule::NoInfiniteValues => "NoInfiniteValues",
            ValidationRule::ReasonableValueRanges(_) => "ReasonableValueRanges",
        }
    }
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub validation_stats: ValidationStats,
    pub feature_quality_scores: HashMap<String, f64>,
}

/// Validation check result
#[derive(Debug, Clone)]
pub struct ValidationCheckResult {
    pub rule_name: String,
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStats {
    pub total_checks: usize,
    pub passed_checks: usize,
    pub failed_checks: usize,
    pub total_errors: usize,
    pub total_warnings: usize,
}

impl ValidationStats {
    pub fn new() -> Self {
        Self {
            total_checks: 0,
            passed_checks: 0,
            failed_checks: 0,
            total_errors: 0,
            total_warnings: 0,
        }
    }

    pub fn add_check_result(&mut self, result: &ValidationCheckResult) {
        self.total_checks += 1;
        if result.is_valid {
            self.passed_checks += 1;
        } else {
            self.failed_checks += 1;
        }
        self.total_errors += result.errors.len();
        self.total_warnings += result.warnings.len();
    }
}

/// Sample validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleValidationResult {
    pub is_valid: bool,
    pub expected_series_count: usize,
    pub actual_series_count: usize,
    pub processed_series: usize,
    pub total_features_generated: usize,
    pub errors: Vec<String>,
}

/// Single file validation result
#[derive(Debug, Clone)]
pub struct SingleFileValidationResult {
    pub file_path: std::path::PathBuf,
    pub is_valid: bool,
    pub feature_count: usize,
    pub row_count: usize,
    pub errors: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    
    fn create_test_config() -> FeatureEngineConfig {
        FeatureEngineConfig::default()
    }
    
    fn create_test_dataframe() -> DataFrame {
        let timestamps = vec![
            Utc::now() - chrono::Duration::hours(3),
            Utc::now() - chrono::Duration::hours(2),
            Utc::now() - chrono::Duration::hours(1),
            Utc::now(),
        ];
        
        df! {
            "timestamp" => timestamps,
            "kpi_value" => [10.0, 15.0, 12.0, 18.0],
            "cell_id" => ["Cell_A", "Cell_A", "Cell_A", "Cell_A"],
        }.unwrap()
    }
    
    #[tokio::test]
    async fn test_feature_validator_creation() {
        let config = create_test_config();
        let validator = FeatureValidator::new(config);
        
        assert!(!validator.validation_rules.is_empty());
    }
    
    #[tokio::test]
    async fn test_validate_required_columns() {
        let config = create_test_config();
        let validator = FeatureValidator::new(config);
        
        let df = create_test_dataframe();
        let required_columns = vec!["timestamp".to_string(), "kpi_value".to_string()];
        
        let result = validator.validate_required_columns(&df, &required_columns).await.unwrap();
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }
    
    #[tokio::test]
    async fn test_validate_feature_quality() {
        let config = create_test_config();
        let validator = FeatureValidator::new(config);
        
        let df = create_test_dataframe();
        let result = validator.validate_feature_quality(&df).await;
        
        assert!(result.is_valid);
    }
    
    #[tokio::test]
    async fn test_calculate_quality_scores() {
        let config = create_test_config();
        let validator = FeatureValidator::new(config);
        
        let df = create_test_dataframe();
        let scores = validator.calculate_feature_quality_scores(&df).await;
        
        assert!(!scores.is_empty());
        assert!(scores.values().all(|&score| score >= 0.0 && score <= 1.0));
    }
}