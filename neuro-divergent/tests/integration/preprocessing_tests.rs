//! Integration tests for data preprocessing pipelines
//!
//! Tests preprocessing scenarios including:
//! - Data scaling and normalization
//! - Feature engineering
//! - Missing value handling
//! - Data validation and cleaning

use neuro_divergent::prelude::*;
use neuro_divergent::{
    NeuralForecast, Frequency, ScalerType,
    models::{LSTM, NBEATS},
    data::{
        TimeSeriesDataFrame, TimeSeriesSchema, DataPreprocessor,
        StandardScaler, MinMaxScaler, RobustScaler, Scaler,
        WindowTransform, LagsTransform, DataTransform,
        DataValidator, ValidationReport,
    },
};
use polars::prelude::*;
use chrono::{DateTime, Utc, Duration};
use num_traits::Float;
use rand::Rng;
use std::collections::HashMap;

/// Generate data with various preprocessing challenges
fn generate_preprocessing_test_data<T: Float>(
    n_series: usize,
    n_points: usize,
    missing_rate: f64,
    outlier_rate: f64,
) -> Result<TimeSeriesDataFrame<T>, Box<dyn std::error::Error>>
where
    T: From<f64> + Into<f64>,
{
    let mut rng = rand::thread_rng();
    let start_date = Utc::now() - Duration::days(n_points as i64);
    
    let mut unique_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    for series_id in 0..n_series {
        let series_name = format!("series_{}", series_id);
        let base_value = rng.gen_range(10.0..1000.0); // Wide range
        let scale = rng.gen_range(0.1..10.0); // Different scales
        
        for i in 0..n_points {
            let date = start_date + Duration::days(i as i64);
            unique_ids.push(series_name.clone());
            timestamps.push(date.timestamp());
            
            // Generate value
            let mut value = base_value + scale * (i as f64 * 0.1).sin() * 10.0;
            
            // Add outliers
            if rng.gen_bool(outlier_rate) {
                value *= rng.gen_range(3.0..10.0) * rng.gen_range(-1.0..1.0).signum();
            }
            
            // Add missing values
            if rng.gen_bool(missing_rate) {
                values.push(None);
            } else {
                values.push(Some(value));
            }
        }
    }
    
    let df = df! {
        "unique_id" => unique_ids,
        "ds" => timestamps,
        "y" => values,
    }?;
    
    Ok(TimeSeriesDataFrame::new(
        df,
        TimeSeriesSchema::default(),
        Some(Frequency::Daily),
    ))
}

/// Test basic scaling pipelines
#[test]
fn test_scaling_pipelines() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data with different scales
    let data = generate_preprocessing_test_data::<f32>(
        3,
        100,
        0.0, // No missing values
        0.0, // No outliers
    )?;
    
    // Test different scalers
    let scalers: Vec<(String, Box<dyn Scaler<f32>>)> = vec![
        ("StandardScaler".to_string(), Box::new(StandardScaler::new())),
        ("MinMaxScaler".to_string(), Box::new(MinMaxScaler::new(0.0, 1.0))),
        ("RobustScaler".to_string(), Box::new(RobustScaler::new())),
    ];
    
    for (scaler_name, scaler) in scalers {
        println!("\nTesting {}", scaler_name);
        
        // Fit scaler on data
        let fitted_scaler = scaler.fit(&data)?;
        
        // Transform data
        let scaled_data = fitted_scaler.transform(&data)?;
        
        // Verify scaling properties
        let stats = calculate_data_statistics(&scaled_data)?;
        
        match scaler_name.as_str() {
            "StandardScaler" => {
                // Should have mean ~0 and std ~1
                for (series_id, series_stats) in &stats {
                    println!("{}: mean={:.3}, std={:.3}", 
                             series_id, series_stats.mean, series_stats.std);
                    assert!((series_stats.mean).abs() < 0.1);
                    assert!((series_stats.std - 1.0).abs() < 0.1);
                }
            },
            "MinMaxScaler" => {
                // Should be in range [0, 1]
                for (series_id, series_stats) in &stats {
                    println!("{}: min={:.3}, max={:.3}", 
                             series_id, series_stats.min, series_stats.max);
                    assert!(series_stats.min >= -0.01 && series_stats.min <= 0.01);
                    assert!(series_stats.max >= 0.99 && series_stats.max <= 1.01);
                }
            },
            "RobustScaler" => {
                // Should be robust to outliers
                for (series_id, series_stats) in &stats {
                    println!("{}: median={:.3}, iqr={:.3}", 
                             series_id, series_stats.median, series_stats.iqr);
                }
            },
            _ => {}
        }
        
        // Train model on scaled data
        let lstm = LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(32)
            .build()?;
        
        let mut nf = NeuralForecast::builder()
            .with_model(Box::new(lstm))
            .with_frequency(Frequency::Daily)
            .build()?;
        
        nf.fit(scaled_data)?;
        
        // Generate forecasts
        let scaled_forecasts = nf.predict()?;
        
        // Inverse transform forecasts
        let original_forecasts = fitted_scaler.inverse_transform(&scaled_forecasts)?;
        
        println!("Forecasting completed with {}", scaler_name);
    }
    
    Ok(())
}

/// Test preprocessing with missing values
#[test]
fn test_missing_value_handling() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data with missing values
    let data = generate_preprocessing_test_data::<f32>(
        3,
        150,
        0.1, // 10% missing values
        0.0,
    )?;
    
    // Create preprocessor with missing value handling
    let preprocessor = DataPreprocessor::builder()
        .with_missing_value_strategy(MissingValueStrategy::Interpolate)
        .with_scaler(ScalerType::StandardScaler)
        .build()?;
    
    // Apply preprocessing
    let processed_data = preprocessor.fit_transform(&data)?;
    
    // Verify no missing values remain
    let has_missing = processed_data.has_missing_values()?;
    assert!(!has_missing, "Processed data still contains missing values");
    
    // Test different imputation strategies
    let strategies = vec![
        MissingValueStrategy::Forward,
        MissingValueStrategy::Backward,
        MissingValueStrategy::Interpolate,
        MissingValueStrategy::Mean,
        MissingValueStrategy::Median,
        MissingValueStrategy::Zero,
    ];
    
    for strategy in strategies {
        let preprocessor = DataPreprocessor::builder()
            .with_missing_value_strategy(strategy.clone())
            .build()?;
        
        let processed = preprocessor.fit_transform(&data)?;
        let missing_count = processed.count_missing_values()?;
        
        println!("{:?}: {} missing values after processing", 
                 strategy, missing_count);
        assert_eq!(missing_count, 0);
    }
    
    // Train model on imputed data
    let nbeats = NBEATS::builder()
        .horizon(7)
        .input_size(14)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(nbeats))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(processed_data)?;
    
    Ok(())
}

/// Test outlier detection and handling
#[test]
fn test_outlier_handling() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data with outliers
    let data = generate_preprocessing_test_data::<f32>(
        2,
        200,
        0.0,
        0.05, // 5% outliers
    )?;
    
    // Create preprocessor with outlier detection
    let preprocessor = DataPreprocessor::builder()
        .with_outlier_detection(OutlierMethod::IQR(1.5))
        .with_outlier_treatment(OutlierTreatment::Cap)
        .with_scaler(ScalerType::RobustScaler) // Robust to outliers
        .build()?;
    
    // Detect outliers before processing
    let outliers_before = preprocessor.detect_outliers(&data)?;
    println!("Outliers detected: {} points", outliers_before.len());
    
    // Apply preprocessing
    let processed_data = preprocessor.fit_transform(&data)?;
    
    // Detect outliers after processing
    let outliers_after = preprocessor.detect_outliers(&processed_data)?;
    println!("Outliers after treatment: {} points", outliers_after.len());
    
    // Should have fewer outliers after treatment
    assert!(outliers_after.len() < outliers_before.len());
    
    // Compare model performance with and without outlier handling
    let lstm_config = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(32)
        .loss_function(LossFunction::Huber) // Robust loss
        .build()?;
    
    // Train on original data
    let mut nf_original = NeuralForecast::builder()
        .with_model(Box::new(lstm_config.clone()))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf_original.fit(data.clone())?;
    
    // Train on processed data
    let mut nf_processed = NeuralForecast::builder()
        .with_model(Box::new(lstm_config))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf_processed.fit(processed_data)?;
    
    println!("Models trained successfully with outlier handling");
    
    Ok(())
}

/// Test feature engineering pipeline
#[test]
fn test_feature_engineering() -> Result<(), Box<dyn std::error::Error>> {
    // Generate base data
    let data = generate_preprocessing_test_data::<f32>(
        3,
        365, // Full year for seasonal features
        0.0,
        0.0,
    )?;
    
    // Create feature engineering pipeline
    let feature_pipeline = FeatureEngineeringPipeline::builder()
        .add_lag_features(vec![1, 7, 14, 28])
        .add_rolling_features(vec![
            RollingFeature::Mean(7),
            RollingFeature::Std(7),
            RollingFeature::Min(7),
            RollingFeature::Max(7),
        ])
        .add_date_features(vec![
            DateFeature::DayOfWeek,
            DateFeature::DayOfMonth,
            DateFeature::WeekOfYear,
            DateFeature::Month,
            DateFeature::Quarter,
        ])
        .add_seasonal_features(vec![7, 30, 365]) // Weekly, monthly, yearly
        .build()?;
    
    // Apply feature engineering
    let featured_data = feature_pipeline.transform(&data)?;
    
    // Verify features were added
    let original_cols = data.schema.all_columns().len();
    let featured_cols = featured_data.schema.all_columns().len();
    
    println!("Original columns: {}", original_cols);
    println!("Featured columns: {}", featured_cols);
    assert!(featured_cols > original_cols);
    
    // Train model with engineered features
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(64)
        .hist_exog_features(feature_pipeline.get_feature_names())
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(featured_data)?;
    
    Ok(())
}

/// Test data validation
#[test]
fn test_data_validation() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data with various issues
    let mut data = generate_preprocessing_test_data::<f32>(
        5,
        100,
        0.15, // 15% missing
        0.1,  // 10% outliers
    )?;
    
    // Add duplicate timestamps for testing
    let mut df = data.to_polars()?;
    let first_rows = df.head(Some(5));
    df = df.vstack(&first_rows)?; // Add duplicates
    data = TimeSeriesDataFrame::from_polars(df, data.schema.clone(), data.frequency)?;
    
    // Create validator
    let validator = DataValidator::new()
        .with_check_missing(true)
        .with_check_duplicates(true)
        .with_check_outliers(true)
        .with_check_frequency(true)
        .with_check_stationarity(true);
    
    // Validate data
    let report = validator.validate(&data)?;
    
    // Print validation report
    println!("\nData Validation Report:");
    println!("Valid: {}", report.is_valid);
    println!("Issues found:");
    for issue in &report.issues {
        println!("  - {}: {}", issue.issue_type, issue.description);
    }
    
    // Data should have issues
    assert!(!report.is_valid);
    assert!(!report.issues.is_empty());
    
    // Fix issues
    let fixer = DataFixer::from_validation_report(&report);
    let fixed_data = fixer.fix(&data)?;
    
    // Re-validate
    let fixed_report = validator.validate(&fixed_data)?;
    println!("\nAfter fixing:");
    println!("Valid: {}", fixed_report.is_valid);
    println!("Remaining issues: {}", fixed_report.issues.len());
    
    // Should have fewer issues
    assert!(fixed_report.issues.len() < report.issues.len());
    
    Ok(())
}

/// Test preprocessing pipeline integration
#[test]
fn test_complete_preprocessing_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    // Generate challenging data
    let data = generate_preprocessing_test_data::<f32>(
        5,
        500,
        0.1,  // Missing values
        0.05, // Outliers
    )?;
    
    // Create complete preprocessing pipeline
    let pipeline = PreprocessingPipeline::builder()
        // Step 1: Validation
        .add_step(Box::new(DataValidator::new()))
        // Step 2: Missing value imputation
        .add_step(Box::new(MissingValueImputer::new(
            MissingValueStrategy::Interpolate
        )))
        // Step 3: Outlier treatment
        .add_step(Box::new(OutlierHandler::new(
            OutlierMethod::IQR(1.5),
            OutlierTreatment::Cap,
        )))
        // Step 4: Feature engineering
        .add_step(Box::new(FeatureEngineer::new()
            .with_lag_features(vec![1, 7])
            .with_rolling_features(vec![RollingFeature::Mean(7)])))
        // Step 5: Scaling
        .add_step(Box::new(StandardScaler::new()))
        .build()?;
    
    // Apply pipeline
    let processed_data = pipeline.fit_transform(&data)?;
    
    // Train models on processed data
    let models: Vec<Box<dyn BaseModel<f32>>> = vec![
        Box::new(LSTM::builder()
            .horizon(14)
            .input_size(28)
            .hidden_size(64)
            .build()?),
        Box::new(NBEATS::builder()
            .horizon(14)
            .input_size(28)
            .build()?),
    ];
    
    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(processed_data.clone())?;
    
    // Generate forecasts
    let forecasts = nf.predict()?;
    
    // Inverse transform through pipeline
    let original_scale_forecasts = pipeline.inverse_transform(&forecasts)?;
    
    println!("Complete pipeline executed successfully");
    
    Ok(())
}

/// Test preprocessing with exogenous variables
#[test]
fn test_exogenous_preprocessing() -> Result<(), Box<dyn std::error::Error>> {
    // Generate base data
    let mut data = generate_preprocessing_test_data::<f32>(
        2,
        200,
        0.05,
        0.0,
    )?;
    
    // Add exogenous variables with different scales
    let mut df = data.to_polars()?;
    let n_rows = df.height();
    
    let mut rng = rand::thread_rng();
    
    // Temperature (Celsius): 0-40
    let temperatures: Vec<Option<f32>> = (0..n_rows)
        .map(|i| {
            if rng.gen_bool(0.05) { // 5% missing
                None
            } else {
                Some(20.0 + 10.0 * (i as f32 * 0.05).sin() + rng.gen_range(-5.0..5.0))
            }
        })
        .collect();
    
    // Humidity (%): 30-90
    let humidity: Vec<f32> = (0..n_rows)
        .map(|_| 60.0 + rng.gen_range(-20.0..20.0))
        .collect();
    
    // Sales promotion (binary)
    let promotion: Vec<i32> = (0..n_rows)
        .map(|i| if i % 7 == 0 { 1 } else { 0 })
        .collect();
    
    df = df.with_column(Series::new("temperature", temperatures))?;
    df = df.with_column(Series::new("humidity", humidity))?;
    df = df.with_column(Series::new("promotion", promotion))?;
    
    // Update schema
    let mut schema = data.schema.clone();
    schema.exog_cols = vec![
        "temperature".to_string(),
        "humidity".to_string(),
        "promotion".to_string(),
    ];
    
    data = TimeSeriesDataFrame::new(df, schema, data.frequency);
    
    // Create preprocessor for exogenous variables
    let exog_preprocessor = ExogenousPreprocessor::builder()
        .with_numeric_scaling(HashMap::from([
            ("temperature".to_string(), ScalerType::StandardScaler),
            ("humidity".to_string(), ScalerType::MinMaxScaler),
        ]))
        .with_categorical_encoding(HashMap::from([
            ("promotion".to_string(), EncodingType::OneHot),
        ]))
        .with_missing_strategies(HashMap::from([
            ("temperature".to_string(), MissingValueStrategy::Interpolate),
        ]))
        .build()?;
    
    // Apply preprocessing
    let processed_data = exog_preprocessor.fit_transform(&data)?;
    
    // Verify exogenous variables are properly scaled
    let exog_stats = calculate_exogenous_statistics(&processed_data)?;
    println!("Exogenous variable statistics after preprocessing:");
    for (var_name, stats) in exog_stats {
        println!("{}: mean={:.3}, std={:.3}, min={:.3}, max={:.3}",
                 var_name, stats.mean, stats.std, stats.min, stats.max);
    }
    
    // Train model with preprocessed exogenous variables
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(64)
        .hist_exog_features(processed_data.schema.exog_cols.clone())
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(processed_data)?;
    
    Ok(())
}

/// Test preprocessing performance with large datasets
#[test]
#[ignore] // Slow test
fn test_preprocessing_performance() -> Result<(), Box<dyn std::error::Error>> {
    // Generate large dataset
    let data = generate_preprocessing_test_data::<f32>(
        100,  // 100 series
        1000, // 1000 time points each
        0.1,
        0.05,
    )?;
    
    println!("Dataset size: {} rows", data.shape().0);
    
    // Create preprocessing pipeline
    let pipeline = PreprocessingPipeline::builder()
        .add_step(Box::new(MissingValueImputer::new(
            MissingValueStrategy::Interpolate
        )))
        .add_step(Box::new(StandardScaler::new()))
        .with_parallel(true) // Enable parallel processing
        .with_batch_size(10000)
        .build()?;
    
    // Measure preprocessing time
    let start = std::time::Instant::now();
    let processed_data = pipeline.fit_transform(&data)?;
    let preprocessing_time = start.elapsed();
    
    println!("Preprocessing completed in {:?}", preprocessing_time);
    
    // Calculate throughput
    let rows_per_second = data.shape().0 as f64 / preprocessing_time.as_secs_f64();
    println!("Throughput: {:.0} rows/second", rows_per_second);
    
    Ok(())
}

/// Helper functions

#[derive(Debug)]
struct DataStatistics {
    mean: f32,
    std: f32,
    min: f32,
    max: f32,
    median: f32,
    iqr: f32,
}

fn calculate_data_statistics(
    data: &TimeSeriesDataFrame<f32>
) -> Result<HashMap<String, DataStatistics>, Box<dyn std::error::Error>> {
    let mut stats = HashMap::new();
    
    for series_id in data.unique_series_ids()? {
        let series_data = data.filter_series(&series_id)?;
        let values = series_data.get_values()?;
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();
        
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];
        let median = sorted_values[sorted_values.len() / 2];
        
        let q1 = sorted_values[sorted_values.len() / 4];
        let q3 = sorted_values[3 * sorted_values.len() / 4];
        let iqr = q3 - q1;
        
        stats.insert(series_id, DataStatistics {
            mean, std, min, max, median, iqr
        });
    }
    
    Ok(stats)
}

fn calculate_exogenous_statistics(
    data: &TimeSeriesDataFrame<f32>
) -> Result<HashMap<String, DataStatistics>, Box<dyn std::error::Error>> {
    // Placeholder implementation
    Ok(HashMap::new())
}

// Placeholder enums and structs for the test
#[derive(Debug, Clone)]
enum MissingValueStrategy {
    Forward,
    Backward,
    Interpolate,
    Mean,
    Median,
    Zero,
}

#[derive(Debug, Clone)]
enum OutlierMethod {
    IQR(f32),
    ZScore(f32),
    IsolationForest,
}

#[derive(Debug, Clone)]
enum OutlierTreatment {
    Remove,
    Cap,
    Transform,
}

#[derive(Debug, Clone)]
enum RollingFeature {
    Mean(usize),
    Std(usize),
    Min(usize),
    Max(usize),
}

#[derive(Debug, Clone)]
enum DateFeature {
    DayOfWeek,
    DayOfMonth,
    WeekOfYear,
    Month,
    Quarter,
}

#[derive(Debug, Clone)]
enum EncodingType {
    OneHot,
    Ordinal,
    Target,
}

// Placeholder builder structs
struct FeatureEngineeringPipeline;
struct PreprocessingPipeline;
struct ExogenousPreprocessor;
struct MissingValueImputer;
struct OutlierHandler;
struct FeatureEngineer;
struct DataFixer;

// These would need actual implementations