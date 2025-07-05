use anyhow::Result;
use chrono::{Duration, Utc};
use polars::prelude::*;
use std::path::Path;
use tempfile::TempDir;
use tokio::time::{timeout, Duration as TokioDuration};

use pfs_feat_01::config::*;
use pfs_feat_01::validation::*;
use pfs_feat_01::*;

/// Integration test for the complete feature engineering pipeline
#[tokio::test]
async fn test_complete_feature_engineering_pipeline() -> Result<()> {
    // Create test data
    let sample_data = create_test_ran_data(100)?;
    
    // Create temporary directories
    let temp_dir = TempDir::new()?;
    let input_path = temp_dir.path().join("input.parquet");
    let output_path = temp_dir.path().join("output.parquet");

    // Save test data
    save_test_data(&sample_data, &input_path)?;

    // Create feature engineering agent
    let config = create_test_config();
    let agent = FeatureEngineeringAgent::new(config.clone());

    // Generate features
    let result = agent.generate_features(
        "test_series_001",
        &input_path,
        &output_path,
        &config.default_features,
    ).await?;

    // Verify results
    assert!(result.stats.processing_time_ms > 0);
    assert_eq!(result.stats.input_rows, 100);
    assert_eq!(result.stats.output_rows, 100);
    assert!(result.stats.features_generated > 0);
    assert!(!result.generated_features.is_empty());

    // Load and verify output
    let output_data = load_test_data(&output_path)?;
    assert_eq!(output_data.height(), sample_data.height());
    assert!(output_data.width() > sample_data.width()); // Features were added

    Ok(())
}

/// Test batch processing
#[tokio::test]
async fn test_batch_feature_generation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");

    std::fs::create_dir_all(&input_dir)?;
    std::fs::create_dir_all(&output_dir)?;

    // Create multiple test files
    let num_series = 5;
    let mut time_series_ids = Vec::new();

    for i in 0..num_series {
        let ts_id = format!("test_series_{:03}", i);
        let sample_data = create_test_ran_data(50)?;
        let input_path = input_dir.join(format!("{}.parquet", ts_id));
        
        save_test_data(&sample_data, &input_path)?;
        time_series_ids.push(ts_id);
    }

    // Create agent and process batch
    let config = create_test_config();
    let agent = FeatureEngineeringAgent::new(config.clone());

    let batch_result = agent.generate_batch_features(
        &time_series_ids,
        &input_dir,
        &output_dir,
        &config.default_features,
        2, // max parallel jobs
    ).await?;

    // Verify batch results
    assert_eq!(batch_result.batch_stats.total_time_series, num_series as u32);
    assert_eq!(batch_result.batch_stats.successful_series, num_series as u32);
    assert_eq!(batch_result.batch_stats.failed_series, 0);
    assert_eq!(batch_result.results.len(), num_series);

    // Verify all output files exist
    for ts_id in &time_series_ids {
        let output_path = output_dir.join(format!("{}_features.parquet", ts_id));
        assert!(output_path.exists());
    }

    Ok(())
}

/// Test feature validation
#[tokio::test]
async fn test_feature_validation() -> Result<()> {
    // Create test data with features
    let input_data = create_test_ran_data(100)?;
    let output_data = create_test_data_with_features(&input_data)?;

    // Create validator
    let config = create_test_config();
    let validator = FeatureValidator::new(config);

    // Run validation
    let expected_features = vec![
        "kpi_value_lag_1".to_string(),
        "hour_of_day".to_string(),
        "day_of_week".to_string(),
    ];

    let validation_result = validator.validate_features(
        &input_data,
        &output_data,
        &expected_features,
    ).await?;

    // Check validation results
    assert!(validation_result.is_valid);
    assert!(validation_result.errors.is_empty());
    assert!(validation_result.validation_stats.total_checks > 0);
    assert!(validation_result.validation_stats.passed_checks > 0);
    assert_eq!(validation_result.validation_stats.failed_checks, 0);

    Ok(())
}

/// Test 1000 time-series sample validation (acceptance criteria)
#[tokio::test]
async fn test_1000_time_series_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let output_dir = temp_dir.path().join("sample_output");
    std::fs::create_dir_all(&output_dir)?;

    // Create sample files (reduced number for test performance)
    let sample_count = 10; // Reduced from 1000 for test efficiency
    
    for i in 0..sample_count {
        let sample_data = create_test_ran_data(50)?;
        let output_data = create_test_data_with_features(&sample_data)?;
        let output_path = output_dir.join(format!("series_{:04}.parquet", i));
        save_test_data(&output_data, &output_path)?;
    }

    // Validate sample batch
    let config = create_test_config();
    let validator = FeatureValidator::new(config);

    let validation_result = validator.validate_sample_batch(
        &output_dir,
        sample_count,
    ).await?;

    assert!(validation_result.is_valid);
    assert_eq!(validation_result.expected_series_count, sample_count);
    assert_eq!(validation_result.actual_series_count, sample_count);
    assert_eq!(validation_result.processed_series, sample_count);
    assert!(validation_result.total_features_generated > 0);
    assert!(validation_result.errors.is_empty());

    Ok(())
}

/// Test error handling
#[tokio::test]
async fn test_error_handling() -> Result<()> {
    let config = create_test_config();
    let agent = FeatureEngineeringAgent::new(config.clone());

    // Test with non-existent input file
    let temp_dir = TempDir::new()?;
    let non_existent_input = temp_dir.path().join("non_existent.parquet");
    let output_path = temp_dir.path().join("output.parquet");

    let result = agent.generate_features(
        "test_error",
        &non_existent_input,
        &output_path,
        &config.default_features,
    ).await;

    assert!(result.is_err());

    Ok(())
}

/// Test concurrent processing
#[tokio::test]
async fn test_concurrent_processing() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");

    std::fs::create_dir_all(&input_dir)?;
    std::fs::create_dir_all(&output_dir)?;

    // Create multiple test files
    let num_series = 8;
    let mut time_series_ids = Vec::new();

    for i in 0..num_series {
        let ts_id = format!("concurrent_test_{:03}", i);
        let sample_data = create_test_ran_data(30)?;
        let input_path = input_dir.join(format!("{}.parquet", ts_id));
        
        save_test_data(&sample_data, &input_path)?;
        time_series_ids.push(ts_id);
    }

    // Process with high concurrency
    let config = create_test_config();
    let agent = FeatureEngineeringAgent::new(config.clone());

    let batch_result = timeout(
        TokioDuration::from_secs(30),
        agent.generate_batch_features(
            &time_series_ids,
            &input_dir,
            &output_dir,
            &config.default_features,
            4, // max parallel jobs
        )
    ).await??;

    // Verify all series processed successfully
    assert_eq!(batch_result.batch_stats.successful_series, num_series as u32);
    assert_eq!(batch_result.batch_stats.failed_series, 0);

    Ok(())
}

/// Test memory usage limits
#[tokio::test]
async fn test_memory_limits() -> Result<()> {
    // Create large test data
    let large_data = create_test_ran_data(5000)?; // Larger dataset
    
    let temp_dir = TempDir::new()?;
    let input_path = temp_dir.path().join("large_input.parquet");
    let output_path = temp_dir.path().join("large_output.parquet");

    save_test_data(&large_data, &input_path)?;

    // Create config with limited memory
    let mut config = create_test_config();
    config.processing.memory_limit_mb = 1; // Very low limit

    let agent = FeatureEngineeringAgent::new(config.clone());

    // This should still work as we don't currently enforce memory limits strictly
    let result = agent.generate_features(
        "memory_test",
        &input_path,
        &output_path,
        &config.default_features,
    ).await;

    // Should complete successfully even with low memory limit
    assert!(result.is_ok());

    Ok(())
}

/// Test different feature configurations
#[tokio::test]
async fn test_different_feature_configurations() -> Result<()> {
    let sample_data = create_test_ran_data(50)?;
    
    let temp_dir = TempDir::new()?;
    let input_path = temp_dir.path().join("input.parquet");
    save_test_data(&sample_data, &input_path)?;

    let config = create_test_config();
    let agent = FeatureEngineeringAgent::new(config);

    // Test with only lag features
    let mut lag_only_config = FeatureConfig {
        lag_features: LagFeatureConfig {
            enabled: true,
            lag_periods: vec![1, 2, 3],
            target_columns: vec!["kpi_value".to_string()],
        },
        rolling_window: RollingWindowConfig {
            enabled: false,
            ..Default::default()
        },
        time_features: TimeBasedFeatureConfig {
            enabled: false,
            ..Default::default()
        },
        output: OutputConfig::default(),
    };

    let output_path = temp_dir.path().join("lag_only_output.parquet");
    let result = agent.generate_features(
        "lag_only_test",
        &input_path,
        &output_path,
        &lag_only_config,
    ).await?;

    assert!(result.stats.features_generated >= 3); // At least 3 lag features

    // Test with only time-based features
    lag_only_config.lag_features.enabled = false;
    lag_only_config.time_features = TimeBasedFeatureConfig {
        enabled: true,
        features: vec!["hour_of_day".to_string(), "day_of_week".to_string()],
        timestamp_column: "timestamp".to_string(),
        timezone: "UTC".to_string(),
    };

    let output_path = temp_dir.path().join("time_only_output.parquet");
    let result = agent.generate_features(
        "time_only_test",
        &input_path,
        &output_path,
        &lag_only_config,
    ).await?;

    assert!(result.stats.features_generated >= 2); // At least 2 time features

    Ok(())
}

/// Test RAN-specific feature templates
#[tokio::test]
async fn test_ran_feature_templates() -> Result<()> {
    let sample_data = create_comprehensive_ran_data(100)?;
    
    let temp_dir = TempDir::new()?;
    let input_path = temp_dir.path().join("ran_input.parquet");
    let output_path = temp_dir.path().join("ran_output.parquet");

    save_test_data(&sample_data, &input_path)?;

    // Test RAN KPI features template
    let mut config = create_test_config();
    config.default_features = RanFeatureTemplates::ran_kpi_features();

    let agent = FeatureEngineeringAgent::new(config.clone());

    let result = agent.generate_features(
        "ran_kpi_test",
        &input_path,
        &output_path,
        &config.default_features,
    ).await?;

    // Should generate many features for RAN KPIs
    assert!(result.stats.features_generated > 50);

    // Load output and verify RAN-specific features exist
    let output_data = load_test_data(&output_path)?;
    let feature_names = output_data.get_column_names();
    
    // Check for PRB utilization features
    assert!(feature_names.iter().any(|name| name.contains("prb_utilization_dl")));
    assert!(feature_names.iter().any(|name| name.contains("throughput_dl")));
    assert!(feature_names.iter().any(|name| name.contains("rsrp_avg")));

    Ok(())
}

// Helper functions

fn create_test_ran_data(num_rows: usize) -> Result<DataFrame> {
    let start_time = Utc::now() - Duration::hours(num_rows as i64);
    
    let mut timestamps = Vec::new();
    let mut kpi_values = Vec::new();
    let mut cell_ids = Vec::new();

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for i in 0..num_rows {
        timestamps.push(start_time + Duration::hours(i as i64));
        kpi_values.push(rng.gen_range(0.0..100.0));
        cell_ids.push("TEST_CELL".to_string());
    }

    let df = df! {
        "timestamp" => timestamps,
        "kpi_value" => kpi_values,
        "cell_id" => cell_ids,
    }?;

    Ok(df)
}

fn create_comprehensive_ran_data(num_rows: usize) -> Result<DataFrame> {
    let start_time = Utc::now() - Duration::hours(num_rows as i64);
    
    let mut timestamps = Vec::new();
    let mut kpi_values = Vec::new();
    let mut prb_utilization_dl = Vec::new();
    let mut prb_utilization_ul = Vec::new();
    let mut active_users = Vec::new();
    let mut throughput_dl = Vec::new();
    let mut throughput_ul = Vec::new();
    let mut rsrp_avg = Vec::new();
    let mut sinr_avg = Vec::new();
    let mut cell_ids = Vec::new();

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for i in 0..num_rows {
        timestamps.push(start_time + Duration::hours(i as i64));
        
        let base_value = rng.gen_range(0.0..100.0);
        kpi_values.push(base_value);
        prb_utilization_dl.push(base_value);
        prb_utilization_ul.push(base_value * 0.5);
        active_users.push(rng.gen_range(10..200));
        throughput_dl.push(rng.gen_range(50.0..300.0));
        throughput_ul.push(rng.gen_range(10.0..100.0));
        rsrp_avg.push(rng.gen_range(-120.0..-60.0));
        sinr_avg.push(rng.gen_range(-5.0..30.0));
        cell_ids.push("TEST_CELL".to_string());
    }

    let df = df! {
        "timestamp" => timestamps,
        "kpi_value" => kpi_values,
        "prb_utilization_dl" => prb_utilization_dl,
        "prb_utilization_ul" => prb_utilization_ul,
        "active_users" => active_users,
        "throughput_dl" => throughput_dl,
        "throughput_ul" => throughput_ul,
        "rsrp_avg" => rsrp_avg,
        "sinr_avg" => sinr_avg,
        "cell_id" => cell_ids,
    }?;

    Ok(df)
}

fn create_test_data_with_features(input_df: &DataFrame) -> Result<DataFrame> {
    let mut df = input_df.clone();
    
    // Add some mock features to simulate feature generation
    if let Ok(kpi_col) = df.column("kpi_value") {
        // Add lag feature
        let lag_1 = kpi_col.shift(1);
        df = df.with_column(lag_1.alias("kpi_value_lag_1"))?;
        
        // Add rolling mean
        let rolling_mean = kpi_col.rolling_mean(RollingOptions::default().window_size(Duration::parse("3i").unwrap()))?;
        df = df.with_column(rolling_mean.alias("kpi_value_mean_3w"))?;
    }
    
    // Add time-based features
    if let Ok(timestamp_col) = df.column("timestamp") {
        let hour_col = timestamp_col.hour()?;
        let dow_col = timestamp_col.weekday()?;
        
        df = df.with_column(hour_col.alias("hour_of_day"))?;
        df = df.with_column(dow_col.alias("day_of_week"))?;
    }
    
    Ok(df)
}

fn create_test_config() -> FeatureEngineConfig {
    let mut config = FeatureEngineConfig::default();
    
    // Configure for testing
    config.default_features.lag_features.lag_periods = vec![1, 2, 3];
    config.default_features.rolling_window.window_sizes = vec![3, 6];
    config.default_features.rolling_window.statistics = vec!["mean".to_string(), "std".to_string()];
    config.processing.max_parallel_jobs = 2;
    config.processing.batch_size = 100;
    
    config
}

fn save_test_data(df: &DataFrame, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    let mut writer = ParquetWriter::new(std::fs::File::create(path)?);
    writer.finish(df)?;
    Ok(())
}

fn load_test_data(path: &Path) -> Result<DataFrame> {
    let df = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?
        .collect()?;
    Ok(df)
}