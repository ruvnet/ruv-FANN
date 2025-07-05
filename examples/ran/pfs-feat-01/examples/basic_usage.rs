use anyhow::Result;
use chrono::{Duration, Utc};
use polars::prelude::*;
use std::path::Path;
use tempfile::TempDir;

use pfs_feat_01::config::*;
use pfs_feat_01::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("PFS-FEAT-01 Basic Usage Example");
    println!("================================");

    // Create sample data
    let sample_data = create_sample_ran_data()?;
    println!("Created sample RAN data with {} rows", sample_data.height());

    // Create temporary directories
    let temp_dir = TempDir::new()?;
    let input_path = temp_dir.path().join("input.parquet");
    let output_path = temp_dir.path().join("output.parquet");

    // Save sample data
    save_dataframe_to_parquet(&sample_data, &input_path)?;
    println!("Saved sample data to: {:?}", input_path);

    // Create feature engineering agent with RAN-specific configuration
    let config = create_ran_feature_config();
    let agent = FeatureEngineeringAgent::new(config.clone());
    println!("Created feature engineering agent");

    // Generate features
    println!("Generating features...");
    let result = agent.generate_features(
        "sample_cell_001",
        &input_path,
        &output_path,
        &config.default_features,
    ).await?;

    println!("Feature generation completed!");
    println!("Processing time: {}ms", result.stats.processing_time_ms);
    println!("Input rows: {}", result.stats.input_rows);
    println!("Output rows: {}", result.stats.output_rows);
    println!("Features generated: {}", result.stats.features_generated);
    println!("Memory usage: {}MB", result.stats.memory_usage_mb);

    // Load and display results
    let output_data = load_dataframe_from_parquet(&output_path)?;
    println!("\nOutput DataFrame schema:");
    for (name, dtype) in output_data.schema().iter() {
        println!("  {}: {:?}", name, dtype);
    }

    println!("\nGenerated features:");
    let input_columns: std::collections::HashSet<_> = sample_data.get_column_names().into_iter().collect();
    let output_columns: std::collections::HashSet<_> = output_data.get_column_names().into_iter().collect();
    let new_features: Vec<_> = output_columns.difference(&input_columns).collect();
    
    for feature in new_features {
        println!("  {}", feature);
    }

    // Demonstrate feature analysis
    analyze_generated_features(&output_data)?;

    Ok(())
}

/// Create sample RAN data
fn create_sample_ran_data() -> Result<DataFrame> {
    let num_rows = 1000;
    let start_time = Utc::now() - Duration::hours(num_rows as i64);
    
    let mut timestamps = Vec::new();
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
        let timestamp = start_time + Duration::hours(i as i64);
        timestamps.push(timestamp);

        // Generate realistic RAN metrics with daily patterns
        let hour = timestamp.hour() as f64;
        let day_factor = if timestamp.weekday().number_from_monday() <= 5 { 1.0 } else { 0.7 }; // Weekday vs weekend
        let hour_factor = 0.5 + 0.5 * (2.0 * std::f64::consts::PI * hour / 24.0).sin(); // Daily pattern

        // PRB utilization (0-100%)
        prb_utilization_dl.push(
            30.0 + 40.0 * hour_factor * day_factor + rng.gen_range(-10.0..10.0)
        );
        prb_utilization_ul.push(
            15.0 + 20.0 * hour_factor * day_factor + rng.gen_range(-5.0..5.0)
        );

        // Active users
        active_users.push(
            (50.0 + 100.0 * hour_factor * day_factor + rng.gen_range(-20.0..20.0)) as i32
        );

        // Throughput (Mbps)
        throughput_dl.push(
            100.0 + 200.0 * hour_factor * day_factor + rng.gen_range(-50.0..50.0)
        );
        throughput_ul.push(
            30.0 + 60.0 * hour_factor * day_factor + rng.gen_range(-15.0..15.0)
        );

        // Radio quality metrics
        rsrp_avg.push(-85.0 + rng.gen_range(-15.0..15.0)); // dBm
        sinr_avg.push(15.0 + rng.gen_range(-10.0..10.0));  // dB

        cell_ids.push("Cell_001".to_string());
    }

    let df = df! {
        "timestamp" => timestamps,
        "kpi_value" => prb_utilization_dl.clone(), // Use PRB DL as main KPI
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

/// Create RAN-specific feature configuration
fn create_ran_feature_config() -> FeatureEngineConfig {
    let mut config = FeatureEngineConfig::default();
    
    // Configure for RAN-specific patterns
    config.default_features = RanFeatureTemplates::ran_kpi_features();
    
    // Adjust processing settings
    config.processing.max_parallel_jobs = 4;
    config.processing.batch_size = 500;
    config.processing.memory_limit_mb = 1024;

    config
}

/// Save DataFrame to Parquet file
fn save_dataframe_to_parquet(df: &DataFrame, path: &Path) -> Result<()> {
    let mut writer = ParquetWriter::new(std::fs::File::create(path)?);
    writer.finish(df)?;
    Ok(())
}

/// Load DataFrame from Parquet file
fn load_dataframe_from_parquet(path: &Path) -> Result<DataFrame> {
    let df = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?
        .collect()?;
    Ok(df)
}

/// Analyze generated features
fn analyze_generated_features(df: &DataFrame) -> Result<()> {
    println!("\nFeature Analysis:");
    println!("================");

    // Display DataFrame info
    println!("DataFrame shape: {} rows × {} columns", df.height(), df.width());

    // Analyze lag features
    let lag_features: Vec<_> = df.get_column_names()
        .into_iter()
        .filter(|name| name.contains("_lag_"))
        .collect();
    
    if !lag_features.is_empty() {
        println!("\nLag Features ({}):", lag_features.len());
        for feature in lag_features {
            if let Ok(col) = df.column(feature) {
                let null_count = col.null_count();
                let null_pct = 100.0 * null_count as f64 / col.len() as f64;
                println!("  {}: {:.1}% null values", feature, null_pct);
            }
        }
    }

    // Analyze rolling window features
    let rolling_features: Vec<_> = df.get_column_names()
        .into_iter()
        .filter(|name| name.contains("_mean_") || name.contains("_std_") || 
                       name.contains("_min_") || name.contains("_max_"))
        .collect();
    
    if !rolling_features.is_empty() {
        println!("\nRolling Window Features ({}):", rolling_features.len());
        for feature in rolling_features.iter().take(5) { // Show first 5
            if let Ok(col) = df.column(feature) {
                let null_count = col.null_count();
                let null_pct = 100.0 * null_count as f64 / col.len() as f64;
                println!("  {}: {:.1}% null values", feature, null_pct);
            }
        }
        if rolling_features.len() > 5 {
            println!("  ... and {} more", rolling_features.len() - 5);
        }
    }

    // Analyze time-based features
    let time_features: Vec<_> = df.get_column_names()
        .into_iter()
        .filter(|name| name.contains("hour_of_day") || name.contains("day_of_week") || 
                       name.contains("is_weekend") || name.contains("is_business_hour"))
        .collect();
    
    if !time_features.is_empty() {
        println!("\nTime-based Features ({}):", time_features.len());
        for feature in time_features {
            if let Ok(col) = df.column(feature) {
                let unique_count = col.n_unique().unwrap_or(0);
                println!("  {}: {} unique values", feature, unique_count);
            }
        }
    }

    // Show sample of data
    println!("\nSample of generated features (first 5 rows):");
    let sample = df.head(Some(5));
    
    // Show only a subset of columns for readability
    let display_columns = df.get_column_names()
        .into_iter()
        .take(10)
        .collect::<Vec<_>>();
    
    if let Ok(sample_subset) = sample.select(display_columns) {
        println!("{}", sample_subset);
    }

    Ok(())
}

/// Demonstrate batch processing
#[allow(dead_code)]
async fn demonstrate_batch_processing() -> Result<()> {
    println!("\nBatch Processing Example:");
    println!("========================");

    let temp_dir = TempDir::new()?;
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");

    std::fs::create_dir_all(&input_dir)?;
    std::fs::create_dir_all(&output_dir)?;

    // Create multiple sample files
    let num_files = 5;
    let mut time_series_ids = Vec::new();

    for i in 0..num_files {
        let ts_id = format!("cell_{:03}", i);
        let sample_data = create_sample_ran_data()?;
        let input_path = input_dir.join(format!("{}.parquet", ts_id));
        
        save_dataframe_to_parquet(&sample_data, &input_path)?;
        time_series_ids.push(ts_id);
    }

    println!("Created {} sample files", num_files);

    // Create agent and process batch
    let config = create_ran_feature_config();
    let agent = FeatureEngineeringAgent::new(config.clone());

    let batch_result = agent.generate_batch_features(
        &time_series_ids,
        &input_dir,
        &output_dir,
        &config.default_features,
        4, // max parallel jobs
    ).await?;

    println!("Batch processing completed!");
    println!("Total processing time: {}ms", batch_result.batch_stats.total_processing_time_ms);
    println!("Successful series: {}", batch_result.batch_stats.successful_series);
    println!("Failed series: {}", batch_result.batch_stats.failed_series);
    println!("Total input rows: {}", batch_result.batch_stats.total_input_rows);
    println!("Total output rows: {}", batch_result.batch_stats.total_output_rows);
    println!("Peak memory usage: {}MB", batch_result.batch_stats.peak_memory_usage_mb);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sample_data() {
        let df = create_sample_ran_data().unwrap();
        assert_eq!(df.height(), 1000);
        assert!(df.width() >= 9); // At least 9 columns
        
        // Check required columns exist
        assert!(df.column("timestamp").is_ok());
        assert!(df.column("kpi_value").is_ok());
        assert!(df.column("prb_utilization_dl").is_ok());
    }

    #[test]
    fn test_ran_feature_config() {
        let config = create_ran_feature_config();
        assert!(config.default_features.lag_features.enabled);
        assert!(config.default_features.rolling_window.enabled);
        assert!(config.default_features.time_features.enabled);
        assert!(!config.default_features.lag_features.lag_periods.is_empty());
    }

    #[tokio::test]
    async fn test_save_and_load_parquet() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.parquet");
        
        let df = create_sample_ran_data().unwrap();
        save_dataframe_to_parquet(&df, &file_path).unwrap();
        
        assert!(file_path.exists());
        
        let loaded_df = load_dataframe_from_parquet(&file_path).unwrap();
        assert_eq!(df.height(), loaded_df.height());
        assert_eq!(df.width(), loaded_df.width());
    }
}