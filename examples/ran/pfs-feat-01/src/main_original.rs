use anyhow::{Context, Result};
use clap::{Arg, Command};
use std::path::Path;
use tokio::signal;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use pfs_feat_01::config::*;
use pfs_feat_01::grpc_service::*;
use pfs_feat_01::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let matches = Command::new("pfs-feat-01")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Time-series feature generation agent for RAN Intelligence Platform")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
                .default_value("config.toml"),
        )
        .arg(
            Arg::new("host")
                .long("host")
                .value_name("HOST")
                .help("gRPC server host")
                .default_value("127.0.0.1"),
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .value_name("PORT")
                .help("gRPC server port")
                .default_value("50051"),
        )
        .arg(
            Arg::new("generate-config")
                .long("generate-config")
                .help("Generate a default configuration file and exit"),
        )
        .arg(
            Arg::new("validate-config")
                .long("validate-config")
                .help("Validate configuration and exit"),
        )
        .arg(
            Arg::new("log-level")
                .long("log-level")
                .value_name("LEVEL")
                .help("Log level (trace, debug, info, warn, error)")
                .default_value("info"),
        )
        .arg(
            Arg::new("one-shot")
                .long("one-shot")
                .help("Run one-shot feature generation instead of starting server")
                .requires_all(&["input", "output"]),
        )
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .value_name("PATH")
                .help("Input file or directory path (for one-shot mode)"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("PATH")
                .help("Output file or directory path (for one-shot mode)"),
        )
        .arg(
            Arg::new("time-series-id")
                .long("time-series-id")
                .value_name("ID")
                .help("Time series ID (for one-shot mode)")
                .default_value("default"),
        )
        .get_matches();

    // Initialize logging
    let log_level = matches.get_one::<String>("log-level").unwrap();
    init_logging(log_level)?;

    info!("Starting PFS-FEAT-01 Feature Engineering Agent v{}", env!("CARGO_PKG_VERSION"));

    // Handle configuration generation
    if matches.get_flag("generate-config") {
        return generate_config_file(matches.get_one::<String>("config").unwrap());
    }

    // Load configuration
    let config_path = matches.get_one::<String>("config").unwrap();
    let mut config = load_configuration(config_path)?;

    // Override config with command line arguments
    if let Some(host) = matches.get_one::<String>("host") {
        config.service.host = host.clone();
    }
    if let Some(port) = matches.get_one::<String>("port") {
        config.service.port = port.parse()
            .context("Invalid port number")?;
    }

    // Validate configuration
    config.validate()
        .map_err(|e| anyhow::anyhow!("Configuration validation failed: {}", e))?;

    if matches.get_flag("validate-config") {
        info!("Configuration is valid");
        return Ok(());
    }

    // Handle one-shot mode
    if matches.get_flag("one-shot") {
        return run_one_shot_mode(&config, &matches).await;
    }

    // Start gRPC server
    run_grpc_server(config).await
}

/// Initialize logging
fn init_logging(log_level: &str) -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(log_level));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(env_filter)
        .init();

    Ok(())
}

/// Generate default configuration file
fn generate_config_file(config_path: &str) -> Result<()> {
    let config = FeatureEngineConfig::default();
    config.to_file(config_path)
        .context("Failed to write configuration file")?;
    
    info!("Generated default configuration file: {}", config_path);
    Ok(())
}

/// Load configuration from file
fn load_configuration(config_path: &str) -> Result<FeatureEngineConfig> {
    if !Path::new(config_path).exists() {
        warn!("Configuration file not found, using default configuration");
        return Ok(FeatureEngineConfig::default());
    }

    FeatureEngineConfig::from_file(config_path)
        .context("Failed to load configuration file")
}

/// Run one-shot feature generation mode
async fn run_one_shot_mode(
    config: &FeatureEngineConfig,
    matches: &clap::ArgMatches,
) -> Result<()> {
    let input_path = matches.get_one::<String>("input").unwrap();
    let output_path = matches.get_one::<String>("output").unwrap();
    let time_series_id = matches.get_one::<String>("time-series-id").unwrap();

    info!("Running one-shot feature generation");
    info!("Input: {}", input_path);
    info!("Output: {}", output_path);
    info!("Time Series ID: {}", time_series_id);

    // Create feature engineering agent
    let agent = FeatureEngineeringAgent::new(config.clone());

    // Generate features
    let result = agent.generate_features(
        time_series_id,
        Path::new(input_path),
        Path::new(output_path),
        &config.default_features,
    ).await
    .context("Feature generation failed")?;

    info!("Feature generation completed successfully");
    info!("Processing time: {}ms", result.stats.processing_time_ms);
    info!("Input rows: {}", result.stats.input_rows);
    info!("Output rows: {}", result.stats.output_rows);
    info!("Features generated: {}", result.stats.features_generated);
    info!("Memory usage: {}MB", result.stats.memory_usage_mb);
    info!("Generated features: {:?}", result.generated_features);

    Ok(())
}

/// Run gRPC server
async fn run_grpc_server(config: FeatureEngineConfig) -> Result<()> {
    info!("Starting gRPC server on {}:{}", config.service.host, config.service.port);

    // Create and start the server
    let server = FeatureEngineeringServer::new(config)
        .context("Failed to create gRPC server")?;

    // Handle shutdown gracefully
    let server_task = tokio::spawn(async move {
        if let Err(e) = server.serve().await {
            error!("gRPC server error: {}", e);
        }
    });

    // Wait for shutdown signal
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("Received shutdown signal, stopping server...");
        }
        Err(err) => {
            error!("Unable to listen for shutdown signal: {}", err);
        }
    }

    // Gracefully shutdown the server
    server_task.abort();
    info!("Server stopped");

    Ok(())
}

/// Feature engineering agent CLI commands
pub mod cli {
    use super::*;
    use std::collections::HashMap;

    /// Validate a single time-series file
    pub async fn validate_single_file(
        config: &FeatureEngineConfig,
        file_path: &str,
    ) -> Result<()> {
        info!("Validating single file: {}", file_path);

        let validator = FeatureValidator::new(config.clone());
        
        // For validation, we need both input and output, so we'll use the file as output
        // and create a minimal input for comparison
        let df = polars::prelude::LazyFrame::scan_parquet(
            file_path, 
            polars::prelude::ScanArgsParquet::default()
        )?
        .collect()?;

        // Create a minimal input DataFrame for comparison
        let input_df = df.select([
            polars::prelude::col("timestamp"),
            polars::prelude::col("kpi_value"),
        ])?;

        let result = validator.validate_features(
            &input_df,
            &df,
            &["timestamp".to_string(), "kpi_value".to_string()],
        ).await?;

        if result.is_valid {
            info!("✓ File validation passed");
        } else {
            error!("✗ File validation failed");
            for error in &result.errors {
                error!("  Error: {}", error);
            }
        }

        for warning in &result.warnings {
            warn!("  Warning: {}", warning);
        }

        info!("Validation stats:");
        info!("  Total checks: {}", result.validation_stats.total_checks);
        info!("  Passed checks: {}", result.validation_stats.passed_checks);
        info!("  Failed checks: {}", result.validation_stats.failed_checks);
        info!("  Total errors: {}", result.validation_stats.total_errors);
        info!("  Total warnings: {}", result.validation_stats.total_warnings);

        Ok(())
    }

    /// Generate sample test data
    pub async fn generate_sample_data(
        output_path: &str,
        series_count: usize,
        rows_per_series: usize,
    ) -> Result<()> {
        use chrono::{DateTime, Duration, Utc};
        use polars::prelude::*;
        use rand::Rng;

        info!("Generating {} sample time-series with {} rows each", series_count, rows_per_series);

        std::fs::create_dir_all(output_path)
            .context("Failed to create output directory")?;

        let mut rng = rand::thread_rng();

        for series_id in 0..series_count {
            let mut timestamps = Vec::new();
            let mut kpi_values = Vec::new();
            let mut cell_ids = Vec::new();

            let start_time = Utc::now() - Duration::hours(rows_per_series as i64);

            for i in 0..rows_per_series {
                timestamps.push(start_time + Duration::hours(i as i64));
                kpi_values.push(rng.gen_range(0.0..100.0));
                cell_ids.push(format!("Cell_{}", series_id));
            }

            let df = df! {
                "timestamp" => timestamps,
                "kpi_value" => kpi_values,
                "cell_id" => cell_ids,
            }?;

            let file_path = format!("{}/series_{}.parquet", output_path, series_id);
            let mut writer = polars::prelude::ParquetWriter::new(
                std::fs::File::create(&file_path)?
            );
            writer.finish(&mut df.clone())?;

            if series_id % 100 == 0 {
                info!("Generated {} series", series_id + 1);
            }
        }

        info!("✓ Generated {} sample time-series files", series_count);
        Ok(())
    }

    /// Benchmark feature generation performance
    pub async fn benchmark_performance(
        config: &FeatureEngineConfig,
        input_directory: &str,
        iterations: usize,
    ) -> Result<()> {
        info!("Benchmarking feature generation performance");

        let agent = FeatureEngineeringAgent::new(config.clone());
        let mut processing_times = Vec::new();

        // List input files
        let entries = std::fs::read_dir(input_directory)
            .context("Failed to read input directory")?;

        let mut input_files = Vec::new();
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "parquet") {
                input_files.push(path);
            }
        }

        if input_files.is_empty() {
            return Err(anyhow::anyhow!("No parquet files found in input directory"));
        }

        info!("Found {} input files", input_files.len());

        for iteration in 0..iterations {
            let start_time = std::time::Instant::now();

            for (i, input_file) in input_files.iter().enumerate() {
                let time_series_id = format!("benchmark_{}_{}_{}", iteration, i, 
                    input_file.file_stem().unwrap().to_str().unwrap());
                let output_file = std::env::temp_dir().join(format!("{}.parquet", time_series_id));

                let _ = agent.generate_features(
                    &time_series_id,
                    input_file,
                    &output_file,
                    &config.default_features,
                ).await?;

                // Clean up temporary output file
                let _ = std::fs::remove_file(output_file);
            }

            let iteration_time = start_time.elapsed();
            processing_times.push(iteration_time.as_millis() as f64);

            info!("Iteration {}: {}ms", iteration + 1, iteration_time.as_millis());
        }

        // Calculate statistics
        let total_time: f64 = processing_times.iter().sum();
        let avg_time = total_time / processing_times.len() as f64;
        let min_time = processing_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_time = processing_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        info!("Benchmark Results:");
        info!("  Iterations: {}", iterations);
        info!("  Files per iteration: {}", input_files.len());
        info!("  Average time per iteration: {:.2}ms", avg_time);
        info!("  Min time: {:.2}ms", min_time);
        info!("  Max time: {:.2}ms", max_time);
        info!("  Total time: {:.2}ms", total_time);
        info!("  Average time per file: {:.2}ms", avg_time / input_files.len() as f64);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_generate_config() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");
        
        let result = generate_config_file(config_path.to_str().unwrap());
        assert!(result.is_ok());
        assert!(config_path.exists());
    }

    #[tokio::test]
    async fn test_load_configuration() {
        // Test loading default config when file doesn't exist
        let result = load_configuration("non_existent_config.toml");
        assert!(result.is_ok());
        
        // Test loading valid config file
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");
        
        let config = FeatureEngineConfig::default();
        config.to_file(config_path.to_str().unwrap()).unwrap();
        
        let loaded_config = load_configuration(config_path.to_str().unwrap());
        assert!(loaded_config.is_ok());
    }

    #[tokio::test]
    async fn test_cli_generate_sample_data() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().to_str().unwrap();
        
        let result = cli::generate_sample_data(output_path, 5, 100).await;
        assert!(result.is_ok());
        
        // Check that files were created
        let entries = std::fs::read_dir(output_path).unwrap();
        let file_count = entries.count();
        assert_eq!(file_count, 5);
    }
}