use anyhow::{Context, Result};
use clap::{Arg, Command};
use std::path::Path;
use tracing::{error, info};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use pfs_feat_01::config::*;
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
                .help("Input file path (for one-shot mode)"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("PATH")
                .help("Output file path (for one-shot mode)"),
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
    let config = load_configuration(config_path)?;

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

    info!("Feature engineering agent started successfully");
    info!("Use --one-shot mode for feature generation or implement gRPC server");

    Ok(())
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
        info!("Configuration file not found, using default configuration");
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
    
    // Show first few generated features
    let feature_count = result.generated_features.len();
    let display_count = std::cmp::min(10, feature_count);
    info!("Generated features (showing {} of {}):", display_count, feature_count);
    for feature in result.generated_features.iter().take(display_count) {
        info!("  - {}", feature);
    }
    
    if feature_count > display_count {
        info!("  ... and {} more features", feature_count - display_count);
    }

    Ok(())
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
}