//! PFS-DATA-01: File-based Data Ingestion Service for RAN Intelligence Platform
//!
//! This service provides high-performance batch data ingestion capabilities for processing
//! CSV and JSON files into normalized Parquet format with standardized schema for RAN
//! intelligence applications.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;

use clap::{Parser, Subcommand};
use tonic::transport::Server;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use pfs_data_01::config::IngestionConfig;
use pfs_data_01::proto::data_ingestion_service_server::DataIngestionServiceServer;
use pfs_data_01::service::DataIngestionServiceImpl;
use pfs_data_01::{IngestionEngine, IngestionResult};

#[derive(Parser)]
#[command(
    name = "pfs-data-01",
    version = "0.1.0",
    about = "RAN Intelligence Platform - File-based Data Ingestion Service",
    long_about = "PFS-DATA-01 provides batch data ingestion capabilities for processing CSV and JSON files into normalized Parquet format with standardized schema for RAN intelligence applications."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,
    
    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, default_value = "info")]
    log_level: String,
    
    /// Enable JSON logging
    #[arg(long)]
    json_logs: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the gRPC server
    Serve {
        /// Server bind address
        #[arg(short, long, default_value = "0.0.0.0:50051")]
        address: String,
        
        /// Enable reflection service
        #[arg(long)]
        reflection: bool,
        
        /// Enable health check service
        #[arg(long)]
        health_check: bool,
    },
    
    /// Process files directly (one-time processing)
    Process {
        /// Input directory containing files to process
        #[arg(short, long)]
        input_dir: PathBuf,
        
        /// Output directory for processed Parquet files
        #[arg(short, long)]
        output_dir: PathBuf,
        
        /// Process files recursively
        #[arg(short, long)]
        recursive: bool,
        
        /// File patterns to match (e.g., "*.csv,*.json")
        #[arg(short, long)]
        patterns: Option<String>,
        
        /// Show detailed progress
        #[arg(short, long)]
        verbose: bool,
    },
    
    /// Validate configuration
    ValidateConfig {
        /// Configuration file to validate
        config_file: PathBuf,
    },
    
    /// Test file processing with sample data
    Test {
        /// Number of test files to generate
        #[arg(short, long, default_value = "5")]
        files: usize,
        
        /// Number of rows per test file
        #[arg(short, long, default_value = "1000")]
        rows: usize,
        
        /// Output directory for test files
        #[arg(short, long, default_value = "test_data")]
        output_dir: PathBuf,
    },
    
    /// Show service information
    Info,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    
    // Initialize logging
    if let Err(e) = init_logging(&cli.log_level, cli.json_logs) {
        eprintln!("Failed to initialize logging: {}", e);
        process::exit(1);
    }
    
    info!("Starting PFS-DATA-01 File-based Data Ingestion Service");
    
    // Load configuration
    let config = match load_config(cli.config).await {
        Ok(config) => config,
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            process::exit(1);
        }
    };
    
    // Execute command
    let result = match cli.command {
        Commands::Serve { address, reflection, health_check } => {
            serve_grpc(config, &address, reflection, health_check).await
        }
        Commands::Process { input_dir, output_dir, recursive, patterns, verbose } => {
            process_files(config, input_dir, output_dir, recursive, patterns, verbose).await
        }
        Commands::ValidateConfig { config_file } => {
            validate_config_file(config_file).await
        }
        Commands::Test { files, rows, output_dir } => {
            run_test(config, files, rows, output_dir).await
        }
        Commands::Info => {
            show_info();
            Ok(())
        }
    };
    
    if let Err(e) = result {
        error!("Command failed: {}", e);
        process::exit(1);
    }
    
    info!("PFS-DATA-01 completed successfully");
}

/// Initialize logging based on configuration
fn init_logging(log_level: &str, json_logs: bool) -> IngestionResult<()> {
    let level = tracing::Level::from_str(log_level)
        .map_err(|_| pfs_data_01::error::IngestionError::config(
            format!("invalid log level: {}", log_level)
        ))?;
    
    let registry = tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(format!("pfs_data_01={}", level)));
    
    if json_logs {
        registry
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        registry
            .with(tracing_subscriber::fmt::layer().pretty())
            .init();
    }
    
    Ok(())
}

/// Load configuration from file or use defaults
async fn load_config(config_path: Option<PathBuf>) -> IngestionResult<IngestionConfig> {
    match config_path {
        Some(path) => {
            info!("Loading configuration from: {:?}", path);
            IngestionConfig::from_file(path)
        }
        None => {
            info!("Using default configuration");
            Ok(IngestionConfig::default())
        }
    }
}

/// Start the gRPC server
async fn serve_grpc(
    config: IngestionConfig,
    address: &str,
    enable_reflection: bool,
    enable_health_check: bool,
) -> IngestionResult<()> {
    let addr = SocketAddr::from_str(address)
        .map_err(|e| pfs_data_01::error::IngestionError::config(
            format!("invalid server address '{}': {}", address, e)
        ))?;
    
    info!("Starting gRPC server on: {}", addr);
    
    // Create service
    let ingestion_service = DataIngestionServiceImpl::new(config);
    let service = DataIngestionServiceServer::new(ingestion_service);
    
    // Build server
    let mut server_builder = Server::builder();
    
    // Add reflection service if enabled
    #[cfg(feature = "reflection")]
    if enable_reflection {
        let reflection_service = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(pfs_data_01::proto::FILE_DESCRIPTOR_SET)
            .build()
            .map_err(|e| pfs_data_01::error::IngestionError::service(
                format!("failed to create reflection service: {}", e)
            ))?;
        server_builder = server_builder.add_service(reflection_service);
        info!("gRPC reflection service enabled");
    }
    
    // Add health check service if enabled
    #[cfg(feature = "health")]
    if enable_health_check {
        let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
        health_reporter
            .set_serving::<DataIngestionServiceServer<DataIngestionServiceImpl>>()
            .await;
        server_builder = server_builder.add_service(health_service);
        info!("gRPC health check service enabled");
    }
    
    // Start server
    info!("PFS-DATA-01 gRPC server listening on {}", addr);
    server_builder
        .add_service(service)
        .serve(addr)
        .await
        .map_err(|e| pfs_data_01::error::IngestionError::service(
            format!("gRPC server error: {}", e)
        ))?;
    
    Ok(())
}

/// Process files directly
async fn process_files(
    mut config: IngestionConfig,
    input_dir: PathBuf,
    output_dir: PathBuf,
    recursive: bool,
    patterns: Option<String>,
    verbose: bool,
) -> IngestionResult<()> {
    info!("Processing files: {:?} -> {:?}", input_dir, output_dir);
    
    // Update configuration
    config.recursive = recursive;
    
    if let Some(patterns_str) = patterns {
        config.file_patterns = patterns_str
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        info!("Using file patterns: {:?}", config.file_patterns);
    }
    
    if verbose {
        config.log_level = "debug".to_string();
    }
    
    // Validate configuration
    config.validate()?;
    
    // Create ingestion engine
    let engine = IngestionEngine::new(config);
    
    // Process directory
    let start_time = std::time::Instant::now();
    let result = engine.process_directory(&input_dir, &output_dir).await?;
    let processing_time = start_time.elapsed();
    
    // Print results
    println!("\n✅ Processing completed successfully!");
    println!("📊 Summary:");
    println!("  • Files processed: {}", result.files_processed);
    println!("  • Files failed: {}", result.files_failed);
    println!("  • Rows processed: {}", result.rows_processed);
    println!("  • Rows failed: {}", result.rows_failed);
    println!("  • Input size: {:.2} MB", result.input_size_bytes as f64 / 1024.0 / 1024.0);
    println!("  • Output size: {:.2} MB", result.output_size_bytes as f64 / 1024.0 / 1024.0);
    println!("  • Processing time: {:.2}s", processing_time.as_secs_f64());
    println!("  • Throughput: {:.2} MB/s", result.throughput_mb_per_second());
    println!("  • Error rate: {:.4}%", result.error_rate() * 100.0);
    println!("  • Compression ratio: {:.2}x", 1.0 / result.compression_ratio());
    
    if result.error_rate() > 0.01 {
        warn!("Error rate ({:.4}%) exceeds recommended threshold (1%)", result.error_rate() * 100.0);
    }
    
    Ok(())
}

/// Validate configuration file
async fn validate_config_file(config_file: PathBuf) -> IngestionResult<()> {
    info!("Validating configuration file: {:?}", config_file);
    
    let config = IngestionConfig::from_file(config_file)?;
    config.validate()?;
    
    println!("✅ Configuration is valid!");
    println!("📋 Configuration summary:");
    println!("  • File patterns: {:?}", config.file_patterns);
    println!("  • Batch size: {}", config.batch_size);
    println!("  • Max concurrent files: {}", config.max_concurrent_files);
    println!("  • Max error rate: {:.2}%", config.max_error_rate * 100.0);
    println!("  • Compression: {}", config.compression_codec);
    println!("  • Row group size: {}", config.row_group_size);
    
    Ok(())
}

/// Run test with sample data
async fn run_test(
    config: IngestionConfig,
    num_files: usize,
    rows_per_file: usize,
    output_dir: PathBuf,
) -> IngestionResult<()> {
    info!("Running test with {} files, {} rows each", num_files, rows_per_file);
    
    // Create test data directory
    let test_data_dir = output_dir.join("input");
    let processed_dir = output_dir.join("output");
    
    tokio::fs::create_dir_all(&test_data_dir).await?;
    tokio::fs::create_dir_all(&processed_dir).await?;
    
    // Generate test files
    println!("📝 Generating {} test files...", num_files);
    for i in 0..num_files {
        let file_path = test_data_dir.join(format!("test_data_{:03}.csv", i));
        generate_test_csv_file(&file_path, rows_per_file).await?;
    }
    
    println!("🔄 Processing test files...");
    
    // Process test data
    let engine = IngestionEngine::new(config);
    let start_time = std::time::Instant::now();
    let result = engine.process_directory(&test_data_dir, &processed_dir).await?;
    let processing_time = start_time.elapsed();
    
    // Validate results
    let expected_total_rows = num_files * rows_per_file;
    let success = result.files_processed == num_files as u64 
        && result.rows_processed >= (expected_total_rows as f64 * 0.99) as u64; // Allow 1% tolerance
    
    // Print test results
    println!("\n🧪 Test Results:");
    println!("  • Status: {}", if success { "✅ PASSED" } else { "❌ FAILED" });
    println!("  • Expected files: {}", num_files);
    println!("  • Processed files: {}", result.files_processed);
    println!("  • Expected rows: ~{}", expected_total_rows);
    println!("  • Processed rows: {}", result.rows_processed);
    println!("  • Error rate: {:.4}%", result.error_rate() * 100.0);
    println!("  • Processing time: {:.2}s", processing_time.as_secs_f64());
    println!("  • Throughput: {:.2} MB/s", result.throughput_mb_per_second());
    
    if !success {
        return Err(pfs_data_01::error::IngestionError::service("test failed"));
    }
    
    println!("\n✅ All tests passed!");
    Ok(())
}

/// Generate a test CSV file
async fn generate_test_csv_file(file_path: &PathBuf, num_rows: usize) -> IngestionResult<()> {
    use std::io::Write;
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;
    
    let mut file = File::create(file_path).await?;
    
    // Write header
    file.write_all(b"timestamp,cell_id,kpi_name,kpi_value,ue_id\n").await?;
    
    // Write data rows
    for i in 0..num_rows {
        let timestamp = chrono::Utc::now() + chrono::Duration::seconds(i as i64);
        let cell_id = format!("cell_{:04}", i % 100);
        let kpi_name = match i % 4 {
            0 => "throughput_dl",
            1 => "throughput_ul", 
            2 => "prb_utilization_dl",
            _ => "active_users",
        };
        let kpi_value = (i as f64 * 1.5 + 10.0) % 100.0;
        let ue_id = format!("ue_{:06}", i % 1000);
        
        let line = format!(
            "{},{},{},{:.2},{}\n",
            timestamp.format("%Y-%m-%d %H:%M:%S%.3f"),
            cell_id,
            kpi_name,
            kpi_value,
            ue_id
        );
        
        file.write_all(line.as_bytes()).await?;
    }
    
    file.flush().await?;
    Ok(())
}

/// Show service information
fn show_info() {
    println!("📡 PFS-DATA-01: File-based Data Ingestion Service");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🎯 Purpose: Batch data ingestion for RAN Intelligence Platform");
    println!("📋 Version: {}", env!("CARGO_PKG_VERSION"));
    println!("🏗️  Built with: Rust + gRPC + Apache Arrow + Parquet");
    println!();
    println!("✨ Features:");
    println!("  • High-performance CSV/JSON to Parquet conversion");
    println!("  • Schema validation and normalization");
    println!("  • Configurable error handling (<0.01% target error rate)");
    println!("  • Concurrent file processing");
    println!("  • Real-time monitoring and metrics");
    println!("  • gRPC API for integration");
    println!("  • Directory watching for automatic processing");
    println!();
    println!("🎯 Performance Targets:");
    println!("  • Throughput: 100GB+ batch processing capability");
    println!("  • Error Rate: <0.01% parsing errors");
    println!("  • Concurrency: Configurable parallel file processing");
    println!("  • Compression: Optimized Parquet output");
    println!();
    println!("📚 Usage:");
    println!("  pfs-data-01 serve                    # Start gRPC server");
    println!("  pfs-data-01 process -i <dir> -o <dir>   # Process files directly");
    println!("  pfs-data-01 test                        # Run test with sample data");
    println!("  pfs-data-01 validate-config <file>      # Validate configuration");
    println!();
    println!("🔗 Part of the RAN Intelligence Platform (ruv-FANN)");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_config_loading() {
        // Test default config
        let config = load_config(None).await.unwrap();
        assert!(config.validate().is_ok());
        
        // Test that we can validate the default config
        assert!(config.batch_size > 0);
        assert!(config.max_concurrent_files > 0);
        assert!(config.max_error_rate >= 0.0 && config.max_error_rate <= 1.0);
    }
    
    #[tokio::test]
    async fn test_generate_test_file() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("test.csv");
        
        generate_test_csv_file(&test_file, 100).await.unwrap();
        
        let content = tokio::fs::read_to_string(&test_file).await.unwrap();
        let lines: Vec<&str> = content.lines().collect();
        
        assert_eq!(lines.len(), 101); // 100 data rows + 1 header
        assert!(lines[0].contains("timestamp,cell_id,kpi_name,kpi_value,ue_id"));
    }
    
    #[test]
    fn test_init_logging() {
        // Test valid log levels
        assert!(init_logging("info", false).is_ok());
        assert!(init_logging("debug", true).is_ok());
        
        // Test invalid log level
        assert!(init_logging("invalid", false).is_err());
    }
}