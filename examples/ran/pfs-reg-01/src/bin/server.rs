//! PFS-REG-01 Model Registry Server Binary

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::signal;
use tracing::{error, info, warn};

use pfs_reg_01::{
    init_tracing, RegistryConfig, ModelRegistryService, FilesystemStorage,
    grpc::ModelRegistryServer, SERVICE_NAME, VERSION,
};

#[derive(Parser)]
#[command(name = SERVICE_NAME)]
#[command(version = VERSION)]
#[command(about = "PFS-REG-01: Model Registry & Lifecycle Service for RAN Intelligence Platform")]
struct Args {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<std::path::PathBuf>,
    
    /// Server host
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    
    /// Server port
    #[arg(short, long, default_value = "50052")]
    port: u16,
    
    /// Database URL
    #[arg(long)]
    database_url: Option<String>,
    
    /// Storage path
    #[arg(long)]
    storage_path: Option<std::path::PathBuf>,
    
    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize tracing
    init_tracing(&args.log_level)?;
    
    info!("Starting {} v{}", SERVICE_NAME, VERSION);
    
    // Load configuration
    let mut config = if let Some(config_path) = args.config {
        info!("Loading configuration from: {:?}", config_path);
        RegistryConfig::from_file(config_path)
            .context("Failed to load configuration from file")?
    } else {
        info!("Loading configuration from environment");
        RegistryConfig::from_env()
            .context("Failed to load configuration from environment")?
    };
    
    // Override config with CLI arguments
    if !args.host.is_empty() {
        config.server.host = args.host;
    }
    if args.port != 0 {
        config.server.port = args.port;
    }
    if let Some(database_url) = args.database_url {
        config.database.url = database_url;
    }
    if let Some(storage_path) = args.storage_path {
        config.storage.base_path = storage_path;
    }
    
    // Validate configuration
    config.validate().context("Configuration validation failed")?;
    
    info!("Configuration loaded successfully");
    info!("Server will listen on: {}:{}", config.server.host, config.server.port);
    info!("Database URL: {}", config.database.url);
    info!("Storage path: {:?}", config.storage.base_path);
    
    // Initialize storage
    let storage = Arc::new(
        FilesystemStorage::new(&config.storage.base_path)
            .context("Failed to initialize storage")?
    );
    
    info!("Storage initialized successfully");
    
    // Initialize model registry service
    let registry_service = ModelRegistryService::new(
        &config.database.url,
        storage,
        config.clone(),
    ).await.context("Failed to initialize model registry service")?;
    
    info!("Model registry service initialized successfully");
    
    // Create gRPC server
    let grpc_server = ModelRegistryServer::new(registry_service, config.clone());
    
    // Parse server address
    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .context("Failed to parse server address")?;
    
    info!("Starting gRPC server on: {}", addr);
    
    // Start the server with graceful shutdown
    tokio::select! {
        result = grpc_server.serve(addr) => {
            match result {
                Ok(_) => info!("gRPC server stopped"),
                Err(e) => error!("gRPC server error: {}", e),
            }
        }
        _ = signal::ctrl_c() => {
            info!("Received shutdown signal");
        }
    }
    
    info!("Shutting down gracefully...");
    
    // TODO: Implement graceful shutdown
    // - Stop accepting new requests
    // - Wait for ongoing requests to complete
    // - Close database connections
    // - Cleanup resources
    
    info!("Shutdown complete");
    Ok(())
}

/// Setup signal handlers for graceful shutdown
async fn setup_signal_handlers() -> Result<()> {
    let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())?;
    let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())?;
    
    tokio::select! {
        _ = sigterm.recv() => {
            info!("Received SIGTERM");
        }
        _ = sigint.recv() => {
            info!("Received SIGINT");
        }
        _ = signal::ctrl_c() => {
            info!("Received Ctrl+C");
        }
    }
    
    Ok(())
}

/// Health check handler
async fn health_check() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // TODO: Implement comprehensive health checks
    // - Database connectivity
    // - Storage accessibility
    // - Service dependencies
    Ok(())
}