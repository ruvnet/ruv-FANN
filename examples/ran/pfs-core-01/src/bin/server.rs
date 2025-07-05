use pfs_core_01::neural_service::neural_service_server::NeuralServiceServer;
use pfs_core_01::{NeuralServiceImpl, ModelManager, ServiceConfig};
use std::path::PathBuf;
use std::sync::Arc;
use tonic::transport::Server;
use tracing::{info, error};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load configuration
    let config = match std::env::var("PFS_CONFIG_FILE") {
        Ok(config_file) => ServiceConfig::from_file(&config_file)?,
        Err(_) => ServiceConfig::default(),
    };

    // Validate configuration
    if let Err(e) = config.validate() {
        error!("Configuration validation failed: {}", e);
        std::process::exit(1);
    }

    info!("Starting PFS-CORE-01 Neural Service...");
    info!("Server configuration: {}:{}", config.server.host, config.server.port);
    info!("Models directory: {:?}", config.storage.models_directory);
    info!("Max concurrent training: {}", config.training.max_concurrent_training);

    // Create model manager
    let model_manager = Arc::new(ModelManager::new(
        config.storage.models_directory.clone(),
        config.storage.max_models,
    ));

    // Initialize model manager
    if let Err(e) = model_manager.initialize().await {
        error!("Failed to initialize model manager: {}", e);
        std::process::exit(1);
    }

    // Create neural service
    let neural_service = NeuralServiceImpl::new(
        model_manager.clone(),
        config.training.max_concurrent_training,
    );

    // Create server address
    let addr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .unwrap();

    info!("Neural Service listening on {}", addr);

    // Start server
    Server::builder()
        .add_service(NeuralServiceServer::new(neural_service))
        .serve(addr)
        .await?;

    Ok(())
}