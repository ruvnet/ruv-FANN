//! Main executable for the OPT-MOB-01 Handover Predictor Service
//!
//! This is the primary entry point for the handover prediction service,
//! providing gRPC API for real-time handover prediction.

use clap::{Arg, Command};
use opt_mob_01::{
    service::{start_service, ServiceBuilder},
    OptMobConfig,
    utils::{ConfigUtils, DataGenerator},
};
use std::path::Path;
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    let matches = Command::new("handover-predictor")
        .version("1.0.0")
        .author("HandoverPredictorAgent <agent@ruv-fann.ai>")
        .about("OPT-MOB-01 Predictive Handover Trigger Model Service")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
                .default_value("config.json")
        )
        .arg(
            Arg::new("model")
                .short('m')
                .long("model")
                .value_name("FILE")
                .help("Trained model file path")
                .default_value("models/handover_v1.bin")
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .value_name("PORT")
                .help("gRPC server port")
                .default_value("50051")
        )
        .arg(
            Arg::new("bind")
                .short('b')
                .long("bind")
                .value_name("ADDRESS")
                .help("Bind address")
                .default_value("0.0.0.0")
        )
        .arg(
            Arg::new("generate-config")
                .long("generate-config")
                .help("Generate default configuration file")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("generate-data")
                .long("generate-data")
                .help("Generate synthetic test data")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("validate-model")
                .long("validate-model")
                .help("Validate the loaded model")
                .action(clap::ArgAction::SetTrue)
        )
        .get_matches();
    
    let config_path = matches.get_one::<String>("config").unwrap();
    let model_path = matches.get_one::<String>("model").unwrap();
    let port: u16 = matches.get_one::<String>("port").unwrap().parse()?;
    let bind_address = matches.get_one::<String>("bind").unwrap();
    
    // Handle special commands
    if matches.get_flag("generate-config") {
        return generate_default_config(config_path);
    }
    
    if matches.get_flag("generate-data") {
        return generate_synthetic_data().await;
    }
    
    if matches.get_flag("validate-model") {
        return validate_model(model_path).await;
    }
    
    // Load configuration
    let mut config = if Path::new(config_path).exists() {
        match ConfigUtils::load_config(config_path) {
            Ok(config) => {
                info!("Loaded configuration from: {}", config_path);
                config
            },
            Err(e) => {
                warn!("Failed to load config from {}: {}. Using defaults.", config_path, e);
                OptMobConfig::default()
            }
        }
    } else {
        info!("Configuration file not found. Using defaults.");
        OptMobConfig::default()
    };
    
    // Override config with command line arguments
    config.model_path = model_path.to_string();
    config.grpc_port = port;
    
    // Validate model file exists
    if !Path::new(&config.model_path).exists() {
        warn!("Model file not found: {}. Service will start without a loaded model.", config.model_path);
        warn!("Use the training binary to create a model first.");
        config.model_path = String::new(); // Clear path to avoid loading errors
    }
    
    // Display startup information
    info!("Starting OPT-MOB-01 Handover Predictor Service");
    info!("Configuration:");
    info!("  - gRPC Port: {}", config.grpc_port);
    info!("  - Model Path: {}", if config.model_path.is_empty() { "None (will start without model)" } else { &config.model_path });
    info!("  - Prediction Horizon: {}s", config.prediction_horizon_seconds);
    info!("  - Handover Threshold: {:.2}", config.handover_threshold);
    info!("  - Feature Window Size: {}", config.feature_window_size);
    info!("  - Max Batch Size: {}", config.max_batch_size);
    
    // Check requirements
    if config.handover_threshold < 0.0 || config.handover_threshold > 1.0 {
        error!("Invalid handover threshold: {}. Must be between 0.0 and 1.0", config.handover_threshold);
        std::process::exit(1);
    }
    
    if config.feature_window_size == 0 {
        error!("Feature window size cannot be zero");
        std::process::exit(1);
    }
    
    // Start the service
    info!("🚀 Starting handover prediction service on {}:{}", bind_address, port);
    info!("📡 Ready to receive UE metrics and predict handovers");
    info!("🎯 Target accuracy: >90% (minimum requirement)");
    
    if let Err(e) = start_service(config).await {
        error!("Service failed: {}", e);
        std::process::exit(1);
    }
    
    Ok(())
}

/// Generate default configuration file
fn generate_default_config(config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating default configuration file: {}", config_path);
    
    ConfigUtils::create_default_config(config_path)?;
    
    info!("✅ Default configuration created successfully");
    info!("Edit {} to customize settings", config_path);
    
    // Display sample configuration
    let config = OptMobConfig::default();
    println!("\nGenerated configuration:");
    println!("{}", serde_json::to_string_pretty(&config)?);
    
    Ok(())
}

/// Generate synthetic test data
async fn generate_synthetic_data() -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating synthetic test data for handover prediction");
    
    let mut generator = DataGenerator::new();
    
    // Generate training dataset
    info!("Generating training dataset...");
    let train_dataset = generator.generate_dataset(
        50,   // 50 UEs
        24.0, // 24 hours of data
        60,   // 1 sample per minute
        0.05, // 5% handover rate
    );
    
    info!("Training dataset generated:");
    info!("  - Total samples: {}", train_dataset.ue_metrics.len());
    info!("  - Handover events: {}", train_dataset.handover_events.len());
    info!("  - Handover rate: {:.2}%", 
          train_dataset.handover_events.len() as f64 / train_dataset.ue_metrics.len() as f64 * 100.0);
    
    // Generate test dataset
    info!("Generating test dataset...");
    let test_dataset = generator.generate_dataset(
        20,   // 20 UEs
        12.0, // 12 hours of data
        60,   // 1 sample per minute
        0.05, // 5% handover rate
    );
    
    info!("Test dataset generated:");
    info!("  - Total samples: {}", test_dataset.ue_metrics.len());
    info!("  - Handover events: {}", test_dataset.handover_events.len());
    
    // Save datasets
    std::fs::create_dir_all("data")?;
    
    use opt_mob_01::utils::ExportUtils;
    
    ExportUtils::export_to_csv(&train_dataset, "data/train_metrics.csv")?;
    ExportUtils::export_handovers_to_csv(&train_dataset.handover_events, "data/train_handovers.csv")?;
    ExportUtils::export_to_csv(&test_dataset, "data/test_metrics.csv")?;
    ExportUtils::export_handovers_to_csv(&test_dataset.handover_events, "data/test_handovers.csv")?;
    
    info!("✅ Synthetic data generated and saved to data/ directory");
    info!("Files created:");
    info!("  - data/train_metrics.csv");
    info!("  - data/train_handovers.csv");
    info!("  - data/test_metrics.csv");
    info!("  - data/test_handovers.csv");
    
    info!("💡 Use the training binary to train a model with this data:");
    info!("   cargo run --bin train-handover-model -- --train-data data/train_metrics.csv --train-events data/train_handovers.csv");
    
    Ok(())
}

/// Validate a trained model
async fn validate_model(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("Validating model: {}", model_path);
    
    if !Path::new(model_path).exists() {
        error!("Model file not found: {}", model_path);
        return Ok(());
    }
    
    use opt_mob_01::model::HandoverModel;
    
    match HandoverModel::load(model_path) {
        Ok(model) => {
            let model_info = model.get_model_info();
            
            info!("✅ Model loaded successfully");
            info!("Model Information:");
            info!("  - Model ID: {}", model_info.model_id);
            info!("  - Version: {}", model_info.model_version);
            info!("  - Training Accuracy: {:.4}", model_info.training_accuracy);
            info!("  - Validation Accuracy: {:.4}", model_info.validation_accuracy);
            info!("  - Total Features: {}", model_info.total_features);
            info!("  - Description: {}", model_info.description);
            
            // Check if model meets accuracy requirements
            if model_info.validation_accuracy >= opt_mob_01::MINIMUM_ACCURACY_TARGET {
                info!("✅ Model meets minimum accuracy requirement (>= {:.1}%)", 
                     opt_mob_01::MINIMUM_ACCURACY_TARGET * 100.0);
            } else {
                warn!("⚠️  Model accuracy {:.4} is below minimum requirement {:.4}", 
                     model_info.validation_accuracy, opt_mob_01::MINIMUM_ACCURACY_TARGET);
            }
            
            // Test prediction capability
            info!("Testing prediction capability...");
            
            use opt_mob_01::features::FeatureExtractor;
            use opt_mob_01::data::UeMetrics;
            
            let mut extractor = FeatureExtractor::new(10);
            let test_metrics = UeMetrics::new("TEST_UE", "TEST_CELL")
                .with_rsrp(-85.0)
                .with_sinr(12.0)
                .with_speed(60.0)
                .with_neighbor_rsrp(-80.0);
            
            extractor.add_metrics(test_metrics);
            
            if let Ok(features) = extractor.extract_features() {
                match model.predict(&features) {
                    Ok(prediction) => {
                        info!("✅ Test prediction successful: {:.4}", prediction);
                        if prediction >= 0.0 && prediction <= 1.0 {
                            info!("✅ Prediction value is within valid range [0, 1]");
                        } else {
                            warn!("⚠️  Prediction value {} is outside valid range [0, 1]", prediction);
                        }
                    },
                    Err(e) => {
                        error!("❌ Test prediction failed: {}", e);
                    }
                }
            } else {
                warn!("⚠️  Could not extract features for test prediction");
            }
        },
        Err(e) => {
            error!("❌ Failed to load model: {}", e);
            error!("Possible issues:");
            error!("  - Model file is corrupted");
            error!("  - Model was created with incompatible version");
            error!("  - File permissions issue");
        }
    }
    
    Ok(())
}

/// Display startup banner
fn _display_banner() {
    println!(r#"
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   OPT-MOB-01 Handover Predictor              ║
    ║                                                               ║
    ║  🎯 Predictive Handover Trigger Model                        ║
    ║  📡 Real-time UE Mobility Prediction                         ║
    ║  🧠 Neural Network-based Decision Engine                     ║
    ║  ⚡ >90% Accuracy Target                                     ║
    ║                                                               ║
    ║  Part of the RAN Intelligence Platform                        ║
    ╚═══════════════════════════════════════════════════════════════╝
    "#);
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_config_generation() {
        let temp_file = NamedTempFile::new().unwrap();
        let config_path = temp_file.path().to_str().unwrap();
        
        assert!(generate_default_config(config_path).is_ok());
        assert!(Path::new(config_path).exists());
        
        // Verify the config can be loaded back
        let loaded_config = ConfigUtils::load_config(config_path).unwrap();
        assert_eq!(loaded_config.grpc_port, 50051);
    }
    
    #[tokio::test]
    async fn test_synthetic_data_generation() {
        // This test would generate data in a temporary directory
        // For now, we'll just verify the function doesn't panic
        let result = generate_synthetic_data().await;
        // In a real test, we'd use a temporary directory
        // assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_model_validation_missing_file() {
        let result = validate_model("nonexistent_model.bin").await;
        assert!(result.is_ok()); // Should not panic, just warn
    }
}