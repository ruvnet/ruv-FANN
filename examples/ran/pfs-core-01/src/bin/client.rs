use pfs_core_01::neural_service::neural_service_client::NeuralServiceClient;
use pfs_core_01::neural_service::*;
use clap::{Arg, Command};
use std::error::Error;
use tonic::transport::Channel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let matches = Command::new("PFS-CORE-01 Client")
        .version("1.0")
        .author("RAN Intelligence Platform Team")
        .about("Client for PFS-CORE-01 Neural Service")
        .arg(
            Arg::new("server")
                .short('s')
                .long("server")
                .value_name("ADDRESS")
                .help("Server address (e.g., http://127.0.0.1:50051)")
                .default_value("http://127.0.0.1:50051"),
        )
        .subcommand(
            Command::new("health")
                .about("Check service health")
        )
        .subcommand(
            Command::new("train")
                .about("Train a neural network")
                .arg(
                    Arg::new("name")
                        .short('n')
                        .long("name")
                        .value_name("NAME")
                        .help("Model name")
                        .required(true),
                )
                .arg(
                    Arg::new("layers")
                        .short('l')
                        .long("layers")
                        .value_name("LAYERS")
                        .help("Layer sizes (e.g., 2,4,1)")
                        .required(true),
                )
                .arg(
                    Arg::new("learning-rate")
                        .short('r')
                        .long("learning-rate")
                        .value_name("RATE")
                        .help("Learning rate")
                        .default_value("0.01"),
                )
                .arg(
                    Arg::new("epochs")
                        .short('e')
                        .long("epochs")
                        .value_name("EPOCHS")
                        .help("Maximum epochs")
                        .default_value("10000"),
                )
                .arg(
                    Arg::new("data-file")
                        .short('f')
                        .long("data-file")
                        .value_name("FILE")
                        .help("Training data file (JSON format)")
                        .required(true),
                )
        )
        .subcommand(
            Command::new("predict")
                .about("Make prediction")
                .arg(
                    Arg::new("model-id")
                        .short('m')
                        .long("model-id")
                        .value_name("ID")
                        .help("Model ID")
                        .required(true),
                )
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .value_name("INPUT")
                        .help("Input vector (e.g., 1.0,2.0)")
                        .required(true),
                )
        )
        .subcommand(
            Command::new("list")
                .about("List models")
                .arg(
                    Arg::new("page")
                        .short('p')
                        .long("page")
                        .value_name("PAGE")
                        .help("Page number")
                        .default_value("0"),
                )
                .arg(
                    Arg::new("size")
                        .short('s')
                        .long("size")
                        .value_name("SIZE")
                        .help("Page size")
                        .default_value("10"),
                )
        )
        .subcommand(
            Command::new("info")
                .about("Get model information")
                .arg(
                    Arg::new("model-id")
                        .short('m')
                        .long("model-id")
                        .value_name("ID")
                        .help("Model ID")
                        .required(true),
                )
        )
        .subcommand(
            Command::new("delete")
                .about("Delete model")
                .arg(
                    Arg::new("model-id")
                        .short('m')
                        .long("model-id")
                        .value_name("ID")
                        .help("Model ID")
                        .required(true),
                )
        )
        .get_matches();

    let server_addr = matches.get_one::<String>("server").unwrap();
    let mut client = NeuralServiceClient::connect(server_addr.clone()).await?;

    match matches.subcommand() {
        Some(("health", _)) => {
            let request = tonic::Request::new(HealthRequest {});
            let response = client.health(request).await?;
            let health = response.into_inner();
            println!("Service Status: {}", health.status);
            println!("Version: {}", health.version);
            println!("Active Models: {}", health.active_models);
            println!("Uptime: {:.2} seconds", health.uptime_seconds);
        }
        Some(("train", sub_matches)) => {
            let name = sub_matches.get_one::<String>("name").unwrap();
            let layers_str = sub_matches.get_one::<String>("layers").unwrap();
            let learning_rate: f64 = sub_matches.get_one::<String>("learning-rate").unwrap().parse()?;
            let epochs: u32 = sub_matches.get_one::<String>("epochs").unwrap().parse()?;
            let data_file = sub_matches.get_one::<String>("data-file").unwrap();

            // Parse layers
            let layers: Vec<u32> = layers_str
                .split(',')
                .map(|s| s.trim().parse())
                .collect::<Result<Vec<_>, _>>()?;

            // Load training data from file
            let training_data = load_training_data(data_file)?;

            let model_config = ModelConfig {
                name: name.clone(),
                description: format!("Model trained with {} layers", layers.len()),
                layers,
                activation: ActivationFunction::Sigmoid as i32,
                learning_rate,
                max_epochs: epochs,
                desired_error: 0.001,
                training_algorithm: TrainingAlgorithm::Backpropagation as i32,
            };

            let request = tonic::Request::new(TrainRequest {
                model_config: Some(model_config),
                training_data: Some(training_data),
                training_config: Some(TrainingConfig::default()),
            });

            println!("Starting training...");
            let response = client.train(request).await?;
            let train_response = response.into_inner();

            println!("Training Status: {}", train_response.status);
            println!("Model ID: {}", train_response.model_id);
            println!("Message: {}", train_response.message);

            if let Some(results) = train_response.results {
                println!("Training Results:");
                println!("  Epochs: {}", results.epochs_completed);
                println!("  Final Error: {:.6}", results.final_error);
                println!("  Training Time: {:.2} seconds", results.training_time_seconds);
                println!("  Validation Error: {:.6}", results.validation_error);
            }
        }
        Some(("predict", sub_matches)) => {
            let model_id = sub_matches.get_one::<String>("model-id").unwrap();
            let input_str = sub_matches.get_one::<String>("input").unwrap();

            let input_vector: Vec<f64> = input_str
                .split(',')
                .map(|s| s.trim().parse())
                .collect::<Result<Vec<_>, _>>()?;

            let request = tonic::Request::new(PredictRequest {
                model_id: model_id.clone(),
                input_vector,
            });

            let response = client.predict(request).await?;
            let predict_response = response.into_inner();

            println!("Prediction Status: {}", predict_response.status);
            println!("Output: {:?}", predict_response.output_vector);
            println!("Confidence: {:.4}", predict_response.confidence);
        }
        Some(("list", sub_matches)) => {
            let page: u32 = sub_matches.get_one::<String>("page").unwrap().parse()?;
            let page_size: u32 = sub_matches.get_one::<String>("size").unwrap().parse()?;

            let request = tonic::Request::new(ListModelsRequest {
                page,
                page_size,
                filter: String::new(),
            });

            let response = client.list_models(request).await?;
            let list_response = response.into_inner();

            println!("Models (Total: {}):", list_response.total_count);
            for model in list_response.models {
                println!("  ID: {}", model.model_id);
                println!("  Name: {}", model.name);
                println!("  Status: {}", model.status);
                println!("  Created: {}", model.created_at);
                println!("  Size: {} bytes", model.size_bytes);
                println!("  ---");
            }
        }
        Some(("info", sub_matches)) => {
            let model_id = sub_matches.get_one::<String>("model-id").unwrap();

            let request = tonic::Request::new(GetModelInfoRequest {
                model_id: model_id.clone(),
            });

            let response = client.get_model_info(request).await?;
            let info_response = response.into_inner();

            println!("Model Information:");
            println!("  ID: {}", info_response.model_id);
            
            if let Some(config) = info_response.config {
                println!("  Name: {}", config.name);
                println!("  Description: {}", config.description);
                println!("  Layers: {:?}", config.layers);
                println!("  Learning Rate: {}", config.learning_rate);
                println!("  Max Epochs: {}", config.max_epochs);
            }
            
            if let Some(metadata) = info_response.metadata {
                println!("  Created: {}", metadata.created_at);
                println!("  Updated: {}", metadata.updated_at);
                println!("  Size: {} bytes", metadata.size_bytes);
                println!("  Parameters: {}", metadata.total_parameters);
                println!("  Version: {}", metadata.version);
            }
        }
        Some(("delete", sub_matches)) => {
            let model_id = sub_matches.get_one::<String>("model-id").unwrap();

            let request = tonic::Request::new(DeleteModelRequest {
                model_id: model_id.clone(),
            });

            let response = client.delete_model(request).await?;
            let delete_response = response.into_inner();

            println!("Delete Status: {}", delete_response.status);
            println!("Message: {}", delete_response.message);
        }
        _ => {
            println!("No subcommand specified. Use --help for usage information.");
        }
    }

    Ok(())
}

fn load_training_data(file_path: &str) -> Result<TrainingData, Box<dyn Error>> {
    let content = std::fs::read_to_string(file_path)?;
    let data: serde_json::Value = serde_json::from_str(&content)?;
    
    let examples = data["examples"].as_array()
        .ok_or("Invalid training data format: missing 'examples' array")?;
    
    let mut training_examples = Vec::new();
    
    for example in examples {
        let inputs = example["inputs"].as_array()
            .ok_or("Invalid training data format: missing 'inputs' array")?
            .iter()
            .map(|v| v.as_f64().ok_or("Invalid input value"))
            .collect::<Result<Vec<_>, _>>()?;
        
        let outputs = example["outputs"].as_array()
            .ok_or("Invalid training data format: missing 'outputs' array")?
            .iter()
            .map(|v| v.as_f64().ok_or("Invalid output value"))
            .collect::<Result<Vec<_>, _>>()?;
        
        training_examples.push(TrainingExample { inputs, outputs });
    }
    
    if training_examples.is_empty() {
        return Err("Training data cannot be empty".into());
    }
    
    let input_size = training_examples[0].inputs.len() as u32;
    let output_size = training_examples[0].outputs.len() as u32;
    
    Ok(TrainingData {
        examples: training_examples,
        input_size,
        output_size,
    })
}