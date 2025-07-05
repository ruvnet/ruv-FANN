use pfs_core_01::neural_service::*;
use pfs_core_01::neural_service::neural_service_client::NeuralServiceClient;
use pfs_core_01::neural_service::neural_service_server::NeuralServiceServer;
use pfs_core_01::{NeuralServiceImpl, ModelManager, ServiceConfig};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tonic::transport::{Channel, Server};
use uuid::Uuid;

struct TestServer {
    _temp_dir: TempDir,
    client: NeuralServiceClient<Channel>,
    server_handle: tokio::task::JoinHandle<()>,
}

impl TestServer {
    async fn new() -> Self {
        let temp_dir = TempDir::new().unwrap();
        let models_dir = temp_dir.path().join("models");
        std::fs::create_dir_all(&models_dir).unwrap();

        let model_manager = Arc::new(ModelManager::new(models_dir, 10));
        model_manager.initialize().await.unwrap();

        let neural_service = NeuralServiceImpl::new(model_manager, 2);

        // Find an available port
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let server = Server::builder()
            .add_service(NeuralServiceServer::new(neural_service))
            .serve(addr);

        let server_handle = tokio::spawn(server);

        // Wait a bit for server to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let client = NeuralServiceClient::connect(format!("http://{}", addr))
            .await
            .unwrap();

        Self {
            _temp_dir: temp_dir,
            client,
            server_handle,
        }
    }

    async fn shutdown(self) {
        self.server_handle.abort();
    }
}

#[tokio::test]
async fn test_health_check() {
    let server = TestServer::new().await;

    let request = tonic::Request::new(HealthRequest {});
    let response = server.client.clone().health(request).await.unwrap();
    let health = response.into_inner();

    assert_eq!(health.status, "healthy");
    assert!(!health.version.is_empty());
    assert!(health.uptime_seconds >= 0.0);

    server.shutdown().await;
}

#[tokio::test]
async fn test_train_and_predict_xor() {
    let server = TestServer::new().await;
    let mut client = server.client.clone();

    // Create XOR training data
    let training_data = TrainingData {
        examples: vec![
            TrainingExample {
                inputs: vec![0.0, 0.0],
                outputs: vec![0.0],
            },
            TrainingExample {
                inputs: vec![0.0, 1.0],
                outputs: vec![1.0],
            },
            TrainingExample {
                inputs: vec![1.0, 0.0],
                outputs: vec![1.0],
            },
            TrainingExample {
                inputs: vec![1.0, 1.0],
                outputs: vec![0.0],
            },
        ],
        input_size: 2,
        output_size: 1,
    };

    let model_config = ModelConfig {
        name: "XOR Test".to_string(),
        description: "XOR function test".to_string(),
        layers: vec![2, 4, 1],
        activation: ActivationFunction::Sigmoid as i32,
        learning_rate: 0.1,
        max_epochs: 1000,
        desired_error: 0.01,
        training_algorithm: TrainingAlgorithm::Backpropagation as i32,
    };

    let training_config = TrainingConfig {
        batch_size: 4,
        shuffle: true,
        validation_split: 0.0,
        patience: 50,
        save_best: true,
    };

    // Train model
    let train_request = tonic::Request::new(TrainRequest {
        model_config: Some(model_config),
        training_data: Some(training_data),
        training_config: Some(training_config),
    });

    let train_response = client.train(train_request).await.unwrap();
    let train_result = train_response.into_inner();

    assert_eq!(train_result.status, "success");
    assert!(!train_result.model_id.is_empty());
    assert!(train_result.results.is_some());

    let model_id = train_result.model_id;
    let results = train_result.results.unwrap();
    assert!(results.epochs_completed > 0);
    assert!(results.training_time_seconds > 0.0);

    // Test predictions
    let test_cases = vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 1.0),
        (vec![1.0, 0.0], 1.0),
        (vec![1.0, 1.0], 0.0),
    ];

    for (input, expected) in test_cases {
        let predict_request = tonic::Request::new(PredictRequest {
            model_id: model_id.clone(),
            input_vector: input,
        });

        let predict_response = client.predict(predict_request).await.unwrap();
        let predict_result = predict_response.into_inner();

        assert_eq!(predict_result.status, "success");
        assert_eq!(predict_result.output_vector.len(), 1);
        
        let output = predict_result.output_vector[0];
        let error = (output - expected).abs();
        assert!(error < 0.2, "Prediction error too large: {} vs {}", output, expected);
    }

    server.shutdown().await;
}

#[tokio::test]
async fn test_model_management() {
    let server = TestServer::new().await;
    let mut client = server.client.clone();

    // Create a simple model
    let model_config = ModelConfig {
        name: "Test Model".to_string(),
        description: "A test model".to_string(),
        layers: vec![2, 3, 1],
        activation: ActivationFunction::Sigmoid as i32,
        learning_rate: 0.01,
        max_epochs: 100,
        desired_error: 0.1,
        training_algorithm: TrainingAlgorithm::Backpropagation as i32,
    };

    let training_data = TrainingData {
        examples: vec![
            TrainingExample {
                inputs: vec![1.0, 2.0],
                outputs: vec![0.5],
            },
        ],
        input_size: 2,
        output_size: 1,
    };

    // Train model
    let train_request = tonic::Request::new(TrainRequest {
        model_config: Some(model_config),
        training_data: Some(training_data),
        training_config: Some(TrainingConfig::default()),
    });

    let train_response = client.train(train_request).await.unwrap();
    let model_id = train_response.into_inner().model_id;

    // Get model info
    let info_request = tonic::Request::new(GetModelInfoRequest {
        model_id: model_id.clone(),
    });

    let info_response = client.get_model_info(info_request).await.unwrap();
    let info_result = info_response.into_inner();

    assert_eq!(info_result.status, "success");
    assert_eq!(info_result.model_id, model_id);
    assert!(info_result.config.is_some());
    assert!(info_result.metadata.is_some());

    let config = info_result.config.unwrap();
    assert_eq!(config.name, "Test Model");
    assert_eq!(config.layers, vec![2, 3, 1]);

    // List models
    let list_request = tonic::Request::new(ListModelsRequest {
        page: 0,
        page_size: 10,
        filter: String::new(),
    });

    let list_response = client.list_models(list_request).await.unwrap();
    let list_result = list_response.into_inner();

    assert_eq!(list_result.status, "success");
    assert!(list_result.total_count >= 1);
    assert!(!list_result.models.is_empty());

    // Delete model
    let delete_request = tonic::Request::new(DeleteModelRequest {
        model_id: model_id.clone(),
    });

    let delete_response = client.delete_model(delete_request).await.unwrap();
    let delete_result = delete_response.into_inner();

    assert_eq!(delete_result.status, "success");

    // Verify model is deleted
    let info_request = tonic::Request::new(GetModelInfoRequest {
        model_id: model_id.clone(),
    });

    let info_response = client.get_model_info(info_request).await;
    assert!(info_response.is_err());

    server.shutdown().await;
}

#[tokio::test]
async fn test_invalid_input_validation() {
    let server = TestServer::new().await;
    let mut client = server.client.clone();

    // Test invalid model config
    let invalid_config = ModelConfig {
        name: "".to_string(),
        description: "".to_string(),
        layers: vec![0], // Invalid: zero neurons
        activation: ActivationFunction::Sigmoid as i32,
        learning_rate: -1.0, // Invalid: negative learning rate
        max_epochs: 0, // Invalid: zero epochs
        desired_error: -1.0, // Invalid: negative error
        training_algorithm: TrainingAlgorithm::Backpropagation as i32,
    };

    let training_data = TrainingData {
        examples: vec![],
        input_size: 0,
        output_size: 0,
    };

    let train_request = tonic::Request::new(TrainRequest {
        model_config: Some(invalid_config),
        training_data: Some(training_data),
        training_config: Some(TrainingConfig::default()),
    });

    let train_response = client.train(train_request).await;
    assert!(train_response.is_err());

    // Test prediction with invalid model ID
    let predict_request = tonic::Request::new(PredictRequest {
        model_id: "nonexistent".to_string(),
        input_vector: vec![1.0, 2.0],
    });

    let predict_response = client.predict(predict_request).await;
    assert!(predict_response.is_err());

    server.shutdown().await;
}

#[tokio::test]
async fn test_concurrent_training() {
    let server = TestServer::new().await;
    let client = server.client.clone();

    let mut handles = Vec::new();

    // Start multiple training tasks concurrently
    for i in 0..3 {
        let mut client_clone = client.clone();
        let handle = tokio::spawn(async move {
            let model_config = ModelConfig {
                name: format!("Concurrent Model {}", i),
                description: format!("Concurrent training test {}", i),
                layers: vec![2, 3, 1],
                activation: ActivationFunction::Sigmoid as i32,
                learning_rate: 0.01,
                max_epochs: 50,
                desired_error: 0.1,
                training_algorithm: TrainingAlgorithm::Backpropagation as i32,
            };

            let training_data = TrainingData {
                examples: vec![
                    TrainingExample {
                        inputs: vec![1.0, 2.0],
                        outputs: vec![0.5],
                    },
                ],
                input_size: 2,
                output_size: 1,
            };

            let train_request = tonic::Request::new(TrainRequest {
                model_config: Some(model_config),
                training_data: Some(training_data),
                training_config: Some(TrainingConfig::default()),
            });

            client_clone.train(train_request).await
        });
        handles.push(handle);
    }

    // Wait for all training tasks to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.into_inner().status, "success");
    }

    server.shutdown().await;
}

#[tokio::test]
async fn test_large_model_training() {
    let server = TestServer::new().await;
    let mut client = server.client.clone();

    // Create a larger model
    let model_config = ModelConfig {
        name: "Large Model".to_string(),
        description: "Large model test".to_string(),
        layers: vec![10, 20, 15, 5],
        activation: ActivationFunction::Relu as i32,
        learning_rate: 0.001,
        max_epochs: 100,
        desired_error: 0.01,
        training_algorithm: TrainingAlgorithm::Rprop as i32,
    };

    // Generate synthetic training data
    let mut examples = Vec::new();
    for i in 0..100 {
        let inputs: Vec<f64> = (0..10).map(|j| (i + j) as f64 / 100.0).collect();
        let outputs: Vec<f64> = (0..5).map(|j| (i + j) as f64 / 200.0).collect();
        examples.push(TrainingExample { inputs, outputs });
    }

    let training_data = TrainingData {
        examples,
        input_size: 10,
        output_size: 5,
    };

    let train_request = tonic::Request::new(TrainRequest {
        model_config: Some(model_config),
        training_data: Some(training_data),
        training_config: Some(TrainingConfig {
            batch_size: 32,
            shuffle: true,
            validation_split: 0.2,
            patience: 10,
            save_best: true,
        }),
    });

    let train_response = client.train(train_request).await.unwrap();
    let train_result = train_response.into_inner();

    assert_eq!(train_result.status, "success");
    assert!(train_result.results.is_some());

    let model_id = train_result.model_id;
    let results = train_result.results.unwrap();
    assert!(results.epochs_completed > 0);
    assert!(results.training_time_seconds > 0.0);

    // Test prediction with the large model
    let predict_request = tonic::Request::new(PredictRequest {
        model_id,
        input_vector: (0..10).map(|i| i as f64 / 10.0).collect(),
    });

    let predict_response = client.predict(predict_request).await.unwrap();
    let predict_result = predict_response.into_inner();

    assert_eq!(predict_result.status, "success");
    assert_eq!(predict_result.output_vector.len(), 5);

    server.shutdown().await;
}