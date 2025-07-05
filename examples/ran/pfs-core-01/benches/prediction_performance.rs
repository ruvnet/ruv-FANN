use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use pfs_core_01::neural_service::*;
use pfs_core_01::{NeuralServiceImpl, ModelManager};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::runtime::Runtime;

struct BenchmarkSetup {
    service: NeuralServiceImpl,
    model_ids: Vec<String>,
    _temp_dir: TempDir,
}

impl BenchmarkSetup {
    async fn new() -> Self {
        let temp_dir = TempDir::new().unwrap();
        let model_manager = Arc::new(ModelManager::new(temp_dir.path().join("models"), 100));
        model_manager.initialize().await.unwrap();
        
        let service = NeuralServiceImpl::new(model_manager, 4);
        
        // Create models for different sizes
        let model_configs = vec![
            (vec![5, 5], "small"),
            (vec![10, 20, 10], "medium"),
            (vec![20, 50, 30, 10], "large"),
            (vec![50, 100, 50, 20], "xlarge"),
        ];
        
        let mut model_ids = Vec::new();
        
        for (layers, name) in model_configs {
            let model_config = ModelConfig {
                name: format!("Benchmark Model {}", name),
                description: format!("Model for prediction benchmarking: {}", name),
                layers: layers.clone(),
                activation: ActivationFunction::Sigmoid as i32,
                learning_rate: 0.01,
                max_epochs: 50,
                desired_error: 0.1,
                training_algorithm: TrainingAlgorithm::Backpropagation as i32,
            };
            
            // Create minimal training data
            let training_data = TrainingData {
                examples: vec![
                    TrainingExample {
                        inputs: (0..layers[0]).map(|i| i as f64 / layers[0] as f64).collect(),
                        outputs: (0..layers[layers.len() - 1]).map(|i| i as f64 / layers[layers.len() - 1] as f64).collect(),
                    }
                ],
                input_size: layers[0],
                output_size: layers[layers.len() - 1],
            };
            
            let train_request = TrainRequest {
                model_config: Some(model_config),
                training_data: Some(training_data),
                training_config: Some(TrainingConfig {
                    batch_size: 1,
                    shuffle: false,
                    validation_split: 0.0,
                    patience: 0,
                    save_best: false,
                }),
            };
            
            let response = service.train(tonic::Request::new(train_request)).await.unwrap();
            let train_response = response.into_inner();
            
            if train_response.status == "success" {
                model_ids.push(train_response.model_id);
            }
        }
        
        Self {
            service,
            model_ids,
            _temp_dir: temp_dir,
        }
    }
}

async fn benchmark_prediction(
    service: &NeuralServiceImpl,
    model_id: &str,
    input_vector: Vec<f64>,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let predict_request = PredictRequest {
        model_id: model_id.to_string(),
        input_vector,
    };

    let response = service.predict(tonic::Request::new(predict_request)).await?;
    let predict_response = response.into_inner();

    if predict_response.status != "success" {
        return Err(format!("Prediction failed: {}", predict_response.message).into());
    }

    Ok(predict_response.output_vector)
}

fn bench_prediction_model_sizes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let setup = rt.block_on(BenchmarkSetup::new());
    
    let model_configs = vec![
        (0, vec![5, 5], "small"),
        (1, vec![10, 20, 10], "medium"),
        (2, vec![20, 50, 30, 10], "large"),
        (3, vec![50, 100, 50, 20], "xlarge"),
    ];
    
    let mut group = c.benchmark_group("prediction_model_sizes");
    group.sample_size(100);
    
    for (idx, layers, name) in model_configs {
        if idx < setup.model_ids.len() {
            let model_id = &setup.model_ids[idx];
            let input_size = layers[0] as usize;
            
            group.bench_with_input(
                BenchmarkId::from_parameter(name),
                &(model_id.clone(), input_size),
                |b, (model_id, input_size)| {
                    b.to_async(&rt).iter(|| async {
                        let input_vector: Vec<f64> = (0..*input_size)
                            .map(|i| i as f64 / *input_size as f64)
                            .collect();
                        
                        let result = benchmark_prediction(&setup.service, model_id, input_vector).await;
                        black_box(result)
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_prediction_batch_sizes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let setup = rt.block_on(BenchmarkSetup::new());
    
    if setup.model_ids.is_empty() {
        return;
    }
    
    let model_id = &setup.model_ids[1]; // Use medium model
    let input_size = 10;
    let batch_sizes = vec![1, 10, 50, 100, 500];
    
    let mut group = c.benchmark_group("prediction_batch_sizes");
    group.sample_size(50);
    
    for batch_size in batch_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter(|| async {
                    let mut handles = Vec::new();
                    
                    for i in 0..batch_size {
                        let service = &setup.service;
                        let model_id = model_id.clone();
                        let input_vector: Vec<f64> = (0..input_size)
                            .map(|j| (i + j) as f64 / (batch_size + input_size) as f64)
                            .collect();
                        
                        let handle = tokio::spawn(async move {
                            benchmark_prediction(service, &model_id, input_vector).await
                        });
                        handles.push(handle);
                    }
                    
                    // Wait for all predictions
                    let mut results = Vec::new();
                    for handle in handles {
                        if let Ok(result) = handle.await {
                            results.push(result);
                        }
                    }
                    
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_prediction_input_sizes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let setup = rt.block_on(BenchmarkSetup::new());
    
    // Create models with different input sizes
    let input_sizes = vec![5, 10, 20, 50];
    let mut test_model_ids = Vec::new();
    
    for input_size in &input_sizes {
        let model_config = ModelConfig {
            name: format!("Input Size Test {}", input_size),
            description: format!("Model with {} inputs", input_size),
            layers: vec![*input_size, (*input_size * 2) as u32, 1],
            activation: ActivationFunction::Sigmoid as i32,
            learning_rate: 0.01,
            max_epochs: 10,
            desired_error: 0.5,
            training_algorithm: TrainingAlgorithm::Backpropagation as i32,
        };
        
        let training_data = TrainingData {
            examples: vec![
                TrainingExample {
                    inputs: (0..*input_size).map(|i| i as f64 / *input_size as f64).collect(),
                    outputs: vec![0.5],
                }
            ],
            input_size: *input_size as u32,
            output_size: 1,
        };
        
        let train_request = TrainRequest {
            model_config: Some(model_config),
            training_data: Some(training_data),
            training_config: Some(TrainingConfig {
                batch_size: 1,
                shuffle: false,
                validation_split: 0.0,
                patience: 0,
                save_best: false,
            }),
        };
        
        if let Ok(response) = rt.block_on(setup.service.train(tonic::Request::new(train_request))) {
            let train_response = response.into_inner();
            if train_response.status == "success" {
                test_model_ids.push((train_response.model_id, *input_size));
            }
        }
    }
    
    let mut group = c.benchmark_group("prediction_input_sizes");
    group.sample_size(100);
    
    for (model_id, input_size) in test_model_ids {
        group.bench_with_input(
            BenchmarkId::from_parameter(input_size),
            &(model_id.clone(), input_size),
            |b, (model_id, input_size)| {
                b.to_async(&rt).iter(|| async {
                    let input_vector: Vec<f64> = (0..*input_size)
                        .map(|i| (i as f64).sin())
                        .collect();
                    
                    let result = benchmark_prediction(&setup.service, model_id, input_vector).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_prediction_concurrent_load(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let setup = rt.block_on(BenchmarkSetup::new());
    
    if setup.model_ids.is_empty() {
        return;
    }
    
    let model_id = &setup.model_ids[1]; // Use medium model
    let concurrency_levels = vec![1, 5, 10, 20, 50];
    
    let mut group = c.benchmark_group("prediction_concurrent_load");
    group.sample_size(20);
    
    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::from_parameter(concurrency),
            &concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
                    let mut handles = Vec::new();
                    
                    for i in 0..100 {
                        let permit = semaphore.clone().acquire_owned().await.unwrap();
                        let service = &setup.service;
                        let model_id = model_id.clone();
                        
                        let handle = tokio::spawn(async move {
                            let _permit = permit;
                            let input_vector: Vec<f64> = (0..10)
                                .map(|j| ((i + j) as f64 / 110.0).sin())
                                .collect();
                            
                            benchmark_prediction(service, &model_id, input_vector).await
                        });
                        handles.push(handle);
                    }
                    
                    // Wait for all predictions
                    let mut results = Vec::new();
                    for handle in handles {
                        if let Ok(result) = handle.await {
                            results.push(result);
                        }
                    }
                    
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_prediction_model_sizes,
    bench_prediction_batch_sizes,
    bench_prediction_input_sizes,
    bench_prediction_concurrent_load
);
criterion_main!(benches);