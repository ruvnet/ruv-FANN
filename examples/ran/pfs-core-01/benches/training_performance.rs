use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use pfs_core_01::neural_service::*;
use pfs_core_01::{NeuralServiceImpl, ModelManager};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::runtime::Runtime;

fn create_training_data(input_size: usize, output_size: usize, num_samples: usize) -> TrainingData {
    let mut examples = Vec::new();
    
    for i in 0..num_samples {
        let inputs: Vec<f64> = (0..input_size)
            .map(|j| ((i + j) as f64 / num_samples as f64).sin())
            .collect();
        
        let outputs: Vec<f64> = (0..output_size)
            .map(|j| ((i + j) as f64 / num_samples as f64).cos())
            .collect();
        
        examples.push(TrainingExample { inputs, outputs });
    }
    
    TrainingData {
        examples,
        input_size: input_size as u32,
        output_size: output_size as u32,
    }
}

fn create_model_config(
    layers: Vec<u32>,
    algorithm: TrainingAlgorithm,
    learning_rate: f64,
    max_epochs: u32,
) -> ModelConfig {
    ModelConfig {
        name: "Benchmark Model".to_string(),
        description: "Model for performance benchmarking".to_string(),
        layers,
        activation: ActivationFunction::Sigmoid as i32,
        learning_rate,
        max_epochs,
        desired_error: 0.001,
        training_algorithm: algorithm as i32,
    }
}

async fn benchmark_training(
    service: &NeuralServiceImpl,
    model_config: ModelConfig,
    training_data: TrainingData,
) -> Result<(String, TrainingResults), Box<dyn std::error::Error>> {
    let train_request = TrainRequest {
        model_config: Some(model_config),
        training_data: Some(training_data),
        training_config: Some(TrainingConfig {
            batch_size: 32,
            shuffle: false, // Disable shuffling for consistent benchmarks
            validation_split: 0.0,
            patience: 0,
            save_best: false,
        }),
    };

    let response = service.train(tonic::Request::new(train_request)).await?;
    let train_response = response.into_inner();

    if train_response.status != "success" {
        return Err(format!("Training failed: {}", train_response.message).into());
    }

    let results = train_response.results
        .ok_or("Training results missing")?;

    Ok((train_response.model_id, results))
}

fn bench_training_algorithms(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let temp_dir = TempDir::new().unwrap();
    let model_manager = Arc::new(ModelManager::new(temp_dir.path().join("models"), 100));
    
    rt.block_on(async {
        model_manager.initialize().await.unwrap();
    });
    
    let service = NeuralServiceImpl::new(model_manager, 4);
    
    let algorithms = vec![
        TrainingAlgorithm::Backpropagation,
        TrainingAlgorithm::Rprop,
        TrainingAlgorithm::Quickprop,
    ];
    
    let mut group = c.benchmark_group("training_algorithms");
    group.sample_size(10);
    
    for algorithm in algorithms {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", algorithm)),
            &algorithm,
            |b, &algorithm| {
                b.to_async(&rt).iter(|| async {
                    let model_config = create_model_config(
                        vec![10, 20, 10, 1],
                        algorithm,
                        0.01,
                        100,
                    );
                    
                    let training_data = create_training_data(10, 1, 100);
                    
                    let result = benchmark_training(&service, model_config, training_data).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_network_sizes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let temp_dir = TempDir::new().unwrap();
    let model_manager = Arc::new(ModelManager::new(temp_dir.path().join("models"), 100));
    
    rt.block_on(async {
        model_manager.initialize().await.unwrap();
    });
    
    let service = NeuralServiceImpl::new(model_manager, 4);
    
    let network_configs = vec![
        ("small", vec![5, 10, 5], 50, 50),
        ("medium", vec![10, 20, 15, 5], 100, 100),
        ("large", vec![20, 50, 30, 10], 200, 50),
    ];
    
    let mut group = c.benchmark_group("network_sizes");
    group.sample_size(10);
    
    for (name, layers, data_size, epochs) in network_configs {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(layers.clone(), data_size, epochs),
            |b, &(ref layers, data_size, epochs)| {
                b.to_async(&rt).iter(|| async {
                    let model_config = create_model_config(
                        layers.clone(),
                        TrainingAlgorithm::Backpropagation,
                        0.01,
                        epochs,
                    );
                    
                    let training_data = create_training_data(layers[0] as usize, layers[layers.len() - 1] as usize, data_size);
                    
                    let result = benchmark_training(&service, model_config, training_data).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_dataset_sizes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let temp_dir = TempDir::new().unwrap();
    let model_manager = Arc::new(ModelManager::new(temp_dir.path().join("models"), 100));
    
    rt.block_on(async {
        model_manager.initialize().await.unwrap();
    });
    
    let service = NeuralServiceImpl::new(model_manager, 4);
    
    let dataset_sizes = vec![100, 500, 1000, 2000];
    
    let mut group = c.benchmark_group("dataset_sizes");
    group.sample_size(10);
    
    for size in dataset_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let model_config = create_model_config(
                        vec![10, 15, 5],
                        TrainingAlgorithm::Backpropagation,
                        0.01,
                        50,
                    );
                    
                    let training_data = create_training_data(10, 5, size);
                    
                    let result = benchmark_training(&service, model_config, training_data).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_learning_rates(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let temp_dir = TempDir::new().unwrap();
    let model_manager = Arc::new(ModelManager::new(temp_dir.path().join("models"), 100));
    
    rt.block_on(async {
        model_manager.initialize().await.unwrap();
    });
    
    let service = NeuralServiceImpl::new(model_manager, 4);
    
    let learning_rates = vec![0.001, 0.01, 0.1, 0.5];
    
    let mut group = c.benchmark_group("learning_rates");
    group.sample_size(10);
    
    for rate in learning_rates {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.3}", rate)),
            &rate,
            |b, &rate| {
                b.to_async(&rt).iter(|| async {
                    let model_config = create_model_config(
                        vec![10, 15, 5],
                        TrainingAlgorithm::Backpropagation,
                        rate,
                        100,
                    );
                    
                    let training_data = create_training_data(10, 5, 200);
                    
                    let result = benchmark_training(&service, model_config, training_data).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_training_algorithms,
    bench_network_sizes,
    bench_dataset_sizes,
    bench_learning_rates
);
criterion_main!(benches);