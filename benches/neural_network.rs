//! Performance benchmarks for ruv-FANN neural network library

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ruv_fann::*;

/// Benchmark network creation with different sizes
fn bench_network_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_creation");

    for layers in &[
        vec![2, 3, 1],          // Small network
        vec![10, 20, 10, 5],    // Medium network
        vec![100, 50, 25, 10],  // Large network
        vec![784, 128, 64, 10], // MNIST-sized network
    ] {
        let total_neurons: usize = layers.iter().sum();
        group.bench_with_input(
            BenchmarkId::new("layers", format!("{layers:?}")),
            &layers,
            |b, layers| {
                b.iter(|| {
                    let network = Network::<f32>::new(black_box(layers));
                    black_box(network);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark forward propagation (inference)
fn bench_forward_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_propagation");

    let configs = vec![
        (vec![2, 3, 1], "XOR-sized"),
        (vec![10, 20, 10], "Small"),
        (vec![100, 50, 25, 10], "Medium"),
        (vec![784, 128, 64, 10], "MNIST-sized"),
    ];

    for (layers, name) in configs {
        let mut network = Network::new(&layers);
        let input = vec![0.5; layers[0]];

        group.bench_function(name, |b| {
            b.iter(|| {
                let output = network.run(black_box(&input));
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark training performance
fn bench_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("training");
    group.sample_size(10); // Reduce sample size for longer benchmarks

    // XOR problem benchmark
    let xor_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let xor_outputs = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    group.bench_function("XOR_100_epochs", |b| {
        b.iter(|| {
            let mut network = Network::new(&[2, 3, 1]);
            network
                .train(black_box(&xor_inputs), black_box(&xor_outputs), 0.1, 100)
                .expect("Training failed");
        });
    });

    // Larger problem benchmark
    let large_inputs: Vec<Vec<f32>> = (0..100)
        .map(|i| vec![i as f32 / 100.0, (i as f32 / 50.0).sin()])
        .collect();

    let large_outputs: Vec<Vec<f32>> = large_inputs
        .iter()
        .map(|inp| vec![(inp[0] + inp[1]) / 2.0])
        .collect();

    group.bench_function("Large_dataset_50_epochs", |b| {
        b.iter(|| {
            let mut network = Network::new(&[2, 10, 5, 1]);
            network
                .train(black_box(&large_inputs), black_box(&large_outputs), 0.1, 50)
                .expect("Training failed");
        });
    });

    group.finish();
}

/// Benchmark different training algorithms
fn bench_training_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_algorithms");
    group.sample_size(10);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let outputs = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let algorithms = vec![
        (TrainingAlgorithm::Backpropagation, "Backpropagation"),
        (TrainingAlgorithm::RProp, "RProp"),
        (TrainingAlgorithm::QuickProp, "QuickProp"),
        (TrainingAlgorithm::Batch, "Batch"),
    ];

    for (algorithm, name) in algorithms {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut network = Network::new(&[2, 4, 1]);
                network.set_training_algorithm(algorithm);
                network
                    .train(black_box(&inputs), black_box(&outputs), 0.01, 500)
                    .expect("Training failed");
            });
        });
    }

    group.finish();
}

/// Benchmark batch processing
fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");

    let mut network = Network::new(&[10, 20, 10, 5]);

    for batch_size in &[1, 10, 32, 64, 128, 256] {
        let inputs: Vec<Vec<f32>> = (0..*batch_size)
            .map(|i| vec![i as f32 / *batch_size as f32; 10])
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let outputs = network.run_batch(black_box(&inputs));
                    black_box(outputs);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark serialization and deserialization
fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    let configs = vec![
        (vec![2, 3, 1], "Small"),
        (vec![10, 20, 10], "Medium"),
        (vec![100, 50, 25, 10], "Large"),
    ];

    for (layers, name) in configs {
        let mut network = Network::new(&layers);
        network.randomize_weights(-1.0, 1.0);

        group.bench_function(format!("{name}_serialize"), |b| {
            b.iter(|| {
                let bytes = network.to_bytes();
                black_box(bytes);
            });
        });

        let bytes = network.to_bytes();

        group.bench_function(format!("{name}_deserialize"), |b| {
            b.iter(|| {
                let loaded =
                    Network::<f32>::from_bytes(black_box(&bytes)).expect("Deserialization failed");
                black_box(loaded);
            });
        });
    }

    group.finish();
}

/// Benchmark different activation functions
fn bench_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_functions");

    let activations = vec![
        (ActivationFunction::Sigmoid, "Sigmoid"),
        (ActivationFunction::Tanh, "Tanh"),
        (ActivationFunction::ReLU, "ReLU"),
        (ActivationFunction::ReLULeaky, "LeakyReLU"),
        (ActivationFunction::Linear, "Linear"),
    ];

    let network_layers = vec![10, 20, 10];
    let input = vec![0.5; 10];

    for (activation, name) in activations {
        group.bench_function(name, |b| {
            let mut network = Network::new(&network_layers);
            // Set activation function for all hidden layers
            for layer_idx in 1..network.num_layers() - 1 {
                network.set_activation_function(layer_idx, activation);
            }

            b.iter(|| {
                let output = network.run(black_box(&input));
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark weight operations
fn bench_weight_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_operations");

    let mut network = Network::new(&[100, 50, 25, 10]);
    let total_weights = network.get_total_connections();

    group.bench_function("get_weights", |b| {
        b.iter(|| {
            let weights = network.get_weights();
            black_box(weights);
        });
    });

    let weights = vec![0.1; total_weights];

    group.bench_function("set_weights", |b| {
        b.iter(|| {
            network
                .set_weights(black_box(&weights))
                .expect("Setting weights failed");
        });
    });

    group.bench_function("randomize_weights", |b| {
        b.iter(|| {
            network.randomize_weights(-1.0, 1.0);
        });
    });

    group.finish();
}

#[cfg(feature = "parallel")]
/// Benchmark parallel training
fn bench_parallel_training(c: &mut Criterion) {
    // use ruv_fann::ParallelTrainingOptions; // Not yet implemented

    let mut group = c.benchmark_group("parallel_training");
    group.sample_size(10);

    // Generate larger dataset
    let inputs: Vec<Vec<f32>> = (0..1000)
        .map(|i| {
            let x = i as f32 / 1000.0;
            vec![x, x.sin(), x.cos()]
        })
        .collect();

    let outputs: Vec<Vec<f32>> = inputs
        .iter()
        .map(|inp| vec![inp.iter().sum::<f32>() / 3.0])
        .collect();

    for threads in &[1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            threads,
            |b, &threads| {
                b.iter(|| {
                    let mut network = Network::new(&[3, 10, 5, 1]);
                    // Note: parallel training not yet implemented
                    /*
                    let options = ParallelTrainingOptions::default()
                        .with_threads(threads)
                        .with_batch_size(32);

                    network.train_parallel(
                        black_box(&inputs),
                        black_box(&outputs),
                        0.01,
                        10,
                        options
                    );
                    */
                    network
                        .train(black_box(&inputs), black_box(&outputs), 0.01, 10)
                        .expect("Training failed");
                });
            },
        );
    }

    group.finish();
}

/// Benchmark cascade network operations
fn bench_cascade_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("cascade_network");
    group.sample_size(10);

    let inputs = [
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let outputs = [vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    group.bench_function("cascade_training", |b| {
        b.iter(|| {
            let network = Network::new(&[2, 2, 1]);
            let mut config = CascadeConfig::<f32>::default();
            config.max_hidden_neurons = 5;
            config.output_max_epochs = 100;
            config.candidate_max_epochs = 100;
            let cascade = CascadeNetwork::new(network, config);
            // Note: train_cascade method would need to be implemented
            black_box(cascade);
        });
    });

    group.finish();
}

// Configure criterion benchmarks
criterion_group!(
    benches,
    bench_network_creation,
    bench_forward_propagation,
    bench_training,
    bench_training_algorithms,
    bench_batch_processing,
    bench_serialization,
    bench_activation_functions,
    bench_weight_operations,
    bench_cascade_network,
);

#[cfg(feature = "parallel")]
criterion_group!(parallel_benches, bench_parallel_training,);

#[cfg(not(feature = "parallel"))]
criterion_main!(benches);

#[cfg(feature = "parallel")]
criterion_main!(benches, parallel_benches);
