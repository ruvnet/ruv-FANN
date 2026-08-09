//! GPU Sweet Spot Performance Benchmark
//!
//! This benchmark tests different network sizes and batch sizes to find
//! the optimal configuration where GPU acceleration provides maximum benefit.

use num_traits::Float;
use rand::prelude::*;
use rand_distr::StandardNormal;
use ruv_fann::training::*;
use ruv_fann::*;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
struct BenchmarkResult {
    network_description: String,
    batch_size: usize,
    cpu_time: Duration,
    gpu_time: Duration,
    speedup: f32,
    cpu_samples_per_sec: f32,
    gpu_samples_per_sec: f32,
    final_cpu_error: f32,
    final_gpu_error: f32,
}

#[derive(Debug, Clone)]
struct TestConfig {
    input_size: usize,
    hidden_layers: Vec<usize>,
    output_size: usize,
    description: String,
    batch_sizes: Vec<usize>,
}

impl TestConfig {
    fn new(
        input: usize,
        hidden: Vec<usize>,
        output: usize,
        desc: &str,
        batches: Vec<usize>,
    ) -> Self {
        Self {
            input_size: input,
            hidden_layers: hidden,
            output_size: output,
            description: desc.to_string(),
            batch_sizes: batches,
        }
    }

    fn total_parameters(&self) -> usize {
        let mut params = 0;
        let mut prev_size = self.input_size;

        for &hidden_size in &self.hidden_layers {
            params += prev_size * hidden_size + hidden_size; // weights + biases
            prev_size = hidden_size;
        }

        params += prev_size * self.output_size + self.output_size; // output layer
        params
    }
}

fn generate_training_data<T: Float + rand::distr::uniform::SampleUniform>(
    input_size: usize,
    output_size: usize,
    samples: usize,
) -> TrainingData<T> {
    let mut rng = rand::rng();
    let normal = StandardNormal;

    let inputs: Vec<Vec<T>> = (0..samples)
        .map(|_| {
            (0..input_size)
                .map(|_| T::from(rng.sample::<f64, _>(normal) * 0.5).unwrap())
                .collect()
        })
        .collect();

    let outputs: Vec<Vec<T>> = (0..samples)
        .map(|_| {
            (0..output_size)
                .map(|_| T::from(rng.random_range(0.0..1.0)).unwrap())
                .collect()
        })
        .collect();

    TrainingData { inputs, outputs }
}

fn benchmark_configuration(
    config: &TestConfig,
    batch_size: usize,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    println!("  📊 Testing batch size: {}", batch_size);

    // Generate training data
    let training_data =
        generate_training_data::<f32>(config.input_size, config.output_size, batch_size);

    // Create identical networks for CPU and GPU
    let mut builder = NetworkBuilder::new().input_layer(config.input_size);

    for &hidden_size in &config.hidden_layers {
        builder = builder.hidden_layer(hidden_size);
    }

    let mut cpu_network = builder.output_layer(config.output_size).build();

    let mut gpu_network = cpu_network.clone();

    // Benchmark CPU training
    let mut cpu_adam = Adam::new(0.001f32)
        .with_beta1(0.9)
        .with_beta2(0.999)
        .with_epsilon(1e-8);

    let cpu_start = Instant::now();
    let mut cpu_final_error = 0.0f32;
    for epoch in 0..5 {
        match cpu_adam.train_epoch(&mut cpu_network, &training_data) {
            Ok(error) => {
                cpu_final_error = error;
                if epoch == 0 || epoch == 4 {
                    println!("      CPU Epoch {}: Error = {:.6}", epoch, error);
                }
            }
            Err(e) => {
                println!("      CPU training error: {}", e);
                break;
            }
        }
    }
    let cpu_time = cpu_start.elapsed();

    // Benchmark GPU training
    let gpu_start = Instant::now();
    let mut gpu_final_error = 0.0f32;

    match GpuAdam::new(0.001f32) {
        Ok(mut gpu_adam) => {
            for epoch in 0..5 {
                match gpu_adam.train_epoch(&mut gpu_network, &training_data) {
                    Ok(error) => {
                        gpu_final_error = error;
                        if epoch == 0 || epoch == 4 {
                            println!("      GPU Epoch {}: Error = {:.6}", epoch, error);
                        }
                    }
                    Err(e) => {
                        println!("      GPU training error: {}", e);
                        break;
                    }
                }
            }
        }
        Err(e) => {
            println!("      GPU initialization failed: {}", e);
            return Ok(BenchmarkResult {
                network_description: config.description.clone(),
                batch_size,
                cpu_time,
                gpu_time: Duration::from_secs(999),
                speedup: 0.0,
                cpu_samples_per_sec: batch_size as f32 / cpu_time.as_secs_f32(),
                gpu_samples_per_sec: 0.0,
                final_cpu_error: cpu_final_error,
                final_gpu_error: 999.0,
            });
        }
    }

    let gpu_time = gpu_start.elapsed();

    // Calculate performance metrics
    let speedup = cpu_time.as_secs_f32() / gpu_time.as_secs_f32();
    let cpu_samples_per_sec = batch_size as f32 / cpu_time.as_secs_f32();
    let gpu_samples_per_sec = batch_size as f32 / gpu_time.as_secs_f32();

    println!("      ✅ CPU: {:.3}s ({:.0} samples/sec) | GPU: {:.3}s ({:.0} samples/sec) | Speedup: {:.2}x", 
             cpu_time.as_secs_f32(), cpu_samples_per_sec,
             gpu_time.as_secs_f32(), gpu_samples_per_sec, speedup);

    Ok(BenchmarkResult {
        network_description: config.description.clone(),
        batch_size,
        cpu_time,
        gpu_time,
        speedup,
        cpu_samples_per_sec,
        gpu_samples_per_sec,
        final_cpu_error: cpu_final_error,
        final_gpu_error: gpu_final_error,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 GPU Sweet Spot Performance Benchmark");
    println!("========================================");
    println!();

    // Test different network sizes and batch sizes to find the sweet spot
    let test_configs = vec![
        TestConfig::new(
            128,
            vec![256, 128],
            16,
            "Medium Network (50K params)",
            vec![1000, 2000, 5000, 10000],
        ),
        TestConfig::new(
            256,
            vec![512, 256],
            32,
            "Large Network (200K params)",
            vec![1000, 2000, 5000, 10000],
        ),
        TestConfig::new(
            512,
            vec![1024, 512],
            64,
            "Very Large Network (800K params)",
            vec![1000, 2000, 5000],
        ),
        TestConfig::new(
            1024,
            vec![2048, 1024],
            128,
            "Huge Network (3.2M params)",
            vec![1000, 2000],
        ),
    ];

    let mut all_results = Vec::new();

    println!(
        "🔧 GPU Info: {}",
        ruv_fann::training::get_gpu_capabilities()
    );
    println!();

    for config in &test_configs {
        println!("🚀 Testing {}", config.description);
        println!(
            "   Architecture: {}-{:?}-{}",
            config.input_size, config.hidden_layers, config.output_size
        );
        println!("   Total parameters: {}", config.total_parameters());
        println!("   =====================================");

        for &batch_size in &config.batch_sizes {
            match benchmark_configuration(config, batch_size) {
                Ok(result) => all_results.push(result),
                Err(e) => println!("  ❌ Benchmark failed: {}", e),
            }
        }
        println!();
    }

    // Find the sweet spot
    println!("📈 Sweet Spot Analysis");
    println!("======================");

    let mut best_speedup = 0.0f32;
    let mut best_config: Option<&BenchmarkResult> = None;

    // Group results by network size
    for config in &test_configs {
        let config_results: Vec<_> = all_results
            .iter()
            .filter(|r| r.network_description == config.description)
            .collect();

        if !config_results.is_empty() {
            println!("\n🔷 {}", config.description);
            println!("┌─────────────┬────────────┬────────────┬──────────────┬─────────────────┐");
            println!("│ Batch Size  │ CPU Time   │ GPU Time   │ Speedup      │ GPU Samples/sec │");
            println!("├─────────────┼────────────┼────────────┼──────────────┼─────────────────┤");

            for result in &config_results {
                println!(
                    "│ {:>9}   │ {:>8.3}s │ {:>8.3}s │ {:>10.2}x │ {:>13.0}   │",
                    result.batch_size,
                    result.cpu_time.as_secs_f32(),
                    result.gpu_time.as_secs_f32(),
                    result.speedup,
                    result.gpu_samples_per_sec
                );

                if result.speedup > best_speedup {
                    best_speedup = result.speedup;
                    best_config = Some(result);
                }
            }

            println!("└─────────────┴────────────┴────────────┴──────────────┴─────────────────┘");
        }
    }

    // Report the sweet spot
    if let Some(best) = best_config {
        println!("\n🎯 **SWEET SPOT FOUND!**");
        println!("   Network: {}", best.network_description);
        println!("   Batch Size: {}", best.batch_size);
        println!("   Speedup: {:.2}x", best.speedup);
        println!(
            "   GPU Throughput: {:.0} samples/sec",
            best.gpu_samples_per_sec
        );

        if best.speedup > 2.0 {
            println!("   🚀 Excellent GPU acceleration!");
        } else if best.speedup > 1.5 {
            println!("   ✅ Good GPU acceleration");
        } else {
            println!("   ⚠️  Modest GPU benefits - try larger networks/batches");
        }
    } else {
        println!("\n❌ No significant GPU acceleration found");
        println!("   Recommendations:");
        println!("   • Try even larger networks (>5M parameters)");
        println!("   • Use larger batch sizes (>10K samples)");
        println!("   • Check GPU memory constraints");
    }

    println!("\n✅ Sweet spot analysis completed!");
    Ok(())
}
