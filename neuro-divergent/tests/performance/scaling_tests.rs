//! Scalability tests for neuro-divergent
//!
//! This module tests how well the library scales with increasing data sizes,
//! model complexity, and computational resources.

use neuro_divergent::models::*;
use neuro_divergent::data::*;
use neuro_divergent::training::*;
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::thread;
use chrono::{DateTime, Utc, TimeZone};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Results from scaling tests
#[derive(Debug, Clone)]
struct ScalingResult {
    data_size: usize,
    model_size: usize,
    threads: usize,
    training_time: Duration,
    inference_time: Duration,
    throughput: f64,
    memory_usage_mb: f64,
    scaling_efficiency: f64,
}

impl ScalingResult {
    fn print_summary(&self) {
        println!("\nScaling Test Result:");
        println!("  Data Size: {}", self.data_size);
        println!("  Model Size: {}", self.model_size);
        println!("  Threads: {}", self.threads);
        println!("  Training Time: {:.2}s", self.training_time.as_secs_f64());
        println!("  Inference Time: {:.2}ms", self.inference_time.as_millis());
        println!("  Throughput: {:.2} samples/sec", self.throughput);
        println!("  Memory Usage: {:.2} MB", self.memory_usage_mb);
        println!("  Scaling Efficiency: {:.2}%", self.scaling_efficiency * 100.0);
    }
}

/// Test data size scaling (linear scaling expected)
#[test]
fn test_data_size_scaling() {
    println!("\n===== DATA SIZE SCALING TEST =====\n");
    
    let data_sizes = vec![1000, 5000, 10000, 50000, 100000];
    let mut results = Vec::new();
    
    for &size in &data_sizes {
        println!("Testing with {} samples...", size);
        
        // Generate data
        let dataset = generate_scaling_dataset(size, 48, 1);
        let training_data = convert_dataset_to_training_data(&dataset);
        
        // Create model
        let config = MLPConfig::new(24, 12)
            .with_hidden_layers(vec![64, 32]);
        let mut model = MLP::new(config).unwrap();
        
        // Setup trainer
        let mut trainer = Trainer::new()
            .with_optimizer(OptimizerType::Adam(Adam::new(0.001, 0.9, 0.999)))
            .with_loss_function(MSELoss::new())
            .build();
        
        let training_config = TrainingConfig {
            max_epochs: 10,
            batch_size: 32,
            validation_frequency: 5,
            patience: None,
            gradient_clip: Some(1.0),
            mixed_precision: false,
            seed: Some(42),
            device: DeviceConfig::Cpu { num_threads: None },
            checkpoint: CheckpointConfig {
                enabled: false,
                save_frequency: 10,
                keep_best_only: true,
                monitor_metric: "loss".to_string(),
                mode: CheckpointMode::Min,
            },
        };
        
        // Measure training time
        let start = Instant::now();
        let _ = trainer.train(&mut model, &training_data, &training_config);
        let training_time = start.elapsed();
        
        // Measure inference time
        let test_batch = dataset.series[0].slice(0, 24).unwrap();
        let start = Instant::now();
        for _ in 0..100 {
            let _ = model.predict(&test_batch);
        }
        let inference_time = start.elapsed() / 100;
        
        // Calculate metrics
        let throughput = size as f64 / training_time.as_secs_f64();
        let memory_estimate = estimate_memory_usage(&model, size);
        
        results.push(ScalingResult {
            data_size: size,
            model_size: model.count_parameters(),
            threads: 1,
            training_time,
            inference_time,
            throughput,
            memory_usage_mb: memory_estimate,
            scaling_efficiency: 1.0, // Will calculate relative to baseline
        });
    }
    
    // Calculate scaling efficiency
    let baseline_throughput = results[0].throughput;
    for (i, result) in results.iter_mut().enumerate() {
        let expected_throughput = baseline_throughput; // Should be constant for linear scaling
        result.scaling_efficiency = result.throughput / expected_throughput;
        result.print_summary();
    }
    
    // Verify reasonable scaling
    let efficiency_threshold = 0.7; // Allow 30% degradation
    for result in &results {
        assert!(result.scaling_efficiency >= efficiency_threshold,
                "Poor scaling efficiency at size {}: {:.2}%", 
                result.data_size, result.scaling_efficiency * 100.0);
    }
}

/// Test model complexity scaling
#[test]
fn test_model_complexity_scaling() {
    println!("\n===== MODEL COMPLEXITY SCALING TEST =====\n");
    
    let model_configs = vec![
        ("tiny", vec![32]),
        ("small", vec![64, 32]),
        ("medium", vec![128, 64, 32]),
        ("large", vec![256, 128, 64, 32]),
        ("xlarge", vec![512, 256, 128, 64, 32]),
    ];
    
    let fixed_data_size = 10000;
    let dataset = generate_scaling_dataset(fixed_data_size, 48, 1);
    let training_data = convert_dataset_to_training_data(&dataset);
    
    let mut results = Vec::new();
    
    for (name, layers) in model_configs {
        println!("Testing {} model...", name);
        
        // Create model
        let config = MLPConfig::new(24, 12)
            .with_hidden_layers(layers.clone());
        let mut model = MLP::new(config).unwrap();
        let model_size = model.count_parameters();
        
        // Setup trainer
        let mut trainer = Trainer::new()
            .with_optimizer(OptimizerType::Adam(Adam::new(0.001, 0.9, 0.999)))
            .with_loss_function(MSELoss::new())
            .build();
        
        let training_config = TrainingConfig {
            max_epochs: 5,
            batch_size: 32,
            validation_frequency: 5,
            patience: None,
            gradient_clip: Some(1.0),
            mixed_precision: false,
            seed: Some(42),
            device: DeviceConfig::Cpu { num_threads: None },
            checkpoint: CheckpointConfig {
                enabled: false,
                save_frequency: 10,
                keep_best_only: true,
                monitor_metric: "loss".to_string(),
                mode: CheckpointMode::Min,
            },
        };
        
        // Measure training time
        let start = Instant::now();
        let _ = trainer.train(&mut model, &training_data, &training_config);
        let training_time = start.elapsed();
        
        // Measure inference time
        let test_batch = dataset.series[0].slice(0, 24).unwrap();
        let start = Instant::now();
        for _ in 0..100 {
            let _ = model.predict(&test_batch);
        }
        let inference_time = start.elapsed() / 100;
        
        let result = ScalingResult {
            data_size: fixed_data_size,
            model_size,
            threads: 1,
            training_time,
            inference_time,
            throughput: fixed_data_size as f64 / training_time.as_secs_f64(),
            memory_usage_mb: estimate_memory_usage(&model, fixed_data_size),
            scaling_efficiency: 1.0,
        };
        
        result.print_summary();
        results.push((name, model_size, result));
    }
    
    // Verify training time scales sub-quadratically with model size
    for i in 1..results.len() {
        let (prev_name, prev_size, prev_result) = &results[i-1];
        let (curr_name, curr_size, curr_result) = &results[i];
        
        let size_ratio = *curr_size as f64 / *prev_size as f64;
        let time_ratio = curr_result.training_time.as_secs_f64() / 
                        prev_result.training_time.as_secs_f64();
        
        println!("\n{} -> {}: size ratio = {:.2}x, time ratio = {:.2}x",
                 prev_name, curr_name, size_ratio, time_ratio);
        
        // Time should scale less than quadratically with size
        assert!(time_ratio < size_ratio * size_ratio,
                "Training time scales super-quadratically from {} to {}", 
                prev_name, curr_name);
    }
}

/// Test parallel processing scaling
#[cfg(feature = "parallel")]
#[test]
fn test_parallel_scaling() {
    println!("\n===== PARALLEL PROCESSING SCALING TEST =====\n");
    
    let thread_counts = vec![1, 2, 4, 8];
    let data_size = 50000;
    let num_series = 100;
    
    // Generate multiple series for parallel processing
    let dataset = generate_multi_series_dataset(num_series, data_size / num_series);
    let mut results = Vec::new();
    
    for &num_threads in &thread_counts {
        println!("Testing with {} threads...", num_threads);
        
        // Configure thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        
        // Create models for each series
        let models: Vec<_> = (0..num_series).map(|_| {
            let config = MLPConfig::new(24, 12)
                .with_hidden_layers(vec![64, 32]);
            let mut model = MLP::new(config).unwrap();
            model.randomize_weights(-0.1, 0.1, Some(42));
            Arc::new(model)
        }).collect();
        
        // Measure parallel training time
        let start = Instant::now();
        pool.install(|| {
            dataset.series.par_iter().zip(models.par_iter()).for_each(|(series, model)| {
                // Simulate training
                for _ in 0..10 {
                    let _ = model.predict(series);
                }
            });
        });
        let training_time = start.elapsed();
        
        // Measure parallel inference time
        let start = Instant::now();
        pool.install(|| {
            let predictions: Vec<_> = dataset.series.par_iter()
                .zip(models.par_iter())
                .map(|(series, model)| model.predict(series))
                .collect();
            predictions
        });
        let inference_time = start.elapsed();
        
        let throughput = data_size as f64 / training_time.as_secs_f64();
        
        results.push(ScalingResult {
            data_size,
            model_size: models[0].count_parameters() * num_series,
            threads: num_threads,
            training_time,
            inference_time,
            throughput,
            memory_usage_mb: estimate_memory_usage(&*models[0], data_size),
            scaling_efficiency: 1.0,
        });
    }
    
    // Calculate scaling efficiency (speedup / ideal speedup)
    let baseline_time = results[0].training_time.as_secs_f64();
    for result in &mut results {
        let ideal_speedup = result.threads as f64;
        let actual_speedup = baseline_time / result.training_time.as_secs_f64();
        result.scaling_efficiency = actual_speedup / ideal_speedup;
        result.print_summary();
    }
    
    // Verify reasonable parallel scaling
    for result in &results {
        let min_efficiency = match result.threads {
            1 => 1.0,
            2 => 0.8,  // 80% efficiency for 2 threads
            4 => 0.6,  // 60% efficiency for 4 threads
            8 => 0.4,  // 40% efficiency for 8 threads
            _ => 0.3,
        };
        
        assert!(result.scaling_efficiency >= min_efficiency,
                "Poor parallel scaling with {} threads: {:.2}%",
                result.threads, result.scaling_efficiency * 100.0);
    }
}

/// Test batch size scaling
#[test]
fn test_batch_size_scaling() {
    println!("\n===== BATCH SIZE SCALING TEST =====\n");
    
    let batch_sizes = vec![8, 16, 32, 64, 128, 256];
    let data_size = 10000;
    
    let dataset = generate_scaling_dataset(data_size, 48, 1);
    let training_data = convert_dataset_to_training_data(&dataset);
    
    let mut results = Vec::new();
    
    for &batch_size in &batch_sizes {
        println!("Testing with batch size {}...", batch_size);
        
        // Create model
        let config = MLPConfig::new(24, 12)
            .with_hidden_layers(vec![128, 64]);
        let mut model = MLP::new(config).unwrap();
        
        // Setup trainer
        let mut trainer = Trainer::new()
            .with_optimizer(OptimizerType::Adam(Adam::new(0.001, 0.9, 0.999)))
            .with_loss_function(MSELoss::new())
            .build();
        
        let training_config = TrainingConfig {
            max_epochs: 5,
            batch_size,
            validation_frequency: 5,
            patience: None,
            gradient_clip: Some(1.0),
            mixed_precision: false,
            seed: Some(42),
            device: DeviceConfig::Cpu { num_threads: None },
            checkpoint: CheckpointConfig {
                enabled: false,
                save_frequency: 10,
                keep_best_only: true,
                monitor_metric: "loss".to_string(),
                mode: CheckpointMode::Min,
            },
        };
        
        // Measure training time
        let start = Instant::now();
        let train_result = trainer.train(&mut model, &training_data, &training_config).unwrap();
        let training_time = start.elapsed();
        
        let result = ScalingResult {
            data_size,
            model_size: model.count_parameters(),
            threads: 1,
            training_time,
            inference_time: Duration::from_millis(0), // Not measured for this test
            throughput: data_size as f64 / training_time.as_secs_f64(),
            memory_usage_mb: estimate_memory_usage(&model, batch_size),
            scaling_efficiency: train_result.final_loss.to_f64().unwrap(),
        };
        
        println!("  Training time: {:.2}s", training_time.as_secs_f64());
        println!("  Throughput: {:.2} samples/sec", result.throughput);
        println!("  Final loss: {:.6}", result.scaling_efficiency);
        
        results.push((batch_size, result));
    }
    
    // Verify that larger batch sizes are more efficient (up to a point)
    let optimal_batch_idx = results.iter()
        .enumerate()
        .max_by(|(_, (_, a)), (_, (_, b))| {
            a.throughput.partial_cmp(&b.throughput).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap();
    
    println!("\nOptimal batch size: {}", batch_sizes[optimal_batch_idx]);
    
    // Throughput should generally increase with batch size up to a point
    for i in 1..optimal_batch_idx {
        assert!(results[i].1.throughput >= results[i-1].1.throughput * 0.95,
                "Throughput decreased significantly from batch {} to {}",
                batch_sizes[i-1], batch_sizes[i]);
    }
}

/// Test memory scaling with data size
#[test]
fn test_memory_scaling() {
    println!("\n===== MEMORY SCALING TEST =====\n");
    
    let data_sizes = vec![1000, 5000, 10000, 50000];
    let window_size = 100;
    
    for &size in &data_sizes {
        println!("\nTesting memory with {} samples...", size);
        
        // Test data structure memory
        let dataset = generate_scaling_dataset(size, window_size, 5);
        let dataset_memory = estimate_dataset_memory(&dataset);
        println!("  Dataset memory: {:.2} MB", dataset_memory);
        
        // Test preprocessing memory
        let values = dataset.series[0].values();
        let mut scaler = StandardScaler::default();
        scaler.fit(&values).unwrap();
        let scaled = scaler.transform(&values).unwrap();
        
        let preprocess_memory = scaled.len() * std::mem::size_of::<f64>() as f64 / 1_048_576.0;
        println!("  Preprocessing memory: {:.2} MB", preprocess_memory);
        
        // Test feature engineering memory
        let lags = vec![1, 2, 3, 6, 12, 24];
        let lag_features = generate_lag_features(&scaled, &lags);
        let feature_memory = lag_features.iter()
            .map(|f| f.len() * std::mem::size_of::<f64>())
            .sum::<usize>() as f64 / 1_048_576.0;
        println!("  Feature memory: {:.2} MB", feature_memory);
        
        // Verify memory scales linearly with data size
        let total_memory = dataset_memory + preprocess_memory + feature_memory;
        let memory_per_sample = total_memory / size as f64 * 1_048_576.0; // bytes per sample
        
        println!("  Total memory: {:.2} MB", total_memory);
        println!("  Memory per sample: {:.2} bytes", memory_per_sample);
        
        // Memory per sample should be relatively constant
        assert!(memory_per_sample < 1000.0, 
                "Excessive memory usage per sample: {:.2} bytes", memory_per_sample);
    }
}

/// Test scalability limits
#[test]
#[ignore] // This test is resource-intensive
fn test_scalability_limits() {
    println!("\n===== SCALABILITY LIMITS TEST =====\n");
    
    let test_configs = vec![
        ("1M samples", 1_000_000, 1, 24),
        ("10K series", 10_000, 10_000, 100),
        ("Deep model", 10_000, 1, 24),
    ];
    
    for (name, total_samples, num_series, seq_len) in test_configs {
        println!("\nTesting {}: {} total samples across {} series...", 
                 name, total_samples, num_series);
        
        let start_time = Instant::now();
        
        // Test data generation
        let gen_start = Instant::now();
        let dataset = if num_series == 1 {
            generate_scaling_dataset(total_samples, seq_len, 1)
        } else {
            generate_multi_series_dataset(num_series, total_samples / num_series)
        };
        let gen_time = gen_start.elapsed();
        println!("  Data generation: {:.2}s", gen_time.as_secs_f64());
        
        // Test model creation
        let model_start = Instant::now();
        let config = if name.contains("Deep") {
            MLPConfig::new(seq_len, 12)
                .with_hidden_layers(vec![512, 256, 128, 64, 32, 16])
        } else {
            MLPConfig::new(seq_len, 12)
                .with_hidden_layers(vec![128, 64])
        };
        let model = MLP::new(config).unwrap();
        let model_time = model_start.elapsed();
        println!("  Model creation: {:.2}s", model_time.as_secs_f64());
        println!("  Model parameters: {}", model.count_parameters());
        
        // Test single inference
        let inference_start = Instant::now();
        let test_input = vec![0.5; seq_len];
        let _ = model.predict(&test_input);
        let inference_time = inference_start.elapsed();
        println!("  Single inference: {:.2}ms", inference_time.as_millis());
        
        let total_time = start_time.elapsed();
        println!("  Total test time: {:.2}s", total_time.as_secs_f64());
        
        // Verify reasonable performance even at scale
        assert!(gen_time.as_secs() < 60, "{} data generation too slow", name);
        assert!(model_time.as_secs() < 10, "{} model creation too slow", name);
        assert!(inference_time.as_millis() < 100, "{} inference too slow", name);
    }
}

/// Helper function to generate scaling dataset
fn generate_scaling_dataset(
    size: usize,
    window_size: usize,
    features: usize,
) -> TimeSeriesDataset<f64> {
    let mut dataset = TimeSeriesDataset::new();
    
    let timestamps: Vec<DateTime<Utc>> = (0..size)
        .map(|i| Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap() + chrono::Duration::hours(i as i64))
        .collect();
    
    let values: Vec<f64> = (0..size)
        .map(|i| {
            let trend = i as f64 * 0.01;
            let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 168.0).sin() * 10.0;
            let noise = ((i * 13) % 7) as f64 * 0.5 - 1.75;
            50.0 + trend + seasonal + noise
        })
        .collect();
    
    let series = TimeSeriesDatasetBuilder::new("scaling_series".to_string())
        .with_frequency("H".to_string())
        .with_values(values)
        .with_timestamps(timestamps)
        .build()
        .unwrap();
    
    dataset.add_series(series);
    dataset
}

/// Helper function to generate multi-series dataset
fn generate_multi_series_dataset(
    num_series: usize,
    samples_per_series: usize,
) -> TimeSeriesDataset<f64> {
    let mut dataset = TimeSeriesDataset::new();
    
    for s in 0..num_series {
        let timestamps: Vec<DateTime<Utc>> = (0..samples_per_series)
            .map(|i| Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap() + chrono::Duration::hours(i as i64))
            .collect();
        
        let values: Vec<f64> = (0..samples_per_series)
            .map(|i| {
                let series_offset = s as f64 * 5.0;
                let trend = i as f64 * 0.01;
                let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 24.0).sin() * 5.0;
                50.0 + series_offset + trend + seasonal
            })
            .collect();
        
        let series = TimeSeriesDatasetBuilder::new(format!("series_{}", s))
            .with_frequency("H".to_string())
            .with_values(values)
            .with_timestamps(timestamps)
            .build()
            .unwrap();
        
        dataset.add_series(series);
    }
    
    dataset
}

/// Helper function to convert dataset to training data
fn convert_dataset_to_training_data(dataset: &TimeSeriesDataset<f64>) -> TrainingData<f64> {
    let window_size = 24;
    let horizon = 12;
    let mut all_inputs = Vec::new();
    let mut all_targets = Vec::new();
    let mut all_metadata = Vec::new();
    
    for series in &dataset.series {
        let values = series.values();
        let mut series_inputs = Vec::new();
        let mut series_targets = Vec::new();
        
        for i in 0..(values.len() - window_size - horizon) {
            let input_window = values[i..i + window_size]
                .iter()
                .map(|&v| vec![v])
                .collect();
            let target_window = values[i + window_size..i + window_size + horizon]
                .iter()
                .map(|&v| vec![v])
                .collect();
            
            series_inputs.push(input_window);
            series_targets.push(target_window);
        }
        
        all_inputs.push(series_inputs);
        all_targets.push(series_targets);
        all_metadata.push(TimeSeriesMetadata {
            id: series.series_id.clone(),
            frequency: series.frequency.clone(),
            seasonal_periods: vec![24, 168],
            scale: Some(1.0),
        });
    }
    
    TrainingData {
        inputs: all_inputs,
        targets: all_targets,
        exogenous: None,
        static_features: None,
        metadata: all_metadata,
    }
}

/// Helper function to estimate memory usage
fn estimate_memory_usage<M: BaseModel<f64>>(model: &M, data_size: usize) -> f64 {
    let param_memory = model.count_parameters() * std::mem::size_of::<f64>();
    let data_memory = data_size * std::mem::size_of::<f64>() * 10; // Rough estimate
    let overhead = (param_memory + data_memory) / 10; // 10% overhead
    
    (param_memory + data_memory + overhead) as f64 / 1_048_576.0
}

/// Helper function to estimate dataset memory
fn estimate_dataset_memory(dataset: &TimeSeriesDataset<f64>) -> f64 {
    let mut total_bytes = 0usize;
    
    for series in &dataset.series {
        // Values
        total_bytes += series.len() * std::mem::size_of::<f64>();
        // Timestamps
        total_bytes += series.len() * std::mem::size_of::<DateTime<Utc>>();
        // Metadata and overhead
        total_bytes += 1000; // Rough estimate
    }
    
    total_bytes as f64 / 1_048_576.0
}