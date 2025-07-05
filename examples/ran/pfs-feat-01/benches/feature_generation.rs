use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use chrono::{Duration, Utc};
use polars::prelude::*;
use std::path::Path;
use tempfile::TempDir;
use tokio::runtime::Runtime;

use pfs_feat_01::config::*;
use pfs_feat_01::*;

/// Benchmark feature generation performance
fn bench_feature_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("feature_generation");
    
    // Test different data sizes
    for size in [100, 500, 1000, 2000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("single_series", size),
            size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let (agent, input_path, output_path, config) = setup_benchmark_data(size).await;
                    
                    let result = agent.generate_features(
                        "bench_series",
                        &input_path,
                        &output_path,
                        &config.default_features,
                    ).await.unwrap();
                    
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark batch processing
fn bench_batch_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("batch_processing");
    
    // Test different batch sizes
    for batch_size in [5, 10, 20].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("batch_series", batch_size),
            batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter(|| async {
                    let (agent, input_dir, output_dir, config, time_series_ids) = 
                        setup_batch_benchmark_data(batch_size, 200).await;
                    
                    let result = agent.generate_batch_features(
                        &time_series_ids,
                        &input_dir,
                        &output_dir,
                        &config.default_features,
                        4, // max parallel jobs
                    ).await.unwrap();
                    
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark different feature types
fn bench_feature_types(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("feature_types");
    
    let size = 1000;
    
    // Benchmark lag features only
    group.bench_function("lag_features_only", |b| {
        b.to_async(&rt).iter(|| async {
            let (agent, input_path, output_path, _) = setup_benchmark_data(size).await;
            
            let config = FeatureConfig {
                lag_features: LagFeatureConfig {
                    enabled: true,
                    lag_periods: vec![1, 2, 3, 6, 12, 24],
                    target_columns: vec!["kpi_value".to_string()],
                },
                rolling_window: RollingWindowConfig { enabled: false, ..Default::default() },
                time_features: TimeBasedFeatureConfig { enabled: false, ..Default::default() },
                output: OutputConfig::default(),
            };
            
            let result = agent.generate_features(
                "lag_bench",
                &input_path,
                &output_path,
                &config,
            ).await.unwrap();
            
            black_box(result);
        });
    });
    
    // Benchmark rolling window features only
    group.bench_function("rolling_window_features_only", |b| {
        b.to_async(&rt).iter(|| async {
            let (agent, input_path, output_path, _) = setup_benchmark_data(size).await;
            
            let config = FeatureConfig {
                lag_features: LagFeatureConfig { enabled: false, ..Default::default() },
                rolling_window: RollingWindowConfig {
                    enabled: true,
                    window_sizes: vec![3, 6, 12, 24],
                    statistics: vec!["mean".to_string(), "std".to_string(), "min".to_string(), "max".to_string()],
                    target_columns: vec!["kpi_value".to_string()],
                },
                time_features: TimeBasedFeatureConfig { enabled: false, ..Default::default() },
                output: OutputConfig::default(),
            };
            
            let result = agent.generate_features(
                "rolling_bench",
                &input_path,
                &output_path,
                &config,
            ).await.unwrap();
            
            black_box(result);
        });
    });
    
    // Benchmark time-based features only
    group.bench_function("time_features_only", |b| {
        b.to_async(&rt).iter(|| async {
            let (agent, input_path, output_path, _) = setup_benchmark_data(size).await;
            
            let config = FeatureConfig {
                lag_features: LagFeatureConfig { enabled: false, ..Default::default() },
                rolling_window: RollingWindowConfig { enabled: false, ..Default::default() },
                time_features: TimeBasedFeatureConfig {
                    enabled: true,
                    features: vec![
                        "hour_of_day".to_string(),
                        "day_of_week".to_string(),
                        "is_weekend".to_string(),
                        "is_business_hour".to_string(),
                    ],
                    timestamp_column: "timestamp".to_string(),
                    timezone: "UTC".to_string(),
                },
                output: OutputConfig::default(),
            };
            
            let result = agent.generate_features(
                "time_bench",
                &input_path,
                &output_path,
                &config,
            ).await.unwrap();
            
            black_box(result);
        });
    });
    
    group.finish();
}

/// Benchmark RAN-specific configurations
fn bench_ran_configurations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("ran_configurations");
    
    let size = 1000;
    
    // Benchmark RAN KPI features
    group.bench_function("ran_kpi_features", |b| {
        b.to_async(&rt).iter(|| async {
            let (agent, input_path, output_path, _) = setup_comprehensive_benchmark_data(size).await;
            
            let config = RanFeatureTemplates::ran_kpi_features();
            
            let result = agent.generate_features(
                "ran_kpi_bench",
                &input_path,
                &output_path,
                &config,
            ).await.unwrap();
            
            black_box(result);
        });
    });
    
    // Benchmark handover prediction features
    group.bench_function("handover_prediction_features", |b| {
        b.to_async(&rt).iter(|| async {
            let (agent, input_path, output_path, _) = setup_handover_benchmark_data(size).await;
            
            let config = RanFeatureTemplates::handover_prediction_features();
            
            let result = agent.generate_features(
                "handover_bench",
                &input_path,
                &output_path,
                &config,
            ).await.unwrap();
            
            black_box(result);
        });
    });
    
    // Benchmark interference detection features
    group.bench_function("interference_detection_features", |b| {
        b.to_async(&rt).iter(|| async {
            let (agent, input_path, output_path, _) = setup_interference_benchmark_data(size).await;
            
            let config = RanFeatureTemplates::interference_detection_features();
            
            let result = agent.generate_features(
                "interference_bench",
                &input_path,
                &output_path,
                &config,
            ).await.unwrap();
            
            black_box(result);
        });
    });
    
    group.finish();
}

/// Benchmark memory usage with different configurations
fn bench_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_usage");
    
    // Test memory usage with large datasets
    for size in [5000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("large_dataset", size),
            size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let (agent, input_path, output_path, config) = setup_benchmark_data(size).await;
                    
                    let result = agent.generate_features(
                        "memory_bench",
                        &input_path,
                        &output_path,
                        &config.default_features,
                    ).await.unwrap();
                    
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

// Helper functions for benchmark setup

async fn setup_benchmark_data(size: usize) -> (FeatureEngineeringAgent, std::path::PathBuf, std::path::PathBuf, FeatureEngineConfig) {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("bench_input.parquet");
    let output_path = temp_dir.path().join("bench_output.parquet");
    
    // Create test data
    let test_data = create_benchmark_ran_data(size);
    save_benchmark_data(&test_data, &input_path);
    
    // Create agent
    let config = create_benchmark_config();
    let agent = FeatureEngineeringAgent::new(config.clone());
    
    // Make paths static for the benchmark
    let static_input = input_path.clone();
    let static_output = output_path.clone();
    
    // Keep temp_dir alive by leaking it (for benchmark purposes only)
    std::mem::forget(temp_dir);
    
    (agent, static_input, static_output, config)
}

async fn setup_batch_benchmark_data(
    batch_size: usize, 
    rows_per_series: usize
) -> (FeatureEngineeringAgent, std::path::PathBuf, std::path::PathBuf, FeatureEngineConfig, Vec<String>) {
    let temp_dir = TempDir::new().unwrap();
    let input_dir = temp_dir.path().join("batch_input");
    let output_dir = temp_dir.path().join("batch_output");
    
    std::fs::create_dir_all(&input_dir).unwrap();
    std::fs::create_dir_all(&output_dir).unwrap();
    
    let mut time_series_ids = Vec::new();
    
    for i in 0..batch_size {
        let ts_id = format!("bench_series_{:03}", i);
        let test_data = create_benchmark_ran_data(rows_per_series);
        let input_path = input_dir.join(format!("{}.parquet", ts_id));
        
        save_benchmark_data(&test_data, &input_path);
        time_series_ids.push(ts_id);
    }
    
    let config = create_benchmark_config();
    let agent = FeatureEngineeringAgent::new(config.clone());
    
    // Keep temp_dir alive
    std::mem::forget(temp_dir);
    
    (agent, input_dir, output_dir, config, time_series_ids)
}

async fn setup_comprehensive_benchmark_data(size: usize) -> (FeatureEngineeringAgent, std::path::PathBuf, std::path::PathBuf, FeatureEngineConfig) {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("comprehensive_input.parquet");
    let output_path = temp_dir.path().join("comprehensive_output.parquet");
    
    let test_data = create_comprehensive_ran_data(size);
    save_benchmark_data(&test_data, &input_path);
    
    let config = create_benchmark_config();
    let agent = FeatureEngineeringAgent::new(config.clone());
    
    std::mem::forget(temp_dir);
    
    (agent, input_path, output_path, config)
}

async fn setup_handover_benchmark_data(size: usize) -> (FeatureEngineeringAgent, std::path::PathBuf, std::path::PathBuf, FeatureEngineConfig) {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("handover_input.parquet");
    let output_path = temp_dir.path().join("handover_output.parquet");
    
    let test_data = create_handover_data(size);
    save_benchmark_data(&test_data, &input_path);
    
    let config = create_benchmark_config();
    let agent = FeatureEngineeringAgent::new(config.clone());
    
    std::mem::forget(temp_dir);
    
    (agent, input_path, output_path, config)
}

async fn setup_interference_benchmark_data(size: usize) -> (FeatureEngineeringAgent, std::path::PathBuf, std::path::PathBuf, FeatureEngineConfig) {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("interference_input.parquet");
    let output_path = temp_dir.path().join("interference_output.parquet");
    
    let test_data = create_interference_data(size);
    save_benchmark_data(&test_data, &input_path);
    
    let config = create_benchmark_config();
    let agent = FeatureEngineeringAgent::new(config.clone());
    
    std::mem::forget(temp_dir);
    
    (agent, input_path, output_path, config)
}

fn create_benchmark_ran_data(size: usize) -> DataFrame {
    let start_time = Utc::now() - Duration::hours(size as i64);
    
    let mut timestamps = Vec::new();
    let mut kpi_values = Vec::new();
    let mut cell_ids = Vec::new();

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for i in 0..size {
        timestamps.push(start_time + Duration::hours(i as i64));
        kpi_values.push(rng.gen_range(0.0..100.0));
        cell_ids.push("BENCH_CELL".to_string());
    }

    df! {
        "timestamp" => timestamps,
        "kpi_value" => kpi_values,
        "cell_id" => cell_ids,
    }.unwrap()
}

fn create_comprehensive_ran_data(size: usize) -> DataFrame {
    let start_time = Utc::now() - Duration::hours(size as i64);
    
    let mut timestamps = Vec::new();
    let mut kpi_values = Vec::new();
    let mut prb_utilization_dl = Vec::new();
    let mut prb_utilization_ul = Vec::new();
    let mut active_users = Vec::new();
    let mut throughput_dl = Vec::new();
    let mut throughput_ul = Vec::new();
    let mut rsrp_avg = Vec::new();
    let mut sinr_avg = Vec::new();
    let mut cell_ids = Vec::new();

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for i in 0..size {
        timestamps.push(start_time + Duration::hours(i as i64));
        
        let base_value = rng.gen_range(0.0..100.0);
        kpi_values.push(base_value);
        prb_utilization_dl.push(base_value);
        prb_utilization_ul.push(base_value * 0.5);
        active_users.push(rng.gen_range(10..200));
        throughput_dl.push(rng.gen_range(50.0..300.0));
        throughput_ul.push(rng.gen_range(10.0..100.0));
        rsrp_avg.push(rng.gen_range(-120.0..-60.0));
        sinr_avg.push(rng.gen_range(-5.0..30.0));
        cell_ids.push("BENCH_CELL".to_string());
    }

    df! {
        "timestamp" => timestamps,
        "kpi_value" => kpi_values,
        "prb_utilization_dl" => prb_utilization_dl,
        "prb_utilization_ul" => prb_utilization_ul,
        "active_users" => active_users,
        "throughput_dl" => throughput_dl,
        "throughput_ul" => throughput_ul,
        "rsrp_avg" => rsrp_avg,
        "sinr_avg" => sinr_avg,
        "cell_id" => cell_ids,
    }.unwrap()
}

fn create_handover_data(size: usize) -> DataFrame {
    let start_time = Utc::now() - Duration::minutes(size as i64);
    
    let mut timestamps = Vec::new();
    let mut kpi_values = Vec::new();
    let mut serving_rsrp = Vec::new();
    let mut serving_sinr = Vec::new();
    let mut neighbor_rsrp_best = Vec::new();
    let mut ue_speed_kmh = Vec::new();
    let mut cqi = Vec::new();

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for i in 0..size {
        timestamps.push(start_time + Duration::minutes(i as i64));
        
        let rsrp = rng.gen_range(-120.0..-60.0);
        kpi_values.push((rsrp + 120.0) / 60.0 * 100.0); // Normalized to 0-100
        serving_rsrp.push(rsrp);
        serving_sinr.push(rng.gen_range(-5.0..30.0));
        neighbor_rsrp_best.push(rng.gen_range(-130.0..-70.0));
        ue_speed_kmh.push(rng.gen_range(0.0..120.0));
        cqi.push(rng.gen_range(1..15));
    }

    df! {
        "timestamp" => timestamps,
        "kpi_value" => kpi_values,
        "serving_rsrp" => serving_rsrp,
        "serving_sinr" => serving_sinr,
        "neighbor_rsrp_best" => neighbor_rsrp_best,
        "ue_speed_kmh" => ue_speed_kmh,
        "cqi" => cqi,
    }.unwrap()
}

fn create_interference_data(size: usize) -> DataFrame {
    let start_time = Utc::now() - Duration::hours(size as i64);
    
    let mut timestamps = Vec::new();
    let mut kpi_values = Vec::new();
    let mut noise_floor_pusch = Vec::new();
    let mut noise_floor_pucch = Vec::new();
    let mut prb_utilization_ul = Vec::new();
    let mut sinr_avg = Vec::new();
    let mut bler_ul = Vec::new();

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for i in 0..size {
        timestamps.push(start_time + Duration::hours(i as i64));
        
        let noise = rng.gen_range(-110.0..-90.0);
        kpi_values.push((noise + 110.0) / 20.0 * 100.0); // Normalized to 0-100
        noise_floor_pusch.push(noise);
        noise_floor_pucch.push(noise + rng.gen_range(-2.0..2.0));
        prb_utilization_ul.push(rng.gen_range(0.0..80.0));
        sinr_avg.push(rng.gen_range(-5.0..25.0));
        bler_ul.push(rng.gen_range(0.0..10.0));
    }

    df! {
        "timestamp" => timestamps,
        "kpi_value" => kpi_values,
        "noise_floor_pusch" => noise_floor_pusch,
        "noise_floor_pucch" => noise_floor_pucch,
        "prb_utilization_ul" => prb_utilization_ul,
        "sinr_avg" => sinr_avg,
        "bler_ul" => bler_ul,
    }.unwrap()
}

fn create_benchmark_config() -> FeatureEngineConfig {
    let mut config = FeatureEngineConfig::default();
    config.processing.max_parallel_jobs = 4;
    config.processing.batch_size = 1000;
    config
}

fn save_benchmark_data(df: &DataFrame, path: &Path) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    
    let mut writer = ParquetWriter::new(std::fs::File::create(path).unwrap());
    writer.finish(df).unwrap();
}

criterion_group!(
    benches,
    bench_feature_generation,
    bench_batch_processing,
    bench_feature_types,
    bench_ran_configurations,
    bench_memory_usage
);
criterion_main!(benches);