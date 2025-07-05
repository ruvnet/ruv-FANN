//! Performance benchmarks for the data ingestion service

use std::path::PathBuf;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tempfile::tempdir;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

use pfs_data_01::config::IngestionConfig;
use pfs_data_01::ingestion::IngestionEngine;

/// Benchmark CSV file processing
async fn benchmark_csv_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("csv_processing");
    
    // Test different file sizes
    let file_sizes = vec![1000, 10000, 100000, 1000000];
    
    for size in file_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("process_csv", size),
            &size,
            |b, &size| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(rt).iter(|| async {
                    let temp_dir = tempdir().unwrap();
                    let input_dir = temp_dir.path().join("input");
                    let output_dir = temp_dir.path().join("output");
                    
                    tokio::fs::create_dir_all(&input_dir).await.unwrap();
                    tokio::fs::create_dir_all(&output_dir).await.unwrap();
                    
                    // Generate test file
                    let test_file = input_dir.join("test.csv");
                    generate_csv_file(&test_file, size).await.unwrap();
                    
                    // Configure for performance
                    let config = IngestionConfig::optimized_for_performance();
                    let engine = IngestionEngine::new(config);
                    
                    // Benchmark processing
                    let result = engine.process_directory(&input_dir, &output_dir).await.unwrap();
                    
                    black_box(result);
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark JSON file processing
async fn benchmark_json_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_processing");
    
    let file_sizes = vec![1000, 10000, 100000];
    
    for size in file_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("process_json", size),
            &size,
            |b, &size| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(rt).iter(|| async {
                    let temp_dir = tempdir().unwrap();
                    let input_dir = temp_dir.path().join("input");
                    let output_dir = temp_dir.path().join("output");
                    
                    tokio::fs::create_dir_all(&input_dir).await.unwrap();
                    tokio::fs::create_dir_all(&output_dir).await.unwrap();
                    
                    // Generate test file
                    let test_file = input_dir.join("test.json");
                    generate_json_file(&test_file, size).await.unwrap();
                    
                    // Configure for performance
                    let config = IngestionConfig::optimized_for_performance();
                    let engine = IngestionEngine::new(config);
                    
                    // Benchmark processing
                    let result = engine.process_directory(&input_dir, &output_dir).await.unwrap();
                    
                    black_box(result);
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark concurrent file processing
async fn benchmark_concurrent_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_processing");
    group.sample_size(10); // Fewer samples for slower benchmarks
    group.measurement_time(Duration::from_secs(30));
    
    let concurrency_levels = vec![1, 2, 4, 8];
    let files_per_level = 20;
    let rows_per_file = 10000;
    
    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::new("concurrent_files", concurrency),
            &concurrency,
            |b, &concurrency| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(rt).iter(|| async {
                    let temp_dir = tempdir().unwrap();
                    let input_dir = temp_dir.path().join("input");
                    let output_dir = temp_dir.path().join("output");
                    
                    tokio::fs::create_dir_all(&input_dir).await.unwrap();
                    tokio::fs::create_dir_all(&output_dir).await.unwrap();
                    
                    // Generate multiple test files
                    for i in 0..files_per_level {
                        let test_file = input_dir.join(format!("test_{}.csv", i));
                        generate_csv_file(&test_file, rows_per_file).await.unwrap();
                    }
                    
                    // Configure with specific concurrency
                    let mut config = IngestionConfig::optimized_for_performance();
                    config.max_concurrent_files = concurrency;
                    let engine = IngestionEngine::new(config);
                    
                    // Benchmark processing
                    let result = engine.process_directory(&input_dir, &output_dir).await.unwrap();
                    
                    black_box(result);
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark different compression codecs
async fn benchmark_compression_codecs(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_codecs");
    group.sample_size(10);
    
    let codecs = vec!["snappy", "gzip", "lz4", "brotli"];
    let file_size = 50000;
    
    for codec in codecs {
        group.bench_with_input(
            BenchmarkId::new("compression", codec),
            codec,
            |b, codec| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(rt).iter(|| async {
                    let temp_dir = tempdir().unwrap();
                    let input_dir = temp_dir.path().join("input");
                    let output_dir = temp_dir.path().join("output");
                    
                    tokio::fs::create_dir_all(&input_dir).await.unwrap();
                    tokio::fs::create_dir_all(&output_dir).await.unwrap();
                    
                    // Generate test file
                    let test_file = input_dir.join("test.csv");
                    generate_csv_file(&test_file, file_size).await.unwrap();
                    
                    // Configure with specific compression
                    let mut config = IngestionConfig::optimized_for_performance();
                    config.compression_codec = codec.to_string();
                    let engine = IngestionEngine::new(config);
                    
                    // Benchmark processing
                    let result = engine.process_directory(&input_dir, &output_dir).await.unwrap();
                    
                    black_box(result);
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark batch size impact
async fn benchmark_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_sizes");
    
    let batch_sizes = vec![1000, 5000, 10000, 25000, 50000];
    let file_size = 100000;
    
    for batch_size in batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &batch_size| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(rt).iter(|| async {
                    let temp_dir = tempdir().unwrap();
                    let input_dir = temp_dir.path().join("input");
                    let output_dir = temp_dir.path().join("output");
                    
                    tokio::fs::create_dir_all(&input_dir).await.unwrap();
                    tokio::fs::create_dir_all(&output_dir).await.unwrap();
                    
                    // Generate test file
                    let test_file = input_dir.join("test.csv");
                    generate_csv_file(&test_file, file_size).await.unwrap();
                    
                    // Configure with specific batch size
                    let mut config = IngestionConfig::optimized_for_performance();
                    config.batch_size = batch_size;
                    let engine = IngestionEngine::new(config);
                    
                    // Benchmark processing
                    let result = engine.process_directory(&input_dir, &output_dir).await.unwrap();
                    
                    black_box(result);
                });
            }
        );
    }
    
    group.finish();
}

/// Generate a CSV test file with specified number of rows
async fn generate_csv_file(file_path: &PathBuf, num_rows: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(file_path).await?;
    
    // Write header
    file.write_all(b"timestamp,cell_id,kpi_name,kpi_value,ue_id,sector_id\n").await?;
    
    // Write data rows with realistic RAN data patterns
    for i in 0..num_rows {
        let timestamp = chrono::Utc::now() - chrono::Duration::seconds((num_rows - i) as i64);
        let cell_id = format!("cell_{:04}", i % 500);
        let sector_id = format!("sector_{:03}", i % 100);
        let kpi_name = match i % 8 {
            0 => "throughput_dl",
            1 => "throughput_ul",
            2 => "prb_utilization_dl",
            3 => "prb_utilization_ul",
            4 => "active_users",
            5 => "rsrp_avg",
            6 => "sinr_avg",
            _ => "latency_avg",
        };
        let kpi_value = match kpi_name {
            "throughput_dl" | "throughput_ul" => (i as f64 * 0.5 + 10.0) % 150.0,
            "prb_utilization_dl" | "prb_utilization_ul" => (i as f64 * 0.1) % 100.0,
            "active_users" => (i % 200) as f64,
            "rsrp_avg" => -120.0 + (i as f64 * 0.01) % 40.0,
            "sinr_avg" => (i as f64 * 0.02) % 30.0,
            "latency_avg" => 1.0 + (i as f64 * 0.001) % 50.0,
            _ => (i as f64 * 1.5) % 100.0,
        };
        let ue_id = format!("ue_{:08}", i % 10000);
        
        let line = format!(
            "{},{},{},{:.3},{},{}\n",
            timestamp.format("%Y-%m-%d %H:%M:%S%.3f"),
            cell_id,
            kpi_name,
            kpi_value,
            ue_id,
            sector_id
        );
        
        file.write_all(line.as_bytes()).await?;
    }
    
    file.flush().await?;
    Ok(())
}

/// Generate a JSON test file with specified number of rows
async fn generate_json_file(file_path: &PathBuf, num_rows: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(file_path).await?;
    
    // Write JSON Lines format
    for i in 0..num_rows {
        let timestamp = chrono::Utc::now() - chrono::Duration::seconds((num_rows - i) as i64);
        let cell_id = format!("cell_{:04}", i % 500);
        let sector_id = format!("sector_{:03}", i % 100);
        let kpi_name = match i % 4 {
            0 => "throughput_dl",
            1 => "prb_utilization_dl",
            2 => "active_users",
            _ => "rsrp_avg",
        };
        let kpi_value = (i as f64 * 1.5 + 10.0) % 100.0;
        let ue_id = format!("ue_{:08}", i % 10000);
        
        let json_obj = serde_json::json!({
            "timestamp": timestamp.format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
            "cell_id": cell_id,
            "kpi_name": kpi_name,
            "kpi_value": kpi_value,
            "ue_id": ue_id,
            "sector_id": sector_id
        });
        
        let line = format!("{}\n", json_obj);
        file.write_all(line.as_bytes()).await?;
    }
    
    file.flush().await?;
    Ok(())
}

/// Wrapper function to run async benchmarks
fn run_async_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    rt.block_on(async {
        benchmark_csv_processing(c).await;
        benchmark_json_processing(c).await;
        benchmark_concurrent_processing(c).await;
        benchmark_compression_codecs(c).await;
        benchmark_batch_sizes(c).await;
    });
}

criterion_group!(benches, run_async_benchmark);
criterion_main!(benches);