//! Integration tests for the PFS-DATA-01 service

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use tempfile::tempdir;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::time::timeout;

use pfs_data_01::config::IngestionConfig;
use pfs_data_01::ingestion::IngestionEngine;
use pfs_data_01::monitoring::IngestionMonitor;
use pfs_data_01::schema::StandardSchema;
use pfs_data_01::storage::ParquetWriter;
use pfs_data_01::watcher::{DirectoryWatcher, WatchConfig};

#[tokio::test]
async fn test_complete_csv_ingestion_workflow() {
    let temp_dir = tempdir().unwrap();
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    
    tokio::fs::create_dir_all(&input_dir).await.unwrap();
    tokio::fs::create_dir_all(&output_dir).await.unwrap();
    
    // Create test CSV file
    let csv_content = r#"timestamp,cell_id,kpi_name,kpi_value,ue_id
2024-01-01 10:00:00.000,cell_001,throughput_dl,45.5,ue_12345
2024-01-01 10:00:01.000,cell_001,throughput_ul,12.3,ue_12345
2024-01-01 10:00:02.000,cell_002,prb_utilization_dl,67.8,ue_12346
2024-01-01 10:00:03.000,cell_002,active_users,15,ue_12347
2024-01-01 10:00:04.000,cell_003,rsrp_avg,-95.2,ue_12348"#;
    
    let test_file = input_dir.join("test_data.csv");
    tokio::fs::write(&test_file, csv_content).await.unwrap();
    
    // Configure and run ingestion
    let config = IngestionConfig::default();
    let engine = IngestionEngine::new(config);
    
    let result = engine.process_directory(&input_dir, &output_dir).await.unwrap();
    
    // Verify results
    assert_eq!(result.files_processed, 1);
    assert_eq!(result.files_failed, 0);
    assert_eq!(result.rows_processed, 5);
    assert_eq!(result.rows_failed, 0);
    assert!(result.error_rate() < 0.01);
    
    // Verify output file exists
    let output_files: Vec<_> = tokio::fs::read_dir(&output_dir)
        .await
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .await
        .unwrap();
    
    assert_eq!(output_files.len(), 1);
    assert!(output_files[0].file_name().to_string_lossy().ends_with(".parquet"));
}

#[tokio::test]
async fn test_complete_json_ingestion_workflow() {
    let temp_dir = tempdir().unwrap();
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    
    tokio::fs::create_dir_all(&input_dir).await.unwrap();
    tokio::fs::create_dir_all(&output_dir).await.unwrap();
    
    // Create test JSON Lines file
    let json_content = r#"{"timestamp": "2024-01-01 10:00:00.000", "cell_id": "cell_001", "kpi_name": "throughput_dl", "kpi_value": 45.5, "ue_id": "ue_12345"}
{"timestamp": "2024-01-01 10:00:01.000", "cell_id": "cell_001", "kpi_name": "throughput_ul", "kpi_value": 12.3, "ue_id": "ue_12345"}
{"timestamp": "2024-01-01 10:00:02.000", "cell_id": "cell_002", "kpi_name": "prb_utilization_dl", "kpi_value": 67.8, "ue_id": "ue_12346"}"#;
    
    let test_file = input_dir.join("test_data.json");
    tokio::fs::write(&test_file, json_content).await.unwrap();
    
    // Configure and run ingestion
    let config = IngestionConfig::default();
    let engine = IngestionEngine::new(config);
    
    let result = engine.process_directory(&input_dir, &output_dir).await.unwrap();
    
    // Verify results
    assert_eq!(result.files_processed, 1);
    assert_eq!(result.files_failed, 0);
    assert_eq!(result.rows_processed, 3);
    assert_eq!(result.rows_failed, 0);
    
    // Verify output file exists
    let output_files: Vec<_> = tokio::fs::read_dir(&output_dir)
        .await
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .await
        .unwrap();
    
    assert_eq!(output_files.len(), 1);
    assert!(output_files[0].file_name().to_string_lossy().ends_with(".parquet"));
}

#[tokio::test]
async fn test_error_handling_malformed_data() {
    let temp_dir = tempdir().unwrap();
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    
    tokio::fs::create_dir_all(&input_dir).await.unwrap();
    tokio::fs::create_dir_all(&output_dir).await.unwrap();
    
    // Create CSV with malformed data
    let csv_content = r#"timestamp,cell_id,kpi_name,kpi_value,ue_id
2024-01-01 10:00:00.000,cell_001,throughput_dl,45.5,ue_12345
invalid_timestamp,cell_001,throughput_ul,12.3,ue_12345
2024-01-01 10:00:02.000,cell_002,prb_utilization_dl,invalid_value,ue_12346
2024-01-01 10:00:03.000,cell_003,active_users,25,ue_12347"#;
    
    let test_file = input_dir.join("test_malformed.csv");
    tokio::fs::write(&test_file, csv_content).await.unwrap();
    
    // Configure to skip malformed rows
    let mut config = IngestionConfig::default();
    config.skip_malformed_rows = true;
    config.max_error_rate = 0.5; // Allow higher error rate for this test
    
    let engine = IngestionEngine::new(config);
    let result = engine.process_directory(&input_dir, &output_dir).await.unwrap();
    
    // Should process valid rows and skip malformed ones
    assert_eq!(result.files_processed, 1);
    assert!(result.rows_processed >= 2); // At least the first and last rows should be processed
    assert!(result.rows_failed > 0); // Some rows should fail
    assert!(result.error_rate() > 0.0);
}

#[tokio::test]
async fn test_large_file_processing() {
    let temp_dir = tempdir().unwrap();
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    
    tokio::fs::create_dir_all(&input_dir).await.unwrap();
    tokio::fs::create_dir_all(&output_dir).await.unwrap();
    
    // Generate a larger test file
    let test_file = input_dir.join("large_test.csv");
    generate_large_csv_file(&test_file, 50000).await.unwrap();
    
    // Configure for performance
    let config = IngestionConfig::optimized_for_performance();
    let engine = IngestionEngine::new(config);
    
    let start_time = std::time::Instant::now();
    let result = engine.process_directory(&input_dir, &output_dir).await.unwrap();
    let processing_time = start_time.elapsed();
    
    // Verify performance requirements
    assert_eq!(result.files_processed, 1);
    assert!(result.rows_processed >= 49000); // Allow for some data quality issues
    assert!(result.error_rate() < 0.01); // Less than 1% error rate
    assert!(result.throughput_mb_per_second() > 1.0); // At least 1 MB/s throughput
    assert!(processing_time < Duration::from_secs(60)); // Complete within 60 seconds
    
    println!("Large file processing stats:");
    println!("  Rows processed: {}", result.rows_processed);
    println!("  Processing time: {:?}", processing_time);
    println!("  Throughput: {:.2} MB/s", result.throughput_mb_per_second());
    println!("  Error rate: {:.4}%", result.error_rate() * 100.0);
}

#[tokio::test]
async fn test_concurrent_file_processing() {
    let temp_dir = tempdir().unwrap();
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    
    tokio::fs::create_dir_all(&input_dir).await.unwrap();
    tokio::fs::create_dir_all(&output_dir).await.unwrap();
    
    // Generate multiple test files
    let num_files = 10;
    let rows_per_file = 1000;
    
    for i in 0..num_files {
        let test_file = input_dir.join(format!("test_{:03}.csv", i));
        generate_test_csv_file(&test_file, rows_per_file).await.unwrap();
    }
    
    // Configure for concurrent processing
    let mut config = IngestionConfig::optimized_for_performance();
    config.max_concurrent_files = 4;
    
    let engine = IngestionEngine::new(config);
    
    let start_time = std::time::Instant::now();
    let result = engine.process_directory(&input_dir, &output_dir).await.unwrap();
    let processing_time = start_time.elapsed();
    
    // Verify concurrent processing worked
    assert_eq!(result.files_processed, num_files);
    assert!(result.rows_processed >= (num_files * rows_per_file * 95 / 100) as u64); // 95% success rate
    assert!(result.error_rate() < 0.01);
    
    // Should be faster than sequential processing
    let expected_sequential_time = Duration::from_millis(num_files as u64 * 100); // Rough estimate
    assert!(processing_time < expected_sequential_time);
    
    println!("Concurrent processing stats:");
    println!("  Files processed: {}", result.files_processed);
    println!("  Processing time: {:?}", processing_time);
    println!("  Throughput: {:.2} MB/s", result.throughput_mb_per_second());
}

#[tokio::test]
async fn test_schema_validation() {
    let temp_dir = tempdir().unwrap();
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    
    tokio::fs::create_dir_all(&input_dir).await.unwrap();
    tokio::fs::create_dir_all(&output_dir).await.unwrap();
    
    // Create CSV with missing required columns
    let csv_content = r#"timestamp,cell_id,value
2024-01-01 10:00:00.000,cell_001,45.5
2024-01-01 10:00:01.000,cell_002,12.3"#;
    
    let test_file = input_dir.join("test_invalid_schema.csv");
    tokio::fs::write(&test_file, csv_content).await.unwrap();
    
    // Use strict schema validation
    let mut config = IngestionConfig::default();
    config.skip_malformed_rows = false;
    
    let engine = IngestionEngine::new(config);
    
    // Should fail due to missing required columns
    let result = engine.process_directory(&input_dir, &output_dir).await;
    assert!(result.is_err() || result.unwrap().error_rate() > 0.5);
}

#[tokio::test]
async fn test_compression_formats() {
    let temp_dir = tempdir().unwrap();
    let input_dir = temp_dir.path().join("input");
    
    tokio::fs::create_dir_all(&input_dir).await.unwrap();
    
    // Generate test file
    let test_file = input_dir.join("test.csv");
    generate_test_csv_file(&test_file, 5000).await.unwrap();
    
    let compression_codecs = vec!["snappy", "gzip", "lz4"];
    
    for codec in compression_codecs {
        let output_dir = temp_dir.path().join(format!("output_{}", codec));
        tokio::fs::create_dir_all(&output_dir).await.unwrap();
        
        // Configure with specific compression
        let mut config = IngestionConfig::default();
        config.compression_codec = codec.to_string();
        
        let engine = IngestionEngine::new(config);
        let result = engine.process_directory(&input_dir, &output_dir).await.unwrap();
        
        // Verify successful processing
        assert_eq!(result.files_processed, 1);
        assert!(result.rows_processed > 0);
        assert!(result.error_rate() < 0.01);
        
        // Verify output file exists
        let output_files: Vec<_> = tokio::fs::read_dir(&output_dir)
            .await
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .await
            .unwrap();
        
        assert_eq!(output_files.len(), 1);
        
        println!("Compression {} - Compression ratio: {:.2}x", 
                 codec, 1.0 / result.compression_ratio());
    }
}

#[tokio::test]
async fn test_parquet_output_validation() {
    let temp_dir = tempdir().unwrap();
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    
    tokio::fs::create_dir_all(&input_dir).await.unwrap();
    tokio::fs::create_dir_all(&output_dir).await.unwrap();
    
    // Generate test file
    let test_file = input_dir.join("test.csv");
    generate_test_csv_file(&test_file, 1000).await.unwrap();
    
    let config = IngestionConfig::default();
    let engine = IngestionEngine::new(config.clone());
    
    let result = engine.process_directory(&input_dir, &output_dir).await.unwrap();
    assert_eq!(result.files_processed, 1);
    
    // Validate the Parquet file
    let parquet_writer = ParquetWriter::new(std::sync::Arc::new(config));
    let output_files: Vec<_> = tokio::fs::read_dir(&output_dir)
        .await
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .await
        .unwrap();
    
    assert_eq!(output_files.len(), 1);
    let parquet_file = output_files[0].path();
    
    // Read and validate metadata
    let validation_result = parquet_writer.validate_file(&parquet_file).await.unwrap();
    assert!(validation_result.is_valid);
    assert!(!validation_result.has_errors());
    assert!(validation_result.row_count > 0);
    assert!(validation_result.file_size_bytes > 0);
    
    println!("Parquet validation:");
    println!("  Rows: {}", validation_result.row_count);
    println!("  Size: {} bytes", validation_result.file_size_bytes);
    println!("  Compression ratio: {:.2}", validation_result.compression_ratio);
}

#[tokio::test]
async fn test_monitoring_and_metrics() {
    let config = IngestionConfig::default();
    let monitor = IngestionMonitor::new(config);
    
    // Test recording file processing
    let job_id = "test_job";
    let file_path = "test.csv";
    
    monitor.record_file_start(job_id, file_path, 1024).await;
    
    let metrics_before = monitor.get_metrics().await;
    assert_eq!(metrics_before.active_files, 1);
    assert_eq!(metrics_before.total_input_size, 1024);
    
    monitor.record_file_completion(
        job_id,
        file_path,
        100,
        2,
        Duration::from_millis(500),
        800,
    ).await;
    
    let metrics_after = monitor.get_metrics().await;
    assert_eq!(metrics_after.active_files, 0);
    assert_eq!(metrics_after.files_processed, 1);
    assert_eq!(metrics_after.rows_processed, 100);
    assert_eq!(metrics_after.rows_failed, 2);
    
    // Test job metrics
    let job_metrics = monitor.get_job_metrics(job_id).await.unwrap();
    assert_eq!(job_metrics.files_processed, 1);
    assert_eq!(job_metrics.rows_processed, 100);
    assert_eq!(job_metrics.rows_failed, 2);
    
    // Test performance summary
    let summary = monitor.get_performance_summary().await;
    assert!(summary.error_rate >= 0.0);
    assert!(summary.throughput_mb_per_second >= 0.0);
}

/// Helper function to generate a large CSV test file
async fn generate_large_csv_file(file_path: &PathBuf, num_rows: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(file_path).await?;
    
    // Write header
    file.write_all(b"timestamp,cell_id,kpi_name,kpi_value,ue_id,sector_id\n").await?;
    
    // Write data rows in batches for memory efficiency
    let batch_size = 1000;
    for batch_start in (0..num_rows).step_by(batch_size) {
        let mut batch_content = String::new();
        let batch_end = (batch_start + batch_size).min(num_rows);
        
        for i in batch_start..batch_end {
            let timestamp = chrono::Utc::now() - chrono::Duration::seconds((num_rows - i) as i64);
            let cell_id = format!("cell_{:04}", i % 1000);
            let sector_id = format!("sector_{:03}", i % 200);
            let kpi_name = match i % 6 {
                0 => "throughput_dl",
                1 => "throughput_ul",
                2 => "prb_utilization_dl",
                3 => "active_users",
                4 => "rsrp_avg",
                _ => "sinr_avg",
            };
            let kpi_value = (i as f64 * 1.5 + 10.0) % 150.0;
            let ue_id = format!("ue_{:08}", i % 50000);
            
            batch_content.push_str(&format!(
                "{},{},{},{:.3},{},{}\n",
                timestamp.format("%Y-%m-%d %H:%M:%S%.3f"),
                cell_id,
                kpi_name,
                kpi_value,
                ue_id,
                sector_id
            ));
        }
        
        file.write_all(batch_content.as_bytes()).await?;
    }
    
    file.flush().await?;
    Ok(())
}

/// Helper function to generate a test CSV file
async fn generate_test_csv_file(file_path: &PathBuf, num_rows: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(file_path).await?;
    
    // Write header
    file.write_all(b"timestamp,cell_id,kpi_name,kpi_value,ue_id\n").await?;
    
    // Write data rows
    for i in 0..num_rows {
        let timestamp = chrono::Utc::now() - chrono::Duration::seconds((num_rows - i) as i64);
        let cell_id = format!("cell_{:04}", i % 100);
        let kpi_name = match i % 4 {
            0 => "throughput_dl",
            1 => "throughput_ul",
            2 => "prb_utilization_dl",
            _ => "active_users",
        };
        let kpi_value = (i as f64 * 1.5 + 10.0) % 100.0;
        let ue_id = format!("ue_{:06}", i % 1000);
        
        let line = format!(
            "{},{},{},{:.2},{}\n",
            timestamp.format("%Y-%m-%d %H:%M:%S%.3f"),
            cell_id,
            kpi_name,
            kpi_value,
            ue_id
        );
        
        file.write_all(line.as_bytes()).await?;
    }
    
    file.flush().await?;
    Ok(())
}