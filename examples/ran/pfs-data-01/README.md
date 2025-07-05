# PFS-DATA-01: File-based Data Ingestion Service

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](./LICENSE)

A high-performance batch data ingestion service for the RAN Intelligence Platform, designed to process CSV and JSON files into normalized Parquet format with standardized schema.

## 🎯 Key Features

- **High-Performance Processing**: Concurrent file processing with configurable parallelism
- **Schema Normalization**: Automatic mapping to standardized RAN data schema
- **Error Handling**: Configurable error rate thresholds with detailed error tracking
- **Multiple Formats**: Support for CSV and JSON input formats
- **Parquet Output**: Optimized columnar storage with configurable compression
- **gRPC API**: Complete service interface for integration
- **Real-time Monitoring**: Built-in metrics and progress tracking
- **Directory Watching**: Automatic processing of new files

## 📊 Performance Targets

- **Throughput**: 100GB+ batch processing capability
- **Error Rate**: <0.01% parsing errors
- **Concurrency**: Configurable parallel file processing
- **Compression**: Optimized Parquet output with multiple codec support

## 🚀 Quick Start

### Prerequisites

- Rust 1.70 or later
- Protocol Buffers compiler (`protoc`)

### Installation

```bash
# Clone the repository
git clone https://github.com/ricable/ruv-FANN.git
cd ruv-FANN/examples/ran/pfs-data-01

# Build the service
cargo build --release
```

### Basic Usage

#### Start the gRPC Server

```bash
# Start server with default configuration
./target/release/pfs-data-01 serve

# Start with custom address and reflection
./target/release/pfs-data-01 serve --address 0.0.0.0:8080 --reflection
```

#### Process Files Directly

```bash
# Process a directory of files
./target/release/pfs-data-01 process \
  --input-dir /path/to/csv/files \
  --output-dir /path/to/parquet/output \
  --recursive

# Process with custom patterns
./target/release/pfs-data-01 process \
  --input-dir ./data \
  --output-dir ./processed \
  --patterns "*.csv,*.json" \
  --verbose
```

#### Run Performance Tests

```bash
# Generate and process test data
./target/release/pfs-data-01 test --files 100 --rows 10000 --output-dir ./test_data

# Run benchmarks
cargo bench
```

## 🔧 Configuration

### Configuration File

Create a TOML configuration file:

```toml
# File processing configuration
file_patterns = ["*.csv", "*.json"]
recursive = true
exclude_patterns = ["*.tmp", "*.processing", ".*"]

# Performance configuration
batch_size = 10000
max_concurrent_files = 4
processing_timeout_seconds = 300

# Error handling
max_error_rate = 0.01
skip_malformed_rows = true
max_retries = 3

# Output configuration
compression_codec = "snappy"
row_group_size = 1000000
enable_statistics = true

# Resource limits
max_memory_mb = 8192
max_file_size_mb = 10240

[schema]
timestamp_column = "timestamp"
cell_id_column = "cell_id"
kpi_name_column = "kpi_name"
kpi_value_column = "kpi_value"
ue_id_column = "ue_id"
sector_id_column = "sector_id"
```

### Optimization Presets

```rust
// For maximum performance
let config = IngestionConfig::optimized_for_performance();

// For memory-constrained environments
let config = IngestionConfig::optimized_for_memory();

// For maximum data quality
let config = IngestionConfig::optimized_for_quality();
```

## 📡 gRPC API

### Service Definition

```protobuf
service DataIngestionService {
    rpc StartIngestion(StartIngestionRequest) returns (StartIngestionResponse);
    rpc StopIngestion(StopIngestionRequest) returns (StopIngestionResponse);
    rpc GetIngestionStatus(GetIngestionStatusRequest) returns (GetIngestionStatusResponse);
    rpc GetIngestionMetrics(GetIngestionMetricsRequest) returns (GetIngestionMetricsResponse);
    rpc ListIngestionJobs(ListIngestionJobsRequest) returns (ListIngestionJobsResponse);
    rpc StreamIngestionEvents(StreamIngestionEventsRequest) returns (stream IngestionEvent);
}
```

### Client Example

```rust
use pfs_data_01::proto::data_ingestion_service_client::DataIngestionServiceClient;
use pfs_data_01::proto::StartIngestionRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = DataIngestionServiceClient::connect("http://127.0.0.1:50051").await?;
    
    let request = StartIngestionRequest {
        input_directory: "/path/to/input".to_string(),
        output_directory: "/path/to/output".to_string(),
        config: None, // Use default config
    };
    
    let response = client.start_ingestion(request).await?;
    println!("Job started: {}", response.into_inner().job_id);
    
    Ok(())
}
```

## 📋 Data Schema

### Standard RAN Data Schema

The service normalizes all input data to a standard schema:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `timestamp` | DateTime | Yes | Measurement timestamp |
| `cell_id` | String | Yes | Cell identifier |
| `kpi_name` | String | Yes | KPI metric name |
| `kpi_value` | Float | Yes | KPI metric value |
| `ue_id` | String | No | User equipment ID |
| `sector_id` | String | No | Sector identifier |

### Supported Input Formats

#### CSV Format

```csv
timestamp,cell_id,kpi_name,kpi_value,ue_id
2024-01-01 10:00:00.000,cell_001,throughput_dl,45.5,ue_12345
2024-01-01 10:00:01.000,cell_001,throughput_ul,12.3,ue_12345
```

#### JSON Lines Format

```json
{"timestamp": "2024-01-01 10:00:00.000", "cell_id": "cell_001", "kpi_name": "throughput_dl", "kpi_value": 45.5, "ue_id": "ue_12345"}
{"timestamp": "2024-01-01 10:00:01.000", "cell_id": "cell_001", "kpi_name": "throughput_ul", "kpi_value": 12.3, "ue_id": "ue_12345"}
```

## 📈 Monitoring and Metrics

### Available Metrics

- **Throughput**: MB/s processing rate
- **Error Rates**: Parsing and validation error percentages
- **Processing Times**: Average file processing duration
- **Resource Usage**: CPU and memory utilization
- **Queue Depth**: Number of files waiting to be processed

### Health Checks

```bash
# Check service health
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check

# Get current metrics
grpcurl -plaintext localhost:50051 pfs.data.v1.DataIngestionService/GetIngestionMetrics
```

## 🧪 Testing

### Unit Tests

```bash
cargo test
```

### Integration Tests

```bash
cargo test --test integration_tests
```

### Performance Benchmarks

```bash
cargo bench
```

### Load Testing

```bash
# Generate large test dataset
./target/release/pfs-data-01 test --files 1000 --rows 100000

# Monitor performance during processing
cargo bench -- --save-baseline baseline_name
```

## 🔍 Troubleshooting

### Common Issues

#### High Error Rate

```bash
# Check error details
./target/release/pfs-data-01 process --input-dir ./data --output-dir ./output --verbose

# Validate configuration
./target/release/pfs-data-01 validate-config config.toml
```

#### Performance Issues

```bash
# Try performance-optimized configuration
export PFS_CONFIG="performance"
./target/release/pfs-data-01 serve --config optimized.toml

# Monitor resource usage
top -p $(pgrep pfs-data-01)
```

#### Schema Validation Errors

Check that your input data matches the expected schema:

```bash
# Example CSV header
timestamp,cell_id,kpi_name,kpi_value,ue_id

# Ensure timestamp format is ISO 8601
2024-01-01 10:00:00.000
```

### Logging

Enable detailed logging:

```bash
# Debug logging
RUST_LOG=debug ./target/release/pfs-data-01 serve

# JSON structured logs
./target/release/pfs-data-01 serve --json-logs --log-level debug
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
cargo install cargo-watch cargo-audit

# Run tests continuously during development
cargo watch -x test

# Check for security vulnerabilities
cargo audit
```

## 📄 License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🔗 Related Projects

- [ruv-FANN](https://github.com/ricable/ruv-FANN) - Fast Artificial Neural Network library
- [RAN Intelligence Platform](../../../) - Complete RAN intelligence solution

## 📞 Support

- GitHub Issues: [Report bugs and request features](https://github.com/ricable/ruv-FANN/issues)
- Documentation: [API Documentation](https://docs.rs/pfs-data-01)
- Examples: [Usage examples](./examples/)

---

Built with ❤️ for the RAN Intelligence Platform