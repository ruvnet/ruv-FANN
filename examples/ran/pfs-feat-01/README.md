# PFS-FEAT-01: Time-series Feature Generation Agent

## Overview

PFS-FEAT-01 is a high-performance time-series feature generation agent designed for the RAN Intelligence Platform. It implements sophisticated feature engineering capabilities specifically tailored for Radio Access Network (RAN) data analysis and machine learning workflows.

## Features

### Core Capabilities

- **Time-series Feature Generation**: Advanced lag features, rolling window statistics, and time-based features
- **High Performance**: Parallel processing with configurable concurrency levels
- **RAN-Specific Templates**: Pre-configured feature sets for common RAN use cases
- **Validation Framework**: Comprehensive validation of generated features
- **gRPC Interface**: High-performance service interface for integration
- **Flexible Configuration**: TOML-based configuration with validation

### Feature Types

#### 1. Lag Features
- Configurable lag periods (t-1, t-2, ..., t-n)
- Support for multiple target columns
- Efficient memory usage for large datasets

#### 2. Rolling Window Statistics
- Multiple window sizes (3, 6, 12, 24, 48, 72 hours)
- Statistical measures: mean, std, min, max, median, quantiles, IQR, skewness
- Optimized for time-series data patterns

#### 3. Time-based Features
- Hour of day, day of week, month, quarter, year
- Business hour detection (9 AM - 5 PM)
- Peak hour detection (8-10 AM, 5-7 PM)
- Weekend detection
- Timezone support

#### 4. RAN-Specific Features
- PRB utilization patterns
- Radio quality metrics (RSRP, SINR)
- Throughput and user activity features
- Handover prediction features
- Interference detection features

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   gRPC Client   │────│ gRPC Service    │────│ Feature Engine  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐             │
                       │   Validation    │─────────────┤
                       │   Framework     │             │
                       └─────────────────┘             │
                                                        │
                       ┌─────────────────┐             │
                       │   Statistics    │─────────────┤
                       │   Collector     │             │
                       └─────────────────┘             │
                                                        │
                       ┌─────────────────┐             │
                       │   Feature       │─────────────┘
                       │   Generators    │
                       └─────────────────┘
```

## Installation

### Prerequisites

- Rust 1.70 or later
- Polars data processing library
- Protocol Buffers compiler

### Build

```bash
cd pfs-feat-01
cargo build --release
```

### Dependencies

The implementation relies on several key dependencies:

- **Polars**: High-performance DataFrame library for data processing
- **Tonic**: gRPC framework for Rust
- **Tokio**: Async runtime
- **Rayon**: Data parallelism
- **Statrs**: Statistical functions
- **Chrono**: Date and time handling

## Usage

### Command Line Interface

#### Start gRPC Server

```bash
./target/release/pfs-feat-01 --config config.toml --host 127.0.0.1 --port 50051
```

#### One-shot Feature Generation

```bash
./target/release/pfs-feat-01 --one-shot \
  --input /path/to/input.parquet \
  --output /path/to/output.parquet \
  --time-series-id "cell_001"
```

#### Generate Default Configuration

```bash
./target/release/pfs-feat-01 --generate-config config.toml
```

#### Validate Configuration

```bash
./target/release/pfs-feat-01 --validate-config --config config.toml
```

### Configuration

The agent uses TOML configuration files. See `config.toml` for a complete example:

```toml
[service]
host = "127.0.0.1"
port = 50051
max_concurrent_requests = 100

[default_features.lag_features]
enabled = true
lag_periods = [1, 2, 3, 6, 12, 24, 48]
target_columns = ["kpi_value", "prb_utilization_dl"]

[default_features.rolling_window]
enabled = true
window_sizes = [3, 6, 12, 24, 48]
statistics = ["mean", "std", "min", "max", "median"]

[processing]
max_parallel_jobs = 8
memory_limit_mb = 4096
```

### gRPC API

#### Generate Features

```protobuf
rpc GenerateFeatures(GenerateFeaturesRequest) returns (GenerateFeaturesResponse);
```

#### Batch Processing

```protobuf
rpc GenerateBatchFeatures(GenerateBatchFeaturesRequest) returns (GenerateBatchFeaturesResponse);
```

#### Validation

```protobuf
rpc ValidateFeatures(ValidateFeaturesRequest) returns (ValidateFeaturesResponse);
```

### Programmatic Usage

```rust
use pfs_feat_01::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create configuration
    let config = FeatureEngineConfig::default();
    
    // Create agent
    let agent = FeatureEngineeringAgent::new(config.clone());
    
    // Generate features
    let result = agent.generate_features(
        "time_series_001",
        Path::new("input.parquet"),
        Path::new("output.parquet"),
        &config.default_features,
    ).await?;
    
    println!("Generated {} features in {}ms", 
             result.stats.features_generated,
             result.stats.processing_time_ms);
    
    Ok(())
}
```

## RAN-Specific Templates

### RAN KPI Features

```rust
let config = RanFeatureTemplates::ran_kpi_features();
```

Features designed for general RAN KPI analysis:
- PRB utilization patterns (DL/UL)
- User activity metrics
- Throughput patterns
- Radio quality trends

### Handover Prediction Features

```rust
let config = RanFeatureTemplates::handover_prediction_features();
```

Optimized for handover prediction models:
- Short-term signal quality trends
- Mobility patterns
- Neighbor cell relationships

### Interference Detection Features

```rust
let config = RanFeatureTemplates::interference_detection_features();
```

Specialized for interference detection:
- Noise floor patterns
- Signal quality degradation
- Uplink interference indicators

## Performance

### Benchmarks

Run benchmarks to measure performance:

```bash
cargo bench
```

### Expected Performance

- **Single Series**: ~1000 rows/second with full feature set
- **Batch Processing**: Linear scaling with CPU cores
- **Memory Usage**: ~1-2MB per 1000 rows with features
- **Parallelism**: Efficient scaling up to CPU core count

### Optimization Tips

1. **Adjust Parallel Jobs**: Set `max_parallel_jobs` to CPU core count
2. **Batch Size**: Optimal batch size is 500-2000 rows
3. **Feature Selection**: Disable unused feature types
4. **Memory Management**: Monitor memory usage with large datasets

## Validation

### Acceptance Criteria

The implementation meets PRD acceptance criteria:

- ✅ **Schema Validation**: Output schema validated for 1000+ time-series
- ✅ **Feature Quality**: Comprehensive quality scoring
- ✅ **Performance**: Sub-second processing for typical datasets
- ✅ **Error Handling**: Robust error detection and reporting

### Validation Framework

```rust
let validator = FeatureValidator::new(config);
let result = validator.validate_sample_batch(
    output_directory,
    1000, // expected series count
).await?;

assert!(result.is_valid);
```

## Testing

### Unit Tests

```bash
cargo test
```

### Integration Tests

```bash
cargo test --test integration_tests
```

### Example Usage

```bash
cargo run --example basic_usage
```

## Monitoring and Statistics

### Built-in Metrics

- Processing time per operation
- Memory usage tracking
- Feature generation statistics
- Error rate monitoring
- Throughput metrics

### Real-time Monitoring

```rust
let stats = agent.get_statistics().await;
println!("Processed {} rows in {}ms", 
         stats.total_rows_processed, 
         stats.total_processing_time_ms);
```

## Error Handling

### Error Types

- **Configuration Errors**: Invalid settings
- **Data Processing Errors**: Malformed input data
- **Feature Generation Errors**: Failed computations
- **Validation Errors**: Quality check failures
- **Memory Errors**: Resource exhaustion

### Recovery Strategies

- Automatic retry for transient failures
- Graceful degradation for resource limits
- Detailed error reporting for debugging

## Development

### Project Structure

```
pfs-feat-01/
├── src/
│   ├── main.rs              # CLI application
│   ├── lib.rs               # Library root
│   ├── config.rs            # Configuration management
│   ├── engine.rs            # Core feature engine
│   ├── features.rs          # Feature generators
│   ├── grpc_service.rs      # gRPC service implementation
│   ├── validation.rs        # Validation framework
│   ├── stats.rs             # Statistics collection
│   └── error.rs             # Error handling
├── proto/
│   └── feature_engineering.proto
├── examples/
│   └── basic_usage.rs
├── tests/
│   └── integration_tests.rs
├── benches/
│   └── feature_generation.rs
├── config.toml
└── README.md
```

### Adding New Features

1. **Create Feature Generator**: Implement `FeatureGenerator` trait
2. **Update Configuration**: Add config options
3. **Add Tests**: Unit and integration tests
4. **Update Documentation**: README and examples

### Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## Troubleshooting

### Common Issues

#### Out of Memory

```bash
# Reduce memory usage
max_parallel_jobs = 2
batch_size = 500
memory_limit_mb = 1024
```

#### Slow Performance

```bash
# Optimize for performance
max_parallel_jobs = 8
streaming_mode = true
disable_validation = true  # For production
```

#### Missing Features

```bash
# Check target columns exist in input data
target_columns = ["kpi_value", "prb_utilization_dl"]
```

### Debugging

Enable debug logging:

```bash
RUST_LOG=debug ./target/release/pfs-feat-01
```

View detailed statistics:

```rust
let stats = agent.get_statistics().await;
for (feature, stats) in stats.feature_stats {
    println!("{}: {} generations", feature, stats.generation_count);
}
```

## License

This implementation is part of the ruv-FANN project and follows the same licensing terms.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review integration tests for examples
3. Open an issue on the repository
4. Contact the RAN Intelligence Platform team

## Roadmap

### Planned Enhancements

- **Streaming Support**: Real-time feature generation
- **GPU Acceleration**: CUDA-based computations
- **Advanced Features**: Fourier transforms, wavelets
- **ML Integration**: Direct model training pipelines
- **Distributed Processing**: Multi-node scaling

### Performance Goals

- **Target**: 10,000+ rows/second processing
- **Scalability**: Linear scaling to 32+ cores
- **Memory**: <100MB for 1M row datasets
- **Latency**: <100ms for real-time features