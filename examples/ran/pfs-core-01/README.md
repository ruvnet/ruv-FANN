# PFS-CORE-01: Rust gRPC Neural Service

PFS-CORE-01 is a high-performance Rust gRPC service that wraps the ruv-FANN library, providing neural network training and prediction capabilities for the RAN Intelligence Platform.

## Features

- **gRPC API**: Modern, efficient remote procedure calls with Protocol Buffers
- **ruv-FANN Integration**: Built on the robust ruv-FANN neural network library
- **Model Management**: Complete lifecycle management for neural network models
- **Concurrent Training**: Support for multiple simultaneous training operations
- **Persistent Storage**: Model serialization and deserialization
- **Performance Monitoring**: Built-in metrics and health checks
- **Comprehensive Testing**: Unit tests, integration tests, and benchmarks

## API Endpoints

### Training
- `Train(TrainRequest) -> TrainResponse`: Train a neural network model with specified configuration and data

### Prediction
- `Predict(PredictRequest) -> PredictResponse`: Make predictions using a trained model

### Model Management
- `GetModelInfo(GetModelInfoRequest) -> GetModelInfoResponse`: Retrieve model information and metadata
- `ListModels(ListModelsRequest) -> ListModelsResponse`: List all available models with filtering
- `DeleteModel(DeleteModelRequest) -> DeleteModelResponse`: Remove a model from storage

### Health Check
- `Health(HealthRequest) -> HealthResponse`: Service health and status information

## Quick Start

### Building

```bash
# Build the service
cargo build --release

# Build with all features
cargo build --release --all-features
```

### Running the Server

```bash
# Run with default configuration
cargo run --bin pfs-core-01-server

# Run with custom configuration
PFS_CONFIG_FILE=config.toml cargo run --bin pfs-core-01-server
```

### Using the Client

```bash
# Check service health
cargo run --bin pfs-core-01-client -- health

# Train XOR model
cargo run --bin pfs-core-01-client -- train \
  --name "XOR Model" \
  --layers "2,4,1" \
  --learning-rate 0.1 \
  --epochs 1000 \
  --data-file examples/xor_training_data.json

# Make prediction
cargo run --bin pfs-core-01-client -- predict \
  --model-id <MODEL_ID> \
  --input "1.0,0.0"

# List models
cargo run --bin pfs-core-01-client -- list --page 0 --size 10

# Get model information
cargo run --bin pfs-core-01-client -- info --model-id <MODEL_ID>

# Delete model
cargo run --bin pfs-core-01-client -- delete --model-id <MODEL_ID>
```

## Configuration

The service can be configured using a TOML file or environment variables:

```toml
[server]
host = "127.0.0.1"
port = 50051
max_connections = 1000
request_timeout_seconds = 300

[storage]
models_directory = "./models"
max_models = 100
cleanup_interval_seconds = 3600

[training]
max_concurrent_training = 4
default_max_epochs = 10000
default_learning_rate = 0.01
default_desired_error = 0.001

[logging]
level = "info"
format = "json"
```

Environment variables use the prefix `PFS_CORE_`:
- `PFS_CORE_SERVER__HOST`
- `PFS_CORE_SERVER__PORT`
- `PFS_CORE_STORAGE__MODELS_DIRECTORY`
- etc.

## Training Data Format

Training data should be provided in JSON format:

```json
{
  "examples": [
    {
      "inputs": [0.0, 0.0],
      "outputs": [0.0]
    },
    {
      "inputs": [0.0, 1.0],
      "outputs": [1.0]
    }
  ]
}
```

## Supported Neural Network Features

### Activation Functions
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- Linear
- Softmax

### Training Algorithms
- Backpropagation (Incremental and Batch)
- RPROP (Resilient Backpropagation)
- QuickProp

### Training Configuration
- Batch size control
- Data shuffling
- Validation splitting
- Early stopping with patience
- Learning rate adjustment

## Testing

```bash
# Run unit tests
cargo test

# Run integration tests
cargo test --test integration_tests

# Run benchmarks
cargo bench
```

## Performance Benchmarks

The service includes comprehensive benchmarks for:

### Training Performance
- Different training algorithms (Backprop, RPROP, QuickProp)
- Various network sizes (small, medium, large)
- Different dataset sizes (100, 500, 1000, 2000 samples)
- Learning rate variations (0.001, 0.01, 0.1, 0.5)

### Prediction Performance
- Model size scaling
- Batch prediction capabilities
- Input size variations
- Concurrent prediction load testing

Run benchmarks with:
```bash
cargo bench --bench training_performance
cargo bench --bench prediction_performance
```

## Architecture

### Components

1. **gRPC Service Layer**: Handles remote procedure calls and request validation
2. **Model Manager**: Manages model lifecycle, storage, and retrieval
3. **ruv-FANN Integration**: Neural network training and prediction
4. **Configuration Management**: Service configuration and validation
5. **Error Handling**: Comprehensive error types and conversions

### Data Flow

1. Client sends training request via gRPC
2. Service validates configuration and training data
3. Model Manager creates and stores new model
4. Neural network training is performed using ruv-FANN
5. Trained model is saved and metadata updated
6. Client can make predictions using the trained model ID

## Error Handling

The service provides detailed error information for:
- Invalid model configurations
- Training failures
- Prediction errors
- Model not found
- Storage issues
- Network problems

## Monitoring and Observability

### Health Checks
- Service status
- Version information
- Active model count
- Uptime tracking

### Logging
- Structured JSON logging
- Configurable log levels
- Training progress tracking
- Performance metrics

## Security Considerations

- Input validation for all parameters
- Model ID validation to prevent path traversal
- Resource limits for concurrent training
- Timeout handling for long-running operations

## Contributing

1. Follow Rust coding standards
2. Add tests for new features
3. Update documentation
4. Run benchmarks for performance-critical changes
5. Ensure all tests pass

## License

This project is licensed under the MIT OR Apache-2.0 license.

## Performance Characteristics

### Training Performance
- Supports concurrent training operations (configurable limit)
- Efficient memory usage with model lifecycle management
- Automatic cleanup of old models when storage limits are reached

### Prediction Performance
- Low-latency predictions (typically < 10ms for small models)
- High concurrent prediction throughput
- Memory-efficient model loading

### Scalability
- Horizontal scaling through multiple service instances
- Persistent model storage for service restarts
- Configurable resource limits and timeouts

## Examples

See the `examples/` directory for:
- XOR training data (`xor_training_data.json`)
- Sine wave approximation data (`sine_wave_data.json`)
- Custom training data formats

## Troubleshooting

### Common Issues

1. **Training fails with "Invalid input"**
   - Check training data format
   - Verify input/output dimensions match model configuration
   - Ensure all numeric values are finite

2. **Model not found errors**
   - Verify model ID is correct
   - Check if model was successfully trained
   - Ensure models directory is accessible

3. **Performance issues**
   - Adjust `max_concurrent_training` setting
   - Monitor memory usage during training
   - Use appropriate model sizes for available resources

4. **Connection refused**
   - Verify server is running
   - Check host and port configuration
   - Ensure firewall allows connections

For more detailed troubleshooting, enable debug logging:
```toml
[logging]
level = "debug"
```