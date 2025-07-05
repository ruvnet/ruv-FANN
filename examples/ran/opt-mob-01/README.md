# OPT-MOB-01: Predictive Handover Trigger Model

## Overview

OPT-MOB-01 is a neural network-based handover prediction system for RAN (Radio Access Network) optimization. It analyzes UE (User Equipment) metrics to predict handover probability and recommend target cells with >90% accuracy.

## Features

- **Real-time Prediction**: gRPC-based service for low-latency handover predictions
- **Advanced Feature Engineering**: 57 time-series features including velocity, acceleration, and mobility patterns
- **Comprehensive Backtesting**: Extensive evaluation framework with >90% accuracy target
- **Neural Network Core**: Built on ruv-FANN for high-performance prediction
- **Production Ready**: Full service with health checks, metrics, and monitoring

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   UE Metrics    │───▶│ Feature Engine  │───▶│ Neural Network  │
│                 │    │                 │    │                 │
│ - RSRP/SINR     │    │ - Time series   │    │ - 57 inputs     │
│ - Speed/Location│    │ - Rolling stats │    │ - 3 hidden      │
│ - Neighbor info │    │ - Mobility      │    │ - 1 output      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐             │
                       │ gRPC Service    │◀────────────┘
                       │                 │
                       │ - Predictions   │
                       │ - Batch API     │
                       │ - Health checks │
                       └─────────────────┘
```

## Quick Start

### 1. Build the Project

```bash
cd /Users/cedric/dev/my-forks/ruv-FANN/examples/ran/opt-mob-01
cargo build --release
```

### 2. Generate Training Data

```bash
cargo run --bin handover-predictor -- --generate-data
```

### 3. Train the Model

```bash
cargo run --bin train-handover-model -- --synthetic --epochs 500 --backtest
```

### 4. Start the Prediction Service

```bash
cargo run --bin handover-predictor -- --model models/handover_v1.bin
```

### 5. Test Predictions

```bash
# The service will be available on localhost:50051
# Use gRPC clients to send UE metrics and receive predictions
```

## Data Requirements

### Input UE Metrics
- **Radio Measurements**: RSRP, SINR, RSRQ, CQI
- **Mobility Information**: Speed, bearing, altitude
- **Neighbor Cell Data**: RSRP/SINR of neighboring cells
- **Network Context**: Technology (LTE/5G), frequency band
- **Load Information**: PRB usage, active users

### Expected Handover Events
- Source and target cell IDs
- Handover timestamp and type
- Success/failure indication
- Performance metrics (preparation/execution time)

## Feature Engineering

The system extracts 57 features from UE metrics:

### Time-Series Features (6)
- Current measurements (RSRP, SINR, speed, etc.)
- Lag features (t-1, t-2, t-3)

### Statistical Features (30)
- Rolling window statistics (5 and 10 samples)
- Mean, standard deviation, min, max, trend

### Mobility Features (8)
- Velocity and acceleration for RSRP/SINR
- Speed categorization and direction stability

### Handover Trigger Features (3)
- A3 event margin calculation
- Time below signal threshold
- Composite handover urgency score

### Contextual Features (10)
- Time-based features (hour, day, weekend)
- Cell load and technology information
- Historical handover patterns

## Model Architecture

### Neural Network
- **Input Layer**: 57 features
- **Hidden Layers**: 128 → 64 → 32 neurons
- **Output Layer**: 1 neuron (handover probability)
- **Activation**: Sigmoid for output, symmetric sigmoid for hidden

### Training Configuration
- **Algorithm**: Resilient Propagation (RProp)
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Early Stopping**: 50 epochs patience
- **Validation Split**: 20%

## API Reference

### gRPC Service

#### PredictHandover
Predict handover for a single UE:

```protobuf
rpc PredictHandover(HandoverPredictionRequest) returns (HandoverPredictionResponse);

message HandoverPredictionRequest {
    string ue_id = 1;
    repeated UeMetric metrics = 2;
    int64 prediction_horizon_seconds = 3;
}

message HandoverPredictionResponse {
    string ue_id = 1;
    double ho_probability = 2;
    string target_cell_id = 3;
    double confidence = 4;
    repeated string candidate_cells = 6;
}
```

#### BatchPredictHandover
Batch prediction for multiple UEs:

```protobuf
rpc BatchPredictHandover(BatchHandoverPredictionRequest) returns (BatchHandoverPredictionResponse);
```

#### GetModelInfo
Get model information and statistics:

```protobuf
rpc GetModelInfo(ModelInfoRequest) returns (ModelInfoResponse);
```

#### HealthCheck
Service health monitoring:

```protobuf
rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
```

## Performance Targets

### Accuracy Requirements
- **Primary Target**: >90% accuracy on backtesting
- **Precision**: >85% (minimize false positives)
- **Recall**: >85% (minimize missed handovers)
- **F1-Score**: >85% (balanced performance)

### Latency Requirements
- **Single Prediction**: <5ms average
- **Batch Processing**: <100ms for 100 UEs
- **Throughput**: >1000 predictions/second

### Memory Requirements
- **Model Size**: <10MB
- **Runtime Memory**: <100MB
- **Feature Buffer**: <1MB per UE

## Configuration

### Default Configuration
```json
{
  "model_path": "models/handover_v1.bin",
  "prediction_horizon_seconds": 30,
  "handover_threshold": 0.5,
  "feature_window_size": 10,
  "sampling_rate_ms": 1000,
  "grpc_port": 50051,
  "max_batch_size": 1000,
  "enable_logging": true
}
```

### Generate Default Config
```bash
cargo run --bin handover-predictor -- --generate-config
```

## Training Guide

### Basic Training
```bash
cargo run --bin train-handover-model -- \
    --synthetic \
    --epochs 1000 \
    --learning-rate 0.001 \
    --hidden-layers 128,64,32
```

### Advanced Training with Backtesting
```bash
cargo run --bin train-handover-model -- \
    --synthetic \
    --epochs 1000 \
    --cross-validation \
    --backtest \
    --output models/handover_production.bin
```

### Training with Real Data
```bash
cargo run --bin train-handover-model -- \
    --train-data data/ue_metrics.csv \
    --train-events data/handover_events.csv \
    --test-data data/test_metrics.csv \
    --test-events data/test_events.csv \
    --epochs 2000 \
    --backtest
```

## Backtesting Framework

### Comprehensive Evaluation
```bash
cargo run --bin backtest-handover-model -- \
    --model models/handover_v1.bin \
    --synthetic \
    --cross-validation 5 \
    --temporal-analysis \
    --feature-importance \
    --confidence-analysis \
    --error-analysis \
    --benchmark \
    --real-time
```

### Evaluation Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Probability Metrics**: AUC-ROC, AUC-PR, Brier Score, Log Loss
- **Confidence Analysis**: Calibration error, reliability diagram
- **Temporal Analysis**: Time-based performance patterns
- **Error Analysis**: False positive/negative patterns

## Deployment

### Production Service
```bash
# Build optimized release
cargo build --release

# Start service
./target/release/handover-predictor \
    --model models/production_model.bin \
    --port 50051 \
    --bind 0.0.0.0
```

### Docker Deployment
```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/handover-predictor /usr/local/bin/
EXPOSE 50051
CMD ["handover-predictor"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: handover-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: handover-predictor
  template:
    metadata:
      labels:
        app: handover-predictor
    spec:
      containers:
      - name: handover-predictor
        image: handover-predictor:latest
        ports:
        - containerPort: 50051
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## Monitoring and Observability

### Health Checks
- **Service Health**: Model loaded and functional
- **Prediction Quality**: Ongoing accuracy monitoring
- **Performance Metrics**: Latency and throughput tracking
- **Resource Usage**: Memory and CPU utilization

### Metrics Collection
- **Request Metrics**: Total requests, success rate, latency
- **Model Metrics**: Prediction distribution, confidence levels
- **Business Metrics**: Handover success rate, false alarm rate

### Logging
- **Structured Logging**: JSON format with correlation IDs
- **Log Levels**: ERROR, WARN, INFO, DEBUG, TRACE
- **Audit Trail**: All predictions with input/output data

## Testing

### Unit Tests
```bash
cargo test
```

### Integration Tests
```bash
cargo test --test integration
```

### Performance Tests
```bash
cargo bench
```

### Model Validation
```bash
cargo run --bin handover-predictor -- --validate-model models/handover_v1.bin
```

## Troubleshooting

### Common Issues

#### Model Loading Errors
```
Error: Model file not found
Solution: Train a model first or check file path
```

#### Low Accuracy
```
Warning: Model accuracy below 90% target
Solutions:
- Increase training data
- Tune hyperparameters
- Improve feature engineering
- Check data quality
```

#### High Latency
```
Warning: Prediction time > 5ms
Solutions:
- Optimize model size
- Use batch predictions
- Enable parallel processing
- Check resource allocation
```

### Debug Commands
```bash
# Validate model
cargo run --bin handover-predictor -- --validate-model models/handover_v1.bin

# Generate test data
cargo run --bin handover-predictor -- --generate-data

# Check configuration
cargo run --bin handover-predictor -- --generate-config
```

## Contributing

### Development Setup
1. Install Rust 1.70+
2. Install Protocol Buffers compiler
3. Clone repository and build dependencies
4. Run tests to verify setup

### Code Style
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Follow Rust naming conventions
- Add comprehensive documentation

### Pull Request Process
1. Create feature branch
2. Implement changes with tests
3. Ensure all tests pass
4. Update documentation
5. Submit pull request

## License

Licensed under the MIT OR Apache-2.0 license.

## Acknowledgments

- Built on ruv-FANN neural network library
- Implements 3GPP handover standards
- Inspired by production RAN optimization systems

---

**HandoverPredictorAgent** - Part of the RAN Intelligence Platform
For technical support and questions, please create an issue in the repository.