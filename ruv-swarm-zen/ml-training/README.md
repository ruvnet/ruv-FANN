# RUV Swarm ML Training Pipeline

Advanced machine learning training pipeline for neuro-divergent models in the RUV Swarm ecosystem. This crate provides comprehensive tools for training LSTM, TCN, and N-BEATS models to predict agent performance and optimize prompts.

## Features

- **Stream Data Loading**: Efficient loading and preprocessing of time-series event data from JSON streams
- **Multiple Model Architectures**:
  - LSTM (Long Short-Term Memory) for sequence modeling
  - TCN (Temporal Convolutional Networks) for efficient temporal processing
  - N-BEATS (Neural Basis Expansion Analysis) for time series forecasting
- **Hyperparameter Optimization**: Advanced optimization methods including Random Search, Bayesian Optimization, Grid Search, and Hyperband
- **Model Evaluation**: Comprehensive evaluation metrics for performance prediction and model selection
- **Feature Engineering**: Automatic feature extraction from performance metrics, prompt data, and temporal patterns

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ruv-swarm-ml-training = "0.1.0"
```

## Quick Start

```rust
use ruv_swarm_ml_training::{TrainingPipeline, TrainingConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure training
    let config = TrainingConfig::default();
    
    // Create pipeline
    let mut pipeline = TrainingPipeline::new(config);
    
    // Run training with your event stream
    let result = pipeline.run(event_stream).await?;
    
    println!("Best model: {}", result.best_model);
    Ok(())
}
```

## Architecture

### Data Flow

1. **Event Stream** → Stream JSON events from agents
2. **Feature Extraction** → Convert events to numerical features
3. **Sequence Creation** → Build time series sequences for training
4. **Model Training** → Train multiple model architectures
5. **Hyperparameter Optimization** → Find optimal parameters
6. **Model Evaluation** → Compare and select best model

### Core Components

#### StreamDataLoader
Loads and processes streaming event data:
```rust
let loader = StreamDataLoader::new(buffer_size, sequence_length);
let dataset = loader.load_from_stream(events).await?;
```

#### Model Implementations

**LSTM Model**:
```rust
let lstm = LSTMModel::new(hidden_size: 128, num_layers: 2);
```

**TCN Model**:
```rust
let tcn = TCNModel::new(num_channels: vec![64, 64, 64], kernel_size: 3);
```

**N-BEATS Model**:
```rust
let nbeats = NBEATSModel::new(
    stack_types: vec![StackType::Trend, StackType::Seasonality],
    num_blocks: 4
);
```

#### Hyperparameter Optimization

```rust
let search_space = SearchSpace {
    parameters: HashMap::from([
        ("learning_rate", ParameterRange::Continuous { min: 0.0001, max: 0.01 }),
        ("hidden_size", ParameterRange::Discrete { values: vec![64.0, 128.0, 256.0] }),
    ]),
};

let optimizer = HyperparameterOptimizer::new(
    search_space,
    OptimizationMethod::BayesianOptimization,
    num_trials: 20,
);

let result = optimizer.optimize(model_factory, dataset, config).await?;
```

## Event Structure

The pipeline expects events in this format:

```rust
StreamEvent {
    timestamp: u64,
    agent_id: String,
    event_type: EventType,
    performance_metrics: PerformanceMetrics {
        latency_ms: f64,
        tokens_per_second: f64,
        memory_usage_mb: f64,
        cpu_usage_percent: f64,
        success_rate: f64,
    },
    prompt_data: Option<PromptData {
        prompt_text: String,
        prompt_tokens: usize,
        response_tokens: usize,
        quality_score: f64,
    }>,
}
```

## Feature Extraction

The pipeline automatically extracts features from events:

- **Performance Features**: Latency, throughput, resource usage
- **Prompt Features**: Token counts, quality scores, response ratios
- **Temporal Features**: Hour of day, day of week, cyclical encodings

## Model Evaluation Metrics

- **MSE** (Mean Squared Error): Overall prediction accuracy
- **MAE** (Mean Absolute Error): Average prediction error
- **R²** (Coefficient of Determination): Variance explained
- **Latency Accuracy**: Percentage of predictions within threshold
- **Success Rate Prediction**: Accuracy of success rate forecasts

## Examples

See the `examples/` directory for:

- `basic_training.rs` - Simple training pipeline usage
- `hyperparameter_search.rs` - Advanced hyperparameter optimization
- `model_comparison.rs` - Comparing different model architectures

## Performance Considerations

- The pipeline is designed for efficient batch processing
- Models support both CPU and GPU training (when available)
- Feature extraction is parallelized for large datasets
- Memory-efficient sliding window approach for sequence generation

## Future Enhancements

- [ ] Integration with actual neural network backends (Candle, Burn)
- [ ] Support for distributed training
- [ ] Online learning capabilities
- [ ] Model ensembling
- [ ] AutoML features
- [ ] WASM compilation support

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.

## Contributing

Contributions are welcome! Please see the [contributing guidelines](../../CONTRIBUTING.md) for details.