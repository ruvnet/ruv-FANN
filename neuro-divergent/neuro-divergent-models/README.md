# neuro-divergent-models

Trainable vanilla recurrent forecasting and reusable numerical operations backed by ruv-FANN. RNN learns both its recurrent weights and its output projection. `LSTM` and `GRU` constructors return explicit unsupported errors; they never substitute a vanilla RNN. Separate low-level LSTM/GRU cells provide gated forward operations, without a gated forecasting trainer.

![A forecasting experiment evaluated on future observations](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/neuro-divergent/docs/assets/forecasting-flow.svg)

## Use a numerical component

```rust
use ndarray::{array, Array1};
use neuro_divergent_models::activations::{ActivationFunction, Softmax};

let logits = array![1.0_f64, 2.0, 3.0];
let probabilities: Array1<f64> = Softmax.forward(&logits);
assert!((probabilities.sum() - 1.0).abs() < 1e-12);
let gradient = Softmax.backward(&logits, &array![0.1, -0.2, 0.3]);
assert!(gradient.sum().abs() < 1e-12);
```

Softmax backward computes a vector-Jacobian product in linear time and storage, with finite-difference and dense-Jacobian checks. Matrix multiplication validates rectangular dimensions and walks contiguous output rows. LayerNorm and BatchNorm calculate centered variance to preserve small variation on large offsets. Loss helpers reject empty, mismatched, and non-finite inputs instead of silently ignoring values.

## Train a multi-step RNN

`input_size` counts features at each time step. The history can have any positive length; `hidden_size` and the forecast `horizon` are independent.

```rust
use neuro_divergent_models::{BaseModel, RNNConfig, TimeSeriesInput};
use neuro_divergent_models::models::RNN;
use neuro_divergent_models::foundation::{DatasetMetadata, TimeSeriesDataset, TimeSeriesSample};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = RNNConfig::<f64>::default_with_horizon(3);
    config.hidden_size = 5;
    config.max_steps = 400;
    config.learning_rate = 0.02;
    config.gradient_clip_val = Some(1.0);

    // Each training window follows y[t + 1] = 0.8 * y[t].
    let samples = (-4..=4).map(|i| {
        let start = i as f64 / 5.0;
        TimeSeriesSample {
            input: TimeSeriesInput::new((0..3).map(|t| start * 0.8_f64.powi(t)).collect()),
            target: (3..6).map(|t| start * 0.8_f64.powi(t)).collect(),
            weight: None,
        }
    }).collect();
    let data = TimeSeriesDataset {
        samples,
        metadata: DatasetMetadata {
            name: None, description: None, num_series: 1, total_samples: 9,
            feature_names: vec![], target_names: vec![],
        },
    };
    let mut model = RNN::new_seeded(config, 17)?;
    let metrics = model.fit(&data)?;
    assert!(metrics.final_loss < 1e-4);
    let restored = RNN::<f64>::from_state(&model.save_state()?)?;
    let forecast = restored.predict(&TimeSeriesInput::new(vec![0.37, 0.296, 0.2368]))?;
    assert_eq!(forecast.forecasts.len(), 3);
    Ok(())
}
```

## Training and model boundaries

RNN uses full backpropagation through time (BPTT) through every stacked recurrent layer and a linear FANN projection that returns all horizon values at once. Each `max_steps` iteration is one full-batch Adam update of weighted mean squared error. Optional global gradient norm clipping and coupled L2 regularization apply to all parameters, including biases. Between-layer dropout is enabled during fitting and disabled for prediction.

For a long history, `set_bptt_truncation(Some(k))` differentiates the final `k` steps and treats earlier states as constants. The forward pass still processes the whole history. Samples start independently from zero states unless the caller sets initial states through `RecurrentState`. Input values are not automatically scaled. The feature order is target, static features, then historical features; future exogenous features and bidirectional training return errors.

Versioned checkpoints store configuration, all learned parameters, initial states, seed, BPTT setting, training history, and metrics. Restoring a checkpoint reproduces predictions. A subsequent `fit` starts a fresh optimizer. Checkpoint input is capped at 256 MiB and configurations at 16 million parameters; these limits are not a hard bound on peak process memory. Invalid shapes, non-finite inputs, and failed numerical updates return errors while preserving the previous fitted model.

The models crate has its own `BaseModel`, configuration, and data types; use the facade's explicit adapter when working with its DataFrames. Its older row DataFrame `to_dataset` and `forecasting::NeuralForecast::predict` methods return unsupported errors because they lack an explicit window/history contract. Basic, advanced, transformer, and specialized architecture source directories are not exported. Enabling `transformer` or `specialized` does not activate those files. Numerical gradient checks and a small synthetic learning test do not establish accuracy, parity, GPU support, or state-of-the-art performance on real forecasting workloads.

## Validate

From the repository root:

```sh
CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 cargo test --manifest-path neuro-divergent/Cargo.toml -p neuro-divergent-models --lib --locked
```

For numerical and recurrent tests without Polars:

```sh
cargo test --manifest-path neuro-divergent/Cargo.toml -p neuro-divergent-models --lib --no-default-features --features serde --locked
```

The 0.2.0 candidate moves to ruv-FANN 0.2.2 and optional Polars 0.45.1. See the [family review](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-forecasting.md) for exact validation and research candidates that still need implementation and evaluation.
