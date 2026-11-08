# ruv-swarm-ml

Fit small dense neural networks to a univariate time series and generate forecasts from the trained weights through `ruv-fann`.

## Fit and forecast

```rust
use ruv_swarm_ml::{ForecastModel, TimeSeriesData};
use ruv_swarm_ml::models::neural_bridge::NeuralForecastModel;

fn main() -> Result<(), String> {
    let mut model = NeuralForecastModel::new_mlp(3, 8, 1)?;
    model.fit(&TimeSeriesData {
        values: (0..16).map(|value| value as f32 / 16.0).collect(),
        timestamps: vec![], frequency: "step".into(), unique_id: "series-a".into(),
    })?;
    let forecast = ForecastModel::predict(&mut model, 2)?;
    assert_eq!(forecast.len(), 2);
    Ok(())
}
```

Prediction before fitting returns an error. Prediction uses the last observed input window and feeds predicted blocks back into that window. Training checks sample shapes and finite values. Output activations are linear so forecasts can represent values outside `[0, 1]`.

## Which model names work?

| Factory name | Current implementation |
| --- | --- |
| `MLP` | Dense feed-forward neural network |
| `DLinear` | Linear network baseline; no seasonal/trend decomposition |
| `MLPMultivariate` | Wider MLP; inputs are still one univariate series |

Other enum variants, including LSTM, TFT and N-BEATS, return an unsupported-model error. They are not silently mapped to MLPs. The available-model catalog and agent selection use the three implementations above. Static/dynamic covariates, probabilistic forecasts and loading model parameters are unsupported. Catalog memory numbers are planning hints, not measured allocation limits.

Ensembles require fitted members. No accuracy, benchmark leadership, automatic reasoning capability or token reduction follows from these APIs. The separate `ruv-swarm-ml-training` package contains experimental backend interfaces; it is not the implementation used here.

## Check it

```sh
cargo test --locked -p ruv-swarm-ml --lib
```

All-target/all-feature native coverage includes 59 tests on Rust 1.88. Tests compare forecasts against trained network outputs, check repeatability, reject unsupported models and validate memory-reservation replacement/switching without double counting. The optional ML path compiles in the base WASM crate's all-feature browser check. This does not establish browser training throughput. The bridge uses CPU training; CUDA is not enabled implicitly.

The versions in this checkout are **unpublished release candidates**. See the [workspace guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm) for all 14 packages and the [review record](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-rust-swarm.md) for checks, dependency findings and remaining limits.

![Animated map of the ruv-FANN packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)
