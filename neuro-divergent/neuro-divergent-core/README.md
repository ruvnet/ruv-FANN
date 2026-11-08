# neuro-divergent-core

Shared forecasting interfaces, configuration, checked Polars data containers, and adapters for ruv-FANN. Use this crate to define a series schema or integrate a FANN network. The separate `neuro-divergent` facade provides complete forecast assembly and chronological backtesting.

![Train on the past, forecast the future, and evaluate after the cutoff](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/neuro-divergent/docs/assets/forecasting-flow.svg)

## Define the columns

```rust
use neuro_divergent_core::data::TimeSeriesSchema;
let schema = TimeSeriesSchema::new("series_id", "timestamp", "value");
assert_eq!(schema.all_columns(), ["series_id", "timestamp", "value"]);
```

`TimeSeriesDataFrame::<f64>::from_csv(path, schema)` reads real CSV files and validates required columns. `from_polars` and `from_parquet` accept other table sources. Dataset conversion sorts observations, preserves timestamp units, rejects duplicate/null/non-finite values, and checks numeric static features remain constant within a series. Declared historical and known exogenous columns become aligned numeric matrices. Time ranges and missing-value summaries come from observed data; fixed regular frequencies are inferred when possible.

`GenericModelConfig::<f64>::save(path)` and `load(path)` serialize validated JSON instead of panicking. Loading is bounded to 16 MiB. System configuration validation reports invalid counts, memory thresholds, unsupported GPU/AMP flags, and I/O limits. Prefer module-qualified imports when data, traits, and integration names overlap.

The FANN adapter propagates checked engine errors and describes actual hidden layers/activations. The training bridge creates windows from the network's real input/output dimensions, includes the last complete window, records training loss, and honors the configured stopping target and exposed learning rate. Its basic rate-based trainer is explicit: unsupported cascade, validation/minibatch/early-stopping options return errors. For chronological evaluation use the facade backtester.

## Validate

From the repository root, use the shared feature configuration to avoid duplicate Polars builds:

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 cargo test --manifest-path neuro-divergent/Cargo.toml --workspace --all-targets --all-features --locked
```

The 0.2.0 candidate uses Polars 0.45.1 and ruv-FANN 0.2.2, which change public APIs. The core trait remains distinct from the model and facade traits. Required-column validation alone does not certify a whole dataset. Calendar frequency inference beyond fixed regular offsets and advanced preprocessing are limited; GPU/freestanding execution is not implemented. See the [family review](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-forecasting.md) for exact results and release evidence.
