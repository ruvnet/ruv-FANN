# Neuro-Divergent

Rust tools for time series forecasting, built on [ruv-FANN](https://github.com/ruvnet/ruv-FANN). Train the implemented RNN through a DataFrame adapter, compare it with honest naive baselines, and evaluate forecasts on chronological windows. Version 0.2.0 is a release candidate; no package has been published by this work.

![A chronological workflow showing the training cutoff, past-only preparation, future predictions, and evaluation](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/neuro-divergent/docs/assets/forecasting-flow.svg)

## Choose a crate

| Crate | Current purpose |
| --- | --- |
| [core](https://github.com/ruvnet/ruv-FANN/blob/main/neuro-divergent/neuro-divergent-core/README.md) | Schemas, checked Polars conversion, configuration, and FANN adapters |
| [data](https://github.com/ruvnet/ruv-FANN/blob/main/neuro-divergent/neuro-divergent-data/README.md) | Scaling, lag/rolling features, differencing, and validation |
| [models](https://github.com/ruvnet/ruv-FANN/blob/main/neuro-divergent/neuro-divergent-models/README.md) | Real recurrent training, activations, numerical helpers, and model checkpoints |
| [training](https://github.com/ruvnet/ruv-FANN/blob/main/neuro-divergent/neuro-divergent-training/README.md) | Losses, metrics, optimizers, schedules, and gradient utilities |
| [registry](https://github.com/ruvnet/ruv-FANN/blob/main/neuro-divergent/neuro-divergent-registry/README.md) | Metadata and bounded plugin descriptor checks |
| `neuro-divergent` | Forecast assembly, naive baselines, RNN adapter, chronological backtests, and pipeline checkpoints |

## Forecast a series

Supply a CSV with string `unique_id`, timestamp `ds`, and numeric `y` columns. The naive baseline repeats the last observation. It is a useful comparison for a trained model.

```rust,no_run
use neuro_divergent::{Frequency, NaiveForecaster, NeuralForecast, TimeSeriesDataFrame, TimeSeriesSchema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = TimeSeriesDataFrame::<f64>::from_csv(
        "series.csv", TimeSeriesSchema::default(), Some(Frequency::Daily),
    )?;
    let model = NaiveForecaster::new("Naive", 7)?;
    let mut pipeline = NeuralForecast::new(vec![Box::new(model)], Frequency::Daily)?;
    pipeline.fit(data)?;
    let forecasts = pipeline.predict()?;
    println!("{} future rows", forecasts.shape().0);
    pipeline.save("forecast.json")?;
    let restored = NeuralForecast::<f64>::load("forecast.json")?;
    Ok(())
}
```

With the `models` feature, wrap a real `models::models::RNN` in `ModelAdapter::new("RNN", model, lookback, frequency)`. Set the model's `input_size` to one for a univariate target. The adapter constructs real sliding training windows and forecasts from the latest lookback. Hidden size and forecast horizon are independent. LSTM/GRU forecasting constructors explicitly report unsupported operation; they are not aliases for RNN. Separate gate cells are available, but dormant transformer/N-BEATS source files are not exported implementations.

## Data and evaluation contracts

History is sorted within each series and checked for finite targets, unique timestamps, and regular frequency. Integer timestamps use Unix milliseconds; Datetime units retain their precision. Month-end and business-day offsets use calendar rules. All models in a pipeline use the same positive horizon. Model outputs are matched by series/time keys and must contain every expected finite future value. `predict()` uses retained training history; `predict_on(history)` accepts a replacement history for known series without changing that retained context. A context ending before its fitted cutoff is rejected.

`CrossValidationConfig::new(windows, horizon)` evaluates expanding chronological prefixes. Step size defaults to the horizon. If `test_size` is set, it must equal `horizon + (windows - 1) * step_size`. Cutoffs are actual observations, including separate cutoffs for unequal series. Even with `refit=false`, the first training prefix is fitted afresh so a previous full-data fit cannot leak into evaluation. Later windows update history without retraining weights. Per-series scalers fit only on training prefixes. Results contain identifiers, future dates, observed targets, cutoffs, and model forecast columns; `metrics_by_model()` scores actual held-out observations.

Pipeline checkpoints store supported baseline/RNN state, history, frequency, schema, and scaler parameters. Reads are bounded to 64 MiB; writes are atomic and unsupported custom-model persistence fails before writing an artifact. Recreate custom model support explicitly rather than expecting arbitrary Rust objects to deserialize.

## Calibrate forecast intervals from past errors

Use `ConformalCalibrator` after generating genuine out-of-sample backtest
predictions. Supply the last time at which every calibration target was known,
the forecast horizon, and coverage fractions such as `0.8` and `0.9`:

```rust
let config = neuro_divergent::conformal::ConformalConfig::new(
    calibration_as_of, 3, vec![0.8, 0.9],
)?;
let calibrator = neuro_divergent::conformal::ConformalCalibrator::fit(&cv, config)?;
let intervals = calibrator.apply(&forecasts, &forecast_origins)?;
```

Here `cv` is a `CrossValidationDataFrame`, `forecasts` is a `ForecastDataFrame`,
and `forecast_origins` maps each series ID to its last observed timestamp.
Calibration targets must be known by `calibration_as_of`; every forecast origin
must be at or after that boundary and every predicted timestamp after its origin.
Horizon means position in each sorted forecast window, rather than an inferred
calendar offset. Inputs are validated and never modified.

Errors are kept separate by series, model and horizon. For `n` calibration
errors and coverage `c`, the radius is the `ceil((n + 1) * c)`-th sorted absolute
error. Too few observations produce an error. Results use long-form rows with
point predictions, lower/upper limits, coverage and calibration metadata. Fixed
record, group, string and output budgets bound the work.

This is a transparent residual/conformal-style baseline. Correlated rolling
errors, changing fitted models or distribution shift can break the assumptions
needed for nominal coverage. The rank correction alone does not establish a
guarantee for arbitrary time series. Automatic pipeline intervals, quantile
regression and bootstrap configuration remain unsupported.

The [AirPassengers example](examples/conformal_intervals.rs) holds the final
12 observations out of calibration and reports coverage, width and interval
score for naive and seasonal baselines:

```sh
cargo run --manifest-path neuro-divergent/Cargo.toml --example conformal_intervals --locked
```

## Build and test

From the repository root:

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 cargo check --manifest-path neuro-divergent/Cargo.toml --workspace --all-targets --all-features --locked
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 cargo test --manifest-path neuro-divergent/Cargo.toml --workspace --all-targets --all-features --locked
cargo test --manifest-path neuro-divergent/Cargo.toml --workspace --doc --locked
```

See the [review and release evidence](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-forecasting.md) for exact gate results, security findings, and remaining capability boundaries. Migrated tests and examples use implemented models and real assertions; there are no validated accuracy, throughput, NeuralForecast parity, or state-of-the-art claims.

The `gpu`, `wasm`, `no_std`, and architecture marker features do not implement extra execution backends. GPU, automatic facade intervals/sampling, and unsupported validation options return errors rather than silently accepting a configuration. Explicit residual calibration is the separate API described above. Forecasting methods are synchronous. Registry generic factories/native executable loading, some advanced preprocessing, and unimplemented architecture families remain outside the supported pipeline. Public Polars types move to 0.45.1 and engine integration to ruv-FANN 0.2.2; downstream callers must migrate from 0.1.0.
