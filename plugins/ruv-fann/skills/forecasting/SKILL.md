---
name: forecasting
description: "Build and validate neuro-divergent univariate RNN or baseline forecasts with keyed dates, chronological splits, past-only scaling, explicit conformal residual intervals, and bounded checkpoints. Use for forecasting APIs, training correctness, evaluation leakage, calendar behavior, or persistence."
---

# Train and evaluate supported forecasts

Work in the independent `neuro-divergent/` workspace. Read its README, the model implementation and `docs/review-forecasting.md` before choosing an architecture. Supported forecasting includes a genuinely trained univariate RNN and naive/seasonal baselines. LSTM/GRU forecasting constructors and unimplemented architecture families return unsupported errors; separate low-level gate cells do not establish a complete forecasting model.

## Define the data contract first

1. Record series IDs, target column, timestamp type, frequency, horizon, lookback, seed and metric. The default schema is string `unique_id`, timestamp `ds`, numeric `y`.
2. Validate finite targets, unique timestamps and regular frequency after sorting within each series. Integer timestamps are Unix milliseconds; datetime units retain their precision.
3. Use the same positive horizon for every pipeline model. Check exact future series/time keys and finite output values rather than accepting only a row count.
4. Keep fitted transforms and model training strictly before each evaluation cutoff. Never fit a scaler on all observations before splitting.

## Establish a runnable baseline

This example reads an existing CSV; it does not generate or overwrite the user's data:

```rust
use neuro_divergent::{Frequency, NaiveForecaster, NeuralForecast,
    TimeSeriesDataFrame, TimeSeriesSchema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = TimeSeriesDataFrame::<f64>::from_csv(
        "series.csv", TimeSeriesSchema::default(), Some(Frequency::Daily),
    )?;
    let baseline = NaiveForecaster::new("Naive", 7)?;
    let mut pipeline = NeuralForecast::new(vec![Box::new(baseline)], Frequency::Daily)?;
    pipeline.fit(data)?;
    let forecast = pipeline.predict()?;
    println!("{} future rows", forecast.shape().0);
    Ok(())
}
```

Use a local path dependency while evaluating the checkout candidate. With the `models` feature, the documented RNN is `models::models::RNN` wrapped by `ModelAdapter::new(name, model, lookback, frequency)`. Set RNN `input_size` to one for the univariate target. Read `neuro-divergent/examples/complete_workflow.rs` for the exact current constructor/configuration; hidden width and horizon are independent.

## Implement and evaluate

- Train through real sliding windows. Validate loss, gradient dimensions, finite parameter updates and seeded repeatability. For recurrent changes, compare gradients with finite differences and distinguish full from truncated backpropagation through time.
- `predict()` uses retained training history. `predict_on(history)` checks replacement history for known series without replacing that retained context; a context ending before the fitted cutoff rejects.
- Join ensemble forecasts by series ID and timestamp. Model column names and horizons must match exactly; never average unrelated positional rows.
- `CrossValidationConfig::new(windows, horizon)` uses expanding chronological prefixes. Its default step is the horizon. If supplied, `test_size` must equal `horizon + (windows - 1) * step_size`.
- Even `refit=false` fits the first prefix afresh. Later windows update history without retraining weights. Confirm cutoffs per series, especially for unequal series lengths.
- `metrics_by_model()` must score actual held-out targets. Include naive and seasonal-naive comparisons on the same cutoffs before interpreting a learned model's error.

## Calibrate intervals explicitly from past residuals

Read `neuro-divergent/src/conformal.rs`, `conformal_data.rs`, and
`conformal_apply.rs`. `ConformalCalibrator` is immutable and separate from the
fitted forecasting pipeline. Supply genuine out-of-sample `CrossValidationDataFrame`
predictions, their observed targets, and a declared calibration `as_of` boundary.
Each `(series, cutoff)` window must contain exactly H distinct target timestamps;
all rows must satisfy `cutoff < target_time <= as_of`. Horizon is the one-based
ordinal position after sorting each complete window, not an inferred calendar gap.

The API sequence is:

```rust
// cv: CrossValidationDataFrame<T>; point: ForecastDataFrame<T>.
// as_of: DateTime<Utc>; origins: BTreeMap<String, DateTime<Utc>>.
use neuro_divergent::{ConformalCalibrator, ConformalConfig};
let config = ConformalConfig::new(as_of, 7, vec![0.80, 0.90])?;
let calibration = ConformalCalibrator::fit(&cv, config)?;
let intervals = calibration.apply(&point, &origins)?;
```

Coverage values are distinct fractions strictly between zero and one. Absolute
errors are calibrated independently for each `(series, model, horizon)` group.
For n residuals, the symmetric radius is sorted error at one-based rank
`ceil((n + 1) * coverage)`. Insufficient samples (`rank > n`) return an error;
the radius is never silently clamped or pooled across unrelated groups.

`apply` requires an explicit origin for exactly every forecast series, each
`origin >= as_of`, with all targets strictly after their origin. Forecast schema,
model names, timestamp representation, and H must match calibration. Unknown
groups, incomplete windows, existing intervals, null/nonfinite values, interval
overflow, and exceeded resource budgets reject. Point forecasts remain unchanged;
the separate long-form result includes point/lower/upper, coverage, horizon,
sample count, forecast origin, and calibration boundary. Inspect `groups()` for
the residual sample counts and radii.

Finite-sample rank correction alone does not guarantee time-series coverage.
Classical marginal coverage relies on appropriate exchangeability of calibration
and future residual scores. Rolling windows can be correlated and come from
different fitted models; distribution shift and model retraining can invalidate
calibration. Report empirical coverage and width on held-out future data, and
refit calibration when the model or residual distribution changes. The frame API
cannot detect hidden model retraining. There is no automatic online update or
calibrator persistence API.

This explicit residual baseline does not enable automatic builder prediction
intervals, bootstrap or quantile interval methods, probabilistic sampling, or
adaptive ACI/PID/AcMCP algorithms. Unsupported builder paths still return errors.
Read [evaluation and migration details](references/evaluation.md) for budgets and
leakage regressions.

## Persist supported state

Pipeline `save(path)`/`load(path)` retain supported baseline/RNN parameters, history, frequency, schema and scaler state. Loads cap input at 64 MiB. Writes are atomic; unsupported custom-model state must fail before replacing a file. Use a fresh task-owned output path, never a device/FIFO or arbitrary user artifact. Do not serialize arbitrary Rust objects or enable native executable plugin loading to make an unsupported model appear loadable.

## Verify

Run from the checkout root. The workspace is substantial; bound jobs and ensure adequate disk before broad checks:

```sh
CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 cargo test --manifest-path neuro-divergent/Cargo.toml --workspace --all-targets --all-features --locked
cargo test --manifest-path neuro-divergent/Cargo.toml --workspace --all-features --doc --locked
cargo test --manifest-path neuro-divergent/Cargo.toml --test conformal_intervals --locked
```

Use the helper's `plan --profile forecasting` to inspect its bounded offline profile before execution. Browser/GPU markers do not add an implemented forecasting execution backend. Exogenous-model mapping, automatic/probabilistic interval builders and sampling, GPU/AMP training and arbitrary executable registry plugins are outside this supported facade.

## Acceptance and diagnosis

Check real learning on a fixed fixture, finite outputs, exact future keys, leakage-free backtests and save/load parity. Date tests should cover month ends, business days, timezone metadata and DST transitions. A daily calendar step can be 23 or 25 elapsed hours. Ambiguous/nonexistent local timestamps must produce typed errors.

For an unsupported architecture, return the unsupported result and describe the implemented alternative. For an unexpectedly strong score, inspect leakage, overlap with pretrained data, key alignment and the naive baseline. Never rename the RNN to imply another architecture or claim SOTA accuracy from an unrelated paper.

Read [evaluation and migration details](references/evaluation.md) when changing metrics, transforms, calendars or checkpoint semantics.
