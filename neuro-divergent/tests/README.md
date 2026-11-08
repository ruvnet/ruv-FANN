# Forecasting checks

The registered tests call the shipped library. Data is deterministic; RNN fixtures
use seed 42. Forecasts are checked against series ids, dates, requested horizons,
observed holdout targets and independent numerical references.

Run from `neuro-divergent/` with Python-independent Rust tooling:

```bash
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
  cargo check --workspace --all-targets --all-features --locked
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
  cargo test --workspace --all-targets --all-features --locked
cargo run --example complete_workflow --locked
cargo run --example ensemble_forecasting --locked
cargo run --example advanced_features --locked
```

All existing registered test modules and five benchmark targets remain registered.
The regular suites have no ignored tests. Model-specific checks require the
`models` feature; registry checks require `registry`. Both are enabled by default.
The `gpu` feature is a reserved marker and adds no GPU implementation.

| Category | What the current checks execute |
| --- | --- |
| Workflow | Baseline and actual trained RNN fit/predict/reset; replacement history; unsorted panels; exact future dates and per-series horizons |
| Ensemble | Actual model columns joined by id and date, deliberately reversed custom-model output, explicit weighted arithmetic, heldout model selection |
| Cross-validation | Expanding chronological windows, observed targets/cutoffs, overlapping windows, unequal series lengths, first-prefix refit that prevents future-fit leakage |
| Persistence | Real baseline/RNN checkpoints, equivalent forecasts and timezone metadata across DST, continued independent use, malformed inputs, concurrent separate files, unsupported custom-state rejection |
| Preprocessing | Shipped scalers, training-only fitted statistics, inverse forecast units, causal lag/rolling alignment, explicit caller-selected missing-value treatment |
| Python-style contract | Roles of id/date/target columns, custom names, timestamp units, New York daily/month-end dates across DST, CSV/Parquet roundtrips and wide output; no Python-runtime or API-parity claim |
| Core/data units | Shipped schemas, result extraction, inclusive temporal filters, typed storage, invertible transforms and property-based scaler checks |
| Model units | Actual dense gradients, BPTT learning, recurrent state, batch equivalence and checkpoints; softmax gradients, normalization, Fourier features and quantile scores |
| Training/registry units | Actual loss gradients and optimizer-state restoration; actual metadata registration/search/removal rather than test-local replicas |
| Memory | Requested heap-byte deltas for actual RNN construction, repeated fitting, inference and ensemble drop, measured on an owning thread with monotonic counters |
| Scaling | Input/forecast storage bounds, panel/horizon shapes, independent threaded equivalence and real RNN width/horizon sweeps |
| Benchmarks | Actual seeded construction, BPTT training, batch inference, preprocessing and local baseline/RNN forecast operations |

The previous fixtures referenced unexported architectures and used test-local
implementations of registries, losses and models. Their duplicate assertions have
been replaced by checks of the real components above. Missing LSTM/GRU forecasting
and unregistered architecture creators are tested as explicit errors; they are
never substituted for the working fit/predict/CV/persistence checks.

Criterion benchmarks report local measurements only. Their names state the work
being measured; fitting and input preparation are outside inference timing.
`cargo test --all-targets` executes their smoke mode. Use `cargo bench --bench
inference_benchmarks` for measurements, recording CPU, compiler, features, thread
count, sample size and distribution when sharing results. No synthetic Python
measurements are supplied.

The memory allocator counts requested bytes in one dedicated thread. It never
resets shared counters, and computes signed differences between monotonic
allocation/deallocation totals. Measured objects are created and freed on that
thread. This detects retained allocations in those operations; it is not process
RSS, a whole-program leak proof, or a claim about arbitrary worker-thread allocators.

Files under `stress/` and `accuracy/` are historical, unregistered source fixtures.
The all-target command covers Cargo's registered targets; these directories have
not been silently added to or used as evidence for the current checks.
