# Forecasting package review and 0.2.0 candidate

Reviewed on 2026-11-07; final validation completed on 2026-11-08. Scope: the six crates in `neuro-divergent/`, their dependency graph, working numerical components, input boundaries, tests, and package guides. These sources are release candidates; nothing was published by this work.

## Package status

| Package | Candidate | Final library tests | Current boundary |
| --- | --- | --- | --- |
| `neuro-divergent` | 0.2.0 | 50 unit tests; 52 integration tests | Real forecast assembly, naive/seasonal baselines, univariate RNN bridge, chronological backtests, scaling, and bounded pipeline checkpoints |
| `neuro-divergent-core` | 0.2.0 | 40 | Real configuration persistence, date ranges, sorted numeric/exogenous conversion, static validation, and dimensioned FANN training windows |
| `neuro-divergent-data` | 0.2.0 | 34 | Scaling, past-only features, correct ordinary/seasonal differencing inverse, arrays and validation |
| `neuro-divergent-models` | 0.2.0 | 31 | Full RNN BPTT with independent hidden/horizon dimensions, seeded initialization, bounded versioned parameter checkpoints |
| `neuro-divergent-registry` | 0.2.0 | 42 | Metadata/catalog and descriptor checks; generic model factory and native executable plugin loading remain unsupported |
| `neuro-divergent-training` | 0.2.0 | 51 | Checked optimizers/losses, stable norms, schedules, and serializable optimizer/scheduler continuation state |

The complete all-target/all-feature gate now passes **354 tests, zero ignored tests, all three example targets, and 15 Criterion smoke cases**. This includes facade 50, core 40, data 34, models 31, registry 42, and training 51 library tests, plus real external integration, numerical, memory, and scaling tests. **16 all-feature workspace doctests also pass**, with zero ignored cases. These checks establish numerical and implementation contracts, not predictive accuracy or NeuralForecast parity.

The model traits in the facade, core, models, and registry crates are distinct interfaces. Their similarly named DataFrame/configuration types are not interchangeable. Only the RNN implementation is exported as a forecasting model; `LSTM` and `GRU` forecasting constructors now return explicit unsupported-operation errors rather than aliasing RNN. Separate LSTM/GRU gate cells exist and are tested. Basic, advanced, transformer, and specialized architecture source directories are not exported. Feature markers do not turn those files into validated implementations.

## Baseline failures and misleading behavior

The original facade resolved published internal crates rather than the neighboring source directories. Local changes could therefore be skipped, while old Polars/Arrow versions remained in the graph. Local workspace builds first failed in Arrow 52 because its `quarter()` call conflicted with a newer Chrono trait method. The data crate also declared four source modules that did not exist; its `deny(missing_docs)` policy exposed undocumented public enum fields. An all-features check later found another nonexistent models `test_utils` module and stale declarations for two nonexistent model examples.

Once the graph was coherent, library test compilation exposed obsolete FANN builder/trainer calls, missing configuration serialization, an invalid gradient-norm expectation, malformed validation fixtures, and exact floating-point equality in a prediction-interval test. The documentation tests contained nonexistent model builders, incompatible facade/model traits, and incorrect CSV signatures. These failures were repaired using real APIs, meaningful assertions, or removal of stale declarations. No missing architecture was replaced with a placeholder implementation.

The facade previously returned successful empty forecasts with horizon one and empty cross-validation results with cutoffs generated from the current wall clock. Those placeholders have been replaced by real keyed model output assembly and expanding-window evaluation. Empty, incomplete, duplicate, misdated, wrong-horizon, and non-finite model outputs return errors.

## Implemented improvements

### Dependency coherence and attack surface

Internal facade dependencies now specify both local paths and matching 0.2.0 versions. All engine consumers resolve the patched local `ruv-fann` **0.2.2**; `cargo tree --workspace --locked -i ruv-fann` shows no published 0.1.x copy.

Polars moved from 0.35 to **0.45.1** in the facade, core, and optional model dependency. This is a minimum targeted migration away from the vulnerable `fast-float` implementation, not a claim to track the newest Polars release. The implementation now handles `Column`, `PlSmallStr`, string access, and `CsvReadOptions` APIs.

Removed unused Arrow/Polars and statistical/interpolation dependencies from the data crate, an unused core dependency from training, unused Candle GPU dependencies, unused Plotly/pprof/jemalloc development dependencies, unused nalgebra dependencies, and unused binary serializer dependencies in models/registry. The registry uses `std::time::Instant` instead of the unmaintained wrapper. Compatible locked dependencies were updated, including affected compression and parsing chains. Optional feature names that remain as markers are documented honestly; enabling `gpu` does not report an available GPU implementation.

### Numerical and performance work

- Softmax backward now computes `y * (g - dot(g, y))` instead of constructing a dense Jacobian: linear rather than quadratic work/storage. Tests compare against the dense formula, finite differences, empty/saturated inputs, and invalid gradient shape. No wall-clock speedup is claimed.
- Matrix multiplication validates nonempty rectangular dimensions and uses an `i-k-j` loop to traverse output rows contiguously. Sliding windows reject zero size/step and avoid overflowing size arithmetic. MSE/MAE and chronological splitting reject invalid numerical inputs rather than silently truncating or propagating NaN.
- StandardScaler computes population variance with a stable single-pass update. Scale-only mode uses standard deviation around the actual mean, rather than root-mean-square around zero. Constant series and large-offset series are covered; invalid refits retain prior fitted statistics.
- Gradient norm uses `hypot`, and clipping scales finite gradients without overflowing squared sums, including when the true norm exceeds the numeric type's finite range.
- Adam validates gradients and stored parameter shape before updates. AMSGrad maxima are reconstructed on restore into a fresh optimizer, and exact resumed steps match uninterrupted steps. The caller must recreate the same beta, epsilon, and AMSGrad settings; optimizer state alone is not a complete training checkpoint.
- Plain cosine annealing reaches its floor without implicitly restarting. Explicit warm restarts derive cycles from absolute steps, so repeated/backward queries are deterministic. Zero-length cycles, invalid restart factors, large constant cycles, and scheduler restoration have regressions.

### Forecasting correctness and engine errors

Lag, rolling, and calendar feature blocks align to their common final timestamps after their different warmups. Rolling windows end before the target observation, fixing direct target leakage. A zero rolling window is rejected. This only protects features generated by the crate; callers must validate external covariates and fit scalers on the past.

Configured value ranges now activate the corresponding validator. Temporal and outlier fixtures reflect the builder's sorting and the statistics actually computed.

Core CSV loading now reads real files instead of panicking. Dataset conversion respects Date/Datetime units, preserves nanosecond precision, converts numeric targets, and rejects null/non-finite values instead of dropping timestamp and target rows independently. Facade CSV, timestamp range, and date filtering have regressions across millisecond, microsecond, and nanosecond columns; legacy Int64 timestamps retain Unix-millisecond semantics.

FANN adapters propagate the engine's checked inference errors. Configuration serialization and builder/trainer tests were updated to the actual 0.2.2 API. Recurrent gate regressions check successful gates and corrupt-network errors. RNN training now uses full backpropagation through time with a real FANN output projection and shared Adam updates; hidden size and horizon are independent. Seeded initialization, finite-difference gradients, learning, and checkpoint continuation are tested by the model owner. The facade constructs real sliding-window samples and converts point outputs to checked series/time rows.

### Plugin descriptor boundaries

Descriptor reads enforce canonical directory allow/block policies, regular-file checks, and a 1 MiB bound before parsing. Blocked paths take precedence, including resolved symlinks. The native-loader entry point authorizes its neighboring descriptor independently of the binary, closing a descriptor-symlink bypass. Plugin compatibility compares parsed semantic versions and rejects malformed bounds rather than comparing version strings lexicographically. SHA-256 verification uses a bounded streaming buffer and checksum policy requires a checksum. Unsupported signature and sandbox policies fail closed.

These checks do not authenticate a mutable descriptor, and canonical path checks do not defend against a hostile local process racing file replacement. Native executable plugin loading remains unsupported. Tests cover allow/block decisions, direct and neighboring-descriptor symlink escape, oversized metadata, missing/matching/tampered checksums, unsupported policies, semantic version ordering, and malformed versions.

## Validation and security evidence

Toolchain: Rust/Cargo **1.99.1**, `cargo-audit` **0.22.2**. Run from the repository root:

```sh
export CARGO_INCREMENTAL=0
export CARGO_BUILD_JOBS=1
export CARGO_PROFILE_DEV_DEBUG=0
cargo check --manifest-path neuro-divergent/Cargo.toml --workspace --all-targets --all-features --locked
cargo check --manifest-path neuro-divergent/Cargo.toml --workspace --no-default-features --locked
cargo test --manifest-path neuro-divergent/Cargo.toml --workspace --all-targets --all-features --locked
cargo test --manifest-path neuro-divergent/Cargo.toml --workspace --all-features --doc --locked
cargo audit --file neuro-divergent/Cargo.lock
cargo tree --manifest-path neuro-divergent/Cargo.toml --workspace --locked -i ruv-fann
```

The final all-target/all-feature test command passed with `CARGO_INCREMENTAL=0`, one build job, and development/test debug information disabled. The focused tests and broad gate use the same feature configuration to avoid duplicate Polars caches. Package guides contain tested code snippets. The self-contained animated [workflow SVG](../neuro-divergent/docs/assets/forecasting-flow.svg) has valid XML, a title/description, static labels, no scripts/external assets, and a reduced-motion rule. README asset/review links use absolute repository URLs so leaf crate archives do not depend on sibling files. Those new URLs become available after the changes reach the referenced main branch.

The original lockfile audit reported **19 vulnerabilities**, plus maintenance/unsoundness warnings. The updated lockfile reports **zero vulnerabilities** and one maintenance warning: `bincode 1.3.3`, [RUSTSEC-2025-0141](https://rustsec.org/advisories/RUSTSEC-2025-0141.html). Bincode remains through the root engine. Unused training checkpoint I/O dependencies and facade development dependencies were removed; the training checkpoint feature now enables actual serializable optimizer/scheduler state. No advisory was suppressed. A serializer migration needs an explicit format/versioning plan rather than silently changing persisted bytes.

Audit database revision: `309392b18f6b17022f82c701f300433126901068` (1,422 advisories; last updated 2026-11-06). Reviewed lockfile SHA-256: `50e70e307a2c5ffd17a43d5ce73320860ead04f8a07518f1400d9cd1b7beccdf`. The lockfile is required to reproduce these dependency results; semver ranges alone do not establish an equivalent audit result.

The initial expanded check found 211 historical facade integration/performance compile errors. Those suites and examples were migrated to real implemented models, baseline or independently coded drift forecasters, checked target/date contracts, actual loss/optimizer operations, and real checkpoint files. They were not disabled or ignored. The expanded check and complete runtime gate subsequently passed; the final receipt records 354 tests without ignored cases. The runtime suite exposed real SGD/loss validation and higher-order differencing bugs, which have production fixes and regression coverage rather than weakened expectations. Independent review also exposed FIFO-open blocking and timezone calendar drift: checkpoint/config readers now preflight regular files and use Unix nonblocking opens followed by file metadata checks, while calendar arithmetic preserves IANA local dates and rejects ambiguous/nonexistent local times. NY DST, month-end, Tokyo date-boundary, FIFO race, and snapshot round-trip tests pass.

### Closed forecasting blockers

- `predict` retains successful raw training history; `predict_on` uses replacement history for known series without mutating that context. Schema, frequency, finite targets, duplicate timestamps, horizon, future dates, and keyed model output completeness are checked. Failed model refits and mutable model access invalidate the pipeline's fitted state.
- Chronological backtesting computes expanding prefixes and actual per-series cutoffs, supports unequal series and explicit refit policies, and always trains the initial prefix afresh even if the caller previously fitted future data. Per-series affine scalers fit only those training observations. Result tables contain actual held-out targets and forecast columns; pooled/per-model metrics score those rows.
- Supported baseline/RNN checkpoints persist model parameters, history, frequency, schema, and scaler state. Writes use private temporary files and atomic rename; reads are bounded to 64 MiB and validate version, numeric precision, model metadata, and scaler shape. Unsupported custom persistence fails before creating or replacing an artifact.
- Core generic configuration save/load no longer panic, core ranges no longer return MIN/MAX sentinels, numeric exogenous/static features are checked and aligned, and missing statistics come from real cells. FANN bridge windows use actual network dimensions and include the final complete window; unsupported bridge modes return errors.
- Ordinary and seasonal integration reconstruct each difference level using its own original-prefix seeds. Polynomial tests cover orders zero through three and seasonal periods one through three. Period/order arithmetic and finite arithmetic are checked; timestamp transformation no longer repeatedly allocates the full timestamp list inside its loop.


### Exact final acceptance receipt

The all-target command ran from `neuro-divergent/`:

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 cargo test --workspace --all-targets --all-features --locked
```

It exited zero. The saved build receipt records 354 passed tests, zero ignored, 15 Criterion cases, and a separate successful all-feature documentation run with 16 doctests; all six crate libraries, facade integration/unit/performance suites, and three examples were included.

The final isolated archive run also exited zero on 2026-11-08. All **26 candidates across five workspaces**, including all six forecasting crates, passed normal Cargo archive verification. The driver packaged dependent families in topological order against the exact previously verified root archive; no verification was skipped. All 34 recorded commands exited zero. Every archive was independently rechecked for SHA-256, size, license payload, and read-only permissions, and all five live checkout lockfile hashes remained unchanged. See the [release process](release-process.md) for the staging contract and reproduction command.

Local immutable evidence is in `/workspace/scratch/release-candidates-final/`: `archives/`, `receipt.json`, `receipt.sha256`, and `publication-order.json`. The receipt SHA-256 is `a9baef95b2fe388e72a5edeebade7635dc54f6481a12b8f0bf6e69d85009a611`; its source snapshot SHA-256 is `00be5a87ec94900e8622ef0a379ba4ba1b32c68c849510fbee1c5d4746373d4c`. These are local release artifacts, not a registry publication.

## Primary sources and implementation choices

- [PyTorch CPU softmax implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/SoftMaxKernel.cpp): the vector-Jacobian product motivates the linear softmax backward implementation; local finite-difference tests verify it independently.
- [Scikit-learn StandardScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): scale-only semantics still require standard deviation, and constant columns need a defined scale. The local implementation retains its own API.
- [PyTorch cosine annealing](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) and [warm restarts](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html): explicit restart semantics inform the scheduler correction; no API equivalence is claimed.
- [Nixtla NeuralForecast cross-validation](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/cross_validation.html): evaluate past-to-future windows with explicit cutoff/refit/step semantics. The implemented backtester uses explicit past-only prefixes, observed cutoffs, and per-window fit policies; this is not a claim of complete API compatibility.
- [PatchTST official implementation](https://github.com/yuqinie98/PatchTST) and [paper](https://arxiv.org/abs/2211.14730): a candidate for a separate implemented-and-evaluated architecture, not evidence that the dormant transformer directory currently works.
- [Google TimesFM primary repository](https://github.com/google-research/timesfm): the current upstream research line includes TimesFM 3.0. Foundation forecasting would require explicit model weights, data/feature contracts, inference backend, licensing review, resource limits, and chronological evaluation. No foundation model was added in this change.

Naive/seasonal baselines and multiple-cutoff evaluation are now implemented. Further accuracy research should compare the trained RNN against these baselines on disclosed datasets. Architecture names, catalog entries, placeholder files, or a successful build do not establish state-of-the-art capability.

## Release concerns

The 0.2.0 candidate deliberately changes public Polars types, FANN error behavior, empty-result placeholders, feature exports, and registry timing types. Review downstream migration and require the complete recorded gate before publication. Publish/verify ruv-FANN 0.2.2 first, then the independent 0.2.0 leaf crates and engine-dependent core/models/training, then the facade. Path dependencies are stripped by Cargo publication and the matching registry versions must already exist.

The supported univariate pipeline now includes adaptation, assembly, backtesting, and persistence. Separate LSTM/GRU forecasting architectures, facade exogenous-model mapping, native executable plugins, GPU/AMP execution, probabilistic intervals/sampling, and advanced preprocessing remain outside that supported contract and return explicit errors where requested. `gpu`, `wasm`, `no_std`, `python-compat`, `mixed_precision`, and architecture markers do not certify additional implementations. Forecasting APIs remain synchronous. Lint warnings include missing facade documentation and overlapping glob re-exports. Passing library checks do not establish MSRV, browser/freestanding support, real-world forecast accuracy, or benchmark gains.
