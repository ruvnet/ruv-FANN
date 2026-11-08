# Prepared inference and historical interval calibration

This iteration implements two research-led changes: immutable prepared CPU
inference and explicit forecast intervals built from past errors. The
[research review](research-sota-2026-11-08.md) records current sources, revisions,
checkpoint licenses and the alternatives deferred. [ADR-007](adr/ADR-007-prepared-inference-and-conformal-calibration.md)
describes the interfaces and statistical boundaries. The machine-readable
[validation report](sota-validation-2026-11-08.json) records command scope and hashes.

## Reusable inference

`Network::prepare()` owns a validated snapshot of weights, connections and
activation settings. Its parameters can be shared between threads, while each
caller reuses an independent workspace. Regular neuron caches are not copied.
Invalid graphs, non-finite parameters and excessive allocation estimates fail
before packing; unsuccessful inference leaves caller output unchanged.

Dense ordered connections use contiguous scalar arithmetic. Sparse, shuffled
and duplicate connections preserve their accumulation order. Production code
adds no unsafe operations, dependency, external engine or quantization format.
Default preparation limits include one workspace in a 64 MiB estimate; allocator
overhead and the original network are outside that estimate.

The 13 new tests cover both f32/f64, all 21 activations, snapshot isolation,
thread sharing, zero steady-state allocations, invalid values, graph structure,
workspace reuse and error behavior. Existing library gates also pass.

### Local measurements

The final Criterion run used the same weights and inputs for both paths, seed
42, f32, default release optimization, no explicit SIMD/native CPU flag, 20
samples, one-second warm-up and two-second measurement windows. Rust compilation
was idle during measurement. The container reports AMD EPYC 9V74 with a four-core
CPU quota. These are mean estimates with bootstrap confidence intervals retained
in [the raw result record](benchmarks/prepared-inference-2026-11-08.json).

| Shape | Existing call | Prepared call | Lower mean latency | Setup + destruction | Estimated break-even |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 → 7 → 3 | 106.39 ns | 102.90 ns | 3.3% | 330.26 ns | 95 calls |
| 100 → 50 → 25 → 10 | 6.20 µs | 4.70 µs | 24.2% | 16.08 µs | 11 calls |
| 784 → 128 → 64 → 10 | 100.44 µs | 94.77 µs | 5.7% | 279.57 µs | 50 calls |

Preparation timing includes parameter packing, workspace allocation and object
destruction. Break-even is **calculated** from those means divided by the mean
per-call saving; a complete cold-start workload was not separately timed.
The dimension-784 case uses random seeded weights, not MNIST data. All three
fixtures assert identical output values before timing. Sparse latency, other
processors, large batches and application-level speed remain unevaluated.

Reproduce the benchmark:

```sh
CARGO_BUILD_JOBS=1 CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0 \
  cargo bench --locked --no-default-features --bench neural_network -- \
  prepared_inference --sample-size 20 --measurement-time 2 --warm-up-time 1
```

## Forecast intervals and held-out evaluation

`ConformalCalibrator::fit` consumes real `CrossValidationDataFrame` predictions.
It separates absolute errors by series, model and ordinal horizon. A coverage
fraction `c` with `n` observations selects sorted rank `ceil((n + 1) * c)`;
insufficient observations produce an error. Every target must be observed by
the explicit calibration boundary. Applying intervals requires an origin for
each series at or after that boundary and strictly later forecast timestamps.

The 18 new tests exercise rank calculations, isolation, input budgets, strings,
timestamps, schemas, nulls, duplicates, numeric overflow and temporal leakage.
Independent review found that converting a large integer to f64 could silently
erase a residual. The implementation now rejects lossy Int64/UInt64 conversion;
dedicated regressions include exact large integers and the problematic pair.

This static residual method is not AcMCP, ACI or PID-Expert. Rolling errors can
be correlated and come from different fitted procedures. Chronology and the
corrected rank do not by themselves establish exchangeability, simultaneous
trajectory coverage or nominal coverage after a distribution shift. Recalibrate
after changing the model, preprocessing or relevant data regime. Existing
automatic interval, quantile and bootstrap builder options remain unsupported.

The example uses the standard 144 AirPassengers monthly observations and
calibrates only outcomes through December 1959. The final 12 months are scored
in four successive three-month forecasts. Later forecast origins may use
already observed earlier evaluation months; none enter the frozen calibration.
Each series/model/horizon has 18 calibration errors and four evaluation outcomes.

| Model | Horizon | Empirical 80% coverage | Empirical 90% coverage | 80% width | 90% width |
| --- | ---: | ---: | ---: | ---: | ---: |
| Last value | 1 | 3/4 | 4/4 | 152 | 202 |
| Last value | 2 | 4/4 | 4/4 | 256 | 304 |
| Last value | 3 | 4/4 | 4/4 | 304 | 394 |
| Seasonal 12 | 1 | 2/4 | 2/4 | 114 | 118 |
| Seasonal 12 | 2 | 4/4 | 4/4 | 114 | 118 |
| Seasonal 12 | 3 | 3/4 | 4/4 | 114 | 136 |

The seasonal first-horizon 90% interval covered only **50%** of these outcomes.
This small example demonstrates inspection and limitations rather than an
accuracy or coverage improvement. Proper interval scores, counts and dataset
attribution are retained in [the evaluation record](benchmarks/airpassengers-conformal-2026-11-08.json).

```sh
CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  cargo run --manifest-path neuro-divergent/Cargo.toml \
  --example conformal_intervals --all-features --locked
```

## Packaging and scope

The nine existing lockfiles remain unchanged. The swarm's actual ordinary and
SIMD WASM builds were repeated against the new root source. Their binary and
binding bytes reproduced exactly; the current-source receipt changed. Clean
Node 22.13 and 24 consumers passed real SQLite, inference, forecasts, MCP and
strict public type checks, with zero audit findings. Previous immutable
archives/receipts were preserved.

Hosted checks also exposed two portability defects: macOS reports a canonical
`/private/var` path for a `/var` temporary directory, and Windows mixes native
and POSIX separators before source hashing. Resolver checks now compare real
file identities; source paths are normalized before deduplication and sorting.
Both failures were reproduced before the fixes. A follow-up Windows checkout
also converted 34 extensionless license files to CRLF. Explicit LF rules now
preserve every one of the 386 hashed source inputs, including with Git's
`core.autocrlf=true` and `core.eol=crlf`. A genuine packing attempt fails before
the rules and passes afterward. Manual LF-to-CRLF source edits still invalidate
the receipt; the verifier does not normalize content or bypass integrity checks.

The updated 59 security/API tests and four tooling tests pass, including real
directory-alias and Git checkout cases. A fresh source-matching WASM receipt and
archive were generated after changing the verifier. The later LF policy and
test additions leave that source hash, binary bytes and npm archive unchanged.

Cargo's ordinary package verification also passed for the core and all six
forecasting crates: seven archives and 13 successful commands. The validation
JSON records their exact SHA-256 values and the immutable source snapshot.
Other package families retain the earlier verification recorded for their
unchanged source; a new full 26-crate release was not published.

Claude plugin 0.1.1 adds instructions for both APIs. Its 39 tests pass; historical
0.1.0 live-model evidence remains explicitly separate. Exact current artifact
and host-validation results are recorded in the validation JSON.

No pretrained checkpoint, Python/PyTorch runtime, ISA-specific engine, external
provider configuration or registry publication is part of this change. Current
foundation-model candidates and their license/runtime requirements are listed
in the research review for a separately reproducible comparison.

## WebAssembly extension

A follow-up exposes both engines through the generated `ruv-swarm` bindings in
ordinary and SIMD binaries. `WasmNeuralNetwork.prepare()` creates an immutable
snapshot; `WasmPreparedNetwork.from_network_json()` imports actual native
`Network<f64>` exports, including sparse/reordered connections and all 21 native
activation variants. JavaScript uses Float64 arrays. `run_into()` reuses numeric
storage, checks intrinsic array metadata, and leaves caller output unchanged on
failure. Trusted intrinsic getters are captured at preparation time; each call
still reads the real array slots and current WASM memory buffer. Shared,
detached, forged and module-owned arrays remain rejected.

The interval algorithm now lives in `ruv_fann::calibration`, which depends only
on the Rust standard library. Both the native DataFrame adapter and
`WasmConformalCalibrator` delegate to it. Native schema and timestamp precision
remain intact; the WASM wire format uses safe-integer Unix milliseconds.
WASM input is limited to 4 MiB, with 10,000 records, 1,000 groups and a
conservative 8 MiB output budget. The statistical limits described above still
apply. The [JavaScript guide](../ruv-swarm/npm/README.md) documents both interfaces.

Independent review found and resolved three defects: changing JavaScript array
lengths could bypass preprocessing limits; raw string exports copied data before
the size check; and the default JSON float parser changed an exact `2^98`
integer token by one ULP. A failing numeric regression was recorded before
enabling `serde_json/float_roundtrip`. That existing feature adds no packages.

### Measured WASM calls

The first working binding repeatedly looked up JavaScript intrinsic functions.
A quiet timing run identified that overhead; caching the same trusted getters
reduced the smallest fixture's prepared-call mean latency by about 90% relative
to that first binding. Validation remains in place.

The final Node 24.22.1 run used one AMD EPYC 9V74 container with a four-core quota,
synthetic fixed weights and inputs, 20 interleaved samples per API, and a warm-up
batch before sampling. Times include JavaScript boundary checks, copies and
result allocations. The ordinary and `+simd128` binaries expose the same scalar
prepared inference kernel; this is not a newly vectorized Float64 backend.

| Binary | Layers | Legacy `run` | Prepared `run` | Prepared `run_into` |
| --- | --- | ---: | ---: | ---: |
| core | 5 → 7 → 3 | 0.458 µs | 1.551 µs | 2.251 µs |
| core | 100 → 50 → 25 → 10 | 8.545 µs | 6.564 µs | 6.805 µs |
| core | 784 → 128 → 64 → 10 | 142.900 µs | 94.889 µs | 93.953 µs |
| simd | 5 → 7 → 3 | 0.420 µs | 1.500 µs | 2.179 µs |
| simd | 100 → 50 → 25 → 10 | 8.977 µs | 6.988 µs | 7.207 µs |
| simd | 784 → 128 → 64 → 10 | 148.402 µs | 103.647 µs | 103.969 µs |

For the medium and large shapes, prepared `run_into` measured about 20–34% lower
mean latency than legacy `run`. The smallest shape takes roughly five times as
long: its stronger boundary checks cost more than the tiny arithmetic task.
Choose using a workload measurement. Every compared output matched exactly on
these six fixtures. Short single-host timings do not establish general speed or
forecast accuracy. Preparation has one observed setup sample per fixture, not a
setup-cost distribution or an end-to-end amortization experiment.

Reproduce after building both artifacts:

```sh
node ruv-swarm/npm/examples/benchmark-prepared.mjs > wasm-timings.json
CHROMIUM_BIN=/usr/bin/chromium node scripts/verify-sota-wasm-browser.cjs
```

The [raw timing record](benchmarks/prepared-wasm-2026-11-08.json) includes every
sample, artifact hashes, build flags and the first-binding comparison.
[WASM validation](wasm-validation-2026-11-08.json) records final test and package
scope. Chromium executes both actual binaries, the documented browser example,
real native-model imports, chronology/precision/resource failures and repeated
buffer reuse. Fresh Node 22/24 consumers exercise the installed package and strict
ESM/CJS declarations. The earlier seven Cargo archives remain historical native
candidates; this extension produces a separate Node/WASM candidate. Registry
publication is outside this validation.
