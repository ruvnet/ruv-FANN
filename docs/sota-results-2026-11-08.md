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
