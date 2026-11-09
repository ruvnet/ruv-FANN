# Dense training, Candle and Node/WASM results

Validated 2026-11-09. This change adds actual reverse-mode training for small
sequential dense networks, reusable workspaces, parameter/input gradients,
SGD/AdamW and independent frozen inference. Rust supports f32/f64. The WASM and
napi-rs APIs use Float64Arrays and the shared Rust engine. The optional adapter
imports actual Candle CPU linear tensors and exports the versioned dense schema.

The [architecture decision](adr/ADR-008-dense-autograd-and-candle-interop.md) and
[source review](research-candle-2026-11-09.md) describe numerical/resource contracts.
The [structured validation record](candle-validation-2026-11-09.json) binds the
actual source/build/proof hashes. No registry release was performed.

## Correctness and deployment

| Check | Observed result |
| --- | --- |
| Rust 1.88 release core | 262 tests passed: library, original inference/allocation, direct packing, tiled/sparse batches, autograd and optimizer allocation checks |
| Core MSRV | Rust 1.81 minimal and serde checks passed |
| Actual Candle 0.11.1 CPU | 18 tests, 280 backward comparisons and 240 SGD/AdamW updates across f32/f64, five activation names and every two-layer bias combination |
| Native napi-rs | 21 tests each on Node 22.13 and 24.22.1; actual native library, hostile arrays, Workers, reentrancy, SDK/HNSW, gradients and three optimizer updates |
| New WASM APIs | 14 focused import/training tests each Node, both core/SIMD variants; full maintained security/runtime suite passed 97 each |
| Extracted npm consumers | Both Node versions passed the native archive and four core/SIMD × SGD/AdamW cases from the actual `ruv-swarm` archive |
| Chromium 156 / Playwright 1.62.1 | Both binaries matched actual Candle, learned a regression, passed 112 rejection cases, and used trained frozen snapshots in four real HNSW lifecycle checks |
| Pinned Rune | Frozen build, replay, fresh verification, real training/inference probe, release correctness and stale-source/plan/object rejection passed |
| Claude plugin 0.1.5 | Eight skills, structural validation and 39 helper/package tests passed; current Claude-host execution was not claimed |
| CI contracts | All 90 workflow tests, actionlint on nine workflows, and exact genuine native/browser receipt guards passed locally |

The local root SQLite addon targets Node 24. For the full existing CLI/MCP suite
on Node 22, a private test loader selected an independently built Node 22
`better-sqlite3` 11.10.0 addon with byte-identical JS sources. That suite passed
97/97. New raw/packed WASM and napi-rs proofs and benchmarks run directly without
that loader. The extracted WASM training consumer needs only Node builtins;
SQLite/CLI dependency installation is outside that archive check.

Candle gradient/optimizer comparisons use f32 `atol=5e-6, rtol=3e-5` and f64
`1e-12` each. Browser maximum absolute error was `2.22e-16`. ReLU's derivative
at zero differs between engines; FANN's zero rule has a separate regression.
The examples demonstrate implementation and learning behavior on small fixtures,
not task-level SOTA model accuracy.

Cargo audits of core/new Candle/new napi-rs locks found zero vulnerabilities;
the dependency-free napi npm package audit also passed. Pinned Ruflo deep scans
of new runtime areas reported zero findings, supplementing independent human
source review. A broad checkout scan also reported existing fixture/demo and
generated-file patterns outside this review; this is not a repository-wide
security-clean claim. All 12 pre-existing locks remain byte-identical. Three new
package locks were added. Private coordinator state was restored byte-for-byte.

## Matched-toolchain performance

These are local shared-host observations. The old core is Git
`d182eb42ae23a48560528c1d9284ee19b9bc510a`. Native baseline/candidate use Rust
1.88, release/thin-LTO/codegen-units1/debug0, deterministic weights/inputs and the
same benchmark computation. WASM baseline/candidate use Rust 1.88,
wasm-pack 0.15.1, wasm-bindgen 0.2.130 and identical release/profile flags.
The archived WASM source was checked against 1,292 Git blobs; its source digest
matches the original receipt. Baseline and candidate build directories are separate.

Each case has 20 raw timing blocks after warmup/calibration. Native cases
alternate baseline/candidate order; Node cases reverse backend order by case.
Blocks are sequential and unpaired, with no claim that V8 tier completion was
fenced. No builds or other benchmarks overlapped timing windows. Native intervals
use 1,000 deterministic unpaired median bootstrap resamples; they describe block
sampling, not variation across machines or days. A ratio above one is faster.

| Native rows | Geometric-mean baseline / candidate time |
| --- | ---: |
| 1 | 0.99× |
| 3 | 0.95× |
| 8 | 2.00× |
| 32 | 1.10× |
| 128 | 0.93× |
| All 60 cases | 1.14× |

Native cases cover f32/f64, Linear/Sigmoid, three shapes and five row counts.
There are 19 faster, 17 slower and 24 uncertain cases under the stated block
interval rule. Small and large-batch losses remain in the raw data. Native tiling
uses only 4–32 rows; other cases retain the original streaming loop.

| End-to-end WASM inference | Node 22.13 | Node 24.22.1 |
| --- | ---: | ---: |
| Core: all 30 f64 cases | 1.06× | 1.09× |
| SIMD: all 30 f64 cases | 1.13× | 1.15× |
| SIMD: 3-row cases | 1.47× | 1.61× |
| SIMD: 8-row cases | 1.24× | 1.25× |
| SIMD: 128-row cases | 0.99× | 1.00× |

WASM tiling stops at eight rows. Larger batches use streaming fallback; the
pilot's repeated large SIMD losses motivated this change. The pilot also used
an old Rust 1.99.1 binary, so it cannot isolate a kernel effect and is preserved
under separately labeled files. Only the matched 1.88 rebuild supplies the table.

Each Node receipt contains 150 inference cases across old/new core/SIMD and the
new native backend, plus 24 new training cases. API timings include intrinsic
array validation and staging/copies. Each engine's single/batch and WASM
before/after outputs agree exactly. Native/WASM exp implementations can differ:
maximum observed cross-backend error was `1.11e-16`, below the explicit
absolute-plus-relative `1e-12` comparison. No FMA or fast-math was enabled.

For the tiny `[3,4,2]` model with eight rows on Node 24, median MSE-backward plus
SGD time was 3.21 µs through napi-rs, 5.09 µs through core WASM and 4.65 µs
through SIMD WASM. Training is a new capability with no prior training-backend
baseline. Allocation tests establish zero hot Rust allocations; setup, JSON and
owned gradient-array copies allocate. Requested-byte estimates are not RSS.

The real native MetaHarness/Darwin run completed and confirmed **no improvement**.
That negative result is retained. It does not justify promoting a tuned setting.

Raw artifacts, including negative outcomes:

- [Performance summary](benchmarks/candle-summary-2026-11-09.json)
- [Native samples](benchmarks/candle-native-2026-11-09.jsonl) and [interval analysis](benchmarks/candle-native-analysis-2026-11-09.jsonl)
- [Node 22 samples](benchmarks/candle-backends-node22-2026-11-09.jsonl) and [Node 24 samples](benchmarks/candle-backends-node24-2026-11-09.jsonl)
- [Native prototype](benchmarks/candle-native-prototype-2026-11-09.jsonl), [Node 22 compiler-confounded pilot](benchmarks/candle-wasm-pilot-node22-2026-11-09.jsonl), and [Node 24 pilot](benchmarks/candle-wasm-pilot-node24-2026-11-09.jsonl)

## Reproduce the checks

Use separate external target directories for baseline, core tests/benchmarks,
napi-rs and the newer Candle toolchain. A shared core/Node target selected
incompatible cached serde features and failed safely during local verification;
a fresh dedicated Node build reproduced the identical final binary.
Prefetch pinned dependencies explicitly; verification runs
offline. Core training and the actual Candle fixture can be exercised with:

```sh
cargo +1.88.0 run --locked --example dense_autograd
cargo +1.88.0 test --locked --test autograd --test autograd_allocations --test prepared_dense --test prepared_dense_kernel
cargo +1.99.1 test --locked --manifest-path integrations/candle/Cargo.toml
cargo +1.99.1 run --quiet --locked --manifest-path integrations/candle/Cargo.toml --example export_fixture > /tmp/candle-fixture.json
```

Set `RUV_FANN_CANDLE_FIXTURE=/tmp/candle-fixture.json`. Build the native package
explicitly as its README describes, and rebuild both WASM variants through
`ruv-swarm/npm/scripts/build.js` after Rust changes. Run:

```sh
node --test bindings/napi/test/*.test.mjs
node --test ruv-swarm/npm/test/wasm-dense-import.test.js ruv-swarm/npm/test/wasm-autograd.test.js
node ruv-swarm/npm/test/browser-training.cjs --output /tmp/new-browser-proof.json
node ruv-swarm/npm/test/check-training-package.mjs --output /tmp/new-wasm-package-proof
node ruv-swarm/npm/test/benchmarks/candle-backends.mjs --baseline /absolute/matched-old-wasm-directory --output /tmp/new-backend-samples.json
```

The baseline directory must contain matching archived glue and core/SIMD binaries
built with the same compiler/profile. The native harness is
`examples/bench_packed.rs`; `--case ID --samples 20` permits alternating old/new
executables across cases. [Rune instructions](../README.md#verify-frozen-builds-with-rune) cover the
pinned real freeze/replay/fresh/negative gate. The new CI workflow repeats the
actual Candle, native, browser, archive and source-evidence contracts.
