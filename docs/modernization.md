# Package modernization review

Review window: 2026-11-07 through 2026-11-08 UTC. Baseline: `f4ca36c2323d08eac822bac43b62040198721cbf`.
Branch: `mission/package-modernization`. No commits, pushes, or package publications were performed.

## Outcome and scope

All 26 Rust packages across five independent workspaces have updated release candidates. Both public npm distributions have been repaired, with real generated WASM in `ruv-swarm` and a deliberately smaller Node source-translation API in `cuda-wasm`. The native `opencv-sys` package now includes an implemented and tested C++ bridge. The forecasting facade now trains, forecasts, evaluates chronological windows, and restores actual checkpoints.

The implementation and its acceptance tests cover the exported behavior described below. Unsupported architectures and backends return useful errors. The [release inventory](release-inventory.json) records exact versions and separately dated registry observations; a candidate in this checkout is not a published release.

Specialists owned core inference/I/O, Rust swarm, npm security, forecasting, CUDA/OpenCV, and research/documentation. A coordinator integrated the work. Independent reviews of numerical and memory boundaries produced further production fixes and regressions. The local Ruflo coordinator was pinned to `@claude-flow/cli@3.25.6`; platform agents performed the work. Prior memory retrieval failed because its table was absent, so no retrieved memory was used as evidence.

## Package changes and detailed evidence

| Family | Candidate versions | Implemented behavior | Evidence |
| --- | --- | --- | --- |
| Core | `ruv-fann 0.2.2` | Allocation-free caller-buffer inference, bounded model/data I/O, checked SIMD, actual WebGPU compute and device-buffer storage | [Core review](review-core.md) |
| Rust swarm | Core/DAA `1.0.7`; transport/persistence `1.0.8`; other packages `1.0.6`; unified WASM `2.0.0` | Scheduling, bounded transport/MCP services, persistence, statistical forecasts, dense WASM inference, and explicit prototype limits | [Rust swarm review](review-rust-swarm.md) |
| Forecasting | All six `0.2.0` | Genuine recurrent training, baselines, dated/keyed forecasts, chronological evaluation, past-only scaling, and versioned checkpoints | [Forecasting review](review-forecasting.md) · [External test migration](../neuro-divergent/tests/README.md) |
| CUDA Rust | `cuda-rust-wasm 0.2.0` | CUDA-subset parsing and Rust/WGSL generation, initialized typed memory, exact allocation ownership, checked raw API contracts | [CUDA/OpenCV review](review-cuda-cv.md) |
| OpenCV | Core/WASM/SDK `4.9.0`; native `opencv-sys 0.2.0` | Portable matrix operations, real browser pixel operations, and actual OpenCV C++ calls through an exception-safe ABI | [Native guide](../opencv-rust/opencv-sys/README.md) · [Review](review-cuda-cv.md) |
| npm | `ruv-swarm 2.0.0`; `cuda-wasm 2.0.0` | Shell-free commands, checked SQL identifiers, bounded MCP framing, source-built WASM receipts, and a genuine external-compiler wrapper | [npm review](review-npm.md) |

Private repository tools, test applications, and generated nested npm manifests are not additional public products. Each package README describes its own supported behavior and dependencies. The root README supplies a working core example and two accessible animated SVG diagrams; family guides explain their more specific workflows.

## Correctness, performance, and security changes

### Neural networks and compute

The core forward pass borrows existing layer values instead of copying every layer. `run_into` reuses caller storage and performs zero allocations across repeated inference calls; `run` makes one final output allocation. Input/output length failures preserve state. Scalar activation derivatives, sparse connections, extreme finite inputs, SIMD tail handling, and gradient dimensions have independent numerical coverage. SIMD wrappers validate safe-call boundaries before unsafe loads or stores, including release builds, and check actual AVX2/FMA availability.

Model and training-data readers bound wire bytes, text lines, declared dimensions, and gzip expansion before the corresponding allocations. `Network::from_bytes` validates decoded topology. Call `Network::validate()` after direct public field edits or generic Serde deserialization; inference does not repeat structural validation on each call. These limits constrain individual operations rather than imposing a process-wide heap quota. Existing binary compatibility is retained, including the documented bincode maintenance risk.

The Rust backend enabled by `gpu` now dispatches every row and batch instead of overwriting initial tiles. Resource arithmetic, device limits, and workgroup counts are checked. The f32 activation path uses the requested shader and the scalar fallback shares canonical definitions. f64 compute retains CPU precision. Managed memory owns actual device buffers with unique handles, exact transfers, readback, lifetime synchronization, and allocation statistics. A device-less legacy pipeline cache returns an unsupported error instead of a phantom compiled object.

The `webgpu`/`wasm-gpu` features expose a separate async Rust browser bridge. The real built WASM class initializes a WebGPU device, multiplies row-major f32 matrices, applies canonical activations, and destroys/reinitializes its device. Buffers are capped at 64 MiB and actual device limits; a matrix call permits at most 67,108,864 multiply-adds. Dimensions, lengths, finite values, activation names, and workgroup counts are checked. Oversized JavaScript typed arrays are rejected before copying into WASM. Invalid inputs and unavailable devices return errors; there is no fabricated CPU fallback or canvas/agent API.

A browser harness loads that generated Rust WASM and exercises its public class. A separate direct-WGSL JavaScript harness checks all seven shader modules and actual Adam/AdamW shader updates. The two gates establish different parts of the execution path. Browser features alone do not enable the native Rust backend, and the browser bridge supports f32 only.

### Forecasting and training

The exported RNN now uses genuine backpropagation through time. Its hidden width and prediction horizon are independent. Seeded construction, full and truncated gradients, dropout masks, learning behavior, f32/f64 state, and versioned checkpoints have regression coverage. LSTM and GRU forecasting constructors explicitly reject unsupported gated architectures.

The primary facade retains successful training history, adapts the real RNN, and includes naive/seasonal baselines. Forecast rows have checked series IDs, actual future timestamps, a declared horizon, finite values, and exact model-column identities. Ensemble outputs are joined by series/time keys. `predict_on` accepts checked replacement history without changing retained training context. Calendar frequencies, integer timestamps, dates, and datetime precision have behavioral tests.

Chronological evaluation fits actual past prefixes, records per-series cutoffs, and scores held-out observations. Even `refit = false` trains its first prefix afresh to prevent leakage from a previously future-fitted model. Per-series scalers fit only those prefixes. Persistence atomically stores supported baseline/RNN parameters, history, frequency, schema, and scaler state; bounded loads reject incompatible or malformed state. Special-file preflight and a checked nonblocking Unix open reject FIFOs without waiting for a writer, including replacement after preflight. Custom model persistence fails before creating or replacing an artifact.

Zoned calendar arithmetic advances local dates before resolving their UTC instants; daily/month-end forecasts preserve timezone metadata through checkpoints and across DST. Ambiguous or nonexistent local timestamps return typed errors. External baseline tests check actual New York midnight predictions, including January/February month ends followed by March 31 at 04:00 UTC, and a daily 23-hour DST transition.

Runtime tests exposed real defects in empty/nonfinite loss handling, ragged optimizer gradients, partial parameter updates, large-offset normalization, and higher-order/seasonal differencing. Production fixes now validate inputs and stage optimizer updates, use centered variance, and reconstruct each difference level from its own seeds. External finite-difference, polynomial, forecast, and persistence assertions were retained through those fixes.

The external suite uses real models and independently defined deterministic baselines. All registered test, example, and benchmark targets compile. Memory tests use monotonic per-thread allocation deltas on an owning thread; they do not reset a racing global allocator counter. Repeated construction, inference, refitting, and ensemble destruction have bounded allocation and exact retained-byte checks. The [migration map](../neuro-divergent/tests/README.md) explains how each historical category maps to current behavior.

### Agents, JavaScript, CUDA, and images

Transport changes serialize coupled ring-buffer operations, enforce recipient/expanded-message budgets, and reserve quotas atomically. JavaScript process calls pass literal argument arrays; permission bypass requires explicit opt-in. SQL update keys use an allowlist separate from bound values. Connection-specific statements and error paths return borrowed pool connections correctly. WASM integrity checks hash and consume the same bytes.

Swarm npm 2.0 deliberately narrows public exports and MCP tools to implemented behavior. Removed prototype exports and fabricated training/DAA/result metrics are breaking API changes, so this candidate uses a major version. Its supported MCP surface performs persisted assignments, actual inference/forecasts, and measured dense-network work; task assignment alone does not execute an external agent.

CUDA memory APIs initialize values before safe access and retain exact allocation ownership/layout. Caller-controlled raw-pointer kernel operations require explicit unsafe contracts; unavailable native dispatch returns an error. Benchmarks now measure implemented CPU memory, dispatch, reference attention, parser/code generation, vector operations, and profiling workloads. Fabricated native-CUDA timings and machine-dependent empty-workload thresholds were removed.

The breaking `cuda-wasm 2.0` Node API invokes a separately installed `cuda2rust >= 0.2.0` executable through `execFile`, with bounded source/output, timeouts, private temporary files, version checks, cleanup, and genuine CJS/ESM/types/CLI exports. It emits Rust or WGSL source. Its published files exclude the old incomplete browser/WASM build scripts. Installing the wrapper does not install a CUDA runtime or compiler binary; local Rust builds work before publication.

Portable image matrices seal supported primitive pixel types and check dimensions, widths, allocation sizes, ROI copies, conversion, and resize arithmetic. The native bridge builds real OpenCV C++ source, exposes an unsafe ownership API, catches C++ exceptions, and defines errors through status values and thread-local details. Native ROI shares OpenCV storage; portable ROI copies pixels. Browser tests execute generated Rust WASM for pixel/canvas operations.

## Reproduce verification

Use current stable Rust, Python 3.11+, and Node 22.13+ for the complete repository. The standalone CUDA Node wrapper accepts Node 22.12+. The core declares Rust 1.81; all swarm and OpenCV packages declare Rust 1.88. Native OpenCV also needs a C++17 compiler, pkg-config, libclang, and OpenCV >= 4.4 development files; see the [native setup guide](../opencv-rust/opencv-sys/README.md). The tested native library is OpenCV 4.10.0 on Debian 13.

```sh
python3 scripts/verify-packages.py --inventory
python3 scripts/verify-packages.py --check
python3 scripts/verify-packages.py --test
python3 scripts/verify-packages.py --audit
python3 scripts/verify-packages.py --docs
python3 -m unittest discover -s scripts/tests -v
```

The driver uses `--workspace --all-targets`, including registered examples and benchmarks. Swarm/forecasting use all features; CUDA uses its portable `serde` configuration. Root all-feature checking, targeted security integrations, and binary targets are included as defined by the driver. It continues after failures and returns nonzero if any command fails. Hardware/browser tests and npm consumers are separate explicit gates. For machines without native OpenCV, use the family guide's `--exclude opencv-sys` commands and report that narrower scope.

The coordinated forecasting gate is:

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
cargo test --manifest-path neuro-divergent/Cargo.toml --workspace --all-targets --all-features --locked
```

Browser compute gates additionally need wasm-pack, the root npm dependencies, and Playwright's Chromium:

```sh
node scripts/verify-webgpu.cjs
node scripts/verify-rust-browser-gpu.cjs --build
```

### Completed acceptance evidence

| Gate | Verified result and scope |
| --- | --- |
| Core native all features/targets | 202 library + 2 allocation + 7 actual compute/memory integration tests; zero ignored; 39 Criterion smoke cases; all examples compile |
| Core additional gates | 7 all-feature doctests, Rust 1.81 default-library check, and minimal native `gpu` check pass |
| Core browser shaders | A direct JavaScript harness compiles/dispatches all 7 WGSL modules; 13,158 matrix values, 21,567 activation values across all 21 functions, and 670 actual Adam/AdamW updates agree with references |
| Core Rust browser bridge | Built WASM returns 13,902 matrix results and 21,567 activation results across all 21 variants; 28 invalid requests reject; shared-buffer growth, WASM-memory input detachment, concurrent calls, device loss, destroy/reinitialize, and metrics are checked; both browser feature target checks pass |
| Rust swarm native | All 14 packages pass 482 tests with zero ignored and 48 Criterion smoke cases under Rust 1.88/all targets/all features; 5 doctests and 14 example builds pass; 11 offline executable examples run successfully |
| Rust swarm WASM | Actual transport runs 2 Node tests; real IndexedDB runs 7 Chromium test groups; selected/all-feature browser targets and ordinary/SIMD builds pass |
| Swarm npm checks | Node 22.13 and 24 each pass 55 real boundary/API/runtime tests, 3 tooling tests, 8 labeled mock examples, and the separate 4-suite/55-assertion legacy runner |
| Swarm npm archive | Installed archive passes 4 API tests, actual SQLite/ordinary-SIMD WASM inference/forecasts, 17-tool MCP protocol checks, ESM/CJS loading, blocked prototype subpaths, strict TypeScript consumers, package validation, and a zero-finding full-graph audit on both Node versions |
| Forecasting all features/targets | 354 tests and 16 all-feature doctests pass; zero ignored; 15 Criterion smoke cases; all 3 registered examples compile; example mains also executed successfully |
| CUDA all targets | 744 tests and 40 Criterion smoke cases pass; 2 optional regression benchmark cases pass; minimal/all-target and optional feature checks pass |
| CUDA npm archive | Final 10-file archive installs on Node 22.12.0 and 24.22.1; each passes 18 tests, genuine compiler integration, CJS/ESM/CLI generation, strict TypeScript consumers, package validation, and a zero-finding full-graph audit |
| OpenCV native/browser | 54 Rust tests and 8 actual Chromium WASM checks pass; native coverage includes 12 integration tests, 8 ABI-layout tests, and linking all 95 baseline symbols |
| Documentation | Core README snippet compiles/runs with output `0.1`; SVG XML, accessible labels, reduced motion, rendering, and local README links validated |

These counts describe specific completed commands, not a combined count of independent capabilities. Criterion smoke execution establishes that a benchmark runs; it does not establish a speedup. Core native compute used Mesa 25.0.7 llvmpipe software Vulkan; browser compute used Chromium/SwiftShader. Physical CUDA/ROCm acceleration and physical GPU throughput were not measured. Two swarm provider examples compile but were not executed against external accounts. Tested remote workflow definitions do not establish execution on other operating systems.

All local implementation, build, test, audit, and package blockers are closed. The coordinated Cargo run verified all 26 archives with 34 successful commands and unchanged live locks in all five workspaces. Both npm installed-archive consumer matrices pass. The historical installation fixture uses the final archive and passes a clean install, native SQLite, actual inference/forecast checks, and a zero-finding audit. The [28-artifact manifest](release-artifacts.json) and final [validation receipt](validation-modernization.json) record candidate identities, command scope, hashes, and timestamps.

## Dependency assessment

All five Cargo lockfiles currently report zero known vulnerabilities and zero unsoundness findings under `cargo-audit 0.22.2` with the review's RustSec database. Maintenance warnings remain, including bincode and selected transitive dependencies; they require follow-up even when the vulnerability count is zero. No advisory was suppressed to obtain those results. Cargo coverage does not include system OpenCV/codec/driver security updates, which follow the operating system's update process.

All four tracked npm lockfile graphs (private root tooling, standalone swarm, standalone CUDA wrapper, and the installation fixture) report zero audit findings including development dependencies. Swarm tooling replaces the affected deep-merge implementation with a maintained compatible source and has a prototype-poisoning regression plus actual Jest/Rollup resolution checks. CUDA npm 2.0 has no npm dependencies. The historical installation fixture was regenerated from the actual final `ruv-swarm 2.0.0` archive; its previous four findings are closed and its real SQLite/inference/forecast checks pass. No tracked lockfile was excluded from the npm audit result.

A clean scan is bounded by the recorded advisory feed, dependency graph, and time. The supplemental Ruflo heuristic scan was not treated as a Rust proof or a complete secret audit. Its synthetic authentication fixtures and process/SQL syntax flags were reviewed against actual execution boundaries. No credential values appear in the reports.

## Archive verification and publication

`scripts/verify-release.py` stages an isolated source snapshot, builds ordinary Cargo-verified archives, inspects normalized package identities and license payloads, and writes hashes and dependency publication order. Dependent families use the already verified root archive in their staging workspace. Registry dependency identities are checked against the original locks; live checkout locks are preserved. The tool does not use `--no-verify` and does not publish.

```sh
# Choose a new empty directory for each immutable source/archive receipt.
python3 scripts/verify-release.py --output /path/to/new-release-directory
```

The final combined Cargo run passed against frozen source: 26 archives total 2,296,265 bytes, all 34 commands succeeded, and all five live Cargo locks remained unchanged. The immutable [Cargo receipt](/workspace/scratch/release-candidates-final/receipt.json) has SHA-256 `a9baef95b2fe388e72a5edeebade7635dc54f6481a12b8f0bf6e69d85009a611`; its [publication plan](/workspace/scratch/release-candidates-final/publication-order.json) satisfies all 30 internal dependency edges. Both npm archives pass fresh prepack checks and installed consumer matrices: swarm has 97 files/447,622 bytes; CUDA has 10 files/12,954 bytes. The [artifact manifest](release-artifacts.json) records all 28 candidate identities. A byte hash identifies an artifact; it does not certify a reproducible or signed build.

Publication needs a deliberate dependency order: verified core FANN first, dependent swarm/forecasting crates after it, native/portable OpenCV leaves before dependent SDK/bindings, and Rust `cuda2rust` support before users rely on the Node compiler prerequisite. The generated publication plan records exact package dependencies. Keep the matching license texts and source/artifact receipts with distribution. Registry publication was not attempted because authentication was unavailable. Physical GPU performance and execution on remote operating systems remain unverified limits; they are not represented as completed local checks.

Workflows use locked resolution, pinned actions, and read-only default permissions. They propagate failed gates and no longer auto-fix source, substitute another crate's WASM after a failed build, or publish competing automated versions. Npm publishing remains an explicit version-matched workflow after validation.

## Research and remaining capability boundaries

[Primary-source research](research-modernization.md) and [ADR-006](adr/ADR-006-inference-and-boundary-safety.md) explain adopted changes and deferred work. This pass adopts checked boundaries, predictable allocation, actual recurrent gradients, past-to-future evaluation, and real shader/browser execution. It provides reusable benchmark code and numerical references.

State-of-the-art forecasting architectures, pretrained foundation models, quantization, packed inference, and a newer WGPU major require separate implementations and reproducible evaluations. The current forecasting contract remains univariate; exogenous-model mapping, intervals/sampling, GPU/AMP training, and executable native plugins are unsupported. Some agent behavior is metadata, heuristic assignment, or a documented demo. Shared-memory transport is in-process. The Rust browser bridge covers bounded f32 compute rather than canvas rendering, persistent shared GPU storage, or browser agent orchestration. Portable WASM image filters and the SDK Python surface cover only their documented subset; optional OpenCV CUDA requires a CUDA-enabled upstream library and real hardware.

No universal latency multiplier, leaderboard accuracy, token saving, SWE-bench solve rate, full upstream API parity, or physical GPU performance is claimed. Future results need disclosed datasets, chronological splits, software/hardware versions, warm-up, allocations/transfers, numerical tolerances, and reproducible commands.
