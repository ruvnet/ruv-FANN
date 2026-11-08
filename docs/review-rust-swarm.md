# Rust swarm package review — 2026-11-07

Scope: all 14 packages in `ruv-swarm/`, reviewed from baseline commit `f4ca36c2323d08eac822bac43b62040198721cbf` on `mission/package-modernization`. Work is staged in the shared checkout. No commit, push, crate publication or release publication was performed by this review.

Rust 1.88 final acceptance passed **482 tests across all 14 packages, 48 Criterion smoke cases and 5 doctests**, with every native target and Cargo feature enabled. All 14 examples build; 11 offline executable smoke checks pass. Actual WASM executes two transport tests under Node and seven grouped IndexedDB checks in Chromium with matching wasm-bindgen 0.2.130. Base/unified all-feature browser builds pass with ordinary and SIMD flags. Minimal/core `no_std` and the DAA, persistence and transport browser configurations compile. These results establish the tested contracts and build configurations; unsupported architecture backends remain explicit errors.

## Package inventory and candidate graph

Official registry observations come from `https://crates.io/api/v1/crates/{name}`, observed 2026-11-07T15:05:36Z. A registry observation is not proof of publication ownership or authority to publish. HTTP 404 means the query did not find that crate at the time of observation. Candidates remain unpublished.

| Package | Checkout baseline | Registry latest stable | Candidate | Rust 1.88 all-target/all-feature coverage |
| --- | --- | --- | --- | --- |
| claude-parser | 1.0.5 | 1.0.5 | 1.0.6 | 10 library tests |
| ruv-swarm-agents | 1.0.5 | 1.0.5 | 1.0.6 | 2 library tests |
| ruv-swarm-benchmarking | 1.0.5 | HTTP 404 | 1.0.6 | 20 library + 4 integration tests |
| ruv-swarm-cli | 1.0.5 | 0.2.0 | 1.0.6 | 2 binary tests; binary check |
| ruv-swarm-core | 1.0.6 | 1.0.6 | 1.0.7 | 173 library tests; 8 Criterion cases; runnable example; minimal/no_std/WASM checks |
| ruv-swarm-daa | 1.0.6 | 1.0.6 | 1.0.7 | 16 library + 2 coordinator binary tests; WASM feature checks |
| ruv-swarm-mcp | 1.0.5 | 0.2.0 | 1.0.6 | 37 library tests; 2 bind-configuration binary tests; both binaries check |
| ruv-swarm-ml | 1.0.5 | 1.0.5 | 1.0.6 | 22 library + 37 integration tests; optional browser path compiles |
| ruv-swarm-ml-training | 1.0.5 | 1.0.5 | 1.0.6 | 9 library + 7 integration tests; 10 Criterion cases |
| ruv-swarm-persistence | 1.0.7 | 1.0.5 | 1.0.8 | 75 native tests; real Chromium IndexedDB 7 groups |
| ruv-swarm-transport | 1.0.7 | 1.0.5 | 1.0.8 | 22 library + 3 integration tests; 17 Criterion cases; 2 actual WASM/Node tests |
| ruv-swarm-wasm | 1.0.5 | 1.0.5 | 1.0.6 | 14 library tests; ordinary/SIMD WASM checks |
| ruv-swarm-wasm-unified | 1.0.5 | HTTP 404 | 2.0.0 | 1 library test; ordinary/SIMD WASM checks |
| swe-bench-adapter | 1.0.5 | 0.2.0 | 1.0.6 | 18 library + 6 integration tests; 13 Criterion cases |

The family workspace version is 1.0.8. Transport and persistence inherit it; other packages have explicit candidate versions. All local path dependencies now carry matching version requirements, including root `ruv-fann` 0.2.2. Three stale repository URLs were corrected to `ruvnet/ruv-FANN`.

Unified 2.0.0 intentionally shrinks the prototype API. Other candidates also change invalid/unsupported-input behavior: errors replace silent model aliases, no-op saves, ignored browser configuration and fabricated evaluations. Review those behavior changes before deciding the final release numbering and migration policy. A candidate version is not a compatibility guarantee.

## Reproducible verification

Environment: Linux `x86_64-unknown-linux-gnu`; rustc 1.88.0 (`6b00bc388`, 2025-06-23); Cargo 1.88.0; cargo-audit 0.22.2. All 14 manifests now declare Rust 1.88, matching the tested minimum and the current MCP SDK requirement. Rust 1.99.1 was also used during earlier baseline/fix work, but the final acceptance below uses 1.88.

Run from `ruv-swarm/` with `CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0` when memory/disk are constrained:

```sh
cargo +1.88.0 metadata --locked --offline --format-version 1 --no-deps
cargo +1.88.0 test --locked --workspace --all-targets --all-features --no-fail-fast
cargo +1.88.0 test --locked --workspace --all-features --doc
cargo +1.88.0 build --locked --workspace --all-features --examples
cargo +1.88.0 run --locked -p ruv-swarm-core --example basic_swarm

cargo +1.88.0 check --locked -p ruv-swarm-core --no-default-features --lib
cargo +1.88.0 check --locked -p ruv-swarm-core --no-default-features --features no_std --lib
cargo +1.88.0 check --locked -p ruv-swarm-core --no-default-features \
  --features wasm --lib --target wasm32-unknown-unknown
for package in ruv-swarm-daa ruv-swarm-persistence ruv-swarm-transport; do
  cargo +1.88.0 check --locked -p "$package" --no-default-features \
    --features wasm --lib --target wasm32-unknown-unknown
done
cargo +1.88.0 check --locked -p ruv-swarm-daa -p ruv-swarm-persistence \
  -p ruv-swarm-transport --all-features --lib --target wasm32-unknown-unknown
cargo +1.88.0 check --locked -p ruv-swarm-wasm -p ruv-swarm-wasm-unified \
  --all-features --lib --target wasm32-unknown-unknown
RUSTFLAGS="-C target-feature=+simd128" cargo +1.88.0 check --locked \
  -p ruv-swarm-wasm -p ruv-swarm-wasm-unified --all-features --lib \
  --target wasm32-unknown-unknown
cargo audit --json --no-fetch
```

All these configurations passed. The selected `wasm` checks were also run separately for DAA, persistence and transport, so the result does not rely solely on their combined dependency feature union. CLI is binary-only; the all-target command includes it, unlike `--workspace --lib` alone. Native all-feature success does not imply native CUDA device availability: this family no longer implicitly enables the root GPU backend.

The core example prints `report-1 -> worker-a`, `report-2 -> worker-b`, `report-3 -> worker-a`. Its source now lives inside the crate. Executed offline examples: `basic_swarm`, `parse_claude_output`, `real_time_monitor`, `ruv_swarm_integration`, `comprehensive_swe_bench_evaluation`, `basic_training`, `hyperparameter_search`, `model_comparison`, `swarm_coordinator_training`, `basic_evaluation`, and `basic_usage`. The last uses an isolated database and ephemeral listener, requiring its own successful post-bind log before checking `/health`. Unsupported built-in training backends produce clearly identified errors; no model results are fabricated. The hyperparameter example dataset was reduced to stay within the real loader budget. `benchmark_integration` and `basic_benchmark` compile but require an external Claude executable/provider; no live provider run is claimed. The fourteenth example is the browser persistence cdylib fixture.

### Actual WASM runtime checks

Use a `wasm-bindgen-test-runner` matching the lock's **0.2.130** version:

```sh
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=/path/to/wasm-bindgen-test-runner \
  cargo +1.88.0 test --locked -p ruv-swarm-transport --no-default-features \
  --features wasm --lib --target wasm32-unknown-unknown
```

Both tests pass in Node and execute Rust in-process transport send/receive/broadcast/close, duplicate-ID rejection and receiver size limits. This is not browser WebSocket or cross-process shared-memory support.

With Playwright and a Chromium browser installed, run the IndexedDB fixture from the repository root:

```sh
cargo +1.88.0 build --manifest-path ruv-swarm/Cargo.toml --locked \
  -p ruv-swarm-persistence --example indexeddb_browser_smoke \
  --no-default-features --features wasm --target wasm32-unknown-unknown
wasm-bindgen ruv-swarm/target/wasm32-unknown-unknown/debug/examples/indexeddb_browser_smoke.wasm \
  --target web --out-dir /tmp/ruv-persistence-browser --out-name persistence_smoke
node ruv-swarm/crates/ruv-swarm-persistence/tests/browser-smoke.cjs /tmp/ruv-persistence-browser
```

Use wasm-bindgen 0.2.130; optionally set `CHROMIUM_BIN` for an installed browser. Seven grouped checks pass in actual Chromium, covering full-width `u64`/Unicode values, indexes, racing claims across two connections, event counter rollback, message updates, aggregation, multi-store rollback, commit barriers and close/reopen persistence. These are persistent IndexedDB writes, not a JavaScript mock.

Evidence retained in `/workspace/scratch/`: `swarm-final130-native.log`, `swarm-final130-status.json`, `swarm-aligned-{docs,examples,browser-full}.log`, `swarm-final130-browser-storage-transport-daa-full.log`, `swarm-transport-wasm-runtime130-final.log`, `persistence-browser-aligned-{build,tests}.log`, `swarm-example-smoke-status.json`, `swarm-audit-aligned.json` and `swarm-package-lists/final130-summary.json`. These are session evidence, not committed release artifacts.

## Correctness and performance changes

### Core scheduling and application handlers

`swarm.rs` and `async_swarm.rs` reject duplicate IDs before changing live agents or loads. Batch extraction preserves FIFO; priority ordering is stable. Round robin now advances across repeated batches instead of returning the first hash-map entry every time. Random selection actually samples; ID ordering makes ties reproducible. Async selection/recording is serialized so simultaneous distributors do not race the bookkeeping.

`ruv-swarm-agents` was a one-line placeholder. It now provides `FunctionAgent<I,O,E,F>` over an application-owned async handler. Tests verify mutable handler state, typed results, unchanged errors and ID/capability validation. It creates no provider session and executes no hidden commands.

Scheduling tests cover seven successive batches with three workers in both synchronous and asynchronous APIs, stable priorities, FIFO and duplicate-ID rejection. This demonstrates assignment behavior, not a complete worker execution runtime or distributed throughput improvement.

### Transport and bounded parsing

The previous ring protected its byte vector while publishing head/tail/size separately. Concurrent producers could overwrite unread data, and concurrent readers could consume the same frame or underflow counters. A single lock now protects the entire frame transaction. Capacity arithmetic is checked; full writes leave earlier frames intact. Wrapped data uses at most two slice copies per slice instead of a modulo loop for every byte. The 4-producer/2-consumer regression verifies 2,000 uniquely delivered frames across many wraps.

WebSockets enforce frame/message budgets, a handshake timeout and decoded-size limits. Gzip is identified by its header, avoiding a compressed-size threshold bug that rejected highly compressible messages. `decompress_with_limit` reads at most limit+1 expanded bytes before rejecting. Send paths avoid an unnecessary full-payload clone. Transport paths snapshot or clone DashMap handles before awaiting sends. Peer-stream EOF now exits its loop.

In-process channels always have positive bounded capacity. Duplicate endpoints cannot hijack another registration, receiver-side budgets are enforced, and broadcast reaches the normal receive channel. `TransportRegistry` clones an `Arc` builder and drops its DashMap guard before awaiting external construction; a gated-builder regression proves same-key replacement can complete during that await. Native networking/shared-memory dependencies are target-specific, so portable WASM does not try to compile Tokio native networking.

`claude-parser` no longer grows an unbounded line buffer. `ParserLimits` default to 1 MiB per physical line, 64 MiB cumulative input and 100,000 retained events. Input includes malformed and blank lines, and quotas apply across calls. Oversized unterminated input and retention boundaries have regressions. Raw malformed lines are no longer logged.

### Persistence and benchmark integrity

In-memory metric aggregation now retains sum/count per `(agent,unit)` group using borrowed keys during its read lock. It avoids per-sample string clones and per-group vectors of all values. The existing 360,000-record one-second guard failed before this change at 1.112509427s on this shared host and passes unchanged afterward. That observation is a regression result under this host's workload, not a portable percentage speedup. A new test verifies grouping, means/counts, inclusive time bounds and an empty result.

The benchmark storage backend uses exact-matched `sqlx-core = 0.8.6` and `sqlx-sqlite = 0.8.6` with explicit private row decoding. Its storage tests pass. A single scenario pair no longer receives a fabricated p-value 0.01, confidence interval or effect size. It uses documented not-calculated sentinels (p=1, interval 0, effect 0); empty trial summaries are finite and do not panic.

The IndexedDB backend now uses the actual Rexie 0.6.2 API with native-only SQLite dependencies. Its record envelope preserves full-width integers, IDs and query indexes. Writes await commit; task claims, message updates and event sequence counters run in the same transaction as their associated records. `write_batch` provides scoped atomic multi-store writes with abort-on-error. Generic `begin_transaction` fails explicitly because later independent storage calls cannot share its lifetime; callers use `write_batch`. Browser `VACUUM` is unsupported, `checkpoint` is a commit barrier, and `storage_size` reports serialized record bytes. Browser futures use local bounds while native `Storage` retains `Send + Sync`.

The SQL builder property test had a false positive: it searched for a bound value anywhere in SQL, so generated `A` or `0` also matched static `AND` or `OFFSET 0`. It now verifies unchanged SQL across changed values, two placeholders and exact parameter order. The minimized regression seed is included in source packages. No test or assertion was removed.

### Real forecasting and honest capability catalogs

The Rust neural bridge now uses the trained network on the last observed input window and feeds output blocks back into that window. It rejects unfitted prediction, invalid dimensions, nonfinite training values, mismatched samples and unsupported covariates. Linear output activations support forecasts outside `[0,1]`. Tests compare the first forecast with direct evaluation of the trained network and verify repeatable prediction.

The factory/catalog expose only MLP, a linear baseline under legacy `DLinear`, and a wider univariate MLP under `MLPMultivariate`. Unsupported LSTM/TFT/etc. names return errors instead of silently instantiating unrelated MLPs. Agent selection and integration examples now use implemented models. Parameter loading returns an unsupported-operation error rather than a successful no-op.

Model-memory planning now checks finite resource limits and reserves replacement/switch deltas rather than double-counting an agent's existing allocation. Oversized replacements fail before mutation; removal frees the reservation. Timestamps use actual Unix milliseconds instead of a mathematical constant. The native bridge explicitly enables CPU parallel/SIMD features, while portable WASM avoids native Rayon threading and implicit GPU dependencies.

The separate training package still has no LSTM/TCN/N-BEATS backend. Its train/predict/save/load methods fail explicitly. The event loader checks positive window/buffer bounds, event count and generated feature elements. Dataset/model evaluation checks shapes and finite values. Numeric random search is implemented with a real PRNG; unsupported methods, categories and invalid ranges return errors. Trial training and validation partitions are distinct, and a regression verifies held-out values do not leak into a trained closed-form mean baseline. Constant-target R² is finite. Historical integration tests now assert supported baselines or explicit backend errors instead of requiring fictitious architectures. It does not claim a new SOTA forecasting architecture.

### Browser inference and actual local assignment

The base WASM network now checks 2–64 nonzero layers, checked parameter arithmetic and a 1,000,000-parameter budget before allocation. Weight/input shapes and finite values are required; overflowing inference fails. Dense inference preserves f64 arithmetic rather than casting every layer to f32. It reuses two layer buffers and advances weight/bias offsets rather than allocating, transposing and re-indexing complete matrices on every layer.

Vector sigmoid/tanh use stable exponentials consistent with scalar tails; inaccurate rational approximations had produced different functions depending on vector length. Regressions cover every remainder length, extremes, NaNs and signed zero. Matrix dimension products and output budgets are checked. Browser microbenchmark outputs use valid JSON and optimization barriers, with a null speedup for a zero timer interval. Those helpers do not establish a universal SIMD gain.

`MemoryPool` rejects overflowing or above 64 MiB dimensions, allocates lazily, accounts outstanding blocks and clears returned bytes. Unknown forecast names no longer silently become last-value models: only linear, mean and naive one-step baselines are accepted, with empty/nonfinite/overflow checks.

Fresh JavaScript artifact tests exposed ignored browser spawn/task JSON. The local orchestrator now parses strict bounded JSON, honors IDs/names/types/capabilities, rejects duplicates/unsupported fields, and assigns only compatible idle workers. Task description/priority/maxAgents are real inputs; statuses distinguish assigned/unassigned, and `update_agent_status` lets the application release workers after its handler completes. It fabricates no neural-network IDs, outputs or execution success. The inventory is capped at 1,000 agents and config at 64 KiB. Three regressions cover configuration, actual assignment and validation before state mutation.

Unified 2.0.0 is a thin re-export of the base bindings. Unused native-thread, ML, SQLite and parallel dependencies were removed. The old core/neural/forecasting/persistence/utils prototypes and `WasmConfig` are no longer compiled or exported; their archival source remains for reference. This change resolved the actual wasm32 Tokio compile failure without claiming the unused integrations were implemented. Standalone core WASM imports were also corrected, and that exact target now passes.

### MCP, CLI and SWE boundaries

MCP WebSocket defaults now account 100 agents, 1,000 tasks, 64 MiB session memory, 10,000 entries,1 MiB messages/values and 24-hour sessions. Quota reservation and accounting share a lock; spawn errors roll back reservations; memory math uses checked addition. Connection permits, frame/message budgets and request timeouts are applied, and JSON-RPC version/live-session checks run before handlers. Tool argument bodies are not logged. Concurrent-reservation and integer-boundary regressions pass.

The final concurrent npm integration gate exposed a hardcoded WebSocket example port: one harness could connect to another harness's server before its own process failed to bind. `RUV_SWARM_MCP_BIND_ADDR` now overrides the address while preserving `127.0.0.1:3000` by default. Parsing rejects invalid environment values before database initialization. A post-bind log includes the actual listener address, allowing the harness to require its own child's successful bind before probing health. Two small binary tests cover default/ephemeral/IPv6 addresses and malformed values.

CLI configuration validates bounds, positive timeouts and HTTP(S) endpoints. API keys are omitted when saving and redacted in Debug output; this does not clean credentials from older saved files. Two tests verify those properties.

SWE evaluation previously ignored its agent and fabricated execution output/patches. The additive mutable typed-agent API now invokes the actual application handler with timeout/error/exit-code/output checks. The legacy shared-agent API and its legacy batch path fail closed. The high-level typed path records one real handler duration instead of simulated repetitions/resource profiles. Tests verify invocation and propagated failure.

The loader no longer creates mock dataset JSON when an instance is absent. IDs reject path separators and invalid characters; symbolic links and documents above 16 MiB are rejected; document/request IDs must match. Cache admission is capped at 1,000 documents and 64 MiB source JSON. Integration tests use explicit local fixtures rather than fabricated downloads. A working-directory option supports an actual local Git checkout. Patch files use unique temporary names instead of overwriting a fixed `.swe_bench_patch.diff`; a sentinel-preservation regression runs actual `git apply`/`git diff --check`. Timed-out child commands are killed. The 13 SWE benchmark cases exercise concrete local parsing/patch/handler fixtures rather than internet access or fabricated end-to-end execution.

## Dependency findings

Audit feed: RustSec commit `309392b18f6b17022f82c701f300433126901068`, updated `2026-11-06T16:34:32+01:00`. Final lockfile SHA-256: `bb0a255f31a0430806bfd8fc08353ecc7e9d0e7a252c9e76afe109a002555568`. The final `--no-fetch` audit uses 1,422 advisory records; its JSON omits Git metadata, so the commit identifies the review feed provisioned by parent integration.

The baseline lock resolved 697 packages and reported 6 vulnerability records plus 8 unsoundness warnings for zstd-safe. The final lock resolves 570 packages and reports **0 vulnerability records, 0 unsoundness warnings and 6 unmaintained-package warnings** with the same feed. No advisory suppressions or risk-acceptance exceptions were added.

| Baseline finding | Change and reachability evidence |
| --- | --- |
| rsa 0.9.8, RUSTSEC-2023-0071, no fixed version | Pulled into the lock by unused SQLx MySQL/macros dependencies. The baseline target/feature tree did not compile RSA. Narrow SQLite components remove the package from the lock entirely. |
| zip 0.6.6, RUSTSEC-2026-0367/0368/0369/0429 | Came from unused optional tensor-engine dependencies in DAA; there were no corresponding backend calls in DAA source. Removed unused tch/Candle dependencies rather than retain an unsupported feature claim. |
| zstd-safe 5.0.2, RUSTSEC-2026-0403 plus 8 unsoundness warnings | Removed with the unused tensor/archive chain. No suppression or forced incompatible codec upgrade. |

Remaining maintenance notices are real follow-up work:

| Dependency | Advisory | Follow-up boundary |
| --- | --- | --- |
| backoff 0.4.0 | RUSTSEC-2025-0012 | Transport retry API migration |
| bincode 1.3.3 | RUSTSEC-2025-0141 | Wire/model codec compatibility and decoded-allocation review |
| instant 0.1.13 | RUSTSEC-2024-0384 | Transitive timer/retry dependencies |
| paste 1.0.15 | RUSTSEC-2024-0436 | Transitive math/dependency graph |
| unsafe-libyaml 0.2.11 | RUSTSEC-2024-0449 | YAML configuration/import compatibility |
| wee_alloc 0.4.5 | RUSTSEC-2022-0054 | Optional allocator paths; default browser build uses the standard allocator |

An advisory count is database-scoped evidence, not a proof that the code is free of vulnerabilities. Parent integration also requires MCP SDK rmcp>=1.4 (the final lock resolves 1.8.0); all 14 manifests declare and were validated with Rust 1.88. Root, Swarm and forecasting locks align wasm-bindgen 0.2.130, js-sys/web-sys 0.3.107 and wasm-bindgen-futures 0.4.80, avoiding an incompatible archive dependency graph. Matching native binding tools were used for browser runtime validation.

## Source package contents and documentation

Every package has an anchored source allowlist. Required compile-time SQLite schema/dashboard assets remain included; database files, Claude session state and other runtime data are absent from all 14 final `cargo package --list --allow-dirty --locked --offline` inspections. This listing check does not verify a registry publish or resolve unpublished candidates from crates.io.

The family README and all 14 package READMEs now explain actual APIs, input limits, verified checks and prototype boundaries. Each links the shared animated package-map SVG; the family uses the relative repository asset and packaged READMEs use a canonical repository URL. Old claims of 84.8% SWE-bench accuracy, 32.3% token reduction, universal 2.8–4.4× acceleration, 27 production models and zero dependencies were removed from these public entry points. Nested historical research/test documents were not all rewritten and remain outside current evidence.

## Remaining release limitations

- All native workspace targets/features, all existing integration targets, 48 Criterion cases and five doctests pass on declared Rust 1.88. All examples build; 11 run offline. Two external-provider examples were not live-tested. CI should preserve this scope instead of reducing it to library-only checks.
- The scheduler assigns work; some async processing, CLI and DAA paths still simulate or use heuristics. This is not a complete autonomous coding runtime.
- MCP stdio and WebSocket use different service paths. Per-session WebSocket quotas are not a comprehensive stdio quota model. Authentication, multi-tenant authorization, session accounting on replacement/expiry and workflow execution remain incomplete. Some older security tests only construct requests.
- Shared-memory cross-process mapping is not implemented. Transport authentication/TLS/reconnect semantics and a complete codec allocation/lifecycle audit need further review.
- ML training scaffolds do not implement LSTM/TCN/N-BEATS. The real ML bridge is univariate and lacks supported model import/probabilistic covariates. Metadata memory estimates are not hard limits.
- Browser compile checks, actual Node/Chromium WASM tests and native numeric tests are recorded separately above. npm artifact rebuilding/provenance and JavaScript adapter validation are coordinated by the npm owner; their final fresh-source receipt is a separate release gate. The 2.0 facade intentionally removes unsupported prototype APIs.
- SWE temporary directories provide no OS/network/credential isolation. Dataset conversion, repository revision pinning and official harness evaluation are still needed before reporting benchmark results. Legacy simulation helpers remain clearly labeled.
- Bench executor modes, quality/resource metrics and parts of the dashboard remain prototype/heuristic paths. No controlled comparative performance experiment or external accuracy benchmark was completed.
- Source listing checks pass for all 14 packages. Coordinator-owned verification of all extracted archives with unpublished path dependencies and final npm receipts remains a separate gate; this report does not claim publication. Do not assume that candidate versions have shipped.

## Primary references and research decisions

1. [Rust `Read::take`](https://doc.rust-lang.org/std/io/trait.Read.html#method.take) — bound expanded/read data before accepting it. The limit+1 pattern distinguishes an exact boundary from an oversized stream.
2. [Tokio WASM support](https://docs.rs/tokio/latest/tokio/#wasm-support) — browser targets support a restricted feature set; compiling native-thread integrations is not a browser capability. The final facade removes unused dependencies instead of hiding the compile error.
3. [DashMap API/locking notes](https://docs.rs/dashmap/latest/dashmap/struct.DashMap.html) — map guards can retain locks; clone/snapshot handles before awaiting external work.
4. [Tungstenite WebSocketConfig](https://docs.rs/tungstenite/0.23.0/tungstenite/protocol/struct.WebSocketConfig.html) — set actual frame/message budgets instead of relying on large library defaults. The pinned crate's source was inspected locally.
5. [SQLx core 0.8.6](https://docs.rs/sqlx-core/0.8.6/sqlx_core/) and [SQLite driver 0.8.6](https://docs.rs/sqlx-sqlite/0.8.6/sqlx_sqlite/) — driver/internal APIs are semver-exempt, so the private component dependencies are exact-pinned together. Their source documentation was inspected before narrowing the dependency graph.
6. [RustSec advisory database](https://github.com/RustSec/advisory-db) — findings and patch/maintenance status are reported against the exact feed/lock hash above.
7. [Rust `black_box`](https://doc.rust-lang.org/std/hint/fn.black_box.html) — best-effort optimization barriers prevent obvious benchmark elimination; they do not replace controlled measurement.
8. [DLinear paper: Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) — a paper's architecture name must not be attached to an unrelated dense baseline. The current legacy DLinear name is explicitly qualified rather than claimed as the paper implementation.
9. [N-BEATS paper](https://arxiv.org/abs/1905.10437) — an architecture requires its actual block structure, learned parameters and evaluated forecasts; the training scaffold is marked unsupported instead of treated as SOTA.
10. [Rexie 0.6.2](https://docs.rs/rexie/0.6.2/rexie/) and [IndexedDB transaction specification](https://w3c.github.io/IndexedDB/#transaction-lifetime) — transaction scope/lifetime and awaited commit determine persistence; independent trait calls cannot pretend to share a completed transaction.
11. [wasm-bindgen testing](https://rustwasm.github.io/docs/wasm-bindgen/wasm-bindgen-test/index.html) — matching binding tools and actual runtime execution distinguish compiled target support from a running browser/Node contract.
12. [Official SWE-bench implementation/harness](https://github.com/SWE-bench/SWE-bench) — a local prompt/patch adapter and mocked metrics do not establish official solve rates or evaluation isolation.

The implementation decision is to expose working, tested baselines and explicit unsupported results before adding architecture names or benchmark claims. Research-informed follow-up should add real model backends, golden/reference outputs, bounded loading and controlled evaluation before advertising a new capability.
