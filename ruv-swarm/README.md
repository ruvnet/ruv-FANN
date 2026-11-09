# ruv-swarm

Rust libraries and JavaScript tooling for assigning work to agents, moving messages, storing state and experimenting with small neural forecasts. An agent is an application worker: it runs the handler you provide. Scheduling a task does not create an AI model or execute that task by itself.

This directory contains **14 Rust workspace packages**. The JavaScript distribution has its own [guide](npm/README.md). The versions below are unpublished release candidates; this review has not published a crate, npm package or release.

![Animated map showing the ruv-FANN package families](../docs/assets/package-map.svg)

## Start with a working example

From this directory:

```sh
cargo run --locked -p ruv-swarm-core --example basic_swarm
```

The example registers two workers and assigns three tasks:

```text
report-1 -> worker-a
report-2 -> worker-b
report-3 -> worker-a
```

Its source is included in [the core crate](crates/ruv-swarm-core/examples/basic_swarm.rs). Use `SwarmConfig` to choose a distribution strategy, register workers with their capabilities, submit `Task` values and inspect the assignments. The application then calls the appropriate worker and records its output.

For a typed worker with a real handler:

```rust
use ruv_swarm_agents::FunctionAgent;
use ruv_swarm_core::Agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut worker = FunctionAgent::new(
        "double", vec!["compute".into()],
        |number: i32| async move { Ok::<i32, std::io::Error>(number * 2) },
    )?;
    assert_eq!(worker.process(21).await?, 42);
    Ok(())
}
```

An application using that snippet needs `ruv-swarm-agents`, `ruv-swarm-core` and Tokio. Use local path dependencies while testing these unpublished candidates.

## Choose a package

| Package | Candidate | What it provides |
| --- | --- | --- |
| [ruv-swarm-core](crates/ruv-swarm-core) | 1.0.7 | Agent/task types, topology and native assignment strategies |
| [ruv-swarm-agents](crates/ruv-swarm-agents) | 1.0.6 | Typed async function handlers with application-owned work |
| [claude-parser](crates/claude-parser) | 1.0.6 | Bounded line-delimited event parsing and exports |
| [ruv-swarm-transport](crates/ruv-swarm-transport) | 1.0.8 | In-process channels, WebSockets and an in-process ring prototype |
| [ruv-swarm-persistence](crates/ruv-swarm-persistence) | 1.0.8 | SQLite, memory and browser IndexedDB records/metrics |
| [ruv-swarm-ml](crates/ruv-swarm-ml) | 1.0.6 | Trained dense univariate forecasts and fitted ensembles |
| [ruv-swarm-mcp](crates/ruv-swarm-mcp) | 1.0.6 | MCP stdio service and separate local WebSocket/JSON-RPC handlers |
| [ruv-swarm-cli](crates/ruv-swarm-cli) | 1.0.6 | Native CLI, configuration and demonstration commands |
| [ruv-swarm-daa](crates/ruv-swarm-daa) | 1.0.7 | Experimental feedback, learning-state and coordination policies |
| [ruv-swarm-wasm](crates/ruv-swarm-wasm) | 1.0.6 | Checked browser inference, forecast baselines and vector helpers |
| [ruv-swarm-wasm-unified](crates/ruv-swarm-wasm-unified) | 2.0.0 | Thin facade over the implemented browser bindings |
| [swe-bench-adapter](crates/swe-bench-adapter) | 1.0.6 | Local dataset/prompt helpers and application-generated patch evaluation |
| [ruv-swarm-ml-training](ml-training) | 1.0.6 | Event features and experimental training interfaces |
| [ruv-swarm-benchmarking](benchmarking) | 1.0.6 | Experimental execution, SQLite results and comparison helpers |

Registry versions differ from checkout versions. On 2026-11-07, the official crates.io API reported core/DAA 1.0.6; parser/agents/ML/training/WASM/transport/persistence 1.0.5; and CLI/MCP/SWE adapter 0.2.0. Benchmarking and unified WASM returned 404. These observations establish neither ownership nor publication readiness. The [review record](../docs/review-rust-swarm.md) includes the full candidate/registry matrix and source references.

## What changed in this review

- Scheduling preserves FIFO order, stably orders priorities and advances round robin across batches. Duplicate worker IDs cannot silently replace an existing worker.
- In-process broadcast reaches `receive`; message budgets apply at the receiving endpoint. WebSocket limits bound both compressed and expanded messages.
- Ring cursors and payload updates share one lock, preventing concurrent frame corruption. Wrapped data uses bulk slice copies.
- Parser input and retained events have explicit budgets. WebSocket MCP quota reservations enforce limits atomically and reject arithmetic overflow.
- In-memory metric aggregation retains a sum/count per group rather than every value. SQLite benchmark storage avoids unused MySQL/RSA dependencies.
- Forecasts use trained network output. Unsupported model names return errors. Browser inference preserves `f64` values, checks shapes and reuses layer buffers.
- Source packages allow only intended code, tests and assets; database files, sessions and other runtime state are excluded.

These changes are supported by regression tests. They do not establish a universal speedup, benchmark lead, token saving or official SWE-bench solve rate.

## Current capability boundaries

**Models.** The Rust ML factory implements an MLP, a linear baseline under the legacy `DLinear` name and a wider MLP under `MLPMultivariate`. The latter remains univariate. LSTM, TFT and other unimplemented names return errors. The separate training package's LSTM/TCN/N-BEATS backends also return errors; no-op saves and mock predictions have been removed.

**Work execution.** The scheduler returns assignments. CLI, DAA and some MCP processing paths retain demonstration or heuristic behavior. A cognitive-pattern label is configuration, not a trained reasoning capability. Use an application-owned handler to perform actual work.

**Communication.** The shared-memory transport's ring currently lives in process memory rather than the mapped OS segment. Cross-process delivery is not implemented by that prototype. Peer authentication, TLS configuration and delivery acknowledgments need application design.

**Evaluation.** The SWE adapter can use a mutable agent that returns an actual `ExecutionResult`; the legacy shared-agent API fails closed. Its temporary-directory setting is not an operating-system sandbox, and its simulation benchmark helpers cannot establish real performance. Use externally isolated execution and the official SWE-bench harness before reporting results.

**Browser support.** Base and unified bindings compile with all Cargo features for `wasm32-unknown-unknown`, using ordinary and SIMD target flags. Core minimal/`no_std` builds and the DAA, transport and persistence browser feature paths also compile. Two transport tests execute actual WASM in Node; seven Chromium checks exercise actual IndexedDB, including rollback, racing task claims and persistence across reopening. Browser transport currently supports in-process messaging; native WebSocket/shared-memory backends are platform-specific. IndexedDB is a separate persistence crate, not part of the unified facade. Generated npm artifacts require a fresh build and provenance receipt.

**Service security.** MCP WebSocket limits cover that handler path. Stdio has a separate service path, and complete authenticated multi-tenant isolation is unfinished. Read the package's scope before exposing a service beyond trusted local clients.

## Verify the checkout

Native packages now require **Rust 1.89** to use patched `rusqlite`/SQLite. The minimal `ruv-swarm-wasm` crate explicitly retains Rust 1.88; unified/all-feature workspace builds use 1.89. SQLite is pinned to 3.54.1 and rmcp to 3.9.0; the benchmark store uses the same SQLite supplier. The current targeted native gate passes 144 library tests, two actual SQLite version/durability gates, and a bounded real MCP stdio protocol check.

Historical acceptance on the previous Rust 1.88 dependency graph passed **482 tests, 48 Criterion smoke cases and 5 doctests**, using all native targets and Cargo features. All 14 examples built; 11 offline executable smoke checks passed. Those counts describe that earlier graph; current verification uses the commands below. Two compiled examples need an external Claude executable/provider and were not exercised against a live service.

```sh
cargo +1.89.0 test --locked --workspace --all-targets --all-features --no-fail-fast
cargo +1.89.0 test --locked --workspace --all-features --doc
cargo +1.89.0 build --locked --workspace --all-features --examples

cargo +1.89.0 check --locked -p ruv-swarm-wasm -p ruv-swarm-wasm-unified \
  --all-features --lib --target wasm32-unknown-unknown
RUSTFLAGS="-C target-feature=+simd128" cargo +1.89.0 check --locked \
  -p ruv-swarm-wasm -p ruv-swarm-wasm-unified \
  --all-features --lib --target wasm32-unknown-unknown
```

For a small machine, set `CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0`. Native timing checks depend on the host and concurrent workload. The [review record](../docs/review-rust-swarm.md) distinguishes runtime checks, compilation checks and capabilities that still return explicit unsupported-operation errors.

## Release status

All 14 packages have staged source candidates and package allowlists. Publishing is a separate gated step: verify the dependency order, generated artifacts, package contents and registry state, then obtain the release owner's authorization. Nothing in this mission was published.

The dependency review removed unused vulnerable archive/tensor/driver dependencies rather than adding audit exceptions. Remaining maintenance advisories and feature gaps are listed in the [review record](../docs/review-rust-swarm.md). See the [repository modernization guide](../docs/modernization.md) for the wider project.

Most packages are MIT OR Apache-2.0; DAA declares MIT. See each manifest and the [MIT](LICENSE-MIT) and [Apache](LICENSE-APACHE) license texts.
