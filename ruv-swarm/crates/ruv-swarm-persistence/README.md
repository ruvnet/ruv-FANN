# ruv-swarm-persistence

Store agent, task, event, message and metric records using native SQLite, in-memory storage or browser IndexedDB through the `Storage` trait.

## Open a database

```rust
use ruv_swarm_persistence::{SqliteStorage, Storage};

async fn open_store() -> Result<(), Box<dyn std::error::Error>> {
    let store = SqliteStorage::new("swarm.db").await?;
    println!("Stored agents: {}", store.list_agents().await?.len());
    Ok(())
}
```

Use `MemoryStorage::new()` for an in-memory store. SQLite uses a connection pool and blocking database operations are moved off async executor threads. See the `Storage` trait for task claiming, record queries and maintenance calls.

## Metric aggregation

In-memory aggregation computes a running sum and count per `(agent, unit)` group within the inclusive requested time range. It allocates scratch space per group rather than retaining every value or cloning group strings per sample. The existing 360,000-record performance regression remains part of the tests; its one-second timing threshold depends on the host.

## Browser IndexedDB

Build with `--no-default-features --features wasm --target wasm32-unknown-unknown` and use `WasmStorage`. SQLite, its connection pool and migrations stay native-only. Browser futures remain local to their JavaScript executor; native `Storage` keeps its `Send + Sync` contract.

IndexedDB writes await transaction commit. Task claiming and message updates are atomic; event insertion and its sequence counter share a transaction. `write_batch` provides genuine multi-store atomic writes and aborts the complete batch on an error. The generic `begin_transaction` API returns an error because later independent trait calls cannot share that IndexedDB transaction; use `write_batch` instead.

`checkpoint` is a commit barrier. Browser `VACUUM` returns an unsupported-operation error. `storage_size` reports serialized record bytes, not the browser's physical quota usage. Records preserve full `u64` values and Unicode. In-memory state is lost on exit.

## Check it

```sh
cargo +1.88.0 test --locked -p ruv-swarm-persistence --all-targets --all-features
cargo +1.88.0 check --locked -p ruv-swarm-persistence --no-default-features \
  --features wasm --lib --target wasm32-unknown-unknown
```

Rust 1.88 native acceptance passes 75 tests. A real Chromium WASM fixture passes seven grouped checks: record/index/large-integer round trips, exactly one winner across racing claims on two connections, event counter rollback, atomic message updates, metric aggregation, multi-store rollback, and close/reopen persistence. The durable fixture is `examples/indexeddb_browser_smoke.rs`, driven by `tests/browser-smoke.cjs` with matching wasm-bindgen 0.2.130 and Playwright. Database files and runtime state are excluded from release packages.

The versions in this checkout are **unpublished release candidates**. See the [workspace guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm) for all 14 packages and the [review record](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-rust-swarm.md) for checks, dependency findings and remaining limits.

![Animated map of the ruv-FANN packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)
