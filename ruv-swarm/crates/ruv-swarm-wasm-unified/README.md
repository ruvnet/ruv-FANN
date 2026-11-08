# ruv-swarm-wasm-unified

A small browser facade over the implemented `ruv-swarm-wasm` bindings. Candidate 2.0.0 deliberately removes the earlier prototype API that advertised neural, forecasting and persistence components without implementing them.

## Build and check

```sh
cargo +1.88.0 check --locked -p ruv-swarm-wasm-unified --all-features --lib --target wasm32-unknown-unknown
cargo test --locked -p ruv-swarm-wasm-unified --lib
wasm-pack build crates/ruv-swarm-wasm-unified --target web --out-dir pkg
```

Use the base module's checked `WasmNeuralNetwork`, `WasmForecastingModel`, memory pools, vector helpers and local swarm state. `get_unified_version()` reports the facade version; `get_version()` reports the base implementation version. The default `simd` feature forwards to the base crate; target instructions still depend on build flags and runtime support.

## Migration from the prototype

The old `core`, `neural`, `forecasting`, `persistence`, `utils` modules and `WasmConfig` are not exported. Their source remains in this checkout for reference but is not compiled. Code using them needs to move to the base API or supply a real application implementation. There is no automatic migration of prototype-created model IDs or storage data.

Unused native-thread, ML, SQLite and parallel dependencies were removed. This lets the actual browser target compile without claiming those integrations work. The facade does not include IndexedDB persistence, 27 neural architectures, background worker scheduling or browser training pipelines. See the base package README for input limits and scope.

The versions in this checkout are **unpublished release candidates**. See the [workspace guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm) for all 14 packages and the [review record](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-rust-swarm.md) for checks, dependency findings and remaining limits.

![Animated map of the ruv-FANN packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)
