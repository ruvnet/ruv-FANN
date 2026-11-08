# ruv-swarm-wasm

JavaScript bindings for a checked dense neural network, simple one-step forecast baselines, vector/matrix helpers and local agent state. Build this crate to generate fresh browser artifacts.

## Build and check

From `ruv-swarm/`:

```sh
rustup target add wasm32-unknown-unknown
cargo check --locked -p ruv-swarm-wasm --target wasm32-unknown-unknown
cargo test --locked -p ruv-swarm-wasm --lib
wasm-pack build crates/ruv-swarm-wasm --target web --out-dir pkg
```

For a SIMD build, use a separate output directory and `RUSTFLAGS="-C target-feature=+simd128"`. A Cargo feature alone does not prove that the runtime can execute SIMD instructions. The exported capability flag reports compilation with `simd128`, not a browser runtime probe.

## JavaScript API

After initializing the generated `wasm-pack` module:

```js
const network = new WasmNeuralNetwork([2, 1], ActivationFunction.Linear);
network.set_weights(new Float64Array([2, 3]));
const output = network.run(new Float64Array([4, 5])); // [23]
network.free();
```

Import those symbols from the generated module. Constructors, weight updates, forecasts and inference throw JavaScript errors for invalid inputs. Network shapes require 2–64 nonzero layers and at most 1,000,000 total weights plus biases. Weights and inputs must have exact dimensions and finite values. Dense inference preserves `f64` arithmetic and reuses two layer buffers instead of allocating/transposing weights and converting them to `f32` for every layer.

`WasmForecastingModel` accepts only `linear`, `mean` and `naive`; these are statistical baselines, not trained recurrent models. Empty, nonfinite or oversized series return errors. A `MemoryPool` accepts positive dimensions with at most 64 MiB total reserved capacity and clears returned blocks before reuse.

Vector sigmoid/tanh lanes now follow the same functions as scalar tails. Native regression tests check every remainder, extremes, inference precision, matrix overflow and allocation limits. Browser compile checks do not establish browser runtime performance or a measured SIMD speedup.

`WasmSwarmOrchestrator` maintains bounded local assignment state. `spawn` accepts JSON with `id` (optional), `name` (optional), `type` and `capabilities`. `orchestrate` accepts `description`, `priority`, optional `maxAgents` and `requiredCapabilities`; it selects compatible idle agents and returns `assigned` or `unassigned`. It preserves the supplied task fields and does not claim task execution. After your handler finishes, call `update_agent_status(id, "idle")` to release that worker.

Unknown configuration fields return errors. Configurations are limited to 64 KiB, inventories to 1,000 agents, descriptions to 16 KiB and capability lists to 64 bounded names. The topology is a local label, not a networking implementation. These objects do not execute application tasks, authenticate peers or provide a distributed swarm runtime. All Cargo features, including `full`/ML, compile for the browser target with Rust 1.88, using ordinary and SIMD flags. Those feature checks do not establish browser training performance or complete distributed execution.

The versions in this checkout are **unpublished release candidates**. See the [workspace guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm) for all 14 packages and the [review record](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-rust-swarm.md) for checks, dependency findings and remaining limits.

![Animated map of the ruv-FANN packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)
