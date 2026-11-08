# ruv-FANN

Rust libraries for neural networks, time-series forecasting, agent coordination, GPU compute, and image processing.

[![Crates.io](https://img.shields.io/crates/v/ruv-fann.svg)](https://crates.io/crates/ruv-fann)
[![API documentation](https://docs.rs/ruv-fann/badge.svg)](https://docs.rs/ruv-fann)
[![CI](https://github.com/ruvnet/ruv-FANN/actions/workflows/ci.yml/badge.svg)](https://github.com/ruvnet/ruv-FANN/actions/workflows/ci.yml)

**Start with `ruv-fann` if you need a small feedforward neural network in Rust.** It provides generic floating-point networks, training algorithms, and model I/O inspired by [FANN](https://github.com/libfann/fann). Other directories contain separate libraries and tools with their own build requirements.

![Repository map: five independent Rust workspaces for neural networks, agent tools, forecasting, CUDA source translation and image processing.](docs/assets/package-map.svg)

## Choose a package

| Location | Available behavior | Candidate versions / guide |
| --- | --- | --- |
| Repository root | `ruv-fann`: build, train, and run feedforward networks; CPU inference and optional WebGPU operations | `0.2.2` · [Core example](examples/basic_usage.rs) · [API](https://docs.rs/ruv-fann) |
| `ruv-swarm/` | Fourteen Rust packages for scheduling, persistence, transport, MCP, CLI, and WASM | `1.0.6`–`1.0.8`; unified WASM `2.0.0` · [Rust guide](ruv-swarm/README.md) |
| `ruv-swarm/npm/` | Node CLI/MCP, SQLite state, dense WASM inference, statistical forecasts, and capability assignment | `ruv-swarm 2.0.0` · [npm guide](ruv-swarm/npm/README.md) |
| `neuro-divergent/` | Six Rust packages: actual RNN training, naive/seasonal baselines, dated forecasts, chronological evaluation, and checkpoints | `0.2.0` · [Forecasting guide](neuro-divergent/README.md) |
| `cuda-wasm/` | Translate a CUDA subset to Rust or WGSL source; Node wrapper requires the external `cuda2rust` compiler | Rust `0.2.0`; npm `2.0.0` · [Translator guide](cuda-wasm/README.md) |
| `opencv-rust/` | Portable Rust matrices, browser image operations, a small SDK, and native OpenCV C++ bindings | Portable/SDK `4.9.0`; native `opencv-sys 0.2.0` · [Image-processing guide](opencv-rust/README.md) |

These are five independent Cargo workspaces with 26 Rust packages and two public npm distributions. Running `cargo test` at the repository root tests the core crate; use the verification driver below to cover the other workspaces. Versions in this checkout are unpublished release candidates. The [release inventory](docs/release-inventory.json) records their exact identities and separate registry observations.

Each family has a defined scope. Forecasting currently implements a univariate RNN and simple baselines; LSTM/GRU forecasting architectures return unsupported errors. Agent assignment records scheduling decisions; external task execution depends on the integration you supply. CUDA translation emits source and does not execute a GPU kernel. The native OpenCV bridge calls an installed C++ library, while the portable packages implement a smaller set of operations.

## Try the neural-network core

Install [Rust](https://www.rust-lang.org/tools/install), then run the included example:

```bash
git clone https://github.com/ruvnet/ruv-FANN.git
cd ruv-FANN
cargo run --locked --example basic_usage
```

For the published crate, run `cargo add ruv-fann` in your application. To try changes from this checkout, use `cargo add ruv-fann --path /path/to/ruv-FANN` instead. Published API documentation describes the published version; new checkout APIs become available after a release.

This example uses explicit weights so you can check the result without training:

```rust
use ruv_fann::{ActivationFunction, NetworkBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut network = NetworkBuilder::<f32>::new()
        .input_layer(2)
        .output_layer_with_activation(1, ActivationFunction::Linear, 1.0)
        .build();

    // Two input weights, followed by the bias weight.
    network.set_weights(&[0.5, -0.25, 0.1])?;
    let outputs = network.run(&[1.0, 2.0])?;
    assert!((outputs[0] - 0.1).abs() < 1e-6);
    println!("Output: {}", outputs[0]);

    Ok(())
}
```

The calculation is `0.5 × 1.0 − 0.25 × 2.0 + 0.1 = 0.1`. A real task needs suitable data and trained weights. [Training examples and algorithms](src/training/mod.rs) include backpropagation, RPROP, Quickprop, Adam, and AdamW.

![Forward-pass diagram: input numbers pass through weighted connections and activation functions; run returns a vector, while run_into writes to a caller-provided output slice.](docs/assets/inference-flow.svg)

For repeated inference, the checkout adds `Network::run_into(inputs, outputs)`. Pass a mutable output slice whose length equals `num_outputs()`. It reuses that buffer and the network's stored neuron values. `run()` remains the convenient API that returns a vector. Check input and output lengths through the returned `Result`.

## Pick the features you need

| Feature | Purpose and limits |
| --- | --- |
| Default features | Enable model I/O, serialization, compression, logging, and parallel-related utilities. CPU inference does not require a GPU. |
| `io` | Enables file formats and training-data I/O. Keep bounds on bytes, decompressed data, and declared model dimensions when loading external data. |
| `simd` / `parallel` | Expose CPU optimization utilities. A feature being enabled does not mean every network or training operation uses vector instructions or multiple threads. Measure the path you call. |
| `gpu` | Enables the native Rust WGPU backend with checked matrix/batch dispatch, f32 activations, and managed device buffers. It needs a compatible device and driver. f64 compute uses the CPU to preserve precision. It does not route every network or trainer through a GPU. |
| `webgpu` / `wasm-gpu` | On `wasm32`, expose the async `WasmGpuBridge` for f32 matrix multiplication and activations in a browser with WebGPU. Buffers are limited to 64 MiB and device limits; matrix calls have a work budget. GPU unavailability returns an error. These features do not enable the native `gpu` backend. |
| `wasm` | Enables WASM-related bindings and dependencies. Browser execution has a separate test gate; target compilation alone does not establish browser behavior. |
| `no_std` | Currently a feature marker. The core still uses the Rust standard library; this is not a supported bare-metal `#![no_std]` build. |

See [Cargo.toml](Cargo.toml) for the precise feature dependency graph. Native Rust `gpu` tests use software Vulkan. Browser tests load the built Rust WASM bridge and verify matrix results, all 21 activations, invalid inputs, and device lifecycle with Chromium/SwiftShader. A separate harness directly checks all seven WGSL modules and Adam/AdamW updates. These checks establish numerical behavior on software adapters. Physical GPU throughput and every experimental training combination remain unevaluated. See the [core review](docs/review-core.md) for limits, tolerances, and commands.

## Build and test

Use a current stable Rust toolchain for full-repository development. The core declares Rust 1.81; the swarm and all four OpenCV packages declare Rust 1.88. Python 3.11 or newer runs the verification tools. Node 22.13 or newer covers the root tooling and swarm distribution; the standalone CUDA wrapper requires Node 22.12 or newer.

The full OpenCV workspace also needs a C++17 compiler, pkg-config, libclang, and OpenCV 4.4 or newer development files. On Debian/Ubuntu these are typically `g++`, `pkg-config`, `libclang-dev`, and `libopencv-dev`. See the [native setup guide](opencv-rust/opencv-sys/README.md) for a checked user-owned prefix option. Portable OpenCV packages can be tested with `--exclude opencv-sys`. Swarm's SQLite npm addon may also need a C++ build toolchain.

```bash
# Inventory every Cargo workspace and npm distribution.
python3 scripts/verify-packages.py --inventory

# Check and test registered libraries, binaries, tests, examples, and benchmarks.
# Install the native OpenCV prerequisites first.
python3 scripts/verify-packages.py --check
python3 scripts/verify-packages.py --test

# Audit all five Cargo lockfiles (requires cargo-audit).
python3 scripts/verify-packages.py --audit

# Validate README links and the SVG diagram sources.
python3 scripts/verify-packages.py --docs
```

The driver uses `--workspace --all-targets`, enables all features for swarm and forecasting, and selects CUDA's portable `serde` configuration. It also runs the targeted safety regressions and reports every failed command. GPU/browser execution, npm consumers, and Cargo archive verification have separate gates recorded in the [modernization report](docs/modernization.md). An inventory or successful metadata resolution is not a passing test suite.

For work on the core alone:

```bash
cargo test --locked --lib
cargo test --locked --doc
cargo check --locked --no-default-features
cargo bench --locked --bench neural_network
```

To build and test the Rust browser GPU bridge, install wasm-pack, the root npm dependencies, and Playwright's Chromium, then run `node scripts/verify-rust-browser-gpu.cjs --build`. This checks generated Rust WASM through the public browser API. The bridge accepts `Float32Array` inputs and implements compute rather than canvas rendering or browser agent orchestration.

For a single family, pass `--workspace neuro-divergent` to the verification driver or use its manifest directly. The family guide explains optional features and native libraries. Run JavaScript commands from the relevant npm directory; the repository root is an integration harness. See each npm guide for a clean install, genuine artifact build, tests, and packed-consumer validation.

## Performance and research

Measure the workload you intend to deploy. Record the commit, compiler, CPU or GPU, target features, model shape, batch size, thread count, warm-up, and numeric tolerance. Compare identical inputs and weights, and include allocation and transfer costs in end-to-end results. Criterion output is written under `target/criterion/`.

This README makes no numerical speedup, forecasting-accuracy, token-saving, or SWE-bench claims. The repository does not yet provide a reproduced evaluation that supports those comparisons. See the [research and adoption decisions](docs/research-modernization.md) for current primary sources and the [modernization report](docs/modernization.md) for package-specific implementation and verification evidence.

## Contribute

Choose a package, reproduce the issue, and add a regression test for a behavior change. Run that package's checks and the relevant integration checks before opening a pull request. Include the commands and results so another contributor can reproduce them.

Model files, training data, plugins, and network messages cross trust boundaries. Changes to those inputs should validate dimensions and resource use before allocating, return useful errors, and preserve format compatibility or document a migration. See the [deserialization decision](docs/adr/ADR-001-deserialization-safety.md) and [migration notes](docs/MIGRATION.md).

## Credits and license

Thanks to the contributors listed in [CONTRIBUTORS.md](CONTRIBUTORS.md), including Ocean (@ohdearquant), Bron (@syndicate604), Jed (@jedarden), and Shep (@elsheppo), and to the FANN, Rust, NeuralForecast, Tokio, Serde, and WebAssembly communities.

See the repository [LICENSE](LICENSE) and each package's manifest for licensing information.
