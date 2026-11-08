---
name: core-inference
description: "Implement, debug, and measure ruv-fann feedforward inference, prepared immutable inference, training boundaries, model I/O, SIMD, or native/browser WebGPU primitives. Use for root-crate numerical, allocation, input-validation, or backend changes."
---

# Work on the neural-network core

Read root `Cargo.toml`, the relevant files under `src/`, and `docs/review-core.md` in the selected checkout. Use `repo-map` if the checkout or feature selection is uncertain. The core is a generic floating-point feedforward library; other workspaces expose different model types and weight layouts.

## Establish a numerical reference

Use explicit weights before training or optimizing. The documented root-crate example includes a bias weight:

```rust
use ruv_fann::{ActivationFunction, NetworkBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut network = NetworkBuilder::<f32>::new()
        .input_layer(2)
        .output_layer_with_activation(1, ActivationFunction::Linear, 1.0)
        .build();
    network.set_weights(&[0.5, -0.25, 0.1])?;
    let mut output = [0.0f32];
    network.run_into(&[1.0, 2.0], &mut output)?;
    assert!((output[0] - 0.1).abs() < 1e-6);
    Ok(())
}
```

For a consumer, use a local `ruv-fann` path dependency against the selected checkout. `run()` returns an owned output vector; `run_into()` writes a caller-owned output slice of exactly `num_outputs()` elements. It reuses the network's stored neuron values. Its repeated-call allocation guarantee is checked separately from latency.

## Prepare once for independent inference workspaces

Read `src/prepared.rs` and the public exports in `src/lib.rs`. `Network::prepare()`
creates an owned `PreparedNetwork<T>` snapshot; subsequent source edits or source
destruction cannot change it. Allocate an `InferenceWorkspace<T>` once for each
concurrent caller, then pass it to `PreparedNetwork::run_into`:

```rust
// Continue with the explicitly weighted network from the example above.
let prepared = network.prepare()?;
let mut workspace = prepared.workspace()?;
let mut packed_output = [0.0f32];
prepared.run_into(&[1.0, 2.0], &mut packed_output, &mut workspace)?;
assert!((packed_output[0] - 0.1).abs() < 1e-6);
```

Share the immutable snapshot across threads when its scalar type permits; keep
each mutable workspace and output exclusive to a call. A workspace can be reused
for another snapshot if its buffers are wide enough. Repeated inference performs
no heap allocations after preparation, workspace, and output allocation. The
workspace may change on error, but the caller output is committed only on
success; the snapshot remains unchanged.

For custom budgets, use `PreparedNetwork::from_network_with_limits(&network,
PreparationLimits { max_bytes, max_neurons, max_connections })`. Validation runs
before packing. Bias neurons must be last, unconnected, and exactly one.
Regular transient neuron sums/values are ignored. Nonfinite parameters/inputs,
invalid indices, inadequate workspace, numeric overflow, and exceeded budgets
return `PreparedError` instead of unchecked inference.

Dense canonical rows use contiguous scalar weights; sparse, shuffled, and
duplicate connections preserve source accumulation order. This is a scalar
packing optimization, with f32/f64 equivalence coverage. It does not implement
an explicit SIMD kernel, XNNPACK backend, or batch API. Measure preparation cost,
reuse count, allocations, and warm inference separately before claiming a gain.

## Implementation procedure

1. Identify the exact method/feature and reproduce the result with known input, weights, expected shape and tolerance. Check sparse connection and bias order; do not assume a dense weight layout.
2. Keep a scalar implementation as the numerical reference. For activation changes, check both values and derivatives, finite extremes and saturated inputs.
3. Validate dimensions and checked products before allocation, indexing, SIMD dispatch or GPU upload. Rejected input/output lengths must leave the network and caller output unchanged.
4. Call `Network::validate()` after direct public-field mutation or generic Serde deserialization. Validated binary/model readers call structural checks, but the hot inference loop deliberately does not revalidate the entire topology.
5. For SIMD, enforce safe method preconditions in release builds and check the actual CPU features used, including FMA. Cover tail lengths and scalar fallback. A configured AVX preference cannot authorize unsupported instructions.
6. For model/data I/O, bound compressed bytes, expanded bytes, individual lines, declared counts and decoded topology. A wire-byte cap does not bound all heap allocation.
7. Run the smallest relevant regression and then the affected feature combination. Preserve serialization compatibility or document a versioned migration.

## Choose the compute path explicitly

| Path | Supported contract and important limit |
| --- | --- |
| CPU `run`/`run_into` | Generic scalar types and canonical activation behavior; benchmark the invoked path |
| `PreparedNetwork::run_into` | Immutable owned scalar parameters and separate reusable workspace; checked finite inputs and canonical unit bias |
| `simd`/`parallel` utilities | Optional optimization utilities; feature selection alone does not route every operation through them |
| Native `gpu` | Real WGPU f32 matrix/batch/activation and managed buffers; f64 computation keeps CPU precision |
| Browser `webgpu`/`wasm-gpu` | Async `WasmGpuBridge` for bounded f32 matrix multiplication and activations; separate from native `gpu` |

The `no_std` feature is a marker, not an implemented bare-metal build. The browser bridge does not implement canvas rendering, agent execution, persistent shared device storage, or an invented CPU fallback. Unavailable devices and unsupported storage APIs return errors.

## Verify the behavior

Run from the checkout root with dependencies already available:

```sh
cargo test --locked --lib
cargo test --locked --test inference_allocations
cargo test --locked --test prepared_inference --test prepared_allocations
cargo test --locked --lib --no-default-features --features simd
cargo test --locked --release --lib tests::core_regressions
cargo test --locked --doc
```

For a broader planned profile:

```sh
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/repoctl.py" plan --repo "/path/to/ruv-FANN" --profile core
```

GPU work needs actual runtime gates, not only all-feature compilation. With the documented toolchain/browser prerequisites installed:

```sh
node scripts/verify-webgpu.cjs
node scripts/verify-rust-browser-gpu.cjs --build
```

The direct WGSL harness and Rust-to-WASM bridge harness cover different paths. A software Vulkan/SwiftShader pass establishes numerical behavior on that adapter. Report physical-device throughput only after measuring it on physical hardware.

## Acceptance and troubleshooting

Accept a change when the explicit-weight reference, boundary regressions and relevant feature/runtime gates pass with stated tolerances. For optimization, include before/after allocations and measured latency on identical workloads; use `security-performance` for the comparison protocol.

If inference diverges, check activation steepness, bias order, sparse connections and scalar precision. If a GPU check fails before dispatch, inspect adapter/driver limits and feature selection. If an untrusted load fails, retain its error and declared resource counts without dumping sensitive training data.

Read [boundary and GPU details](references/boundaries.md) for resource caps and device-lifecycle checks.
