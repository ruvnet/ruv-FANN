# Candle linear models in ruv-FANN

This optional Rust crate converts actual CPU `candle_nn::Linear` tensors into
owned ruv-FANN prepared inference. It pins `candle-core` and `candle-nn` 0.11.1
with default features disabled. The core ruv-FANN crate acquires no Candle
dependency. The adapter uses the reviewed public tensor APIs; it does not copy
Candle's tensor engine or require a model download. The upstream API review used
[Candle commit cfcbd718](https://github.com/huggingface/candle/tree/cfcbd7186d13271843ce7eb41d25659bd0835850),
licensed MIT OR Apache-2.0.

```rust
use candle_core::{Device, Tensor};
use candle_nn::Linear;
use ruv_fann::ActivationFunction;
use ruv_fann_candle::{prepare_f32, LinearStage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let linear = Linear::new(
        Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?,
        Some(Tensor::new(&[0.5f32, -0.25], &Device::Cpu)?),
    );
    let prepared = prepare_f32(&[LinearStage {
        layer: &linear,
        activation: ActivationFunction::Linear,
        steepness: 1.0,
    }])?;
    let mut output = [0.0f32; 2];
    prepared.run_into(&[10.0, 100.0], &mut output, &mut prepared.workspace()?)?;
    assert_eq!(output, [210.5, 429.75]);
    Ok(())
}
```

`prepare_f64` is the strict Float64 counterpart. The explicit-limit variants are
`prepare_f32_with_limits` and `prepare_f64_with_limits`. Weights have logical
shape `[outputs, inputs]`, and a present bias has shape `[outputs]`. Adjacent
stages must match. Activation and steepness use ruv-FANN's documented semantics;
they are supplied explicitly rather than inferred from an autograd graph.

All stages' shapes, dtypes, CPU placement and resource counts are checked before
numerical data is copied. Only f32-to-f32 and f64-to-f64 are accepted. There is no
implicit dtype conversion, GPU transfer or tensor-file loading. Non-contiguous
strides, broadcasts and nonzero storage offsets are read in logical row-major
order using safe Candle storage guards. No temporary contiguous Candle tensor is
created. One guard is held at a time.

Stop concurrent source-variable updates while preparing. The resulting snapshot
has its own parameters; after return, source tensors may be updated or dropped.

## Bounds and numeric behavior

The adapter accepts at most 64 stages and honors the core's neuron/connection
limits. Its `max_bytes` check is deliberately stricter than the core's
packed-only limit: a conservative estimate includes temporary copied values,
the final packed values, descriptor allowances and one ordinary workspace. It
excludes caller-owned Candle storage, stack, allocator slack and process-wide
state. Thus a model can fit the core alone but fail the adapter's copy budget.

The requested-byte estimate uses 64 bytes per output row, 64 bytes per layer,
512 bytes per stage of import metadata, and 1 KiB fixed allowance in addition to
both parameter copies and workspace buffers. These are conservative allowances
for the current adapter/core layouts, not public Rust ABI sizes.

FANN retains its own per-neuron product/addition order. Candle GEMM can use
different reduction order or FMA, so the external CPU oracle tests use explicit
f32 tolerances (`atol=5e-6`, `rtol=2e-5`) and f64 tolerances (`1e-12` each).
FANN scalar-versus-batch output comparisons remain exact, including zero signs.
These tests do not establish accuracy of an untrained or arbitrary user model.

## Export to WASM or napi-rs

`export_dense_f64_json` emits the core's `ruv-fann.dense-layers/1` interchange
schema from actual Candle tensors. The current WASM and native Node prepared
bindings accept that Float64 schema through `from_dense_layers_json`.
`export_dense_f32_json` targets the Rust Float32 parser; interpreting those
numbers in Float64 does not preserve Float32 arithmetic. Convert deliberately
in the source application when a different precision is intended.

Export uses the same metadata checks and finite-value validation, and streams
JSON into a capped writer: at most 4 MiB UTF-8 and 64 layers, with no returned
partial output when the cap is exceeded. A metadata-only size lower bound rejects
provably oversized JSON before copying source parameters. This interchange
describes sequential dense layers, rather than arbitrary Candle operators.

## Training and gradients

The exported JSON can initialize ruv-FANN's `AutogradNetwork` for dense MSE
training with SGD or AdamW, then produce an owned inference snapshot. It supports
an explicit sequence of Linear, Sigmoid, Tanh and ReLU activations; it does not
reconstruct a Candle autograd graph or train a transformer.

The independent CPU training tests use actual Candle `Var`, `loss::mse`,
`loss.backward()` and optimizer steps. They compare loss, predictions, parameter
gradients, input gradients and three updates, with both bias choices at each
layer and multiple input rows/outputs. f32 uses `atol=5e-6`, `rtol=3e-5`; f64
uses `1e-12` each. The ReLU gradient oracle stays away from zero: Candle chooses
derivative 1 at exactly zero while ruv-FANN preserves derivative 0 there.
Parameter/gradient order is each layer's row-major weights, then optional bias.

## Verify

The adapter is tested with Rust 1.99.1; its declared minimum is 1.99. The separate
core and binding crates retain their own toolchain requirements.

```sh
CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 \
  cargo +stable test --locked --manifest-path integrations/candle/Cargo.toml

cargo +stable run --locked --manifest-path integrations/candle/Cargo.toml \
  --example export_fixture > /tmp/candle-f64-fixture.json
```

The example emits `{oracle, dense_json, input, expected, ...}` from real Candle
Float64 CPU operations for downstream WASM/Node conformance checks. It includes
the MSE target/loss, parameter/input gradients and expected SGD/AdamW updates.
The fixture contains deterministic numerical data, not downloaded weights. Tests cover
stacked layers, all bias combinations, logical strides/offsets, dtype/shape/byte
rejection, source ownership and bounded JSON export. CUDA/Metal execution is not
tested or claimed.
