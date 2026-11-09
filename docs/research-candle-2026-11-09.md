# Candle source review and adoption decisions

Reviewed 2026-11-09 against upstream
[`huggingface/candle` cfcbd7186d13271843ce7eb41d25659bd0835850](https://github.com/huggingface/candle/tree/cfcbd7186d13271843ce7eb41d25659bd0835850),
dated 2026-11-01, and published Candle 0.11.1. The repository and crate manifests
declare **MIT OR Apache-2.0**. No model weights or external datasets were fetched.
The implementation independently adapts dense layout/workspace ideas and calls
public Candle APIs in an optional adapter; it does not copy the tensor engine.

| Primary source | Observation | Implementation decision |
| --- | --- | --- |
| [Linear](https://github.com/huggingface/candle/blob/cfcbd7186d13271843ce7eb41d25659bd0835850/candle-nn/src/linear.rs) | Weight is `[out,in]`; forward transposes it, with optional bias. Contiguous batch dimensions can be flattened before matmul. | Canonical borrowed dense descriptors and direct owned packing; explicitly preserve legacy WASM's separate input-major layout. |
| [CPU backend](https://github.com/huggingface/candle/blob/cfcbd7186d13271843ce7eb41d25659bd0835850/candle-core/src/cpu_backend/mod.rs) | Matmul handles layout-dependent batching; a broadcast RHS can be reused where strides permit. | Reuse input loads across four dense outputs and bounded sample blocks. Retain original reduction order and sparse fallback. |
| [Layout](https://github.com/huggingface/candle/blob/cfcbd7186d13271843ce7eb41d25659bd0835850/candle-core/src/layout.rs) | Contiguity, nonzero storage offsets and logical indexing are distinct. | Safe storage guards plus public `Tensor::strided_index()`; validate before copying, including transposed/narrowed/broadcast tensors. |
| [Backpropagation](https://github.com/huggingface/candle/blob/cfcbd7186d13271843ce7eb41d25659bd0835850/candle-core/src/backprop.rs) | Reverse traversal accumulates adjoints into a gradient store. | Fixed sequential reverse traversal with reusable tape and explicit parameter/input gradients; no claim of general tensor autograd. |
| [Optimizers](https://github.com/huggingface/candle/blob/cfcbd7186d13271843ce7eb41d25659bd0835850/candle-nn/src/optim.rs) | SGD and AdamW update variable tensors from gradients, with Adam moments/bias correction and decoupled weight decay. | Explicit revision-bound SGD/AdamW, staged atomic updates, and actual three-step Candle CPU comparisons. |
| [MSE loss](https://github.com/huggingface/candle/blob/cfcbd7186d13271843ce7eb41d25659bd0835850/candle-nn/src/loss.rs) | Squared error is averaged over all elements. | Mean over rows times outputs, derivative `2*(prediction-target)/N`; test multirow/multioutput normalization. |

The optional adapter pins `candle-core =0.11.1` and `candle-nn =0.11.1` with
default features off. Its tested/declared Rust toolchain is 1.99.1/1.99. The
independent napi-rs crate pins `napi =3.16.1`, `napi-derive =3.7.0`, and
`napi-build =2.6.0`; these require Rust 1.88. Core remains compatible with 1.81.
New package graphs have separate checked-in locks; existing locks are preserved.

The Candle CPU oracle uses real tensors, `Var`, `loss::mse`, `loss.backward()` and
optimizer state, rather than a second handwritten derivative implementation.
There are 280 backward comparisons and 240 optimizer updates across f32/f64,
five supported activation names, all two-layer bias combinations and three
input rows with two outputs. f32 gradient/optimizer tolerances are `atol=5e-6`,
`rtol=3e-5`; f64 uses `1e-12` for both. ReLU's zero derivative differs between
the engines, so that external comparison stays away from its nondifferentiable
point; FANN's derivative zero there has a separate regression.

Native tiling has a real archived pre-change executable, built with the same
Rust 1.88 thin-LTO profile and deterministic benchmark harness as the candidate.
The initial experiment showed both gains and regressions, motivating a separate
fallback loop and conservative native dispatch. Shared-host timing and bootstrap
block intervals do not prove a universal speedup. Final raw measurements and
backend/compiler provenance belong in the accompanying benchmark receipt.

The final WASM comparison also rebuilds the archived source with Rust 1.88.
An earlier artifact pilot mixed an old Rust 1.99.1 build with a new 1.88 build;
it is retained as a labeled historical observation, not kernel evidence. Tiling
now stops at eight WASM rows, retaining streaming behavior for larger batches.
See [actual results and raw samples](candle-results-2026-11-09.md).

[`ruvnet/rune` 92425512](https://github.com/ruvnet/rune/tree/92425512f0f742e21b747f58f7555104446fc9de)
is the already pinned verification tool. This change expands its actual core
probe and release tests to dense import, autograd and optimizers. Rune evidence
attests the core executable; Candle, NAPI and browser proofs are separate gates.
No model calls, autonomous source promotion or provider spend is required.

See [the architecture decision](adr/ADR-008-dense-autograd-and-candle-interop.md),
[the runnable training example](../examples/dense_autograd.rs),
[Candle adapter](../integrations/candle/README.md) and
[native Node package](../bindings/napi/README.md).
