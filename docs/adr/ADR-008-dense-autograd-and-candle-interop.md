# ADR-008: Dense reverse-mode training and Candle interoperability

Date: 2026-11-09. Status: implemented; validation is recorded separately.

## Problem and scope

Users need to train small dense FANN networks, inspect gradients, and deploy the
same weights through Rust, browser WASM and Node. Candle provides a useful CPU
oracle and a common linear-layer layout. The user specifically prioritized
training and autograd. Existing mutable FANN training and legacy WASM layouts
must retain their behavior.

## Representation and differentiation

`DenseLayer<T>` describes one affine layer with row-major `[outputs, inputs]`
weights, optional bias, activation and steepness. An owned `AutogradNetwork<T>`
copies and validates these parameters. It supports f32/f64 and the analytic
rules for Linear, Sigmoid, Tanh, SigmoidSymmetric and ReLU. ReLU uses derivative
zero at zero and preserves FANN's existing steepness behavior. Unsupported rules
return an error during construction.

The explicit `AutogradWorkspace<T>` owns a fixed-capacity activation tape,
adjoints, parameter/input gradients and proposed parameters. MSE averages over
all `rows * outputs` elements. Reverse traversal computes gradients of the actual
forward graph. A backward pass replaces the previous gradients. Forward,
backward, SGD and AdamW reuse all numerical buffers after setup.

Each graph has an identity and checked parameter revision. Workspaces and AdamW
states bind to that graph. A successful parameter update consumes its gradients;
stale or foreign gradients cannot authorize another update. SGD/manual parameter
replacement invalidates an existing AdamW revision, requiring explicit reset.
SGD and AdamW stage proposed parameters and moments and validate them before
committing. Failures leave model parameters and optimizer state unchanged;
workspace scratch may be invalidated.

## Interchange and deployment

`ruv-fann.dense-layers/1` JSON records sequential dense layers and their current
weights. Parsing/export is capped at 4 MiB UTF-8 and 64 layers. It is a weight
checkpoint; optimizer moments, gradients and arbitrary Candle graphs are not
serialized. `prepare()` produces an independently owned immutable inference
snapshot that survives later training and model destruction.

The optional `integrations/candle` crate pins Candle 0.11.1 in its own lockfile.
It validates CPU device, exact dtype, rank, shape, adjacency, limits and finite
values before safe logical strided copies. Core FANN has no Candle dependency.
The adapter handles non-contiguous and offset tensors without creating another
contiguous tensor. Source mutation must be externally serialized during export.

`WasmTrainableNetwork` and napi-rs `NativeTrainableNetwork` call the actual core.
They accept primitive JSON/numbers and intrinsic ordinary Float64Arrays. They
reject shared/detached/spoofed arrays and copy into preallocated staging, avoiding
live JS/Rust buffer aliases. WASM additionally rejects its own linear-memory
views. Array metadata access uses captured intrinsics. Both bridges cap combined
requested training storage at 64 MiB, capacity at 4096 rows and per-call policy
work at 100 million units. Work estimates are conservative policy counts,
not instruction counts or latency guarantees.

## Inference optimization

Compatible dense rows can reuse input loads across four output accumulators.
Each accumulator preserves FANN's multiplication/addition order; no FMA,
reassociation, numerical unsafe, new BLAS dependency or quantization is added.
Sparse, shuffled and duplicate edges retain their original path. Native tiling
is restricted to 4–32 active rows after observed small/large-batch regressions;
WASM tiling is restricted to at most eight active rows after repeated SIMD
large-batch regressions; larger batches use the original streaming path. Its
scalar/SIMD comparison rebuilds both revisions with Rust 1.88. Measurements
retain negative outcomes.

## Verification and tradeoffs

Finite differences cover supported derivatives in both precisions. Independent
Candle `loss.backward()` tests compare predictions, mean loss, parameter/input
gradients and three SGD/AdamW updates, including all bias combinations and
multiple rows/outputs. Candle GEMM can reassociate or fuse, so external parity
uses explicit tolerances. FANN's scalar/batch comparison remains exact.

Allocation tests, convergence, overflow rollback, identity/revision misuse,
work/memory limits, hostile JS arrays, Workers, snapshot lifetimes and packed
consumers verify the surrounding contracts. Pinned Rune freezes, replays and
freshly verifies the real core probe, then tests source/plan/object tampering.
Separate CI executes Candle, napi-rs and both generated WASM binaries.

The tape scales with capacity and layer widths; model and optimizer storage are
additional. Requested-byte limits exclude caller arrays, allocator slack and
process-wide runtime memory. This fixed sequential graph adds no general tensor
operators, transformer trainer, GPU backward pass, mixed precision or native
binary registry release. Those need their own numerical and deployment proof.
