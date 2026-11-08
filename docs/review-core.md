# Core review and implementation evidence

Reviewed 2026-11-07, starting at `f4ca36c2323d08eac822bac43b62040198721cbf`.
This report covers the root `ruv-fann` crate. Other Rust workspaces and the npm
distribution have separate reviews. Changes are compatible with existing
network serialization and preserve the public error enum.

## Outcome

The core now offers `Network::run_into(inputs, outputs)` for inference into a
caller-owned slice. Adjacent layers borrow each other's neuron values instead
of allocating temporary output vectors. The existing `run()` API still returns
an owned vector. An allocator regression test confirms zero allocations for 64
repeated `run_into` calls and one final-vector allocation for `run` on a
five-layer network. This allocation result does not establish a universal
latency improvement.

All declared scalar activation variants now have implementations and matching
derivatives. Previously, GELU, Swish, the `LeakyRelu` alias, and several FANN
variants silently used identity output or derivative. GELU uses the documented
tanh approximation. Existing sigmoid slope behavior is preserved and its enum
documentation corrected to match. GELU's derivative handles saturated finite
inputs without the `0 * infinity` numerical failure. The SIMD helper uses the
same canonical GELU/Swish derivative implementation to avoid numerical drift.

The security changes check unsafe CPU boundaries and validate resource counts
before constructing models or retaining training rows. Their contracts and
tradeoffs are recorded in [ADR-006](adr/ADR-006-inference-and-boundary-safety.md).

## Confirmed findings and fixes

| Finding | Exposure and evidence | Remediation |
| --- | --- | --- |
| High: out-of-bounds SIMD access through safe APIs | `CpuSimdOps` used only `debug_assert!` before raw AVX loads/stores, so optimized builds lacked the safety checks | Always-on buffer validation and checked dimension multiplication before every unsafe dispatch; release-mode regression tests |
| High: unsupported CPU instructions | User configuration could force AVX2; matrix kernels also used FMA without checking it | Hardware-filtered AVX2/FMA preferences and explicit target-feature annotations, with scalar fallback |
| High: counts can drive excessive allocation | FANN text `layer_sizes` could request `usize::MAX` neurons; training loader reserved attacker-controlled sample counts | Checked topology/container estimates before construction, no dataset reservation from header, bounded lines and total input |
| High: streaming decompression has no output cap | Exported `decompress_data` copied expanded data until EOF; only the private byte helper had a cap | Default 1 GiB output cap; public `decompress_data_with_limit` writes at most the chosen prefix and errors on excess |
| Medium: network topology validation uses the wrong layer | `from_bytes` checked a source index against the current width plus 1024, rejecting some valid wide fan-in and accepting invalid indices | Public `validate()` checks the actual previous layer and finite model state; binary network loads call it |
| Correctness: scalar activation fallbacks change the requested model | Several public enum selections returned identity output or derivative | Exhaustive implementations; finite-difference derivatives for all 19 differentiable variants |
| Correctness: gradient matrix dimensions are wrong | Parallel helper supplied inner-product dimensions for an outer product | Use `(errors.len(), activations.len(), 1)` and verify the expected 2-by-3 gradient |
| Correctness: empty and single-layer builders panic | Unsigned subtraction and the one-element middle-layer slice underflowed | Saturating range lengths and empty/single-layer regressions |

The SIMD findings are confirmed by direct inspection of reachable safe methods,
not scanner-only signals. No exploit network traffic or production service was
used. Invalid-shape tests require an ordinary Rust panic before SIMD execution;
they do not deliberately execute invalid memory accesses.

## Limits and compatibility

| Input boundary | Limit |
| --- | --- |
| FANN text model and estimated decoded topology | 256 MiB each by default; `read_network_with_limit` selects a custom budget |
| Individual text line, including ignored parameters | 16 MiB |
| Training input and header allocation estimate | 1 GiB each |
| Default expanded streaming gzip output | 1 GiB; callers can select another output cap |
| Existing binary/JSON input limits | 256 MiB; existing custom-limit functions are now publicly re-exported |

Training streams reject zero-width, truncated, non-finite, or overflowing rows
and reuse the temporary sample buffers. Writers reject inconsistent dataset
headers rather than indexing past their row vectors. Budgets include estimates
of row containers and scalar values; they are not hard limits on process heap
usage. Parser buffers, vector spare capacity, caller retention and concurrency
also consume memory.

`run_into` requires exactly `num_outputs()` output elements. Invalid input or
output lengths leave the network and output unchanged; it uses the existing
`InputSizeMismatch` and `DataSizeMismatch` variants. Model fields remain public.
Call `validate()` after direct field edits or deserialization through generic
Serde APIs. Structural validation is deliberately not repeated on the hot path.

Readers now reject malformed models and non-finite training rows previously
accepted or silently ignored. Fixed-format bincode bytes remain compatible.
Changing a previously unimplemented activation from identity to its documented
function is an intentional correctness change; retrain models that accidentally
depended on the identity fallback.

## Validation

The initial locked default library baseline passed 173 tests on Rust 1.99.1,
x86_64 Linux. The baseline lockfile SHA-256 is
`53b1ea836b334ee6ce3670339336a855ea380656b7ab115832b1734973f54193`.
The coordinator refreshed dependencies and prepared candidate version 0.2.2.
Final checks passed with lockfile SHA-256
`5f372915ca5af4a98992a728f16a331b8965d26b5cffea2ccb95368ad850ec39`:

| Check | Result |
| --- | --- |
| Default library | 191 passed |
| Inference allocation regression | 2 passed |
| Library with no default features | 108 passed |
| Library with standalone `simd`, no defaults | 118 passed |
| Default doctests | 5 passed |
| Release-mode core regression suite | 18 passed |
| All-feature library, CI GPU-unavailability mode | 196 passed |
| Rust 1.81.0 standalone SIMD library check | Passed |
| Formatting for changed Rust files and diff whitespace check | Passed |

Regression sources are [core_regressions.rs](../src/tests/core_regressions.rs)
and [inference_allocations.rs](../tests/inference_allocations.rs). They cover
scalar numerical derivatives, finite saturated GELU, sparse connection order,
invalid sizes without state mutation, decoding indices and bias placement,
text-line and allocation limits, truncated rows, expanded gzip caps, unsafe SIMD
dimensions, irregular vector tails, and the gradient outer product.

Commands:

```bash
cargo test --locked --lib
cargo test --locked --test inference_allocations
cargo test --locked --lib --no-default-features
cargo test --locked --lib --no-default-features --features simd
cargo test --locked --doc
cargo test --locked --release --lib tests::core_regressions
RUV_FANN_CI_TESTING=1 cargo test --locked --all-features --lib
cargo +1.81.0 check --locked --lib --no-default-features --features simd
```

The final all-target/all-feature run uses an actual Mesa software Vulkan
adapter, with no ignored tests: **202 library tests, 2 allocation tests, 7
compute/memory integration tests and 39 Criterion smoke cases pass**. All example
targets compile. Seven all-feature doctests and the Rust 1.81 default-library
check pass after the final changes. Software execution verifies API behavior
and arithmetic; these results measure no physical GPU throughput.

## WebGPU blockers closed

Large batch dispatch previously submitted repeated tiles without offsets,
overwriting the first rows and leaving later rows/batches unwritten. Dispatch
now covers the complete checked two-dimensional grid. The regression validates
129 rows, 79 columns and 101 vectors, including every tail element. Buffer sizes,
workgroup counts and dimension products are checked before allocation/dispatch.
f64 matrix and activation operations retain CPU precision rather than silently
narrowing through f32 GPU storage.

The public activation API now dispatches the requested WGSL kernel for large
f32 arrays, and its fallback uses the canonical scalar definitions for all 21
enum variants. Stable sigmoid tails, unscaled ReLU conventions and saturated
GELU are covered against the neuron implementation. Native tests use a relative
budget of 3e-5. Browser checks use 4e-5, with 1e-3 for the software adapter's
trigonometric intrinsics; this is an explicit acceptance tolerance.

The managed-memory API previously returned size-as-handle tokens, discarded
uploads and downloaded zeros. It now owns real device buffers with unique,
nonrecycled handles, exact f32/f64 byte storage, bounded prefix writes, staging
readback, concurrent lifetime protection and allocation statistics. The managed
budget is the device's maximum buffer size with at most 4,096 allocations;
reported availability excludes transient buffers and does not report free VRAM.
Invalid, freed and cross-backend handles fail. f64 byte storage does not imply
f64 shader support.

The separate device-less legacy `PipelineCache` can no longer return a phantom
pipeline/layout as successful compilation. It returns `UnsupportedOperation`,
and regression tests verify it records no compiled object or cache hit. The
actual `WebGPUBackend` still caches its real device-bound pipelines. Hardware
compute-unit count and bandwidth are reported as unknown instead of fabricated
measurements.

`node scripts/verify-webgpu.cjs` executes the repository shaders in Chromium
through Playwright. All seven WGSL modules compile, 13,158 matrix results and
21,567 activation results agree with their references, and 670 real Adam/AdamW
updates agree across five optimizer steps. This caught the optimizer shader's
read-only gradient binding, duplicated bias correction and missing tail bounds.
These are shader tests, not a claim that every experimental GPU trainer is
implemented. Native execution honors `WGPU_BACKEND`; software adapters require
`RUV_FANN_ALLOW_SOFTWARE_GPU=1`. CI runs both actual native and browser dispatch.

## Final fallback and browser integration checks

The legacy device-less `ComputeContext::GpuMemoryManager` and the CPU/SIMD
vector backends now reject unsupported handle-based storage instead of returning
random/size-based handles, discarding uploads or returning empty downloads.
CPU callers can use the actual `CpuMemoryManager`; GPU callers use the real
`WebGPUBackend` storage. All CPU/SIMD activation fallbacks now share the 21
canonical definitions, with parity regressions. Built-in training losses return
NaN for empty, mismatched or non-finite inputs instead of scoring a truncated
zip as zero error. These three final regression cases are included in the
202-library-test count above.

The old browser bridge failed to compile and advertised worker, storage and DAA
prototypes. It is replaced with a small async `WasmGpuBridge`, using actual
WebGPU requests, buffers, pipeline compilation, dispatch and mapped readback.
This browser API is separate from native blocking `ComputeBackend`.
`webgpu` and `wasm-gpu` compile on wasm32. Native `gpu` now compiles with default
features disabled after adding its required serde/logging feature edges.
Root output supports Rust libraries and WASM cdylibs; native `gpu` is not a
supported wasm32 profile.

`node scripts/verify-rust-browser-gpu.cjs --build <output>` builds the Rust class
with wasm-pack 0.15.1 and matching wasm-bindgen 0.2.130, then runs Chromium.
The final frozen-source run validates **13,902 matrix results**, **21,567
activation results across 21 variants**, and **28 rejected requests**, including
invalid dimensions, unsupported activations, spoofed typed-array prototypes,
concurrent requests, device loss and lifecycle recovery. Growable shared memory
and a 2-million-value input backed by this module's own WASM memory are checked
through real dispatch and forced memory detachment. Outputs have copied
ownership. An independent lifecycle probe confirms freeing an actively borrowed
object is rejected while its Rust operation safely completes.

Limits are 64 MiB per buffer, actual device storage/buffer/workgroup limits and
64 million multiply-adds per matrix operation. Inputs must be genuine
Float32Arrays with finite values. Intrinsic brand/metadata getters reject
prototype spoofing. Fixed-length JS-owned snapshots precede Rust vector
allocation; bounded JS string handles and exact integer JS dimensions prevent
implicit allocation/coercion before validation. The class serializes operations,
cleans temporary buffers on errors and reports completed operations/returned
bytes. These are not process-wide memory or wall-clock quotas.

Final WASM SHA-256:
`ad860f2ba686bd70aa6a5193de665de9bc6c06028281dd9232d8b0369b9eade4`.
Source, lock, tool and artifact identities accompany the final validation
receipt. Root/swarm binding locks share the 0.2.130/0.3.107/0.4.80 family with
forecasting. This is software-adapter evidence, not physical GPU throughput or
full GPU model training.

## Performance experiment

The existing Criterion benchmark now compares three paths using the same
topology and deterministic weights (`0.01`): returning `run`, caller-owned
`run_into`, and an explicit copying reference that retains the old per-layer
vectors. It includes XOR, small, medium, and MNIST-size fully connected models.

```bash
cargo bench --locked --bench neural_network -- forward_propagation --quick
```

Quick samples collected while the swarm compiled other workspaces were
inconsistent between model sizes and paths. They are exploratory evidence,
not a release speed claim. Use a quiet host, normal Criterion sampling,
recorded CPU configuration, and representative sparsity/precision before
publishing comparative latency results. The allocation counter is the stable
acceptance evidence for this iteration.

A subsequent focused `forward_propagation/Medium` quick run on the same shared
AMD EPYC 9V74 x86_64 host produced intervals of 5.270–5.365 microseconds for
`run`, 5.192–5.289 microseconds for `run_into`, and 5.340–5.412 microseconds for
the copying reference. These samples show that the earlier extreme differences
were not reproducible. They remain quick samples, not a statistically
established end-to-end speedup. The reproducible benchmark code, rather than a
headline multiplier, is the release artifact.

## Primary sources and research boundary

- [Rust architecture intrinsics and runtime CPU detection](https://doc.rust-lang.org/core/arch/index.html)
  support the required safety contract for CPU dispatch; preference flags alone
  do not make intrinsics safe.
- [Hendrycks and Gimpel: Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415)
  defines GELU; this implementation uses its tanh approximation rather than
  claiming an exact normal-CDF kernel.
- [Ramachandran, Zoph and Le: Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
  defines Swish; the implementation uses beta equal to one, matching the
  existing public activation description.

These are established numerical building blocks, not evidence that this
feedforward library outperforms current foundation models. Packed weights,
quantization, hardware GPU measurements and broader backend integration need
their own evaluation. See [the research review](research-modernization.md).

## Remaining boundaries

Generic `read_binary<T>` and `read_json<T>` bound encoded input, not arbitrary
custom Serde deserializer CPU work or aggregate concurrent process memory.
The binary format still uses the existing bincode dependency; its maintenance
advisory remains a coordinator-owned migration decision. Binary reader/writer
configuration helpers still have existing unimplemented encoding options; this
iteration does not claim to repair them. The verified compute kernels run on software Vulkan and browser SwiftShader.
Physical GPU performance and every experimental training/backend combination
remain outside the validated contract.
