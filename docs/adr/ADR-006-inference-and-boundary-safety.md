# ADR-006: Reuse inference storage and validate external data

## Status

Accepted for the compatible modernization iteration, 2026-11-07. Implementation and executed checks are recorded in [the core review](../review-core.md) and [the modernization report](../modernization.md).

## Context

The core forward pass copied each preceding layer's outputs into a new vector. Repeated small-network inference therefore performed avoidable allocations. Removing those copies is a narrower change than replacing the network's public representation with packed weights or a new backend.

External model and training-data formats also contain counts and indices that can drive allocation or execution. Existing byte limits did not consistently constrain text lines, declared topology, or streaming decompression. Deserialization must validate both resource use and model structure.

The earlier [deserialization ADR](ADR-001-deserialization-safety.md) establishes the safety goal. [ADR-002](ADR-002-private-struct-fields.md), [ADR-004](ADR-004-unified-error-types.md), and [ADR-005](ADR-005-simd-integration.md) describe broader API and backend changes. Their accepted design status does not establish that all of those changes are implemented. This decision defines the scope of the current compatible changes.

## Decision

### 1. Reuse live neuron state during inference

Borrow adjacent layers using disjoint mutable slices. Read source neuron values directly while calculating the next layer, preserving connection order, bias behavior, activation selection, and the generic scalar type. Do not build a temporary output vector for each layer.

Keep `Network::run(&mut self, inputs) -> Result<Vec<T>, NetworkError>` and add:

```text
Network::run_into(&mut self, inputs: &[T], outputs: &mut [T])
    -> Result<(), NetworkError>
```

The caller's output slice must have exactly `num_outputs()` elements. Return the existing `DataSizeMismatch` variant for the wrong output length and `InputSizeMismatch` for the wrong input length. These dimension errors leave both network state and the output slice unchanged. `run()` still returns an owned vector; `run_into()` performs no inference allocation with valid dimensions.

This does not introduce a packed-weight cache. Network fields remain publicly mutable in this iteration, so a cache would need an invalidation contract that does not currently exist.

### 2. Make structural validation explicit

Expose `Network::validate()` for callers that edit public fields or deserialize through generic Serde APIs. `Network::from_bytes()` invokes it before returning a network.

Validation checks nonempty regular neurons, finite numeric model state and weights, a finite connection rate in `[0, 1]`, source indices against the actual preceding layer, and bias placement/state. An input layer cannot have incoming connections.

Avoid an additional full structural scan on every inference call. Direct public mutation and third-party deserialization therefore require the caller to validate before execution. `run()` and `run_into()` check their slice dimensions; they are not substitutes for validation of a manually edited model.

### 3. Bound text inputs and declared allocations

Use checked arithmetic for dimension products and allocation estimates before constructing a topology or training-data buffers. Apply these default budgets:

| Boundary | Default limit |
| --- | --- |
| FANN text model input | 256 MiB |
| Estimated decoded FANN topology allocation | 256 MiB |
| Individual text line, including ignored fields | 16 MiB |
| Training-data input | 1 GiB |
| Training-data header estimate, including rows and `f32` values | 1 GiB |
| Decompressed streaming output | 1 GiB |

`FannReader::read_network_with_limit` allows a caller-selected input/topology budget. The line cap remains separate. Reject invalid/overflowing shapes and non-finite parameters instead of attempting the declared allocation. Reuse training-data streaming buffers and do not preallocate a whole dataset solely from an unchecked sample count.

These are input and allocation estimates, not a hard upper bound on total process memory. A parser's temporary storage, allocator overhead, concurrent calls, and the caller's retained data also consume memory.

### 4. Cap expanded data at the writer boundary

Add `decompress_data_with_limit(reader, writer, max_bytes)` and make the existing streaming decompressor use the default cap. Write at most the permitted expanded prefix and read a sentinel to detect excess data. Return an error if the stream exceeds its budget.

A failure may leave that prefix in the writer. Callers needing atomic replacement must write to a temporary destination and promote it only on success. A compressed-input byte limit and a decompressed-output byte limit protect different resources.

### 5. Verify CPU helper contracts independently

Validate vector/matrix dimensions and checked size arithmetic before unsafe SIMD kernels, including release builds. Filter CPU dispatch for every instruction feature actually used, with scalar fallback. Tests must cover the gradient outer product and shapes that do not align to a vector width.

This validates the exposed helpers; it does not assert that all inference or training now executes through a new SIMD backend. Extending backend integration remains a separate measured change.

## Alternatives and tradeoffs

A packed contiguous backend could improve locality and vectorization, but replacing public mutable structures would widen this change and require a cache or migration policy. Retaining the current representation permits the allocation reduction without that API break.

Privatizing every field and replacing every public error type would improve invariant control, but would break direct field access and exhaustive error handling. Preserve those contracts here and expose explicit validation. A future version can adopt the broader API changes with migration guidance.

A new serialization dependency or integer encoding could improve maintenance prospects, but changing the bincode format silently would break stored models. Maintain bounded decoding of the existing format and plan a versioned migration. The bincode maintenance advisory remains a recorded risk.

## Acceptance evidence

Run the core library tests, the allocation regression test, default-feature doctests, and the no-default-features compile check. The regression sources are [core_regressions.rs](../../src/tests/core_regressions.rs) and [inference_allocations.rs](../../tests/inference_allocations.rs).

```bash
cargo test --locked --lib
cargo test --locked --test inference_allocations
cargo test --locked --doc
cargo check --locked --no-default-features
```

The allocation test checks 64 repeated `run_into` calls with zero allocations and checks that `run` allocates only its final output vector. Structural and I/O tests must reject corrupt indices, oversized counts, unbounded lines, non-finite values, malformed training rows, and decompression beyond an exact cap, while valid round trips continue to pass. SIMD checks require numerical reference comparisons and invalid-shape cases.

Benchmark release builds at fixed model shapes and inputs before making latency claims. Allocation counts explain the removed work; they do not establish a universal speedup. See the [research review](../research-modernization.md) for the primary sources, measurement requirements, and deferred GPU/quantization work.
