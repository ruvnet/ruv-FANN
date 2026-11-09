# Dense training, Candle and native Node procedure

Read source in the selected checkout before executing these commands. Use a
task-owned external build/evidence directory, retain locks, and check resource
caps before allocating. The core requires Rust 1.81, the WASM/napi-rs builds use
1.88, and the optional Candle adapter is tested with 1.99.1 (declared 1.99).

## Establish the training contract

1. Describe sequential affine layers with `[outputs, inputs]` row-major weights,
   optional bias, activation and steepness. Borrowed `DenseLayer` inputs are
   copied into the owned `AutogradNetwork`. No source tensor graph is inferred.
2. Allocate `model.workspace(max_rows, max_bytes)` and, for AdamW,
   `AdamW::new(&model, config, max_bytes)`. The combined application budget must
   include model, tape, optimizer, staging, caller data and concurrent workers.
3. Call `backward_mse(inputs, targets, rows, &mut tape)`. Input/target matrices are
   row-major, and loss divides by `rows * outputs`. Read `parameter_gradients`,
   `input_gradients`, and `outputs` before the update. Parameter/gradient order is
   each layer's weights, then optional bias.
4. Call `sgd_step` or `adamw_step`. A successful update consumes gradients.
   Manual parameter replacement/SGD makes previous AdamW state stale; configure
   it anew. A failed pass can invalidate tape contents, while failed optimizer
   steps preserve parameters and moments. Never reuse stale gradients.
5. Call `prepare()` for an independent serving snapshot. Further training or
   freeing the trainer cannot mutate it. Save `to_dense_layers_json()` only as a
   topology/weight checkpoint; resuming optimizer history requires separate state.

`examples/dense_autograd.rs` learns `y=2*x+1` with real AdamW updates. Run:

```sh
cargo +1.88.0 run --locked --example dense_autograd
cargo +1.88.0 test --locked --test autograd --test autograd_allocations
cargo +1.88.0 test --locked --test prepared_dense --test prepared_dense_kernel
```

Accept derivatives only after f32/f64 finite differences, multirow/multioutput
normalization, convergence and rollback/stale-identity checks pass. Repeated
forward/backward/SGD/AdamW calls must allocate zero Rust numerical buffers after
setup; array getters, serialization and public-boundary copies have other costs.

## Use actual Candle as an independent oracle

The optional `integrations/candle` crate pins CPU Candle 0.11.1 with default
features off. It validates rank, dtype, device, shapes, finite values, strides
and budgets before copying through safe storage guards. Serialize source-variable
updates during export. Do not silently cast dtype or move GPU tensors to CPU.

```sh
cargo +1.99.1 test --locked --manifest-path integrations/candle/Cargo.toml
cargo +1.99.1 run --quiet --locked --manifest-path integrations/candle/Cargo.toml --example export_fixture
```

Capture the second command's stdout in an external fixture file and set
`RUV_FANN_CANDLE_FIXTURE` to its absolute path for the JS/browser tests. The
fixture comes from actual `Var`, MSE/backward and three optimizer updates, not
another manually coded derivative. Compare loss, predictions, parameter/input
gradients and updates with stated tolerances. Native/WASM exp implementations
can differ by an ulp; each engine's scalar/batch arithmetic remains exact.
ReLU zero has a different Candle subgradient; test FANN's zero rule separately.

## Cross the WASM and napi-rs boundaries

Both trainable classes expose `from_dense_layers_json(json, max_rows)`,
`mse_backward`, gradient/prediction getters, `sgd_step`, `configure_adamw`,
`adamw_step`, `prepare`, weight JSON export and explicit lifecycle cleanup.
The Node WASM facade provides `runtime.createTrainableNetwork(json, maxRows)`.
Native Node builds explicitly from `bindings/napi`; it never downloads a binary
or silently falls back to WASM.

```sh
node bindings/napi/scripts/build.mjs --target-dir /absolute/external/build-directory
node --test bindings/napi/test/*.test.mjs
node --test ruv-swarm/npm/test/wasm-dense-import.test.js ruv-swarm/npm/test/wasm-autograd.test.js
node ruv-swarm/npm/test/browser-training.cjs --output /absolute/external/new-browser-proof.json
```

The native helper needs prefetched pinned dependencies. Verify the resolved
compiler, actual Cargo artifact and before/after input hashes. For WASM changes,
use the repository's genuine core+SIMD rebuild and source-aware provenance gate.
Never rewrite a receipt to authorize old bytes after source changes.

JS inputs must be primitive bounded strings/numbers and real ordinary
Float64Arrays. Test wrong dtype, prototype/tag spoofs, proxies, shared/detached
buffers, metadata getters, aliases and overflow. WASM must reject views into its
own memory. Capacity is at most4096, JSON4MiB/64layers, requested training storage
64MiB and conservative per-call work100M. Requested bytes are not process RSS.

Finally verify a frozen trained model through the existing encoder, real
RuVector search and optional MetaHarness. Record actual retrieval and confirmation
results; a completed tuning run may correctly report no confirmed improvement.
Use pinned Rune frozen/replay/fresh/negative gates for the actual core, and keep
Candle, native Node, browser and package-consumer evidence distinct.
