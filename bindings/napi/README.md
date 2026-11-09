# Native ruv-FANN for Node

This package runs the real Rust `PreparedNetwork<f64>` through Node-API8. It
provides the same prepared/batch method shape used by the optional neural-search
and MetaHarness integrations. It contains no WASM fallback or automatic download.

Build explicitly from this repository after fetching locked Cargo dependencies:

```sh
cargo +1.88.0 fetch --locked --manifest-path bindings/napi/Cargo.toml
node bindings/napi/scripts/build.mjs --target-dir /absolute/shared/target
node --test bindings/napi/test/*.test.mjs
node bindings/napi/test/packed-consumer.mjs --output /absolute/new-consumer-evidence
```

The helper requires preinstalled Rust1.88.0 and builds offline. Its current
platform is recorded in `build-info.json`; the packed addon is specific to that
platform and architecture. CJS and ESM loaders verify the local binary hash.
Source/lock/compiler provenance and the actual addon SHA256 are recorded without
changing the root lockfile. This source build helper supports Linux and macOS;
other hosts need an explicit native build setup rather than a downloaded binary.

```js
import {NativePreparedNetwork} from '@ruv-fann/napi';

const denseJson = JSON.stringify({
  schema: 'ruv-fann.dense-layers/1',
  layers: [{inputs: 2, outputs: 1, weights: [0.5, -0.25], bias: [0.1],
    activation: 'Linear', steepness: 1}],
});
const model = NativePreparedNetwork.from_dense_layers_json(denseJson);
const output = model.run(new Float64Array([1, 2]));
const batch = model.create_batch(128);
model.free(); // The batch owns a shared immutable snapshot and remains usable.
batch.free();
```

`from_network_json` accepts a native serialized `Network<f64>`, preserving sparse
connections, biases and activation behavior. `from_dense_layers_json` accepts the
versioned out-by-in dense format above. The model imports own their parameters;
source JSON strings may be discarded after construction.

Train fixed dense models with actual reverse-mode MSE gradients and SGD or AdamW:

```js
import {NativeTrainableNetwork} from '@ruv-fann/napi';

const training = NativeTrainableNetwork.from_dense_layers_json(denseJson, 32);
training.configure_adamw(0.005, 0.9, 0.99, 1e-8, 0.01);
const loss = training.mse_backward(inputRows, targetRows, rows);
const gradients = training.parameter_gradients(); // Owned JS Float64Array.
training.adamw_step(); // Recompute backward before the next update.
const snapshot = training.prepare(); // Independent immutable inference snapshot.
const checkpoint = training.to_dense_layers_json(); // Parameters, without optimizer state.
training.free();
snapshot.free();
```

Training supports Linear, Sigmoid, SigmoidSymmetric, Tanh and ReLU.
Gradients use layer order: row-major weights followed by optional biases. MSE
is the mean across active rows and outputs. Failed parameter updates preserve
the model; successful updates consume the current gradients. `sgd_step(rate)`
provides an alternative to AdamW. Training capacity is1..4096 rows, with the
same64MiB combined model/workspace/optimizer/staging limit and100million
conservative forward/reverse or optimizer work units. Optimizer reconfiguration
accounts for both old and new storage during replacement. Dense checkpoints
round-trip f64 values; they contain model data, rather than executable code.

Set `RUV_FANN_CANDLE_FIXTURE` to an actual Candle CPU oracle fixture when running
source tests and the packed consumer. If supplied, its losses, parameter/input
gradients and three sequential SGD/AdamW updates must match; it is never skipped.
The source tests also check finite differences, actual loss reduction, existing
WASM/SDK/RuVector compatibility and direct-addon malformed inputs. Run
`test/tuner-compat.mjs --output /absolute/new-proof.json` separately during an
idle measurement window for actual Darwin/native compatibility evidence.

The native exports validate primitive JSON strings before UTF-8 allocation
(4MiB limit), genuine nonshared/nondetached/nonresizable Float64Arrays, exact
dimensions, finite inputs, 64MiB combined model/workspace/staging budgets and
batch rows1..4096. Conservative batch work is limited to100million units.
The addon captures the standard resizable-buffer getter at module registration;
ordinary host intrinsics are required at that point. Instance tag/property
spoofing cannot replace the checks. Validation can reenter JS safely before
native inference state is borrowed.

`run_into` snapshots inputs into private Rust staging, computes there, then
commits outputs. Input/output aliases are supported and failures preserve caller
output. `free` is idempotent; other operations reject freed handles. Native
`run_into` reuses its Rust buffers, while `run` allocates an owned JS result.
Caller buffers never become Rust slices and no asynchronous JS references are
borrowed. Numerical checks, including FFI abuse tests, must pass on the actual
`.node` export; the JS loader is not the security boundary.
