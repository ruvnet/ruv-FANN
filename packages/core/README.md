# ruv-fann

Typed dense neural inference and training over real ruv-FANN WebAssembly. The only
npm dependency is exactly `ruv-fann-wasm@0.1.0`. ESM for Node >=20.9 and browsers.

```js
import { loadRuntime } from 'ruv-fann';
const runtime = await loadRuntime({ simd: 'auto' });
const model = {
  schema: 'ruv-fann.dense-layers/1',
  layers: [{ inputs: 1, outputs: 1, weights: [2], bias: [1],
    activation: 'Linear', steepness: 1 }]
};
const prepared = runtime.prepare(model, { maxRows: 128 });
try {
  console.log(prepared.predict(Float64Array.of(1, 2, 3))); // [3, 5, 7]
  const output = new Float64Array(3);
  prepared.predictInto(Float64Array.of(1, 2, 3), output);
} finally { prepared.dispose(); }

const learned = runtime.train(model, {
  inputs: Float64Array.of(1, 2), targets: Float64Array.of(2, 4),
  rows: 2, epochs: 100, learningRate: 0.05, optimizer: 'adamw'
});
console.log(learned.initialLoss, learned.finalLoss, learned.model);
```

`loadRuntime({simd?: boolean | 'auto'})` initializes a cached `FannRuntime` with
`backend: 'wasm' | 'wasm-simd'`. Auto uses a real SIMD feature probe; explicit
SIMD rejects when unavailable. Node reads installed assets without network access.
For browser deployment, preserve the raw package's `src`/`dist` asset URLs and
see the `ruv-fann-wasm` README. No native addon or CommonJS entry is provided.

`runtime.prepare(modelOrJSON, {maxRows?: number})` owns an immutable packed
snapshot and one reusable batch. It exposes `inputSize`, `outputSize`, `maxRows`,
`estimatedBytes`, `predict(inputs, rows?)`, `predictInto(inputs, outputs, rows?)`,
and `dispose()`. Inputs are contiguous row-major `Float64Array`s. Omitted rows
are inferred from exact input length; explicit rows must match the whole input.
Outputs must have exactly `rows * outputSize` elements. Zero-row inference is
supported. Returned arrays are caller-owned copies; input/output aliasing is safe.
Changes to the source model do not change a prepared snapshot. Disposal is
idempotent, and subsequent predictions throw. Validation and numerical errors
preserve the whole supplied output buffer. A model does not expose raw WASM views.

The versioned model uses flat `[outputs, inputs]` weights, optional per-output
`bias` (array or null), finite `steepness`, and case-sensitive native activation
names such as `Linear`, `Sigmoid`, `Tanh` and `ReLU` (see `DenseActivation` types
for all inference names). Adjacent layer dimensions must match. Unknown model
fields, shape errors, nonfinite values and unsupported schema versions reject.
Existing activation numerical conventions are preserved, including unscaled ReLU.

`runtime.train(modelOrJSON, options)` performs synchronous full-batch MSE training
in WASM, returning `{model, initialLoss, finalLoss, epochs, parameterCount}`. Every
epoch performs exactly one fresh optimizer update; final loss is evaluated after
the last update. SGD is the default. AdamW uses beta1=0.9, beta2=0.999,
epsilon=1e-8, weightDecay=0; state is fresh per call and is not serialized.
Training supports Linear, Sigmoid, SigmoidSymmetric, Tanh and ReLU. Caller model,
inputs and targets are never changed. Training allocations are freed even when
an update fails. Large runs should be scheduled in a worker to keep the UI free.

Bounds are exported as `LIMITS`: up to 4096 rows (prepared default 128), 64 layers,
65536 elements per dimension, 1 million parameters, 4 MiB UTF-8 model JSON,
64 MiB native storage per prepared batch/training object, and 1–1000 epochs.
Total synchronous training is also capped at 100 million conservative work units:
`(epochs + 1) * rows * (32 * parameters + 8 * (inputs + outputs) + 1)
+ epochs * 64 * parameters`. This uses the existing native training/update cost
policy across the whole call. Peak process memory also includes model conversion,
caller arrays, temporary JS copies, allocator slack and shared backend memory;
`estimatedBytes` is a conservative native object estimate, not process RSS.

Only finite primitive numbers and JS-owned, nonshared, nondetached Float64Arrays
are accepted at array boundaries. Exact lengths and row limits are enforced before
inference/training. WASM memory views, proxies and spoofed arrays reject.

Repository validation: build raw assets with `node scripts/npm-modules-build.mjs`,
run both package tests, then `node scripts/npm-modules-pack-test.mjs` for isolated
installed-tarball validation. `npm run benchmark` reports batch-vs-row timing with
matching results; it does not enforce noisy timing thresholds.

MIT licensed. Source: https://github.com/ruvnet/ruv-FANN/tree/main/packages/core
