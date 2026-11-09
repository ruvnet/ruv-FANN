# ruv-FANN integrations

Use a ruv-FANN network to produce vectors, search them with RuVector, and measure
batch configurations with MetaHarness. This optional package connects the real
implementations; the Rust core has no new dependency on these packages.

| Package | Role |
| --- | --- |
| ruv-FANN prepared WASM | Runs fixed neural weights, including native `Network<f64>` JSON exports |
| `@ruvector/wasm` 0.2.1 | Stores Float32 neural outputs and performs exact flat or approximate HNSW search |
| `@metaharness/darwin` 0.13.0 | Searches batch capacity and returns a measured recommendation |
| ruv-swarm | Already exposes the prepared WASM network; applications can pass its handle directly |
| Rune and CI | Remain the separate build-verification and acceptance checks |

## Run the checkout example

Requires Node 22.13+. From the repository root:

```sh
npm ci --prefix integrations/ruv-fann --ignore-scripts --omit=optional
npm run build --prefix integrations/ruv-fann
npm test --prefix integrations/ruv-fann
npm run test:e2e --prefix integrations/ruv-fann
npm run example --prefix integrations/ruv-fann
```

The example imports the supplied native model fixture into actual SIMD WASM,
indexes its neural outputs in actual RuVector HNSW, queries the index, and calls
the actual Darwin numeric engine. It downloads no model and needs no provider key.
The package is prepared in this checkout; registry publication is a separate step.

## Connect your own network

Pass a real `WasmPreparedNetwork`, obtained from `network.prepare()`,
`WasmPreparedNetwork.from_network_json(json)`, or ruv-swarm's
`runtime.createPreparedNetwork(json)`. These objects retain fixed weights.

With the Rust core's default Serde feature, export the native network using
`serde_json::to_string(&network)?`, then import that JSON into the prepared WASM
runtime. The tests exercise existing dense and sparse native exports.

```js
import { createNeuralIndex } from '@ruv-fann/integrations';
import { createRuvectorStore } from '@ruv-fann/integrations/ruvector';
import { ruvectorWasmBytes } from '@ruv-fann/integrations/node';

// prepared is your real ruv-FANN WasmPreparedNetwork.
const store = await createRuvectorStore({
  wasmBytes: await ruvectorWasmBytes(),
  dimensions: prepared.output_size,
  metric: 'euclidean',
  useHnsw: true,
});
const index = createNeuralIndex({ prepared, store, modelId: 'my-fixed-model' });
try {
  index.add({ id: 'sample', input: sampleFeatures, metadata: { category: 'example' } });
  const matches = index.search(queryFeatures, { k: 5 });
  console.log(matches); // id, distance, similarity/score, metadata
} finally {
  index.close(); // frees the owned batch and store
  prepared.free(); // the application owns the prepared model
}
```

Inputs are finite, row-major Float64 arrays. The network computes Float64 outputs;
RuVector stores their Float32 conversions. Conversion overflow and unsafe vector
norms are rejected. `score`/`similarity` is higher-is-better; raw `distance` is
lower-is-better. Cosine similarity requires a nonzero vector. A neural output is
only useful for semantic search if the model has been trained for that purpose;
this package supplies no pretrained text embedding model.

`createEncoder(prepared, { batchRows: 128 })` also encodes flattened multiple-row
inputs without an index. Its `close()` frees its own batch, retaining `prepared`.
Changing model weights requires a new prepared snapshot and rebuilding the index.
`modelId` is an application label, not proof of model identity.

## Tune inference batches

```js
import { tuneBatchRows } from '@ruv-fann/integrations/metaharness';
const receipt = await tuneBatchRows({
  prepared, inputs: representativeRows, modelId: 'my-fixed-model',
  baselineBatchRows: 128, generations: 2, children: 2, samples: 20,
});
console.log(receipt.recommendation, receipt.confirmation);
```

The Node-only tuner changes batch capacity from 1 to 128. It processes identical
rows and requires exact finite Float64 reference outputs. Search observations
and fresh paired confirmation samples remain separate. Only a changed candidate
with a 95% timing-block bootstrap lower speed ratio above 1.05 can replace the
requested baseline in the recommendation. The function never applies that setting.
This measures inference batching, not ANN search throughput or retrieval quality.

The budget permits at most 12 dispatches with concurrency one; identical capacities
reuse an observation. The deadline is cooperative inside a trusted application,
not a process sandbox. No shell evaluator, model call or source editing is exposed.
Use the repository's separate private MetaHarness CLI when a parent process wall
deadline and source-aware experiment receipts are needed.

## Browser and ownership boundaries

The main and `./ruvector` exports support modern browser ESM. Supply the pinned
RuVector WASM bytes explicitly, and use an import map or a bundler that supports
`import.meta.resolve` for `@ruvector/wasm`. The `./node` asset helper
and `./metaharness` tuner are Node-only and are not imported by the main export.
The active database reports `indexType`; requesting HNSW cannot silently certify
that it ran. HNSW is approximate. Use `useHnsw: false` for an exact reference.

The index takes ownership of an empty matching store and its own encoder. It never
frees the supplied prepared network. Duplicate IDs, malformed dimensions, invalid
metadata and excessive resource requests are rejected. This release does not
provide persistent index/model synchronization or cross-owner memory sharing.
