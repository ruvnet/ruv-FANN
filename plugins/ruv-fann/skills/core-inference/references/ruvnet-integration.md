# Connect ruv-FANN models to RuVector and MetaHarness

Use the selected checkout's `integrations/ruv-fann` package. It contains the
optional `@ruv-fann/integrations` 0.1.0 candidate, pinning `@ruvector/wasm` 0.2.1
and `@metaharness/darwin` 0.13.0 in a separate lock. The Rust core and other
package locks stay separate.

## Establish real inference

Accept a real `WasmPreparedNetwork` from `network.prepare()`,
`WasmPreparedNetwork.from_network_json(json)`, or ruv-swarm's
`runtime.createPreparedNetwork(json)`. Native `Network<f64>` JSON retains sparse
connection order and activations. Inputs are numerical Float64 features. The
package supplies no pretrained text embeddings or training procedure.

From the checkout root:

```sh
npm ci --prefix integrations/ruv-fann --ignore-scripts --omit=optional
npm run build --prefix integrations/ruv-fann
npm test --prefix integrations/ruv-fann
npm run test:e2e --prefix integrations/ruv-fann
npm run example --prefix integrations/ruv-fann
```

## Apply the adapters correctly

`createEncoder` owns a reusable batch, not the prepared network. Its exact
Float64 outputs are converted to Float32 only when inserted or queried in
RuVector. Reject overflow, invalid norms, dimensions and excessive metadata.
Cosine zero vectors are invalid. Preserve score direction: similarity is higher
is better, raw distance is lower is better. HNSW is approximate; compare it with
flat search and inspect the actual database `indexType`.

`createNeuralIndex` takes ownership of an empty matching store. Closing the index
frees that store and its encoder but leaves the caller's prepared snapshot alive.
Change model weights only by creating a new snapshot and rebuilding the index.
The supplied `modelId` is a label, not independent proof of weights or lineage.

The browser exports require native ESM/import maps or a compatible bundler. WASM
bytes are supplied explicitly. The Node-only `./node` helper reads the pinned
installed asset locally. The `./metaharness` tuner imports real `evolveNumeric`,
uses bounded dispatches and fresh confirmation, and never auto-applies a setting.
Its deadline is cooperative in a trusted application; it is not an OS sandbox.

## Require end-to-end evidence

Test native dense and sparse model exports on both actual FANN binaries, exact
batched output equivalence, actual RuVector flat/HNSW retrieval, metadata,
deletion, ownership, real Darwin tuning, TypeScript declarations and an extracted
npm package consumer. Browser checks cover inference and retrieval separately
from the Node-only tuner. Preserve raw samples and failed/inconclusive results.
Never require a shared-runner speedup or equate vector similarity with semantic
accuracy. Registry publication and fresh Claude host certification are separate.
