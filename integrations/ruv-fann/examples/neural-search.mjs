/** Runnable checkout demo: a native ruv-FANN model, RuVector and MetaHarness. */
import { readFile } from 'node:fs/promises';
import { createNeuralIndex } from '../src/index.mjs';
import { createRuvectorStore } from '../src/ruvector.mjs';
import { ruvectorWasmBytes } from '../src/node.mjs';
import { tuneBatchRows } from '../src/metaharness.mjs';
import init, { WasmPreparedNetwork } from '../../../ruv-swarm/npm/wasm/ruv_swarm_wasm.js';

const bytes = await readFile(new URL('../../../ruv-swarm/npm/wasm/ruv_swarm_simd.wasm', import.meta.url));
await init({ module_or_path: bytes });
const fixture = JSON.parse(await readFile(new URL('../../../ruv-swarm/crates/ruv-swarm-wasm/tests/fixtures/prepared-dense.json', import.meta.url)));
const prepared = WasmPreparedNetwork.from_network_json(JSON.stringify(fixture.network));
const store = await createRuvectorStore({ wasmBytes: await ruvectorWasmBytes(),
  dimensions: prepared.output_size, metric: 'euclidean', useHnsw: true });
const index = createNeuralIndex({ prepared, store, modelId: 'native-dense-demo', batchRows: 128 });
try {
  const first = Float64Array.from(fixture.input);
  const second = Float64Array.from(first, x => x + 0.25);
  index.add({ id: 'first', input: first, metadata: { label: 'first numerical sample' } });
  index.add({ id: 'second', input: second, metadata: { label: 'second numerical sample' } });
  const nearest = index.search(first, { k: 2 });
  const inputs = Float64Array.from({ length: 131 * prepared.input_size }, (_, i) =>
    fixture.input[i % prepared.input_size] + Math.floor(i / prepared.input_size) / 100);
  const tuning = await tuneBatchRows({ prepared, inputs, modelId: index.modelId,
    baselineBatchRows: 128, generations: 1, children: 1, samples: 5 });
  console.log(JSON.stringify({ kind: 'actual-ruv-fann-ruvector-metaharness', indexType: index.indexType,
    dimensions: { input: prepared.input_size, neuralOutput: prepared.output_size }, nearest,
    tuning: { status: tuning.status, confirmation: tuning.confirmation.status, recommendation: tuning.recommendation },
    scope: 'Numerical model-output similarity, not pretrained text semantics. Tuning is a recommendation only.' }, null, 2));
} finally { index.close(); prepared.free(); }
