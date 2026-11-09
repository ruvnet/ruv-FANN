/** Real native backend through the existing public FANN/RuVector integration API. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
import { createEncoder, createNeuralIndex } from '../../../integrations/ruv-fann/src/index.mjs';
import { createRuvectorStore } from '../../../integrations/ruv-fann/src/ruvector.mjs';
import { dense, network } from './fixtures.mjs';
const { NativePreparedNetwork } = createRequire(import.meta.url)('../ruv_fann.node');

test('existing encoder and actual HNSW search work with the native prepared interface', async () => {
  const prepared = NativePreparedNetwork.from_dense_layers_json(dense());
  const wasmBytes = await readFile(new URL('../../../integrations/ruv-fann/node_modules/@ruvector/wasm/pkg/ruvector_wasm_bg.wasm', import.meta.url));
  const store = await createRuvectorStore({ wasmBytes, dimensions: 2, metric: 'euclidean', useHnsw: true });
  const encoder = createEncoder(prepared, { batchRows: 3, maxRows: 7 });
  try {
    const input = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]), expected = [];
    for (let first = 0; first < input.length; first += 2) expected.push(...prepared.run(input.subarray(first, first + 2)));
    assert.deepEqual([...encoder.encode(input)], expected);
    const index = createNeuralIndex({ prepared, store, modelId: 'native-actual-source', batchRows: 3 });
    try {
      index.add({ id: 'one', input: new Float64Array([1, 0]), metadata: { backend: 'native' } });
      index.add({ id: 'two', input: new Float64Array([0, 1]) });
      const result = index.search(new Float64Array([1, 0]));
      assert.equal(index.indexType, 'hnsw'); assert.equal(result[0].id, 'one');
      assert.equal(result[0].distance, 0); assert.equal(result[0].metadata.backend, 'native');
    } finally { index.close(); }
    assert.equal(prepared.run(new Float64Array([1, 2])).length, 2);
  } finally { encoder.close(); store.close(); prepared.free(); }
});

test('native and actual core/SIMD WASM imports agree on the same native sparse JSON', async () => {
  const json = network({ sparse: true }), native = NativePreparedNetwork.from_network_json(json);
  try {
    for (const variant of ['core', 'simd']) {
      const glue = new URL('../../../ruv-swarm/npm/wasm/ruv_swarm_wasm.js', import.meta.url);
      glue.searchParams.set('native-parity', variant);
      const module = await import(glue.href);
      module.initSync({ module: await readFile(new URL('../../../ruv-swarm/npm/wasm/' + (variant === 'simd' ? 'ruv_swarm_simd.wasm' : 'ruv_swarm_wasm_bg.wasm'), import.meta.url)) });
      const wasm = module.WasmPreparedNetwork.from_network_json(json);
      try {
        const input = new Float64Array([0.2, -0.7]);
        assert.deepEqual([...native.run(input)], [...wasm.run(input)]);
      } finally { wasm.free(); }
    }
  } finally { native.free(); }
});
