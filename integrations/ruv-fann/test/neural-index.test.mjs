/** Boundary ownership fixtures and actual FANN + actual RuVector composition. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createNeuralIndex } from '../src/index.mjs';
import { createRuvectorStore } from '../src/ruvector.mjs';

const wasmBytes = new Uint8Array(await readFile(new URL('../node_modules/@ruvector/wasm/pkg/ruvector_wasm_bg.wasm', import.meta.url)));
const storeFor = options => createRuvectorStore({ wasmBytes, dimensions: 2, useHnsw: false, ...options });

function fixture(transform = input => [input[0] * 2 + input[1], input[0] + input[1] * 3]) {
  const state = { batches: 0, freed: 0, inference: 0, preparedFreed: 0 };
  const prepared = { input_size: 2, output_size: 2, estimated_bytes: 128,
    run(input) { return new Float64Array(transform(input)); }, free() { state.preparedFreed++; },
    create_batch(rows) {
      state.batches++;
      return { input_size: 2, output_size: 2, max_rows: rows, estimated_bytes: 512,
        run_into(input, output, active) {
          state.inference++;
          for (let row = 0; row < active; row++) output.set(transform(input.subarray(row * 2, row * 2 + 2)), row * 2);
        }, free() { state.freed++; } };
    } };
  return { prepared, state };
}

test('neural index owns encoder/store, preserves metadata and leaves prepared caller-owned', async () => {
  const { prepared, state } = fixture(), store = await storeFor();
  const index = createNeuralIndex({ prepared, store, modelId: 'caller-label-v1', batchRows: 3 });
  assert.equal(index.modelId, 'caller-label-v1'); assert.equal(index.indexType, 'flat');
  index.add({ id: 'east', input: new Float64Array([1, 0]), metadata: { title: 'east' } });
  index.add({ id: 'north', input: new Float64Array([0, 1]), metadata: { title: 'north' } });
  const hits = index.search(new Float64Array([1, 0]), { k: 2 });
  assert.equal(hits[0].id, 'east'); assert.equal(hits[0].metadata.title, 'east');
  assert.equal(hits[0].score, hits[0].similarity); assert.ok(Number.isFinite(hits[0].distance));
  assert.equal(index.size, 2); assert.equal(index.delete('east'), true); assert.equal(index.delete('east'), false);
  index.close(); index.close();
  assert.equal(state.freed, 1); assert.equal(state.preparedFreed, 0);
  assert.deepEqual([...prepared.run(new Float64Array([1, 0]))], [2, 1]);
  assert.throws(() => store.size, /closed/); assert.throws(() => index.size, /closed/);
  assert.throws(() => index.add({ id: 'closed', input: new Float64Array([1, 1]) }), /closed/);
  assert.throws(() => index.search(new Float64Array([1, 1])), /closed/);
});

test('nonempty or mismatched stores are rejected before encoder ownership transfers', async () => {
  const { prepared, state } = fixture();
  const occupied = await storeFor(), wrong = await storeFor({ dimensions: 3 });
  try {
    occupied.insert({ id: 'preexisting', vector: new Float32Array([1, 0]) });
    for (const store of [occupied, wrong])
      assert.throws(() => createNeuralIndex({ prepared, store, modelId: 'model' }), /new empty compatible/);
    assert.equal(state.batches, 0); assert.equal(state.freed, 0); assert.equal(occupied.size, 1);
  } finally { occupied.close(); wrong.close(); }
});

test('duplicate IDs, invalid metadata and configured capacity fail before inference', async () => {
  const { prepared, state } = fixture(), store = await storeFor();
  const index = createNeuralIndex({ prepared, store, modelId: 'model', maxEntries: 1 });
  try {
    assert.throws(() => index.add({ id: 'bad', input: new Float64Array([1, 1]), metadata: { value: Infinity } }));
    assert.equal(state.inference, 0);
    index.add({ id: 'one', input: new Float64Array([1, 0]) });
    const calls = state.inference;
    assert.throws(() => index.add({ id: 'one', input: new Float64Array([0, 1]) }), /duplicate/);
    assert.throws(() => index.add({ id: 'two', input: new Float64Array([0, 1]) }), /budget/);
    assert.equal(state.inference, calls); assert.equal(index.size, 1);
  } finally { index.close(); }
});

test('exactly one finite input row and bounded search parameters are required', async () => {
  const { prepared, state } = fixture(), store = await storeFor();
  const index = createNeuralIndex({ prepared, store, modelId: 'model' });
  try {
    for (const input of [new Float64Array(), new Float64Array(3), new Float64Array(4),
      new Float64Array([NaN, 1]), new Float32Array(2)]) {
      assert.throws(() => index.add({ id: 'bad', input })); assert.throws(() => index.search(input));
    }
    assert.equal(state.inference, 0);
    assert.throws(() => index.search(new Float64Array([1, 0]), { k: 1001 }));
    assert.throws(() => index.search(new Float64Array([1, 0]), { filter: true }));
  } finally { index.close(); }
});

test('reentrant input getters cannot exceed the configured neural-index capacity', async () => {
  const { prepared } = fixture(), store = await storeFor({ maxEntries: 2 });
  const index = createNeuralIndex({ prepared, store, modelId: 'reentrant-fixture', maxEntries: 1 });
  try {
    assert.throws(() => index.add({ id: 'outer', get input() {
      index.add({ id: 'inner', input: new Float64Array([1, 0]) });
      return new Float64Array([0, 1]);
    } }), /budget/);
    assert.equal(index.size, 1);
    assert.equal(index.search(new Float64Array([1, 0]))[0].id, 'inner');
  } finally { index.close(); }
});

test('finite doubles that overflow f32 or underflow cosine vectors never enter the store', async () => {
  for (const transform of [() => [Number.MAX_VALUE, 1], () => [Number.MIN_VALUE, 0], () => [0, 0]]) {
    const { prepared } = fixture(transform), store = await storeFor();
    const index = createNeuralIndex({ prepared, store, modelId: 'range-fixture' });
    try {
      assert.throws(() => index.add({ id: 'bad', input: new Float64Array([1, 0]) }), /f32|norm/);
      assert.equal(index.size, 0);
    } finally { index.close(); }
  }
});

test('zero embeddings remain valid for euclidean search', async () => {
  const { prepared } = fixture(() => [0, 0]), store = await storeFor({ metric: 'euclidean' });
  const index = createNeuralIndex({ prepared, store, modelId: 'zero-euclidean' });
  try {
    index.add({ id: 'zero', input: new Float64Array([1, 2]) });
    assert.equal(index.search(new Float64Array([3, 4]))[0].distance, 0);
  } finally { index.close(); }
});

test('invalid model labels/options do not take ownership of caller resources', async () => {
  const { prepared, state } = fixture(), store = await storeFor();
  try {
    for (const options of [{ modelId: '' }, { modelId: '\ud800' }, { modelId: 'ok', extra: 1 },
      { modelId: 'ok', batchRows: 0 }, { modelId: 'ok', maxEntries: 10001 }])
      assert.throws(() => createNeuralIndex({ prepared, store, ...options }));
    assert.equal(store.size, 0); assert.equal(state.batches, 0); assert.equal(state.preparedFreed, 0);
  } finally { store.close(); }
});

test('actual FANN WASM embedding composes with real HNSW and expected numerical nearest input', async () => {
  const module = await import('../../../ruv-swarm/npm/wasm/ruv_swarm_wasm.js');
  module.initSync({ module: await readFile(new URL('../../../ruv-swarm/npm/wasm/ruv_swarm_wasm_bg.wasm', import.meta.url)) });
  const network = new module.WasmNeuralNetwork(new Uint32Array([2, 2]), module.ActivationFunction.Linear);
  network.set_weights(new Float64Array([2, 1, 1, 3]));
  const prepared = network.prepare(); network.free();
  const store = await storeFor({ useHnsw: true, metric: 'euclidean' });
  const index = createNeuralIndex({ prepared, store, modelId: 'actual-wasm-fixed-weights' });
  try {
    index.add({ id: 'a', input: new Float64Array([1, 0]), metadata: { input: [1, 0] } });
    index.add({ id: 'b', input: new Float64Array([0, 1]), metadata: { input: [0, 1] } });
    index.add({ id: 'c', input: new Float64Array([-1, -1]), metadata: { input: [-1, -1] } });
    const hits = index.search(new Float64Array([1, 0]), { k: 3 });
    assert.equal(index.indexType, 'hnsw'); assert.equal(hits[0].id, 'a'); assert.equal(hits[0].distance, 0);
    assert.ok(hits[1].distance > 0); assert.deepEqual(hits[0].metadata.input, [1, 0]);
    index.close(); assert.equal(prepared.run(new Float64Array([1, 0])).length, 2);
  } finally { index.close(); prepared.free(); }
});
