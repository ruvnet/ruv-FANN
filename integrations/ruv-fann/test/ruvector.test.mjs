/** Tests use the actual pinned published RuVector WASM binary, never a JS search fallback. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRuvectorStore, VECTOR_LIMITS } from '../src/ruvector.mjs';

const bytes = new Uint8Array(await readFile(new URL('../node_modules/@ruvector/wasm/pkg/ruvector_wasm_bg.wasm', import.meta.url)));
const factory = options => createRuvectorStore({ wasmBytes: bytes, dimensions: 2, ...options });

for (const useHnsw of [false, true]) test(`real ${useHnsw ? 'HNSW' : 'flat'} index returns ordered distance/similarity and deletion`, async () => {
  const store = await factory({ useHnsw });
  try {
    assert.equal(store.indexType, useHnsw ? 'hnsw' : 'flat'); assert.equal(store.size, 0);
    const metadata = { title: 'east', nested: { value: 1 } };
    store.insert({ id: 'east', vector: new Float32Array([1, 0]), metadata });
    store.insert({ id: 'north', vector: new Float32Array([0, 1]) });
    store.insert({ id: 'west', vector: new Float32Array([-1, 0]) });
    metadata.nested.value = 99;
    const hits = store.search({ vector: new Float32Array([1, 0]), k: 3 });
    assert.deepEqual(hits.map(item => item.id), ['east', 'north', 'west']);
    assert.deepEqual(hits.map(item => item.distance), [0, 1, 2]);
    assert.deepEqual(hits.map(item => item.score), [1, 0, -1]);
    assert.equal(hits[0].similarity, hits[0].score); assert.equal(hits[0].metadata.nested.value, 1);
    hits[0].metadata.nested.value = 777;
    assert.equal(store.search({ vector: new Float32Array([1, 0]), k: 1 })[0].metadata.nested.value, 1);
    assert.equal(store.delete('east'), true); assert.equal(store.delete('east'), false); assert.equal(store.size, 2);
    assert.ok(store.search({ vector: new Float32Array([1, 0]), k: 3 }).every(item => item.id !== 'east'));
  } finally { store.close(); }
  store.close(); assert.throws(() => store.search({ vector: new Float32Array([1, 0]), k: 1 }), /closed/);
  assert.throws(() => store.size, /closed/);
});

test('all advertised metrics preserve finite raw distance and higher-is-better score', async () => {
  for (const metric of ['cosine', 'dotproduct', 'euclidean', 'manhattan']) {
    const store = await factory({ metric, useHnsw: false });
    try {
      store.insert({ id: 'same', vector: new Float32Array([1, 0]) });
      store.insert({ id: 'other', vector: new Float32Array([0, 1]) });
      const hits = store.search({ vector: new Float32Array([1, 0]), k: 2 });
      assert.equal(hits[0].id, 'same'); assert.ok(hits[0].score > hits[1].score);
      for (const hit of hits) {
        assert.ok(Number.isFinite(hit.distance));
        const expected = metric === 'cosine' ? 1 - hit.distance : metric === 'dotproduct' ? -hit.distance : 1 / (1 + hit.distance);
        assert.equal(hit.score, expected);
      }
    } finally { store.close(); }
  }
});

test('explicit bytes initialize without fetch and are isolated from the public module singleton', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = () => { throw new Error('unexpected network initialization'); };
  const publicModule = await import('@ruvector/wasm');
  publicModule.initSync({ module: bytes });
  const originalSearch = publicModule.VectorDB.prototype.search;
  publicModule.VectorDB.prototype.search = () => { throw new Error('shared public singleton'); };
  let store;
  try {
    store = await factory({ useHnsw: true }); store.insert({ id: 'owned', vector: new Float32Array([1, 1]) });
    assert.equal(store.search({ vector: new Float32Array([1, 1]), k: 1 })[0].id, 'owned');
  } finally {
    store?.close(); globalThis.fetch = originalFetch; publicModule.VectorDB.prototype.search = originalSearch;
  }
});

test('concurrent factories share explicit initialization and different byte identities fail closed', async () => {
  const stores = await Promise.all([factory({ useHnsw: false }), factory({ useHnsw: true })]);
  try { assert.deepEqual(stores.map(item => item.indexType), ['flat', 'hnsw']); }
  finally { for (const store of stores) store.close(); }
  await assert.rejects(factory({ wasmBytes: new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]) }), /different explicit bytes/);
});

test('input type, dimension, duplicate and finite f32 norm checks precede database mutation', async () => {
  const store = await factory({ maxEntries: 1 });
  try {
    for (const vector of [new Float32Array(3), new Float64Array(2), new Float32Array([0, 0]),
      new Float32Array([Infinity, 1]), new Float32Array([1e30, 1]), new Float32Array([1e-30, 0]),
      new Float32Array(new SharedArrayBuffer(8)), Object.create(Float32Array.prototype)])
      assert.throws(() => store.insert({ id: 'invalid', vector }));
    const detached = new Float32Array(2); structuredClone(detached.buffer, { transfer: [detached.buffer] });
    assert.throws(() => store.insert({ id: 'detached', vector: detached }));
    assert.equal(store.size, 0);
    store.insert({ id: 'one', vector: new Float32Array([1, 0]) });
    assert.throws(() => store.insert({ id: 'one', vector: new Float32Array([0, 1]) }), /duplicate/);
    assert.throws(() => store.insert({ id: 'two', vector: new Float32Array([0, 1]) }), /budget/);
    assert.throws(() => store.search({ vector: new Float32Array([1, 0]), k: 1001 }), /k/);
    assert.deepEqual(store.search({ vector: new Float32Array([1, 0]), k: 0 }), []);
  } finally { store.close(); }
});

test('metadata is bounded acyclic JSON and never executes accessors or toJSON callbacks', async () => {
  const store = await factory(); let getters = 0;
  const accessor = { get secret() { getters++; return 1; } }, cycle = {}; cycle.self = cycle;
  const toJSON = { toJSON() { throw new Error('metadata callback executed'); } };
  let deep = {}; for (let index = 0; index < 10; index++) deep = { deep };
  try {
    for (const metadata of [accessor, cycle, toJSON, { value: NaN }, { value: 'x'.repeat(16385) }, deep])
      assert.throws(() => store.insert({ id: 'invalid', vector: new Float32Array([1, 0]), metadata }));
    assert.equal(getters, 0); assert.equal(store.size, 0);
    assert.throws(() => store.insert({ id: '\ud800', vector: new Float32Array([1, 0]) }), /Unicode/);
    assert.throws(() => store.insert({ id: 'x'.repeat(257), vector: new Float32Array([1, 0]) }));
  } finally { store.close(); }
});

test('changing vector getters are captured once before validation and copying', async () => {
  const store = await factory(); let insertReads = 0, searchReads = 0;
  try {
    store.insert({ id: 'one', get vector() { insertReads++; return insertReads === 1 ? new Float32Array([1, 0]) : new Float32Array(9999); } });
    const hits = store.search({ k: 1, get vector() { searchReads++; return searchReads === 1 ? new Float32Array([1, 0]) : new Float32Array([NaN]); } });
    assert.equal(insertReads, 1); assert.equal(searchReads, 1); assert.equal(hits[0].id, 'one');
  } finally { store.close(); }
});

test('query result logical byte budget is rejected before native search allocations', async () => {
  const store = await factory({ dimensions: VECTOR_LIMITS.maxDimensions, metric: 'euclidean', useHnsw: false });
  try { assert.throws(() => store.search({ vector: new Float32Array(65536), k: 257 }), /byte budget/); }
  finally { store.close(); }
});

test('invalid factory options/byte views are rejected without network fallback', async () => {
  for (const options of [{ dimensions: 0 }, { dimensions: 65537 }, { metric: 'unknown' }, { useHnsw: 'yes' },
    { maxEntries: 10001 }, { extra: 1 }, { wasmBytes: new Uint8Array() },
    { wasmBytes: new Uint8Array(new SharedArrayBuffer(8)) }, { wasmBytes: new Uint8Array(16 * 1024 * 1024 + 1) }])
    await assert.rejects(factory(options));
});
