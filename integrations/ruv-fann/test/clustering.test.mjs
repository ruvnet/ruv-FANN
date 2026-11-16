import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { init } from '@ruv-fann/integrations/clustering';
import { trainProductQuantizer, encodeProductQuantizer, decodeProductQuantizer } from '@ruv-fann/integrations/clustering/quantization';
import { partition, postingList } from '@ruv-fann/integrations/clustering/ivf';
import { createClusteringEvaluator } from '@ruv-fann/integrations/clustering/metaharness';
import { createClusteringWorker } from '@ruv-fann/integrations/clustering/cloudflare';

await init();

test('optional integrations expose shared Rust/WASM clustering without a prepared network', () => {
  const data = new Float32Array([1, 3, 5, 7]);
  const model = trainProductQuantizer(data, 2, 2, { clusters: 1 });
  assert.deepEqual(Array.from(model.centroids), [3, 5]);
  assert.deepEqual(Array.from(decodeProductQuantizer(model, encodeProductQuantizer(model, data))), [3, 5, 3, 5]);
  const result = partition(data, 2, { clusters: 1 });
  assert.deepEqual(Array.from(postingList(result, 0)), [0, 1]);
});

test('optional MetaHarness clustering alias evaluates held-out data', async () => {
  const evaluator = createClusteringEvaluator({
    training: new Float32Array([1, 3, 5, 7]), holdout: new Float32Array([3, 5, 7, 9]), dimensions: 2,
    baseline: { clusters: 1, epochs: 2, batchSize: 2, peakStep: 0.5, seed: 42 },
    domain: { peakStep: { min: 0.25, max: 1, type: 'float', scale: 'linear' } },
    budget: { maxEvaluations: 1, maxDistanceCoordinates: 10000 },
  });
  const score = await evaluator.evaluate(evaluator.baselineGenome, 'fann-integrations-baseline');
  assert.equal(score.raw.holdoutMse, 16);
  assert.equal(evaluator.receipt().sourceApplied, false);
});

test('optional Cloudflare clustering alias executes a local Worker fit', async () => {
  const wasmModule = await WebAssembly.compile(await readFile(new URL(import.meta.resolve('@ruvector/e-means/wasm'))));
  const worker = createClusteringWorker({ wasmModule, allowUnauthenticated: true, maxWork: 10000 });
  const response = await worker.fetch(new Request('https://local.test/fit', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: [1, 3, 5, 7], dimensions: 2, options: { clusters: 1 } }),
  }));
  assert.equal(response.status, 200);
  assert.deepEqual((await response.json()).centroids, [3, 5]);
});
