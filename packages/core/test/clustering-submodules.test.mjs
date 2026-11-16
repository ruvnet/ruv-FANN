import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { init } from 'ruv-fann/clustering';
import { trainProductQuantizer, encodeProductQuantizer, decodeProductQuantizer } from 'ruv-fann/clustering/quantization';
import { partition, postingList } from 'ruv-fann/clustering/ivf';
import { createClusteringEvaluator } from 'ruv-fann/clustering/metaharness';
import { createClusteringWorker } from 'ruv-fann/clustering/cloudflare';

await init();

test('clustering submodules expose real shared codebook codecs and IVF posting lists', () => {
  const data = new Float32Array([0, 2, 4, 6, 8, 10]);
  const model = trainProductQuantizer(data, 2, 2, { clusters: 1, batchSize: 2 });
  const codes = encodeProductQuantizer(model, data);
  assert.deepEqual(codes, new Uint8Array(6));
  assert.deepEqual(Array.from(decodeProductQuantizer(model, codes)), [4, 6, 4, 6, 4, 6]);
  const result = partition(data, 2, { clusters: 1, batchSize: 2 });
  assert.deepEqual(Array.from(result.model.centroids), [4, 6]);
  assert.deepEqual(Array.from(postingList(result, 0)), [0, 1, 2]);
});

test('clustering MetaHarness alias measures a real held-out numerical objective', async () => {
  const evaluator = createClusteringEvaluator({
    training: new Float32Array([1, 3, 5, 7]), holdout: new Float32Array([3, 5, 7, 9]), dimensions: 2,
    baseline: { clusters: 1, epochs: 2, batchSize: 2, peakStep: 0.5, seed: 42 },
    domain: { peakStep: { min: 0.25, max: 1, type: 'float', scale: 'linear' } },
    budget: { maxEvaluations: 1, maxDistanceCoordinates: 10000 },
  });
  const score = await evaluator.evaluate(evaluator.baselineGenome, 'fann-core-baseline');
  assert.equal(score.raw.trainMse, 8);
  assert.equal(score.raw.holdoutMse, 16);
  assert.equal(evaluator.receipt().trials.length, 1);
  assert.equal(evaluator.receipt().sourceApplied, false);
});

test('clustering Cloudflare alias executes a local Worker request with real WASM', async () => {
  const wasmModule = await WebAssembly.compile(await readFile(new URL(import.meta.resolve('@ruvector/e-means/wasm'))));
  const worker = createClusteringWorker({ wasmModule, allowUnauthenticated: true, maxWork: 10000 });
  const response = await worker.fetch(new Request('https://local.test/fit', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: [1, 3, 5, 7], dimensions: 2, options: { clusters: 1 } }),
  }));
  assert.equal(response.status, 200);
  assert.deepEqual((await response.json()).centroids, [3, 5]);
});
