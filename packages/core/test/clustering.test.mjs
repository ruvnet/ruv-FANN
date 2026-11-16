import test from 'node:test';
import assert from 'node:assert/strict';
import { init, fit, predict } from 'ruv-fann/clustering';

await init();

test('clustering capability trains and predicts using the shared WASM dependency', () => {
  const data = new Float32Array([-10, -9, -9, -10, 9, 10, 10, 9]);
  const options = { clusters: 2, epochs: 6, batchSize: 2, seed: 42 };
  const model = fit(data, 2, options);
  assert.ok(model.centroids instanceof Float32Array);
  assert.ok(model.assignments instanceof Uint32Array);
  assert.equal(model.assignments[0], model.assignments[1]);
  assert.equal(model.assignments[2], model.assignments[3]);
  assert.notEqual(model.assignments[0], model.assignments[2]);
  assert.ok(model.mse < 1);
  assert.deepEqual(predict(data, 2, model.centroids), model.assignments);
  assert.deepEqual(fit(data, 2, options).centroids, model.centroids);
});

test('clustering capability uses the last-epoch mean and rejects malformed inputs', () => {
  const model = fit(new Float32Array([0, 2, 4, 6, 8, 10]), 2, { clusters: 1, batchSize: 2 });
  assert.deepEqual(Array.from(model.centroids), [4, 6]);
  assert.throws(() => fit(new Float32Array([NaN]), 1, { clusters: 1 }));
  assert.throws(() => fit(new Float32Array([1, 2, 3]), 2, { clusters: 1 }));
  assert.throws(() => fit(new Float32Array(), 1, { clusters: 1 }));
});
