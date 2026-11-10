import assert from 'node:assert/strict';
import { test } from 'node:test';
import { loadRuntime, LIMITS } from '../src/index.js';
import { loadWasm, supportsSIMD } from 'ruv-fann-wasm';
const model = (weight = 2, bias = 1, activation = 'Linear') => ({ schema: 'ruv-fann.dense-layers/1',
  layers: [{ inputs: 1, outputs: 1, weights: [weight], bias: [bias], activation, steepness: 1 }] });
const training = (extra = {}) => ({ inputs: Float64Array.of(1, 2), targets: Float64Array.of(2, 4),
  rows: 2, epochs: 1, learningRate: 0.1, ...extra });
function close(actual, expected, tolerance = 1e-12) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) <= tolerance, `${value} != ${expected[index]}`));
}
for (const simd of [false, ...(supportsSIMD() ? [true] : [])]) {
  test(`prepared batches equal rows, own snapshots, support aliasing and disposal (${simd})`, async () => {
    const runtime = await loadRuntime({ simd });
    assert.equal(runtime, await loadRuntime({ simd }));
    assert.equal(runtime.backend, simd ? 'wasm-simd' : 'wasm');
    const source = model();
    const prepared = runtime.prepare(source, { maxRows: 3 });
    assert.equal(prepared.inputSize, 1); assert.equal(prepared.outputSize, 1);
    assert.equal(prepared.maxRows, 3); assert.ok(prepared.estimatedBytes > 0 && prepared.estimatedBytes <= LIMITS.maxBytes);
    source.layers[0].weights[0] = 99;
    close(prepared.predict(Float64Array.of(1, 2, 3)), [3, 5, 7]);
    for (let row = 1; row <= 3; row++) close(prepared.predict(Float64Array.of(row)), [2 * row + 1]);
    const alias = Float64Array.of(1, 2, 3); prepared.predictInto(alias, alias); close(alias, [3, 5, 7]);
    close(prepared.predict(new Float64Array(0)), []);
    prepared.dispose(); prepared.dispose();
    assert.throws(() => prepared.predict(Float64Array.of(1)), /disposed/);
    assert.throws(() => prepared.predictInto(Float64Array.of(1), new Float64Array(1)), /disposed/);
    const independent = runtime.prepare(JSON.stringify(model()));
    close(independent.predict(Float64Array.of(2)), [5]); independent.dispose();
  });

  test(`exact SGD/AdamW updates and actual loss decrease (${simd})`, async () => {
    const runtime = await loadRuntime({ simd });
    const source = model(0, 0), snapshot = JSON.stringify(source);
    const one = runtime.train(source, training());
    assert.equal(one.initialLoss, 10); assert.equal(one.epochs, 1); assert.equal(one.parameterCount, 2);
    close(one.model.layers[0].weights, [1]); close(one.model.layers[0].bias, [0.6]);
    assert.ok(Math.abs(one.finalLoss - 1.06) < 1e-12);
    const adam = runtime.train(source, training({ optimizer: 'adamw' }));
    close(adam.model.layers[0].weights, [0.1 * 10 / (10 + 1e-8)]);
    close(adam.model.layers[0].bias, [0.1 * 6 / (6 + 1e-8)]);
    for (const optimizer of ['sgd', 'adamw']) {
      const result = runtime.train(source, { inputs: Float64Array.of(-2, -1, 0, 1, 2),
        targets: Float64Array.of(-3, -1, 1, 3, 5), rows: 5, epochs: 500, learningRate: 0.05, optimizer });
      assert.ok(result.finalLoss < 1e-8 && result.finalLoss < result.initialLoss / 1000);
      close(result.model.layers[0].weights, [2], 1e-4); close(result.model.layers[0].bias, [1], 1e-4);
      const frozen = runtime.prepare(result.model); close(frozen.predict(Float64Array.of(2)), [5], 1e-4); frozen.dispose();
    }
    assert.equal(JSON.stringify(source), snapshot);
  });

  test(`intrinsic array checks, exact bounds and atomic output errors (${simd})`, async () => {
    const runtime = await loadRuntime({ simd });
    const raw = await loadWasm({ simd });
    const prepared = runtime.prepare(model(), { maxRows: 2 });
    const detached = new Float64Array(1); structuredClone(detached.buffer, { transfer: [detached.buffer] });
    const input = Float64Array.of(1);
    for (const invalid of [[1], new Float32Array(1), new Proxy(input, {}), Object.create(Float64Array.prototype),
      detached, new Float64Array(new SharedArrayBuffer(8)), new Float64Array(raw.memory.buffer, 0, 1),
      Float64Array.of(NaN), Float64Array.of(Infinity)]) {
      const output = Float64Array.of(123);
      assert.throws(() => prepared.predictInto(invalid, output)); assert.equal(output[0], 123);
      assert.throws(() => runtime.train(model(), training({ inputs: invalid })));
    }
    for (const rows of [-1, 0.5, 3, NaN, '1', null]) {
      const output = Float64Array.of(123);
      assert.throws(() => prepared.predictInto(input, output, rows)); assert.equal(output[0], 123);
    }
    assert.throws(() => prepared.predict(Float64Array.of(1, 2, 3)));
    assert.throws(() => prepared.predictInto(input, new Float64Array(2)));
    assert.throws(() => prepared.predictInto(input, new Float64Array(raw.memory.buffer, 0, 1)));
    const overflow = runtime.prepare(model(1e308, 0));
    const output = Float64Array.of(123);
    assert.throws(() => overflow.predictInto(Float64Array.of(1e308), output)); assert.equal(output[0], 123);
    overflow.dispose(); prepared.dispose();
  });
}

test('public model/options validation enforces JSON, shape, finite values, epochs and total work', async () => {
  const runtime = await loadRuntime({ simd: false });
  for (const bad of [null, [], {}, { ...model(), schema: 'other' }, { ...model(), extra: 1 },
    model(Infinity), model(NaN), { ...model(), layers: [] },
    { ...model(), layers: [{ ...model().layers[0], weights: [1, 2] }] },
    { ...model(), layers: [{ ...model().layers[0], outputs: 1.5 }] },
    ' '.repeat(LIMITS.maxModelBytes + 1), model(1, 0, 'invented')]) assert.throws(() => runtime.prepare(bad));
  for (const maxRows of [0, 4097, 1.5, '1', null]) assert.throws(() => runtime.prepare(model(), { maxRows }));
  assert.throws(() => runtime.prepare(model(), { other: true }));
  for (const epochs of [0, -1, 1.5, 1001, Infinity, '1', null])
    assert.throws(() => runtime.train(model(), training({ epochs })), /epochs/);
  for (const learningRate of [0, -1, Infinity, NaN, '0.1', null])
    assert.throws(() => runtime.train(model(), training({ learningRate })));
  assert.throws(() => runtime.train(model(), training({ optimizer: 'fake' })));
  assert.throws(() => runtime.train(model(), training({ optimizer: null })));
  assert.throws(() => runtime.train(model(), training({ other: true })));
  assert.throws(() => runtime.train(model(), training({ rows: 4097 })));
  assert.throws(() => runtime.train(model(), training({ targets: Float64Array.of(1) })));
  assert.throws(() => runtime.train(model(1, 0, 'Gaussian'), training()));
  const wide = { schema: 'ruv-fann.dense-layers/1', layers: [{ inputs: 100, outputs: 1,
    weights: Array(100).fill(0), bias: [0], activation: 'Linear', steepness: 1 }] };
  assert.throws(() => runtime.train(wide, { inputs: new Float64Array(10000), targets: new Float64Array(100),
    rows: 100, epochs: 1000, learningRate: 0.1 }), /total work budget/);
});
