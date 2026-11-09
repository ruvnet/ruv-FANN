/** Direct native-addon training; no JS/WASM algorithm fallback. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { trainingJson, trainingProof, close } from './training-proof.mjs';
const { NativeTrainableNetwork } = createRequire(import.meta.url)('../ruv_fann.node');

test('actual native reverse-mode, finite-difference, SGD/AdamW learning and immutable snapshot proofs', async () => {
  const receipt = await trainingProof(NativeTrainableNetwork);
  assert.equal(receipt.actualAutograd, true); assert.equal(receipt.finiteDifferences, true);
  if (process.env.RUV_FANN_CANDLE_FIXTURE) assert.equal(receipt.candle.verified, true);
});

test('all training arrays/scalars are checked inside direct addon exports', () => {
  const model = NativeTrainableNetwork.from_dense_layers_json(trainingJson(), 2);
  const input = new Float64Array([2, -1]), targets = new Float64Array([0.7]);
  try {
    for (const value of [new Float32Array(2), new Float64Array(new SharedArrayBuffer(16)),
      new Float64Array(new ArrayBuffer(16, { maxByteLength: 32 })), new Proxy(input, {}), Object.create(Float64Array.prototype)])
      assert.throws(() => model.mse_backward(value, targets, 1));
    const detached = new Float64Array(1); structuredClone(detached.buffer, { transfer: [detached.buffer] });
    assert.throws(() => model.mse_backward(input, detached, 1));
    assert.throws(() => model.mse_backward(input, new Float64Array([NaN]), 1));
    for (const rows of [0, -1, 3, NaN, '1', 1n]) assert.throws(() => model.mse_backward(input, targets, rows));
    const old = model.to_dense_layers_json();
    for (const lr of [0, -1, NaN, Infinity, '0.1', new Number(0.1)]) assert.throws(() => model.sgd_step(lr));
    assert.equal(model.to_dense_layers_json(), old);
  } finally { model.free(); }
});

test('training resource/work caps reject pathological capacity before workspace allocation', () => {
  for (const rows of [0, 4097, NaN, '2', undefined])
    assert.throws(() => NativeTrainableNetwork.from_dense_layers_json(trainingJson(), rows));
  assert.throws(() => NativeTrainableNetwork.from_dense_layers_json(trainingJson().replace('Linear', 'Gelu'), 2));
});

test('oracle comparisons reject nonfinite references and unbounded tolerances', () => {
  for (const value of [Infinity, -Infinity, NaN, '0', null]) {
    assert.throws(() => close([0], [value]), /finite/);
    assert.throws(() => close([value], [0]), /finite/);
  }
  for (const tolerance of [Infinity, NaN, -1, 1, '0'])
    assert.throws(() => close([0], [1], tolerance), /tolerance/);
  close([1], [1], 0);
});
