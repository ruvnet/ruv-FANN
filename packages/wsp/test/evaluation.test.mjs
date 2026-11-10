import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { loadRuntime } from 'ruv-fann';
import { applyWeightPatch, fingerprintModel, evaluateWeightPatch } from '../src/index.js';
import { xorModel, xorData, linearModel, edit, patch, linearData, strictPolicy } from './fixtures.mjs';
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');

for (const simd of [false, true]) {
  test(`hand-authored gradient-free XOR construction and repair on actual WASM (SIMD=${simd})`, async () => {
    const original = xorModel(), before = JSON.stringify(original);
    const baseHash = await fingerprintModel(original);
    const proposal = patch(baseHash, [edit(1, 1, 'weights', 0), edit(1, 1, 'weights', 1)]);
    const runtime = await loadRuntime({ simd });
    const snapshot = runtime.prepare(original, { maxRows: 4 });
    try {
      const evaluation = await evaluateWeightPatch(original, proposal, xorData(), strictPolicy(), { simd });
      assert.equal(evaluation.backend, simd ? 'wasm-simd' : 'wasm');
      assert.equal(evaluation.schema, 'ruv-fann.weight-evaluation/1');
      assert.equal(evaluation.accepted, true); assert.deepEqual(evaluation.reasons, []);
      assert.equal(evaluation.baseline.mse, 0.5); assert.equal(evaluation.candidate.mse, 0);
      assert.equal(evaluation.baseline.maxAbsoluteError, 1); assert.equal(evaluation.candidate.maxAbsoluteError, 0);
      assert.equal(evaluation.candidateHash, await fingerprintModel(evaluation.model));
      assert.equal(evaluation.baseHash, baseHash);
      const candidate = runtime.prepare(evaluation.model, { maxRows: 4 });
      try {
        assert.deepEqual([...candidate.predict(Float64Array.from(xorData().inputs))], [0, 1, 1, 0]);
        assert.deepEqual([...snapshot.predict(Float64Array.from(xorData().inputs))], [0, 0, 0, 0]);
      } finally { candidate.dispose(); }
      assert.equal(JSON.stringify(original), before);
      assert.deepEqual(evaluation.model, await applyWeightPatch(original, proposal));
      assert.doesNotThrow(() => JSON.stringify(evaluation));
    } finally { snapshot.dispose(); }
  });
}

test('each policy threshold rejects independently; equality accepts and regression is explicit', async () => {
  const source = linearModel(), baseHash = await fingerprintModel(source), proposed = patch(baseHash, [edit(1)]);
  const insufficient = await evaluateWeightPatch(source, proposed, linearData(), { maxMSE: 1, maxRegressionMSE: 0 });
  assert.equal(insufficient.accepted, false); assert.deepEqual(insufficient.reasons, ['candidate MSE exceeds maxMSE']);
  const absolute = await evaluateWeightPatch(source, proposed, linearData(), { maxMSE: 10, maxRegressionMSE: 0, maxAbsoluteError: 1 });
  assert.equal(absolute.accepted, false); assert.deepEqual(absolute.reasons, ['candidate max absolute error exceeds maxAbsoluteError']);
  const equality = await evaluateWeightPatch(source, proposed, linearData(), { maxMSE: 2.5, maxRegressionMSE: 0, maxAbsoluteError: 2 });
  assert.equal(equality.accepted, true);
  const repaired = await applyWeightPatch(source, patch(baseHash));
  const regression = await evaluateWeightPatch(repaired, patch(await fingerprintModel(repaired), [edit(0)]),
    linearData(), { maxMSE: 20, maxRegressionMSE: 0 });
  assert.equal(regression.accepted, false); assert.deepEqual(regression.reasons, ['candidate MSE regression exceeds maxRegressionMSE']);
});

test('canonical model, dataset dimensions and policy are independently SHA256 bound', async () => {
  const source = linearModel(0), proposal = patch(await fingerprintModel(source));
  const data = linearData(), policy = { maxMSE: 0, maxRegressionMSE: 0, maxAbsoluteError: 0 };
  const result = await evaluateWeightPatch(source, proposal, data, policy);
  assert.equal(result.baseHash, hash({ schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 1, outputs: 1, weights: [0], bias: [0], activation: 'Linear', steepness: 1 }
  ] }));
  assert.equal(result.datasetHash, hash({ schema: 'ruv-fann.weight-dataset/1', inputSize: 1, outputSize: 1,
    rows: data.rows, inputs: data.inputs, targets: data.targets }));
  assert.equal(result.policyHash, hash({ schema: 'ruv-fann.weight-policy/1', ...policy }));
  const reordered = await evaluateWeightPatch(source, proposal, { targets: [2, 4], rows: 2, inputs: [1, 2] },
    { maxAbsoluteError: 0, maxRegressionMSE: 0, maxMSE: 0 });
  assert.equal(reordered.datasetHash, result.datasetHash); assert.equal(reordered.policyHash, result.policyHash);
  const other = await evaluateWeightPatch(source, proposal, { inputs: [1, 2], targets: [2, 3], rows: 2 }, policy);
  assert.notEqual(other.datasetHash, result.datasetHash);
  const alteredPolicy = await evaluateWeightPatch(source, proposal, data, { maxMSE: 0.01, maxRegressionMSE: 0 });
  assert.notEqual(alteredPolicy.policyHash, result.policyHash);
});

test('multi-batch evaluation with two outputs preserves row/target ordering and exact metrics', async () => {
  const source = { schema: 'ruv-fann.dense-layers/1', layers: [{ inputs: 2, outputs: 2,
    weights: [0, 0, 0, 0], bias: [0, 0], activation: 'Linear', steepness: 1 }] };
  const rows = 97, inputs = [], targets = [];
  for (let row = 0; row < rows; row++) {
    inputs.push(row, -row); targets.push(row + 3, -row - 4);
  }
  const proposal = patch(await fingerprintModel(source), [edit(1, 0, 'weights', 0), edit(1, 0, 'weights', 3),
    edit(1, 0, 'bias', 0), edit(-1, 0, 'bias', 1)]);
  const result = await evaluateWeightPatch(source, proposal, { inputs, targets, rows }, { maxMSE: 6.5, maxRegressionMSE: 0 });
  // Candidate errors are -2,+3 on every row, including the last partial batch.
  assert.equal(result.candidate.mse, 6.5); assert.equal(result.candidate.maxAbsoluteError, 3);
  assert.equal(result.accepted, true); assert.equal(result.rows, rows);
  let baseline = 0;
  targets.forEach((value, index) => { baseline += (value * value - baseline) / (index + 1); });
  assert.equal(result.baseline.mse, baseline); assert.equal(result.baseline.maxAbsoluteError, rows + 3);
});

test('finite online MSE survives sum overflow, while squared/difference/prediction overflow reject', async () => {
  const large = value => ({ schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 1, outputs: 1, weights: [value], bias: [0], activation: 'Linear', steepness: 1 },
    { inputs: 1, outputs: 1, weights: [1], bias: [0], activation: 'Linear', steepness: 1 }
  ] });
  const source = large(1e154), proposal = patch(await fingerprintModel(source), [edit(1, 1)]);
  const accepted = await evaluateWeightPatch(source, proposal, { inputs: [1, 1, 1, 1], targets: [0, 0, 0, 0], rows: 4 },
    { maxMSE: Number.MAX_VALUE, maxRegressionMSE: 0 });
  assert.equal(accepted.accepted, true); assert.equal(accepted.candidate.mse, 1e308);
  for (const [value, input, target] of [[1e155, 1, 0], [1e308, 1, -1e308], [1e308, 2, 0]]) {
    const model = large(value), patchValue = patch(await fingerprintModel(model), [edit(1, 1)]);
    await assert.rejects(evaluateWeightPatch(model, patchValue, { inputs: [input], targets: [target], rows: 1 },
      { maxMSE: Number.MAX_VALUE, maxRegressionMSE: Number.MAX_VALUE }));
  }
  // Failure must not poison or leak an unusable native snapshot into a later evaluation.
  const simple = linearModel(), repaired = await evaluateWeightPatch(simple, patch(await fingerprintModel(simple)), linearData(), strictPolicy());
  assert.equal(repaired.accepted, true);
});
