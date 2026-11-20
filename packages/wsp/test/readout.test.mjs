import assert from 'node:assert/strict';
import test from 'node:test';
import { applyWeightPatch, createWeightWorkspace, proposeReadoutDelta, traceWeightModel } from '../src/index.js';
import { linearModel } from './fixtures.mjs';

const probes = { rows: 5, inputs: [-2, -1, 0, 1, 2] };
const options = { simd: false };
async function request(model, output = 0) {
  const { baseHash, inputHash } = await traceWeightModel(model, probes, options);
  return { baseHash, inputHash, output, deltas: probes.inputs.map(x => x + 1) };
}

test('fits additive deltas on a nonzero base and rolls back the evaluated edit', async () => {
  const model = linearModel(2, [0.5]);
  const spec = await request(model);
  const batch = structuredClone(probes);
  const pending = proposeReadoutDelta(model, batch, spec, options);
  batch.inputs.fill(100); spec.deltas.fill(0); model.layers[0].weights[0] = 7;
  const patch = await pending;
  const workspace = await createWeightWorkspace(linearModel(2, [0.5]), options);
  const proposal = await workspace.propose(patch, { rows: 2, inputs: [-3, 4], targets: [-7.5, 13.5] },
    { maxMSE: 1e-5, maxRegressionMSE: 0 });
  assert.ok(proposal.evaluation.accepted);
  assert.equal(workspace.snapshot().revision, 1);
  assert.equal(workspace.promote(proposal.id, 1).revision, 2);
  assert.equal(workspace.rollback(1, 2).hash, patch.baseHash);
});

test('hidden permutations and duplicate features preserve fitted behavior and untouched outputs', async () => {
  const observed = [];
  for (const order of [[1, 1, -1], [-1, 1, 1]]) {
    const model = { schema: 'ruv-fann.dense-layers/1', layers: [
      { inputs: 1, outputs: 3, weights: order, bias: [0, 0, 0], activation: 'ReLU', steepness: 1 },
      { inputs: 3, outputs: 2, weights: [...order, 0, 0, 0], bias: [0, 0], activation: 'Linear', steepness: 1 }
    ] };
    const original = structuredClone(model);
    const patch = await proposeReadoutDelta(model, probes, await request(model, 1), options);
    assert.ok(patch.edits.every(e => e.layer === 1 && (e.parameter === 'weights' ? e.index >= 3 : e.index === 1)));
    const candidate = await applyWeightPatch(model, patch);
    assert.deepEqual(model, original);
    const holdout = { inputs: [-1.5, 3.5], rows: 2 };
    const old = (await traceWeightModel(model, holdout, options)).layers.at(-1).activations;
    const next = (await traceWeightModel(candidate, holdout, options)).layers.at(-1).activations;
    assert.deepEqual([next[0], next[2]], [old[0], old[2]]);
    assert.ok(Math.abs(next[1] + 0.5) < 0.003 && Math.abs(next[3] - 4.5) < 0.003);
    observed.push(next[1], next[3]);
  }
  assert.ok(Math.abs(observed[0] - observed[2]) < 1e-10 && Math.abs(observed[1] - observed[3]) < 1e-10);
});

test('rejects stale and reordered probes, unsupported heads, budgets and malformed values', async () => {
  const model = linearModel(), spec = await request(model);
  const propose = (m, b, s, o = options) => proposeReadoutDelta(m, b, s, o);
  assert.equal(await propose(model, probes, { ...spec, deltas: Array(5).fill(0) }), null);
  await assert.rejects(propose(linearModel(3), probes, spec), /hash mismatch/i);
  await assert.rejects(propose(model, { rows: 5, inputs: probes.inputs.toReversed() }, spec), /hash mismatch/i);
  await assert.rejects(propose(model, probes, { ...spec, output: 1 }), /output/);
  await assert.rejects(propose(model, probes, spec, { limits: { maxEdits: 1 } }), /row including bias/);
  await assert.rejects(propose(model, probes, spec, { limits: { maxAbsDelta: 0.01 } }), /maxAbsDelta/);
  await assert.rejects(propose(model, { inputs: [1], rows: 1 }, spec), /too few probes/);
  await assert.rejects(propose(model, probes, { ...spec, deltas: [NaN, 0, 0, 0, 0] }), /finite/);
  const hole = Array(5);
  Object.defineProperty(hole, 0, { get() { assert.fail('caller getter invoked'); } });
  await assert.rejects(propose(model, probes, { ...spec, deltas: hole }), /own data/);
  model.layers[0].activation = 'Tanh';
  await assert.rejects(propose(model, probes, spec), /Linear head/);
});
