import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createWeightWorkspace, fingerprintModel } from '../src/index.js';
import { linearModel, edit, patch, linearData, strictPolicy } from './fixtures.mjs';

test('workspace promotion requires an internal accepted proposal, with exact rollback and monotonic revisions', async () => {
  const workspace = await createWeightWorkspace(linearModel(), { simd: false });
  const start = workspace.snapshot();
  assert.equal(start.revision, 1);
  const proposed = await workspace.propose(patch(start.hash), linearData(), strictPolicy());
  assert.equal(workspace.snapshot().revision, 1); // Evaluation alone cannot promote.
  assert.match(proposed.id, /^[a-f0-9]{32}$/);
  assert.equal(proposed.revision, 1);
  const id = proposed.id;
  proposed.evaluation.accepted = false; proposed.evaluation.model.layers[0].weights[0] = 999;
  proposed.evaluation.candidateHash = 'f'.repeat(64); proposed.evaluation.policy.maxMSE = 999;
  proposed.revision = 999; proposed.id = 'forged';
  const promoted = workspace.promote(id, 1);
  assert.equal(promoted.revision, 2); assert.equal(promoted.model.layers[0].weights[0], 2);
  assert.equal(promoted.hash, await fingerprintModel(promoted.model));
  promoted.model.layers[0].weights[0] = -999;
  assert.equal(workspace.snapshot().model.layers[0].weights[0], 2);
  const next = await workspace.propose(patch(workspace.snapshot().hash, [edit(3)]),
    { inputs: [1, 2], targets: [3, 6], rows: 2 }, strictPolicy());
  assert.equal(workspace.promote(next.id, 2).revision, 3);
  const rolled = workspace.rollback(2, 3);
  assert.equal(rolled.revision, 4); assert.equal(rolled.model.layers[0].weights[0], 2);
  assert.equal(rolled.hash, promoted.hash);
  const restored = workspace.rollback(1, 4);
  assert.equal(restored.revision, 5); assert.equal(restored.hash, start.hash);
  assert.equal(restored.model.layers[0].weights[0], 0);
  const history = workspace.history();
  assert.deepEqual(history.map(entry => entry.action), ['initial', 'promote', 'promote', 'rollback', 'rollback']);
  assert.equal(history[3].targetRevision, 2); assert.equal(history[3].previousRevision, 3);
  assert.ok(history.every(entry => !Object.hasOwn(entry, 'model')));
  history[0].hash = 'forged'; start.model.layers[0].weights[0] = 999;
  assert.notEqual(workspace.history()[0].hash, 'forged'); assert.equal(workspace.snapshot().model.layers[0].weights[0], 0);
});

test('forged receipts, rejected evaluation tampering, cross-workspace ids and stale CAS never authorize promotion', async () => {
  const workspace = await createWeightWorkspace(linearModel());
  const hash = workspace.snapshot().hash;
  const rejected = await workspace.propose(patch(hash, [edit(1)]), linearData(), strictPolicy());
  assert.equal(rejected.evaluation.accepted, false);
  rejected.evaluation.accepted = true; rejected.evaluation.model.layers[0].weights[0] = 2;
  assert.throws(() => workspace.promote(rejected.id, 1), /not accepted/);
  assert.throws(() => workspace.promote(rejected.evaluation, 1), /id/);
  assert.throws(() => workspace.promote('f'.repeat(32), 1), /unknown/);
  const other = await createWeightWorkspace(linearModel());
  const external = await other.propose(patch(hash), linearData(), strictPolicy());
  assert.throws(() => workspace.promote(external.id, 1), /unknown/);
  const accepted = await workspace.propose(patch(hash), linearData(), strictPolicy());
  assert.throws(() => workspace.promote(accepted.id, 2), /stale/);
  assert.throws(() => workspace.promote(accepted.id, Number.MAX_SAFE_INTEGER + 1), /expectedRevision/);
  workspace.promote(accepted.id, 1);
  assert.throws(() => workspace.promote(accepted.id, 1), /stale/);
  assert.throws(() => workspace.promote(accepted.id, 2), /unknown/);
  assert.throws(() => workspace.rollback(1, 1), /stale/);
  assert.throws(() => workspace.rollback(999, 2), /retained/);
  assert.throws(() => workspace.rollback(Number.MAX_SAFE_INTEGER + 1, 2), /targetRevision/);
});

test('propose snapshots data immediately and concurrent promotion invalidates in-flight evaluation', async () => {
  const workspace = await createWeightWorkspace(linearModel()), hash = workspace.snapshot().hash;
  const supplied = patch(hash), data = linearData(), policy = strictPolicy();
  const evaluating = workspace.propose(supplied, data, policy);
  supplied.edits[0].value = 999; data.inputs.fill(0); data.targets.fill(999); policy.maxMSE = 999;
  const original = await evaluating;
  assert.equal(original.evaluation.accepted, true); assert.equal(original.evaluation.model.layers[0].weights[0], 2);
  const pending = workspace.propose(patch(hash), linearData(), strictPolicy());
  workspace.promote(original.id, 1);
  await assert.rejects(pending, /stale during evaluation/);
});

test('workspace caps and evicts proposals/history; old hashes do not bypass revision lineage', async () => {
  const workspace = await createWeightWorkspace(linearModel()), hash = workspace.snapshot().hash;
  let first, last;
  for (let index = 0; index < 17; index++) {
    last = await workspace.propose(patch(hash), linearData(), strictPolicy());
    first ??= last;
  }
  assert.notEqual(first.id, last.id); assert.throws(() => workspace.promote(first.id, 1), /expired/);
  workspace.promote(last.id, 1);
  const rollback = workspace.rollback(1, 2); assert.equal(rollback.hash, hash); assert.equal(rollback.revision, 3);
  assert.throws(() => workspace.promote(first.id, 3), /stale/);
  for (let index = 0; index < 32; index++) {
    const current = workspace.snapshot();
    const noChange = await workspace.propose(patch(current.hash, [edit(0)]), { inputs: [1], targets: [0], rows: 1 }, strictPolicy());
    workspace.promote(noChange.id, current.revision);
  }
  const history = workspace.history();
  assert.equal(history.length, 32); assert.equal(history.at(-1).revision, 35); assert.equal(history[0].revision, 4);
  assert.throws(() => workspace.rollback(1, 35), /retained/);
  assert.equal(workspace.rollback(4, 35).revision, 36);
  assert.equal(workspace.history().length, 32);
});

test('in-flight evaluations are bounded, not merely retained proposal records', async () => {
  const workspace = await createWeightWorkspace(linearModel()), hash = workspace.snapshot().hash;
  const results = await Promise.allSettled(Array.from({ length: 17 }, () => workspace.propose(patch(hash), linearData(), strictPolicy())));
  assert.equal(results.filter(result => result.status === 'fulfilled').length, 16);
  const rejection = results.find(result => result.status === 'rejected');
  assert.match(rejection.reason.message, /in-flight/);
});
