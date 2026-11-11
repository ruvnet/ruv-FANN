import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fingerprintModel } from 'ruv-fann-wsp';
import { createAgentDBRegistry, RegistryPersistenceError } from '../src/index.js';
import { openDemoRegistry } from '../../../tooling/agentdb/runtime.mjs';

const model = () => ({ schema: 'ruv-fann.dense-layers/1', layers: [{
  inputs: 1, outputs: 1, weights: [0], bias: [0], activation: 'Linear', steepness: 1
}] });
const data = () => ({ inputs: [1, 2], targets: [2, 4], rows: 2 });
const policy = () => ({ maxMSE: 0, maxRegressionMSE: 0 });
const patch = (hash, value = 2) => ({ schema: 'ruv-fann.weight-patch/1', baseHash: hash,
  edits: [{ layer: 0, parameter: 'weights', index: 0, value }] });
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
async function fixture(t) {
  const directory = await mkdtemp(join(tmpdir(), 'wsp-agentdb-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return join(directory, 'weights.db');
}
async function raw(filename) {
  let createDatabase, WeightPrograms;
  try {
    ({ createDatabase } = await import('../../../tooling/agentdb/node_modules/agentdb/dist/src/db-fallback.js'));
    ({ WeightPrograms } = await import('../../../tooling/agentdb/node_modules/agentdb/dist/src/controllers/WeightPrograms.js'));
  } catch (error) {
    throw new Error('AgentDB test fixture missing; npm ci --prefix tooling/agentdb --ignore-scripts --omit=optional', { cause: error });
  }
  const db = await createDatabase(filename), programs = new WeightPrograms(db, { simd: false });
  return { db, programs,
    cleanup({ poisoned }) { programs.dispose(); if (db.save) db.save = () => {}; db.close(); } };
}

test('real AgentDB create, proposal, promotion and rollback survive each close/reopen', async t => {
  const filename = await fixture(t);
  let registry = await openDemoRegistry(filename);
  const initial = await registry.create('linear', model());
  assert.equal(initial.revision, 1);
  assert.equal(registry.status().durability, 'caller-flush');
  await registry.close();
  registry = await openDemoRegistry(filename);
  assert.equal((await registry.get('linear')).hash, initial.hash);
  const proposed = await registry.propose('linear', patch(initial.hash), data(), policy());
  assert.equal(proposed.evaluation.accepted, true);
  assert.equal(proposed.evaluation.backend, 'wasm');
  assert.equal(proposed.evaluation.baseline.mse, 10);
  assert.equal(proposed.evaluation.candidate.mse, 0);
  await registry.close();
  registry = await openDemoRegistry(filename);
  const promoted = await registry.promote('linear', proposed.proposalId, 1);
  assert.equal(promoted.revision, 2);
  assert.equal(promoted.model.layers[0].weights[0], 2);
  assert.equal(promoted.hash, await fingerprintModel(promoted.model));
  await registry.close();
  registry = await openDemoRegistry(filename);
  const restored = await registry.rollback('linear', 1, 2);
  assert.equal(restored.hash, initial.hash);
  assert.equal(restored.revision, 3);
  await registry.close();
  registry = await openDemoRegistry(filename);
  t.after(() => registry.close());
  const final = await registry.get('linear');
  assert.deepEqual(final.history.map(row => row.operation), ['rollback', 'promote', 'create']);
  assert.equal(final.revision, 3);
  assert.equal(final.model.layers[0].weights[0], 0);
  assert.equal((await registry.list())[0].hash, initial.hash);
});

test('rejected proposals, forged receipts/IDs and detached result edits cannot promote', async t => {
  const registry = await openDemoRegistry(await fixture(t)); t.after(() => registry.close());
  const initial = await registry.create('test', model());
  const rejected = await registry.propose('test', patch(initial.hash, -1), data(), policy());
  assert.equal(rejected.evaluation.accepted, false);
  rejected.evaluation.accepted = true;
  await assert.rejects(registry.promote('test', rejected.proposalId, 1), /rejected/);
  await assert.rejects(registry.promote('test', { accepted: true }, 1), /proposalId/);
  await assert.rejects(registry.promote('test', 'forged-internal-id', 1), /not found/);
  const good = await registry.propose('test', patch(initial.hash), data(), policy());
  good.evaluation.model.layers[0].weights[0] = 100;
  const promoted = await registry.promote('test', good.proposalId, 1);
  assert.equal(promoted.model.layers[0].weights[0], 2);
  assert.equal(promoted.hash, good.evaluation.candidateHash);
});

test('concurrent promotions serialize; exactly one expected revision succeeds', async t => {
  const registry = await openDemoRegistry(await fixture(t)); t.after(() => registry.close());
  const initial = await registry.create('cas', model());
  const first = await registry.propose('cas', patch(initial.hash), data(), policy());
  const second = await registry.propose('cas', patch(initial.hash), data(), policy());
  const results = await Promise.allSettled([
    registry.promote('cas', first.proposalId, 1), registry.promote('cas', second.proposalId, 1)
  ]);
  assert.deepEqual(results.map(r => r.status), ['fulfilled', 'rejected']);
  assert.match(results[1].reason.message, /stale|revision/i);
  assert.equal(results[1].reason.code, 'WSP_REGISTRY_STALE');
  assert.equal((await registry.get('cas')).revision, 2);
  await assert.rejects(registry.rollback('cas', 1, 1), /stale|revision/i);
});

test('queued inputs snapshot before awaiting preceding persistence; reads wait for flush', async t => {
  const source = await raw(await fixture(t)), entered = deferred(), release = deferred();
  let flushes = 0;
  const registry = createAgentDBRegistry(source.programs, {
    flush: async () => { if (++flushes === 1) { entered.resolve(); await release.promise; } source.db.save?.(); },
    close: state => source.cleanup(state)
  });
  t.after(() => registry.close());
  const initial = model(), hash = await fingerprintModel(initial);
  const creating = registry.create('snapshot', initial);
  await entered.promise;
  const candidate = patch(hash), dataset = data(), criteria = policy();
  const proposing = registry.propose('snapshot', candidate, dataset, criteria);
  let readFinished = false;
  const reading = registry.get('snapshot').then(record => { readFinished = true; return record; });
  candidate.edits[0].value = 100; dataset.targets[0] = -100; criteria.maxMSE = 100;
  initial.layers[0].weights[0] = 100;
  await Promise.resolve(); assert.equal(readFinished, false);
  release.resolve();
  await creating;
  const proposal = await proposing;
  assert.equal(proposal.evaluation.accepted, true);
  assert.equal(proposal.evaluation.candidate.mse, 0);
  assert.equal(proposal.evaluation.model.layers[0].weights[0], 2);
  assert.equal((await reading).model.layers[0].weights[0], 0);
});

test('failed real-controller flush rejects acknowledgement and poisons subsequent work', async t => {
  const filename = await fixture(t), source = await raw(filename);
  let count = 0, closeState;
  const registry = createAgentDBRegistry(source.programs, {
    flush: () => { if (++count === 3) throw new Error('controlled disk flush failure'); source.db.save?.(); },
    close: state => { closeState = state; source.cleanup(state); }
  });
  const initial = await registry.create('failure', model());
  const proposal = await registry.propose('failure', patch(initial.hash), data(), policy());
  await assert.rejects(registry.promote('failure', proposal.proposalId, 1), error =>
    error instanceof RegistryPersistenceError && error.code === 'WSP_REGISTRY_PERSISTENCE');
  assert.equal(registry.status().poisoned, true);
  await assert.rejects(registry.get('failure'), /poisoned/);
  await assert.rejects(registry.rollback('failure', 1, 2), /poisoned/);
  await registry.close();
  assert.equal(closeState.poisoned, true);
  const reopened = await openDemoRegistry(filename); t.after(() => reopened.close());
  // This injected failure happened before save, and cleanup deliberately did not retry.
  assert.equal((await reopened.get('failure')).revision, 1);
  assert.equal((await reopened.get('failure')).hash, initial.hash);
});

test('queue capacity and close are bounded; cleanup runs once after queued work', async t => {
  const source = await raw(await fixture(t)), entered = deferred(), release = deferred();
  let cleanups = 0;
  const registry = createAgentDBRegistry(source.programs, {
    maxPending: 1,
    flush: async () => { entered.resolve(); await release.promise; source.db.save?.(); },
    close: state => { cleanups++; source.cleanup(state); }
  });
  const creating = registry.create('bounded', model());
  await entered.promise;
  await assert.rejects(registry.get('bounded'), /queue is full/);
  const closing = registry.close(), again = registry.close();
  assert.equal(closing, again);
  await assert.rejects(registry.get('bounded'), /closing/);
  release.resolve(); await creating; await closing;
  assert.equal(cleanups, 1);
  assert.equal(registry.status().closed, true);
  await assert.rejects(registry.list(), /closed/);
});

test('plain data, identifiers, queue options, unknown fields and accessors fail safely', async t => {
  const registry = await openDemoRegistry(await fixture(t)); t.after(() => registry.close());
  let getters = 0;
  const source = model(); Object.defineProperty(source.layers[0].weights, '0', { get() { getters++; return 0; } });
  await assert.rejects(registry.create('getter', source), /accessors/); assert.equal(getters, 0);
  for (const bad of ['bad/id', '', 'x'.repeat(129), null]) await assert.rejects(registry.get(bad));
  for (const limit of [0, 101, NaN, '1']) await assert.rejects(registry.list(limit));
  await assert.rejects(registry.create('huge', 'x'.repeat(2 * 1024 * 1024 + 1)), /bound/);
  await assert.rejects(registry.create('unsafe', Object.assign(model(), { toJSON() { throw Error('executed'); } })), /JSON tree/);
  await assert.rejects(registry.create('unknown', { ...model(), unrecognized: true }), /unknown|field/i);
  await assert.rejects(registry.create('missing'), /arguments/);
  assert.throws(() => createAgentDBRegistry({ get get() { getters++; return () => {}; } }), /lacks create/);
  const rawSource = await raw(':memory:');
  try {
    for (const config of [{ maxPending: 17 }, { maxPending: 0 }, { extra: true }, { flush: true }])
      assert.throws(() => createAgentDBRegistry(rawSource.programs, config));
    const getterOptions = { get flush() { getters++; return () => {}; } };
    assert.throws(() => createAgentDBRegistry(rawSource.programs, getterOptions), /accessor/);
    assert.equal(getters, 0);
  } finally { rawSource.cleanup({ poisoned: false }); }
});

test('real sql.js concurrent file change is rejected rather than overwriting another owner', async t => {
  const filename = await fixture(t);
  const first = await openDemoRegistry(filename), initial = await first.create('shared', model());
  const stale = await openDemoRegistry(filename);
  try {
    await first.propose('shared', patch(initial.hash), data(), policy());
    await assert.rejects(stale.create('other', model()), /persistence failed/);
    assert.equal(stale.status().poisoned, true);
    await stale.close();
    await first.close();
    const reopened = await openDemoRegistry(filename); t.after(() => reopened.close());
    assert.equal(await reopened.get('other'), null);
    assert.equal((await reopened.get('shared')).proposals.length, 1);
  } finally { await stale.close(); await first.close(); }
});
