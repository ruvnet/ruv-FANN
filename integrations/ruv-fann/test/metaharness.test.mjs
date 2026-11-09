// Actual installed Darwin, with a deterministic synthetic prepared backend and
// clock. These are API/resource tests, not hardware performance measurements.
import test, {after, before} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp, readdir, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {performance} from 'node:perf_hooks';
import {tuneBatchRows} from '../src/metaharness.mjs';
import {pairedRatio} from '../src/timing.mjs';

const tensorHash = values => createHash('sha256').update(Buffer.from(values.buffer, values.byteOffset, values.byteLength)).digest('hex');
const inputs = () => Float64Array.from({length: 256}, (_, i) => (i % 17 - 8) / 16);
const archives = async () => new Set((await readdir(tmpdir())).filter(name => name.startsWith('ruv-fann-darwin-')));
const originalTemp = Object.fromEntries(['TMPDIR', 'TMP', 'TEMP'].map(name => [name, process.env[name]]));
let privateTemp;

before(async () => {
  privateTemp = await mkdtemp(join(tmpdir(), `ruv-fann-tuner-tests-${process.pid}-`));
  for (const name of Object.keys(originalTemp)) process.env[name] = privateTemp;
  assert.equal(tmpdir(), privateTemp);
});

after(async () => {
  for (const [name, value] of Object.entries(originalTemp)) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
  if (privateTemp) await rm(privateTemp, {recursive: true, force: true});
});

function clock(t) {
  const state = {now: 0};
  t.mock.method(performance, 'now', () => (state.now += 0.001));
  return state;
}

function synthetic(settings = {}) {
  const state = {freed: 0, created: 0, active: 0, peak: 0, batchCalls: 0, referenceCalls: 0};
  const row = (values, first = 0) => [values[first] + values[first + 1], values[first] - values[first + 1]];
  const prepared = {
    input_size: 2, output_size: 2, estimated_bytes: 256,
    run(input) {
      state.referenceCalls++;
      settings.onReference?.(input, state);
      const output = new Float64Array(row(input));
      if (settings.corruptReferenceAfterBatch && state.batchCalls) output[0] += 1;
      return output;
    },
    create_batch(capacity) {
      state.created++; state.active++; state.peak = Math.max(state.peak, state.active);
      let closed = false;
      return {input_size: 2, output_size: 2, max_rows: capacity, estimated_bytes: 1024,
        run_into(input, output, rows) {
          assert(!closed);
          state.batchCalls++;
          settings.onBatch?.(input, state);
          for (let i = 0; i < rows; i++) output.set(row(input, i * 2), i * 2);
          if (settings.wrongOutput) output[0] += 1;
          if (settings.nonfiniteOutput) output[0] = NaN;
        },
        free() { if (!closed) { closed = true; state.active--; } },
      };
    },
    free() { state.freed++; },
  };
  return {prepared, state};
}

test('real Darwin retains the requested baseline128 with fresh no-gain confirmation', async t => {
  clock(t);
  const before = await archives();
  const {prepared, state} = synthetic();
  const input = inputs(), expected = new Float64Array(input.length);
  for (let i = 0; i < input.length; i += 2) expected.set([input[i] + input[i + 1], input[i] - input[i + 1]], i);
  const receipt = await tuneBatchRows({prepared, inputs: input, modelId: 'synthetic-labelled-model', samples: 5});
  assert.equal(receipt.schema, 'ruv-fann.batch-tuning/1');
  assert.equal(receipt.status, 'passed');
  assert.deepEqual(receipt.engine, {package: '@metaharness/darwin', version: '0.13.0', algorithm: 'evolveNumeric'});
  assert.equal(receipt.options.baselineBatchRows, 128);
  assert.equal(receipt.baseline.genome.batchRows, 128);
  assert.equal(receipt.confirmation.verified, true);
  assert.equal(receipt.confirmation.status, 'not_improved');
  assert.equal(receipt.confirmation.faster, false);
  assert.deepEqual(receipt.recommendation, {batchRows: 128});
  assert.equal(receipt.context.inputSha256, tensorHash(input));
  assert.equal(receipt.context.outputSha256, tensorHash(expected));
  assert.match(receipt.modelIdScope, /not.*attestation/);
  assert.equal(receipt.trialBudget.dispatchCount, 2);
  const unique = new Set(receipt.trials.map(value => value.genome.batchRows)).size;
  assert.equal(receipt.trialBudget.actualMeasurementCount, unique);
  assert.equal(receipt.trialBudget.cacheHits, receipt.trials.length - unique);
  assert.equal(receipt.trialBudget.concurrency, 1);
  assert.equal(receipt.confirmation.baselineSamplesNs.length, 5);
  assert.equal(receipt.confirmation.candidateSamplesNs.length, 5);
  assert.equal(receipt.confirmation.maximumAbsoluteError, 0);
  assert.equal(receipt.temporaryArchiveRemoved, true);
  assert.equal(receipt.modelCalls, 0);
  assert.equal(receipt.sourceApplied, false);
  assert.equal(state.active, 0); assert.equal(state.peak, 1); assert.equal(state.freed, 0);
  assert.deepEqual(await archives(), before);
});

test('real Darwin dispatch and measurement budgets cover growing parent population', async t => {
  clock(t);
  const {prepared, state} = synthetic();
  const receipt = await tuneBatchRows({prepared, inputs: inputs(), modelId: 'bounded',
    baselineBatchRows: 1, generations: 3, children: 2, samples: 5});
  const budget = receipt.trialBudget;
  assert.equal(budget.worstCaseDispatches, 11);
  assert.equal(budget.dispatchCount, receipt.trials.length);
  assert(budget.dispatchCount <= 11);
  assert.equal(budget.actualMeasurementCount + budget.cacheHits, budget.dispatchCount);
  assert.equal(budget.actualMeasurementCount, new Set(receipt.trials.map(value => value.genome.batchRows)).size);
  assert(budget.cacheHits > 0, 'seeded real Darwin run must exercise cached capacities');
  assert.equal(receipt.baseline.genome.batchRows, 1);
  assert(receipt.trials.every(value => Number.isFinite(value.score.primary) && value.score.primary > 0));
  assert(receipt.trials.every(value => value.score.raw.maximumAbsoluteError === 0));
  assert.equal(receipt.confirmation.workloadSha256, receipt.context.workloadSha256);
  assert.equal(state.active, 0); assert.equal(state.peak, 1); assert.equal(state.freed, 0);
});

test('inputs are synchronously snapshotted before caller mutation during awaits', async t => {
  clock(t);
  const {prepared} = synthetic();
  const input = inputs(), original = tensorHash(input);
  const pending = tuneBatchRows({prepared, inputs: input, modelId: 'snapshot', samples: 5});
  input.fill(99);
  const receipt = await pending;
  assert.equal(receipt.context.inputSha256, original);
  assert.notEqual(receipt.context.inputSha256, tensorHash(input));
});

test('intrinsic input length ignores an own throwing length getter', async t => {
  clock(t);
  const {prepared} = synthetic();
  const input = inputs();
  Object.defineProperty(input, 'length', {get() { throw new Error('shadow length was invoked'); }});
  const receipt = await tuneBatchRows({prepared, inputs: input, modelId: 'intrinsic length', samples: 5});
  assert.equal(receipt.context.totalRows, 128);
});

test('strict options reject fabricated baselines, budgets, identities and evaluator injection', async () => {
  const {prepared, state} = synthetic();
  for (const change of [
    {baselineBatchRows: 0}, {baselineBatchRows: 129}, {baselineBatchRows: '128'},
    {generations: 4}, {children: 3}, {samples: 4}, {samples: 21}, {seed: -1},
    {timeoutMs: NaN}, {timeoutMs: Infinity}, {timeoutMs: 99}, {timeoutMs: 120001},
    {modelId: ''}, {modelId: 'x'.repeat(257)}, {engine: () => {}}, {evaluator: {}},
  ]) await assert.rejects(tuneBatchRows({prepared, inputs: inputs(), modelId: 'invalid', ...change}));
  assert.equal(state.created, 0); assert.equal(state.freed, 0);
});

test('invalid, oversized-row, nonfinite, spoofed, shared and detached inputs reject before batches', async () => {
  const {prepared, state} = synthetic();
  const nonfinite = inputs(); nonfinite[3] = NaN;
  const spoofed = new Float32Array(256); Object.setPrototypeOf(spoofed, Float64Array.prototype);
  const detached = inputs(); structuredClone(detached.buffer, {transfer: [detached.buffer]});
  for (const input of [[], new Float64Array(254), new Float64Array(2050), nonfinite,
    spoofed, detached, new Float64Array(new SharedArrayBuffer(2048))])
    await assert.rejects(tuneBatchRows({prepared, inputs: input, modelId: 'bad input'}));
  assert.equal(state.created, 0); assert.equal(state.freed, 0);
});

test('working-buffer cap includes maximum encoder staging before allocation or reference inference', async () => {
  let called = false;
  const prepared = {input_size: 1, output_size: 17000,
    create_batch() { called = true; throw new Error('must not allocate batch'); },
    run() { called = true; throw new Error('must not allocate reference'); }};
  await assert.rejects(tuneBatchRows({prepared, inputs: new Float64Array(128), modelId: 'cap'}), /64 MiB/);
  assert.equal(called, false);
});

test('backend mutation of the private input snapshot fails before creating any batch', async () => {
  const {prepared, state} = synthetic({onReference(input) { input[0] = 99; }});
  await assert.rejects(tuneBatchRows({prepared, inputs: inputs(), modelId: 'mutation'}), /changed tuning inputs/);
  assert.equal(state.created, 0); assert.equal(state.freed, 0);
});

test('incorrect/nonfinite batched output rejects and frees every encoder and temporary archive', async t => {
  clock(t);
  const before = await archives();
  for (const settings of [{wrongOutput: true}, {nonfiniteOutput: true},
    {onBatch() { throw new Error('backend failure'); }}]) {
    const {prepared, state} = synthetic(settings);
    await assert.rejects(tuneBatchRows({prepared, inputs: inputs(), modelId: 'failure', samples: 5}));
    assert.equal(state.active, 0); assert.equal(state.peak, 1); assert.equal(state.freed, 0);
  }
  assert.deepEqual(await archives(), before);
});

test('reference outputs are rechecked after timing to reject changed model behavior', async t => {
  clock(t);
  const before = await archives();
  const {prepared, state} = synthetic({corruptReferenceAfterBatch: true});
  await assert.rejects(tuneBatchRows({prepared, inputs: inputs(), modelId: 'changed model', samples: 5}));
  assert.equal(state.active, 0); assert.equal(state.freed, 0);
  assert.deepEqual(await archives(), before);
});

test('cooperative deadline failure cleans the owned batch and archive', async t => {
  const timer = clock(t), before = await archives();
  const {prepared, state} = synthetic({onBatch() { timer.now = 100000; }});
  await assert.rejects(tuneBatchRows({prepared, inputs: inputs(), modelId: 'deadline', samples: 5,
    timeoutMs: 100}), /deadline/);
  assert.equal(state.active, 0); assert.equal(state.freed, 0);
  assert.deepEqual(await archives(), before);
});

test('paired timing confidence rejects malformed samples and preserves equal timing', () => {
  assert.deepEqual(pairedRatio([1, 1, 1, 1, 1], [1, 1, 1, 1, 1]), {meanSpeedRatio: 1, ci95: [1, 1]});
  for (const candidate of [[1], [1, 1, 0, 1, 1], [1, 1, Infinity, 1, 1]])
    assert.throws(() => pairedRatio([1, 1, 1, 1, 1], candidate));
});
