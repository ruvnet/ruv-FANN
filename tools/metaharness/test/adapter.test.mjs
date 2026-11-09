/** Synthetic boundary evidence tests. Real production WASM tests live separately. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, symlink, rm, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { checkedConfirmation, checkedGenome, checkedScore, runEvolution } from '../src/evolution.mjs';
import { assertNoScoreModel, boundedProcess, childEnvironment, fileBytes, finiteJson, parseArgs, validateOptions } from '../src/guards.mjs';

const options = { variant: 'core', model: 'tiny', totalRows: 128, samples: 5,
  generations: 2, children: 2, seed: 7, timeoutMs: 10000 };
const context = { variant: 'core', model: 'tiny', totalRows: 128,
  ...Object.fromEntries(['workloadSha256', 'outputSha256', 'inputSha256', 'weightsSha256',
    'sourceSha256', 'binarySha256', 'glueSha256', 'receiptSha256'].map(key => [key, 'a'.repeat(64)])) };
const card = (id, rows, more = {}) => ({ variantId: id, primary: rows * 100, regressed: false,
  noopRate: 0, costPerWin: 0, raw: { ...context, batchRows: rows, samplesNs: [10, 11, 10, 11, 10],
    maximumAbsoluteError: 0, meanNs: 10.4, medianNs: 10, rowsPerSecond: rows * 100, estimatedBytes: 256,
    cacheHit: false, observationId: id }, ...more });
function confirmation(baselineRows = 1, winnerRows = 2) {
  const faster = winnerRows !== baselineRows;
  return { status: faster ? 'passed' : 'not_improved', verified: true, faster,
    baselineBatchRows: baselineRows, candidateBatchRows: winnerRows, meanSpeedRatio: faster ? 2 : 1,
    ci95: faster ? [1.8, 2.2] : [0.9, 1.1], minimumImprovement: 0.05, maximumAbsoluteError: 0,
    baselineSamplesNs: [20, 20, 20, 20, 20], candidateSamplesNs: Array(5).fill(faster ? 10 : 20),
    workloadSha256: context.workloadSha256, scope: 'Synthetic test fixture, never benchmark evidence.' };
}
async function temporary(run) {
  const directory = await mkdtemp(join(tmpdir(), 'metaharness-adapter-'));
  try { return await run(directory); } finally { await rm(directory, { recursive: true, force: true }); }
}
function fixtureEvaluator({ regressed = false, invalid = false } = {}) {
  const cache = new Map(); let closed = false, verified = false;
  return { context,
    async evaluate(genome, id) {
      if (cache.has(genome.batchRows)) {
        const score = structuredClone(cache.get(genome.batchRows));
        return { ...score, variantId: id, raw: { ...score.raw, cacheHit: true } };
      }
      const score = card(id, genome.batchRows);
      if (regressed && genome.batchRows > 1) { score.regressed = true; score.primary = score.raw.rowsPerSecond = 999999; }
      if (invalid) score.raw.maximumAbsoluteError = 1;
      cache.set(genome.batchRows, structuredClone(score)); return score;
    },
    async confirmWinner({ baselineGenome, winnerGenome }) { return confirmation(baselineGenome.batchRows, winnerGenome.batchRows); },
    async verifyInputs() { verified = true; return { status: 'passed', sourceSha256: context.sourceSha256 }; },
    close() { closed = true; }, get state() { return { closed, verified }; } };
}

test('strict flags and integer limits reject unknown, duplicate, NaN and overflow values', () => {
  for (const args of [['plan', '--wat', '1'], ['plan', '--seed', '1', '--seed', '2'],
    ['optimize'], ['plan', '--output', '/tmp/no'], ['plan', '--samples', 'NaN'],
    ['plan', '--seed', '4294967296'], ['plan', '--children', '2oops'], ['plan', '--generations', '4']])
    assert.throws(() => parseArgs(args));
  assert.equal(parseArgs(['plan', '--generations', '3', '--children', '2']).generations, 3);
  assert.equal(parseArgs(['plan']).baselineBatchRows, 1);
  assert.equal(parseArgs(['plan', '--baseline-batch-rows', '128']).baselineBatchRows, 128);
  for (const value of ['0', '129', 'NaN', '1.5'])
    assert.throws(() => parseArgs(['plan', '--baseline-batch-rows', value]));
  assert.throws(() => validateOptions({ samples: Infinity }));
});

test('genome can vary only finite integer batchRows in the approved range', () => {
  for (const value of [{ batchRows: 0 }, { batchRows: 129 }, { batchRows: NaN }, { batchRows: 1.1 },
    { batchRows: 2, source: 'edit' }, { batchRows: '2' }]) assert.throws(() => checkedGenome(value));
  assert.deepEqual(checkedGenome({ batchRows: 128 }), { batchRows: 128 });
});

test('score rejects nonfinite, numerical errors and mismatched frozen workload identities', () => {
  assert.equal(checkedScore(card('v', 2), 'v', { batchRows: 2 }, 5, context).primary, 200);
  for (const change of [score => { score.primary = Infinity; }, score => { score.raw.samplesNs[0] = NaN; },
    score => { score.raw.maximumAbsoluteError = 1; }, score => { score.raw.workloadSha256 = 'b'.repeat(64); },
    score => { score.raw.outputSha256 = 'b'.repeat(64); }, score => { score.raw.model = 'large'; },
    score => { score.raw.rowsPerSecond = 900; }, score => { score.evaluatorError = 'failed'; }]) {
    const score = card('v', 2); change(score);
    assert.throws(() => checkedScore(score, 'v', { batchRows: 2 }, 5, context));
  }
});

test('confirmation rejects bypassed confidence policy, wrong samples and inconsistent ratios', () => {
  assert.equal(checkedConfirmation(confirmation(), { batchRows: 1 }, { batchRows: 2 }, context, 5).faster, true);
  for (const change of [value => { value.ci95 = [1.01, 1.03]; }, value => { value.workloadSha256 = 'b'.repeat(64); },
    value => { value.baselineSamplesNs.pop(); }, value => { value.meanSpeedRatio = 3; },
    value => { value.maximumAbsoluteError = 1; }, value => { value.verified = false; }]) {
    const value = confirmation(); change(value);
    assert.throws(() => checkedConfirmation(value, { batchRows: 1 }, { batchRows: 2 }, context, 5));
  }
});

test('actual installed Darwin engine stays within two-parent growth and records cached candidates', () => temporary(async output => {
  const evaluator = fixtureEvaluator();
  const result = await runEvolution(options, output, evaluator);
  assert.equal(result.status, 'passed'); assert.equal(result.modelCalls, 0); assert.equal(result.sourceApplied, false);
  assert.equal(result.trialBudget.worstCaseDispatches, 7);
  assert.equal(result.trials.length, result.trialBudget.dispatchCount);
  assert.ok(result.trialBudget.dispatchCount <= 7);
  assert.equal(result.trialBudget.actualMeasurementCount + result.trialBudget.cacheHits, result.trials.length);
  assert.equal(result.inputVerification, 'passed');
  assert.deepEqual(evaluator.state, { closed: true, verified: true });
  assert.ok(JSON.parse(await readFile(join(output, 'darwin/archive.json'))).length > 1);
}));

test('regressed upstream archive best is rejected and verified baseline retained', () => temporary(async output => {
  const evaluator = fixtureEvaluator({ regressed: true });
  const result = await runEvolution({ ...options, generations: 1 }, output, evaluator);
  assert.equal(result.winner.genome.batchRows, 1);
  assert.equal(result.winnerRejected.reason, 'regressed archive winner');
  assert.equal(result.confirmation.status, 'not_improved');
  assert.deepEqual(result.recommendation, { batchRows: 1 });
}));

test('real Darwin with synthetic scores retains configured baseline128 when a candidate is unconfirmed', () => temporary(async output => {
  const evaluator = fixtureEvaluator();
  const evaluate = evaluator.evaluate.bind(evaluator);
  evaluator.evaluate = async (genome, id) => {
    const score = await evaluate(genome, id);
    score.primary = score.raw.rowsPerSecond = genome.batchRows === 128 ? 100 : 200;
    return score;
  };
  evaluator.confirmWinner = async ({ baselineGenome, winnerGenome }) => ({
    ...confirmation(baselineGenome.batchRows, winnerGenome.batchRows), status: 'not_improved', faster: false,
    meanSpeedRatio: 1, ci95: [0.98, 1.02], baselineSamplesNs: Array(5).fill(20), candidateSamplesNs: Array(5).fill(20) });
  const result = await runEvolution({ ...options, baselineBatchRows: 128 }, output, evaluator);
  assert.equal(result.baseline.genome.batchRows, 128);
  assert.notEqual(result.winner.genome.batchRows, 128);
  assert.equal(result.confirmation.baselineBatchRows, 128);
  assert.equal(result.confirmation.status, 'not_improved');
  assert.deepEqual(result.recommendation, { batchRows: 128 });
  assert.deepEqual(evaluator.state, { closed: true, verified: true });
}));

test('incorrect measurements fail without optimistic fallback and always close evaluator', () => temporary(async output => {
  const evaluator = fixtureEvaluator({ invalid: true });
  await assert.rejects(runEvolution(options, output, evaluator), /correctness/);
  assert.equal(evaluator.state.closed, true); assert.equal(evaluator.state.verified, false);
}));

test('a preexisting Darwin archive is never resumed', () => temporary(async output => {
  await mkdir(join(output, 'darwin'));
  const evaluator = fixtureEvaluator();
  await assert.rejects(runEvolution(options, output, evaluator), /EEXIST/);
  assert.equal(evaluator.state.closed, true);
}));

test('independent dispatch breaker does not rely on upstream generation implementation', () => temporary(async output => {
  const evaluator = fixtureEvaluator();
  const hostileEngine = async config => {
    for (let index = 0; index < 20; index++) await config.evaluator.evaluate({ batchRows: index + 1 }, 'synthetic-' + index);
  };
  await assert.rejects(runEvolution({ ...options, generations: 1, children: 1 }, output, evaluator, hostileEngine), /dispatch/);
  assert.equal(evaluator.state.closed, true);
}));

test('special and symlink inputs fail before file reads', () => temporary(async root => {
  await writeFile(join(root, 'ordinary'), 'safe');
  await symlink(join(root, 'ordinary'), join(root, 'linked'));
  await assert.rejects(fileBytes(root, 'linked'), /symlink/);
  await mkdir(join(root, 'directory'));
  await assert.rejects(fileBytes(root, 'directory'), /non-file/);
  await assert.rejects(fileBytes(root, '../escape'), /escapes/);
}));

test('child process omits provider credentials and Node injection variables', () => temporary(async cwd => {
  const output = await boundedProcess(['--input-type=module', '-e', 'console.log(JSON.stringify(process.env))'],
    { cwd, timeoutMs: 3000 });
  const value = JSON.parse(output.stdout);
  assert.deepEqual(Object.keys(value).sort(), Object.keys(childEnvironment()).sort());
  assert.equal(value.ANTHROPIC_API_KEY, undefined); assert.equal(value.NODE_OPTIONS, undefined);
}));

test('process-group deadline and output cap fail closed', () => temporary(async cwd => {
  await assert.rejects(boundedProcess(['-e', 'while(true){}'], { cwd, timeoutMs: 100 }), /timeout/);
  await assert.rejects(boundedProcess(['-e', 'console.log("x".repeat(100000))'],
    { cwd, timeoutMs: 3000, maxBytes: 1024 }), /output_limit/);
  await assert.rejects(boundedProcess(['-e', 'process.exit(7)'], { cwd, timeoutMs: 3000 }), /7/);
}));

test('non-JSON and unbounded evidence cannot be serialized as successful measurements', () => {
  assert.throws(() => finiteJson({ value: Infinity }));
  assert.throws(() => finiteJson({ execute() {} }));
  assert.throws(() => finiteJson({ raw: new Float64Array(2) }));
});

test('score rejects an ancestor optional embedding model without loading its code', () => temporary(async root => {
  const analyzer = join(root, 'node_modules/metaharness/dist/analyze-repo.js');
  await mkdir(join(root, 'node_modules/metaharness/dist'), { recursive: true });
  await writeFile(analyzer, '// Resolution fixture only.\n');
  const model = join(root, 'node_modules/@ruvector/ruvllm');
  await mkdir(model, { recursive: true });
  await writeFile(join(model, 'package.json'), JSON.stringify({ name: '@ruvector/ruvllm', main: 'index.js' }));
  await writeFile(join(model, 'index.js'), 'throw new Error("MODEL CODE MUST NEVER BE LOADED");\n');
  assert.throws(() => assertNoScoreModel(analyzer), /resolvable optional embedding model/);
}));
