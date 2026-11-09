import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createEvaluator, pairedRatio} from '../src/wasm-evaluator.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');

test('paired confirmation preserves pairing and rejects invalid clock observations', () => {
  const ratio = pairedRatio([2, 4, 6, 8, 10], [1, 2, 3, 4, 5]);
  assert.equal(ratio.meanSpeedRatio, 2);
  assert.deepEqual(ratio.ci95, [2, 2]);
  for (const values of [[0, 1, 2, 3, 4], [NaN, 1, 2, 3, 4], [Infinity, 1, 2, 3, 4]]) {
    assert.throws(() => pairedRatio(values, [1, 2, 3, 4, 5]));
  }
});

for (const variant of ['core', 'simd']) {
  test(`${variant}: real WASM, odd tails, invariant work and deduplicated observations`, async () => {
    const evaluator = await createEvaluator({repo, variant, model: 'tiny', totalRows: 131, samples: 5});
    try {
      const a = await evaluator.evaluate({batchRows: 1}, 'baseline');
      const b = await evaluator.evaluate({batchRows: 17}, 'tail');
      const cached = await evaluator.evaluate({batchRows: 17}, 'repeat');
      assert.equal(a.raw.totalRows, b.raw.totalRows);
      assert.equal(b.raw.callsPerWorkload, 8);
      assert.equal(a.raw.outputSha256, b.raw.outputSha256);
      assert.equal(a.raw.workloadSha256, b.raw.workloadSha256);
      assert.equal(b.raw.maximumAbsoluteError, 0);
      assert.equal(cached.raw.cacheHit, true);
      assert.equal(cached.raw.observationId, 'tail');
      assert.equal(cached.variantId, 'repeat');
      assert.deepEqual(evaluator.stats, {measurementCount: 2, cacheHits: 1});
      cached.raw.samplesNs[0] = -1;
      assert.ok((await evaluator.evaluate({batchRows: 17}, 'another')).raw.samplesNs[0] > 0);
      const confirmation = await evaluator.confirmWinner({baselineGenome: {batchRows: 1}, winnerGenome: {batchRows: 17}, variantId: 'fresh'});
      assert.equal(confirmation.verified, true);
      assert.equal(confirmation.maximumAbsoluteError, 0);
      assert.equal(confirmation.baselineSamplesNs.length, 5);
      assert.ok(confirmation.ci95.every(Number.isFinite));
      assert.equal((await evaluator.verifyInputs()).status, 'passed');
      for (const genome of [{batchRows: NaN}, {batchRows: 1.5}, {batchRows: 0}, {batchRows: 129}, {batchRows: 2, shell: 1}, []]) {
        await assert.rejects(evaluator.evaluate(genome, 'bad'));
      }
    } finally { evaluator.close(); }
    await assert.rejects(evaluator.evaluate({batchRows: 2}, 'closed'));
  });
}

test('workload selection and dimensions fail before loading code', async () => {
  for (const options of [{model: 'unknown'}, {variant: 'gpu'}, {samples: 0}, {totalRows: 0}, {timeoutMs: Infinity}]) {
    await assert.rejects(createEvaluator({repo, ...options}));
  }
});
