import test from 'node:test';
import assert from 'node:assert/strict';
import { createLab, benchmarkRuntime } from '../src/engine.js';
import { evaluateRequest } from '../src/api.js';

for (const simd of [false, true]) for (const name of ['robot', 'room']) {
  test(`${name} ${simd ? 'SIMD' : 'scalar'} rejects shortcuts, promotes a gated repair, restores original hash`, async () => {
    const lab = await createLab(name, simd), baseline = lab.snapshot();
    assert.throws(() => lab.promote(), /gates/);
    const bad = await lab.evaluate(lab.scenario.shortcut);
    assert.equal(bad.regression.suites[0].evaluation.accepted, true);
    assert.equal(bad.accepted, false); assert.equal(lab.snapshot().hash, baseline.hash);
    assert.throws(() => lab.promote(), /gates/);
    const good = await lab.evaluate(lab.scenario.good);
    assert.equal(good.accepted, true); assert.equal(lab.snapshot().hash, baseline.hash);
    lab.invalidate(); assert.throws(() => lab.promote(), /gates/);
    await lab.evaluate(lab.scenario.good);
    assert.equal(lab.promote().revision, 2);
    const restored = lab.rollback(); assert.equal(restored.revision, 3); assert.equal(restored.hash, baseline.hash);
    assert.throws(() => lab.workspace.promote('stale', 1), /revision/i);
  });
}
test('development-only robot search finds an edit that fails independent evaluation', async () => {
  const lab = await createLab('robot'), found = await lab.search();
  assert.ok(found.result.evaluations <= 24); assert.equal(found.result.bestScore, 0);
  assert.equal((await lab.evaluate(found.weights)).accepted, false);
});
test('scalar and SIMD benchmark predictions agree with arithmetic reference', async () => {
  for (const simd of [false, true]) assert.ok((await benchmarkRuntime(simd)).maxError < 1e-12);
});
const request = (body, headers = { 'Content-Type': 'application/json' }) => new Request('https://lab.test/api/evaluate', { method: 'POST', headers, body: JSON.stringify(body) });
test('public API enforces fixed policies, bounded inputs, methods, bytes and encoding', async () => {
  const valid = { scenario: 'robot', weights: [1, -1], baseWeights: [0, 0] };
  const success = await evaluateRequest(request(valid)); assert.equal(success.status, 200);
  assert.equal((await success.json()).accepted, true);
  const rejected = await evaluateRequest(request({ ...valid, weights: [1, 0] }));
  assert.equal(rejected.status, 200); assert.equal((await rejected.json()).accepted, false);
  for (const invalid of [{ ...valid, policy: { maxMSE: 100 } }, { ...valid, scenario: '__proto__' }, { ...valid, weights: [1, 3] }, { ...valid, baseWeights: null }, { ...valid, weights: ['1', -1] }]) {
    assert.equal((await evaluateRequest(request(invalid))).status, 422);
  }
  assert.equal((await evaluateRequest(new Request('https://lab.test/api/evaluate'))).status, 405);
  assert.equal((await evaluateRequest(request(valid, { 'Content-Type': 'text/plain' }))).status, 415);
  assert.equal((await evaluateRequest(request({ ...valid, padding: 'x'.repeat(1024) }))).status, 413);
  assert.equal((await evaluateRequest(new Request('https://lab.test/api/evaluate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: Uint8Array.of(255) }))).status, 400);
});
