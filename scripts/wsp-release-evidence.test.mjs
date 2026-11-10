import assert from 'node:assert/strict';
import { test } from 'node:test';
import { validateWorkbenchProof, validateSecurity, validateBenchmark } from './wsp-release-evidence.mjs';

test('release evidence rejects missing, false or misrepresented runtime/security proof', () => {
  const proof = { workbench: { traceBackend: 'wasm-simd', sensitivityMethod: 'central-finite-difference', ablationHashBound: true, analyticDerivative: true },
    security: { checks: 19, failed: 0, getterCalls: 0, status: 'passed' } };
  validateWorkbenchProof(proof);
  for (const mutate of [p => delete p.workbench, p => p.workbench.sensitivityMethod = 'autograd', p => p.workbench.traceBackend = 'mock',
    p => p.security.checks = 0, p => p.security.failed = 1, p => p.security.getterCalls = 1, p => p.workbench.analyticDerivative = false]) {
    const changed = structuredClone(proof); mutate(changed); assert.throws(() => validateWorkbenchProof(changed));
  }
});
test('dependency audit cannot authorize another archive, source, advisory count or incomplete result', () => {
  const report = { schema: 'ruv-fann-wsp.npm-security/1', status: 'passed', sourceSha: 'a'.repeat(40), tarballSha256: 'b'.repeat(64),
    vulnerabilities: { total: 0, info: 0, low: 0, moderate: 0, high: 0, critical: 0 } };
  validateSecurity(report, report.sourceSha, report.tarballSha256);
  assert.throws(() => validateSecurity(report, 'c'.repeat(40), report.tarballSha256));
  assert.throws(() => validateSecurity(report, report.sourceSha, 'd'.repeat(64)));
  for (const mutate of [p => p.vulnerabilities.high = 1, p => delete p.vulnerabilities.low, p => p.status = 'unknown']) {
    const changed = structuredClone(report); mutate(changed); assert.throws(() => validateSecurity(changed, report.sourceSha, report.tarballSha256));
  }
});
test('benchmark gate fails closed before accepting missing or unrelated source evidence', async () => {
  await assert.rejects(validateBenchmark({}, 'a'.repeat(40)));
  await assert.rejects(validateBenchmark({ schema: 'ruv-fann.workbench-benchmark/1', status: 'passed', sourceSha: 'b'.repeat(40) }, 'a'.repeat(40)));
  await assert.rejects(validateBenchmark({ schema: 'ruv-fann.workbench-benchmark/1', status: 'passed', sourceSha: 'a'.repeat(40), sourceWorktreeClean: false }, 'a'.repeat(40)));
});
