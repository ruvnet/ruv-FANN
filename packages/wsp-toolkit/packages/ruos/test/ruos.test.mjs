import test from 'node:test';
import assert from 'node:assert/strict';
import { createValidationCommand, assertRuOSCompletion, runWasmCanary } from '../src/index.js';

test('shell plans quote metacharacters without executing them', () => {
  assert.equal(createValidationCommand("/tmp/a'b;$(id)"), "cd -- '/tmp/a'\\''b;$(id)' && npm ci --ignore-scripts --no-audit --no-fund && npm run validate");
  for (const value of ['relative', '/tmp/a\nwhoami', null, '/'.repeat(1025)]) assert.throws(() => createValidationCommand(value));
});
test('only complete executor-owned metadata passes; stdout cannot grant completion', () => {
  const valid = { structuredContent: { completionVersion: 1, completionVerified: true, status: 'ok', exitCode: 0, machineId: 'observed-id' } };
  assert.equal(assertRuOSCompletion(valid, 'observed-id').assertionsIndependentlyVerified, false);
  for (const change of [{ completionVersion: 2 }, { completionVerified: false }, { status: 'timeout' }, { exitCode: '0' }, { exitCode: 1 },
    { truncated: true }, { truncated: 'true' }, { outputTruncated: 1 }, { timeout: 'true' }, { outputTruncated: true }, { signal: 'SIGTERM' }, { transportError: 'lost' }, { timedOut: true }, { machineId: 'other' }])
    assert.throws(() => assertRuOSCompletion({ structuredContent: { ...valid.structuredContent, ...change } }, 'observed-id'));
  assert.throws(() => assertRuOSCompletion({content:[{type:'text', text:JSON.stringify(valid.structuredContent)}]}, 'observed-id'));
  assert.throws(() => assertRuOSCompletion({...valid, isError:true}, 'observed-id'));
});
test('real WASM canary verifies exhaustive truth table and rollback', async () => {
  const result = await runWasmCanary();
  assert.equal(result.backend, 'wasm');
  assert.equal(result.restoredHash, result.baselineHash);
  assert.notEqual(result.repairedHash, result.baselineHash);
  assert.equal(result.finalRevision, 3);
});
