#!/usr/bin/env node
/** Actual extracted-archive consumer of the public WASM training facade.
 * This path needs only Node builtins. SQLite/CLI dependency installation is
 * outside this check; no source node_modules graph is linked into the package.
 */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { externalEvidencePath, sourceSnapshot, repo, sha } from '../../../integrations/ruv-fann/test/e2e/fixture.mjs';

if (process.argv.length !== 4 || process.argv[2] !== '--output' || !process.env.RUV_FANN_CANDLE_FIXTURE) {
  throw new Error('Set RUV_FANN_CANDLE_FIXTURE and use --output NEW_EXTERNAL_DIRECTORY');
}
const output = await externalEvidencePath(process.argv[3]);
await mkdir(output, { mode: 0o700 });
const before = await sourceSnapshot(), npmRoot = join(repo, 'ruv-swarm/npm');
execFileSync(process.execPath, ['scripts/validate-package.js'], { cwd: npmRoot, timeout: 30000, stdio: 'pipe' });
const [packed] = JSON.parse(execFileSync('npm', ['pack', '--ignore-scripts', '--json', '--pack-destination', output],
  { cwd: npmRoot, encoding: 'utf8', timeout: 60000, maxBuffer: 2 * 1024 * 1024 }));
assert.match(packed.filename, /^ruv-swarm-[0-9.]+\.tgz$/);
const tarball = join(output, packed.filename);
assert.ok((await readFile(tarball)).length <= 8 * 1024 * 1024);
const entries = execFileSync('tar', ['-tzf', tarball], { encoding: 'utf8', timeout: 10000, maxBuffer: 2 * 1024 * 1024 }).trim().split('\n');
assert.ok(entries.length < 2048);
for (const entry of entries) {
  assert.ok(entry.startsWith('package/') && !entry.split('/').includes('..'));
  assert.ok(!entry.split('/').some(part => ['node_modules', '.git', '.swarm', '.claude-flow'].includes(part) || part.startsWith('.env')));
}
const listing = execFileSync('tar', ['-tvzf', tarball], { encoding: 'utf8', timeout: 10000, maxBuffer: 2 * 1024 * 1024 });
assert.ok(listing.split('\n').filter(Boolean).every(line => line.startsWith('-') || line.startsWith('d')));
execFileSync('tar', ['-xzf', tarball, '-C', output, '--no-same-owner', '--no-same-permissions'], { timeout: 10000 });
const extracted = join(output, 'package');
execFileSync(process.execPath, ['scripts/validate-package.js', '--artifact-only'], { cwd: extracted, timeout: 30000, stdio: 'pipe' });
await mkdir(join(output, 'node_modules'));
await symlink(extracted, join(output, 'node_modules/ruv-swarm'), 'dir');
const source = `import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { RuvSwarm } from 'ruv-swarm';
const f = JSON.parse(await readFile(process.env.RUV_FANN_CANDLE_FIXTURE, 'utf8'));
assert.match(f.oracle, /candle-core\\/candle-nn 0\\.11\\.1 CPU f64/);
function close(actual, expected) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Number.isFinite(actual[i]) && Number.isFinite(expected[i]));
    assert.ok(Math.abs(actual[i] - expected[i]) <= 1e-12 * (1 + Math.abs(expected[i])));
  }
}
const cases = [];
for (const useSIMD of [false, true]) {
  const runtime = await RuvSwarm.initialize({ useSIMD });
  assert.equal(JSON.parse(runtime._bindings.get_features()).prepared_batch_simd, useSIMD);
  for (const optimizer of ['sgd', 'adamw']) {
    const model = runtime.createTrainableNetwork(f.dense_json, 1);
    if (optimizer === 'adamw') {
      const c = f.adamw_config;
      model.configure_adamw(c.learning_rate, c.beta1, c.beta2, c.epsilon, c.weight_decay);
    }
    assert.equal(f[optimizer + '_steps'].length, 3);
    for (const step of f[optimizer + '_steps']) {
      close([model.mse_backward(Float64Array.from(f.input), Float64Array.from(f.training_target), 1)], [step.loss]);
      close(model.predictions(), step.predictions);
      close(model.parameter_gradients(), step.parameter_gradients);
      close(model.input_gradients(), step.input_gradients);
      if (optimizer === 'sgd') model.sgd_step(f.sgd_learning_rate); else model.adamw_step();
      const parameters = JSON.parse(model.to_dense_layers_json()).layers.flatMap(l => [...l.weights, ...(l.bias ?? [])]);
      close(parameters, step.parameters);
    }
    const json = model.to_dense_layers_json(), prepared = model.prepare();
    const input = Float64Array.from(f.input), expected = Array.from(prepared.run(input));
    model.free();
    assert.deepEqual(Array.from(prepared.run(input)), expected);
    const restored = runtime.createDensePreparedNetwork(json);
    assert.deepEqual(Array.from(restored.run(input)), expected);
    const batch = prepared.create_batch(3);
    prepared.free(); restored.free();
    const batchOutput = new Float64Array(expected.length * 3);
    batch.run_into(Float64Array.from([...f.input, ...f.input, ...f.input]), batchOutput, 3);
    assert.deepEqual(Array.from(batchOutput), [...expected, ...expected, ...expected]);
    batch.free();
    cases.push({ useSIMD, optimizer, steps: 3, gradients: true, frozenLifetime: true, checkpoint: true, batchLifetime: true });
  }
}
console.log(JSON.stringify({ status: 'passed', cases }));
`;
await writeFile(join(output, 'consumer.mjs'), source, { flag: 'wx' });
const result = JSON.parse(execFileSync(process.execPath, ['consumer.mjs'], { cwd: output, encoding: 'utf8', timeout: 30000,
  maxBuffer: 1024 * 1024, env: { ...process.env } }));
assert.equal(result.status, 'passed');
assert.equal(result.cases.length, 4);
assert.deepEqual(await sourceSnapshot(), before);
const receipt = { schema: 'ruv-fann.wasm-training-package/1', status: 'passed', node: process.version,
  version: packed.version, tarballSha256: sha(await readFile(tarball)), entries: entries.length,
  actualCandle: true, actualCoreAndSimd: true, actualBarePackageImport: true,
  dependencyScope: 'Only Node builtins and extracted package files; no dependency graph installed or linked. CLI/SQLite are outside this check.',
  fixtureSha256: sha(await readFile(process.env.RUV_FANN_CANDLE_FIXTURE)),
  inputsBefore: before, inputsAfter: before, unchanged: true, cases: result.cases };
await writeFile(join(output, 'receipt.json'), `${JSON.stringify(receipt, null, 2) }\n`, { flag: 'wx' });
console.log(JSON.stringify({ status: 'passed', receipt: join(output, 'receipt.json'), cases: result.cases.length }));
