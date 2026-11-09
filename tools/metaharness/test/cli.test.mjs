/** Real CLI/projection checks; these do not run WASM benchmarks or model providers. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm, writeFile, symlink } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import { digest, fileBytes, inputSnapshot, packagePins } from '../src/guards.mjs';

const repo = resolve(fileURLToPath(new URL('../../..', import.meta.url)));
const cli = resolve(fileURLToPath(new URL('../src/cli.mjs', import.meta.url)));
const invoke = args => spawnSync(process.execPath, [cli, ...args], { cwd: repo,
  encoding: 'utf8', timeout: 15000, maxBuffer: 1024 * 1024 });
const json = result => { assert.equal(result.error, undefined); return JSON.parse(result.stdout); };
async function temporary(run) {
  const dir = await mkdtemp(join(tmpdir(), 'metaharness-cli-'));
  try { return await run(dir); } finally { await rm(dir, { recursive: true, force: true }); }
}

test('actual private registry packages match exact lock identities and integrity', async () => {
  const pins = await packagePins();
  assert.equal(pins.metaharness.version, '0.4.21'); assert.equal(pins.darwin.version, '0.13.0');
  assert.match(pins.darwin.npmIntegrity, /^sha512-/);
  assert.equal(pins.referenceCommit, 'b27878cf39d3ada6b2c4d1bcc3e9258a70c6cfd6');
});

test('real plan is read-only and captures both WASM artifacts, ten original locks and all tool sources', async () => {
  const before = await inputSnapshot(repo);
  const result = invoke(['plan', '--repo', repo]);
  const report = json(result); assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.equal(report.status, 'planned'); assert.equal(report.readOnly, true);
  assert.equal(report.options.baselineBatchRows, 1);
  assert.equal(report.sourceApplied, false); assert.equal(report.modelCalls, 0);
  const originalLocks = Object.keys(before).filter(key => key.startsWith('repo/') && /(?:Cargo\.lock|package-lock\.json)$/.test(key));
  assert.equal(originalLocks.length, 10);
  for (const name of ['ruv_swarm_wasm_bg.wasm', 'ruv_swarm_simd.wasm', 'ruv_swarm_wasm.js'])
    assert.match(report.inputsBefore['repo/ruv-swarm/npm/wasm/' + name], /^[a-f0-9]{64}$/);
  for (const name of ['cli', 'guards', 'evolution', 'wasm-evaluator'])
    assert.match(report.inputsBefore[`tool/src/${name}.mjs`], /^[a-f0-9]{64}$/);
  const after = await inputSnapshot(repo);
  for (const name of originalLocks) assert.equal(after[name], before[name]);
});

test('CLI rejects unknown/duplicate flags, nonfinite or excessive budgets as JSON', () => {
  for (const args of [['plan', '--oops', '1'], ['plan', '--samples', 'Infinity'],
    ['plan', '--generations', '100'], ['plan', '--variant', 'provider'],
    ['plan', '--seed', '1', '--seed', '2'], ['optimize', '--repo', repo]]) {
    const result = invoke(args); assert.equal(result.status, 2); assert.equal(json(result).status, 'invalid_input');
  }
});

test('CLI never overwrites an existing external receipt', () => temporary(async dir => {
  const marker = join(dir, 'receipt.json'); await writeFile(marker, 'immutable original receipt\n');
  const result = invoke(['score', '--repo', repo, '--output', dir]);
  assert.equal(result.status, 2); assert.match(json(result).error, /already exists/);
  assert.equal(await readFile(marker, 'utf8'), 'immutable original receipt\n');
}));

test('upstream MetaHarness scores only the bounded public projection without executing repo source', () => temporary(async dir => {
  const output = join(dir, "score $(touch NEVER); 'literal'");
  const before = await inputSnapshot(repo);
  const result = invoke(['score', '--repo', repo, '--output', output, '--timeout-ms', '5000']);
  assert.equal(result.status, 0, result.stdout + result.stderr);
  const summary = json(result), receipt = JSON.parse(await readFile(summary.receiptPath));
  assert.equal(receipt.status, 'passed'); assert.equal(receipt.staticAnalysisOnly, true);
  assert.equal(receipt.scorecard.schema, 1); assert.equal(receipt.modelCalls, 0); assert.equal(receipt.sourceApplied, false);
  assert.deepEqual(receipt.inputsBefore, receipt.inputsAfter);
  const files = await readdir(join(output, 'score-input'));
  assert.ok(files.includes('Cargo.toml')); assert.ok(files.includes('README.md'));
  for (const name of ['.mcp.json', '.claude', '.swarm', '.env']) assert.ok(!files.includes(name));
  assert.deepEqual(before, await inputSnapshot(repo));
}));

test('regular-file boundary rejects FIFO and oversized input before opening it', () => temporary(async dir => {
  const fifo = join(dir, 'fifo');
  const created = spawnSync('mkfifo', [fifo], { encoding: 'utf8' });
  assert.equal(created.status, 0, created.stderr);
  await assert.rejects(fileBytes(dir, 'fifo'), /non-file/);
  await writeFile(join(dir, 'large'), Buffer.alloc(2048));
  await assert.rejects(fileBytes(dir, 'large', 1024), /at most/);
}));

test('symlink output ancestry is rejected without writing its target', () => temporary(async dir => {
  await symlink(dir, join(dir, 'alias'));
  const result = invoke(['score', '--repo', repo, '--output', join(dir, 'alias/output')]);
  assert.equal(result.status, 2); assert.match(json(result).error, /symlink ancestry/);
  assert.deepEqual(await readdir(dir), ['alias']);
}));

test('dependency entrypoint and actual CLI implementation hashes are included', async () => {
  const inputs = await inputSnapshot(repo);
  for (const name of ['bin', 'index', 'repo-scorecard', 'analyze-repo', 'constraints']) {
    const relative = `node_modules/metaharness/dist/${name}.js`;
    const toolRoot = resolve(fileURLToPath(new URL('..', import.meta.url)));
    assert.equal(inputs['tool/' + relative], digest(await fileBytes(toolRoot, relative)));
  }
});
