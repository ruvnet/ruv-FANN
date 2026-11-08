'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { execFileSync } = require('node:child_process');
const api = require('../dist/index.js');
const pkg = require('../package.json');

// This is an executable test double, not a production compiler fallback.
function fixture(t, body = `process.stdout.write(JSON.stringify({ args, source: fs.readFileSync(args[2], 'utf8') }));`, version = '0.2.0') {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'cuda-wrapper-test-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const binary = path.join(directory, 'compiler ;$(touch INJECTED).js');
  fs.writeFileSync(binary, `#!${process.execPath}\nconst fs = require('node:fs');\nconst args = process.argv.slice(2);\nif (args[0] === '--version') { process.stdout.write('cuda2rust ${version}\\n'); } else { ${body} }\n`, { mode: 0o700 });
  return { directory, binary };
}
const posix = { skip: process.platform === 'win32' ? 'Executable JS test fixtures require POSIX; real compiler integration is separate.' : false };

function rejectsCode(promise, code) {
  return assert.rejects(promise, error => error instanceof api.CudaWasmError && error.code === code);
}

test('CJS, ESM and package version expose only the documented source API', async () => {
  const esm = await import(pathToFileURL(path.resolve(__dirname, '../dist/index.mjs')).href);
  assert.equal(api.getVersion(), '2.0.0');
  assert.equal(api.getVersion(), pkg.version);
  assert.equal(esm.transpileCuda, api.transpileCuda);
  assert.equal(esm.default.transpileFile, api.transpileFile);
  assert.equal(api.analyzeKernel, undefined);
  assert.equal(api.benchmark, undefined);
  assert.equal(pkg.browser, undefined);
  assert.equal(pkg.exports['./wasm'], undefined);
});

test('source goes through a private file and fixed argv without a shell', posix, async t => {
  const { binary, directory } = fixture(t);
  const source = '__global__ void f() { /* $(`touch NO`) ; café */ }';
  for (const target of ['rust', 'wgsl']) {
    const result = await api.transpileCuda(source, { binary, target });
    const record = JSON.parse(result.code);
    assert.equal(record.source, source);
    assert.deepEqual(record.args.slice(0, 2), [target === 'rust' ? 'transpile' : 'generate-wgsl', '--input']);
    assert.equal(record.args.length, 3);
    assert.equal(fs.existsSync(record.args[2]), false, 'private source is removed after success');
    assert.equal(result.compilerVersion, '0.2.0');
    assert.equal(result.target, target);
  }
  assert.equal(fs.existsSync(path.join(directory, 'INJECTED')), false);
});

test('regular UTF-8 files use a bounded snapshot, including paths with spaces', posix, async t => {
  const { binary, directory } = fixture(t);
  const input = path.join(directory, 'source ; $(touch OTHER).cu');
  fs.writeFileSync(input, '__global__ void f() {}');
  const result = await api.transpileFile(input, { binary });
  assert.equal(JSON.parse(result.code).source, '__global__ void f() {}');
  fs.writeFileSync(input, Buffer.from([0xff, 0xfe]));
  await rejectsCode(api.transpileFile(input, { binary }), 'E_INPUT');
  await rejectsCode(api.transpileFile(directory, { binary }), 'E_INPUT');
});

test('rejects bad source, oversized headers and unsupported targets before spawning', async t => {
  const missing = path.join(os.tmpdir(), 'nonexistent-cuda2rust-for-input-tests');
  for (const source of ['', ' ', null, '\ud800']) {
    await rejectsCode(api.transpileCuda(source, { binary: missing }), 'E_INPUT');
  }
  await rejectsCode(api.transpileCuda('x'.repeat(api.MAX_INPUT_BYTES + 1), { binary: missing }), 'E_INPUT_LIMIT');
  await rejectsCode(api.transpileCuda('x', { binary: missing, target: 'wasm' }), 'E_TARGET');
  for (const options of [{ timeoutMs: 0 }, { timeoutMs: 120001 }, { maxOutputBytes: Infinity }, { maxOutputBytes: 16777217 }, { shell: true }]) {
    await rejectsCode(api.transpileCuda('x', options), 'E_OPTIONS');
  }
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'cuda-input-test-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const input = path.join(directory, 'large.cu');
  fs.writeFileSync(input, 'x'.repeat(api.MAX_INPUT_BYTES + 1));
  await rejectsCode(api.transpileFile(input, { binary: missing }), 'E_INPUT_LIMIT');
});

test('missing executables produce an actionable stable error', async t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'cuda-missing-test-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  await assert.rejects(api.transpileCuda('x', { binary: path.join(directory, 'missing') }), error => {
    assert.equal(error.code, 'E_COMPILER_NOT_FOUND');
    assert.match(error.message, /CUDA2RUST_BINARY/);
    return true;
  });
});

test('rejects an older compiler version', posix, async t => {
  const { binary } = fixture(t, '', '0.1.6');
  await rejectsCode(api.getCompilerVersion({ binary }), 'E_COMPILER_VERSION');
});

test('propagates nonzero exits and bounds/sanitizes diagnostic text', posix, async t => {
  const { binary } = fixture(t, `process.stderr.write('bad\\u001b[31m input'); process.exitCode = 9;`);
  await assert.rejects(api.transpileCuda('x', { binary }), error => {
    assert.equal(error.code, 'E_COMPILER');
    assert.equal(error.exitCode, 9);
    assert.match(error.compilerStderr, /bad/);
    assert.equal(error.compilerStderr.includes('\u001b'), false);
    return true;
  });
});

test('kills a timed-out compiler and removes its private input', posix, async t => {
  const { binary, directory } = fixture(t,
    `fs.writeFileSync(__filename + '.record', args[2]); setInterval(() => {}, 1000);`);
  await rejectsCode(api.transpileCuda('x', { binary, timeoutMs: 500 }), 'E_TIMEOUT');
  const input = fs.readFileSync(binary + '.record', 'utf8');
  assert.equal(fs.existsSync(input), false);
  assert.ok(directory);
});

test('rejects excessive or empty compiler output', posix, async t => {
  const large = fixture(t, `process.stdout.write('x'.repeat(100000));`);
  await rejectsCode(api.transpileCuda('x', { binary: large.binary, maxOutputBytes: 4096 }), 'E_OUTPUT_LIMIT');
  const empty = fixture(t, `process.exitCode = 0;`);
  await rejectsCode(api.transpileCuda('x', { binary: empty.binary }), 'E_COMPILER');
});

test('CLI help and version need no compiler; unknown legacy commands fail', () => {
  const cli = path.resolve(__dirname, '../cli/index.js');
  assert.equal(execFileSync(process.execPath, [cli, '--version'], { encoding: 'utf8' }).trim(), pkg.version);
  assert.match(execFileSync(process.execPath, [cli, '--help'], { encoding: 'utf8' }), /source text only/);
  assert.throws(() => execFileSync(process.execPath, [cli, 'benchmark', 'x.cu'], { stdio: 'pipe' }));
});

test('CLI protects the input and an existing output file', posix, t => {
  const { binary, directory } = fixture(t);
  const input = path.join(directory, 'input.cu');
  const output = path.join(directory, 'output.rs');
  fs.writeFileSync(input, 'kernel');
  fs.writeFileSync(output, 'keep');
  const cli = path.resolve(__dirname, '../cli/index.js');
  const base = [cli, 'transpile', input, '--binary', binary];
  assert.throws(() => execFileSync(process.execPath, [...base, '--output', input], { stdio: 'pipe' }));
  assert.throws(() => execFileSync(process.execPath, [...base, '--output', output], { stdio: 'pipe' }));
  assert.equal(fs.readFileSync(output, 'utf8'), 'keep');
  execFileSync(process.execPath, [...base, '--output', output, '--force']);
  assert.equal(JSON.parse(fs.readFileSync(output, 'utf8')).source, 'kernel');
  assert.equal(fs.readFileSync(input, 'utf8'), 'kernel');
  const alias = path.join(directory, 'alias.rs');
  fs.linkSync(input, alias);
  assert.throws(() => execFileSync(process.execPath, [...base, '--output', alias, '--force'], { stdio: 'pipe' }));
  assert.equal(fs.readFileSync(input, 'utf8'), 'kernel');
});
