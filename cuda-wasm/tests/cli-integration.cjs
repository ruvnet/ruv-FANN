'use strict';
// Explicit real-compiler gate. Run after building cuda2rust and setting CUDA2RUST_BINARY.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { pathToFileURL } = require('node:url');
const api = require('../dist/index.js');
const source = '__global__ void copy(float* a, float* b) { int i = threadIdx.x; b[i] = a[i]; }';

test('actual cuda2rust generates Rust and WGSL through CJS, ESM and the CLI', async t => {
  assert.ok(process.env.CUDA2RUST_BINARY, 'Set CUDA2RUST_BINARY to a built cuda2rust >=0.2.0');
  assert.match(await api.getCompilerVersion(), /^\d+\.\d+\.\d+$/);
  const rust = await api.transpileCuda(source);
  assert.match(rust.code, /fn copy/);
  assert.equal(rust.target, 'rust');
  const esm = await import(pathToFileURL(path.resolve(__dirname, '../dist/index.mjs')).href);
  const wgsl = await esm.transpileCuda(source, { target: 'wgsl' });
  assert.match(wgsl.code, /@compute/);
  assert.equal(wgsl.target, 'wgsl');
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'cuda-real-cli-'));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  const input = path.join(directory, 'kernel with spaces.cu');
  const output = path.join(directory, 'kernel.wgsl');
  await fs.writeFile(input, source);
  const generated = execFileSync(process.execPath, [path.resolve(__dirname, '../cli/index.js'),
    'transpile', input, '--target', 'wgsl'], { encoding: 'utf8', timeout: 30000 });
  assert.equal(generated, wgsl.code);
  execFileSync(process.execPath, [path.resolve(__dirname, '../cli/index.js'),
    'transpile', input, '--target', 'wgsl', '--output', output], { timeout: 30000 });
  assert.equal(await fs.readFile(output, 'utf8'), wgsl.code);
  // This checks genuine source generation. Shader compilation/execution and
  // numerical equivalence are deliberately separate hardware/compiler gates.
});
