const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { validatePackage } = require('../scripts/validate-package.js');

function fixture(t, pkg = { main: './index.cjs' }) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cuda-pkg-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify(pkg));
  return dir;
}

test('artifact gate accepts a loadable main', t => {
  const dir = fixture(t);
  fs.writeFileSync(path.join(dir, 'index.cjs'), 'module.exports = { available: true };');
  assert.deepEqual(validatePackage(dir), []);
});

test('artifact gate rejects missing main dependencies', t => {
  const dir = fixture(t);
  fs.writeFileSync(path.join(dir, 'index.cjs'), 'require("./missing-parser")');
  assert.ok(validatePackage(dir).some(error => error.includes('cannot load')));
});

test('artifact gate checks every export and rejects traversal', t => {
  const dir = fixture(t, { main: './index.cjs', exports: { '.': { types: './types.d.ts', import: './index.mjs' }, './bad': '../outside' } });
  const errors = validatePackage(dir);
  assert.ok(errors.some(error => error.includes('types.d.ts')));
  assert.ok(errors.some(error => error.includes('index.mjs')));
  assert.ok(errors.some(error => error.includes('escapes')));
});

test('artifact gate checks WASM bytes', t => {
  const dir = fixture(t, { main: './index.cjs', exports: { './wasm': './module.wasm' } });
  fs.writeFileSync(path.join(dir, 'index.cjs'), 'module.exports = {};');
  fs.writeFileSync(path.join(dir, 'module.wasm'), 'not WASM');
  assert.ok(validatePackage(dir).some(error => error.includes('WebAssembly')));
});

test('postinstall leaves consumer directory unchanged', t => {
  const dir = fixture(t);
  const before = fs.readdirSync(dir);
  execFileSync(process.execPath, [path.resolve(__dirname, '../scripts/postinstall.js')], { cwd: dir });
  assert.deepEqual(fs.readdirSync(dir), before);
});

test('future CUDA versions retain real API and CLI checks', t => {
  const dir = fixture(t, { name: 'cuda-wasm', version: '2.0.1', main: './index.cjs' });
  fs.writeFileSync(path.join(dir, 'index.cjs'), 'module.exports = {};');
  assert.ok(validatePackage(dir).some(error => error.includes('cjs source API')));
});

test('validator does not execute an entry point symlink outside the package', { skip: process.platform === 'win32' }, t => {
  const dir = fixture(t);
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'cuda-outside-test-'));
  t.after(() => fs.rmSync(outside, { recursive: true, force: true }));
  const marker = path.join(outside, 'executed');
  const external = path.join(outside, 'external.cjs');
  fs.writeFileSync(external, `require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'bad');`);
  fs.symlinkSync(external, path.join(dir, 'index.cjs'));
  assert.ok(validatePackage(dir).some(error => error.includes('symlink escapes')));
  assert.equal(fs.existsSync(marker), false);
});
