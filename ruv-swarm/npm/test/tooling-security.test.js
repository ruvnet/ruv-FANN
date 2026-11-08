import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const require = createRequire(import.meta.url);
const merge = require('deepmerge');

test('development config merging rejects prototype poisoning in target, source, and nested arrays', () => {
  const poison = () => JSON.parse('{"__proto__":{"untrustedConfig":true},"allowed":1}');
  for (const result of [merge(poison(), {}), merge({}, poison()), merge({ nested: poison() }, {}),
    merge({ list: [poison()] }, {})]) {
    const actual = result.nested || result.list?.[0] || result;
    assert.equal(Object.getPrototypeOf(actual), Object.prototype);
    assert.equal(actual.untrustedConfig, undefined);
    assert.equal(Object.hasOwn(actual, '__proto__'), false);
    assert.equal(actual.allowed, 1);
  }
  const symbol = Symbol('config');
  const target = { nested: { left: true }, list: [1], [symbol]: { left: true } };
  const result = merge(target, { nested: { right: true }, list: [2], [symbol]: { right: true } });
  assert.deepEqual(result, { nested: { left: true, right: true }, list: [1, 2], [symbol]: { left: true, right: true } });
  assert.deepEqual(target.list, [1]);
});

async function fixture(t) {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'ruv-tooling-'));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  return directory;
}

test('actual Jest preset normalization preserves nested globals and array concatenation', async t => {
  const directory = await fixture(t);
  await fs.writeFile(path.join(directory, 'preset.cjs'), 'module.exports = { globals: { nested: { left: true }, list: ["preset"] } };');
  const configuration = { rootDir: directory, preset: './preset.cjs',
    globals: { nested: { right: true }, list: ['project'] }, testMatch: ['**/not-run.test.cjs'] };
  const file = path.join(directory, 'jest.config.json');
  await fs.writeFile(file, JSON.stringify(configuration));
  const executable = path.join(path.dirname(require.resolve('jest/package.json')), 'bin/jest.js');
  const output = execFileSync(process.execPath, [executable, '--showConfig', '--config', file], { encoding: 'utf8', cwd: directory });
  const config = JSON.parse(output).configs[0];
  assert.deepEqual(config.globals, { nested: { left: true, right: true }, list: ['preset', 'project'] });
});

async function verifyRollupResolver(directory) {
  const dependency = path.join(directory, 'node_modules/fixture-dependency');
  await fs.mkdir(dependency, { recursive: true });
  await fs.writeFile(path.join(dependency, 'package.json'), JSON.stringify({ name: 'fixture-dependency', version: '1.0.0', module: 'index.js' }));
  await fs.writeFile(path.join(dependency, 'index.js'), 'export const answer = 42;');
  const entry = path.join(directory, 'entry.mjs');
  await fs.writeFile(entry, 'export { answer } from "fixture-dependency";');
  const { rollup } = await import('rollup');
  const { nodeResolve } = require('@rollup/plugin-node-resolve');
  const bundle = await rollup({ input: entry, plugins: [nodeResolve({ moduleDirectories: ['node_modules'] })] });
  try {
    const { output } = await bundle.generate({ format: 'es' });
    const watchedFiles = await Promise.all(bundle.watchFiles.map(file => fs.realpath(file)));
    assert.ok(watchedFiles.includes(await fs.realpath(path.join(dependency, 'index.js'))));
    const exports = await import(`data:text/javascript;base64,${Buffer.from(output[0].code).toString('base64')}`);
    assert.equal(exports.answer, 42);
  } finally {
    await bundle.close();
  }
}

test('actual Rollup resolver bundles a local package with the replacement merge dependency', async t => {
  await verifyRollupResolver(await fixture(t));
});

test('actual Rollup resolver recognizes watched dependencies through a directory alias', async t => {
  const directory = await fixture(t);
  const target = path.join(directory, 'real');
  const alias = path.join(directory, 'alias');
  await fs.mkdir(target);
  await fs.symlink(target, alias, process.platform === 'win32' ? 'junction' : 'dir');
  await verifyRollupResolver(alias);
});
