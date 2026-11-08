import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { canonicalSourcePaths, recordWasmBuild, rustSourceDigest, verifyWasmBuild } from '../scripts/wasm-provenance.js';

test('source paths normalize mixed Windows separators before deduplication and sorting', () => {
  // Fixed recipe inputs use POSIX spelling; filesystem traversal uses native
  // separators. This is the mixed collection encountered by Windows prepack.
  const posix = ['ruv-swarm/Cargo.lock', 'ruv-swarm/crates/pkg/Cargo.toml',
    'ruv-swarm/npm/package.json', 'ruv-swarm/npm/src/index.js', 'src/lib.rs'];
  const mixed = ['src\\lib.rs', 'ruv-swarm/npm/package.json', 'ruv-swarm\\crates\\pkg\\Cargo.toml',
    'ruv-swarm/Cargo.lock', 'ruv-swarm\\npm\\src\\index.js', 'ruv-swarm\\Cargo.lock'];
  assert.deepEqual(canonicalSourcePaths(mixed, path.win32.sep), posix);
  assert.deepEqual(canonicalSourcePaths(mixed, path.win32.sep), canonicalSourcePaths(posix, path.posix.sep));
});

test('source path canonicalization preserves literal POSIX backslashes', () => {
  const paths = ['src/strange\\name.rs', 'src/strange/name.rs'];
  assert.deepEqual(canonicalSourcePaths(paths, path.posix.sep), ['src/strange/name.rs', 'src/strange\\name.rs']);
});

test('Git autocrlf checkout preserves extensionless build inputs and binary bytes', async t => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'ruv-checkout-'));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  const source = path.join(directory, 'source');
  const checkout = path.join(directory, 'checkout');
  await fs.mkdir(source);
  await fs.mkdir(checkout);
  const files = new Map([
    ...['LICENSE', 'LICENSE-MIT', 'LICENSE-APACHE', 'ruv-swarm/LICENSE-MIT',
      'ruv-swarm/crates/example/LICENSE-APACHE', 'rust-toolchain', 'ruv-swarm/rust-toolchain',
      '.cargo/config', 'ruv-swarm/.cargo/config', 'src/lib.rs', 'Cargo.toml', 'Cargo.lock',
      'package.json', 'ruv-swarm/npm/src/index.js', 'src/shader.wgsl'].map(file => [file, Buffer.from('first line\nsecond line\n')]),
    ['ruv-swarm/npm/wasm/core.wasm', Buffer.from([0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 0, 10, 13, 10])],
  ]);
  await fs.writeFile(path.join(source, '.gitattributes'), await fs.readFile(new URL('../../../.gitattributes', import.meta.url)));
  for (const [file, bytes] of files) {
    await fs.mkdir(path.dirname(path.join(source, file)), { recursive: true });
    await fs.writeFile(path.join(source, file), bytes);
  }
  const git = args => execFileSync('git', ['-c', 'core.autocrlf=true', '-c', 'core.eol=crlf',
    '-c', 'core.safecrlf=false', '-c', `core.hooksPath=${path.join(directory, 'no-hooks')}`, ...args],
  { cwd: source, stdio: 'pipe' });
  git(['init', '--quiet']);
  git(['add', '--', '.']);
  git([`--git-dir=${path.join(source, '.git')}`, `--work-tree=${checkout}`, 'checkout-index', '--all', '--force']);
  for (const [file, expected] of files) {
    assert.deepEqual(await fs.readFile(path.join(checkout, file)), expected, `Checkout changed raw bytes: ${file}`);
  }
});

async function fixture(t) {
  const repository = await fs.mkdtemp(path.join(os.tmpdir(), 'ruv-build-'));
  t.after(() => fs.rm(repository, { recursive: true, force: true }));
  const npmRoot = path.join(repository, 'ruv-swarm/npm');
  for (const directory of ['src', '.cargo', 'ruv-swarm/crates', 'ruv-swarm/ml-training', 'ruv-swarm/benchmarking',
    'ruv-swarm/npm/wasm', 'ruv-swarm/npm/src', 'ruv-swarm/npm/bin', 'ruv-swarm/npm/scripts']) {
    await fs.mkdir(path.join(repository, directory), { recursive: true });
  }
  for (const file of ['Cargo.toml', 'Cargo.lock', 'ruv-swarm/Cargo.toml', 'ruv-swarm/Cargo.lock',
    'package.json', 'package-lock.json', 'ruv-swarm/npm/package.json', 'ruv-swarm/npm/package-lock.json',
    'ruv-swarm/npm/scripts/build.js', 'ruv-swarm/npm/scripts/wasm-provenance.js',
    'ruv-swarm/npm/scripts/validate-package.js', 'ruv-swarm/npm/wasm/wasm-bindings-loader.mjs']) {
    await fs.writeFile(path.join(repository, file), '# fixture');
  }
  await fs.writeFile(path.join(repository, 'src/lib.rs'), 'pub fn example() {}');
  await fs.writeFile(path.join(npmRoot, 'wasm/core.wasm'), Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]));
  await fs.writeFile(path.join(npmRoot, 'wasm/core.js'), 'export default async function init() {}');
  return { repository, npmRoot, sourceSha256: await rustSourceDigest(repository) };
}

async function record({ npmRoot, sourceSha256 }) {
  await recordWasmBuild(npmRoot, {
    sourceSha256, artifacts: ['core.wasm', 'core.js'], toolchain: { rustc: 'fixture', wasmPack: 'fixture' },
    configurations: [{ name: 'core', artifacts: ['core.wasm'], target: 'wasm32-unknown-unknown', bindingTarget: 'web',
      profile: 'release', defaultFeatures: true, features: [], rustFlags: '' }],
  });
}

test('missing build evidence blocks publication even for valid WASM bytes', async t => {
  const { npmRoot } = await fixture(t);
  await assert.rejects(verifyWasmBuild(npmRoot), /Missing WASM build receipt/);
});

test('receipt checks current sources and exact built artifact bytes', async t => {
  const data = await fixture(t);
  await record(data);
  await verifyWasmBuild(data.npmRoot);
  await fs.writeFile(path.join(data.npmRoot, 'wasm/core.wasm'), 'changed bytes');
  await assert.rejects(verifyWasmBuild(data.npmRoot), /no matching build evidence/);
});

test('unrebuilt legacy WASM cannot acquire provenance by residing in output directory', async t => {
  const data = await fixture(t);
  await record(data);
  await fs.writeFile(path.join(data.npmRoot, 'wasm/legacy.wasm'), Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]));
  await assert.rejects(verifyWasmBuild(data.npmRoot), /legacy.wasm/);
});

test('new Rust source modules invalidate the receipt', async t => {
  const data = await fixture(t);
  await record(data);
  await fs.writeFile(path.join(data.repository, 'src/new.rs'), 'pub fn added() {}');
  await assert.rejects(verifyWasmBuild(data.npmRoot), /does not match/);
});

test('source changes during compilation prevent writing a build receipt', async t => {
  const data = await fixture(t);
  await fs.appendFile(path.join(data.repository, 'Cargo.lock'), '\n# changed dependency');
  await assert.rejects(record(data), /changed while WASM was building/);
});

test('receipt rejects raw LF-to-CRLF source changes without normalizing content', async t => {
  const data = await fixture(t);
  const license = path.join(data.repository, 'LICENSE');
  await fs.writeFile(license, 'license first line\nlicense second line\n');
  data.sourceSha256 = await rustSourceDigest(data.repository);
  await record(data);
  await fs.writeFile(license, 'license first line\r\nlicense second line\r\n');
  await assert.rejects(verifyWasmBuild(data.npmRoot), /does not match/);
});


test('unpublished optional artifacts do not block verified release files', async t => {
  const data = await fixture(t);
  await record(data);
  await fs.writeFile(path.join(data.npmRoot, 'wasm/legacy.wasm'), 'legacy unsupported');
  await verifyWasmBuild(data.npmRoot, data.repository, ['core.wasm']);
});

test('generated JavaScript glue is verified along with its WASM binary', async t => {
  const data = await fixture(t);
  await record(data);
  await fs.appendFile(path.join(data.npmRoot, 'wasm/core.js'), '\nexport const changed = true;');
  await assert.rejects(verifyWasmBuild(data.npmRoot), /core.js/);
});

test('build scripts, new Cargo config, runtime glue, and manifests invalidate evidence', async t => {
  for (const file of ['build.rs', '.cargo/config.toml', 'LICENSE', 'ruv-swarm/npm/src/runtime.js',
    'ruv-swarm/npm/scripts/build.js', 'ruv-swarm/npm/package.json']) {
    await t.test(file, async subtest => {
      const data = await fixture(subtest);
      await record(data);
      await fs.appendFile(path.join(data.repository, file), '\n// changed build input');
      await assert.rejects(verifyWasmBuild(data.npmRoot), /does not match/);
    });
  }
});

test('consumer verification works without any Rust checkout', async t => {
  const data = await fixture(t);
  await record(data);
  const consumer = await fs.mkdtemp(path.join(os.tmpdir(), 'ruv-consumer-'));
  t.after(() => fs.rm(consumer, { recursive: true, force: true }));
  await fs.cp(data.npmRoot, consumer, { recursive: true });
  await fs.rm(data.repository, { recursive: true, force: true });
  await verifyWasmBuild(consumer, null, ['core.wasm', 'core.js']);
});

test('SIMD artifacts require an explicit SIMD compiler configuration', async t => {
  const data = await fixture(t);
  await fs.copyFile(path.join(data.npmRoot, 'wasm/core.wasm'), path.join(data.npmRoot, 'wasm/core-simd.wasm'));
  await assert.rejects(recordWasmBuild(data.npmRoot, {
    sourceSha256: data.sourceSha256, artifacts: ['core-simd.wasm'], toolchain: { rustc: 'fixture', wasmPack: 'fixture' },
    configurations: [{ artifacts: ['core-simd.wasm'], target: 'wasm32-unknown-unknown', bindingTarget: 'web',
      profile: 'release', defaultFeatures: true, features: [], rustFlags: '' }],
  }), /artifact configuration/);
});
