/** Build-boundary checks perform no Cargo builds or dependency installation. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm, readFile, symlink } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { rejectOverrides, lintOnlyConfig, configPreflight } from '../scripts/preflight.mjs';
import { externalPath } from '../scripts/paths.mjs';

test('compiler, target, wrapper and hidden flag overrides fail before Cargo can select a stale host artifact', () => {
  for (const name of ['RUSTC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'RUSTFLAGS',
    'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_TARGET', 'CARGO_BUILD_RUSTC',
    'CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER', 'CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS'])
    assert.throws(() => rejectOverrides({ [name]: '/not-the-recorded-compiler' }), /override/);
  rejectOverrides({ CARGO_BUILD_JOBS: '2', RUSTFLAGS: '', CARGO_TARGET_DIR: '/explicit-command-line-wins' });
});

test('only fully parsed level/name lint flag pairs are accepted from the known root config', async () => {
  const actual = await readFile(new URL('../../../.cargo/config.toml', import.meta.url), 'utf8');
  assert.equal(lintOnlyConfig(actual), true);
  for (const text of ['[build]\ntarget="aarch64-unknown-linux-gnu"', actual + '\n[build]\nrustc="evil"',
    actual.replace('"-A", "dead_code"', '"-Clinker=evil"'), '[target.x86_64]\nlinker="evil"'])
    assert.equal(lintOnlyConfig(text), false);
});

test('nested and Cargo-home configs cannot silently change compiler/target provenance', async () => {
  const root = await mkdtemp(join(tmpdir(), 'napi-build-boundary-'));
  const repository = join(root, 'repository'), directory = join(repository, 'bindings/napi');
  const cargoHome = join(root, 'cargo-home');
  await mkdir(directory, { recursive: true }); await mkdir(cargoHome);
  try {
    await configPreflight(directory, repository, { CARGO_HOME: cargoHome });
    await mkdir(join(repository, 'bindings/.cargo'));
    await writeFile(join(repository, 'bindings/.cargo/config.toml'), '[build]\ntarget="wrong"');
    await assert.rejects(configPreflight(directory, repository, { CARGO_HOME: cargoHome }), /ambient/);
    await rm(join(repository, 'bindings/.cargo'), { recursive: true });
    await writeFile(join(cargoHome, 'config.toml'), '[build]\nrustc-wrapper="wrong"');
    await assert.rejects(configPreflight(directory, repository, { CARGO_HOME: cargoHome }), /ambient/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('proof and shared build outputs reject source overlap and symlink/special ancestry', async () => {
  const root = await mkdtemp(join(tmpdir(), 'napi-output-boundary-'));
  const repository = join(root, 'repository'), external = join(root, 'repository-sibling');
  await mkdir(repository); await mkdir(external);
  try {
    for (const path of [repository, root, join(repository, 'proof.json'), join(repository, '..pack')])
      await assert.rejects(externalPath(path, repository), /external/);
    assert.equal(await externalPath(join(external, 'new'), repository), join(external, 'new'));
    await assert.rejects(externalPath(external, repository), /exists/);
    assert.equal(await externalPath(external, repository, { existingDirectory: true }), external);
    await symlink(repository, join(root, 'redirect'));
    await assert.rejects(externalPath(join(root, 'redirect/new'), repository), /symlink/);
    await writeFile(join(external, 'ordinary-file'), 'x');
    await assert.rejects(externalPath(join(external, 'ordinary-file/new'), repository), /ordinary/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
