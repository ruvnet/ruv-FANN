#!/usr/bin/env node
/** Explicit native source build. No dependency install, binary download or fallback. */
import { lstat, mkdir, readFile, readdir, copyFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { configPreflight } from './preflight.mjs';
import { externalPath } from './paths.mjs';

const directory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repository = resolve(directory, '../..');
const arguments_ = process.argv.slice(2);
if (arguments_.length !== 2 || arguments_[0] !== '--target-dir' || !arguments_[1])
  throw new Error('Usage: node scripts/build.mjs --target-dir <shared build directory>');
const target = await externalPath(arguments_[1], repository, { existingDirectory: true });
if (!['linux', 'darwin'].includes(process.platform)) throw new Error('This build helper supports Linux/macOS; no platform SDK is downloaded.');
const commands = [];
const checkedConfigs = await configPreflight(directory, repository);
let actualRustcPath;
function command(program, args) {
  const child = spawnSync(program, args, { cwd: directory, encoding: 'utf8', timeout: 600000,
    maxBuffer: 8 * 1024 * 1024, env: { ...process.env, CARGO_INCREMENTAL: '0', CARGO_BUILD_JOBS: '2',
      CARGO_PROFILE_DEV_DEBUG: '0', CARGO_PROFILE_RELEASE_DEBUG: '0', CARGO_PROFILE_RELEASE_LTO: 'thin',
      CARGO_PROFILE_RELEASE_CODEGEN_UNITS: '1', CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS: 'true',
      CARGO_PROFILE_RELEASE_OPT_LEVEL: '3', CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS: 'false',
      ...(actualRustcPath ? { RUSTC: actualRustcPath } : {}),
      CARGO_NET_OFFLINE: 'true', RUSTUP_AUTO_INSTALL: '0' } });
  commands.push({ program, args, exitCode: child.status });
  if (child.error || child.status !== 0) throw new Error(`${program} failed: ${child.error?.message ?? child.stderr}`);
  if (child.stderr) process.stderr.write(child.stderr);
  return child.stdout.trim();
}
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
async function sources() {
  const files = ['Cargo.toml', 'Cargo.lock', 'bindings/napi/Cargo.toml', 'bindings/napi/Cargo.lock',
    'bindings/napi/build.rs', 'bindings/napi/scripts/build.mjs', 'bindings/napi/scripts/preflight.mjs',
    'bindings/napi/scripts/paths.mjs'];
  let entries = 0, totalBytes = 0;
  async function walk(name) {
    const items = await readdir(join(repository, name), { withFileTypes: true });
    if (items.length > 2048) throw new Error('Source directory exceeds entry budget');
    for (const item of items) {
      if (++entries > 4096) throw new Error('Source discovery exceeds entry budget');
      const relative = name + '/' + item.name;
      if (item.isSymbolicLink() || (!item.isDirectory() && !item.isFile())) throw new Error('Source symlink/special file is unsupported');
      if (item.isDirectory()) await walk(relative);
      else if (/\.(?:rs|wgsl)$/.test(item.name)) files.push(relative);
    }
  }
  await walk('src'); await walk('bindings/napi/src');
  for (const name of ['.cargo/config', '.cargo/config.toml', 'rust-toolchain', 'rust-toolchain.toml']) {
    try { await lstat(join(repository, name)); files.push(name); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  const result = {};
  for (const name of [...new Set(files)].sort()) {
    const path = join(repository, name), info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink() || info.size > 8 * 1024 * 1024) throw new Error('Source file exceeds its ordinary-file budget');
    totalBytes += info.size;
    if (totalBytes > 32 * 1024 * 1024) throw new Error('Build source exceeds 32MiB');
    result[name] = hash(await readFile(path));
  }
  return result;
}
const available = command('rustup', ['toolchain', 'list']);
if (!available.split('\n').some(line => line.startsWith('1.88.0-'))) throw new Error('Install Rust1.88.0 explicitly before building');
actualRustcPath = command('rustup', ['which', '--toolchain', '1.88.0', 'rustc']);
const rustc = command(actualRustcPath, ['-vV']);
const rustcSha256 = hash(await readFile(actualRustcPath));
const cargo = command('cargo', ['+1.88.0', '--version']);
const inputsBefore = await sources();
const messages = command('cargo', ['+1.88.0', 'build', '--release', '--locked', '--offline', '--message-format=json-render-diagnostics', '--manifest-path',
  join(directory, 'Cargo.toml'), '--target-dir', target]);
const inputsAfter = await sources();
if (JSON.stringify(inputsBefore) !== JSON.stringify(inputsAfter)) throw new Error('Build inputs changed during compilation');
if (JSON.stringify(checkedConfigs) !== JSON.stringify(await configPreflight(directory, repository)))
  throw new Error('Cargo configuration changed during compilation');
if (hash(await readFile(actualRustcPath)) !== rustcSha256) throw new Error('Actual Rust compiler changed during build');
const filename = process.platform === 'darwin' ? 'libruv_fann_napi.dylib' : 'libruv_fann_napi.so';
const artifacts = messages.split('\n').filter(Boolean).map(line => JSON.parse(line)).filter(message =>
  message.reason === 'compiler-artifact' && message.target?.name === 'ruv_fann_napi' &&
  message.target?.kind?.includes('cdylib') && resolve(message.manifest_path) === join(directory, 'Cargo.toml'));
const outputs = artifacts.flatMap(message => message.filenames).filter(path => path.endsWith('/' + filename));
if (outputs.length !== 1 || resolve(outputs[0]) !== join(target, 'release', filename))
  throw new Error('Cargo did not identify exactly the expected host native artifact');
const source = outputs[0];
const binary = join(directory, 'ruv_fann.node');
await mkdir(directory, { recursive: true }); await copyFile(source, binary);
const receipt = { schema: 'ruv-fann.napi-build/1', platform: process.platform, arch: process.arch,
  nodeApi: 8, rustc, rustcPath: actualRustcPath, rustcSha256, cargo, node: process.version, binarySha256: hash(await readFile(binary)),
  coreLockSha256: inputsBefore['Cargo.lock'], napiLockSha256: inputsBefore['bindings/napi/Cargo.lock'],
  pins: { napi: '3.16.1', derive: '3.7.0', build: '2.6.0' }, inputsBefore, inputsAfter, commands,
  flags: { CARGO_PROFILE_RELEASE_DEBUG: '0', RUSTFLAGS: process.env.RUSTFLAGS ?? '',
    CARGO_ENCODED_RUSTFLAGS: process.env.CARGO_ENCODED_RUSTFLAGS ?? '',
    lto: 'thin', codegenUnits: 1, optLevel: 3, debugAssertions: false,
    checkedConfigs: Object.fromEntries(Object.entries(checkedConfigs).map(([path, text]) => [path, hash(text)])) },
  scope: 'Actual source-built native addon for this platform; Node-API8. No download or WASM fallback.' };
await writeFile(join(directory, 'build-info.json'), JSON.stringify(receipt, null, 2) + '\n');
process.stdout.write(JSON.stringify({ status: 'passed', binary, receipt: join(directory, 'build-info.json'), binarySha256: receipt.binarySha256 }) + '\n');
