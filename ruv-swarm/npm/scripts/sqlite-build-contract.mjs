/** Shared, fail-closed contract for the package-owned SQLite native binding. */
import { createHash } from 'node:crypto';
import { constants, openSync, closeSync, readSync, fstatSync, lstatSync, fchmodSync, unlinkSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const PACKAGE_ROOT = fileURLToPath(new URL('../', import.meta.url));
export const BETTER_SQLITE_VERSION = '12.11.1';
export const NODE_GYP_VERSION = '12.4.0';
export const RECEIPT_SCHEMA = 'ruv-swarm.sqlite-build/1';
export const SOURCE_FILES = Object.freeze(['sqlite3.c', 'sqlite3.h', 'sqlite3ext.h']);
export const BUILDER_FILES = Object.freeze([
  'sqlite-build.mjs', 'sqlite-build-source.mjs', 'sqlite-build-process.mjs',
  'sqlite-build-contract.mjs', 'sqlite-source.json',
]);
export const MAX_BINARY_BYTES = 32 * 1024 * 1024;
const hex = value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

export function readBoundedFile(filename, maximum) {
  if (!Number.isSafeInteger(maximum) || maximum < 1) {
    throw new Error('Invalid SQLite file bound');
  }
  const invalid = () => new Error(`Invalid, changed or oversized SQLite build file: ${ path.basename(filename)}`);
  const before = lstatSync(filename, { bigint: true });
  if (!before.isFile() || before.isSymbolicLink() || before.size === 0n || before.size > BigInt(maximum)) {
    throw invalid();
  }
  const descriptor = openSync(filename,
    constants.O_RDONLY | (constants.O_NOFOLLOW || 0) | (constants.O_NONBLOCK || 0));
  try {
    const initial = fstatSync(descriptor, { bigint: true });
    if (!initial.isFile() || initial.dev !== before.dev || initial.ino !== before.ino ||
      initial.size === 0n || initial.size > BigInt(maximum)) {
      throw invalid();
    }
    // At most max+1 bytes can be allocated/read, including a growth sentinel.
    const bytes = Buffer.allocUnsafe(Number(initial.size) + 1);
    let count = 0;
    while (count < bytes.length) {
      const read = readSync(descriptor, bytes, count, bytes.length - count, null);
      if (read === 0) {
        break;
      }
      count += read;
    }
    const after = fstatSync(descriptor, { bigint: true }), current = lstatSync(filename, { bigint: true });
    if (BigInt(count) !== initial.size || after.size !== initial.size || after.mtimeNs !== initial.mtimeNs ||
      after.ctimeNs !== initial.ctimeNs || !current.isFile() || current.isSymbolicLink() ||
      current.dev !== initial.dev || current.ino !== initial.ino) {
      throw invalid();
    }
    return bytes.subarray(0, count);
  } finally {
    closeSync(descriptor);
  }
}

export function readSourceContract(root = PACKAGE_ROOT) {
  const source = JSON.parse(readBoundedFile(path.join(root, 'scripts/sqlite-source.json'), 16384));
  if (source.schema !== 'ruv-swarm.sqlite-source/1' || source.version !== '3.54.1' ||
    source.archive?.url !== 'https://static.crates.io/crates/libsqlite3-sys/libsqlite3-sys-0.39.1.crate' ||
    source.archive.prefix !== 'libsqlite3-sys-0.39.1/sqlite3/' || !hex(source.archive.sha256) ||
    !Number.isSafeInteger(source.archive.bytes) || source.archive.bytes <= 0 || source.archive.bytes > 32 * 1024 * 1024 ||
    typeof source.sourceId !== 'string' || !/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [0-9a-f]{64}$/.test(source.sourceId)) {
    throw new Error('Invalid pinned SQLite source contract');
  }
  if (!source.files || Object.keys(source.files).sort().join(',') !== [...SOURCE_FILES].sort().join(',')) {
    throw new Error('SQLite source contract must pin exactly three amalgamation files');
  }
  for (const name of SOURCE_FILES) {
    const entry = source.files[name];
    if (!entry || !hex(entry.sha256) || !Number.isSafeInteger(entry.bytes) || entry.bytes <= 0 || entry.bytes > 16 * 1024 * 1024) {
      throw new Error(`Invalid SQLite amalgamation pin: ${ name}`);
    }
  }
  return source;
}

/** Remove only an ordinary artifact we opened; never chmod a symlink target. */
export function removeBuildFile(filename) {
  let descriptor;
  try {
    descriptor = openSync(filename, constants.O_RDONLY | (constants.O_NOFOLLOW || 0) | (constants.O_NONBLOCK || 0));
  } catch (error) {
    if (error.code === 'ENOENT') {
      return;
    }
    throw error;
  }
  try {
    const stat = fstatSync(descriptor, { bigint: true }), leaf = lstatSync(filename, { bigint: true });
    if (!stat.isFile() || !leaf.isFile() || leaf.isSymbolicLink() || stat.dev !== leaf.dev || stat.ino !== leaf.ino) {
      throw new Error('Refusing to replace a nonregular SQLite build artifact');
    }
    fchmodSync(descriptor, 0o600);
    const current = lstatSync(filename, { bigint: true });
    if (!current.isFile() || current.isSymbolicLink() || current.dev !== stat.dev || current.ino !== stat.ino) {
      throw new Error('SQLite build artifact changed before replacement');
    }
    unlinkSync(filename);
  } finally {
    closeSync(descriptor);
  }
}

export function runtimeIdentity() {
  const { platform, arch, versions } = process;
  if (!/^[a-z0-9_]+$/.test(platform) || !/^[a-z0-9_]+$/.test(arch) || !/^[1-9]\d*$/.test(versions.modules ?? '')) {
    throw new Error('Unsupported SQLite native runtime identity');
  }
  return { platform, arch, nodeAbi: versions.modules };
}

export function bindingPaths(root = PACKAGE_ROOT, identity = runtimeIdentity()) {
  const directory = path.join(root, 'native/sqlite',
    `${identity.platform }-${ identity.arch }-abi${ identity.nodeAbi}`);
  return { directory, binary: path.join(directory, 'better_sqlite3.node'), receipt: path.join(directory, 'receipt.json') };
}

export function builderHashes(root = PACKAGE_ROOT) {
  return Object.fromEntries(BUILDER_FILES.map(name => [
    name, sha256(readBoundedFile(path.join(root, 'scripts', name), 256 * 1024)),
  ]));
}

function sameKeys(actual, expected) {
  return actual && typeof actual === 'object' && !Array.isArray(actual) &&
    Object.keys(actual).sort().join(',') === Object.keys(expected).sort().join(',');
}

export function assertEngineIdentity(identity, source) {
  if (!identity || identity.version !== source.version || identity.sourceId !== source.sourceId) {
    throw new Error('SQLite runtime version/source_id does not match the patched source');
  }
  const version = identity.version.split('.').map(Number);
  if (version.length !== 3 || version.some(value => !Number.isSafeInteger(value) || value < 0) ||
    version[0] < 3 || (version[0] === 3 && version[1] < 54)) {
    throw new Error('SQLite runtime is older than the required 3.54.0 security floor');
  }
}

export function makeReceipt(source, binary, root = PACKAGE_ROOT) {
  return {
    schema: RECEIPT_SCHEMA, betterSqlite3Version: BETTER_SQLITE_VERSION, nodeGypVersion: NODE_GYP_VERSION,
    ...runtimeIdentity(), nodeVersion: process.versions.node,
    sqliteVersion: source.version, sqliteSourceId: source.sourceId,
    archiveSha256: source.archive.sha256,
    sourceFiles: Object.fromEntries(SOURCE_FILES.map(name => [name, source.files[name].sha256])),
    builderFiles: builderHashes(root), binarySha256: sha256(binary), binaryBytes: binary.length,
  };
}

export function assertBuildReceipt(receipt, source, binary, root = PACKAGE_ROOT, identity = runtimeIdentity()) {
  const expected = makeReceipt(source, binary, root);
  // Node patch versions sharing an ABI are compatible; preserve the actual
  // build version as evidence, without rejecting a normal Node patch upgrade.
  expected.nodeVersion = receipt?.nodeVersion;
  if (!sameKeys(receipt, expected) || typeof receipt.nodeVersion !== 'string' ||
    !/^\d+\.\d+\.\d+$/.test(receipt.nodeVersion)) {
    throw new Error('Missing or stale SQLite build receipt');
  }
  for (const key of Object.keys(expected)) {
    if (key === 'sourceFiles' || key === 'builderFiles') {
      if (!sameKeys(receipt[key], expected[key]) ||
        Object.keys(expected[key]).some(name => receipt[key][name] !== expected[key][name])) {
        throw new Error('SQLite build source hash mismatch');
      }
    } else if (receipt[key] !== expected[key]) {
      throw new Error(`SQLite build receipt mismatch: ${ key}`);
    }
  }
  for (const key of ['platform', 'arch', 'nodeAbi']) {
    if (receipt[key] !== identity[key]) {
      throw new Error('SQLite build runtime/ABI mismatch');
    }
  }
}
