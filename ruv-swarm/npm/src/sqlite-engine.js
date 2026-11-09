/** Load only the package-owned, source-verified patched SQLite binding. */
import { createRequire } from 'node:module';
import path from 'node:path';
import {
  PACKAGE_ROOT, BETTER_SQLITE_VERSION, MAX_BINARY_BYTES, readBoundedFile,
  readSourceContract, bindingPaths, assertBuildReceipt, assertEngineIdentity,
} from '../scripts/sqlite-build-contract.mjs';

const require = createRequire(import.meta.url);
const repair = `Run npm run build:sqlite in ${ PACKAGE_ROOT } using this Node runtime.`;

function verifiedBinding() {
  try {
    const source = readSourceContract(), paths = bindingPaths();
    const receipt = JSON.parse(readBoundedFile(paths.receipt, 16384));
    const binary = readBoundedFile(paths.binary, MAX_BINARY_BYTES);
    assertBuildReceipt(receipt, source, binary);
    const manifest = require.resolve('better-sqlite3/package.json');
    const wrapper = JSON.parse(readBoundedFile(manifest, 65536));
    if (wrapper.version !== BETTER_SQLITE_VERSION) {
      throw new Error('The pinned better-sqlite3 wrapper is missing');
    }
    // The normal addon resolver/prebuilt directories are never consulted.
    const Database = require('better-sqlite3');
    const probe = new Database(':memory:', { nativeBinding: paths.binary });
    let identity;
    try {
      identity = probe.prepare('SELECT sqlite_version() AS version, sqlite_source_id() AS sourceId').get();
      assertEngineIdentity(identity, source);
    } finally {
      probe.close();
    }
    return { Database, paths, receipt, identity };
  } catch (cause) {
    const error = new Error(`Patched SQLite validation failed: ${ cause.message }. ${ repair}`, { cause });
    error.code = 'ERR_RUV_SWARM_SQLITE_UNVERIFIED';
    throw error;
  }
}

/** Read-only native/source/ABI verification; opens only a private memory DB. */
export function verifySqliteEngine() {
  const { paths, receipt, identity } = verifiedBinding();
  return {
    schema: 'ruv-swarm.sqlite-verification/1', status: 'passed',
    sqliteVersion: identity.version, sqliteSourceId: identity.sourceId,
    nativeBinding: path.relative(PACKAGE_ROOT, paths.binary), receipt,
  };
}

/**
 * Validation happens at construction, so --ignore-scripts installations can
 * still import non-database WASM/help modules. Every user DB follows a successful
 * receipt/binary/ABI check and an actual patched-engine identity probe.
 */
export default function Database(filename, options = {}) {
  if (options === null) {
    options = {};
  }
  if (typeof options !== 'object' || Array.isArray(options) ||
    ![Object.prototype, null].includes(Object.getPrototypeOf(options))) {
    throw new TypeError('SQLite options must be a plain record');
  }
  const owned = {};
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (!['readonly', 'fileMustExist', 'timeout', 'verbose'].includes(key) || !Object.hasOwn(descriptor, 'value')) {
      throw new TypeError('Unsupported SQLite option or accessor');
    }
    owned[key] = descriptor.value;
  }
  const engine = verifiedBinding();
  return new engine.Database(filename, { ...owned, nativeBinding: engine.paths.binary });
}
