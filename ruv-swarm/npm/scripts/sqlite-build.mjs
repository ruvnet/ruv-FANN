#!/usr/bin/env node
/** Force a source build against the pinned patched SQLite, with no prebuilt fallback. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import {
  PACKAGE_ROOT, BETTER_SQLITE_VERSION, SOURCE_FILES, MAX_BINARY_BYTES,
  readSourceContract, readBoundedFile, bindingPaths, runtimeIdentity,
  removeBuildFile, makeReceipt, builderHashes,
} from './sqlite-build-contract.mjs';
import { nodeGypPackage, sourceFiles } from './sqlite-build-source.mjs';
import { runBuildProcess } from './sqlite-build-process.mjs';

const require = createRequire(import.meta.url);
async function directory(filename) {
  try {
    await fs.mkdir(filename);
  } catch (error) {
    if (error.code !== 'EEXIST') {
      throw error;
    }
  }
  const stat = await fs.lstat(filename);
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    throw new Error('SQLite build directory must not be a symlink');
  }
}

export async function buildSqlite(archive) {
  if (process.env.NODE_TLS_REJECT_UNAUTHORIZED === '0' || process.env.npm_config_strict_ssl === 'false') {
    throw new Error('SQLite source builds require TLS certificate verification');
  }
  const inputsBefore = builderHashes();
  const source = readSourceContract(), paths = bindingPaths(), identity = runtimeIdentity();
  const manifest = require.resolve('better-sqlite3/package.json');
  const wrapper = JSON.parse(readBoundedFile(manifest, 65536));
  if (wrapper.version !== BETTER_SQLITE_VERSION) {
    throw new Error(`Install better-sqlite3 exactly ${ BETTER_SQLITE_VERSION}`);
  }
  const gyp = nodeGypPackage(), wrapperDirectory = path.dirname(manifest);
  await directory(path.join(PACKAGE_ROOT, 'native'));
  await directory(path.join(PACKAGE_ROOT, 'native/sqlite'));
  const lockPath = path.join(PACKAGE_ROOT, 'native/sqlite/.build.lock');
  let lock;
  try {
    lock = await fs.open(lockPath, 'wx', 0o600);
  } catch (cause) {
    throw new Error('SQLite build is locked; finish the other build before retrying', { cause });
  }
  let temporary, stagedBinary, stagedReceipt, failure;
  try {
    await lock.writeFile(JSON.stringify({ pid: process.pid, ...identity }));
    await directory(paths.directory);
    // A failed rebuild cannot silently continue using a stale receipt.
    removeBuildFile(paths.receipt);
    temporary = await fs.mkdtemp(path.join(PACKAGE_ROOT, 'native/sqlite/.source-'));
    const files = await sourceFiles(source, archive);
    for (const name of SOURCE_FILES) {
      await fs.writeFile(path.join(temporary, name), files.get(name), { flag: 'wx', mode: 0o444 });
    }
    const env = {
      ...process.env,
      PATH: path.dirname(process.execPath) + path.delimiter + (process.env.PATH || ''),
      npm_config_build_from_source: 'true', npm_config_force_build: '1', force_build: '1',
      npm_config_sqlite3: temporary, npm_config_jobs: '2',
      npm_config_disturl: 'https://nodejs.org/download/release',
    };
    delete env.GYP_DEFINES;
    delete env.npm_config_runtime;
    if (!env.npm_config_cafile) {
      env.npm_config_cafile = env.NODE_EXTRA_CA_CERTS || env.SSL_CERT_FILE || '';
    }
    await runBuildProcess(process.execPath, [
      path.join(gyp.directory, 'bin/node-gyp.js'), 'rebuild', '--release', '--jobs=2',
      '--force_build=1', `--sqlite3=${ temporary}`, `--target=${ process.versions.node}`,
      `--arch=${ process.arch}`, '--dist-url=https://nodejs.org/download/release',
    ], { cwd: wrapperDirectory, env });
    const binary = readBoundedFile(path.join(wrapperDirectory, 'build/Release/better_sqlite3.node'), MAX_BINARY_BYTES);
    stagedBinary = path.join(paths.directory, `.build-${ process.pid }.node`);
    await fs.writeFile(stagedBinary, binary, { flag: 'wx', mode: 0o444 });
    // A loaded Windows DLL cannot be renamed. Probe in a bounded subprocess
    // which exits (and closes its binding) before publishing either artifact.
    const probe = [
      "import {createRequire} from 'node:module';",
      'const {assertEngineIdentity}=await import(process.argv[4]);',
      'const require=createRequire(process.argv[1]); const Database=require(process.argv[1]);',
      "const db=new Database(':memory:',{nativeBinding:process.argv[2]});",
      "try{assertEngineIdentity(db.prepare('SELECT sqlite_version() AS version, sqlite_source_id() AS sourceId').get(),JSON.parse(process.argv[3]));}",
      'finally{db.close();}',
    ].join('\n');
    await runBuildProcess(process.execPath, [
      '--input-type=module', '-e', probe, require.resolve('better-sqlite3'), stagedBinary,
      JSON.stringify(source), new URL('./sqlite-build-contract.mjs', import.meta.url).href,
    ], { timeout: 15000, maximumOutput: 16384 });
    const receipt = makeReceipt(source, binary);
    if (JSON.stringify(receipt.builderFiles) !== JSON.stringify(inputsBefore)) {
      throw new Error('SQLite builder/source inputs changed during compilation');
    }
    stagedReceipt = path.join(paths.directory, `.receipt-${ process.pid }.json`);
    await fs.writeFile(stagedReceipt, `${JSON.stringify(receipt, null, 2) }\n`, { flag: 'wx', mode: 0o444 });
    if (process.platform === 'win32') {
      removeBuildFile(paths.binary);
    }
    await fs.rename(stagedBinary, paths.binary); stagedBinary = undefined;
    await fs.rename(stagedReceipt, paths.receipt); stagedReceipt = undefined;
    return { schema: 'ruv-swarm.sqlite-build-result/1', status: 'passed',
      nativeBinding: path.relative(PACKAGE_ROOT, paths.binary), receipt };
  } catch (error) {
    failure = error; throw error;
  } finally {
    const cleanup = await Promise.allSettled([
      Promise.resolve().then(() => {
        if (stagedBinary) {
          removeBuildFile(stagedBinary);
        }
      }),
      Promise.resolve().then(() => {
        if (stagedReceipt) {
          removeBuildFile(stagedReceipt);
        }
      }),
      temporary ? fs.rm(temporary, { recursive: true, force: true }) : Promise.resolve(),
    ]);
    try {
      await lock.close();
    } finally {
      removeBuildFile(lockPath);
    }
    const rejected = cleanup.find(result => result.status === 'rejected');
    if (rejected && !failure) {
      throw new Error('SQLite temporary build cleanup failed', { cause: rejected.reason });
    }
  }
}

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 1 && args[0] === '--verify') {
    const { verifySqliteEngine } = await import('../src/sqlite-engine.js');
    console.log(JSON.stringify(verifySqliteEngine(), null, 2)); return;
  }
  if (args.length !== 0 && !(args.length === 2 && args[0] === '--archive' && args[1])) {
    throw new Error('Use sqlite-build.mjs [--archive PINNED_CRATE] or --verify');
  }
  console.log(JSON.stringify(await buildSqlite(args[1] ? path.resolve(args[1]) : undefined), null, 2));
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  main().catch(error => {
    console.error(error.message); process.exitCode = 1;
  });
}
