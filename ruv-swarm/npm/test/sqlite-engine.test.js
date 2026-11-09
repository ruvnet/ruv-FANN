import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';
import Database, { verifySqliteEngine } from '../src/sqlite-engine.js';
import {
  PACKAGE_ROOT, BUILDER_FILES, SOURCE_FILES, readSourceContract, runtimeIdentity,
  bindingPaths, sha256, makeReceipt, assertBuildReceipt, assertEngineIdentity, removeBuildFile,
} from '../scripts/sqlite-build-contract.mjs';
import { selectedTarFiles, checkedArchive } from '../scripts/sqlite-build-source.mjs';
import { runBuildProcess } from '../scripts/sqlite-build-process.mjs';

const source = readSourceContract();
const copy = value => JSON.parse(JSON.stringify(value));

async function isolatedPackage(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ruv-sqlite-guard-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  await fs.mkdir(path.join(root, 'src'));
  await fs.mkdir(path.join(root, 'scripts'));
  await fs.writeFile(path.join(root, 'package.json'), '{"type":"module"}\n');
  await fs.copyFile(path.join(PACKAGE_ROOT, 'src/sqlite-engine.js'), path.join(root, 'src/sqlite-engine.js'));
  for (const file of BUILDER_FILES) {
    await fs.copyFile(
      path.join(PACKAGE_ROOT, 'scripts', file), path.join(root, 'scripts', file),
    );
  }
  const module = await import(pathToFileURL(path.join(root, 'src/sqlite-engine.js')).href);
  return { root, Database: module.default };
}

function tar(entries) {
  const chunks = [];
  for (const { name, bytes = Buffer.alloc(0), type = '0' } of entries) {
    const header = Buffer.alloc(512);
    header.write(name, 0, 100);
    header.write('0000644\0', 100);
    header.write('0000000\0', 108);
    header.write('0000000\0', 116);
    header.write(`${bytes.length.toString(8).padStart(11, '0') }\0`, 124);
    header.write('00000000000\0', 136);
    header.fill(32, 148, 156);
    header.write(type, 156);
    const checksum = header.reduce((sum, byte) => sum + byte, 0);
    header.write(`${checksum.toString(8).padStart(6, '0') }\0 `, 148);
    chunks.push(header, bytes, Buffer.alloc((512 - bytes.length % 512) % 512));
  }
  return Buffer.concat([...chunks, Buffer.alloc(1024)]);
}

function fixture() {
  const bytes = new Map(SOURCE_FILES.map(name => [name, Buffer.from(`verified ${ name}`)]));
  const contract = {
    archive: { prefix: 'pinned/sqlite3/' },
    files: Object.fromEntries([...bytes].map(([name, value]) => [name, { bytes: value.length, sha256: sha256(value) }])),
  };
  const entries = [...bytes].map(([name, value]) => ({ name: contract.archive.prefix + name, bytes: value }));
  return { bytes, contract, entries };
}

test('source contract pins the patched public-domain amalgamation and exact source identity', () => {
  assert.equal(source.version, '3.54.1');
  assert.equal(source.license, 'public-domain');
  assert.equal(source.archive.sha256, '14a5f01b87b92705d07b433c06f96cbc764637d81a0bffb5734ce17cf0f9a2b3');
  assert.equal(source.archive.bytes, 2759388);
  assert.deepEqual(Object.keys(source.files).sort(), [...SOURCE_FILES].sort());
  assertEngineIdentity({ version: source.version, sourceId: source.sourceId }, source);
  for (const identity of [
    { version: '3.49.2', sourceId: source.sourceId },
    { version: '3.53.4', sourceId: source.sourceId },
    { version: '3.54.1', sourceId: 'unknown' }, null,
  ]) {
    assert.throws(() => assertEngineIdentity(identity, source), /version\/source_id/);
  }
});

test('archive extraction selects only the three pinned regular files', () => {
  const { bytes, contract, entries } = fixture();
  const result = selectedTarFiles(tar([
    { name: 'pinned/ignored.txt', bytes: Buffer.from('not extracted') }, ...entries,
  ]), contract);
  assert.equal(result.size, 3);
  for (const [name, value] of bytes) {
    assert.deepEqual(result.get(name), value);
  }
  const archive = Buffer.from('archive bytes');
  const pin = { archive: { bytes: archive.length, sha256: sha256(archive) } };
  assert.equal(checkedArchive(archive, pin), archive);
  assert.throws(() => checkedArchive(Buffer.from('changed bytes'), pin), /size\/checksum/);
});

test('archive extraction rejects traversal, links, duplicates, truncation and corrupt hashes', () => {
  const { contract, entries } = fixture();
  for (const name of ['../escape', '/absolute', 'pinned\\escape']) {
    assert.throws(() => selectedTarFiles(tar([{ name }, ...entries]), contract), /Unsafe/);
  }
  assert.throws(() => selectedTarFiles(tar([{ name: 'pinned/link', type: '2' }, ...entries]), contract), /nonregular/);
  assert.throws(() => selectedTarFiles(tar([...entries, entries[0]]), contract), /Duplicate/);
  assert.throws(() => selectedTarFiles(tar(entries.slice(1)), contract), /Incomplete/);
  assert.throws(() => selectedTarFiles(tar(entries).subarray(0, 520), contract), /Truncated/);
  const corrupt = tar(entries); corrupt[0] ^= 1;
  assert.throws(() => selectedTarFiles(corrupt, contract), /header checksum/);
  const bad = copy(contract); bad.files['sqlite3.c'].sha256 = '0'.repeat(64);
  assert.throws(() => selectedTarFiles(tar(entries), bad), /source checksum/);
});

test('receipt verification rejects stale wrapper, source, builder, binary and ABI metadata', () => {
  const binary = Buffer.from('receipt fixture');
  const valid = makeReceipt(source, binary);
  assertBuildReceipt(valid, source, binary);
  for (const [field, value] of [
    ['schema', 'old'], ['betterSqlite3Version', '11.10.0'], ['nodeGypVersion', '13.1.0'],
    ['sqliteVersion', '3.53.4'], ['sqliteSourceId', 'old'], ['nodeAbi', '999'],
    ['arch', 'unknown'], ['platform', 'unknown'], ['archiveSha256', '0'.repeat(64)],
    ['binarySha256', '0'.repeat(64)], ['binaryBytes', binary.length + 1],
  ]) {
    const receipt = copy(valid); receipt[field] = value;
    assert.throws(() => assertBuildReceipt(receipt, source, binary), /receipt mismatch/);
  }
  for (const field of ['sourceFiles', 'builderFiles']) {
    const receipt = copy(valid);
    receipt[field][Object.keys(receipt[field])[0]] = '0'.repeat(64);
    assert.throws(() => assertBuildReceipt(receipt, source, binary), /source hash mismatch/);
  }
  assert.throws(() => assertBuildReceipt({}, source, binary), /stale/);
  assert.throws(() => assertBuildReceipt(valid, source, Buffer.from('different binary')), /receipt mismatch/);
  const patchUpgrade = copy(valid); patchUpgrade.nodeVersion = '22.13.1';
  assertBuildReceipt(patchUpgrade, source, binary);
});

test('missing receipt fails before creating a user DB while non-database modules still import', async t => {
  const isolated = await isolatedPackage(t);
  const filename = path.join(isolated.root, 'must-not-open.db');
  assert.throws(() => new isolated.Database(filename), error =>
    error.code === 'ERR_RUV_SWARM_SQLITE_UNVERIFIED' && /build:sqlite/.test(error.message));
  await assert.rejects(fs.access(filename));
  const { RuvSwarm } = await import('../src/index.js');
  assert.equal(typeof RuvSwarm.initialize, 'function');
  const help = execFileSync(process.execPath, [path.join(PACKAGE_ROOT, 'bin/ruv-swarm-secure.js'), '--help'], {
    timeout: 15000, encoding: 'utf8', env: { ...process.env, RUV_SWARM_DB_PATH: filename },
  });
  assert.match(help, /ruv-swarm/i);
  await assert.rejects(fs.access(filename));
});

test('runtime rejects wrong and stale receipts before touching any user DB', async t => {
  const isolated = await isolatedPackage(t), paths = bindingPaths(isolated.root);
  const filename = path.join(isolated.root, 'must-not-open.db'), binary = Buffer.from('not a native binding');
  await fs.mkdir(paths.directory, { recursive: true });
  await fs.writeFile(paths.binary, binary);
  const valid = makeReceipt(source, binary, isolated.root);
  for (const edit of [
    r => {
      r.schema = 'old';
    }, r => {
      r.nodeAbi = '999';
    },
    r => {
      r.sqliteVersion = '3.53.4';
    }, r => {
      r.sqliteSourceId = 'old';
    },
    r => {
      r.binarySha256 = '0'.repeat(64);
    },
    r => {
      r.sourceFiles['sqlite3.c'] = '0'.repeat(64);
    },
    r => {
      r.builderFiles['sqlite-build.mjs'] = '0'.repeat(64);
    },
  ]) {
    const receipt = copy(valid); edit(receipt);
    await fs.writeFile(paths.receipt, JSON.stringify(receipt));
    assert.throws(() => new isolated.Database(filename), { code: 'ERR_RUV_SWARM_SQLITE_UNVERIFIED' });
    await assert.rejects(fs.access(filename));
  }
  await fs.writeFile(paths.receipt, '{invalid');
  assert.throws(() => new isolated.Database(filename), { code: 'ERR_RUV_SWARM_SQLITE_UNVERIFIED' });
  await fs.writeFile(paths.receipt, JSON.stringify(valid));
  await fs.rm(paths.binary);
  assert.throws(() => new isolated.Database(filename), { code: 'ERR_RUV_SWARM_SQLITE_UNVERIFIED' });
  await assert.rejects(fs.access(filename));
});

test('caller native bindings and accessor options cannot bypass the runtime gateway', async t => {
  const isolated = await isolatedPackage(t);
  let reads = 0;
  for (const options of [
    { nativeBinding: {} }, { nativeBinding: '/old/better_sqlite3.node' },
    { get readonly() {
      reads++; return true;
    } },
  ]) {
    assert.throws(() => new isolated.Database(':memory:', options), /Unsupported SQLite option/);
  }
  assert.equal(reads, 0);
});

test('bounded source reads reject deterministic growth and symlink/FIFO replacement', {
  skip: process.platform === 'win32' ? 'POSIX no-follow/FIFO regression' : false,
}, async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ruv-sqlite-file-race-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const helper = pathToFileURL(path.join(PACKAGE_ROOT, 'scripts/sqlite-build-contract.mjs')).href;
  for (const kind of ['growth', 'symlink', 'fifo']) {
    const filename = path.join(root, kind), target = path.join(root, `${kind }-target`);
    const script = [
      "import fs from 'node:fs'; import assert from 'node:assert/strict';",
      "import {execFileSync} from 'node:child_process'; import {syncBuiltinESMExports} from 'node:module';",
      `const {readBoundedFile}=await import(${ JSON.stringify(helper) });`,
      `const filename=${ JSON.stringify(filename) }, target=${ JSON.stringify(target) }, kind=${ JSON.stringify(kind) };`,
      "fs.writeFileSync(filename,'safe'); fs.writeFileSync(target,'not read');",
      'const open=fs.openSync, read=fs.readSync; let changed=false, reads=0, maximum=0;',
      "fs.openSync=(name,...args)=>{if(kind!=='growth'&&name===filename&&!changed){changed=true;fs.unlinkSync(filename);",
      "if(kind==='symlink')fs.symlinkSync(target,filename);else execFileSync('mkfifo',[filename],{timeout:1000});} return open(name,...args);};",
      "fs.readSync=(fd,buffer,...args)=>{reads++;maximum=Math.max(maximum,buffer.length);if(kind==='growth'&&!changed){changed=true;fs.appendFileSync(filename,Buffer.alloc(128));}return read(fd,buffer,...args);};",
      'syncBuiltinESMExports(); assert.throws(()=>readBoundedFile(filename,16));',
      "if(kind==='growth'){assert.ok(reads>0);assert.ok(maximum<=17);} console.log(JSON.stringify({kind,blocked:true,maximum}));",
    ].join('\n');
    const result = JSON.parse(execFileSync(process.execPath, ['--input-type=module', '-e', script], {
      encoding: 'utf8', timeout: 3000,
    }));
    assert.equal(result.blocked, true);
  }
});

test('read-only artifact replacement never chmods or removes a symlink target', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ruv-sqlite-publication-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const file = path.join(root, 'old.node'), target = path.join(root, 'target.node');
  await fs.writeFile(file, 'old', { mode: 0o444 });
  removeBuildFile(file);
  await assert.rejects(fs.access(file));
  removeBuildFile(file);
  if (process.platform !== 'win32') {
    await fs.writeFile(target, 'untouched', { mode: 0o444 });
    const originalMode = (await fs.stat(target)).mode & 0o777;
    await fs.symlink(target, file);
    assert.throws(() => removeBuildFile(file));
    assert.equal(await fs.readFile(target, 'utf8'), 'untouched');
    assert.equal((await fs.stat(target)).mode & 0o777, originalMode);
    assert.equal((await fs.lstat(file)).isSymbolicLink(), true);
  }
});

test('compiler deadlines and output limits terminate descendants before cleanup', {
  skip: process.platform === 'win32' ? 'POSIX process-group regression' : false,
}, async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ruv-sqlite-compiler-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const worker = path.join(root, 'worker.cjs');
  await fs.writeFile(worker, [
    "const {spawn}=require('node:child_process'),fs=require('node:fs');",
    "const writer='const fs=require(\"node:fs\");setInterval(()=>fs.appendFileSync(process.argv[1],\"x\"),10)';",
    "spawn(process.execPath,['-e',writer,process.argv[2]],{stdio:'inherit'});",
    "if(process.argv[3]==='parent-exit')process.exit(0);",
    "if(process.argv[3]==='output')setTimeout(()=>process.stdout.write(Buffer.alloc(8192)),150);",
    'setInterval(()=>{},1000);',
  ].join('\n'));
  const silent = { write() {} };
  for (const mode of ['deadline', 'output']) {
    const filename = path.join(root, mode);
    await fs.writeFile(filename, '');
    await assert.rejects(runBuildProcess(process.execPath, [worker, filename, mode], {
      timeout: mode === 'deadline' ? 350 : 2000, maximumOutput: 256, output: silent, errorOutput: silent,
    }), mode === 'deadline' ? /time limit/ : /byte limit/);
    const stopped = await fs.readFile(filename, 'utf8');
    await new Promise(resolve => setTimeout(resolve, 100));
    assert.equal(await fs.readFile(filename, 'utf8'), stopped, 'Compiler descendant kept writing after cleanup');
  }
  const earlyExit = path.join(root, 'compiler-parent-exit');
  await fs.writeFile(earlyExit, '');
  await runBuildProcess(process.execPath, [worker, earlyExit, 'parent-exit'], {
    timeout: 5000, maximumOutput: 256, output: silent, errorOutput: silent,
  });
  const afterExit = await fs.readFile(earlyExit, 'utf8');
  await new Promise(resolve => setTimeout(resolve, 100));
  assert.equal(await fs.readFile(earlyExit, 'utf8'), afterExit, 'Compiler parent exit left a live descendant');
  const filename = path.join(root, 'parent-exit'), supervisor = path.join(root, 'supervisor.mjs');
  await fs.writeFile(filename, '');
  const helper = pathToFileURL(path.join(PACKAGE_ROOT, 'scripts/sqlite-build-process.mjs')).href;
  await fs.writeFile(supervisor, [
    `import {runBuildProcess} from ${ JSON.stringify(helper) };`,
    `runBuildProcess(process.execPath,${ JSON.stringify([worker, filename, 'deadline'])
    },{output:{write(){}},errorOutput:{write(){}}}).catch(()=>{});`,
    'setTimeout(()=>process.exit(0),350);',
  ].join('\n'));
  execFileSync(process.execPath, [supervisor], { timeout: 3000 });
  const stopped = await fs.readFile(filename, 'utf8');
  await new Promise(resolve => setTimeout(resolve, 100));
  assert.equal(await fs.readFile(filename, 'utf8'), stopped, 'Parent exit left a live compiler descendant');
});

test('native SQLite is the actual patched source and preserves durable parameterized data', async t => {
  const report = verifySqliteEngine();
  assert.equal(report.status, 'passed');
  assert.equal(report.sqliteVersion, source.version);
  assert.equal(report.sqliteSourceId, source.sourceId);
  assert.equal(report.receipt.nodeAbi, runtimeIdentity().nodeAbi);
  assert.match(report.receipt.binarySha256, /^[0-9a-f]{64}$/);
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ruv-sqlite-durable-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const filename = path.join(root, 'state.db'), literal = "'); DROP TABLE records; --";
  let db = new Database(filename);
  try {
    db.exec('CREATE TABLE records (id INTEGER PRIMARY KEY, value TEXT NOT NULL)');
    db.prepare('INSERT INTO records(value) VALUES (?)').run(literal);
    assert.equal(db.prepare('SELECT count(*) AS count FROM records').get().count, 1);
  } finally {
    db.close();
  }
  db = new Database(filename, { readonly: true, fileMustExist: true });
  try {
    assert.equal(db.prepare('SELECT value FROM records WHERE id = ?').get(1).value, literal);
    assert.equal(db.prepare('SELECT sqlite_version() AS version').get().version, '3.54.1');
  } finally {
    db.close();
  }
});
