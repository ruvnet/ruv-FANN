#!/usr/bin/env node
/** Actual source archive -> normal install lifecycle -> patched DB -> public CLI.
 * Dependencies and native sources are installed in a new external consumer.
 */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { delimiter, dirname, join } from 'node:path';
import { externalEvidencePath, sourceSnapshot, repo, sha } from '../../../integrations/ruv-fann/test/e2e/fixture.mjs';

if (process.argv.length !== 4 || process.argv[2] !== '--output') {
  throw new Error('Use --output NEW_EXTERNAL_DIRECTORY');
}
const output = await externalEvidencePath(process.argv[3]);
await mkdir(output, { mode: 0o700 });
const before = await sourceSnapshot(), npmRoot = join(repo, 'ruv-swarm/npm');
const env = { ...process.env, PATH: dirname(process.execPath) + delimiter + process.env.PATH,
  npm_config_ignore_scripts: 'false', npm_config_jobs: '2' };
execFileSync(process.execPath, ['scripts/validate-package.js'], { cwd: npmRoot, timeout: 30000, stdio: 'pipe', env });
const [packed] = JSON.parse(execFileSync('npm', ['pack', '--ignore-scripts', '--json', '--pack-destination', output],
  { cwd: npmRoot, encoding: 'utf8', timeout: 60000, maxBuffer: 2 * 1024 * 1024, env }));
assert.match(packed.filename, /^ruv-swarm-[0-9.]+\.tgz$/);
const tarball = join(output, packed.filename), tarBytes = await readFile(tarball);
assert.ok(tarBytes.length <= 8 * 1024 * 1024);
const entries = execFileSync('tar', ['-tzf', tarball], { encoding: 'utf8', timeout: 10000, maxBuffer: 2 * 1024 * 1024 }).trim().split('\n');
assert.ok(entries.length < 2048);
for (const entry of entries) {
  assert.ok(entry.startsWith('package/') && !entry.split('/').includes('..'));
  assert.ok(!entry.split('/').some(part => ['node_modules', '.git', '.swarm', '.claude-flow', 'native'].includes(part) || part.startsWith('.env')));
}
const consumer = join(output, 'consumer'); await mkdir(consumer, { mode: 0o700 });
await writeFile(join(consumer, 'package.json'), `${JSON.stringify({ private: true, type: 'module',
  dependencies: { 'ruv-swarm': `file:../${packed.filename}` } }, null, 2) }\n`, { flag: 'wx' });
const installed = execFileSync('npm', ['install', '--ignore-scripts=false', '--foreground-scripts', '--no-audit', '--no-fund'],
  { cwd: consumer, encoding: 'utf8', timeout: 600000, maxBuffer: 16 * 1024 * 1024, env });
await writeFile(join(output, 'install.log'), installed, { flag: 'wx' });
const packageRoot = join(consumer, 'node_modules/ruv-swarm');
const manifest = JSON.parse(await readFile(join(packageRoot, 'package.json'), 'utf8'));
assert.equal(manifest.name, 'ruv-swarm');
assert.equal(manifest.version, packed.version);
assert.equal(manifest.scripts.postinstall, 'node scripts/sqlite-build.mjs');
const engine = JSON.parse(execFileSync(process.execPath, ['scripts/sqlite-build.mjs', '--verify'],
  { cwd: packageRoot, encoding: 'utf8', timeout: 30000, maxBuffer: 1024 * 1024, env }));
assert.equal(engine.schema, 'ruv-swarm.sqlite-verification/1');
assert.equal(engine.status, 'passed');
assert.equal(engine.sqliteVersion, '3.54.1');
assert.match(engine.sqliteSourceId, /^\d{4}-\d{2}-\d{2} /);
const cli = join(packageRoot, manifest.bin['ruv-swarm']);
const database = join(consumer, 'private-state.sqlite');
const cliEnv = { ...env, RUV_SWARM_DB_PATH: database };
const call = args => JSON.parse(execFileSync(process.execPath, [cli, ...args], {
  cwd: consumer, encoding: 'utf8', timeout: 30000, maxBuffer: 1024 * 1024, env: cliEnv,
}));
call(['init', 'mesh', '2']);
const literal = "Literal ' ; DROP TABLE agents; -- 日本語";
call(['spawn', 'coder', 'Archive agent']);
const task = call(['orchestrate', literal]);
assert.equal(task.status, 'assigned');
assert.equal(typeof task.id, 'string');
assert.equal(call(['status']).total_agents, 1); // a fresh process reopens the DB
assert.ok((await readFile(database)).length > 0);
const reader = join(consumer, 'read-task.mjs');
await writeFile(reader, `import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const [module, database, id, expected] = process.argv.slice(2);
const { default: Database } = await import(pathToFileURL(module));
const db = new Database(database, { readonly: true, fileMustExist: true });
try {
  assert.equal(db.prepare('SELECT description FROM tasks WHERE id = ?').get(id).description, expected);
} finally { db.close(); }
`, { flag: 'wx' });
execFileSync(process.execPath, [reader, join(packageRoot, 'src/sqlite-engine.js'), database, task.id, literal],
  { cwd: consumer, encoding: 'utf8', timeout: 30000, maxBuffer: 1024 * 1024, env: cliEnv });
assert.deepEqual(await sourceSnapshot(), before);
const receipt = { schema: 'ruv-swarm.sqlite-package/1', status: 'passed', node: process.version,
  nodeAbi: process.versions.modules, platform: process.platform, arch: process.arch,
  actualLifecycleInstall: true, actualCliPersistence: true, sqliteVersion: engine.sqliteVersion,
  sqliteSourceId: engine.sqliteSourceId, sourceId: engine.sqliteSourceId,
  binarySha256: engine.receipt.binarySha256,
  tarballSha256: sha(tarBytes), packageVersion: packed.version, engine,
  literalArgumentPreservedThroughProcessBoundary: true, cliProcessCount: 4,
  inputsBefore: before, inputsAfter: before, unchanged: true,
  scope: 'Actual external archive installation with lifecycle enabled, source-built per-ABI SQLite verification, and four public CLI processes. No checkout dependency symlink or native-binary copy.' };
await writeFile(join(output, 'receipt.json'), `${JSON.stringify(receipt, null, 2) }\n`, { flag: 'wx' });
console.log(JSON.stringify({ status: 'passed', receipt: join(output, 'receipt.json'), sqliteVersion: engine.sqliteVersion }));
