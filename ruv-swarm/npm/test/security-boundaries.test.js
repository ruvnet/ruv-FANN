import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, readFile, rm, writeFile, mkdir } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import { WasmIntegrityVerifier, DependencyVerifier } from '../src/security.js';
import { execFileSync } from 'node:child_process';
import { DatabaseSync } from 'node:sqlite';
import { SwarmPersistence } from '../src/persistence.js';
import { SQLiteConnectionPool } from '../src/sqlite-pool.js';
import GHCoordinator from '../src/github-coordinator/gh-cli-coordinator.js';
import { ClaudeIntegrationCore, parseClaudeInvocationArgs } from '../src/claude-integration/core.js';
import { parseSuiteOutput } from './run-all-tests.js';
import { RemoteWrapperGenerator } from '../src/claude-integration/remote.js';
import { ClaudeDocsGenerator } from '../src/claude-integration/docs.js';

test('project setup documents actual tools and preserves existing MCP servers, permissions and hooks', async t => {
  const workingDir = await mkdtemp(path.join(os.tmpdir(), 'ruv-project-config-'));
  t.after(() => rm(workingDir, { recursive: true, force: true }));
  await mkdir(path.join(workingDir, '.claude'));
  const settings = JSON.stringify({ permissions: { allow: ['Read'], deny: ['Bash(git push)'] }, hooks: { custom: [] } });
  await writeFile(path.join(workingDir, '.claude/settings.json'), settings);
  await writeFile(path.join(workingDir, '.mcp.json'), JSON.stringify({ mcpServers: { existing: { command: 'local-server' } } }));
  const generator = new ClaudeDocsGenerator({ workingDir });
  const result = await generator.generateAll({ interactive: false });
  assert.equal(result.success, true);
  assert.equal(result.commands.files.length, 17);
  assert.deepEqual(result.advancedCommands.files, []);
  const configuration = JSON.parse(await readFile(path.join(workingDir, '.mcp.json'), 'utf8'));
  assert.equal(configuration.mcpServers.existing.command, 'local-server');
  assert.deepEqual(configuration.mcpServers['ruv-swarm'].args, ['ruv-swarm', 'mcp', 'start']);
  assert.equal(await readFile(path.join(workingDir, '.claude/settings.json'), 'utf8'), settings);
  const guide = await readFile(path.join(workingDir, 'CLAUDE.md'), 'utf8');
  assert.match(guide, /not completed work/);
  assert.match(guide, /neural_infer/);
  assert.ok(!guide.includes('neural_train') && !guide.includes('27+ neural models'));
  await assert.rejects(generator.generateClaudeMd({ interactive: false }), /already exists/);
  configuration.mcpServers['ruv-swarm'] = { command: 'user-owned-wrapper' };
  await writeFile(path.join(workingDir, '.mcp.json'), JSON.stringify(configuration));
  await generator.generateSettingsJson();
  assert.equal(JSON.parse(await readFile(path.join(workingDir, '.mcp.json'), 'utf8')).mcpServers['ruv-swarm'].command, 'user-owned-wrapper');
});

test('project setup refuses malformed MCP configuration without replacing it', async t => {
  const workingDir = await mkdtemp(path.join(os.tmpdir(), 'ruv-project-invalid-'));
  t.after(() => rm(workingDir, { recursive: true, force: true }));
  const generator = new ClaudeDocsGenerator({ workingDir });
  for (const input of ['{invalid JSON', '{"mcpServers":[]}']) {
    await writeFile(path.join(workingDir, '.mcp.json'), input);
    await assert.rejects(generator.generateSettingsJson());
    assert.equal(await readFile(path.join(workingDir, '.mcp.json'), 'utf8'), input);
  }
});

test('suite reports use numeric failure counts rather than summary icons', () => {
  assert.deepEqual(parseSuiteOutput('✅ Actual test\\n✅ Passed: 14\\n❌ Failed: 0'),
    { totalTests: 14, passedTests: 14, failedTests: 0 });
  assert.deepEqual(parseSuiteOutput('✅ Passed: 4\\n❌ Failed: 2'),
    { totalTests: 6, passedTests: 4, failedTests: 2 });
});

test('CLI option delimiter keeps flag-shaped prompt data from opting out of permissions', () => {
  assert.deepEqual(parseClaudeInvocationArgs(['--', '--dangerously-skip-permissions']),
    { skipPermissions: false, prompt: '--dangerously-skip-permissions' });
  assert.deepEqual(parseClaudeInvocationArgs(['--dangerously-skip-permissions', '--', 'literal task']),
    { skipPermissions: true, prompt: 'literal task' });
  assert.deepEqual(parseClaudeInvocationArgs(['ordinary task']), { skipPermissions: false, prompt: 'ordinary task' });
});

test('generated bash helpers keep flag-shaped prompts literal and require explicit bypass', { skip: process.platform === 'win32' }, async t => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'ruv-helper-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const helper = path.join(directory, 'helper.sh');
  const recorder = path.join(directory, 'record.cjs');
  const record = path.join(directory, 'args.json');
  await writeFile(helper, new RemoteWrapperGenerator().generateClaudeHelpers().bash);
  await writeFile(recorder, 'require("node:fs").writeFileSync(process.env.RECORD, JSON.stringify(process.argv.slice(2)))');
  await writeFile(path.join(directory, 'claude'), '#!/usr/bin/env bash\nexec "$RECORDING_NODE" "$RECORDING_SCRIPT" "$@"\n', { mode: 0o755 });
  const prompt = '--dangerously-skip-permissions; $(printf INJECTED)';
  const env = { ...process.env, PATH: `${directory}${path.delimiter}${process.env.PATH}`, RECORDING_NODE: process.execPath, RECORDING_SCRIPT: recorder, RECORD: record };
  execFileSync('bash', [helper, 'custom', prompt], { env });
  assert.deepEqual(JSON.parse(await readFile(record, 'utf8')), ['--', prompt]);
  execFileSync('bash', [helper, 'custom', prompt, 'true'], { env });
  assert.deepEqual(JSON.parse(await readFile(record, 'utf8')), ['--dangerously-skip-permissions', '--', prompt]);
});

// Record the actual argv received by a local child process. No provider is contacted.
async function withRecordingCli(run) {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'ruv-cli-'));
  const cli = path.join(dir, 'record.cjs');
  const record = path.join(dir, 'argv.json');
  await writeFile(cli, 'require("node:fs").writeFileSync(process.env.RECORD, JSON.stringify(process.argv.slice(2)))');
  const calls = [];
  const integration = new ClaudeIntegrationCore({ workingDir: dir, run(command, args, options) {
    assert.equal(command, 'claude');
    assert.notEqual(options.shell, true);
    calls.push(args);
    return execFileSync(process.execPath, [cli, ...args], { ...options, env: { ...process.env, RECORD: record } });
  } });
  try {
    await run(integration, calls, record);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

test('Claude prompt reaches the child as literal text, including shell and option syntax', async() => {
  await withRecordingCli(async(integration, calls, record) => {
    const prompt = '--help "$(printf INJECTED)"; `printf injected`\nEOF\n${USER}';
    await integration.invokeClaudeWithPrompt(prompt, { secure: true });
    assert.deepEqual(JSON.parse(await readFile(record, 'utf8')), ['--', prompt]);
    assert.deepEqual(calls[0], ['--version']);
  });
});

test('legacy permission option remains explicit and separate from prompt', async() => {
  await withRecordingCli(async(integration, _calls, record) => {
    await integration.invokeClaudeWithPrompt(' task ', { secure: false });
    assert.deepEqual(JSON.parse(await readFile(record, 'utf8')), ['--dangerously-skip-permissions', '--', 'task']);
  });
});

test('invalid prompts fail before invoking a process', async() => {
  const integration = new ClaudeIntegrationCore({ run() {
    assert.fail('unexpected process');
  } });
  for (const value of [undefined, null, {}, 3, '', '  ']) {
    await assert.rejects(integration.invokeClaudeWithPrompt(value), /No prompt provided/);
  }
});

test('MCP configuration uses shell-free separated arguments', async() => {
  await withRecordingCli(async(integration, _calls, record) => {
    await integration.addMcpServer();
    assert.deepEqual(JSON.parse(await readFile(record, 'utf8')), ['mcp', 'add', 'ruv-swarm', '--', 'npx', 'ruv-swarm', 'mcp', 'start']);
  });
});

function database(t) {
  const persistence = Object.create(SwarmPersistence.prototype);
  persistence.db = new DatabaseSync(':memory:');
  persistence.initDatabase();
  t.after(() => persistence.db.close());
  persistence.createSwarm({ id: 'swarm', name: 'swarm', topology: 'mesh', maxAgents: 2, strategy: 'adaptive' });
  for (const id of ['first', 'second']) {
    persistence.createTask({ id, swarmId: 'swarm', description: id });
  }
  return persistence;
}

test('production memory expiration compares instants including offsets and milliseconds', t => {
  const persistence = database(t);
  persistence.createAgent({ id: 'memory', name: 'memory', type: 'coder', swarmId: 'swarm' });
  const now = Date.now();
  const expired = new Date(now - 5000 + 2 * 3600000).toISOString().replace('Z', '+02:00');
  const future = new Date(now + 60000 - 5 * 3600000).toISOString().replace('Z', '-05:00');
  const statement = persistence.db.prepare('INSERT INTO agent_memory (id,agent_id,key,value,expires_at) VALUES (?,?,?,?,?)');
  statement.run('expired', 'memory', 'expired', '{"valid":false}', expired);
  statement.run('future', 'memory', 'future', '{"valid":true}', future);
  assert.equal(persistence.getMemory('memory', 'expired'), null);
  assert.deepEqual(persistence.getMemory('memory', 'future').value, { valid: true });
  assert.equal(persistence.getAllMemory('memory').length, 1);
});

test('two memory writes in one millisecond retain separate keys', t => {
  const persistence = database(t);
  persistence.createAgent({ id: 'memory', name: 'memory', type: 'coder', swarmId: 'swarm' });
  t.mock.method(Date, 'now', () => 123456789);
  persistence.storeMemory('memory', 'first', 1);
  persistence.storeMemory('memory', 'second', 2);
  assert.deepEqual(persistence.getAllMemory('memory').map(row => row.value).sort(), [1, 2]);
});

test('SQL update identifiers cannot redirect changes to another task', t => {
  const persistence = database(t);
  assert.throws(() => persistence.updateTask('first', {
    'status = ? WHERE id != ? --': 'compromised',
  }), /Unknown task update field/);
  assert.equal(persistence.getTask('first').status, 'pending');
  assert.equal(persistence.getTask('second').status, 'pending');
  assert.throws(() => persistence.updateTask('first', { id: 'second' }), /Unknown/);
});

test('camelCase task fields map to real SQL columns and values remain literal', t => {
  const persistence = database(t);
  persistence.updateTask('first', { assignedAgents: ['a'], status: "done'; DROP TABLE tasks; --", result: { ok: true } });
  const task = persistence.getTask('first');
  assert.deepEqual(task.assigned_agents, ['a']);
  assert.deepEqual(task.result, { ok: true });
  assert.equal(task.status, "done'; DROP TABLE tasks; --");
  assert.equal(persistence.getTask('second').status, 'pending');
  assert.throws(() => persistence.updateTask('first', {}), /At least one/);
  assert.throws(() => persistence.updateTask('first', { assignedAgents: [], assigned_agents: [] }), /Duplicate/);
});

test('neural updates accept known data columns and reject owner/SQL changes', t => {
  const persistence = database(t);
  persistence.createAgent({ id: 'agent', name: 'agent', type: 'coder', swarmId: 'swarm' });
  persistence.db.prepare('INSERT INTO neural_networks (id,agent_id,architecture) VALUES (?,?,?)').run('n', 'agent', '{}');
  persistence.updateNeuralNetwork('n', { trainingData: { epochs: 2 }, weights: [1, 2] });
  const row = persistence.db.prepare('SELECT * FROM neural_networks WHERE id = ?').get('n');
  assert.deepEqual(JSON.parse(row.training_data), { epochs: 2 });
  assert.deepEqual(JSON.parse(row.weights), [1, 2]);
  assert.throws(() => persistence.updateNeuralNetwork('n', { agent_id: 'victim' }), /Unknown neural/);
});

test('GitHub labels and comments stay in separate literal arguments', async() => {
  const coordinator = Object.create(GHCoordinator.prototype);
  coordinator.config = { owner: 'ruvnet', repo: 'ruv-FANN', labelPrefix: 'swarm-' };
  const calls = [];
  coordinator.run = (command, args, options) => {
    calls.push({ command, args, options });
    return '[]';
  };
  await coordinator.getAvailableTasks({ label: '"$(printf injected)"', state: 'open' });
  assert.ok(calls[0].args.includes('--label="$(printf injected)"'));
  assert.equal(calls[0].command, 'gh');
  assert.notEqual(calls[0].options.shell, true);
  await coordinator.updateTaskProgress('x', 12, '"; $(printf injected)');
  assert.ok(calls[1].args.some(arg => arg.startsWith('--body=') && arg.includes('$(printf injected)')));
  await assert.rejects(coordinator.updateTaskProgress('x', '--help', 'x'), /Invalid issue/);
  coordinator.config.owner = '--help';
  await assert.rejects(coordinator.getAvailableTasks(), /Invalid GitHub/);
});


test('omitting secure preserves normal Claude permission checks', async() => {
  await withRecordingCli(async(integration, _calls, record) => {
    await integration.invokeClaudeWithPrompt('task');
    assert.deepEqual(JSON.parse(await readFile(record, 'utf8')), ['--', 'task']);
  });
});


test('WASM loading returns only bytes that passed verification despite file replacement', async() => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'ruv-wasm-'));
  const wasm = path.join(dir, 'module.wasm');
  const verified = Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]);
  const tampered = Buffer.from('tampered bytes');
  await writeFile(wasm, verified);
  const verifier = new WasmIntegrityVerifier(path.join(dir, 'checksums.json'));
  verifier.initialized = true;
  verifier.knownHashes = { get() {
    writeFileSync(wasm, tampered);
    return createHash('sha256').update(verified).digest('hex');
  } };
  try {
    assert.deepEqual(await verifier.loadVerifiedWasm(wasm), verified);
    assert.deepEqual(await readFile(wasm), tampered);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('dependency verification rejects traversal and malformed package names', async() => {
  for (const name of ['../secrets', '/tmp/pkg', '@scope/../../pkg', '', null, {}, 'name\\suffix']) {
    await assert.rejects(DependencyVerifier.verifyNpmPackage(name, '1.0.0'), /Invalid npm package name/);
  }
});


test('SQL statement cache is isolated by connection and bounded', () => {
  const pool = Object.create(SQLiteConnectionPool.prototype);
  pool.preparedStatements = new WeakMap();
  const first = { prepare: sql => ({ owner: 'first', sql }) };
  const second = { prepare: sql => ({ owner: 'second', sql }) };
  const a = pool.getPreparedStatement(first, 'SELECT 1');
  assert.equal(pool.getPreparedStatement(first, 'SELECT 1'), a);
  assert.equal(pool.getPreparedStatement(second, 'SELECT 1').owner, 'second');
  for (let i = 0; i < 256; i++) {
    pool.getPreparedStatement(first, `SELECT ${i}`);
  }
  assert.equal(pool.preparedStatements.get(first).size, 128);
});

test('failed SQL reads always release the borrowed connection', async() => {
  const pool = Object.create(SQLiteConnectionPool.prototype);
  const connection = {};
  let released = 0;
  pool.acquireReaderConnection = async() => connection;
  pool.getPreparedStatement = () => {
    throw new Error('invalid query');
  };
  pool.releaseReaderConnection = value => {
    assert.equal(value, connection); released++;
  };
  pool.emit = () => {};
  await assert.rejects(pool.read('invalid SQL'), /invalid query/);
  assert.equal(released, 1);
});


test('pool initialization failure is observable without an unhandled rejection', async() => {
  class FailedPool extends SQLiteConnectionPool {
    async initializeWriteConnection() {
      throw new Error('expected initialization failure');
    }
  }
  const pool = new FailedPool(':memory:');
  const errors = [];
  pool.on('error', error => errors.push(error.message));
  await assert.rejects(pool.initialization, /expected initialization failure/);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(errors, ['expected initialization failure']);
});
