import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { spawn, spawnSync } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import os from 'node:os';
import path from 'node:path';
import { EnhancedMCPTools } from '../src/mcp-tools-enhanced.js';
import { DAAService } from '../src/daa-service.js';

const cli = fileURLToPath(new URL('../bin/ruv-swarm-secure.js', import.meta.url));
async function directory(t, cleanup = true) {
  const result = await mkdtemp(path.join(os.tmpdir(), 'ruv-node-runtime-'));
  if (cleanup) {
    t.after(() => rm(result, { recursive: true, force: true }));
  }
  return result;
}

test('Node MCP persists actual assignments and executes real inference, forecasts, and benchmarks', async t => {
  const location = await directory(t, false);
  const dbPath = path.join(location, 'state.sqlite');
  let service = new EnhancedMCPTools(null, { dbPath });
  t.after(async() => {
    await service.destroy(); await rm(location, { recursive: true, force: true });
  });
  await service.initialize();
  const names = service.getAllToolDefinitions().map(tool => tool.name);
  assert.ok(names.includes('neural_infer') && names.includes('forecast_predict'));
  assert.ok(!names.includes('neural_train') && !names.some(name => name.startsWith('daa_')));
  const swarm = await service.tools.swarm_init({ topology: 'mesh', maxAgents: 2 });
  const agent = await service.tools.agent_spawn({ swarmId: swarm.id, type: 'coder', name: 'Actual coder', capabilities: ['code'] });
  const task = await service.tools.task_orchestrate({ swarmId: swarm.id, task: 'Use the real requested description', requiredCapabilities: ['code'] });
  assert.equal(task.status, 'assigned');
  assert.deepEqual(task.assignedAgents, [agent.id]);
  const results = await service.tools.task_results({ taskId: task.id });
  assert.equal(results.status, 'assigned');
  assert.equal(results.result, null);
  assert.deepEqual(results.recorded_results, []);
  await assert.rejects(service.tools.task_results({ taskId: 'missing' }), /Task not found/);
  assert.deepEqual((await service.tools.neural_infer({ layers: [2, 1], weights: [2, -1], input: [3, 4] })).output, [2]);
  assert.deepEqual((await service.tools.forecast_predict({ model: 'linear', input: [1, 3, 5] })).output, [7]);
  const benchmark = await service.tools.benchmark_run({ iterations: 10 });
  assert.equal(benchmark.checksum, 20);
  assert.ok(benchmark.elapsed_ms >= 0 && benchmark.operations_per_second > 0);
  assert.equal((await service.tools.features_detect()).neural_training, false);
  await assert.rejects(service.tools.neural_train({}), { code: 'UNSUPPORTED_FEATURE' });
  await assert.rejects(service.tools.swarm_init({ enableNeuralAgents: true }), /Unsupported argument/);
  await service.destroy();
  service = new EnhancedMCPTools(null, { dbPath });
  await service.initialize();
  assert.equal((await service.tools.swarm_status({})).total_agents, 1);
  assert.equal((await service.tools.task_status({ taskId: task.id })).status, 'assigned');
  assert.equal((await service.tools.agent_list({})).agents[0].status, 'busy');
  await service.tools.agent_update_status({ agentId: agent.id, status: 'idle' });
  assert.equal((await service.tools.agent_list({})).agents[0].status, 'idle');
});

test('public CLI reports real values and rejects simulated training and autonomous execution', async t => {
  const cwd = await directory(t);
  for (const args of [['init', 'mesh', '2'], ['spawn', 'coder', 'Local coder'], ['orchestrate', 'Assign this actual task'], ['status']]) {
    const run = spawnSync(process.execPath, [cli, ...args], { cwd, encoding: 'utf8', timeout: 15000 });
    assert.equal(run.status, 0, run.stderr);
    const record = JSON.parse(run.stdout);
    if (args[0] === 'orchestrate') {
      assert.equal(record.status, 'assigned');
    }
    if (args[0] === 'status') {
      assert.equal(record.total_agents, 1);
    }
  }
  for (const [args, expected] of [
    [['neural', 'infer', '--layers', '2,1', '--weights', '2,-1', '--input', '3,4'], 2],
    [['neural', 'forecast', '--model', 'linear', '--input', '1,3,5'], 7],
  ]) {
    const run = spawnSync(process.execPath, [cli, ...args], { cwd, encoding: 'utf8', timeout: 15000 });
    assert.equal(run.status, 0, run.stderr);
    assert.deepEqual(JSON.parse(run.stdout).output, [expected]);
  }
  const training = spawnSync(process.execPath, [cli, 'neural', 'train'], { cwd, encoding: 'utf8', timeout: 15000 });
  assert.notEqual(training.status, 0);
  assert.match(training.stderr, /not implemented/);
  const daa = new DAAService();
  assert.equal(daa.getCapabilities().autonomousLearning, false);
  await assert.rejects(daa.initialize(), { code: 'UNSUPPORTED_FEATURE' });
});

test('MCP metadata quotas roll back assignments and bound monitoring snapshots', async t => {
  const location = await directory(t, false);
  const service = new EnhancedMCPTools(null, { dbPath: path.join(location, 'limits.sqlite'),
    maxSwarms: 1, maxAgents: 1, maxTasks: 1 });
  t.after(async() => {
    await service.destroy(); await rm(location, { recursive: true, force: true });
  });
  const swarm = await service.tools.swarm_init({ maxAgents: 2 });
  await assert.rejects(service.tools.swarm_init({}), /Swarm resource limit/);
  const agent = await service.tools.agent_spawn({ swarmId: swarm.id });
  await assert.rejects(service.tools.agent_spawn({ swarmId: swarm.id }), /Agent resource limit/);
  await service.tools.task_orchestrate({ task: 'First assignment' });
  await service.tools.agent_update_status({ agentId: agent.id, status: 'idle' });
  await assert.rejects(service.tools.task_orchestrate({ task: 'Over quota' }), /Task resource limit/);
  assert.equal((await service.tools.agent_list({})).agents[0].status, 'idle');
  assert.equal((await service.tools.swarm_status({})).swarms[0].agent_details[0].status, 'idle');
  assert.deepEqual((await service.tools.persistence_stats({})).counts, { swarms: 1, agents: 1, tasks: 1, task_results: 0 });
  await assert.rejects(service.tools.swarm_monitor({ duration: 1, interval: 1 }), /at least 10/);
  const monitor = await service.tools.swarm_monitor({ duration: 25, interval: 10 });
  assert.ok(monitor.snapshots.length > 0 && monitor.snapshots.length <= 100);
  assert.equal(monitor.snapshots[0].agent_counts.idle, 1);
  assert.ok(!Object.hasOwn(monitor.snapshots[0], 'agent_details'));
});

test('real stdio MCP only invokes registered tools and returns failures as JSON-RPC errors', async t => {
  const cwd = await directory(t, false);
  const child = spawn(process.execPath, [cli, 'mcp', 'start'], { cwd,
    env: { ...process.env, RUV_SWARM_DB_PATH: path.join(cwd, 'mcp.sqlite') }, stdio: ['pipe', 'pipe', 'pipe'] });
  const exited = new Promise(resolve => child.once('close', resolve));
  t.after(async() => {
    child.kill('SIGTERM'); await exited; await rm(cwd, { recursive: true, force: true });
  });
  let stderr = '';
  child.stderr.on('data', data => {
    stderr += data;
  });
  let id = 0;
  const pending = new Map();
  createInterface({ input: child.stdout }).on('line', line => {
    const frame = JSON.parse(line);
    if (pending.has(frame.id)) {
      pending.get(frame.id)(frame); pending.delete(frame.id);
    }
  });
  async function request(method, params = {}) {
    const requestId = ++id;
    const frame = new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(requestId); reject(new Error(`MCP timeout: ${stderr}`));
      }, 15000);
      pending.set(requestId, value => {
        clearTimeout(timer); resolve(value);
      });
    });
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: requestId, method, params })}\n`);
    return frame;
  }
  const tools = await request('tools/list');
  assert.ok(tools.result.tools.some(tool => tool.name === 'neural_infer'));
  assert.ok(!tools.result.tools.some(tool => tool.name === 'neural_train'));
  const inference = await request('tools/call', { name: 'neural_infer', arguments: { layers: [2, 1], weights: [2, -1], input: [3, 4] } });
  assert.deepEqual(JSON.parse(inference.result.content[0].text).output, [2]);
  const missing = await request('tools/call', { name: 'task_results', arguments: { taskId: 'missing' } });
  assert.ok(missing.error && !missing.result);
  assert.match(missing.error.message, /Task not found/);
  const training = await request('tools/call', { name: 'neural_train', arguments: {} });
  assert.equal(training.error.data.code, 'UNSUPPORTED_FEATURE');
  for (const name of ['destroy', 'constructor', 'toString']) {
    const result = await request('tools/call', { name });
    assert.equal(result.error.code, -32601);
  }
});

test('every shipped CLI drains a real inference reply before exiting on stdin EOF', async t => {
  const cwd = await directory(t);
  for (const filename of ['ruv-swarm-secure.js', 'ruv-swarm-clean.js', 'ruv-swarm-secure-heartbeat.js']) {
    const executable = fileURLToPath(new URL(`../bin/${filename}`, import.meta.url));
    const request = { jsonrpc: '2.0', id: 42, method: 'tools/call', params: {
      name: 'neural_infer', arguments: { layers: [2, 1], weights: [2, -1], input: [3, 4] },
    } };
    const result = spawnSync(process.execPath, [executable, 'mcp', 'start'], {
      cwd, input: `${JSON.stringify(request)}\n`, encoding: 'utf8', timeout: 10000,
    });
    assert.equal(result.status, 0, `${filename}: ${result.error || result.stderr}`);
    const reply = result.stdout.trim().split('\n').map(JSON.parse).find(frame => frame.id === 42);
    assert.deepEqual(JSON.parse(reply.result.content[0].text).output, [2]);
  }
});
