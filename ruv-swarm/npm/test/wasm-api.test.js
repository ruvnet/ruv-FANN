import assert from 'node:assert/strict';
import { test } from 'node:test';
import { RuvSwarm } from '../src/index.js';
import { WasmBindingsLoader } from '../wasm/wasm-bindings-loader.mjs';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

test('bindings loader uses actual module memory and never substitutes predictions on failure', async t => {
  const loader = await new WasmBindingsLoader().initialize();
  assert.ok(loader.memory instanceof WebAssembly.Memory);
  assert.equal(loader.getTotalMemoryUsage(), loader.memory.buffer.byteLength);
  assert.equal(loader.isPlaceholder, false);
  const directory = await mkdtemp(path.join(os.tmpdir(), 'ruv-invalid-wasm-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const missing = new WasmBindingsLoader(directory);
  await assert.rejects(missing.initialize(), /ENOENT/);
  assert.equal(missing.initialized, false);
  assert.equal(missing.create_neural_network, undefined);
  await writeFile(path.join(directory, 'ruv_swarm_wasm_bg.wasm'), Buffer.from('invalid WASM'));
  await writeFile(path.join(directory, 'ruv_swarm_wasm.js'), 'export default async ({module_or_path}) => WebAssembly.instantiate(module_or_path)');
  await assert.rejects(missing.initialize(), WebAssembly.CompileError);
  assert.equal(missing.initialized, false);
});

for (const useSIMD of [false, true]) {
  test(`generated WASM performs actual inference and statistical forecasts (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    assert.match(RuvSwarm.getVersion(), /^\d+\.\d+\.\d+/);
    assert.ok(RuvSwarm.getMemoryUsage() > 0);
    assert.equal(typeof RuvSwarm.getRuntimeFeatures().threadsAvailable, 'boolean');
    const network = runtime.createNeuralNetwork([2, 1], 'Linear');
    try {
      network.set_weights(Float64Array.of(2, -1));
      assert.deepEqual(Array.from(network.run(Float64Array.of(3, 4))), [2]);
      assert.throws(() => network.run(Float64Array.of(3)));
      assert.throws(() => network.set_weights(Float64Array.of(NaN, 1)));
    } finally {
      network.free();
    }
    for (const [name, expected] of [['linear', 7], ['mean', 3], ['naive', 5]]) {
      const model = runtime.createForecastingModel(name);
      try {
        assert.deepEqual(Array.from(model.predict(Float64Array.of(1, 3, 5))), [expected]);
      } finally {
        model.free();
      }
    }
    assert.throws(() => runtime.createForecastingModel('lstm'));
    assert.throws(() => runtime.createNeuralNetwork([2.5, 1]));
  });
}

test('published facade preserves caller metadata and actual capability assignments', async() => {
  const runtime = await RuvSwarm.initialize();
  const swarm = await runtime.createSwarm({ name: 'actual', topology: 'ring', maxAgents: 1 });
  try {
    const agent = await swarm.spawn({ id: 'coder-1', name: 'Actual coder', type: 'coder', capabilities: ['code'] });
    assert.equal(agent.id, 'coder-1');
    assert.equal(agent.agentType, 'coder');
    assert.deepEqual(agent.getCapabilities(), ['code']);
    assert.equal(agent.status, 'idle');
    assert.deepEqual(swarm.getAgents(), ['coder-1']);
    const task = await swarm.orchestrate({ description: 'Compile the actual input', priority: 'high', requiredCapabilities: ['code'] });
    assert.equal(task.description, 'Compile the actual input');
    assert.equal(task.priority, 'high');
    assert.equal(task.status, 'assigned');
    assert.deepEqual(task.assignedAgents, ['coder-1']);
    assert.equal(agent.status, 'busy');
    agent.reset();
    assert.equal(agent.status, 'idle');
    const unmatched = await swarm.orchestrate({ description: 'Requires a different skill', requiredCapabilities: ['vision'] });
    assert.equal(unmatched.status, 'unassigned');
    assert.deepEqual(unmatched.assignedAgents, []);
    const status = swarm.getStatus();
    assert.equal(status.name, 'actual');
    assert.equal(status.topology, 'ring');
    assert.equal(status.agentCount, 1);
    await assert.rejects(swarm.spawn({ type: 'coder' }), /agent limit/);
    await assert.rejects(agent.execute({ description: 'No configured executor' }), /external task execution/);
    await assert.rejects(swarm.orchestrate({ description: 'Unsupported DAG', dependencies: ['previous'] }), /Dependency scheduling/);
  } finally {
    swarm.free();
  }
});
