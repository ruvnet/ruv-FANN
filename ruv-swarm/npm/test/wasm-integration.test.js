/**
 * Comprehensive test suite for WASM integration
 * Tests progressive loading, neural networks, and swarm orchestration
 */

import { RuvSwarm } from '../src/index-enhanced';
import { WasmModuleLoader } from '../src/wasm-loader';
import { EnhancedMCPTools } from '../src/mcp-tools-enhanced';
import { NeuralNetworkManager } from '../src/neural-network-manager';

import assert from 'assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Test utilities
function assertApprox(actual, expected, tolerance = 0.01) {
  assert(Math.abs(actual - expected) < tolerance,
    `Expected ${actual} to be approximately ${expected} (tolerance: ${tolerance})`);
}

async function measureTime(fn) {
  const start = performance.now();
  const result = await fn();
  const time = performance.now() - start;
  return { result, time };
}

// Test suite
class WasmIntegrationTests {
  constructor() {
    this.testResults = [];
    this.ruvSwarm = null;
    this.mcpTools = null;
  }

  async setup() {
    console.log('🔧 Setting up test environment...');

    // Initialize RuvSwarm with test configuration
    this.ruvSwarm = await RuvSwarm.initialize({
      loadingStrategy: 'progressive',
      enablePersistence: false, // Disable for tests
      enableNeuralNetworks: true,
      enableForecasting: false, // Start with disabled
      useSIMD: true,
      debug: true,
    });

    // Initialize MCP tools
    this.mcpTools = new EnhancedMCPTools();
    await this.mcpTools.initialize();

    console.log('✅ Test environment ready\n');
  }

  async runAll() {
    await this.setup();

    const tests = [
      this.testWasmModuleLoading,
      this.testProgressiveLoading,
      this.testSwarmCreation,
      this.testAgentSpawning,
      this.testTaskOrchestration,
      this.testNeuralNetworks,
      this.testMCPTools,
      this.testMemoryManagement,
      this.testPerformance,
      this.testBackwardCompatibility,
    ];

    let passed = 0;
    let failed = 0;

    for (const test of tests) {
      try {
        console.log(`\n🧪 Running: ${test.name}`);
        const { time } = await measureTime(() => test.call(this));
        console.log(`✅ ${test.name} passed (${time.toFixed(1)}ms)`);
        passed++;
        this.testResults.push({ test: test.name, status: 'passed', time });
      } catch (error) {
        console.error(`❌ ${test.name} failed: ${error.message}`);
        failed++;
        this.testResults.push({ test: test.name, status: 'failed', error: error.message });
      }
    }

    this.printSummary(passed, failed);
  }

  async testWasmModuleLoading() {
    // Test 1: Module loader initialization
    const loader = new WasmModuleLoader();
    assert(loader.modules instanceof Map, 'Modules should be a Map');
    assert(loader.moduleManifest.core, 'Core module should be in manifest');

    // Test 2: Progressive loading strategy
    await loader.initialize('progressive');
    assert.equal(loader.loadingStrategy, 'progressive');

    // Test 3: Core module loading
    const coreModule = await loader.loadModule('core');
    assert(coreModule, 'Core module should load');
    assert(coreModule.exports || coreModule.isPlaceholder, 'Core module should have exports or be placeholder');

    // Test 4: Module status
    const status = loader.getModuleStatus();
    assert(status.core.loaded || status.core.loading, 'Core module should be loaded or loading');
  }

  async testProgressiveLoading() {
    const loader = new WasmModuleLoader();

    // Test 1: Progressive strategy loads only core modules
    await loader.initialize('progressive');
    const status1 = loader.getModuleStatus();
    assert(status1.core.loaded || status1.core.loading, 'Core should be loaded');
    assert(!status1.neural.loaded, 'Neural should not be loaded yet');

    // Test 2: On-demand loading
    const neuralModule = await loader.loadModule('neural');
    assert(neuralModule, 'Neural module should load on demand');

    // Test 3: Memory usage tracking
    const memoryUsage = loader.getTotalMemoryUsage();
    assert(typeof memoryUsage === 'number', 'Memory usage should be a number');
  }

  async testSwarmCreation() {
    // Test 1: Create swarm with default config
    const swarm1 = await this.ruvSwarm.createSwarm({
      name: 'test-swarm-1',
      topology: 'mesh',
      maxAgents: 5,
    });
    assert(swarm1.id, 'Swarm should have an ID');
    assert.equal(this.ruvSwarm.activeSwarms.size, 1, 'Should have 1 active swarm');

    // Test 2: Create swarm with different topologies
    const topologies = ['star', 'hierarchical', 'ring'];
    for (const topology of topologies) {
      const swarm = await this.ruvSwarm.createSwarm({
        name: `test-${topology}`,
        topology,
        maxAgents: 3,
      });
      assert(swarm.id, `${topology} swarm should be created`);
    }

    // Test 3: Swarm status
    const status = await swarm1.getStatus();
    assert(status.agents, 'Status should have agents info');
    assert(status.tasks, 'Status should have tasks info');
  }

  async testAgentSpawning() {
    const swarm = await this.ruvSwarm.createSwarm({
      name: 'agent-test-swarm',
      maxAgents: 10,
    });

    // Test 1: Spawn different agent types
    const agentTypes = ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'];
    const agents = [];

    for (const type of agentTypes) {
      const agent = await swarm.spawn({
        type,
        name: `test-${type}`,
        enableNeuralNetwork: true,
      });
      assert(agent.id, `${type} agent should have ID`);
      assert.equal(agent.type, type, `Agent type should be ${type}`);
      assert(agent.cognitivePattern, 'Agent should have cognitive pattern');
      agents.push(agent);
    }

    // Test 2: Agent metrics
    for (const agent of agents) {
      const metrics = await agent.getMetrics();
      assert(typeof metrics.memoryUsage === 'number', 'Memory usage should be a number');
    }

    // Test 3: Swarm agent count
    assert.equal(swarm.agents.size, agentTypes.length, 'Swarm should have all spawned agents');
  }

  async testTaskOrchestration() {
    const swarm = await this.ruvSwarm.createSwarm({
      name: 'task-test-swarm',
      maxAgents: 5,
    });

    // Spawn some agents
    await swarm.spawn({ type: 'researcher' });
    await swarm.spawn({ type: 'coder' });
    await swarm.spawn({ type: 'analyst' });

    // Test 1: Create task
    const task = await swarm.orchestrate({
      description: 'Test task for orchestration',
      priority: 'high',
    });
    assert(task.id, 'Task should have ID');
    assert.equal(task.description, 'Test task for orchestration');

    // Test 2: Task status
    const status = await task.getStatus();
    assert(status.id, 'Task status should have ID');
    assert(typeof status.progress === 'number', 'Progress should be a number');

    // Test 3: Multiple tasks with dependencies
    const task1 = await swarm.orchestrate({
      description: 'Parent task',
      priority: 'medium',
    });

    const task2 = await swarm.orchestrate({
      description: 'Child task',
      dependencies: [task1.id],
    });
    assert(task2.id !== task1.id, 'Tasks should have unique IDs');
  }

  async testNeuralNetworks() {
    if (!this.ruvSwarm.features.neural_networks) {
      console.log('⚠️  Neural networks not available, skipping detailed tests');
      return;
    }

    const nnManager = new NeuralNetworkManager(this.ruvSwarm.wasmLoader);

    // Test 1: Create neural network for agent
    const network = await nnManager.createAgentNeuralNetwork('test-agent-1', {
      template: 'deep_analyzer',
    });
    assert(network, 'Neural network should be created');

    // Test 2: Forward pass
    const input = new Array(128).fill(0.5);
    const output = await network.forward(input);
    assert(output instanceof Float32Array || Array.isArray(output), 'Output should be array-like');

    // Test 3: Training
    const trainingData = {
      samples: Array(10).fill(null).map(() => ({
        input: new Array(128).fill(Math.random()),
        target: new Array(128).fill(Math.random()),
      })),
    };

    const metrics = await network.train(trainingData, {
      epochs: 1,
      batchSize: 5,
    });
    assert(typeof metrics.loss === 'number', 'Loss should be a number');
    assert(metrics.epochs_trained >= 1, 'Should have trained at least 1 epoch');

    // Test 4: Collaborative learning setup
    const network2 = await nnManager.createAgentNeuralNetwork('test-agent-2');
    const session = await nnManager.enableCollaborativeLearning(
      ['test-agent-1', 'test-agent-2'],
      { strategy: 'federated' },
    );
    assert(session.id, 'Collaborative session should have ID');
  }

  async testMCPTools() {
    // Test 1: swarm_init
    const initResult = await this.mcpTools.swarm_init({
      topology: 'mesh',
      maxAgents: 3,
    });
    assert(initResult.id, 'Swarm init should return ID');
    assert(initResult.features, 'Should include features');

    // Test 2: agent_spawn
    const spawnResult = await this.mcpTools.agent_spawn({
      type: 'researcher',
      name: 'mcp-test-agent',
    });
    assert(spawnResult.agent.id, 'Agent spawn should return agent ID');
    assert(spawnResult.swarm_info, 'Should include swarm info');

    // Test 3: task_orchestrate
    const taskResult = await this.mcpTools.task_orchestrate({
      task: 'Test MCP task orchestration',
      priority: 'medium',
    });
    assert(taskResult.taskId, 'Task orchestrate should return task ID');

    // Test 4: swarm_status
    const statusResult = await this.mcpTools.swarm_status({
      verbose: true,
    });
    assert(statusResult.active_swarms >= 0, 'Should have active swarms count');

    // Test 5: features_detect
    const features = await this.mcpTools.features_detect({
      category: 'all',
    });
    assert(features.runtime, 'Should detect runtime features');
    assert(features.wasm, 'Should detect WASM features');

    // Test 6: benchmark_run
    const benchmarks = await this.mcpTools.benchmark_run({
      type: 'wasm',
      iterations: 3,
    });
    assert(benchmarks.results, 'Should have benchmark results');
  }

  async testMemoryManagement() {
    // Test 1: Initial memory usage
    const initialMemory = await this.mcpTools.memory_usage({ detail: 'summary' });
    assert(typeof initialMemory.total_mb === 'number', 'Total memory should be a number');
    assert(typeof initialMemory.wasm_mb === 'number', 'WASM memory should be a number');

    // Test 2: Memory growth with swarm creation
    const beforeSwarms = initialMemory.total_mb;

    // Create multiple swarms
    for (let i = 0; i < 3; i++) {
      await this.ruvSwarm.createSwarm({
        name: `memory-test-${i}`,
        maxAgents: 5,
      });
    }

    const afterSwarms = await this.mcpTools.memory_usage({ detail: 'summary' });
    assert(afterSwarms.total_mb >= beforeSwarms, 'Memory should increase or stay same');

    // Test 3: Detailed memory report
    const detailedMemory = await this.mcpTools.memory_usage({ detail: 'detailed' });
    assert(detailedMemory.wasm_modules, 'Should have WASM modules breakdown');

    // Test 4: Per-agent memory
    const swarm = await this.ruvSwarm.createSwarm({ name: 'agent-memory-test' });
    await swarm.spawn({ type: 'researcher' });
    await swarm.spawn({ type: 'coder' });

    const agentMemory = await this.mcpTools.memory_usage({ detail: 'by-agent' });
    assert(Array.isArray(agentMemory.agents), 'Should have agents array');
    assert(agentMemory.agents.length >= 2, 'Should have at least 2 agents');
  }

  async testPerformance() {
    // Test 1: Swarm creation performance
    const swarmTimes = [];
    for (let i = 0; i < 5; i++) {
      const { time } = await measureTime(() =>
        this.ruvSwarm.createSwarm({
          name: `perf-swarm-${i}`,
          maxAgents: 10,
        }),
      );
      swarmTimes.push(time);
    }

    const avgSwarmTime = swarmTimes.reduce((a, b) => a + b) / swarmTimes.length;
    assert(avgSwarmTime < 1000, 'Average swarm creation should be under 1s');

    // Test 2: Agent spawning performance
    const swarm = await this.ruvSwarm.createSwarm({ name: 'perf-agent-test' });
    const agentTimes = [];

    for (let i = 0; i < 10; i++) {
      const { time } = await measureTime(() =>
        swarm.spawn({ type: 'researcher' }),
      );
      agentTimes.push(time);
    }

    const avgAgentTime = agentTimes.reduce((a, b) => a + b) / agentTimes.length;
    assert(avgAgentTime < 500, 'Average agent spawn should be under 500ms');

    // Test 3: WASM module loading performance
    const loader = new WasmModuleLoader();
    const { time: loadTime } = await measureTime(() =>
      loader.initialize('progressive'),
    );
    assert(loadTime < 2000, 'Progressive loading should complete under 2s');
  }

  async testBackwardCompatibility() {
    // Test 1: Legacy API still works
    const { RuvSwarm: LegacyRuvSwarm } = await import('../src/index.js');
    assert(LegacyRuvSwarm, 'Legacy RuvSwarm should be available');

    // Test 2: Old initialization pattern
    const legacy = await LegacyRuvSwarm.initialize({
      wasmPath: path.join(__dirname, '..', 'wasm'),
      useSIMD: false,
      debug: false,
    });
    assert(legacy, 'Legacy initialization should work');

    // Test 3: Check version compatibility
    const version = RuvSwarm.getVersion();
    assert(version === '0.2.0', 'Version should be 0.2.0');
  }

  printSummary(passed, failed) {
    console.log(`\n${ '='.repeat(60)}`);
    console.log('📊 Test Summary');
    console.log('='.repeat(60));
    console.log(`✅ Passed: ${passed}`);
    console.log(`❌ Failed: ${failed}`);
    console.log(`📈 Success Rate: ${((passed / (passed + failed)) * 100).toFixed(1)}%`);

    console.log('\n📋 Detailed Results:');
    this.testResults.forEach(result => {
      const icon = result.status === 'passed' ? '✅' : '❌';
      const time = result.time ? ` (${result.time.toFixed(1)}ms)` : '';
      const error = result.error ? ` - ${result.error}` : '';
      console.log(`${icon} ${result.test}${time}${error}`);
    });

    console.log('\n💾 Memory Usage:');
    console.log(`Peak Memory: ${(process.memoryUsage().heapUsed / 1024 / 1024).toFixed(2)}MB`);

    if (failed > 0) {
      process.exit(1);
    }
  }
}

// Run tests if called directly
// Direct execution block
{
  const tests = new WasmIntegrationTests();
  tests.runAll().catch(error => {
    console.error('❌ Test suite failed:', error);
    process.exit(1);
  });
}

export { WasmIntegrationTests };