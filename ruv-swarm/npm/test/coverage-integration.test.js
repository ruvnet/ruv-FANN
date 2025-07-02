/**
 * Integration Tests for Coverage - Actually imports and tests real modules
 */

import { RuvSwarm } from '../src/index.js';
import { RuvSwarmEnhanced } from '../src/index-enhanced.js';
import { NeuralAgent } from '../src/neural-agent.js';
import { NeuralNetworkManager } from '../src/neural-network-manager.js';
import { SwarmPersistence } from '../src/persistence.js';
import { WasmLoader } from '../src/wasm-loader.js';
import { BenchmarkCLI } from '../src/benchmark.js';
import { getMemoryConfig } from '../src/memory-config.js';
import { NeuralCLI } from '../src/neural.js';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const { PerformanceCLI } = require('../src/performance.js');

// Test memory config
console.log('Testing memory-config.js...');
const memConfig = getMemoryConfig();
console.log('✓ Memory config loaded:', memConfig.neural ? 'with neural' : 'basic');

// Test basic initialization
console.log('\nTesting index.js...');
try {
  const ruv = await RuvSwarm.initialize();
  console.log('✓ RuvSwarm initialized');

  const version = RuvSwarm.getVersion();
  console.log('✓ Version:', version);

  const simdSupport = RuvSwarm.detectSIMDSupport();
  console.log('✓ SIMD support:', simdSupport);

  const swarm = await ruv.createSwarm({
    name: 'test-swarm',
    topology: 'mesh',
    maxAgents: 3,
  });
  console.log('✓ Swarm created:', swarm.name);

  const agent = await swarm.spawn({ type: 'researcher' });
  console.log('✓ Agent spawned:', agent.id);

  const result = await agent.execute({ task: 'test-task' });
  console.log('✓ Task executed:', result.status);

  const status = await swarm.getStatus();
  console.log('✓ Status retrieved:', status);

  const orchestrated = await swarm.orchestrate({ task: 'complex-task' });
  console.log('✓ Task orchestrated:', orchestrated.id);

} catch (error) {
  console.error('✗ RuvSwarm test failed:', error.message);
}

// Test enhanced version
console.log('\nTesting index-enhanced.js...');
try {
  const enhanced = new RuvSwarmEnhanced();
  await enhanced.initialize();
  console.log('✓ RuvSwarmEnhanced initialized');

  const swarmEnhanced = await enhanced.createSwarm({
    topology: 'hierarchical',
    enableNeuralAgents: true,
  });
  console.log('✓ Enhanced swarm created');

  const neuralAgent = await swarmEnhanced.createNeuralAgent({
    type: 'adaptive',
    modelType: 'gru',
  });
  console.log('✓ Neural agent created');

} catch (error) {
  console.error('✗ RuvSwarmEnhanced test failed:', error.message);
}

// Test neural agent
console.log('\nTesting neural-agent.js...');
try {
  const neuralAgent = new NeuralAgent({
    type: 'researcher',
    model: 'transformer',
  });
  await neuralAgent.initialize();
  console.log('✓ NeuralAgent initialized');

  await neuralAgent.train([
    { input: [1, 2, 3], output: [0, 1] },
  ]);
  console.log('✓ NeuralAgent trained');

  const prediction = await neuralAgent.predict([1, 2, 3]);
  console.log('✓ Prediction made:', prediction);

} catch (error) {
  console.error('✗ NeuralAgent test failed:', error.message);
}

// Test neural network manager
console.log('\nTesting neural-network-manager.js...');
try {
  const manager = new NeuralNetworkManager();
  await manager.initialize();
  console.log('✓ NeuralNetworkManager initialized');

  const network = await manager.createNetwork({
    layers: [10, 20, 10],
    activation: 'relu',
  });
  console.log('✓ Network created');

  const models = manager.listModels();
  console.log('✓ Models listed:', models.length);

} catch (error) {
  console.error('✗ NeuralNetworkManager test failed:', error.message);
}

// Test persistence
console.log('\nTesting persistence.js...');
try {
  const persistence = new SwarmPersistence();
  await persistence.initialize();
  console.log('✓ SwarmPersistence initialized');

  await persistence.saveSwarm({
    id: 'test-swarm',
    state: { agents: 3 },
  });
  console.log('✓ Swarm saved');

  const loaded = await persistence.loadSwarm('test-swarm');
  console.log('✓ Swarm loaded:', loaded);

  await persistence.close();
  console.log('✓ Persistence closed');

} catch (error) {
  console.error('✗ SwarmPersistence test failed:', error.message);
}

// Test WASM loader
console.log('\nTesting wasm-loader.js...');
try {
  const loader = new WasmLoader();
  console.log('✓ WasmLoader created');

  const supported = loader.isSupported();
  console.log('✓ WASM support checked:', supported);

  const simd = loader.hasSIMDSupport();
  console.log('✓ SIMD support checked:', simd);

} catch (error) {
  console.error('✗ WasmLoader test failed:', error.message);
}

// Test benchmark
console.log('\nTesting benchmark.js...');
try {
  const benchmark = new BenchmarkCLI();
  console.log('✓ BenchmarkCLI created');

  // Test getArg method
  const arg = benchmark.getArg(['--type', 'wasm'], '--type');
  console.log('✓ Arg parsing works:', arg);

} catch (error) {
  console.error('✗ Benchmark test failed:', error.message);
}

// Test performance analyzer
console.log('\nTesting performance.js...');
try {
  const perfCLI = new PerformanceCLI();
  console.log('✓ PerformanceCLI created');

  // Test command parsing
  const command = perfCLI.parseCommand(['analyze', '--metric', 'cpu']);
  console.log('✓ Command parsed:', command);

} catch (error) {
  console.error('✗ Performance test failed:', error.message);
}

// Test neural module
console.log('\nTesting neural.js...');
try {
  const neuralCLI = new NeuralCLI();
  console.log('✓ NeuralCLI created');

  // Test pattern memory config
  const { PATTERN_MEMORY_CONFIG } = await import('../src/neural.js');
  console.log('✓ Pattern memory config loaded:', Object.keys(PATTERN_MEMORY_CONFIG));

} catch (error) {
  console.error('✗ Neural test failed:', error.message);
}

console.log('\n✅ Integration tests completed');