#!/usr/bin/env node

/**
 * Docker Performance Test Suite for ruv-swarm v1.0.6
 * Comprehensive performance benchmarks across all features
 */

import { RuvSwarm, NeuralAgent } from '../src/index.js';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { performance } from 'perf_hooks';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('================================================');
console.log('ruv-swarm v1.0.6 Performance Benchmark Suite');
console.log('================================================');
console.log(`Date: ${new Date().toISOString()}`);
console.log(`Node Version: ${process.version}`);
console.log(`Platform: ${process.platform}`);
console.log(`Architecture: ${process.arch}`);
console.log('');

const results = {
  testSuite: 'performance-benchmarks',
  version: '1.0.6',
  timestamp: new Date().toISOString(),
  environment: {
    nodeVersion: process.version,
    platform: process.platform,
    arch: process.arch,
    cpus: require('os').cpus().length,
    memory: require('os').totalmem(),
  },
  benchmarks: [],
  summary: {},
};

// Benchmark utilities
function benchmark(name, fn, iterations = 1000) {
  return new Promise(async(resolve) => {
    console.log(`Running: ${name}`);
    const timings = [];

    // Warmup
    for (let i = 0; i < 10; i++) {
      await fn();
    }

    // Actual benchmark
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await fn();
      const end = performance.now();
      timings.push(end - start);

      if (i % 100 === 0) {
        process.stdout.write('.');
      }
    }
    console.log(' Done!');

    // Calculate statistics
    timings.sort((a, b) => a - b);
    const stats = {
      name,
      iterations,
      min: timings[0],
      max: timings[timings.length - 1],
      mean: timings.reduce((a, b) => a + b) / timings.length,
      median: timings[Math.floor(timings.length / 2)],
      p95: timings[Math.floor(timings.length * 0.95)],
      p99: timings[Math.floor(timings.length * 0.99)],
    };

    results.benchmarks.push(stats);
    console.log(`  Mean: ${stats.mean.toFixed(3)}ms, P95: ${stats.p95.toFixed(3)}ms`);
    resolve(stats);
  });
}

// Benchmark tests
async function benchmarkSwarmCreation() {
  console.log('\n1. Swarm Creation Benchmarks');
  console.log('============================');

  await benchmark('Swarm Creation (Small)', () => {
    const swarm = new RuvSwarm({ maxAgents: 4 });
    return swarm;
  });

  await benchmark('Swarm Creation (Medium)', () => {
    const swarm = new RuvSwarm({ maxAgents: 16 });
    return swarm;
  });

  await benchmark('Swarm Creation (Large)', () => {
    const swarm = new RuvSwarm({ maxAgents: 64 });
    return swarm;
  }, 100);
}

async function benchmarkAgentOperations() {
  console.log('\n2. Agent Operation Benchmarks');
  console.log('=============================');

  const swarm = new RuvSwarm({ maxAgents: 32 });

  await benchmark('Agent Spawn', () => {
    swarm.spawnAgent(`agent-${Date.now()}`, 'researcher');
  });

  await benchmark('Agent Communication', async() => {
    const agent1 = swarm.agents[0];
    const agent2 = swarm.agents[1];
    if (agent1 && agent2) {
      await agent1.sendMessage(agent2.id, { type: 'test', data: 'benchmark' });
    }
  });

  await benchmark('Agent Task Assignment', async() => {
    const agent = swarm.agents[0];
    if (agent) {
      await agent.assignTask({ type: 'analyze', data: 'benchmark-data' });
    }
  });
}

async function benchmarkNeuralOperations() {
  console.log('\n3. Neural Network Benchmarks');
  console.log('============================');

  const agent = new NeuralAgent('neural-bench', 'researcher');
  await agent.initialize();

  await benchmark('Neural Forward Pass', async() => {
    const input = new Float32Array(128).fill(0.5);
    await agent.neuralNetwork.forward(input);
  });

  await benchmark('Neural Training Step', async() => {
    const input = new Float32Array(128).fill(0.5);
    const target = new Float32Array(64).fill(0.8);
    await agent.neuralNetwork.train(input, target);
  }, 100);

  await benchmark('Pattern Recognition', async() => {
    const pattern = { type: 'test', features: new Array(32).fill(0.5) };
    await agent.recognizePattern(pattern);
  });
}

async function benchmarkMemoryOperations() {
  console.log('\n4. Memory Operation Benchmarks');
  console.log('==============================');

  const swarm = new RuvSwarm({ maxAgents: 8 });

  await benchmark('Memory Store', () => {
    const key = `key-${Date.now()}`;
    const value = { data: 'test', timestamp: Date.now(), array: new Array(100).fill(0) };
    swarm.memory.store(key, value);
  });

  await benchmark('Memory Retrieve', () => {
    const key = Object.keys(swarm.memory.data)[0];
    return swarm.memory.retrieve(key);
  });

  await benchmark('Memory Pattern Match', () => {
    return swarm.memory.search('test');
  });
}

async function benchmarkTaskOrchestration() {
  console.log('\n5. Task Orchestration Benchmarks');
  console.log('================================');

  const swarm = new RuvSwarm({
    topology: 'hierarchical',
    maxAgents: 16,
  });

  // Spawn agents
  for (let i = 0; i < 8; i++) {
    swarm.spawnAgent(`worker-${i}`, 'researcher');
  }

  await benchmark('Simple Task Orchestration', async() => {
    await swarm.orchestrateTask({
      type: 'analyze',
      data: 'benchmark-task',
      priority: 'high',
    });
  }, 100);

  await benchmark('Complex Task Orchestration', async() => {
    await swarm.orchestrateTask({
      type: 'multi-phase',
      phases: ['collect', 'analyze', 'synthesize'],
      data: new Array(1000).fill(0),
      priority: 'critical',
    });
  }, 10);
}

async function benchmarkWASMSpecific() {
  console.log('\n6. WASM-Specific Benchmarks');
  console.log('===========================');

  // Direct WASM function calls if available
  try {
    const wasmModule = global._wasmModule || global.__ruv_swarm_wasm;
    if (wasmModule && wasmModule.benchmark_operation) {
      await benchmark('Direct WASM Call', () => {
        return wasmModule.benchmark_operation();
      });
    } else {
      console.log('  ⚠️  Direct WASM benchmarks not available');
    }
  } catch (error) {
    console.log('  ⚠️  WASM benchmark error:', error.message);
  }
}

async function generatePerformanceReport() {
  // Calculate aggregate statistics
  const aggregateStats = {
    totalBenchmarks: results.benchmarks.length,
    avgMeanTime: results.benchmarks.reduce((sum, b) => sum + b.mean, 0) / results.benchmarks.length,
    avgP95Time: results.benchmarks.reduce((sum, b) => sum + b.p95, 0) / results.benchmarks.length,
    fastestOperation: results.benchmarks.reduce((min, b) => b.mean < min.mean ? b : min),
    slowestOperation: results.benchmarks.reduce((max, b) => b.mean > max.mean ? b : max),
  };

  results.summary = aggregateStats;

  // Performance grade
  let grade = 'A';
  if (aggregateStats.avgMeanTime > 10) {
    grade = 'B';
  }
  if (aggregateStats.avgMeanTime > 50) {
    grade = 'C';
  }
  if (aggregateStats.avgMeanTime > 100) {
    grade = 'D';
  }
  if (aggregateStats.avgMeanTime > 500) {
    grade = 'F';
  }

  results.summary.performanceGrade = grade;

  // Save results
  const resultsPath = path.join(__dirname, '..', 'test-results', 'performance-benchmarks.json');
  await fs.mkdir(path.dirname(resultsPath), { recursive: true });
  await fs.writeFile(resultsPath, JSON.stringify(results, null, 2));

  console.log('\n================================================');
  console.log('Performance Benchmark Summary');
  console.log('================================================');
  console.log(`Total Benchmarks: ${aggregateStats.totalBenchmarks}`);
  console.log(`Average Mean Time: ${aggregateStats.avgMeanTime.toFixed(3)}ms`);
  console.log(`Average P95 Time: ${aggregateStats.avgP95Time.toFixed(3)}ms`);
  console.log(`Fastest Operation: ${aggregateStats.fastestOperation.name} (${aggregateStats.fastestOperation.mean.toFixed(3)}ms)`);
  console.log(`Slowest Operation: ${aggregateStats.slowestOperation.name} (${aggregateStats.slowestOperation.mean.toFixed(3)}ms)`);
  console.log(`Performance Grade: ${grade}`);
  console.log('');
  console.log(`Results saved to: ${resultsPath}`);
}

// Run all benchmarks
async function runBenchmarks() {
  try {
    await benchmarkSwarmCreation();
    await benchmarkAgentOperations();
    await benchmarkNeuralOperations();
    await benchmarkMemoryOperations();
    await benchmarkTaskOrchestration();
    await benchmarkWASMSpecific();
    await generatePerformanceReport();
  } catch (error) {
    console.error('Benchmark suite failed:', error);
    process.exit(1);
  }
}

runBenchmarks();
