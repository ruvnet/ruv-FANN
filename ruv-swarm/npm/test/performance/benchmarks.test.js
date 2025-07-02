/**
 * Performance benchmarks for ruv-swarm
 */

import { RuvSwarm  } from '../../src/index-enhanced';
import { NeuralNetwork  } from '../../src/neural-agent';
import { SwarmPersistence  } from '../../src/persistence';
import assert from 'assert';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Benchmark utilities
class BenchmarkRunner {
  constructor(name) {
    this.name = name;
    this.results = [];
  }

  async run(fn, iterations = 100) {
    const times = [];

    // Warmup
    for (let i = 0; i < 10; i++) {
      await fn();
    }

    // Actual benchmark
    for (let i = 0; i < iterations; i++) {
      const start = process.hrtime.bigint();
      await fn();
      const end = process.hrtime.bigint();
      times.push(Number(end - start) / 1e6); // Convert to milliseconds
    }

    // Calculate statistics
    times.sort((a, b) => a - b);
    const mean = times.reduce((sum, t) => sum + t, 0) / times.length;
    const median = times[Math.floor(times.length / 2)];
    const p95 = times[Math.floor(times.length * 0.95)];
    const p99 = times[Math.floor(times.length * 0.99)];
    const min = times[0];
    const max = times[times.length - 1];

    const result = {
      name: this.name,
      iterations,
      mean,
      median,
      p95,
      p99,
      min,
      max,
      ops_per_second: 1000 / mean,
    };

    this.results.push(result);
    return result;
  }

  report() {
    console.log(`\n${ '='.repeat(80)}`);
    console.log(`Benchmark Results: ${this.name}`);
    console.log('='.repeat(80));

    for (const result of this.results) {
      console.log(`\n${result.name}:`);
      console.log(`  Iterations: ${result.iterations}`);
      console.log(`  Mean: ${result.mean.toFixed(3)}ms`);
      console.log(`  Median: ${result.median.toFixed(3)}ms`);
      console.log(`  P95: ${result.p95.toFixed(3)}ms`);
      console.log(`  P99: ${result.p99.toFixed(3)}ms`);
      console.log(`  Min: ${result.min.toFixed(3)}ms`);
      console.log(`  Max: ${result.max.toFixed(3)}ms`);
      console.log(`  Ops/sec: ${result.ops_per_second.toFixed(2)}`);
    }

    console.log(`\n${ '='.repeat(80)}`);
  }
}

describe('Performance Benchmarks', () => {
  let ruvSwarm;
  const testDbPath = path.join(__dirname, 'benchmark.db');

  beforeEach(async() => {
    // Clean up
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }

    // Reset global state
    global._ruvSwarmInstance = null;
    global._ruvSwarmInitialized = 0;
  });

  afterEach(() => {
    if (ruvSwarm && ruvSwarm.persistence) {
      ruvSwarm.persistence.close();
    }
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  describe('Core Operations Benchmarks', () => {
    it('should benchmark RuvSwarm initialization', async() => {
      const benchmark = new BenchmarkRunner('RuvSwarm Initialization');

      const result = await benchmark.run(async() => {
        global._ruvSwarmInstance = null;
        await RuvSwarm.initialize({ enablePersistence: false });
      }, 50);

      benchmark.report();
      assert(result.mean < 100); // Should initialize in less than 100ms
    });

    it('should benchmark swarm creation', async() => {
      ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });
      const benchmark = new BenchmarkRunner('Swarm Creation');

      const result = await benchmark.run(async() => {
        await ruvSwarm.createSwarm({
          name: 'benchmark-swarm',
          topology: 'mesh',
          maxAgents: 10,
        });
      });

      benchmark.report();
      assert(result.mean < 10); // Should create swarm in less than 10ms
    });

    it('should benchmark agent spawning', async() => {
      ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });
      const swarm = await ruvSwarm.createSwarm({ name: 'agent-benchmark' });
      const benchmark = new BenchmarkRunner('Agent Spawning');

      const result = await benchmark.run(async() => {
        await swarm.spawn({
          type: 'researcher',
          enableNeuralNetwork: false,
        });
      });

      benchmark.report();
      assert(result.mean < 5); // Should spawn agent in less than 5ms
    });

    it('should benchmark task orchestration', async() => {
      ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });
      const swarm = await ruvSwarm.createSwarm({ name: 'task-benchmark' });

      // Spawn agents for task assignment
      for (let i = 0; i < 5; i++) {
        await swarm.spawn({ type: 'researcher' });
      }

      const benchmark = new BenchmarkRunner('Task Orchestration');

      const result = await benchmark.run(async() => {
        await swarm.orchestrate({
          description: 'Benchmark task',
          priority: 'medium',
        });
      });

      benchmark.report();
      assert(result.mean < 10); // Should orchestrate in less than 10ms
    });
  });

  describe('Neural Network Benchmarks', () => {
    it('should benchmark neural network forward propagation', async() => {
      const network = new NeuralNetwork({
        networkLayers: [100, 200, 100, 50],
        activationFunction: 'relu',
        learningRate: 0.5,
        momentum: 0.2,
      });

      const input = new Array(100).fill(0).map(() => Math.random());
      const benchmark = new BenchmarkRunner('Neural Network Forward Pass');

      const result = await benchmark.run(async() => {
        network.forward(input);
      }, 1000);

      benchmark.report();
      assert(result.ops_per_second > 1000); // Should handle > 1000 ops/sec
    });

    it('should benchmark neural network training', async() => {
      const network = new NeuralNetwork({
        networkLayers: [50, 100, 50, 25],
        activationFunction: 'sigmoid',
        learningRate: 0.5,
        momentum: 0.2,
      });

      const input = new Array(50).fill(0).map(() => Math.random());
      const target = new Array(25).fill(0).map(() => Math.random());
      const benchmark = new BenchmarkRunner('Neural Network Training');

      const result = await benchmark.run(async() => {
        network.train(input, target);
      }, 500);

      benchmark.report();
      assert(result.ops_per_second > 100); // Should handle > 100 training ops/sec
    });
  });

  describe('Persistence Benchmarks', () => {
    let persistence;

    beforeEach(() => {
      persistence = new SwarmPersistence(testDbPath);
    });

    afterEach(() => {
      if (persistence) {
        persistence.close();
      }
    });

    it('should benchmark swarm persistence', async() => {
      const benchmark = new BenchmarkRunner('Swarm Persistence');
      let swarmId = 0;

      const result = await benchmark.run(async() => {
        persistence.createSwarm({
          id: `swarm-${swarmId++}`,
          name: 'Benchmark Swarm',
          topology: 'mesh',
          maxAgents: 10,
        });
      });

      benchmark.report();
      assert(result.mean < 5); // Should persist in less than 5ms
    });

    it('should benchmark agent persistence', async() => {
      // Create parent swarm
      persistence.createSwarm({
        id: 'parent-swarm',
        name: 'Parent',
        topology: 'mesh',
        maxAgents: 100,
      });

      const benchmark = new BenchmarkRunner('Agent Persistence');
      let agentId = 0;

      const result = await benchmark.run(async() => {
        persistence.createAgent({
          id: `agent-${agentId++}`,
          swarmId: 'parent-swarm',
          name: 'Benchmark Agent',
          type: 'researcher',
        });
      });

      benchmark.report();
      assert(result.mean < 5); // Should persist in less than 5ms
    });

    it('should benchmark memory operations', async() => {
      // Create parent entities
      persistence.createSwarm({
        id: 'memory-swarm',
        name: 'Memory Test',
        topology: 'mesh',
        maxAgents: 10,
      });

      persistence.createAgent({
        id: 'memory-agent',
        swarmId: 'memory-swarm',
        name: 'Memory Agent',
        type: 'researcher',
      });

      const benchmark = new BenchmarkRunner('Memory Store/Retrieve');
      let keyId = 0;

      const result = await benchmark.run(async() => {
        const key = `key-${keyId++}`;
        const data = { value: Math.random(), timestamp: Date.now() };

        // Store
        persistence.storeAgentMemory('memory-agent', key, data);

        // Retrieve
        persistence.getAgentMemory('memory-agent', key);
      });

      benchmark.report();
      assert(result.mean < 10); // Should complete in less than 10ms
    });

    it('should benchmark query performance', async() => {
      // Populate with test data
      const swarmId = 'query-swarm';
      persistence.createSwarm({
        id: swarmId,
        name: 'Query Test',
        topology: 'mesh',
        maxAgents: 100,
      });

      // Create 100 agents
      for (let i = 0; i < 100; i++) {
        persistence.createAgent({
          id: `agent-${i}`,
          swarmId,
          name: `Agent ${i}`,
          type: i % 2 === 0 ? 'researcher' : 'coder',
        });
      }

      const benchmark = new BenchmarkRunner('Agent Query');

      const result = await benchmark.run(async() => {
        persistence.getSwarmAgents(swarmId);
      });

      benchmark.report();
      assert(result.mean < 20); // Should query 100 agents in less than 20ms
    });
  });

  describe('Concurrent Operations Benchmarks', () => {
    it('should benchmark concurrent swarm operations', async() => {
      ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });
      const benchmark = new BenchmarkRunner('Concurrent Swarm Operations');

      const result = await benchmark.run(async() => {
        // Create multiple swarms concurrently
        const promises = [];
        for (let i = 0; i < 5; i++) {
          promises.push(
            ruvSwarm.createSwarm({
              name: `concurrent-swarm-${i}`,
              topology: 'mesh',
            }),
          );
        }
        await Promise.all(promises);
      }, 50);

      benchmark.report();
      assert(result.mean < 50); // Should handle 5 concurrent creates in < 50ms
    });

    it('should benchmark concurrent agent operations', async() => {
      ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });
      const swarm = await ruvSwarm.createSwarm({ name: 'concurrent-test' });
      const benchmark = new BenchmarkRunner('Concurrent Agent Spawning');

      const result = await benchmark.run(async() => {
        // Spawn multiple agents concurrently
        const promises = [];
        for (let i = 0; i < 10; i++) {
          promises.push(
            swarm.spawn({
              type: i % 2 === 0 ? 'researcher' : 'coder',
              name: `agent-${i}`,
            }),
          );
        }
        await Promise.all(promises);
      }, 20);

      benchmark.report();
      assert(result.mean < 100); // Should spawn 10 agents in < 100ms
    });
  });

  describe('Memory Usage Benchmarks', () => {
    it('should measure memory usage for large swarms', async() => {
      ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });

      const initialMemory = process.memoryUsage();

      // Create large swarm
      const swarm = await ruvSwarm.createSwarm({
        name: 'memory-test-swarm',
        maxAgents: 1000,
      });

      // Spawn many agents
      const agents = [];
      for (let i = 0; i < 100; i++) {
        agents.push(await swarm.spawn({ type: 'researcher' }));
      }

      // Create many tasks
      for (let i = 0; i < 50; i++) {
        await swarm.orchestrate({
          description: `Memory test task ${i}`,
        });
      }

      const finalMemory = process.memoryUsage();

      const memoryIncrease = {
        heapUsed: (finalMemory.heapUsed - initialMemory.heapUsed) / 1024 / 1024,
        external: (finalMemory.external - initialMemory.external) / 1024 / 1024,
        rss: (finalMemory.rss - initialMemory.rss) / 1024 / 1024,
      };

      console.log('\nMemory Usage:');
      console.log(`  Heap increase: ${memoryIncrease.heapUsed.toFixed(2)} MB`);
      console.log(`  External increase: ${memoryIncrease.external.toFixed(2)} MB`);
      console.log(`  RSS increase: ${memoryIncrease.rss.toFixed(2)} MB`);

      // Should not use excessive memory
      assert(memoryIncrease.heapUsed < 100); // Less than 100MB for 100 agents
    });
  });

  describe('Scalability Benchmarks', () => {
    it('should benchmark scalability with increasing agents', async() => {
      ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });
      const swarm = await ruvSwarm.createSwarm({ name: 'scalability-test' });

      const agentCounts = [10, 50, 100, 200];
      const results = [];

      for (const count of agentCounts) {
        // Spawn agents
        const start = process.hrtime.bigint();
        for (let i = 0; i < count; i++) {
          await swarm.spawn({ type: 'researcher' });
        }
        const end = process.hrtime.bigint();

        const timeMs = Number(end - start) / 1e6;
        const timePerAgent = timeMs / count;

        results.push({
          agents: count,
          totalTime: timeMs,
          timePerAgent,
        });
      }

      console.log('\nScalability Results:');
      for (const result of results) {
        console.log(`  ${result.agents} agents: ${result.totalTime.toFixed(2)}ms total, ${result.timePerAgent.toFixed(3)}ms per agent`);
      }

      // Time per agent should not increase significantly
      const firstTimePerAgent = results[0].timePerAgent;
      const lastTimePerAgent = results[results.length - 1].timePerAgent;
      assert(lastTimePerAgent < firstTimePerAgent * 2); // Should not double
    });
  });

  describe('Real-world Scenario Benchmarks', () => {
    it('should benchmark a realistic workflow', async() => {
      ruvSwarm = await RuvSwarm.initialize({
        enablePersistence: true,
        enableNeuralNetworks: true,
      });

      if (ruvSwarm.persistence) {
        ruvSwarm.persistence.close();
        ruvSwarm.persistence = new SwarmPersistence(testDbPath);
      }

      const benchmark = new BenchmarkRunner('Realistic Workflow');

      const result = await benchmark.run(async() => {
        // Create swarm
        const swarm = await ruvSwarm.createSwarm({
          name: 'project-swarm',
          topology: 'hierarchical',
          strategy: 'specialized',
        });

        // Spawn diverse agents
        const agents = await Promise.all([
          swarm.spawn({ type: 'researcher', capabilities: ['research', 'documentation'] }),
          swarm.spawn({ type: 'coder', capabilities: ['javascript', 'python'] }),
          swarm.spawn({ type: 'analyst', capabilities: ['analysis', 'testing'] }),
          swarm.spawn({ type: 'optimizer', capabilities: ['performance', 'optimization'] }),
        ]);

        // Orchestrate multiple tasks
        const tasks = await Promise.all([
          swarm.orchestrate({
            description: 'Research best practices',
            priority: 'high',
            requiredCapabilities: ['research'],
          }),
          swarm.orchestrate({
            description: 'Implement core features',
            priority: 'high',
            requiredCapabilities: ['javascript'],
          }),
          swarm.orchestrate({
            description: 'Analyze performance',
            priority: 'medium',
            requiredCapabilities: ['analysis'],
          }),
        ]);

        // Wait for some execution
        await new Promise(resolve => setTimeout(resolve, 10));

        // Get status
        await swarm.getStatus();

        // Store some memory
        if (ruvSwarm.persistence) {
          for (const agent of agents) {
            ruvSwarm.persistence.storeAgentMemory(
              agent.id,
              'workflow_state',
              { completed: true },
            );
          }
        }
      }, 20);

      benchmark.report();
      console.log('\nRealistic workflow completed successfully');
      assert(result.mean < 200); // Should complete in less than 200ms
    });
  });
});

// Run benchmarks
// Direct execution block
{
  console.log('Running Performance Benchmarks...');
  console.log('This may take a few minutes...\n');
  require('../../node_modules/.bin/jest');
}