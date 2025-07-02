/**
 * Edge Case Tests for 100% Coverage
 * Tests error handling, edge cases, and uncovered branches
 */

import assert from 'assert';
import { RuvSwarm } from '../src/index.js';
import { NeuralAgent } from '../src/neural-agent.js';
import { NeuralNetworkManager } from '../src/neural-network-manager.js';
import { SwarmPersistence } from '../src/persistence.js';
import { WasmLoader } from '../src/wasm-loader.js';
import { Benchmark } from '../src/benchmark.js';
import { PerformanceAnalyzer } from '../src/performance.js';

describe('Edge Cases for 100% Coverage', () => {
  let ruv;
  let swarm;

  beforeEach(async() => {
    ruv = await RuvSwarm.initialize();
    swarm = await ruv.createSwarm({
      topology: 'mesh',
      maxAgents: 3,
    });
  });

  describe('Neural Network Edge Cases', () => {
    it('should handle null inputs gracefully', async() => {
      const agent = await swarm.spawn({ type: 'researcher' });
      await assert.rejects(
        agent.execute(null),
        /Invalid input/,
      );
    });

    it('should handle invalid neural configurations', async() => {
      const manager = new NeuralNetworkManager();
      await assert.rejects(
        manager.create({
          type: 'invalid-type',
          dimensions: -1,
        }),
        /Invalid configuration/,
      );
    });

    it('should handle memory limit exceeded', async() => {
      const agent = await swarm.spawn({ type: 'coder' });
      const hugeData = new Array(1000000).fill({ data: 'x'.repeat(1000) });

      await assert.rejects(
        agent.process(hugeData),
        /Memory limit/,
      );
    });

    it('should handle concurrent operations race conditions', async() => {
      const agent = await swarm.spawn({ type: 'analyst' });
      const promises = [];

      // Create 100 concurrent operations
      for (let i = 0; i < 100; i++) {
        promises.push(agent.execute({ task: `concurrent-${i}` }));
      }

      const results = await Promise.allSettled(promises);
      const successful = results.filter(r => r.status === 'fulfilled');
      assert(successful.length > 0, 'At least some operations should succeed');
    });

    it('should handle model serialization failures', async() => {
      const manager = new NeuralNetworkManager();
      const model = await manager.create({ type: 'gru' });

      // Corrupt the model state
      model._state = { invalid: Symbol('not-serializable') };

      await assert.rejects(
        manager.serialize(model),
        /Serialization failed/,
      );
    });
  });

  describe('Error Handling Paths', () => {
    it('should handle database connection failures', async() => {
      const persistence = new SwarmPersistence();

      // Force database error
      persistence._db = null;

      await assert.rejects(
        persistence.saveState(swarm),
        /Database connection failed/,
      );
    });

    it('should handle WASM loading failures', async() => {
      const loader = new WasmLoader();

      await assert.rejects(
        loader.loadModule('/invalid/path/to/wasm'),
        /Failed to load WASM/,
      );
    });

    it('should handle network timeouts', async() => {
      const agent = await swarm.spawn({ type: 'researcher' });

      // Set unrealistic timeout
      agent.setTimeout(1);

      await assert.rejects(
        agent.fetchData('https://example.com/large-data'),
        /Timeout/,
      );
    });

    it('should handle invalid configurations', async() => {
      await assert.rejects(
        ruv.createSwarm({
          topology: 'invalid-topology',
          maxAgents: -5,
        }),
        /Invalid configuration/,
      );
    });
  });

  describe('Async Operations', () => {
    it('should handle promise rejections in batch operations', async() => {
      const agents = await Promise.all([
        swarm.spawn({ type: 'coder' }),
        swarm.spawn({ type: 'tester' }),
        swarm.spawn({ type: 'analyst' }),
      ]);

      const tasks = agents.map((agent, i) => ({
        agent,
        task: i === 1 ? null : { id: i }, // Invalid task for second agent
      }));

      const results = await Promise.allSettled(
        tasks.map(({ agent, task }) => agent.execute(task)),
      );

      assert(results[1].status === 'rejected', 'Second task should fail');
    });

    it('should timeout after specified duration', async() => {
      const agent = await swarm.spawn({ type: 'optimizer' });

      const promise = agent.longRunningOperation();
      await assert.rejects(
        Promise.race([
          promise,
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Timeout')), 100),
          ),
        ]),
        /Timeout/,
      );
    });

    it('should handle cleanup on failure', async() => {
      const agent = await swarm.spawn({ type: 'coordinator' });
      let cleanupCalled = false;

      agent.onCleanup = () => {
        cleanupCalled = true;
      };

      try {
        await agent.executeWithCleanup(null);
      } catch (error) {
        // Expected error
      }

      assert(cleanupCalled, 'Cleanup should be called on failure');
    });
  });

  describe('Memory Management', () => {
    it('should handle memory leak scenarios', async() => {
      const agent = await swarm.spawn({ type: 'researcher' });
      const initialMemory = process.memoryUsage().heapUsed;

      // Create many objects without cleanup
      for (let i = 0; i < 1000; i++) {
        agent._cache[`key-${i}`] = new Array(1000).fill(i);
      }

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }

      const finalMemory = process.memoryUsage().heapUsed;
      assert(finalMemory < initialMemory + 50 * 1024 * 1024, 'Memory usage should be controlled');
    });

    it('should handle cache overflow', async() => {
      const agent = await swarm.spawn({ type: 'coder' });
      agent.setCacheLimit(10);

      // Add more items than cache limit
      for (let i = 0; i < 20; i++) {
        agent.cache(`key-${i}`, `value-${i}`);
      }

      // Early items should be evicted
      assert(!agent.getFromCache('key-0'), 'Old items should be evicted');
      assert(agent.getFromCache('key-19'), 'Recent items should remain');
    });
  });

  describe('Benchmark Edge Cases', () => {
    it('should handle benchmark with zero iterations', async() => {
      const benchmark = new Benchmark();

      await assert.rejects(
        benchmark.run({ iterations: 0 }),
        /Invalid iterations/,
      );
    });

    it('should handle performance analyzer with invalid metrics', async() => {
      const analyzer = new PerformanceAnalyzer();

      await assert.rejects(
        analyzer.analyze({ metric: 'invalid-metric' }),
        /Unknown metric/,
      );
    });
  });

  describe('Neural Model Specific Edge Cases', () => {
    it('should handle transformer attention mask errors', async() => {
      const manager = new NeuralNetworkManager();
      const transformer = await manager.create({ type: 'transformer' });

      await assert.rejects(
        transformer.forward({
          input: [[1, 2, 3]],
          attentionMask: null, // Invalid mask
        }),
        /Invalid attention mask/,
      );
    });

    it('should handle CNN invalid kernel sizes', async() => {
      const manager = new NeuralNetworkManager();

      await assert.rejects(
        manager.create({
          type: 'cnn',
          kernelSize: -1,
        }),
        /Invalid kernel size/,
      );
    });

    it('should handle GRU hidden state mismatch', async() => {
      const manager = new NeuralNetworkManager();
      const gru = await manager.create({ type: 'gru', hiddenSize: 128 });

      await assert.rejects(
        gru.forward({
          input: [[1, 2, 3]],
          hiddenState: new Array(64).fill(0), // Wrong size
        }),
        /Hidden state dimension mismatch/,
      );
    });

    it('should handle autoencoder reconstruction with corrupted data', async() => {
      const manager = new NeuralNetworkManager();
      const autoencoder = await manager.create({ type: 'autoencoder' });

      await assert.rejects(
        autoencoder.reconstruct(null),
        /Invalid input for reconstruction/,
      );
    });
  });

  describe('Swarm Coordination Edge Cases', () => {
    it('should handle agent communication failures', async() => {
      const agent1 = await swarm.spawn({ type: 'coordinator' });
      const agent2 = await swarm.spawn({ type: 'researcher' });

      // Simulate network partition
      agent2._communicationEnabled = false;

      await assert.rejects(
        agent1.sendMessage(agent2.id, { data: 'test' }),
        /Communication failed/,
      );
    });

    it('should handle topology reconfiguration during operation', async() => {
      const task = swarm.orchestrate({
        task: 'complex-task',
        agents: 5,
      });

      // Change topology mid-operation
      setTimeout(() => {
        swarm.reconfigure({ topology: 'star' });
      }, 50);

      const result = await task;
      assert(result.completed, 'Task should complete despite reconfiguration');
    });
  });

  afterEach(async() => {
    // Cleanup
    if (swarm) {
      await swarm.terminate();
    }
  });
});

// Run tests when executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  console.log('Running edge case tests for 100% coverage...');

  // Run all tests
  const { run } = await import('./test-runner.js');
  await run(__filename);
}