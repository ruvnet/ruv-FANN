/**
 * Integration tests for JavaScript-WASM communication
 * Tests bidirectional data flow, callbacks, and complex interactions
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { RuvSwarm } from '../../src/index-enhanced.js';
import { WasmModuleLoader } from '../../src/wasm-loader.js';
import { NeuralNetworkManager } from '../../src/neural-network-manager.js';
import { PersistenceManager } from '../../src/persistence.js';

describe('JS-WASM Communication Integration Tests', () => {
  let ruvSwarm;
  let wasmLoader;
  let neuralManager;
  let persistenceManager;

  beforeAll(async() => {
    // Initialize all components
    ruvSwarm = await RuvSwarm.initialize({
      loadingStrategy: 'progressive',
      enablePersistence: true,
      enableNeuralNetworks: true,
      useSIMD: true,
      debug: false,
    });

    wasmLoader = ruvSwarm.wasmLoader;
    neuralManager = ruvSwarm.neuralManager;
    persistenceManager = ruvSwarm.persistenceManager;
  });

  afterAll(async() => {
    if (ruvSwarm) {
      await ruvSwarm.cleanup();
    }
  });

  describe('Data Type Marshalling', () => {
    it('should correctly marshal primitive types', async() => {
      const swarm = await ruvSwarm.createSwarm({
        name: 'test-swarm',
        maxAgents: 5,
      });

      // Test number marshalling
      const agent = await swarm.spawn({
        type: 'researcher',
        priority: 0.75,
      });
      expect(agent.priority).toBe(0.75);

      // Test string marshalling
      agent.updateMetadata({ description: 'Test agent with special chars: 日本語 🚀' });
      const metadata = await agent.getMetadata();
      expect(metadata.description).toBe('Test agent with special chars: 日本語 🚀');

      // Test boolean marshalling
      agent.setActive(true);
      expect(agent.isActive).toBe(true);
    });

    it('should correctly marshal arrays', async() => {
      const data = {
        floatArray: new Float32Array([1.1, 2.2, 3.3, 4.4]),
        intArray: new Int32Array([10, 20, 30, 40]),
        uint8Array: new Uint8Array([255, 128, 64, 0]),
      };

      // Test array passing to WASM
      const result = await wasmLoader.processArrays(data);

      expect(result.floatSum).toBeCloseTo(11.0);
      expect(result.intSum).toBe(100);
      expect(result.uint8Sum).toBe(447);
    });

    it('should correctly marshal complex objects', async() => {
      const config = {
        network: {
          type: 'lstm',
          layers: [
            { units: 128, activation: 'tanh' },
            { units: 64, activation: 'relu' },
            { units: 32, activation: 'sigmoid' },
          ],
          optimizer: {
            type: 'adam',
            learningRate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
          },
        },
        training: {
          epochs: 100,
          batchSize: 32,
          validationSplit: 0.2,
        },
      };

      const network = await neuralManager.createNetwork(config.network);
      const networkConfig = await network.getConfiguration();

      expect(networkConfig.layers).toHaveLength(3);
      expect(networkConfig.optimizer.learningRate).toBe(0.001);
    });
  });

  describe('Callback Mechanisms', () => {
    it('should handle synchronous callbacks from WASM', async() => {
      let callbackExecuted = false;
      let callbackData = null;

      const swarm = await ruvSwarm.createSwarm({
        name: 'callback-test',
        onAgentUpdate: (agentId, status) => {
          callbackExecuted = true;
          callbackData = { agentId, status };
        },
      });

      const agent = await swarm.spawn({ type: 'coder' });
      await agent.execute({ task: 'test-task' });

      expect(callbackExecuted).toBe(true);
      expect(callbackData).toBeDefined();
      expect(callbackData.agentId).toBe(agent.id);
    });

    it('should handle asynchronous callbacks from WASM', async() => {
      const events = [];

      const swarm = await ruvSwarm.createSwarm({
        name: 'async-callback-test',
        onTaskProgress: async(taskId, progress) => {
          events.push({ taskId, progress, timestamp: Date.now() });
          await new Promise(resolve => setTimeout(resolve, 10)); // Simulate async work
        },
      });

      const taskId = await swarm.orchestrate({
        task: 'Complex multi-step task',
        steps: 5,
      });

      // Wait for task completion
      await swarm.waitForTask(taskId);

      expect(events.length).toBeGreaterThan(0);
      expect(events[events.length - 1].progress).toBe(100);
    });

    it('should handle error callbacks from WASM', async() => {
      let errorCaught = false;
      let errorMessage = '';

      const swarm = await ruvSwarm.createSwarm({
        name: 'error-callback-test',
        onError: (error) => {
          errorCaught = true;
          errorMessage = error.message;
        },
      });

      // Trigger an error by exceeding max agents
      try {
        for (let i = 0; i < 20; i++) {
          await swarm.spawn({ type: 'researcher' });
        }
      } catch (e) {
        // Expected
      }

      expect(errorCaught).toBe(true);
      expect(errorMessage).toContain('max agents');
    });
  });

  describe('Memory Sharing', () => {
    it('should share memory efficiently between JS and WASM', async() => {
      const size = 1024 * 1024; // 1MB
      const sharedBuffer = new SharedArrayBuffer(size);
      const jsView = new Float32Array(sharedBuffer);

      // Fill from JS side
      for (let i = 0; i < jsView.length; i++) {
        jsView[i] = Math.random();
      }

      // Process in WASM
      const result = await wasmLoader.processSharedMemory(sharedBuffer);

      expect(result.sum).toBeGreaterThan(0);
      expect(result.mean).toBeCloseTo(0.5, 1);
      expect(result.processed).toBe(jsView.length);
    });

    it('should handle concurrent memory access safely', async() => {
      const buffer = new SharedArrayBuffer(1024);
      const view = new Int32Array(buffer);
      Atomics.store(view, 0, 0);

      const promises = [];
      const numWorkers = 4;
      const incrementsPerWorker = 1000;

      for (let i = 0; i < numWorkers; i++) {
        promises.push(wasmLoader.atomicIncrement(buffer, 0, incrementsPerWorker));
      }

      await Promise.all(promises);

      const finalValue = Atomics.load(view, 0);
      expect(finalValue).toBe(numWorkers * incrementsPerWorker);
    });

    it('should manage memory lifecycle correctly', async() => {
      const initialMemory = await wasmLoader.getMemoryStats();

      // Allocate and process large data
      const allocations = [];
      for (let i = 0; i < 10; i++) {
        const data = new Float32Array(100000).fill(i);
        const ptr = await wasmLoader.allocateAndProcess(data);
        allocations.push(ptr);
      }

      const afterAllocMemory = await wasmLoader.getMemoryStats();
      expect(afterAllocMemory.used).toBeGreaterThan(initialMemory.used);

      // Clean up
      for (const ptr of allocations) {
        await wasmLoader.deallocate(ptr);
      }

      const afterCleanupMemory = await wasmLoader.getMemoryStats();
      expect(afterCleanupMemory.used).toBeLessThanOrEqual(afterAllocMemory.used);
    });
  });

  describe('Stream Processing', () => {
    it('should handle streaming data from JS to WASM', async() => {
      const stream = wasmLoader.createDataStream();
      const chunks = [];

      for (let i = 0; i < 100; i++) {
        const chunk = new Float32Array(1000).fill(i);
        chunks.push(chunk);
        await stream.write(chunk);
      }

      const result = await stream.finalize();
      expect(result.chunksProcessed).toBe(100);
      expect(result.totalElements).toBe(100000);
    });

    it('should handle streaming results from WASM to JS', async() => {
      const results = [];
      const resultStream = wasmLoader.createResultStream({
        onData: (data) => results.push(data),
        bufferSize: 1024,
      });

      // Start computation that produces streaming results
      await wasmLoader.computeStreamingResults(resultStream.id, {
        iterations: 50,
        dataPerIteration: 1000,
      });

      await resultStream.waitForCompletion();

      expect(results.length).toBe(50);
      expect(results[0].length).toBe(1000);
    });

    it('should handle backpressure in streaming', async() => {
      let processingDelay = 50; // ms
      let bufferedCount = 0;

      const stream = wasmLoader.createDataStream({
        highWaterMark: 10,
        onBackpressure: () => {
          bufferedCount = stream.getBufferedCount();
          processingDelay = 10; // Speed up processing
        },
      });

      // Write data faster than it can be processed
      const writePromises = [];
      for (let i = 0; i < 20; i++) {
        writePromises.push(stream.write(new Float32Array(1000)));
        await new Promise(resolve => setTimeout(resolve, 10));
      }

      await Promise.all(writePromises);
      expect(bufferedCount).toBeGreaterThan(0);
      expect(bufferedCount).toBeLessThanOrEqual(10);
    });
  });

  describe('Complex Workflow Integration', () => {
    it('should handle neural network training workflow', async() => {
      // Create network
      const network = await neuralManager.createNetwork({
        type: 'mlp',
        layers: [
          { units: 10, activation: 'relu' },
          { units: 5, activation: 'softmax' },
        ],
      });

      // Generate training data
      const trainingData = {
        inputs: [],
        targets: [],
      };

      for (let i = 0; i < 100; i++) {
        trainingData.inputs.push(new Float32Array(10).map(() => Math.random()));
        const target = new Float32Array(5).fill(0);
        target[Math.floor(Math.random() * 5)] = 1;
        trainingData.targets.push(target);
      }

      // Train with progress callbacks
      const progressHistory = [];
      const trainResult = await network.train(trainingData, {
        epochs: 10,
        batchSize: 10,
        onProgress: (epoch, loss) => {
          progressHistory.push({ epoch, loss });
        },
      });

      expect(progressHistory).toHaveLength(10);
      expect(progressHistory[9].loss).toBeLessThan(progressHistory[0].loss);
      expect(trainResult.finalLoss).toBeDefined();
    });

    it('should handle swarm orchestration workflow', async() => {
      const swarm = await ruvSwarm.createSwarm({
        name: 'orchestration-test',
        topology: 'hierarchical',
        maxAgents: 10,
      });

      // Spawn different types of agents
      const agents = await Promise.all([
        swarm.spawn({ type: 'coordinator', role: 'lead' }),
        swarm.spawn({ type: 'researcher', specialization: 'data' }),
        swarm.spawn({ type: 'coder', language: 'javascript' }),
        swarm.spawn({ type: 'tester', framework: 'jest' }),
      ]);

      // Create complex task
      const task = {
        id: 'complex-workflow',
        steps: [
          { type: 'research', description: 'Analyze requirements' },
          { type: 'design', description: 'Create architecture' },
          { type: 'implement', description: 'Write code' },
          { type: 'test', description: 'Validate implementation' },
        ],
        dependencies: {
          design: ['research'],
          implement: ['design'],
          test: ['implement'],
        },
      };

      const orchestrationResult = await swarm.orchestrate(task);

      expect(orchestrationResult.completed).toBe(true);
      expect(orchestrationResult.steps).toHaveLength(4);
      expect(orchestrationResult.agentsUsed).toHaveLength(4);
    });

    it('should handle persistence workflow', async() => {
      // Create and train a network
      const network = await neuralManager.createNetwork({
        type: 'lstm',
        inputSize: 20,
        hiddenSize: 50,
        outputSize: 10,
      });

      await network.train({
        inputs: Array(50).fill(null).map(() => new Float32Array(20).map(() => Math.random())),
        targets: Array(50).fill(null).map(() => new Float32Array(10).map(() => Math.random())),
      });

      // Save to persistence
      const saveResult = await persistenceManager.saveNetwork(network.id, {
        metadata: {
          name: 'test-lstm',
          version: '1.0.0',
          trainedAt: new Date().toISOString(),
        },
      });

      expect(saveResult.success).toBe(true);
      expect(saveResult.size).toBeGreaterThan(0);

      // Load from persistence
      const loadedNetwork = await persistenceManager.loadNetwork(saveResult.id);
      expect(loadedNetwork.id).toBeDefined();

      // Verify loaded network works
      const testInput = new Float32Array(20).map(() => Math.random());
      const output = await loadedNetwork.predict(testInput);
      expect(output).toHaveLength(10);
    });
  });

  describe('Error Propagation', () => {
    it('should propagate WASM errors to JS with context', async() => {
      try {
        await wasmLoader.executeInvalidOperation({
          operation: 'divide_by_zero',
          context: { value: 42 },
        });
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.message).toContain('division by zero');
        expect(error.wasmStack).toBeDefined();
        expect(error.context.value).toBe(42);
      }
    });

    it('should handle memory allocation errors gracefully', async() => {
      try {
        // Try to allocate impossibly large amount
        await wasmLoader.allocate(Number.MAX_SAFE_INTEGER);
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.message).toContain('memory allocation failed');
        expect(error.code).toBe('ENOMEM');
      }
    });

    it('should recover from WASM panics', async() => {
      const beforePanic = await wasmLoader.getState();

      try {
        await wasmLoader.triggerPanic('test panic');
      } catch (error) {
        expect(error.message).toContain('panic');
      }

      // Verify module can recover
      const afterPanic = await wasmLoader.getState();
      expect(afterPanic.healthy).toBe(true);
      expect(afterPanic.lastError).toContain('panic');
    });
  });

  describe('Performance Monitoring', () => {
    it('should track JS-WASM call overhead', async() => {
      const metrics = await wasmLoader.enableMetrics();

      // Make various calls
      for (let i = 0; i < 100; i++) {
        await wasmLoader.simpleOperation(i);
      }

      const stats = await metrics.getStatistics();
      expect(stats.totalCalls).toBe(100);
      expect(stats.averageOverhead).toBeLessThan(1); // Less than 1ms overhead
      expect(stats.maxOverhead).toBeLessThan(5); // Max 5ms overhead
    });

    it('should measure data transfer performance', async() => {
      const sizes = [1024, 10240, 102400, 1024000]; // 1KB to 1MB
      const results = [];

      for (const size of sizes) {
        const data = new Float32Array(size / 4).fill(1.0);
        const start = performance.now();

        const result = await wasmLoader.processData(data);

        const time = performance.now() - start;
        const throughput = (size / 1024) / (time / 1000); // KB/s

        results.push({ size, time, throughput });
      }

      // Verify throughput scales reasonably
      expect(results[3].throughput).toBeGreaterThan(1000); // At least 1MB/s for large transfers
    });
  });
});
