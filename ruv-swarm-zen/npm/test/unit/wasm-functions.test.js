/**
 * Unit tests for WASM functions
 * Tests all exported WASM functions with comprehensive coverage
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { WasmModuleLoader } from '../../src/wasm-loader.js';
import { performance } from 'perf_hooks';

describe('WASM Functions Unit Tests', () => {
  let wasmModule;
  let loader;

  beforeAll(async() => {
    loader = new WasmModuleLoader();
    wasmModule = await loader.loadModule('core');
  });

  afterAll(async() => {
    if (loader) {
      await loader.cleanup();
    }
  });

  describe('Core WASM Functions', () => {
    it('should initialize WASM module correctly', () => {
      expect(wasmModule).toBeDefined();
      expect(wasmModule.exports).toBeDefined();
    });

    it('should detect SIMD support', () => {
      const simdSupported = wasmModule.exports.detectSIMDSupport();
      expect(typeof simdSupported).toBe('boolean');
    });

    it('should get version information', () => {
      const version = wasmModule.exports.getVersion();
      expect(version).toMatch(/^\d+\.\d+\.\d+$/);
    });

    it('should get memory usage statistics', () => {
      const stats = wasmModule.exports.getMemoryUsage();
      expect(stats).toHaveProperty('heapUsed');
      expect(stats).toHaveProperty('heapTotal');
      expect(stats.heapUsed).toBeGreaterThanOrEqual(0);
      expect(stats.heapTotal).toBeGreaterThan(0);
    });
  });

  describe('Agent Functions', () => {
    let agentId;

    it('should create an agent', () => {
      const agentType = 'researcher';
      const capabilities = ['research', 'analysis'];
      agentId = wasmModule.exports.createAgent(agentType, capabilities);
      expect(agentId).toBeDefined();
      expect(typeof agentId).toBe('string');
    });

    it('should get agent status', () => {
      expect(agentId).toBeDefined();
      const status = wasmModule.exports.getAgentStatus(agentId);
      expect(status).toHaveProperty('id', agentId);
      expect(status).toHaveProperty('type', 'researcher');
      expect(status).toHaveProperty('status');
      expect(status).toHaveProperty('capabilities');
    });

    it('should update agent state', () => {
      expect(agentId).toBeDefined();
      const newState = { status: 'busy', currentTask: 'test-task-1' };
      const result = wasmModule.exports.updateAgentState(agentId, newState);
      expect(result).toBe(true);

      const status = wasmModule.exports.getAgentStatus(agentId);
      expect(status.status).toBe('busy');
    });

    it('should list all agents', () => {
      const agents = wasmModule.exports.listAgents();
      expect(Array.isArray(agents)).toBe(true);
      expect(agents.length).toBeGreaterThan(0);
      expect(agents.some(a => a.id === agentId)).toBe(true);
    });

    it('should remove an agent', () => {
      expect(agentId).toBeDefined();
      const result = wasmModule.exports.removeAgent(agentId);
      expect(result).toBe(true);

      const agents = wasmModule.exports.listAgents();
      expect(agents.some(a => a.id === agentId)).toBe(false);
    });
  });

  describe('Swarm Functions', () => {
    let swarmId;

    it('should create a swarm', () => {
      const config = {
        name: 'test-swarm',
        topology: 'mesh',
        maxAgents: 10,
      };
      swarmId = wasmModule.exports.createSwarm(config);
      expect(swarmId).toBeDefined();
      expect(typeof swarmId).toBe('string');
    });

    it('should get swarm status', () => {
      expect(swarmId).toBeDefined();
      const status = wasmModule.exports.getSwarmStatus(swarmId);
      expect(status).toHaveProperty('id', swarmId);
      expect(status).toHaveProperty('name', 'test-swarm');
      expect(status).toHaveProperty('topology', 'mesh');
      expect(status).toHaveProperty('agentCount', 0);
      expect(status).toHaveProperty('maxAgents', 10);
    });

    it('should add agent to swarm', () => {
      expect(swarmId).toBeDefined();
      const agentId = wasmModule.exports.createAgent('coder', ['coding', 'testing']);
      const result = wasmModule.exports.addAgentToSwarm(swarmId, agentId);
      expect(result).toBe(true);

      const status = wasmModule.exports.getSwarmStatus(swarmId);
      expect(status.agentCount).toBe(1);
    });

    it('should orchestrate task in swarm', async() => {
      expect(swarmId).toBeDefined();
      const task = {
        description: 'Test task',
        priority: 'high',
        requiredCapabilities: ['coding'],
      };

      const taskId = await wasmModule.exports.orchestrateTask(swarmId, task);
      expect(taskId).toBeDefined();
      expect(typeof taskId).toBe('string');
    });

    it('should get swarm metrics', () => {
      expect(swarmId).toBeDefined();
      const metrics = wasmModule.exports.getSwarmMetrics(swarmId);
      expect(metrics).toHaveProperty('tasksCompleted');
      expect(metrics).toHaveProperty('tasksInProgress');
      expect(metrics).toHaveProperty('averageCompletionTime');
      expect(metrics).toHaveProperty('agentUtilization');
    });

    it('should remove a swarm', () => {
      expect(swarmId).toBeDefined();
      const result = wasmModule.exports.removeSwarm(swarmId);
      expect(result).toBe(true);

      expect(() => wasmModule.exports.getSwarmStatus(swarmId)).toThrow();
    });
  });

  describe('Neural Network Functions', () => {
    let networkId;

    it('should create a neural network', () => {
      const config = {
        type: 'lstm',
        inputSize: 10,
        hiddenSize: 20,
        outputSize: 5,
        layers: 2,
      };
      networkId = wasmModule.exports.createNeuralNetwork(config);
      expect(networkId).toBeDefined();
      expect(typeof networkId).toBe('string');
    });

    it('should get network architecture', () => {
      expect(networkId).toBeDefined();
      const arch = wasmModule.exports.getNetworkArchitecture(networkId);
      expect(arch).toHaveProperty('type', 'lstm');
      expect(arch).toHaveProperty('layers');
      expect(arch.layers.length).toBe(2);
    });

    it('should perform forward pass', () => {
      expect(networkId).toBeDefined();
      const input = new Float32Array(10).fill(0.5);
      const output = wasmModule.exports.forward(networkId, input);
      expect(output).toBeInstanceOf(Float32Array);
      expect(output.length).toBe(5);
    });

    it('should train network', () => {
      expect(networkId).toBeDefined();
      const trainingData = {
        inputs: [new Float32Array(10).fill(0.5)],
        targets: [new Float32Array(5).fill(0.8)],
      };
      const loss = wasmModule.exports.train(networkId, trainingData, { epochs: 1, learningRate: 0.01 });
      expect(typeof loss).toBe('number');
      expect(loss).toBeGreaterThanOrEqual(0);
    });

    it('should save and load network weights', () => {
      expect(networkId).toBeDefined();
      const weights = wasmModule.exports.getNetworkWeights(networkId);
      expect(weights).toBeInstanceOf(Float32Array);

      const newNetworkId = wasmModule.exports.createNeuralNetwork({
        type: 'lstm',
        inputSize: 10,
        hiddenSize: 20,
        outputSize: 5,
        layers: 2,
      });

      const result = wasmModule.exports.setNetworkWeights(newNetworkId, weights);
      expect(result).toBe(true);
    });

    it('should remove a neural network', () => {
      expect(networkId).toBeDefined();
      const result = wasmModule.exports.removeNeuralNetwork(networkId);
      expect(result).toBe(true);
    });
  });

  describe('Memory Management Functions', () => {
    it('should allocate memory', () => {
      const size = 1024;
      const ptr = wasmModule.exports.allocate(size);
      expect(ptr).toBeGreaterThan(0);

      wasmModule.exports.deallocate(ptr);
    });

    it('should copy memory between JS and WASM', () => {
      const data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const ptr = wasmModule.exports.allocateFloat32Array(data.length);

      wasmModule.exports.copyFloat32ArrayToWasm(data, ptr);
      const result = wasmModule.exports.copyFloat32ArrayFromWasm(ptr, data.length);

      expect(result).toEqual(data);
      wasmModule.exports.deallocateFloat32Array(ptr);
    });

    it('should handle memory pressure', () => {
      const memoryBefore = wasmModule.exports.getMemoryUsage();

      // Allocate a large amount of memory
      const allocations = [];
      for (let i = 0; i < 100; i++) {
        allocations.push(wasmModule.exports.allocate(1024 * 1024)); // 1MB each
      }

      const memoryAfter = wasmModule.exports.getMemoryUsage();
      expect(memoryAfter.heapUsed).toBeGreaterThan(memoryBefore.heapUsed);

      // Cleanup
      allocations.forEach(ptr => wasmModule.exports.deallocate(ptr));

      // Force garbage collection if available
      if (wasmModule.exports.collectGarbage) {
        wasmModule.exports.collectGarbage();
      }
    });
  });

  describe('SIMD Operations', () => {
    it('should perform SIMD vector addition', () => {
      if (!wasmModule.exports.detectSIMDSupport()) {
        console.log('SIMD not supported, skipping test');
        return;
      }

      const a = new Float32Array([1, 2, 3, 4]);
      const b = new Float32Array([5, 6, 7, 8]);
      const result = wasmModule.exports.simdVectorAdd(a, b);

      expect(result).toEqual(new Float32Array([6, 8, 10, 12]));
    });

    it('should perform SIMD matrix multiplication', () => {
      if (!wasmModule.exports.detectSIMDSupport()) {
        console.log('SIMD not supported, skipping test');
        return;
      }

      const a = new Float32Array([1, 2, 3, 4]); // 2x2 matrix
      const b = new Float32Array([5, 6, 7, 8]); // 2x2 matrix
      const result = wasmModule.exports.simdMatMul(a, 2, 2, b, 2, 2);

      expect(result).toEqual(new Float32Array([19, 22, 43, 50]));
    });

    it('should benchmark SIMD vs non-SIMD operations', () => {
      const size = 1000000;
      const a = new Float32Array(size).fill(1.0);
      const b = new Float32Array(size).fill(2.0);

      // Non-SIMD benchmark
      const nonSimdStart = performance.now();
      const nonSimdResult = wasmModule.exports.vectorAddNonSIMD(a, b);
      const nonSimdTime = performance.now() - nonSimdStart;

      // SIMD benchmark (if supported)
      if (wasmModule.exports.detectSIMDSupport()) {
        const simdStart = performance.now();
        const simdResult = wasmModule.exports.vectorAddSIMD(a, b);
        const simdTime = performance.now() - simdStart;

        console.log(`Non-SIMD time: ${nonSimdTime.toFixed(2)}ms`);
        console.log(`SIMD time: ${simdTime.toFixed(2)}ms`);
        console.log(`Speedup: ${(nonSimdTime / simdTime).toFixed(2)}x`);

        expect(simdTime).toBeLessThan(nonSimdTime);
      }
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid agent ID', () => {
      expect(() => wasmModule.exports.getAgentStatus('invalid-id')).toThrow();
    });

    it('should handle invalid swarm configuration', () => {
      const invalidConfig = {
        name: '',
        topology: 'invalid',
        maxAgents: -1,
      };
      expect(() => wasmModule.exports.createSwarm(invalidConfig)).toThrow();
    });

    it('should handle memory allocation failures', () => {
      const hugeSize = Number.MAX_SAFE_INTEGER;
      expect(() => wasmModule.exports.allocate(hugeSize)).toThrow();
    });

    it('should handle neural network errors gracefully', () => {
      const invalidConfig = {
        type: 'unknown',
        inputSize: -1,
        hiddenSize: 0,
        outputSize: 0,
      };
      expect(() => wasmModule.exports.createNeuralNetwork(invalidConfig)).toThrow();
    });
  });

  describe('Performance Benchmarks', () => {
    it('should benchmark agent creation performance', () => {
      const iterations = 1000;
      const start = performance.now();

      for (let i = 0; i < iterations; i++) {
        const agentId = wasmModule.exports.createAgent('researcher', ['research']);
        wasmModule.exports.removeAgent(agentId);
      }

      const time = performance.now() - start;
      const avgTime = time / iterations;

      console.log(`Agent creation average time: ${avgTime.toFixed(3)}ms`);
      expect(avgTime).toBeLessThan(1); // Should be less than 1ms per agent
    });

    it('should benchmark neural network inference', () => {
      const network = wasmModule.exports.createNeuralNetwork({
        type: 'mlp',
        inputSize: 100,
        hiddenSize: 50,
        outputSize: 10,
        layers: 3,
      });

      const input = new Float32Array(100).fill(0.5);
      const iterations = 1000;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        wasmModule.exports.forward(network, input);
      }
      const time = performance.now() - start;

      const avgTime = time / iterations;
      console.log(`Neural network inference average time: ${avgTime.toFixed(3)}ms`);
      expect(avgTime).toBeLessThan(0.5); // Should be less than 0.5ms per inference

      wasmModule.exports.removeNeuralNetwork(network);
    });

    it('should benchmark memory operations', () => {
      const size = 1024 * 1024; // 1MB
      const data = new Float32Array(size / 4).fill(1.0);
      const iterations = 100;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const ptr = wasmModule.exports.allocateFloat32Array(data.length);
        wasmModule.exports.copyFloat32ArrayToWasm(data, ptr);
        const result = wasmModule.exports.copyFloat32ArrayFromWasm(ptr, data.length);
        wasmModule.exports.deallocateFloat32Array(ptr);
      }
      const time = performance.now() - start;

      const avgTime = time / iterations;
      const throughput = (size * iterations) / (time / 1000) / (1024 * 1024); // MB/s

      console.log(`Memory operation average time: ${avgTime.toFixed(3)}ms`);
      console.log(`Memory throughput: ${throughput.toFixed(2)} MB/s`);
      expect(throughput).toBeGreaterThan(100); // Should be at least 100 MB/s
    });
  });
});
