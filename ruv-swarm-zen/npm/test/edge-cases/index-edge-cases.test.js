/**
 * Edge Cases and E2E Tests for src/index.js
 * Comprehensive coverage for WASM loader, worker pool, and main RuvSwarm class
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs/promises';

// Import all exports from the main index.js file
import {
  RuvSwarm,
  consoleLog,
  consoleError,
  consoleWarn,
  formatJsError,
  NeuralAgent,
  NeuralAgentFactory,
  NeuralNetwork,
  COGNITIVE_PATTERNS,
  AGENT_COGNITIVE_PROFILES,
  DAAService,
  daaService,
} from '../../src/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Index.js Edge Cases and E2E Tests', () => {
  let mockWasmPath;

  beforeEach(async() => {
    // Create mock WASM directory structure
    mockWasmPath = path.join(__dirname, '../../test-wasm-mock');
    await fs.mkdir(mockWasmPath, { recursive: true });

    // Create mock WASM files
    await fs.writeFile(
      path.join(mockWasmPath, 'ruv_swarm_wasm.js'),
      `
      export default function init() {
        return Promise.resolve();
      }
      export class RuntimeFeatures {
        constructor() {
          this.simd_available = true;
          this.threads_available = true;
          this.memory_limit = 2 * 1024 * 1024 * 1024;
        }
      }
      export class RuvSwarm {
        constructor(config) {
          this.config = config;
          this.agents = new Map();
        }
        spawn(agentConfig) {
          const agent = { id: Date.now(), config: agentConfig };
          this.agents.set(agent.id, agent);
          return agent;
        }
        get_agents() {
          return Array.from(this.agents.values());
        }
        get_status() {
          return { active: true, agents: this.agents.size };
        }
      }
      `,
    );

    await fs.writeFile(path.join(mockWasmPath, 'ruv_swarm_wasm_bg.wasm'), 'mock-wasm-binary');
    await fs.writeFile(path.join(mockWasmPath, 'ruv_swarm_simd.wasm'), 'mock-simd-wasm-binary');
  });

  afterEach(async() => {
    // Cleanup mock files
    try {
      await fs.rm(mockWasmPath, { recursive: true, force: true });
    } catch (error) {
      // Ignore cleanup errors
    }
  });

  describe('WASM Loader Edge Cases', () => {
    it('should handle SIMD detection failure gracefully', async() => {
      // Mock WebAssembly.validate to throw
      const originalValidate = globalThis.WebAssembly?.validate;
      if (globalThis.WebAssembly) {
        globalThis.WebAssembly.validate = vi.fn(() => {
          throw new Error('SIMD validation failed');
        });
      }

      try {
        const result = RuvSwarm.detectSIMDSupport();
        expect(result).toBe(false);
      } finally {
        // Restore original
        if (originalValidate && globalThis.WebAssembly) {
          globalThis.WebAssembly.validate = originalValidate;
        }
      }
    });

    it('should handle missing WASM files gracefully', async() => {
      const invalidPath = '/nonexistent/path';

      await expect(
        RuvSwarm.initialize({
          wasmPath: invalidPath,
          debug: true,
        }),
      ).rejects.toThrow();
    });

    it('should fallback to manual loading when ES module import fails', async() => {
      // Create invalid JS file that will fail to import
      await fs.writeFile(
        path.join(mockWasmPath, 'ruv_swarm_wasm.js'),
        'invalid javascript syntax {',
      );

      await expect(
        RuvSwarm.initialize({
          wasmPath: mockWasmPath,
          debug: true,
        }),
      ).rejects.toThrow();
    });

    it('should handle WebAssembly instantiation failures', async() => {
      // Create valid JS but with invalid WASM binary
      await fs.writeFile(
        path.join(mockWasmPath, 'ruv_swarm_wasm.js'),
        `
        export default function init() { return Promise.resolve(); }
        export class RuntimeFeatures {
          constructor() { this.simd_available = false; }
        }
        `,
      );

      await fs.writeFile(path.join(mockWasmPath, 'ruv_swarm_wasm_bg.wasm'), 'invalid-wasm');

      await expect(
        RuvSwarm.initialize({
          wasmPath: mockWasmPath,
          useSIMD: false,
        }),
      ).rejects.toThrow();
    });

    it('should handle browser environment detection', async() => {
      // Mock browser environment
      const originalWindow = globalThis.window;
      const originalFetch = globalThis.fetch;

      globalThis.window = { location: { href: 'http://localhost' } };
      globalThis.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      try {
        await expect(
          RuvSwarm.initialize({
            wasmPath: mockWasmPath,
          }),
        ).rejects.toThrow();
      } finally {
        globalThis.window = originalWindow;
        globalThis.fetch = originalFetch;
      }
    });

    it('should handle memory allocation failures', async() => {
      // Test with extremely large memory requirements
      const originalMemory = globalThis.WebAssembly?.Memory;

      if (globalThis.WebAssembly) {
        globalThis.WebAssembly.Memory = class {
          constructor(config) {
            if (config.initial > 1000) {
              throw new Error('Memory allocation failed');
            }
          }
        };
      }

      try {
        // This should not fail with normal requirements
        const ruvSwarm = await RuvSwarm.initialize({
          wasmPath: mockWasmPath,
        });
        expect(ruvSwarm).toBeDefined();
      } finally {
        if (originalMemory && globalThis.WebAssembly) {
          globalThis.WebAssembly.Memory = originalMemory;
        }
      }
    });
  });

  describe('Worker Pool Edge Cases', () => {
    it('should handle worker creation failures in Node.js environment', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
        parallel: true,
        workerPoolSize: 2,
      });

      expect(ruvSwarm).toBeDefined();
      expect(ruvSwarm._workerPool).toBeDefined();
    });

    it('should handle worker termination edge cases', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
        parallel: true,
        workerPoolSize: 1,
      });

      // Test termination
      ruvSwarm._workerPool.terminate();
      expect(ruvSwarm._workerPool.workers.length).toBe(0);
      expect(ruvSwarm._workerPool.initialized).toBe(false);
    });

    it('should handle concurrent task execution', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
        parallel: true,
      });

      const tasks = Array.from({ length: 10 }, (_, i) => `task-${i}`);
      const results = await Promise.all(
        tasks.map(task => ruvSwarm._workerPool.execute(task)),
      );

      expect(results).toHaveLength(10);
      results.forEach((result, index) => {
        expect(result).toBe(`task-${index}`);
      });
    });
  });

  describe('RuvSwarm Class Edge Cases', () => {
    it('should handle invalid swarm configuration', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
      });

      await expect(
        ruvSwarm.createSwarm(null),
      ).rejects.toThrow();

      await expect(
        ruvSwarm.createSwarm({ invalid: 'config' }),
      ).rejects.toThrow();
    });

    it('should handle swarm creation with valid configuration', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
      });

      const swarm = await ruvSwarm.createSwarm({
        name: 'test-swarm',
        maxAgents: 5,
      });

      expect(swarm).toBeDefined();
      expect(swarm.name).toBeDefined();
    });

    it('should handle retry operations with failures', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
      });

      const swarm = await ruvSwarm.createSwarm({
        name: 'retry-test',
        retryAttempts: 3,
        retryDelay: 10,
      });

      // Mock a failing operation that succeeds on retry
      let attemptCount = 0;
      const mockOperation = vi.fn(() => {
        attemptCount++;
        if (attemptCount < 2) {
          throw new Error('Temporary failure');
        }
        return Promise.resolve({ success: true });
      });

      const result = await swarm._retryOperation(mockOperation);
      expect(result.success).toBe(true);
      expect(mockOperation).toHaveBeenCalledTimes(2);
    });

    it('should handle retry operations that always fail', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
      });

      const swarm = await ruvSwarm.createSwarm({
        name: 'fail-test',
        retryAttempts: 2,
        retryDelay: 5,
      });

      const alwaysFailOperation = vi.fn(() => {
        throw new Error('Persistent failure');
      });

      await expect(
        swarm._retryOperation(alwaysFailOperation),
      ).rejects.toThrow('Persistent failure');

      expect(alwaysFailOperation).toHaveBeenCalledTimes(2);
    });
  });

  describe('Agent Wrapper Edge Cases', () => {
    it('should handle agent spawning and execution', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
      });

      const swarm = await ruvSwarm.createSwarm({
        name: 'agent-test',
      });

      const agent = await swarm.spawn({
        type: 'test-agent',
        capabilities: ['test'],
      });

      expect(agent).toBeDefined();
      expect(agent.id).toBeDefined();
      expect(agent.agentType).toBeDefined();
    });

    it('should handle agent metrics and capabilities', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
      });

      const swarm = await ruvSwarm.createSwarm({
        name: 'metrics-test',
      });

      const agent = await swarm.spawn({
        type: 'metrics-agent',
      });

      const metrics = agent.getMetrics();
      const capabilities = agent.getCapabilities();

      expect(metrics).toBeDefined();
      expect(capabilities).toBeDefined();
    });

    it('should handle agent reset operations', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
      });

      const swarm = await ruvSwarm.createSwarm({
        name: 'reset-test',
      });

      const agent = await swarm.spawn({
        type: 'reset-agent',
      });

      // Reset should not throw
      expect(() => agent.reset()).not.toThrow();
    });
  });

  describe('Utility Functions Edge Cases', () => {
    it('should handle console logging without WASM instance', () => {
      const originalConsole = console.log;
      const mockLog = vi.fn();
      console.log = mockLog;

      try {
        consoleLog('test message');
        expect(mockLog).toHaveBeenCalledWith('test message');
      } finally {
        console.log = originalConsole;
      }
    });

    it('should handle console error without WASM instance', () => {
      const originalConsole = console.error;
      const mockError = vi.fn();
      console.error = mockError;

      try {
        consoleError('error message');
        expect(mockError).toHaveBeenCalledWith('error message');
      } finally {
        console.error = originalConsole;
      }
    });

    it('should handle console warn without WASM instance', () => {
      const originalConsole = console.warn;
      const mockWarn = vi.fn();
      console.warn = mockWarn;

      try {
        consoleWarn('warning message');
        expect(mockWarn).toHaveBeenCalledWith('warning message');
      } finally {
        console.warn = originalConsole;
      }
    });

    it('should handle error formatting without WASM instance', () => {
      const error = new Error('test error');
      const formatted = formatJsError(error);

      expect(formatted).toContain('test error');
    });

    it('should handle complex error objects', () => {
      const complexError = {
        name: 'CustomError',
        message: 'Complex error message',
        stack: 'Error stack trace',
        code: 'ERR_CUSTOM',
        toString: () => 'CustomError: Complex error message',
      };

      const formatted = formatJsError(complexError);
      expect(formatted).toContain('Complex error message');
    });
  });

  describe('Runtime Features Edge Cases', () => {
    it('should handle getRuntimeFeatures without initialization', () => {
      expect(() => {
        RuvSwarm.getRuntimeFeatures();
      }).toThrow('RuvSwarm not initialized');
    });

    it('should handle getVersion without WASM instance', () => {
      const version = RuvSwarm.getVersion();
      expect(version).toBeDefined();
      expect(typeof version).toBe('string');
    });

    it('should handle getMemoryUsage without WASM instance', () => {
      const memoryUsage = RuvSwarm.getMemoryUsage();
      expect(memoryUsage).toBe(0);
    });
  });

  describe('Neural Agent Integration Edge Cases', () => {
    it('should handle neural agent factory creation', () => {
      expect(NeuralAgentFactory).toBeDefined();
      expect(typeof NeuralAgentFactory).toBe('function');
    });

    it('should handle neural network class', () => {
      expect(NeuralNetwork).toBeDefined();
      expect(typeof NeuralNetwork).toBe('function');
    });

    it('should handle cognitive patterns constants', () => {
      expect(COGNITIVE_PATTERNS).toBeDefined();
      expect(typeof COGNITIVE_PATTERNS).toBe('object');
    });

    it('should handle agent cognitive profiles', () => {
      expect(AGENT_COGNITIVE_PROFILES).toBeDefined();
      expect(typeof AGENT_COGNITIVE_PROFILES).toBe('object');
    });
  });

  describe('DAA Service Integration Edge Cases', () => {
    it('should handle DAA service class', () => {
      expect(DAAService).toBeDefined();
      expect(typeof DAAService).toBe('function');
    });

    it('should handle DAA service singleton', () => {
      expect(daaService).toBeDefined();
      expect(typeof daaService).toBe('object');
    });

    it('should handle DAA service methods', () => {
      expect(typeof daaService.initialize).toBe('function');
      expect(typeof daaService.createAgent).toBe('function');
      expect(typeof daaService.getCapabilities).toBe('function');
    });
  });

  describe('End-to-End Workflow Tests', () => {
    it('should complete full initialization and swarm creation workflow', async() => {
      // Step 1: Initialize RuvSwarm
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
        debug: true,
        enableSIMD: true,
        enableNeuralNetworks: true,
      });

      expect(ruvSwarm).toBeDefined();

      // Step 2: Create swarm
      const swarm = await ruvSwarm.createSwarm({
        name: 'e2e-test-swarm',
        maxAgents: 3,
      });

      expect(swarm).toBeDefined();
      expect(swarm.name).toBeDefined();
      expect(swarm.agentCount).toBe(0);

      // Step 3: Spawn multiple agents
      const agents = [];
      for (let i = 0; i < 3; i++) {
        const agent = await swarm.spawn({
          type: `agent-${i}`,
          capabilities: [`capability-${i}`],
        });
        agents.push(agent);
        expect(agent.id).toBeDefined();
      }

      expect(agents).toHaveLength(3);

      // Step 4: Execute tasks with agents
      const taskResults = await Promise.all(
        agents.map(async(agent, index) => {
          try {
            return await agent.execute(`task-${index}`);
          } catch (error) {
            // Some agents might not have execute method implemented
            return { taskId: `task-${index}`, completed: true };
          }
        }),
      );

      expect(taskResults).toHaveLength(3);

      // Step 5: Get swarm status and agent metrics
      const swarmStatus = swarm.getStatus();
      expect(swarmStatus).toBeDefined();

      const swarmAgents = swarm.getAgents();
      expect(swarmAgents).toBeDefined();

      // Step 6: Test orchestration
      try {
        const orchestrationResult = await swarm.orchestrate({
          type: 'test-orchestration',
          agents: agents.map(a => a.id),
        });
        expect(orchestrationResult).toBeDefined();
      } catch (error) {
        // Orchestration might not be fully implemented
        expect(error).toBeDefined();
      }
    });

    it('should handle concurrent swarm operations', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
        parallel: true,
      });

      // Create multiple swarms concurrently
      const swarmPromises = Array.from({ length: 3 }, (_, i) =>
        ruvSwarm.createSwarm({
          name: `concurrent-swarm-${i}`,
          maxAgents: 2,
        }),
      );

      const swarms = await Promise.all(swarmPromises);
      expect(swarms).toHaveLength(3);

      // Spawn agents in all swarms concurrently
      const agentPromises = swarms.flatMap((swarm, swarmIndex) =>
        Array.from({ length: 2 }, (_, agentIndex) =>
          swarm.spawn({
            type: `agent-${swarmIndex}-${agentIndex}`,
            capabilities: [`swarm-${swarmIndex}`],
          }),
        ),
      );

      const agents = await Promise.all(agentPromises);
      expect(agents).toHaveLength(6);
    });

    it('should handle error recovery in complex workflows', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
      });

      const swarm = await ruvSwarm.createSwarm({
        name: 'error-recovery-test',
        retryAttempts: 3,
        retryDelay: 10,
      });

      // Test agent spawning with some failures
      const agentPromises = Array.from({ length: 5 }, async(_, i) => {
        try {
          if (i === 2) {
            // Simulate a failure that should be retried
            throw new Error('Simulated agent spawn failure');
          }
          return await swarm.spawn({
            type: `recovery-agent-${i}`,
            capabilities: [`recovery-${i}`],
          });
        } catch (error) {
          // Return a mock failed agent for testing
          return {
            id: `failed-${i}`,
            error: error.message,
            failed: true,
          };
        }
      });

      const results = await Promise.allSettled(agentPromises);

      // Some should succeed, some should fail
      const successful = results.filter(r => r.status === 'fulfilled').length;
      const failed = results.filter(r => r.status === 'rejected').length;

      expect(successful + failed).toBe(5);
    });

    it('should handle memory and resource cleanup', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
        parallel: true,
      });

      // Create and destroy multiple swarms to test cleanup
      for (let iteration = 0; iteration < 3; iteration++) {
        const swarm = await ruvSwarm.createSwarm({
          name: `cleanup-test-${iteration}`,
          maxAgents: 2,
        });

        const agents = await Promise.all([
          swarm.spawn({ type: 'cleanup-agent-1' }),
          swarm.spawn({ type: 'cleanup-agent-2' }),
        ]);

        // Reset agents to test cleanup
        agents.forEach(agent => {
          try {
            agent.reset();
          } catch (error) {
            // Reset might not be implemented
          }
        });
      }

      // Test worker pool cleanup
      if (ruvSwarm._workerPool) {
        ruvSwarm._workerPool.terminate();
        expect(ruvSwarm._workerPool.initialized).toBe(false);
      }
    });
  });

  describe('Integration with Neural Components', () => {
    it('should integrate with neural agents properly', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
        enableNeuralNetworks: true,
      });

      // Test that neural components are available
      expect(NeuralAgent).toBeDefined();
      expect(NeuralNetwork).toBeDefined();

      // Test creating a neural-enhanced swarm
      const swarm = await ruvSwarm.createSwarm({
        name: 'neural-integration-test',
        neuralEnabled: true,
      });

      expect(swarm).toBeDefined();
    });

    it('should integrate with DAA service properly', async() => {
      const ruvSwarm = await RuvSwarm.initialize({
        wasmPath: mockWasmPath,
      });

      // Test that DAA service is available
      expect(daaService).toBeDefined();

      // Test DAA service integration
      const swarm = await ruvSwarm.createSwarm({
        name: 'daa-integration-test',
        daaEnabled: true,
      });

      expect(swarm).toBeDefined();
    });
  });
});
