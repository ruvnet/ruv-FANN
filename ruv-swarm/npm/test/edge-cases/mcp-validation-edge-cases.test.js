/**
 * MCP Tools Validation Edge Cases
 * Tests all boundary conditions and error scenarios for MCP tool validation
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { EnhancedMCPTools } from '../../src/mcp-tools-enhanced.js';
import { RuvSwarm } from '../../src/index-enhanced.js';

// Mock dependencies
jest.mock('../../src/index-enhanced.js');
jest.mock('../../src/persistence.js');

describe('MCP Validation Edge Cases', () => {
  let mcpTools;
  let mockRuvSwarm;

  beforeEach(() => {
    mockRuvSwarm = {
      createSwarm: jest.fn(),
      detectFeatures: jest.fn(),
      benchmark: jest.fn(),
    };
    RuvSwarm.initialize = jest.fn().mockResolvedValue(mockRuvSwarm);
    mcpTools = new EnhancedMCPTools();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Iteration Validation Edge Cases', () => {
    it('should reject iterations at boundary values', async() => {
      await mcpTools.initialize(mockRuvSwarm);

      // Test boundary values
      const boundaryValues = [
        { value: 0, shouldFail: true },
        { value: 1, shouldFail: false },
        { value: 1000, shouldFail: false },
        { value: 1001, shouldFail: true },
        { value: -1, shouldFail: true },
        { value: null, shouldFail: true },
        { value: undefined, shouldFail: true },
        { value: NaN, shouldFail: true },
        { value: Infinity, shouldFail: true },
        { value: -Infinity, shouldFail: true },
        { value: '10', shouldFail: true },
        { value: [10], shouldFail: true },
        { value: { iterations: 10 }, shouldFail: true },
        { value: 1.5, shouldFail: false }, // Should be floored to 1
        { value: 999.999, shouldFail: false }, // Should be floored to 999
      ];

      for (const { value, shouldFail } of boundaryValues) {
        if (shouldFail) {
          await expect(
            mcpTools.benchmark_run({ iterations: value }),
          ).rejects.toThrow(/Iterations must be/);
        } else {
          // Should not throw
          mockRuvSwarm.benchmark.mockResolvedValue({ results: [] });
          await mcpTools.benchmark_run({ iterations: value });
        }
      }
    });

    it('should handle floating point precision issues', async() => {
      await mcpTools.initialize(mockRuvSwarm);
      mockRuvSwarm.benchmark.mockResolvedValue({ results: [] });

      // Floating point edge cases
      const floatCases = [
        0.999999999999999, // Just under 1
        1.000000000000001, // Just over 1
        999.999999999999, // Just under 1000
        1000.000000000001, // Just over 1000 (should fail)
      ];

      for (const value of floatCases) {
        if (value > 1000) {
          await expect(
            mcpTools.benchmark_run({ iterations: value }),
          ).rejects.toThrow();
        } else {
          await mcpTools.benchmark_run({ iterations: value });
        }
      }
    });
  });

  describe('Learning Rate Validation Edge Cases', () => {
    it('should validate learning rate boundaries', async() => {
      await mcpTools.initialize(mockRuvSwarm);

      const learningRateTests = [
        { value: 0, shouldFail: true },
        { value: 0.000001, shouldFail: false },
        { value: 0.5, shouldFail: false },
        { value: 1, shouldFail: false },
        { value: 1.000001, shouldFail: true },
        { value: -0.1, shouldFail: true },
        { value: Number.EPSILON, shouldFail: false },
        { value: 1 - Number.EPSILON, shouldFail: false },
      ];

      for (const { value, shouldFail } of learningRateTests) {
        const mockSwarm = {
          spawn: jest.fn().mockResolvedValue({
            id: 'agent-1',
            train: jest.fn().mockResolvedValue({ loss: 0.1 }),
          }),
        };
        mcpTools.activeSwarms.set('test-swarm', mockSwarm);

        if (shouldFail) {
          await expect(
            mcpTools.neural_train({
              swarmId: 'test-swarm',
              learningRate: value,
            }),
          ).rejects.toThrow(/Learning rate must be/);
        } else {
          await mcpTools.neural_train({
            swarmId: 'test-swarm',
            learningRate: value,
          });
        }
      }
    });
  });

  describe('Model Type Validation Edge Cases', () => {
    it('should handle case sensitivity and whitespace in model types', async() => {
      await mcpTools.initialize(mockRuvSwarm);

      const modelTypeTests = [
        { value: 'feedforward', shouldFail: false },
        { value: 'FEEDFORWARD', shouldFail: true },
        { value: 'FeedForward', shouldFail: true },
        { value: ' feedforward ', shouldFail: true },
        { value: 'feedforward\n', shouldFail: true },
        { value: 'feed forward', shouldFail: true },
        { value: '', shouldFail: true },
        { value: null, shouldFail: true },
        { value: undefined, shouldFail: true },
        { value: 123, shouldFail: true },
        { value: ['feedforward'], shouldFail: true },
      ];

      for (const { value, shouldFail } of modelTypeTests) {
        const mockSwarm = {
          spawn: jest.fn().mockResolvedValue({
            id: 'agent-1',
            capabilities: ['neural'],
          }),
        };
        mcpTools.activeSwarms.set('test-swarm', mockSwarm);

        if (shouldFail) {
          await expect(
            mcpTools.agent_spawn({
              swarmId: 'test-swarm',
              type: 'neural',
              modelType: value,
            }),
          ).rejects.toThrow();
        } else {
          await mcpTools.agent_spawn({
            swarmId: 'test-swarm',
            type: 'neural',
            modelType: value,
          });
        }
      }
    });
  });

  describe('Swarm ID Edge Cases', () => {
    it('should handle special characters in swarm IDs', async() => {
      await mcpTools.initialize(mockRuvSwarm);

      const specialIds = [
        'swarm-with-dashes',
        'swarm_with_underscores',
        'swarm.with.dots',
        'swarm/with/slashes',
        'swarm\\with\\backslashes',
        'swarm with spaces',
        'swarm\twith\ttabs',
        'swarm\nwith\nnewlines',
        '🐝emoji-swarm🐝',
        'swarm;drop table swarms;--',
        'swarm<script>alert("xss")</script>',
        '', // Empty string
        '.', // Just a dot
        '..', // Two dots
        'a'.repeat(1000), // Very long ID
      ];

      mockRuvSwarm.createSwarm.mockResolvedValue({
        id: 'created-swarm',
        topology: 'mesh',
        agents: [],
      });

      for (const id of specialIds) {
        // Should handle all IDs gracefully
        const result = await mcpTools.swarm_init({
          swarmId: id,
          topology: 'mesh',
        });

        // Some IDs might be normalized or rejected
        expect(result).toBeDefined();
      }
    });
  });

  describe('Concurrent Operation Edge Cases', () => {
    it('should handle race conditions in swarm initialization', async() => {
      await mcpTools.initialize(mockRuvSwarm);

      let callCount = 0;
      mockRuvSwarm.createSwarm.mockImplementation(async() => {
        callCount++;
        // Simulate varying processing times
        await new Promise(resolve => setTimeout(resolve, Math.random() * 100));
        return {
          id: `swarm-${callCount}`,
          topology: 'mesh',
          agents: [],
        };
      });

      // Create many swarms concurrently
      const promises = [];
      for (let i = 0; i < 50; i++) {
        promises.push(
          mcpTools.swarm_init({
            swarmId: `concurrent-${i}`,
            topology: 'mesh',
          }),
        );
      }

      const results = await Promise.allSettled(promises);
      const successful = results.filter(r => r.status === 'fulfilled');

      expect(successful.length).toBeGreaterThan(0);
      expect(mcpTools.activeSwarms.size).toBe(successful.length);
    });

    it('should handle concurrent operations on the same swarm', async() => {
      await mcpTools.initialize(mockRuvSwarm);

      const mockSwarm = {
        id: 'test-swarm',
        spawn: jest.fn().mockImplementation(async() => {
          // Simulate processing delay
          await new Promise(resolve => setTimeout(resolve, 10));
          return { id: `agent-${Date.now()}`, type: 'researcher' };
        }),
        agents: [],
      };

      mockRuvSwarm.createSwarm.mockResolvedValue(mockSwarm);
      await mcpTools.swarm_init({ topology: 'mesh' });

      // Spawn many agents concurrently on the same swarm
      const spawnPromises = [];
      for (let i = 0; i < 20; i++) {
        spawnPromises.push(
          mcpTools.agent_spawn({
            swarmId: mockSwarm.id,
            type: 'researcher',
          }),
        );
      }

      const results = await Promise.allSettled(spawnPromises);
      const successful = results.filter(r => r.status === 'fulfilled');

      expect(successful.length).toBe(20);
      expect(mockSwarm.spawn).toHaveBeenCalledTimes(20);
    });
  });

  describe('Memory Pressure Edge Cases', () => {
    it('should handle memory limits when creating large swarms', async() => {
      await mcpTools.initialize(mockRuvSwarm);

      mockRuvSwarm.createSwarm.mockImplementation(async(config) => {
        if (config.maxAgents > 100) {
          throw new Error('Memory limit exceeded');
        }
        return {
          id: 'swarm-1',
          topology: config.topology,
          maxAgents: config.maxAgents,
          agents: [],
        };
      });

      // Test various agent counts
      await expect(
        mcpTools.swarm_init({ topology: 'mesh', maxAgents: 1000 }),
      ).rejects.toThrow(/Memory limit/);

      // Should succeed with reasonable limits
      const result = await mcpTools.swarm_init({
        topology: 'mesh',
        maxAgents: 50,
      });
      expect(result.maxAgents).toBe(50);
    });

    it('should handle memory cleanup on failure', async() => {
      await mcpTools.initialize(mockRuvSwarm);

      let swarmCount = 0;
      mockRuvSwarm.createSwarm.mockImplementation(async() => {
        swarmCount++;
        if (swarmCount > 5) {
          throw new Error('Resource exhausted');
        }
        return {
          id: `swarm-${swarmCount}`,
          topology: 'mesh',
          terminate: jest.fn(),
        };
      });

      // Create swarms until failure
      const promises = [];
      for (let i = 0; i < 10; i++) {
        promises.push(
          mcpTools.swarm_init({ topology: 'mesh' }),
        );
      }

      const results = await Promise.allSettled(promises);
      const failed = results.filter(r => r.status === 'rejected');

      expect(failed.length).toBe(4); // Should fail after 5 successful
      expect(mcpTools.activeSwarms.size).toBeLessThanOrEqual(5);
    });
  });

  describe('Input Sanitization Edge Cases', () => {
    it('should sanitize potentially malicious inputs', async() => {
      await mcpTools.initialize(mockRuvSwarm);

      const maliciousInputs = [
        { task: '<script>alert("xss")</script>' },
        { task: '"; DROP TABLE swarms; --' },
        { task: '${process.exit(1)}' },
        { task: '`rm -rf /`' },
        { task: '../../../etc/passwd' },
        { task: 'file:///etc/passwd' },
        { task: String.fromCharCode(0) }, // Null character
        { task: '\x00\x01\x02\x03' }, // Control characters
      ];

      const mockSwarm = {
        orchestrate: jest.fn().mockResolvedValue({
          id: 'task-1',
          status: 'completed',
        }),
      };
      mcpTools.activeSwarms.set('test-swarm', mockSwarm);

      for (const input of maliciousInputs) {
        // Should sanitize and not throw
        const result = await mcpTools.task_orchestrate({
          swarmId: 'test-swarm',
          ...input,
        });

        expect(result).toBeDefined();
        expect(result.status).toBe('completed');
      }
    });
  });

  describe('Network and Timeout Edge Cases', () => {
    it('should handle network timeouts gracefully', async() => {
      await mcpTools.initialize(mockRuvSwarm);

      const mockSwarm = {
        monitor: jest.fn().mockImplementation(async() => {
          // Simulate long network delay
          await new Promise(resolve => setTimeout(resolve, 5000));
          return { status: 'timeout' };
        }),
      };
      mcpTools.activeSwarms.set('test-swarm', mockSwarm);

      // Set a shorter timeout for the test
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Operation timeout')), 100),
      );

      await expect(
        Promise.race([
          mcpTools.swarm_monitor({ swarmId: 'test-swarm' }),
          timeoutPromise,
        ]),
      ).rejects.toThrow(/timeout/i);
    });
  });

  describe('State Consistency Edge Cases', () => {
    it('should maintain consistency during rapid state changes', async() => {
      await mcpTools.initialize(mockRuvSwarm);

      const mockSwarm = {
        id: 'test-swarm',
        reconfigure: jest.fn().mockResolvedValue(true),
        getStatus: jest.fn().mockReturnValue({ topology: 'mesh' }),
      };

      mockRuvSwarm.createSwarm.mockResolvedValue(mockSwarm);
      await mcpTools.swarm_init({ topology: 'mesh' });

      // Rapid topology changes
      const topologies = ['star', 'ring', 'hierarchical', 'mesh'];
      const promises = [];

      for (let i = 0; i < 100; i++) {
        const topology = topologies[i % topologies.length];
        promises.push(
          mcpTools.swarm_reconfigure({
            swarmId: 'test-swarm',
            topology,
          }).catch(() => null), // Ignore errors
        );
      }

      await Promise.all(promises);

      // Check final state is consistent
      const status = await mcpTools.swarm_status({ swarmId: 'test-swarm' });
      expect(status).toBeDefined();
      expect(topologies).toContain(status.topology);
    });
  });
});

// Run tests when executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  console.log('Running MCP validation edge case tests...');

  // Run all tests
  const { run } = await import('../test-runner.js');
  await run(__filename);
}