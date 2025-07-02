/**
 * Error Handling and Recovery Edge Cases
 * Tests error propagation, recovery mechanisms, and failure scenarios
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { RuvSwarm } from '../../src/index-enhanced.js';
import { EnhancedMCPTools } from '../../src/mcp-tools-enhanced.js';
import { SwarmPersistence } from '../../src/persistence.js';

describe('Error Handling and Recovery Edge Cases', () => {
  let mcpTools;
  let mockRuvSwarm;

  beforeEach(() => {
    mockRuvSwarm = {
      createSwarm: jest.fn(),
      benchmark: jest.fn(),
    };
    mcpTools = new EnhancedMCPTools();
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Error Propagation Edge Cases', () => {
    it('should handle nested error chains', async() => {
      const createNestedError = () => {
        try {
          throw new Error('Level 1 error');
        } catch (level1) {
          try {
            throw new Error(`Level 2: ${level1.message}`);
          } catch (level2) {
            try {
              throw new Error(`Level 3: ${level2.message}`);
            } catch (level3) {
              throw new Error(`Level 4: ${level3.message}`);
            }
          }
        }
      };

      mockRuvSwarm.createSwarm.mockImplementation(() => {
        createNestedError();
      });

      await mcpTools.initialize(mockRuvSwarm);

      await expect(mcpTools.swarm_init({ topology: 'mesh' }))
        .rejects.toThrow(/Level 4.*Level 3.*Level 2.*Level 1/);
    });

    it('should handle error aggregation from multiple sources', async() => {
      const errors = [];

      const failingOperations = [
        () => Promise.reject(new Error('Database error')),
        () => Promise.reject(new Error('Network error')),
        () => Promise.reject(new Error('Validation error')),
        () => Promise.reject(new Error('Permission error')),
      ];

      const results = await Promise.allSettled(
        failingOperations.map(op => op()),
      );

      results.forEach(result => {
        if (result.status === 'rejected') {
          errors.push(result.reason.message);
        }
      });

      expect(errors).toHaveLength(4);
      expect(errors).toContain('Database error');
      expect(errors).toContain('Network error');
      expect(errors).toContain('Validation error');
      expect(errors).toContain('Permission error');
    });

    it('should handle circular error references', async() => {
      const createCircularError = () => {
        const error1 = new Error('Error 1');
        const error2 = new Error('Error 2');

        // Create circular reference
        error1.cause = error2;
        error2.cause = error1;

        return error1;
      };

      const circularError = createCircularError();

      // Should handle circular references without infinite loops
      expect(() => {
        JSON.stringify(circularError, null, 2);
      }).toThrow(/circular|Converting/);

      // Custom error handler should handle this gracefully
      const safeErrorString = String(circularError);
      expect(safeErrorString).toContain('Error 1');
    });
  });

  describe('Recovery Mechanism Edge Cases', () => {
    it('should implement exponential backoff with jitter', async() => {
      let attemptCount = 0;
      const maxAttempts = 5;
      const backoffTimes = [];

      const unreliableOperation = async() => {
        attemptCount++;
        const backoffTime = Math.pow(2, attemptCount - 1) * 100; // Exponential backoff
        const jitter = Math.random() * 100; // Add jitter
        const totalWait = backoffTime + jitter;

        backoffTimes.push(totalWait);

        if (attemptCount < maxAttempts) {
          throw new Error(`Attempt ${attemptCount} failed`);
        }

        return `Success after ${attemptCount} attempts`;
      };

      const retryWithBackoff = async(operation, maxRetries = 5) => {
        for (let i = 0; i < maxRetries; i++) {
          try {
            return await operation();
          } catch (error) {
            if (i === maxRetries - 1) {
              throw error;
            }

            const backoffTime = backoffTimes[i];
            await new Promise(resolve => setTimeout(resolve, backoffTime));
          }
        }
      };

      const result = await retryWithBackoff(unreliableOperation);

      expect(result).toContain('Success');
      expect(attemptCount).toBe(maxAttempts);
      expect(backoffTimes).toHaveLength(maxAttempts);

      // Verify exponential growth with jitter
      expect(backoffTimes[1]).toBeGreaterThan(backoffTimes[0]);
      expect(backoffTimes[2]).toBeGreaterThan(backoffTimes[1]);
    });

    it('should handle recovery from corrupted state', async() => {
      class StatefulService {
        constructor() {
          this.state = { healthy: true, data: [] };
        }

        async operation() {
          if (!this.state.healthy) {
            throw new Error('Service in corrupted state');
          }
          return 'Operation successful';
        }

        corruptState() {
          this.state.healthy = false;
          this.state.data = null;
        }

        async recover() {
          // Simulate recovery process
          this.state = { healthy: true, data: [] };
          await new Promise(resolve => setTimeout(resolve, 100));
        }

        async safeOperation() {
          try {
            return await this.operation();
          } catch (error) {
            if (error.message.includes('corrupted')) {
              await this.recover();
              return await this.operation();
            }
            throw error;
          }
        }
      }

      const service = new StatefulService();

      // First operation should succeed
      const result1 = await service.safeOperation();
      expect(result1).toBe('Operation successful');

      // Corrupt the state
      service.corruptState();

      // Operation should fail but recover
      const result2 = await service.safeOperation();
      expect(result2).toBe('Operation successful');
      expect(service.state.healthy).toBe(true);
    });

    it('should handle cascading failures with circuit breaker', async() => {
      class CircuitBreaker {
        constructor(threshold = 5, timeout = 60000) {
          this.failureThreshold = threshold;
          this.timeout = timeout;
          this.failureCount = 0;
          this.lastFailureTime = null;
          this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
        }

        async execute(operation) {
          if (this.state === 'OPEN') {
            if (Date.now() - this.lastFailureTime > this.timeout) {
              this.state = 'HALF_OPEN';
            } else {
              throw new Error('Circuit breaker is OPEN');
            }
          }

          try {
            const result = await operation();
            this.onSuccess();
            return result;
          } catch (error) {
            this.onFailure();
            throw error;
          }
        }

        onSuccess() {
          this.failureCount = 0;
          this.state = 'CLOSED';
        }

        onFailure() {
          this.failureCount++;
          this.lastFailureTime = Date.now();

          if (this.failureCount >= this.failureThreshold) {
            this.state = 'OPEN';
          }
        }
      }

      const circuitBreaker = new CircuitBreaker(3, 1000);
      let operationCount = 0;

      const flakyOperation = async() => {
        operationCount++;
        if (operationCount <= 5) {
          throw new Error(`Operation failed (${operationCount})`);
        }
        return 'Success';
      };

      // First 3 failures should succeed in failing
      for (let i = 0; i < 3; i++) {
        await expect(circuitBreaker.execute(flakyOperation))
          .rejects.toThrow(/Operation failed/);
      }

      expect(circuitBreaker.state).toBe('OPEN');

      // Next attempts should fail immediately due to circuit breaker
      await expect(circuitBreaker.execute(flakyOperation))
        .rejects.toThrow(/Circuit breaker is OPEN/);

      // Wait for timeout and try again
      await new Promise(resolve => setTimeout(resolve, 1100));

      // Should succeed now (operationCount > 5)
      const result = await circuitBreaker.execute(flakyOperation);
      expect(result).toBe('Success');
      expect(circuitBreaker.state).toBe('CLOSED');
    });
  });

  describe('Resource Cleanup on Failure', () => {
    it('should cleanup resources even when cleanup fails', async() => {
      const resources = [];
      let cleanupSuccesses = 0;
      let cleanupFailures = 0;

      class FailingResource {
        constructor(id, shouldFailCleanup = false) {
          this.id = id;
          this.shouldFailCleanup = shouldFailCleanup;
          this.acquired = true;
          resources.push(this);
        }

        async cleanup() {
          if (this.shouldFailCleanup) {
            cleanupFailures++;
            throw new Error(`Cleanup failed for resource ${this.id}`);
          }
          this.acquired = false;
          cleanupSuccesses++;
        }
      }

      const acquireResources = async() => {
        const acquired = [];
        try {
          // Acquire resources, some will fail cleanup
          for (let i = 0; i < 10; i++) {
            const shouldFail = i % 3 === 0; // Every 3rd resource fails cleanup
            acquired.push(new FailingResource(i, shouldFail));
          }

          // Simulate operation failure
          throw new Error('Operation failed');

        } catch (error) {
          // Cleanup all resources, even if some cleanups fail
          const cleanupPromises = acquired.map(resource =>
            resource.cleanup().catch(err => ({ error: err.message })),
          );

          const cleanupResults = await Promise.allSettled(cleanupPromises);

          // Count successful cleanups
          const successfulCleanups = cleanupResults.filter(
            result => result.status === 'fulfilled' && !result.value?.error,
          ).length;

          throw new Error(`Operation failed. Cleaned up ${successfulCleanups}/${acquired.length} resources`);
        }
      };

      await expect(acquireResources()).rejects.toThrow(/Operation failed/);

      expect(cleanupSuccesses).toBe(7); // 7 resources should cleanup successfully
      expect(cleanupFailures).toBe(3); // 3 resources should fail cleanup
    });

    it('should handle nested resource cleanup failures', async() => {
      const cleanupLog = [];

      class NestedResource {
        constructor(id, children = []) {
          this.id = id;
          this.children = children;
          this.acquired = true;
        }

        async cleanup() {
          cleanupLog.push(`Cleaning up ${this.id}`);

          // Cleanup children first
          const childCleanupPromises = this.children.map(child =>
            child.cleanup().catch(error => {
              cleanupLog.push(`Child cleanup failed: ${error.message}`);
              return { error };
            }),
          );

          await Promise.all(childCleanupPromises);

          // Then cleanup self
          if (this.id.includes('fail')) {
            throw new Error(`Failed to cleanup ${this.id}`);
          }

          this.acquired = false;
          cleanupLog.push(`Cleaned up ${this.id}`);
        }
      }

      // Create nested resource structure
      const leaf1 = new NestedResource('leaf-1');
      const leaf2 = new NestedResource('leaf-2-fail');
      const leaf3 = new NestedResource('leaf-3');

      const branch1 = new NestedResource('branch-1', [leaf1, leaf2]);
      const branch2 = new NestedResource('branch-2-fail', [leaf3]);

      const root = new NestedResource('root', [branch1, branch2]);

      await expect(root.cleanup()).rejects.toThrow(/Failed to cleanup/);

      // Check cleanup log
      expect(cleanupLog).toContain('Cleaning up root');
      expect(cleanupLog).toContain('Cleaning up branch-1');
      expect(cleanupLog).toContain('Cleaning up leaf-1');
      expect(cleanupLog).toContain('Child cleanup failed: Failed to cleanup leaf-2-fail');
    });
  });

  describe('Error Context Preservation', () => {
    it('should preserve error context through async boundaries', async() => {
      const createContextualError = (context) => {
        const error = new Error('Base error');
        error.context = context;
        error.timestamp = Date.now();
        return error;
      };

      const asyncOperation1 = async() => {
        await new Promise(resolve => setImmediate(resolve));
        throw createContextualError({ operation: 'async-1', step: 'validation' });
      };

      const asyncOperation2 = async() => {
        try {
          await asyncOperation1();
        } catch (error) {
          const wrappedError = new Error(`Wrapped: ${error.message}`);
          wrappedError.originalError = error;
          wrappedError.context = { operation: 'async-2', step: 'processing' };
          throw wrappedError;
        }
      };

      const asyncOperation3 = async() => {
        try {
          await asyncOperation2();
        } catch (error) {
          const finalError = new Error(`Final: ${error.message}`);
          finalError.errorChain = [error];
          finalError.context = { operation: 'async-3', step: 'finalization' };
          throw finalError;
        }
      };

      try {
        await asyncOperation3();
      } catch (error) {
        expect(error.message).toContain('Final: Wrapped: Base error');
        expect(error.context.operation).toBe('async-3');
        expect(error.errorChain[0].originalError.context.operation).toBe('async-1');
      }
    });

    it('should handle error context serialization', async() => {
      const createComplexError = () => {
        const error = new Error('Complex error');
        error.context = {
          user: { id: 123, name: 'test' },
          operation: 'data-processing',
          metadata: {
            timestamp: new Date(),
            circular: {}, // Will create circular reference
          },
        };

        // Create circular reference
        error.context.metadata.circular.self = error.context.metadata;

        return error;
      };

      const complexError = createComplexError();

      // Test safe serialization
      const safeSerialize = (obj) => {
        const seen = new WeakSet();
        return JSON.stringify(obj, (key, value) => {
          if (typeof value === 'object' && value !== null) {
            if (seen.has(value)) {
              return '[Circular]';
            }
            seen.add(value);
          }
          return value;
        });
      };

      const serialized = safeSerialize({
        message: complexError.message,
        context: complexError.context,
      });

      expect(serialized).toContain('Complex error');
      expect(serialized).toContain('[Circular]');
      expect(serialized).toContain('data-processing');
    });
  });

  describe('Timeout and Cancellation Edge Cases', () => {
    it('should handle operation cancellation during execution', async() => {
      const { AbortController } = globalThis;

      if (!AbortController) {
        // Skip if AbortController not available
        return;
      }

      const controller = new AbortController();
      const { signal } = controller;

      const longRunningOperation = async(signal) => {
        for (let i = 0; i < 1000; i++) {
          if (signal.aborted) {
            throw new Error('Operation was cancelled');
          }

          await new Promise(resolve => setTimeout(resolve, 10));
        }
        return 'Completed';
      };

      // Cancel after 100ms
      setTimeout(() => controller.abort(), 100);

      await expect(longRunningOperation(signal))
        .rejects.toThrow('Operation was cancelled');
    });

    it('should handle timeout with resource cleanup', async() => {
      const resources = [];

      const operationWithResources = async(timeoutMs) => {
        const acquiredResources = [];
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error('Operation timeout')), timeoutMs);
        });

        try {
          const workPromise = (async() => {
            // Acquire resources
            for (let i = 0; i < 5; i++) {
              const resource = { id: i, acquired: true };
              acquiredResources.push(resource);
              resources.push(resource);
              await new Promise(resolve => setTimeout(resolve, 50));
            }

            // Do work (this will timeout)
            await new Promise(resolve => setTimeout(resolve, 1000));
            return 'Work completed';
          })();

          return await Promise.race([workPromise, timeoutPromise]);
        } catch (error) {
          // Cleanup resources on timeout
          acquiredResources.forEach(resource => {
            resource.acquired = false;
          });
          throw error;
        }
      };

      await expect(operationWithResources(200)).rejects.toThrow('Operation timeout');

      // Check that resources were cleaned up
      const acquiredResources = resources.filter(r => r.acquired);
      expect(acquiredResources).toHaveLength(0);
    });
  });

  describe('Error Recovery Strategies', () => {
    it('should implement retry with different strategies', async() => {
      const strategies = {
        immediate: (attempt) => 0,
        linear: (attempt) => attempt * 100,
        exponential: (attempt) => Math.pow(2, attempt) * 100,
        fibonacci: (() => {
          const fib = [100, 100];
          return (attempt) => {
            if (attempt < 2) {
              return fib[attempt];
            }
            const next = fib[0] + fib[1];
            fib[0] = fib[1];
            fib[1] = next;
            return next;
          };
        })(),
      };

      for (const [strategyName, backoffFn] of Object.entries(strategies)) {
        let attempts = 0;
        const maxAttempts = 4;

        const retryWithStrategy = async(operation, strategy) => {
          for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
              return await operation();
            } catch (error) {
              if (attempt === maxAttempts - 1) {
                throw error;
              }

              const delay = strategy(attempt);
              await new Promise(resolve => setTimeout(resolve, Math.min(delay, 1000)));
            }
          }
        };

        const flakyOperation = async() => {
          attempts++;
          if (attempts < 3) {
            throw new Error(`${strategyName} attempt ${attempts} failed`);
          }
          return `${strategyName} succeeded`;
        };

        const result = await retryWithStrategy(flakyOperation, backoffFn);
        expect(result).toContain('succeeded');

        // Reset for next strategy
        attempts = 0;
      }
    });
  });
});

// Run tests when executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  console.log('Running error handling and recovery edge case tests...');

  // Run all tests
  const { run } = await import('../test-runner.js');
  await run(__filename);
}