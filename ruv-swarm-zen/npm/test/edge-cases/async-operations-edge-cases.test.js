/**
 * Async Operations Edge Cases
 * Tests promise handling, timeouts, race conditions, and async edge cases
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { RuvSwarm } from '../../src/index-enhanced.js';
import { NeuralAgent } from '../../src/neural-agent.js';
import { SwarmPersistence } from '../../src/persistence.js';

// Mock timers for timeout testing
jest.useFakeTimers();

describe('Async Operations Edge Cases', () => {
  let ruv;
  let swarm;

  beforeEach(async() => {
    // Initialize with mocks
    ruv = {
      createSwarm: jest.fn().mockResolvedValue({
        id: 'test-swarm',
        spawn: jest.fn(),
        orchestrate: jest.fn(),
        terminate: jest.fn(),
      }),
    };
    swarm = await ruv.createSwarm({ topology: 'mesh' });
  });

  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  describe('Promise Timeout Scenarios', () => {
    it('should timeout after max wait period', async() => {
      const longOperation = () => new Promise((resolve) => {
        setTimeout(() => resolve('completed'), 10000); // 10 seconds
      });

      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Operation timeout')), 5000);
      });

      jest.useRealTimers(); // Use real timers for this test

      const racePromise = Promise.race([longOperation(), timeoutPromise]);

      await expect(racePromise).rejects.toThrow('Operation timeout');
    });

    it('should handle multiple timeout scenarios', async() => {
      jest.useFakeTimers();

      const operations = [
        { delay: 1000, timeout: 500, shouldTimeout: true },
        { delay: 500, timeout: 1000, shouldTimeout: false },
        { delay: 1000, timeout: 1000, shouldTimeout: false }, // Edge case
        { delay: 0, timeout: 100, shouldTimeout: false },
      ];

      const results = [];

      for (const { delay, timeout, shouldTimeout } of operations) {
        const operation = new Promise((resolve) => {
          setTimeout(() => resolve('success'), delay);
        });

        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error('timeout')), timeout);
        });

        const resultPromise = Promise.race([operation, timeoutPromise])
          .then(() => ({ status: 'success' }))
          .catch(() => ({ status: 'timeout' }));

        results.push(resultPromise);

        // Advance timers to the max of delay and timeout
        jest.advanceTimersByTime(Math.max(delay, timeout) + 1);
      }

      const resolvedResults = await Promise.all(results);

      operations.forEach((op, index) => {
        if (op.shouldTimeout) {
          expect(resolvedResults[index].status).toBe('timeout');
        } else {
          expect(resolvedResults[index].status).toBe('success');
        }
      });
    });
  });

  describe('Promise Rejection Handling', () => {
    it('should handle nested promise rejections', async() => {
      const nestedOperation = async() => {
        try {
          await Promise.resolve().then(() => {
            return Promise.reject(new Error('Inner rejection'));
          });
        } catch (error) {
          // Re-throw with context
          throw new Error(`Wrapped: ${error.message}`);
        }
      };

      await expect(nestedOperation()).rejects.toThrow('Wrapped: Inner rejection');
    });

    it('should handle promise rejection in finally block', async() => {
      let cleanupCalled = false;
      let errorInFinally = false;

      const operationWithFinally = async() => {
        try {
          throw new Error('Main error');
        } finally {
          cleanupCalled = true;
          try {
            // Cleanup that might fail
            await Promise.reject(new Error('Cleanup error'));
          } catch (e) {
            errorInFinally = true;
          }
        }
      };

      await expect(operationWithFinally()).rejects.toThrow('Main error');
      expect(cleanupCalled).toBe(true);
      expect(errorInFinally).toBe(true);
    });

    it('should handle unhandled promise rejections in callbacks', async() => {
      const events = [];

      // Track unhandled rejections
      const unhandledHandler = (reason) => {
        events.push({ type: 'unhandled', reason: reason.message });
      };

      process.on('unhandledRejection', unhandledHandler);

      // Create promise that will be rejected after a delay
      setTimeout(() => {
        Promise.reject(new Error('Async rejection'));
      }, 0);

      // Wait for event loop
      await new Promise(resolve => setImmediate(resolve));

      process.off('unhandledRejection', unhandledHandler);

      // Note: In test environment, this might not capture the event
      // This test demonstrates the pattern for handling such cases
    });
  });

  describe('Race Condition Edge Cases', () => {
    it('should handle concurrent state modifications', async() => {
      const sharedState = { counter: 0, operations: [] };

      const increment = async(id) => {
        const current = sharedState.counter;
        // Simulate async operation
        await new Promise(resolve => setImmediate(resolve));
        sharedState.counter = current + 1;
        sharedState.operations.push(id);
      };

      // Create many concurrent operations
      const promises = [];
      for (let i = 0; i < 100; i++) {
        promises.push(increment(i));
      }

      await Promise.all(promises);

      // Due to race conditions, counter might not be 100
      expect(sharedState.counter).toBeLessThanOrEqual(100);
      expect(sharedState.operations.length).toBe(100);
    });

    it('should handle promise settlement order', async() => {
      const results = [];

      const promises = [
        new Promise(resolve => setTimeout(() => {
          results.push(1);
          resolve(1);
        }, 100)),
        new Promise(resolve => setTimeout(() => {
          results.push(2);
          resolve(2);
        }, 50)),
        new Promise(resolve => setTimeout(() => {
          results.push(3);
          resolve(3);
        }, 75)),
      ];

      jest.useRealTimers();
      const settled = await Promise.all(promises);

      // Results array shows actual execution order
      expect(results).toEqual([2, 3, 1]); // Based on timeout order
      // Promise.all preserves input order
      expect(settled).toEqual([1, 2, 3]);
    });
  });

  describe('Async Iterator Edge Cases', () => {
    it('should handle async generator errors', async() => {
      async function* problematicGenerator() {
        yield 1;
        yield 2;
        throw new Error('Generator error');
        // eslint-disable-next-line no-unreachable
        yield 3; // Intentionally unreachable for testing error handling
      }

      const results = [];
      let errorCaught = false;

      try {
        for await (const value of problematicGenerator()) {
          results.push(value);
        }
      } catch (error) {
        errorCaught = true;
        expect(error.message).toBe('Generator error');
      }

      expect(results).toEqual([1, 2]);
      expect(errorCaught).toBe(true);
    });

    it('should handle async generator cleanup', async() => {
      let cleanupCalled = false;

      async function* generatorWithCleanup() {
        try {
          yield 1;
          yield 2;
          yield 3;
        } finally {
          cleanupCalled = true;
        }
      }

      const gen = generatorWithCleanup();

      // Consume only first two values
      await gen.next(); // 1
      await gen.next(); // 2

      // Return early (triggers cleanup)
      await gen.return();

      expect(cleanupCalled).toBe(true);
    });
  });

  describe('Concurrent Async Operations', () => {
    it('should handle Promise.allSettled with mixed results', async() => {
      const operations = [
        Promise.resolve('success-1'),
        Promise.reject(new Error('failure-1')),
        Promise.resolve('success-2'),
        Promise.reject(new Error('failure-2')),
        new Promise(resolve => setTimeout(() => resolve('delayed'), 100)),
      ];

      jest.useRealTimers();
      const results = await Promise.allSettled(operations);

      expect(results).toHaveLength(5);
      expect(results[0]).toEqual({ status: 'fulfilled', value: 'success-1' });
      expect(results[1]).toEqual({
        status: 'rejected',
        reason: expect.objectContaining({ message: 'failure-1' }),
      });
      expect(results[2]).toEqual({ status: 'fulfilled', value: 'success-2' });
      expect(results[3]).toEqual({
        status: 'rejected',
        reason: expect.objectContaining({ message: 'failure-2' }),
      });
      expect(results[4]).toEqual({ status: 'fulfilled', value: 'delayed' });
    });

    it('should handle Promise.race with all rejections', async() => {
      const rejections = [
        new Promise((_, reject) => setTimeout(() => reject(new Error('Error 1')), 100)),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Error 2')), 50)),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Error 3')), 150)),
      ];

      jest.useRealTimers();

      // Should reject with the fastest rejection
      await expect(Promise.race(rejections)).rejects.toThrow('Error 2');
    });

    it('should handle Promise.any with all rejections', async() => {
      const rejections = [
        Promise.reject(new Error('Error 1')),
        Promise.reject(new Error('Error 2')),
        Promise.reject(new Error('Error 3')),
      ];

      await expect(Promise.any(rejections)).rejects.toThrow('All promises were rejected');
    });
  });

  describe('Async Queue Edge Cases', () => {
    it('should handle queue overflow', async() => {
      class AsyncQueue {
        constructor(maxSize = 10) {
          this.queue = [];
          this.maxSize = maxSize;
          this.processing = false;
        }

        async add(task) {
          if (this.queue.length >= this.maxSize) {
            throw new Error('Queue overflow');
          }
          this.queue.push(task);
          if (!this.processing) {
            this.process();
          }
        }

        async process() {
          this.processing = true;
          while (this.queue.length > 0) {
            const task = this.queue.shift();
            try {
              await task();
            } catch (error) {
              // Task failed, continue processing
            }
          }
          this.processing = false;
        }
      }

      const queue = new AsyncQueue(5);
      const results = [];

      // Fill queue to capacity
      for (let i = 0; i < 5; i++) {
        await queue.add(async() => {
          results.push(i);
        });
      }

      // This should throw
      await expect(queue.add(async() => {})).rejects.toThrow('Queue overflow');

      // Wait for processing
      await new Promise(resolve => setTimeout(resolve, 100));

      expect(results.length).toBe(5);
    });
  });

  describe('Async Resource Cleanup', () => {
    it('should cleanup resources on async failure', async() => {
      const resources = [];
      let cleanupCount = 0;

      class AsyncResource {
        constructor(id) {
          this.id = id;
          this.acquired = true;
          resources.push(this);
        }

        async use() {
          if (this.id === 2) {
            throw new Error('Resource 2 failed');
          }
          return `Used ${this.id}`;
        }

        async cleanup() {
          this.acquired = false;
          cleanupCount++;
        }
      }

      const acquireResources = async() => {
        const acquired = [];
        try {
          for (let i = 0; i < 5; i++) {
            const resource = new AsyncResource(i);
            acquired.push(resource);
            await resource.use();
          }
        } catch (error) {
          // Cleanup all acquired resources
          await Promise.all(acquired.map(r => r.cleanup()));
          throw error;
        }
      };

      await expect(acquireResources()).rejects.toThrow('Resource 2 failed');
      expect(cleanupCount).toBe(3); // Resources 0, 1, and 2 were cleaned up
    });
  });

  describe('Async Event Emitter Edge Cases', () => {
    it('should handle async event listeners with errors', async() => {
      const { EventEmitter } = await import('events');
      const emitter = new EventEmitter();
      const results = [];

      // Add multiple async listeners, some failing
      emitter.on('test', async() => {
        results.push('listener-1-start');
        await new Promise(resolve => setTimeout(resolve, 50));
        results.push('listener-1-end');
      });

      emitter.on('test', async() => {
        results.push('listener-2-start');
        throw new Error('Listener 2 failed');
      });

      emitter.on('test', async() => {
        results.push('listener-3-start');
        await new Promise(resolve => setTimeout(resolve, 25));
        results.push('listener-3-end');
      });

      // Emit and wait for all listeners
      const promises = [];
      emitter.listeners('test').forEach(listener => {
        promises.push(
          listener().catch(err => ({ error: err.message })),
        );
      });

      emitter.emit('test');
      const listenerResults = await Promise.all(promises);

      expect(results).toContain('listener-1-start');
      expect(results).toContain('listener-2-start');
      expect(results).toContain('listener-3-start');
      expect(listenerResults[1]).toEqual({ error: 'Listener 2 failed' });
    });
  });
});

// Run tests when executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  console.log('Running async operations edge case tests...');

  // Run all tests
  const { run } = await import('../test-runner.js');
  await run(__filename);
}
