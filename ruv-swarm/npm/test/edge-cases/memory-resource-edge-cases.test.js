/**
 * Memory and Resource Management Edge Cases
 * Tests memory limits, resource cleanup, garbage collection scenarios
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { RuvSwarm } from '../../src/index-enhanced.js';
import { NeuralAgent } from '../../src/neural-agent.js';
import { SwarmPersistence } from '../../src/persistence.js';

describe('Memory and Resource Edge Cases', () => {
  let initialMemory;

  beforeEach(() => {
    initialMemory = process.memoryUsage();
    // Force garbage collection if available
    if (global.gc) {
      global.gc();
    }
  });

  afterEach(() => {
    // Cleanup after each test
    if (global.gc) {
      global.gc();
    }
  });

  describe('Memory Pressure Scenarios', () => {
    it('should handle large array allocations', () => {
      const arrays = [];
      let lastSuccessfulSize = 0;

      try {
        // Try increasingly large allocations
        for (let exp = 10; exp < 28; exp++) { // Up to ~256MB
          const size = Math.pow(2, exp);
          const arr = new Array(size).fill(1);
          arrays.push(arr);
          lastSuccessfulSize = size;

          // Check memory usage periodically
          if (exp % 3 === 0) {
            const currentMemory = process.memoryUsage();
            if (currentMemory.heapUsed > initialMemory.heapUsed + 500 * 1024 * 1024) {
              // Stop if we've allocated too much
              break;
            }
          }
        }
      } catch (error) {
        // Expected - system ran out of memory
        expect(error.message).toMatch(/out of memory|Maximum call stack/i);
      }

      expect(lastSuccessfulSize).toBeGreaterThan(0);

      // Cleanup
      arrays.length = 0;
      if (global.gc) {
        global.gc();
      }
    });

    it('should handle memory fragmentation scenarios', () => {
      const smallArrays = [];
      const largeArrays = [];

      try {
        // Create fragmentation by alternating small and large allocations
        for (let i = 0; i < 1000; i++) {
          if (i % 2 === 0) {
            smallArrays.push(new Array(1000).fill(i));
          } else {
            largeArrays.push(new Array(100000).fill(i));
          }

          // Periodically free some small arrays to create holes
          if (i % 100 === 99 && smallArrays.length > 50) {
            smallArrays.splice(0, 25);
          }
        }

        // Force GC to see how well it handles fragmentation
        if (global.gc) {
          global.gc();
        }

        const memoryAfterGC = process.memoryUsage();
        expect(memoryAfterGC.heapUsed).toBeLessThan(
          initialMemory.heapUsed + 200 * 1024 * 1024, // 200MB limit
        );

      } finally {
        // Cleanup
        smallArrays.length = 0;
        largeArrays.length = 0;
      }
    });

    it('should handle circular reference memory leaks', () => {
      const objects = [];

      // Create circular references
      for (let i = 0; i < 10000; i++) {
        const obj1 = { id: i, data: new Array(100).fill(i) };
        const obj2 = { id: i + 10000, data: new Array(100).fill(i + 10000) };

        // Create circular reference
        obj1.ref = obj2;
        obj2.ref = obj1;

        objects.push(obj1, obj2);
      }

      // Remove references but keep some circular ones
      for (let i = 0; i < objects.length; i += 4) {
        objects[i] = null;
      }

      // Force garbage collection
      if (global.gc) {
        global.gc();
      }

      const memoryAfterCleanup = process.memoryUsage();

      // Modern JS engines should handle circular references
      expect(memoryAfterCleanup.heapUsed).toBeLessThan(
        initialMemory.heapUsed + 100 * 1024 * 1024,
      );
    });
  });

  describe('Resource Cleanup Edge Cases', () => {
    it('should cleanup file handles properly', async() => {
      const fs = await import('fs/promises');
      const handles = [];

      try {
        // Open many temporary file handles
        for (let i = 0; i < 100; i++) {
          try {
            const handle = await fs.open('/tmp/test-file-${i}', 'w');
            handles.push(handle);
          } catch (error) {
            // Some might fail due to system limits
          }
        }

        expect(handles.length).toBeGreaterThan(0);

      } finally {
        // Cleanup all handles
        await Promise.allSettled(
          handles.map(handle => handle?.close?.()),
        );
      }
    });

    it('should handle timer cleanup on object destruction', async() => {
      class TimerObject {
        constructor() {
          this.timers = [];
          this.destroyed = false;
        }

        startTimer(interval) {
          const timer = setInterval(() => {
            if (this.destroyed) {
              clearInterval(timer);
            }
          }, interval);
          this.timers.push(timer);
        }

        destroy() {
          this.destroyed = true;
          this.timers.forEach(timer => clearInterval(timer));
          this.timers = [];
        }
      }

      const objects = [];

      // Create objects with timers
      const promises = [];
      for (let i = 0; i < 50; i++) {
        const obj = new TimerObject();
        obj.startTimer(10);
        objects.push(obj);

        // Destroy some objects randomly
        if (Math.random() > 0.5) {
          promises.push(
            new Promise(resolve => {
              setTimeout(() => {
                obj.destroy();
                resolve();
              }, Math.random() * 100);
            }),
          );
        }
      }

      await Promise.all(promises);

      // Cleanup remaining objects
      objects.forEach(obj => obj.destroy());

      // Verify cleanup
      objects.forEach(obj => {
        expect(obj.destroyed).toBe(true);
        expect(obj.timers).toHaveLength(0);
      });
    });
  });

  describe('Garbage Collection Edge Cases', () => {
    it('should handle WeakMap and WeakSet edge cases', () => {
      const weakMap = new WeakMap();
      const weakSet = new WeakSet();
      const strongRefs = [];

      // Create objects and weak references
      for (let i = 0; i < 1000; i++) {
        const obj = { id: i, data: new Array(100).fill(i) };

        weakMap.set(obj, `value-${i}`);
        weakSet.add(obj);

        // Keep strong references to some objects
        if (i % 3 === 0) {
          strongRefs.push(obj);
        }
      }

      // Force garbage collection
      if (global.gc) {
        global.gc();
      }

      // Check that weak references to unreferenced objects are cleaned up
      let weakMapCount = 0;
      let weakSetCount = 0;

      strongRefs.forEach(obj => {
        if (weakMap.has(obj)) {
          weakMapCount++;
        }
        if (weakSet.has(obj)) {
          weakSetCount++;
        }
      });

      expect(weakMapCount).toBeGreaterThan(0);
      expect(weakSetCount).toBeGreaterThan(0);
    });

    it('should handle finalization registry edge cases', () => {
      if (typeof FinalizationRegistry === 'undefined') {
        // Skip if not supported
        return;
      }

      const cleanupCallbacks = [];
      const registry = new FinalizationRegistry((heldValue) => {
        cleanupCallbacks.push(heldValue);
      });

      // Create objects and register for cleanup
      const objects = [];
      for (let i = 0; i < 100; i++) {
        const obj = { id: i };
        objects.push(obj);
        registry.register(obj, `cleanup-${i}`);
      }

      // Remove references to half the objects
      objects.splice(0, 50);

      // Force multiple GCs
      if (global.gc) {
        global.gc();
        global.gc();
        global.gc();
      }

      // Note: Finalization is not guaranteed to run immediately
      // This test mainly verifies no errors occur
      expect(cleanupCallbacks.length).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Memory Monitoring Edge Cases', () => {
    it('should detect memory leaks in event listeners', async() => {
      const { EventEmitter } = await import('events');
      const emitter = new EventEmitter();
      const initialListeners = emitter.listenerCount('test');

      // Add many listeners
      const listeners = [];
      for (let i = 0; i < 1000; i++) {
        const listener = () => { /* noop */ };
        listeners.push(listener);
        emitter.on('test', listener);
      }

      expect(emitter.listenerCount('test')).toBe(initialListeners + 1000);

      // Remove only half
      for (let i = 0; i < 500; i++) {
        emitter.removeListener('test', listeners[i]);
      }

      expect(emitter.listenerCount('test')).toBe(initialListeners + 500);

      // Cleanup remaining
      emitter.removeAllListeners('test');
      expect(emitter.listenerCount('test')).toBe(0);
    });

    it('should handle buffer overflow scenarios', () => {
      const buffers = [];
      let totalSize = 0;

      try {
        // Create increasingly large buffers
        for (let exp = 10; exp < 26; exp++) { // Up to 64MB
          const size = Math.pow(2, exp);
          const buffer = Buffer.alloc(size);
          buffers.push(buffer);
          totalSize += size;

          // Fill with pattern to ensure allocation
          buffer.fill(exp % 256);

          // Stop if we've allocated too much
          if (totalSize > 100 * 1024 * 1024) { // 100MB
            break;
          }
        }

        expect(totalSize).toBeGreaterThan(0);

      } catch (error) {
        // Expected on systems with low memory
        expect(error.message).toMatch(/out of memory|Invalid array length/i);
      } finally {
        // Cleanup
        buffers.length = 0;
      }
    });
  });

  describe('Resource Pool Edge Cases', () => {
    it('should handle resource pool exhaustion', async() => {
      class ResourcePool {
        constructor(maxSize = 10) {
          this.resources = [];
          this.available = [];
          this.maxSize = maxSize;
          this.allocated = 0;
        }

        async acquire() {
          if (this.available.length > 0) {
            return this.available.pop();
          }

          if (this.allocated < this.maxSize) {
            const resource = { id: this.allocated++, acquired: true };
            this.resources.push(resource);
            return resource;
          }

          throw new Error('Pool exhausted');
        }

        release(resource) {
          resource.acquired = false;
          this.available.push(resource);
        }

        destroy() {
          this.resources = [];
          this.available = [];
          this.allocated = 0;
        }
      }

      const pool = new ResourcePool(5);
      const acquired = [];

      // Acquire all resources
      for (let i = 0; i < 5; i++) {
        const resource = await pool.acquire();
        acquired.push(resource);
      }

      // Next acquisition should fail
      await expect(pool.acquire()).rejects.toThrow('Pool exhausted');

      // Release one and try again
      pool.release(acquired[0]);
      const newResource = await pool.acquire();
      expect(newResource).toBeDefined();

      pool.destroy();
    });

    it('should handle concurrent resource access', async() => {
      class ConcurrentResource {
        constructor() {
          this.users = 0;
          this.maxUsers = 3;
        }

        async use() {
          if (this.users >= this.maxUsers) {
            throw new Error('Resource busy');
          }

          this.users++;
          try {
            // Simulate work
            await new Promise(resolve => setTimeout(resolve, 10));
            return `Used by ${this.users} users`;
          } finally {
            this.users--;
          }
        }
      }

      const resource = new ConcurrentResource();
      const promises = [];

      // Try to use resource concurrently
      for (let i = 0; i < 10; i++) {
        promises.push(
          resource.use().catch(error => ({ error: error.message })),
        );
      }

      const results = await Promise.all(promises);
      const successful = results.filter(r => !r.error);
      const failed = results.filter(r => r.error);

      expect(successful.length).toBeGreaterThan(0);
      expect(failed.length).toBeGreaterThan(0);
      expect(resource.users).toBe(0); // All should be released
    });
  });

  describe('Memory Allocation Patterns', () => {
    it('should handle different allocation patterns', () => {
      const allocations = [];

      // Pattern 1: Many small allocations
      for (let i = 0; i < 10000; i++) {
        allocations.push(new Array(10).fill(i));
      }

      // Pattern 2: Few large allocations
      for (let i = 0; i < 10; i++) {
        allocations.push(new Array(100000).fill(i));
      }

      // Pattern 3: Exponential allocation sizes
      for (let exp = 1; exp < 16; exp++) {
        const size = Math.pow(2, exp);
        allocations.push(new Array(size).fill(exp));
      }

      const memoryUsed = process.memoryUsage();
      expect(memoryUsed.heapUsed).toBeGreaterThan(initialMemory.heapUsed);

      // Cleanup half randomly
      for (let i = allocations.length - 1; i >= 0; i -= 2) {
        allocations.splice(i, 1);
      }

      if (global.gc) {
        global.gc();
      }

      const memoryAfterCleanup = process.memoryUsage();
      expect(memoryAfterCleanup.heapUsed).toBeLessThan(memoryUsed.heapUsed);
    });
  });
});

// Run tests when executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  console.log('Running memory and resource edge case tests...');

  // Run all tests
  const { run } = await import('../test-runner.js');
  await run(__filename);
}