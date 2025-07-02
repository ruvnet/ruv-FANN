/**
 * Edge case and error handling tests for persistence module
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import Database from 'better-sqlite3';
import { promises as fs } from 'fs';
import path from 'path';

// Mock modules
jest.mock('better-sqlite3');
jest.mock('fs', () => ({
  promises: {
    access: jest.fn(),
    mkdir: jest.fn(),
    unlink: jest.fn(),
  },
}));

// Import after mocking
import { PersistenceManager } from '../src/persistence.js';

describe('PersistenceManager Edge Cases', () => {
  let persistence;
  let mockDb;
  let mockStmt;

  beforeEach(() => {
    mockStmt = {
      run: jest.fn(),
      get: jest.fn(),
      all: jest.fn(),
      finalize: jest.fn(),
    };

    mockDb = {
      prepare: jest.fn().mockReturnValue(mockStmt),
      exec: jest.fn(),
      close: jest.fn(),
      transaction: jest.fn((fn) => fn),
      pragma: jest.fn(),
    };

    Database.mockReturnValue(mockDb);
    persistence = new PersistenceManager();
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Database Connection Issues', () => {
    it('should handle database initialization failure', async () => {
      Database.mockImplementation(() => {
        throw new Error('Cannot open database');
      });

      await expect(async () => {
        new PersistenceManager();
      }).rejects.toThrow('Cannot open database');
    });

    it('should handle database corruption', async () => {
      mockDb.exec.mockImplementation(() => {
        throw new Error('Database disk image is malformed');
      });

      await expect(persistence.initialize()).rejects.toThrow('malformed');
    });

    it('should retry on database lock', async () => {
      let attempts = 0;
      mockStmt.run.mockImplementation(() => {
        attempts++;
        if (attempts < 3) {
          throw new Error('database is locked');
        }
        return { changes: 1 };
      });

      await persistence.initialize();
      const result = await persistence.storeMemory('test', { data: 'value' });
      
      expect(result).toBe(true);
      expect(attempts).toBe(3);
    });
  });

  describe('Memory Storage Edge Cases', () => {
    beforeEach(async () => {
      await persistence.initialize();
    });

    it('should handle extremely large memory values', async () => {
      const largeData = {
        array: new Array(10000).fill('x'.repeat(100)),
        nested: {}
      };
      
      // Create deeply nested object
      let current = largeData.nested;
      for (let i = 0; i < 100; i++) {
        current.next = { level: i };
        current = current.next;
      }

      mockStmt.run.mockReturnValue({ changes: 1 });
      
      const result = await persistence.storeMemory('large', largeData);
      expect(result).toBe(true);
    });

    it('should handle circular references in memory data', async () => {
      const circularData = { name: 'test' };
      circularData.self = circularData;

      await expect(persistence.storeMemory('circular', circularData))
        .rejects.toThrow('Converting circular structure');
    });

    it('should handle special characters in keys', async () => {
      const specialKeys = [
        'key/with/slashes',
        'key with spaces',
        'key"with"quotes',
        'key\'with\'apostrophes',
        'key\\with\\backslashes',
        'key\nwith\nnewlines',
        'key\twith\ttabs',
        '🔑emoji-key🗝️',
        'key;drop table memories;--',
      ];

      mockStmt.run.mockReturnValue({ changes: 1 });
      mockStmt.get.mockReturnValue({ value: '{"test": true}' });

      for (const key of specialKeys) {
        const stored = await persistence.storeMemory(key, { test: true });
        expect(stored).toBe(true);
        
        const retrieved = await persistence.retrieveMemory(key);
        expect(retrieved).toEqual({ test: true });
      }
    });

    it('should handle null and undefined values', async () => {
      mockStmt.run.mockReturnValue({ changes: 1 });
      
      await expect(persistence.storeMemory(null, {}))
        .rejects.toThrow();
      
      await expect(persistence.storeMemory(undefined, {}))
        .rejects.toThrow();
      
      await expect(persistence.storeMemory('key', null))
        .rejects.toThrow();
      
      await expect(persistence.storeMemory('key', undefined))
        .rejects.toThrow();
    });

    it('should handle concurrent memory operations', async () => {
      mockStmt.run.mockReturnValue({ changes: 1 });
      
      const promises = [];
      for (let i = 0; i < 100; i++) {
        promises.push(persistence.storeMemory(`key-${i}`, { index: i }));
      }
      
      const results = await Promise.all(promises);
      expect(results.every(r => r === true)).toBe(true);
    });
  });

  describe('Memory Retrieval Edge Cases', () => {
    beforeEach(async () => {
      await persistence.initialize();
    });

    it('should handle corrupted JSON data', async () => {
      mockStmt.get.mockReturnValue({ value: 'invalid json{' });
      
      const result = await persistence.retrieveMemory('corrupted');
      expect(result).toBeNull();
    });

    it('should handle missing memory gracefully', async () => {
      mockStmt.get.mockReturnValue(undefined);
      
      const result = await persistence.retrieveMemory('nonexistent');
      expect(result).toBeNull();
    });

    it('should handle database errors during retrieval', async () => {
      mockStmt.get.mockImplementation(() => {
        throw new Error('Disk I/O error');
      });
      
      await expect(persistence.retrieveMemory('key'))
        .rejects.toThrow('Disk I/O error');
    });
  });

  describe('Memory Listing Edge Cases', () => {
    beforeEach(async () => {
      await persistence.initialize();
    });

    it('should handle complex glob patterns', async () => {
      const patterns = [
        '**/deep/path/**',
        '[a-z]*[0-9]',
        '?(a|b|c)',
        '!(exclude)*',
        '@(pattern1|pattern2)',
      ];

      mockStmt.all.mockReturnValue([
        { key: 'match1', value: '{}', metadata: '{}' },
        { key: 'match2', value: '{}', metadata: '{}' },
      ]);

      for (const pattern of patterns) {
        const results = await persistence.listMemory(pattern);
        expect(Array.isArray(results)).toBe(true);
      }
    });

    it('should handle empty results', async () => {
      mockStmt.all.mockReturnValue([]);
      
      const results = await persistence.listMemory('*');
      expect(results).toEqual([]);
    });

    it('should handle very large result sets', async () => {
      const largeResults = Array(10000).fill(null).map((_, i) => ({
        key: `key-${i}`,
        value: '{"data": "test"}',
        metadata: '{"accessed": 0}',
      }));
      
      mockStmt.all.mockReturnValue(largeResults);
      
      const results = await persistence.listMemory('*');
      expect(results.length).toBe(10000);
    });
  });

  describe('Neural Model Edge Cases', () => {
    beforeEach(async () => {
      await persistence.initialize();
    });

    it('should handle invalid model data', async () => {
      await expect(persistence.saveNeuralModel('agent1', 'model1', null))
        .rejects.toThrow();
      
      await expect(persistence.saveNeuralModel('agent1', 'model1', 'not-an-object'))
        .rejects.toThrow();
    });

    it('should handle model name conflicts', async () => {
      const modelData = { weights: [1, 2, 3] };
      
      mockStmt.run.mockReturnValue({ changes: 1 });
      
      // Save first model
      await persistence.saveNeuralModel('agent1', 'model1', modelData);
      
      // Try to save with same name - should update
      const updatedData = { weights: [4, 5, 6] };
      await persistence.saveNeuralModel('agent1', 'model1', updatedData);
      
      expect(mockStmt.run).toHaveBeenCalledTimes(2);
    });

    it('should handle large neural models', async () => {
      const largeModel = {
        weights: new Array(1000000).fill(0).map(() => Math.random()),
        biases: new Array(10000).fill(0).map(() => Math.random()),
        architecture: {
          layers: Array(100).fill({ neurons: 1000 }),
        },
      };

      mockStmt.run.mockReturnValue({ changes: 1 });
      
      const result = await persistence.saveNeuralModel('agent1', 'large', largeModel);
      expect(result).toBe(true);
    });
  });

  describe('Training Data Edge Cases', () => {
    beforeEach(async () => {
      await persistence.initialize();
    });

    it('should handle empty training data', async () => {
      await expect(persistence.saveTrainingData('session1', []))
        .rejects.toThrow();
    });

    it('should handle malformed training data', async () => {
      const malformedData = [
        { input: [1, 2], output: [1] },
        { input: [1], output: [1, 2] }, // Mismatched dimensions
        { input: null, output: [1] },
        { output: [1] }, // Missing input
      ];

      mockStmt.run.mockReturnValue({ changes: 1 });
      
      // Should filter out invalid entries
      const result = await persistence.saveTrainingData('session1', malformedData);
      expect(result).toBe(true);
    });

    it('should handle training data with NaN or Infinity', async () => {
      const problematicData = [
        { input: [1, NaN, 3], output: [1] },
        { input: [1, 2, 3], output: [Infinity] },
        { input: [-Infinity, 2, 3], output: [1] },
      ];

      mockStmt.run.mockReturnValue({ changes: 1 });
      
      const result = await persistence.saveTrainingData('session1', problematicData);
      expect(result).toBe(true);
    });
  });

  describe('Database Maintenance Edge Cases', () => {
    beforeEach(async () => {
      await persistence.initialize();
    });

    it('should handle cleanup during active operations', async () => {
      // Simulate active operations
      const storePromises = Array(10).fill(null).map((_, i) => 
        persistence.storeMemory(`active-${i}`, { data: i })
      );

      // Run cleanup while operations are in progress
      const cleanupPromise = persistence.cleanup();

      await Promise.all([...storePromises, cleanupPromise]);
      
      // Should complete without errors
      expect(mockDb.exec).toHaveBeenCalledWith('VACUUM');
    });

    it('should handle database close with pending operations', async () => {
      const pendingOps = [];
      
      // Create operations that will be pending
      for (let i = 0; i < 5; i++) {
        pendingOps.push(
          new Promise(resolve => {
            setTimeout(() => {
              persistence.storeMemory(`pending-${i}`, { data: i })
                .then(resolve);
            }, 100);
          })
        );
      }

      // Close immediately
      await persistence.close();
      
      // Pending operations should handle gracefully
      await expect(Promise.all(pendingOps)).rejects.toThrow();
    });

    it('should handle VACUUM failure', async () => {
      mockDb.exec.mockImplementation((sql) => {
        if (sql === 'VACUUM') {
          throw new Error('database or disk is full');
        }
      });

      await expect(persistence.cleanup()).rejects.toThrow('full');
    });
  });

  describe('Transaction Edge Cases', () => {
    beforeEach(async () => {
      await persistence.initialize();
    });

    it('should rollback on transaction failure', async () => {
      const rollbackFn = jest.fn();
      mockDb.transaction.mockImplementation((fn) => {
        try {
          return fn();
        } catch (e) {
          rollbackFn();
          throw e;
        }
      });

      mockStmt.run.mockImplementation(() => {
        throw new Error('constraint violation');
      });

      await expect(persistence.storeMemory('key', { data: 'test' }))
        .rejects.toThrow('constraint violation');
      
      expect(rollbackFn).toHaveBeenCalled();
    });

    it('should handle nested transactions', async () => {
      let transactionDepth = 0;
      mockDb.transaction.mockImplementation((fn) => {
        transactionDepth++;
        const result = fn();
        transactionDepth--;
        return result;
      });

      mockStmt.run.mockReturnValue({ changes: 1 });
      
      // Simulate operations that might create nested transactions
      await Promise.all([
        persistence.storeMemory('key1', { data: 1 }),
        persistence.storeMemory('key2', { data: 2 }),
        persistence.storeMemory('key3', { data: 3 }),
      ]);

      expect(transactionDepth).toBe(0);
    });
  });

  describe('Resource Limits', () => {
    beforeEach(async () => {
      await persistence.initialize();
    });

    it('should handle memory pressure', async () => {
      // Simulate low memory by storing many large objects
      const promises = [];
      
      for (let i = 0; i < 100; i++) {
        const largeObject = {
          data: new Array(10000).fill('x'.repeat(100)),
          index: i,
        };
        promises.push(persistence.storeMemory(`mem-${i}`, largeObject));
      }

      mockStmt.run.mockReturnValue({ changes: 1 });
      
      // Should handle without crashing
      const results = await Promise.allSettled(promises);
      const successful = results.filter(r => r.status === 'fulfilled').length;
      expect(successful).toBeGreaterThan(0);
    });

    it('should enforce reasonable limits on batch operations', async () => {
      const hugeBatch = Array(10000).fill(null).map((_, i) => ({
        key: `batch-${i}`,
        value: { data: i },
      }));

      // Should process in chunks rather than all at once
      const results = [];
      for (let i = 0; i < hugeBatch.length; i += 100) {
        const chunk = hugeBatch.slice(i, i + 100);
        const chunkResults = await Promise.all(
          chunk.map(item => persistence.storeMemory(item.key, item.value))
        );
        results.push(...chunkResults);
      }

      expect(results.length).toBe(10000);
    });
  });
});