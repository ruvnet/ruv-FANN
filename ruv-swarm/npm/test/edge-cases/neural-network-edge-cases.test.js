/**
 * Neural Network Edge Cases
 * Tests numerical stability, gradient issues, and model training edge cases
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { NeuralNetworkManager } from '../../src/neural-network-manager.js';
import { NeuralAgent } from '../../src/neural-agent.js';

describe('Neural Network Edge Cases', () => {
  let manager;

  beforeEach(() => {
    manager = new NeuralNetworkManager();
  });

  describe('Numerical Stability Edge Cases', () => {
    it('should handle NaN inputs gracefully', async() => {
      const config = {
        type: 'feedforward',
        layers: [4, 8, 4],
        activation: 'relu',
      };

      const network = await manager.create(config);

      const invalidInputs = [
        [NaN, 1, 2, 3],
        [1, NaN, 2, 3],
        [1, 2, NaN, 3],
        [1, 2, 3, NaN],
        [NaN, NaN, NaN, NaN],
      ];

      for (const input of invalidInputs) {
        const result = await network.forward(input);

        // Network should handle NaN inputs without crashing
        expect(result).toBeDefined();
        expect(Array.isArray(result)).toBe(true);

        // Output should either be valid numbers or NaN (consistent behavior)
        result.forEach(value => {
          expect(typeof value).toBe('number');
        });
      }
    });

    it('should handle Infinity inputs', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 4, 2],
        activation: 'tanh',
      };

      const network = await manager.create(config);

      const infinityInputs = [
        [Infinity, 1],
        [1, Infinity],
        [-Infinity, 1],
        [1, -Infinity],
        [Infinity, -Infinity],
        [Infinity, Infinity],
        [-Infinity, -Infinity],
      ];

      for (const input of infinityInputs) {
        const result = await network.forward(input);

        expect(result).toBeDefined();
        expect(Array.isArray(result)).toBe(true);
        expect(result.length).toBe(2);

        // tanh should clamp infinite values to -1 or 1
        result.forEach(value => {
          expect(Math.abs(value)).toBeLessThanOrEqual(1);
        });
      }
    });

    it('should handle very small numbers (underflow)', async() => {
      const config = {
        type: 'feedforward',
        layers: [3, 5, 3],
        activation: 'sigmoid',
      };

      const network = await manager.create(config);

      const smallNumbers = [
        [Number.MIN_VALUE, 1e-100, 1e-308],
        [1e-323, 1e-324, 0], // Near machine epsilon
        [-Number.MIN_VALUE, -1e-100, -1e-308],
      ];

      for (const input of smallNumbers) {
        const result = await network.forward(input);

        expect(result).toBeDefined();
        expect(result.length).toBe(3);

        // Should handle underflow gracefully
        result.forEach(value => {
          expect(isFinite(value)).toBe(true);
          expect(value).toBeGreaterThanOrEqual(0);
          expect(value).toBeLessThanOrEqual(1);
        });
      }
    });

    it('should handle very large numbers (overflow)', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 3, 2],
        activation: 'relu',
      };

      const network = await manager.create(config);

      const largeNumbers = [
        [Number.MAX_VALUE, 1e100],
        [1e308, 1e307],
        [-Number.MAX_VALUE, -1e100],
      ];

      for (const input of largeNumbers) {
        const result = await network.forward(input);

        expect(result).toBeDefined();
        expect(result.length).toBe(2);

        // ReLU should pass through positive values or zero
        result.forEach(value => {
          expect(value).toBeGreaterThanOrEqual(0);
          expect(isFinite(value) || value === Infinity).toBe(true);
        });
      }
    });
  });

  describe('Gradient Edge Cases', () => {
    it('should handle vanishing gradients', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 10, 10, 10, 10, 10, 2], // Deep network
        activation: 'sigmoid', // Prone to vanishing gradients
      };

      const network = await manager.create(config);

      const trainingData = [];
      for (let i = 0; i < 100; i++) {
        trainingData.push({
          input: [Math.random(), Math.random()],
          target: [Math.random(), Math.random()],
        });
      }

      const initialWeights = network.getWeights();

      // Train for a few epochs
      for (let epoch = 0; epoch < 10; epoch++) {
        let totalLoss = 0;

        for (const sample of trainingData) {
          const output = await network.forward(sample.input);
          const loss = await network.backward(sample.target);
          totalLoss += loss;
        }

        const avgLoss = totalLoss / trainingData.length;
        expect(avgLoss).toBeGreaterThanOrEqual(0);
        expect(isFinite(avgLoss)).toBe(true);
      }

      const finalWeights = network.getWeights();

      // Check that some learning occurred (weights changed)
      let weightsChanged = false;
      for (let i = 0; i < initialWeights.length; i++) {
        if (Math.abs(initialWeights[i] - finalWeights[i]) > 1e-6) {
          weightsChanged = true;
          break;
        }
      }

      // Even with vanishing gradients, some change should occur
      expect(weightsChanged).toBe(true);
    });

    it('should handle exploding gradients', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 5, 2],
        activation: 'linear',
        learningRate: 10.0, // High learning rate
      };

      const network = await manager.create(config);

      // Initialize with large weights to cause exploding gradients
      const weights = network.getWeights();
      for (let i = 0; i < weights.length; i++) {
        weights[i] = (Math.random() - 0.5) * 10; // Large initial weights
      }
      network.setWeights(weights);

      const trainingData = {
        input: [1, 1],
        target: [0, 0],
      };

      let maxGradient = 0;
      const gradientHistory = [];

      for (let epoch = 0; epoch < 20; epoch++) {
        try {
          await network.forward(trainingData.input);
          const loss = await network.backward(trainingData.target);

          // Track gradient magnitude
          const currentWeights = network.getWeights();
          const gradientMagnitude = Math.sqrt(
            currentWeights.reduce((sum, w) => sum + w * w, 0),
          );

          gradientHistory.push(gradientMagnitude);
          maxGradient = Math.max(maxGradient, gradientMagnitude);

          // Network should handle large gradients without crashing
          expect(isFinite(loss)).toBe(true);
          expect(isFinite(gradientMagnitude)).toBe(true);

        } catch (error) {
          // If training fails due to numerical issues, that's acceptable
          expect(error.message).toMatch(/numerical|overflow|gradient/i);
          break;
        }
      }

      // Should have detected large gradients
      expect(maxGradient).toBeGreaterThan(1);
    });
  });

  describe('Model Architecture Edge Cases', () => {
    it('should handle single neuron networks', async() => {
      const config = {
        type: 'feedforward',
        layers: [1, 1],
        activation: 'sigmoid',
      };

      const network = await manager.create(config);

      const result = await network.forward([0.5]);
      expect(result).toHaveLength(1);
      expect(result[0]).toBeGreaterThan(0);
      expect(result[0]).toBeLessThan(1);
    });

    it('should handle networks with skip connections', async() => {
      const config = {
        type: 'residual',
        layers: [4, 8, 8, 4],
        activation: 'relu',
        skipConnections: true,
      };

      const network = await manager.create(config);

      const input = [1, 2, 3, 4];
      const result = await network.forward(input);

      expect(result).toHaveLength(4);
      result.forEach(value => {
        expect(isFinite(value)).toBe(true);
      });
    });

    it('should handle extremely wide networks', async() => {
      const config = {
        type: 'feedforward',
        layers: [10, 1000, 10], // Very wide hidden layer
        activation: 'relu',
      };

      const network = await manager.create(config);

      const input = new Array(10).fill(0).map(() => Math.random());
      const result = await network.forward(input);

      expect(result).toHaveLength(10);
      result.forEach(value => {
        expect(isFinite(value)).toBe(true);
      });
    });

    it('should handle extremely deep networks', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2], // Very deep
        activation: 'relu',
      };

      const network = await manager.create(config);

      const input = [0.5, 0.5];
      const result = await network.forward(input);

      expect(result).toHaveLength(2);
      result.forEach(value => {
        expect(isFinite(value)).toBe(true);
      });
    });
  });

  describe('Training Data Edge Cases', () => {
    it('should handle empty training data', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 3, 2],
        activation: 'sigmoid',
      };

      const network = await manager.create(config);

      await expect(network.train([])).rejects.toThrow(/empty|no data/i);
    });

    it('should handle inconsistent input dimensions', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 3, 2],
        activation: 'sigmoid',
      };

      const network = await manager.create(config);

      const inconsistentData = [
        { input: [1, 2], target: [0, 1] },
        { input: [1, 2, 3], target: [0, 1] }, // Wrong input size
        { input: [1], target: [0, 1] }, // Wrong input size
      ];

      await expect(network.train(inconsistentData)).rejects.toThrow(/dimension|size/i);
    });

    it('should handle inconsistent output dimensions', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 3, 2],
        activation: 'sigmoid',
      };

      const network = await manager.create(config);

      const inconsistentData = [
        { input: [1, 2], target: [0, 1] },
        { input: [1, 2], target: [0, 1, 2] }, // Wrong target size
        { input: [1, 2], target: [0] }, // Wrong target size
      ];

      await expect(network.train(inconsistentData)).rejects.toThrow(/dimension|size/i);
    });

    it('should handle duplicate training samples', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 3, 2],
        activation: 'sigmoid',
      };

      const network = await manager.create(config);

      const duplicateData = [];
      for (let i = 0; i < 100; i++) {
        duplicateData.push({ input: [1, 2], target: [0, 1] });
      }

      const result = await network.train(duplicateData, { epochs: 10 });

      // Should converge quickly on duplicate data
      expect(result.finalLoss).toBeLessThan(result.initialLoss);
    });
  });

  describe('Activation Function Edge Cases', () => {
    it('should handle unknown activation functions', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 3, 2],
        activation: 'unknown_activation',
      };

      await expect(manager.create(config)).rejects.toThrow(/activation|unknown/i);
    });

    it('should handle custom activation functions', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 3, 2],
        activation: (x) => Math.max(0, Math.min(1, x)), // Custom clamp function
      };

      const network = await manager.create(config);

      const result = await network.forward([0.5, -0.5]);

      expect(result).toHaveLength(2);
      result.forEach(value => {
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThanOrEqual(1);
      });
    });

    it('should handle activation functions with extreme outputs', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 3, 2],
        activation: (x) => x > 0 ? 1e10 : -1e10, // Extreme activation
      };

      const network = await manager.create(config);

      const result = await network.forward([1, -1]);

      expect(result).toHaveLength(2);
      result.forEach(value => {
        expect(Math.abs(value)).toBeGreaterThan(1000);
      });
    });
  });

  describe('Batch Processing Edge Cases', () => {
    it('should handle single sample batches', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 3, 2],
        activation: 'relu',
      };

      const network = await manager.create(config);

      const singleBatch = [{ input: [1, 2], target: [0, 1] }];
      const result = await network.trainBatch(singleBatch);

      expect(result.loss).toBeGreaterThanOrEqual(0);
      expect(isFinite(result.loss)).toBe(true);
    });

    it('should handle very large batches', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 3, 2],
        activation: 'relu',
      };

      const network = await manager.create(config);

      const largeBatch = [];
      for (let i = 0; i < 10000; i++) {
        largeBatch.push({
          input: [Math.random(), Math.random()],
          target: [Math.random(), Math.random()],
        });
      }

      const result = await network.trainBatch(largeBatch);

      expect(result.loss).toBeGreaterThanOrEqual(0);
      expect(isFinite(result.loss)).toBe(true);
    });

    it('should handle batches with mixed sample qualities', async() => {
      const config = {
        type: 'feedforward',
        layers: [2, 3, 2],
        activation: 'sigmoid',
      };

      const network = await manager.create(config);

      const mixedBatch = [
        { input: [1, 2], target: [0, 1] }, // Normal
        { input: [0, 0], target: [0, 0] }, // Zero input
        { input: [1000, -1000], target: [1, 0] }, // Extreme input
        { input: [0.1, 0.1], target: [0.9, 0.9] }, // Small input, large target
      ];

      const result = await network.trainBatch(mixedBatch);

      expect(result.loss).toBeGreaterThanOrEqual(0);
      expect(isFinite(result.loss)).toBe(true);
    });
  });

  describe('Memory Management in Neural Networks', () => {
    it('should handle memory cleanup during training', async() => {
      const config = {
        type: 'feedforward',
        layers: [10, 100, 100, 10],
        activation: 'relu',
      };

      const network = await manager.create(config);

      const trainingData = [];
      for (let i = 0; i < 1000; i++) {
        trainingData.push({
          input: new Array(10).fill(0).map(() => Math.random()),
          target: new Array(10).fill(0).map(() => Math.random()),
        });
      }

      const initialMemory = process.memoryUsage();

      // Train for multiple epochs
      for (let epoch = 0; epoch < 50; epoch++) {
        await network.train(trainingData, { epochs: 1 });

        // Force garbage collection periodically
        if (epoch % 10 === 0 && global.gc) {
          global.gc();
        }
      }

      const finalMemory = process.memoryUsage();

      // Memory usage should not grow unboundedly
      const memoryGrowth = finalMemory.heapUsed - initialMemory.heapUsed;
      expect(memoryGrowth).toBeLessThan(100 * 1024 * 1024); // Less than 100MB growth
    });
  });
});

// Run tests when executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  console.log('Running neural network edge case tests...');

  // Run all tests
  const { run } = await import('../test-runner.js');
  await run(__filename);
}