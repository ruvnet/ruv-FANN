/**
 * Comprehensive test suite for neural network manager
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';

// Mock the persistence manager
jest.mock('../src/persistence.js', () => ({
  PersistenceManager: jest.fn().mockImplementation(() => ({
    initialize: jest.fn().mockResolvedValue(true),
    saveNeuralModel: jest.fn().mockResolvedValue(true),
    loadNeuralModel: jest.fn().mockResolvedValue(null),
    saveTrainingData: jest.fn().mockResolvedValue(true),
    getTrainingHistory: jest.fn().mockResolvedValue([]),
  })),
}));

// Import after mocking
import { NeuralNetworkManager } from '../src/neural-network-manager.js';

describe('NeuralNetworkManager Comprehensive Tests', () => {
  let manager;
  let mockPersistence;

  beforeEach(() => {
    manager = new NeuralNetworkManager();
    mockPersistence = manager.persistence;
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Network Creation and Configuration', () => {
    it('should create network with default configuration', () => {
      const network = manager.createNetwork('test-agent');

      expect(network).toBeDefined();
      expect(network.id).toBe('test-agent');
      expect(network.layers).toHaveLength(3);
      expect(network.layers[0].neurons).toBe(10);
      expect(network.layers[1].neurons).toBe(20);
      expect(network.layers[2].neurons).toBe(5);
    });

    it('should create network with custom configuration', () => {
      const config = {
        layers: [8, 16, 32, 4],
        activation: 'relu',
        learningRate: 0.001,
        momentum: 0.8,
        dropout: 0.2,
      };

      const network = manager.createNetwork('custom-agent', config);

      expect(network.layers).toHaveLength(4);
      expect(network.config.activation).toBe('relu');
      expect(network.config.learningRate).toBe(0.001);
      expect(network.config.momentum).toBe(0.8);
      expect(network.config.dropout).toBe(0.2);
    });

    it('should handle invalid layer configurations', () => {
      const invalidConfigs = [
        { layers: [] },
        { layers: [10] },
        { layers: [0, 10, 5] },
        { layers: [-5, 10, 5] },
        { layers: [10, 0, 5] },
      ];

      for (const config of invalidConfigs) {
        expect(() => manager.createNetwork('invalid', config)).toThrow();
      }
    });

    it('should validate activation functions', () => {
      const validActivations = ['sigmoid', 'tanh', 'relu', 'leaky_relu'];

      for (const activation of validActivations) {
        const network = manager.createNetwork(`test-${activation}`, { activation });
        expect(network.config.activation).toBe(activation);
      }

      expect(() =>
        manager.createNetwork('invalid', { activation: 'unknown' }),
      ).toThrow();
    });
  });

  describe('Weight Initialization', () => {
    it('should initialize weights with Xavier method', () => {
      const network = manager.createNetwork('xavier-test');

      // Check weight distribution
      for (const layer of network.layers) {
        if (layer.weights) {
          const weights = layer.weights.flat();
          const mean = weights.reduce((a, b) => a + b) / weights.length;
          const variance = weights.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / weights.length;

          expect(Math.abs(mean)).toBeLessThan(0.1);
          expect(variance).toBeGreaterThan(0);
          expect(variance).toBeLessThan(1);
        }
      }
    });

    it('should initialize biases to small values', () => {
      const network = manager.createNetwork('bias-test');

      for (const layer of network.layers) {
        if (layer.biases) {
          for (const bias of layer.biases) {
            expect(Math.abs(bias)).toBeLessThan(0.1);
          }
        }
      }
    });
  });

  describe('Forward Propagation', () => {
    beforeEach(() => {
      manager.createNetwork('forward-test', {
        layers: [3, 4, 2],
        activation: 'sigmoid',
      });
    });

    it('should perform forward pass correctly', () => {
      const input = [0.5, 0.3, 0.8];
      const output = manager.forward('forward-test', input);

      expect(output).toHaveLength(2);
      expect(output.every(v => v >= 0 && v <= 1)).toBe(true);
    });

    it('should handle batch forward propagation', () => {
      const batch = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
      ];

      for (const input of batch) {
        const output = manager.forward('forward-test', input);
        expect(output).toHaveLength(2);
      }
    });

    it('should apply dropout during training', () => {
      manager.createNetwork('dropout-test', {
        layers: [10, 20, 10],
        dropout: 0.5,
      });

      const input = new Array(10).fill(0.5);

      // Run multiple times to check dropout randomness
      const outputs = [];
      for (let i = 0; i < 10; i++) {
        manager.networks.get('dropout-test').training = true;
        outputs.push(manager.forward('dropout-test', input));
      }

      // Outputs should vary due to dropout
      const allSame = outputs.every(o =>
        JSON.stringify(o) === JSON.stringify(outputs[0]),
      );
      expect(allSame).toBe(false);
    });

    it('should not apply dropout during inference', () => {
      manager.createNetwork('inference-test', {
        layers: [5, 10, 3],
        dropout: 0.5,
      });

      const input = new Array(5).fill(0.5);

      // Set to inference mode
      manager.networks.get('inference-test').training = false;

      // Run multiple times
      const outputs = [];
      for (let i = 0; i < 5; i++) {
        outputs.push(manager.forward('inference-test', input));
      }

      // Outputs should be deterministic
      const allSame = outputs.every(o =>
        JSON.stringify(o) === JSON.stringify(outputs[0]),
      );
      expect(allSame).toBe(true);
    });

    it('should throw error for non-existent network', () => {
      expect(() => manager.forward('non-existent', [1, 2, 3])).toThrow();
    });

    it('should validate input dimensions', () => {
      expect(() => manager.forward('forward-test', [1, 2])).toThrow();
      expect(() => manager.forward('forward-test', [1, 2, 3, 4])).toThrow();
    });
  });

  describe('Activation Functions', () => {
    it('should correctly apply sigmoid activation', () => {
      const sigmoid = manager.activations.sigmoid;

      expect(sigmoid(0)).toBeCloseTo(0.5);
      expect(sigmoid(100)).toBeCloseTo(1);
      expect(sigmoid(-100)).toBeCloseTo(0);
    });

    it('should correctly apply tanh activation', () => {
      const tanh = manager.activations.tanh;

      expect(tanh(0)).toBeCloseTo(0);
      expect(tanh(100)).toBeCloseTo(1);
      expect(tanh(-100)).toBeCloseTo(-1);
    });

    it('should correctly apply ReLU activation', () => {
      const relu = manager.activations.relu;

      expect(relu(5)).toBe(5);
      expect(relu(-5)).toBe(0);
      expect(relu(0)).toBe(0);
    });

    it('should correctly apply Leaky ReLU activation', () => {
      const leakyRelu = manager.activations.leaky_relu;

      expect(leakyRelu(5)).toBe(5);
      expect(leakyRelu(-5)).toBeCloseTo(-0.05);
      expect(leakyRelu(0)).toBe(0);
    });

    it('should compute derivatives correctly', () => {
      const sigmoidDeriv = manager.activationDerivatives.sigmoid;
      const tanhDeriv = manager.activationDerivatives.tanh;
      const reluDeriv = manager.activationDerivatives.relu;

      // Sigmoid derivative peaks at 0.5
      expect(sigmoidDeriv(0.5)).toBeCloseTo(0.25);

      // Tanh derivative peaks at 0
      expect(tanhDeriv(0)).toBeCloseTo(1);

      // ReLU derivative
      expect(reluDeriv(5)).toBe(1);
      expect(reluDeriv(-5)).toBe(0);
    });
  });

  describe('Training and Backpropagation', () => {
    beforeEach(() => {
      manager.createNetwork('train-test', {
        layers: [2, 3, 1],
        learningRate: 0.1,
        activation: 'sigmoid',
      });
    });

    it('should train on single sample', async() => {
      const trainingData = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ];

      const initialError = manager.calculateError('train-test', trainingData);

      await manager.train('train-test', trainingData, { epochs: 100, verbose: false });

      const finalError = manager.calculateError('train-test', trainingData);
      expect(finalError).toBeLessThan(initialError);
    });

    it('should support mini-batch training', async() => {
      const trainingData = Array(100).fill(null).map(() => ({
        input: [Math.random(), Math.random()],
        output: [Math.random()],
      }));

      await manager.train('train-test', trainingData, {
        epochs: 10,
        batchSize: 10,
        verbose: false,
      });

      // Should complete without errors
      expect(manager.networks.get('train-test').stats.epochs).toBe(10);
    });

    it('should apply momentum correctly', async() => {
      manager.createNetwork('momentum-test', {
        layers: [2, 3, 1],
        learningRate: 0.1,
        momentum: 0.9,
      });

      const trainingData = [
        { input: [0.5, 0.5], output: [0.75] },
      ];

      // Train for a few epochs to build up momentum
      await manager.train('momentum-test', trainingData, {
        epochs: 5,
        verbose: false,
      });

      const network = manager.networks.get('momentum-test');
      expect(network.velocities).toBeDefined();

      // Velocities should be non-zero
      const hasNonZeroVelocity = network.velocities.some(v =>
        v && v.weights && v.weights.some(row => row.some(w => w !== 0)),
      );
      expect(hasNonZeroVelocity).toBe(true);
    });

    it('should implement early stopping', async() => {
      const trainingData = Array(50).fill(null).map(() => ({
        input: [Math.random(), Math.random()],
        output: [Math.random()],
      }));

      const validationData = Array(10).fill(null).map(() => ({
        input: [Math.random(), Math.random()],
        output: [Math.random()],
      }));

      const result = await manager.train('train-test', trainingData, {
        epochs: 1000,
        validationData,
        earlyStopPatience: 5,
        verbose: false,
      });

      // Should stop before max epochs
      expect(result.epochs).toBeLessThan(1000);
    });

    it('should handle gradient clipping', async() => {
      manager.createNetwork('clip-test', {
        layers: [2, 3, 1],
        learningRate: 10, // Very high to cause gradient explosion
        gradientClip: 1.0,
      });

      const trainingData = [
        { input: [100, 100], output: [0.0001] }, // Extreme values
      ];

      await manager.train('clip-test', trainingData, {
        epochs: 10,
        verbose: false,
      });

      // Network should remain stable despite high learning rate
      const output = manager.forward('clip-test', [1, 1]);
      expect(output.every(v => !isNaN(v) && isFinite(v))).toBe(true);
    });
  });

  describe('Error Calculation', () => {
    beforeEach(() => {
      manager.createNetwork('error-test', {
        layers: [2, 2],
      });
    });

    it('should calculate MSE correctly', () => {
      const data = [
        { input: [0, 0], output: [0, 1] },
        { input: [1, 1], output: [1, 0] },
      ];

      const error = manager.calculateError('error-test', data);
      expect(error).toBeGreaterThan(0);
      expect(error).toBeLessThan(1);
    });

    it('should handle empty data', () => {
      expect(() => manager.calculateError('error-test', [])).toThrow();
    });

    it('should validate output dimensions', () => {
      const invalidData = [
        { input: [0, 0], output: [0, 1, 2] }, // Wrong output size
      ];

      expect(() => manager.calculateError('error-test', invalidData)).toThrow();
    });
  });

  describe('Model Persistence', () => {
    it('should save model correctly', async() => {
      manager.createNetwork('save-test');

      await manager.saveModel('save-test', 'test-model');

      expect(mockPersistence.saveNeuralModel).toHaveBeenCalledWith(
        'save-test',
        'test-model',
        expect.objectContaining({
          layers: expect.any(Array),
          config: expect.any(Object),
          stats: expect.any(Object),
        }),
      );
    });

    it('should load model correctly', async() => {
      const savedModel = {
        layers: [
          { neurons: 3, weights: [[1, 2], [3, 4], [5, 6]], biases: [0.1, 0.2, 0.3] },
          { neurons: 2, weights: [[7, 8, 9], [10, 11, 12]], biases: [0.4, 0.5] },
        ],
        config: { activation: 'relu', learningRate: 0.01 },
        stats: { epochs: 100 },
      };

      mockPersistence.loadNeuralModel.mockResolvedValue(savedModel);

      const loaded = await manager.loadModel('load-test', 'test-model');
      expect(loaded).toBe(true);

      const network = manager.networks.get('load-test');
      expect(network.layers).toHaveLength(2);
      expect(network.config.activation).toBe('relu');
    });

    it('should handle missing models gracefully', async() => {
      mockPersistence.loadNeuralModel.mockResolvedValue(null);

      const loaded = await manager.loadModel('missing', 'non-existent');
      expect(loaded).toBe(false);
    });
  });

  describe('Advanced Features', () => {
    it('should support learning rate scheduling', async() => {
      manager.createNetwork('lr-schedule', {
        layers: [2, 3, 1],
        learningRate: 0.1,
        learningRateDecay: 0.95,
      });

      const trainingData = [
        { input: [0, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ];

      const network = manager.networks.get('lr-schedule');
      const initialLR = network.config.learningRate;

      await manager.train('lr-schedule', trainingData, {
        epochs: 10,
        verbose: false,
      });

      expect(network.config.learningRate).toBeLessThan(initialLR);
      expect(network.config.learningRate).toBeCloseTo(
        initialLR * Math.pow(0.95, 10),
      );
    });

    it('should support batch normalization', () => {
      manager.createNetwork('batch-norm', {
        layers: [10, 20, 10],
        batchNorm: true,
      });

      const network = manager.networks.get('batch-norm');

      // Check that batch norm parameters are initialized
      for (let i = 1; i < network.layers.length; i++) {
        const layer = network.layers[i];
        expect(layer.batchNorm).toBeDefined();
        expect(layer.batchNorm.gamma).toHaveLength(layer.neurons);
        expect(layer.batchNorm.beta).toHaveLength(layer.neurons);
      }
    });

    it('should handle network ensemble predictions', () => {
      // Create multiple networks
      for (let i = 0; i < 3; i++) {
        manager.createNetwork(`ensemble-${i}`, {
          layers: [2, 3, 1],
        });
      }

      const input = [0.5, 0.5];
      const predictions = [];

      for (let i = 0; i < 3; i++) {
        predictions.push(manager.forward(`ensemble-${i}`, input)[0]);
      }

      // Calculate ensemble average
      const ensemblePrediction = predictions.reduce((a, b) => a + b) / predictions.length;

      expect(ensemblePrediction).toBeGreaterThan(0);
      expect(ensemblePrediction).toBeLessThan(1);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle NaN values in forward propagation', () => {
      manager.createNetwork('nan-test');

      const input = [NaN, 1, 2];
      expect(() => manager.forward('nan-test', input)).toThrow();
    });

    it('should handle Infinity values in training', async() => {
      manager.createNetwork('inf-test', {
        layers: [2, 2],
      });

      const trainingData = [
        { input: [Infinity, 1], output: [0, 1] },
      ];

      await expect(manager.train('inf-test', trainingData)).rejects.toThrow();
    });

    it('should recover from numerical instability', async() => {
      manager.createNetwork('unstable', {
        layers: [2, 10, 1],
        learningRate: 100, // Extremely high
      });

      const trainingData = [
        { input: [1000, 1000], output: [0.0001] },
      ];

      // Should handle without crashing
      try {
        await manager.train('unstable', trainingData, {
          epochs: 5,
          verbose: false,
        });
      } catch (e) {
        expect(e.message).toContain('numerical');
      }
    });

    it('should validate network integrity', () => {
      const network = manager.createNetwork('integrity-test');

      // Corrupt network structure
      network.layers[1].weights = null;

      expect(() => manager.forward('integrity-test', [1, 2, 3])).toThrow();
    });
  });

  describe('Performance and Memory', () => {
    it('should handle large networks efficiently', () => {
      const start = Date.now();

      manager.createNetwork('large-network', {
        layers: [100, 200, 200, 100, 50],
      });

      const creationTime = Date.now() - start;
      expect(creationTime).toBeLessThan(1000); // Should create in under 1 second

      // Test forward pass performance
      const input = new Array(100).fill(0.5);
      const forwardStart = Date.now();

      for (let i = 0; i < 10; i++) {
        manager.forward('large-network', input);
      }

      const forwardTime = Date.now() - forwardStart;
      expect(forwardTime).toBeLessThan(100); // 10 passes in under 100ms
    });

    it('should clean up resources properly', async() => {
      // Create and destroy multiple networks
      for (let i = 0; i < 100; i++) {
        manager.createNetwork(`temp-${i}`, {
          layers: [10, 20, 10],
        });
      }

      expect(manager.networks.size).toBe(100);

      // Clear networks
      manager.networks.clear();
      expect(manager.networks.size).toBe(0);
    });
  });
});