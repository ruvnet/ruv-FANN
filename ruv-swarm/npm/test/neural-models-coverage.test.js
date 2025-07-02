/**
 * Neural Models Coverage Tests
 * Tests all neural model implementations for 100% coverage
 */

import assert from 'assert';
import {
  AutoencoderModel,
  CNNModel,
  GNNModel,
  GRUModel,
  ResNetModel,
  TransformerModel,
  BaseNeuralModel,
} from '../src/neural-models/index.js';

describe('Neural Models 100% Coverage', () => {
  describe('BaseNeuralModel', () => {
    it('should handle abstract method calls', () => {
      const base = new BaseNeuralModel();

      assert.throws(
        () => base.forward(),
        /Not implemented/,
      );

      assert.throws(
        () => base.backward(),
        /Not implemented/,
      );
    });

    it('should handle parameter initialization edge cases', () => {
      const base = new BaseNeuralModel();

      // Test with invalid dimensions
      assert.throws(
        () => base.initializeParameters({ dimensions: -1 }),
        /Invalid dimensions/,
      );

      // Test with null config
      assert.throws(
        () => base.initializeParameters(null),
        /Configuration required/,
      );
    });

    it('should handle serialization of complex states', () => {
      const base = new BaseNeuralModel();
      base._state = {
        weights: new Float32Array([1, 2, 3]),
        metadata: { version: 1 },
        training: { epochs: 100 },
      };

      const serialized = base.serialize();
      const deserialized = BaseNeuralModel.deserialize(serialized);

      assert.deepEqual(
        Array.from(deserialized._state.weights),
        [1, 2, 3],
      );
    });
  });

  describe('AutoencoderModel', () => {
    let autoencoder;

    beforeEach(() => {
      autoencoder = new AutoencoderModel({
        inputDim: 784,
        hiddenDim: 128,
        latentDim: 32,
      });
    });

    it('should handle encoding with invalid inputs', async() => {
      await assert.rejects(
        autoencoder.encode(null),
        /Invalid input/,
      );

      await assert.rejects(
        autoencoder.encode([[1, 2]]), // Wrong dimension
        /Dimension mismatch/,
      );
    });

    it('should handle decoding with invalid latent vectors', async() => {
      await assert.rejects(
        autoencoder.decode(null),
        /Invalid latent vector/,
      );

      await assert.rejects(
        autoencoder.decode([[1, 2]]), // Wrong latent dimension
        /Latent dimension mismatch/,
      );
    });

    it('should handle training with corrupted data', async() => {
      const corruptedData = [
        { input: null, target: [1, 2, 3] },
        { input: [1, 2, 3], target: null },
      ];

      await assert.rejects(
        autoencoder.train(corruptedData),
        /Invalid training data/,
      );
    });

    it('should handle regularization edge cases', async() => {
      autoencoder.setRegularization({
        l1: -0.1, // Invalid negative regularization
        l2: 0.01,
      });

      await assert.rejects(
        autoencoder.train([]),
        /Invalid regularization/,
      );
    });
  });

  describe('CNNModel', () => {
    let cnn;

    beforeEach(() => {
      cnn = new CNNModel({
        inputChannels: 3,
        outputClasses: 10,
        kernelSizes: [3, 5, 7],
      });
    });

    it('should handle invalid image dimensions', async() => {
      await assert.rejects(
        cnn.forward({ image: null }),
        /Invalid image input/,
      );

      await assert.rejects(
        cnn.forward({
          image: new Array(3).fill(new Array(32).fill(0)), // Missing dimension
        }),
        /Invalid image dimensions/,
      );
    });

    it('should handle pooling with invalid parameters', async() => {
      cnn._poolingSize = -1;

      await assert.rejects(
        cnn.forward({ image: new Array(3).fill(new Array(32).fill(new Array(32).fill(0))) }),
        /Invalid pooling size/,
      );
    });

    it('should handle batch normalization edge cases', async() => {
      cnn.enableBatchNorm(true);

      // Single sample (batch size 1) should handle differently
      const result = await cnn.forward({
        image: [new Array(3).fill(new Array(32).fill(new Array(32).fill(0)))],
      });

      assert(result, 'Should handle batch size 1');
    });

    it('should handle dropout during inference', async() => {
      cnn.setDropout(0.5);
      cnn.eval(); // Set to evaluation mode

      const result1 = await cnn.forward({
        image: new Array(3).fill(new Array(32).fill(new Array(32).fill(1))),
      });

      const result2 = await cnn.forward({
        image: new Array(3).fill(new Array(32).fill(new Array(32).fill(1))),
      });

      // Results should be identical in eval mode
      assert.deepEqual(result1, result2, 'Dropout should be disabled in eval mode');
    });
  });

  describe('GNNModel', () => {
    let gnn;

    beforeEach(() => {
      gnn = new GNNModel({
        nodeDim: 64,
        edgeDim: 32,
        hiddenDim: 128,
      });
    });

    it('should handle invalid graph structures', async() => {
      await assert.rejects(
        gnn.forward({ nodes: null, edges: [[0, 1]] }),
        /Invalid nodes/,
      );

      await assert.rejects(
        gnn.forward({
          nodes: [[1, 2, 3]],
          edges: null,
        }),
        /Invalid edges/,
      );
    });

    it('should handle disconnected graphs', async() => {
      const result = await gnn.forward({
        nodes: [[1, 2], [3, 4], [5, 6]],
        edges: [[0, 1]], // Node 2 is disconnected
      });

      assert(result, 'Should handle disconnected nodes');
    });

    it('should handle self-loops in graphs', async() => {
      const result = await gnn.forward({
        nodes: [[1, 2], [3, 4]],
        edges: [[0, 0], [0, 1], [1, 1]], // Self-loops
      });

      assert(result, 'Should handle self-loops');
    });

    it('should handle message passing failures', async() => {
      gnn._messagePassingEnabled = false;

      await assert.rejects(
        gnn.forward({
          nodes: [[1, 2], [3, 4]],
          edges: [[0, 1]],
        }),
        /Message passing disabled/,
      );
    });
  });

  describe('GRUModel', () => {
    let gru;

    beforeEach(() => {
      gru = new GRUModel({
        inputSize: 100,
        hiddenSize: 256,
        numLayers: 2,
      });
    });

    it('should handle sequence length mismatches', async() => {
      await assert.rejects(
        gru.forward({
          sequence: [[1, 2, 3]], // Wrong input size
          lengths: [10], // Mismatched length
        }),
        /Sequence length mismatch/,
      );
    });

    it('should handle bidirectional processing edge cases', async() => {
      gru.setBidirectional(true);

      const result = await gru.forward({
        sequence: [new Array(100).fill(0)],
        lengths: [1], // Single timestep
      });

      assert(result, 'Should handle single timestep in bidirectional mode');
    });

    it('should handle gradient clipping edge cases', async() => {
      gru.setGradientClipping(-1); // Invalid negative clipping

      await assert.rejects(
        gru.backward({ gradOutput: [[1, 2, 3]] }),
        /Invalid gradient clipping/,
      );
    });

    it('should handle hidden state reset', async() => {
      // Process a sequence
      await gru.forward({
        sequence: [new Array(100).fill(1)],
        lengths: [1],
      });

      // Reset hidden state
      gru.resetHiddenState();

      // Process another sequence
      const result = await gru.forward({
        sequence: [new Array(100).fill(2)],
        lengths: [1],
      });

      assert(result, 'Should process after hidden state reset');
    });
  });

  describe('ResNetModel', () => {
    let resnet;

    beforeEach(() => {
      resnet = new ResNetModel({
        numClasses: 1000,
        layers: [3, 4, 6, 3], // ResNet-50 configuration
        inputChannels: 3,
      });
    });

    it('should handle skip connection failures', async() => {
      resnet._skipConnections = false;

      const result = await resnet.forward({
        image: new Array(3).fill(new Array(224).fill(new Array(224).fill(0))),
      });

      assert(result, 'Should work without skip connections (plain network)');
    });

    it('should handle identity mapping edge cases', async() => {
      // Test with very deep network
      const deepResnet = new ResNetModel({
        numClasses: 10,
        layers: [10, 10, 10, 10], // Very deep
        inputChannels: 1,
      });

      const result = await deepResnet.forward({
        image: new Array(1).fill(new Array(32).fill(new Array(32).fill(1))),
      });

      assert(result, 'Should handle very deep architectures');
    });

    it('should handle bottleneck architecture edge cases', async() => {
      resnet.useBottleneck(true);

      await assert.rejects(
        resnet.forward({
          image: new Array(2).fill(new Array(224).fill(new Array(224).fill(0))), // Wrong channels
        }),
        /Channel dimension mismatch/,
      );
    });
  });

  describe('TransformerModel', () => {
    let transformer;

    beforeEach(() => {
      transformer = new TransformerModel({
        dModel: 512,
        nHeads: 8,
        nLayers: 6,
        vocabSize: 10000,
      });
    });

    it('should handle attention mask edge cases', async() => {
      const result = await transformer.forward({
        input: [[1, 2, 3, 4]],
        mask: null, // No mask (full attention)
      });

      assert(result, 'Should work without attention mask');
    });

    it('should handle padding mask edge cases', async() => {
      await assert.rejects(
        transformer.forward({
          input: [[1, 2, 0, 0]], // Padded sequence
          paddingMask: [[1, 1, 1, 1]], // Wrong mask (no padding indicated)
        }),
        /Padding mask mismatch/,
      );
    });

    it('should handle position encoding overflow', async() => {
      // Very long sequence
      const longSequence = new Array(5000).fill(1);

      await assert.rejects(
        transformer.forward({
          input: [longSequence],
        }),
        /Sequence too long/,
      );
    });

    it('should handle multi-head attention failures', async() => {
      transformer._heads[0] = null; // Corrupt one attention head

      await assert.rejects(
        transformer.forward({
          input: [[1, 2, 3, 4]],
        }),
        /Attention head failure/,
      );
    });

    it('should handle layer normalization edge cases', async() => {
      transformer.disableLayerNorm();

      const result = await transformer.forward({
        input: [[1, 2, 3, 4]],
      });

      assert(result, 'Should work without layer normalization');
    });
  });

  describe('Model Ensemble Edge Cases', () => {
    it('should handle ensemble with mixed model types', async() => {
      const models = [
        new CNNModel({ inputChannels: 3, outputClasses: 10 }),
        new GRUModel({ inputSize: 100, hiddenSize: 128 }),
        new TransformerModel({ dModel: 256, nHeads: 4, vocabSize: 1000 }),
      ];

      // Ensemble should handle incompatible models
      const ensemble = { models, vote: 'majority' };

      await assert.rejects(
        ensemblePredict(ensemble, { data: [1, 2, 3] }),
        /Incompatible model types/,
      );
    });
  });
});

// Helper function for ensemble prediction
async function ensemblePredict(ensemble, input) {
  const predictions = await Promise.all(
    ensemble.models.map(model => model.forward(input)),
  );

  if (!predictions.every(p => p.length === predictions[0].length)) {
    throw new Error('Incompatible model types in ensemble');
  }

  return predictions;
}

// Run tests when executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  console.log('Running neural models coverage tests...');

  // Run all tests
  const { run } = await import('./test-runner.js');
  await run(__filename);
}