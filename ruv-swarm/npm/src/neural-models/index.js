/**
 * Neural Models Index
 * Exports all available neural network architectures
 */

export { NeuralModel } from './base.js';
export { TransformerModel } from './transformer.js';
export { CNNModel } from './cnn.js';
export { GRUModel } from './gru.js';
export { AutoencoderModel } from './autoencoder.js';
export { GNNModel } from './gnn.js';
export { ResNetModel } from './resnet.js';
export { VAEModel } from './vae.js';
export { LSTMModel } from './lstm.js';

// Model factory for easy instantiation
export const createNeuralModel = (type, config = {}) => {
  const models = {
    transformer: () => import('./transformer.js').then(m => new m.TransformerModel(config)),
    cnn: () => import('./cnn.js').then(m => new m.CNNModel(config)),
    gru: () => import('./gru.js').then(m => new m.GRUModel(config)),
    autoencoder: () => import('./autoencoder.js').then(m => new m.AutoencoderModel(config)),
    gnn: () => import('./gnn.js').then(m => new m.GNNModel(config)),
    resnet: () => import('./resnet.js').then(m => new m.ResNetModel(config)),
    vae: () => import('./vae.js').then(m => new m.VAEModel(config)),
    lstm: () => import('./lstm.js').then(m => new m.LSTMModel(config)),
  };

  if (!models[type]) {
    throw new Error(`Unknown neural model type: ${type}. Available types: ${Object.keys(models).join(', ')}`);
  }

  return models[type]();
};

// Model configurations presets
export const MODEL_PRESETS = {
  // Transformer presets
  transformer: {
    small: {
      dimensions: 256,
      heads: 4,
      layers: 3,
      ffDimensions: 1024,
      dropoutRate: 0.1,
    },
    base: {
      dimensions: 512,
      heads: 8,
      layers: 6,
      ffDimensions: 2048,
      dropoutRate: 0.1,
    },
    large: {
      dimensions: 1024,
      heads: 16,
      layers: 12,
      ffDimensions: 4096,
      dropoutRate: 0.1,
    },
  },

  // CNN presets
  cnn: {
    mnist: {
      inputShape: [28, 28, 1],
      convLayers: [
        { filters: 32, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' },
        { filters: 64, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' },
      ],
      denseLayers: [128],
      outputSize: 10,
      dropoutRate: 0.5,
    },
    cifar10: {
      inputShape: [32, 32, 3],
      convLayers: [
        { filters: 32, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' },
        { filters: 64, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' },
        { filters: 128, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' },
      ],
      denseLayers: [256, 128],
      outputSize: 10,
      dropoutRate: 0.5,
    },
    imagenet: {
      inputShape: [224, 224, 3],
      convLayers: [
        { filters: 64, kernelSize: 7, stride: 2, padding: 'same', activation: 'relu' },
        { filters: 128, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' },
        { filters: 256, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' },
        { filters: 512, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' },
      ],
      denseLayers: [4096, 4096],
      outputSize: 1000,
      dropoutRate: 0.5,
    },
  },

  // GRU presets
  gru: {
    text_classification: {
      inputSize: 300, // Word embedding size
      hiddenSize: 128,
      numLayers: 2,
      outputSize: 2, // Binary classification
      bidirectional: true,
      dropoutRate: 0.2,
    },
    sequence_generation: {
      inputSize: 128,
      hiddenSize: 512,
      numLayers: 3,
      outputSize: 10000, // Vocabulary size
      bidirectional: false,
      dropoutRate: 0.3,
    },
    time_series: {
      inputSize: 10, // Feature dimensions
      hiddenSize: 64,
      numLayers: 2,
      outputSize: 1, // Regression
      bidirectional: false,
      dropoutRate: 0.1,
    },
  },

  // Autoencoder presets
  autoencoder: {
    mnist_compress: {
      inputSize: 784,
      encoderLayers: [512, 256, 128],
      bottleneckSize: 32,
      activation: 'relu',
      outputActivation: 'sigmoid',
      dropoutRate: 0.1,
    },
    image_denoise: {
      inputSize: 4096, // 64x64 grayscale
      encoderLayers: [2048, 1024, 512],
      bottleneckSize: 256,
      activation: 'relu',
      outputActivation: 'sigmoid',
      denoisingNoise: 0.3,
      dropoutRate: 0.2,
    },
    vae_generation: {
      inputSize: 784,
      encoderLayers: [512, 256],
      bottleneckSize: 20,
      activation: 'relu',
      outputActivation: 'sigmoid',
      variational: true,
      dropoutRate: 0.1,
    },
  },

  // GNN presets
  gnn: {
    social_network: {
      nodeDimensions: 128,
      edgeDimensions: 64,
      hiddenDimensions: 256,
      outputDimensions: 128,
      numLayers: 3,
      aggregation: 'mean',
    },
    molecular: {
      nodeDimensions: 64,
      edgeDimensions: 32,
      hiddenDimensions: 128,
      outputDimensions: 64,
      numLayers: 4,
      aggregation: 'sum',
    },
    knowledge_graph: {
      nodeDimensions: 256,
      edgeDimensions: 128,
      hiddenDimensions: 512,
      outputDimensions: 256,
      numLayers: 2,
      aggregation: 'max',
    },
  },

  // ResNet presets
  resnet: {
    resnet18: {
      numBlocks: 4,
      blockDepth: 2,
      hiddenDimensions: 512,
      initialChannels: 64,
    },
    resnet34: {
      numBlocks: 6,
      blockDepth: 3,
      hiddenDimensions: 512,
      initialChannels: 64,
    },
    resnet50: {
      numBlocks: 8,
      blockDepth: 3,
      hiddenDimensions: 1024,
      initialChannels: 128,
    },
  },

  // VAE presets
  vae: {
    mnist_vae: {
      inputSize: 784,
      encoderLayers: [512, 256],
      latentDimensions: 20,
      decoderLayers: [256, 512],
      betaKL: 1.0,
    },
    cifar_vae: {
      inputSize: 3072,
      encoderLayers: [1024, 512, 256],
      latentDimensions: 128,
      decoderLayers: [256, 512, 1024],
      betaKL: 0.5,
    },
    beta_vae: {
      inputSize: 784,
      encoderLayers: [512, 256],
      latentDimensions: 10,
      decoderLayers: [256, 512],
      betaKL: 4.0, // Higher beta for disentanglement
    },
  },

  // LSTM presets
  lstm: {
    text_generation: {
      inputSize: 128,
      hiddenSize: 512,
      numLayers: 2,
      outputSize: 10000,
      bidirectional: false,
      returnSequence: true,
    },
    sentiment_analysis: {
      inputSize: 300,
      hiddenSize: 256,
      numLayers: 2,
      outputSize: 2,
      bidirectional: true,
      returnSequence: false,
    },
    time_series_forecast: {
      inputSize: 10,
      hiddenSize: 128,
      numLayers: 3,
      outputSize: 1,
      bidirectional: false,
      returnSequence: false,
    },
  },
};

// Utility function to get preset configuration
export const getModelPreset = (modelType, presetName) => {
  if (!MODEL_PRESETS[modelType]) {
    throw new Error(`No presets available for model type: ${modelType}`);
  }

  if (!MODEL_PRESETS[modelType][presetName]) {
    throw new Error(`No preset named '${presetName}' for model type: ${modelType}`);
  }

  return MODEL_PRESETS[modelType][presetName];
};