/**
 * Autoencoder Neural Network Model
 * For dimensionality reduction, feature learning, and data compression
 */

import { NeuralModel } from './base.js';

class AutoencoderModel extends NeuralModel {
  constructor(config = {}) {
    super('autoencoder');

    // Autoencoder configuration
    this.config = {
      inputSize: config.inputSize || 784, // e.g., 28x28 flattened image
      encoderLayers: config.encoderLayers || [512, 256, 128, 64], // Progressive compression
      bottleneckSize: config.bottleneckSize || 32, // Latent space dimension
      decoderLayers: config.decoderLayers || null, // Mirror of encoder if not specified
      activation: config.activation || 'relu',
      outputActivation: config.outputActivation || 'sigmoid',
      dropoutRate: config.dropoutRate || 0.1,
      sparseRegularization: config.sparseRegularization || 0.01,
      denoisingNoise: config.denoisingNoise || 0, // For denoising autoencoder
      variational: config.variational || false, // For VAE
      ...config,
    };

    // Set decoder layers as mirror of encoder if not specified
    if (!this.config.decoderLayers) {
      this.config.decoderLayers = [...this.config.encoderLayers].reverse();
    }

    // Initialize network components
    this.encoderWeights = [];
    this.encoderBiases = [];
    this.decoderWeights = [];
    this.decoderBiases = [];

    // For variational autoencoder
    if (this.config.variational) {
      this.muLayer = null;
      this.logVarLayer = null;
    }

    this.initializeWeights();
  }

  initializeWeights() {
    let lastSize = this.config.inputSize;

    // Initialize encoder layers
    for (const units of this.config.encoderLayers) {
      this.encoderWeights.push(this.createWeight([lastSize, units]));
      this.encoderBiases.push(new Float32Array(units).fill(0));
      lastSize = units;
    }

    // Bottleneck layer
    if (this.config.variational) {
      // For VAE: separate layers for mean and log variance
      this.muLayer = {
        weight: this.createWeight([lastSize, this.config.bottleneckSize]),
        bias: new Float32Array(this.config.bottleneckSize).fill(0),
      };
      this.logVarLayer = {
        weight: this.createWeight([lastSize, this.config.bottleneckSize]),
        bias: new Float32Array(this.config.bottleneckSize).fill(0),
      };
      lastSize = this.config.bottleneckSize;
    } else {
      // Standard autoencoder bottleneck
      this.encoderWeights.push(this.createWeight([lastSize, this.config.bottleneckSize]));
      this.encoderBiases.push(new Float32Array(this.config.bottleneckSize).fill(0));
      lastSize = this.config.bottleneckSize;
    }

    // Initialize decoder layers
    for (const units of this.config.decoderLayers) {
      this.decoderWeights.push(this.createWeight([lastSize, units]));
      this.decoderBiases.push(new Float32Array(units).fill(0));
      lastSize = units;
    }

    // Output layer (reconstruction)
    this.decoderWeights.push(this.createWeight([lastSize, this.config.inputSize]));
    this.decoderBiases.push(new Float32Array(this.config.inputSize).fill(0));
  }

  createWeight(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    const weight = new Float32Array(size);

    // Xavier/Glorot initialization
    const scale = Math.sqrt(2.0 / (shape[0] + shape[1]));

    for (let i = 0; i < size; i++) {
      weight[i] = (Math.random() * 2 - 1) * scale;
    }

    weight.shape = shape;
    return weight;
  }

  async forward(input, training = false) {
    // Add noise for denoising autoencoder
    let x = input;
    if (training && this.config.denoisingNoise > 0) {
      x = this.addNoise(input, this.config.denoisingNoise);
    }

    // Encode
    const encodingResult = await this.encode(x, training);

    // Decode
    const reconstruction = await this.decode(encodingResult.latent, training);

    return {
      reconstruction,
      latent: encodingResult.latent,
      mu: encodingResult.mu,
      logVar: encodingResult.logVar,
    };
  }

  async encode(input, training = false) {
    let x = input;

    // Pass through encoder layers
    for (let i = 0; i < this.encoderWeights.length; i++) {
      x = this.dense(x, this.encoderWeights[i], this.encoderBiases[i]);

      // Apply activation
      if (this.config.activation === 'relu') {
        x = this.relu(x);
      } else if (this.config.activation === 'tanh') {
        x = this.tanh(x);
      } else if (this.config.activation === 'sigmoid') {
        x = this.sigmoid(x);
      }

      // Apply dropout if training (except last layer)
      if (training && this.config.dropoutRate > 0 && i < this.encoderWeights.length - 1) {
        x = this.dropout(x, this.config.dropoutRate);
      }
    }

    // Handle variational autoencoder
    if (this.config.variational) {
      const mu = this.dense(x, this.muLayer.weight, this.muLayer.bias);
      const logVar = this.dense(x, this.logVarLayer.weight, this.logVarLayer.bias);

      // Reparameterization trick
      const latent = training ? this.reparameterize(mu, logVar) : mu;

      return { latent, mu, logVar };
    }

    return { latent: x, mu: null, logVar: null };
  }

  async decode(latent, training = false) {
    let x = latent;

    // Pass through decoder layers
    for (let i = 0; i < this.decoderWeights.length; i++) {
      x = this.dense(x, this.decoderWeights[i], this.decoderBiases[i]);

      // Apply activation (use output activation for last layer)
      if (i === this.decoderWeights.length - 1) {
        if (this.config.outputActivation === 'sigmoid') {
          x = this.sigmoid(x);
        } else if (this.config.outputActivation === 'tanh') {
          x = this.tanh(x);
        }
        // 'linear' means no activation
      } else {
        // Hidden layers
        if (this.config.activation === 'relu') {
          x = this.relu(x);
        } else if (this.config.activation === 'tanh') {
          x = this.tanh(x);
        } else if (this.config.activation === 'sigmoid') {
          x = this.sigmoid(x);
        }

        // Apply dropout if training
        if (training && this.config.dropoutRate > 0) {
          x = this.dropout(x, this.config.dropoutRate);
        }
      }
    }

    return x;
  }

  dense(input, weights, biases) {
    const [batchSize, inputSize] = input.shape;
    const outputSize = biases.length;

    const output = new Float32Array(batchSize * outputSize);

    for (let b = 0; b < batchSize; b++) {
      for (let o = 0; o < outputSize; o++) {
        let sum = biases[o];

        for (let i = 0; i < inputSize; i++) {
          sum += input[b * inputSize + i] * weights[i * outputSize + o];
        }

        output[b * outputSize + o] = sum;
      }
    }

    output.shape = [batchSize, outputSize];
    return output;
  }

  addNoise(input, noiseLevel) {
    const noisy = new Float32Array(input.length);

    for (let i = 0; i < input.length; i++) {
      // Add Gaussian noise
      const noise = (Math.random() - 0.5) * 2 * noiseLevel;
      noisy[i] = Math.max(0, Math.min(1, input[i] + noise));
    }

    noisy.shape = input.shape;
    return noisy;
  }

  reparameterize(mu, logVar) {
    // VAE reparameterization trick: z = mu + sigma * epsilon
    const [batchSize, latentSize] = mu.shape;

    const z = new Float32Array(batchSize * latentSize);

    for (let b = 0; b < batchSize; b++) {
      for (let l = 0; l < latentSize; l++) {
        const idx = b * latentSize + l;
        const epsilon = this.sampleGaussian(); // N(0, 1)
        const sigma = Math.exp(0.5 * logVar[idx]);
        z[idx] = mu[idx] + sigma * epsilon;
      }
    }

    z.shape = mu.shape;
    return z;
  }

  sampleGaussian() {
    // Box-Muller transform for sampling from standard normal distribution
    let u = 0, v = 0;
    while (u === 0) {
      u = Math.random();
    }
    while (v === 0) {
      v = Math.random();
    }
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  calculateLoss(input, output, mu = null, logVar = null) {
    const [batchSize] = input.shape;

    // Reconstruction loss (MSE or binary cross-entropy)
    let reconstructionLoss = 0;

    if (this.config.outputActivation === 'sigmoid') {
      // Binary cross-entropy for outputs in [0, 1]
      for (let i = 0; i < input.length; i++) {
        const epsilon = 1e-7;
        const pred = Math.max(epsilon, Math.min(1 - epsilon, output.reconstruction[i]));
        reconstructionLoss -= input[i] * Math.log(pred) + (1 - input[i]) * Math.log(1 - pred);
      }
    } else {
      // MSE for continuous outputs
      for (let i = 0; i < input.length; i++) {
        const diff = input[i] - output.reconstruction[i];
        reconstructionLoss += diff * diff;
      }
    }

    reconstructionLoss /= batchSize;

    // KL divergence for VAE
    let klLoss = 0;
    if (this.config.variational && mu && logVar) {
      for (let i = 0; i < mu.length; i++) {
        klLoss += -0.5 * (1 + logVar[i] - mu[i] * mu[i] - Math.exp(logVar[i]));
      }
      klLoss /= batchSize;
    }

    // Sparsity regularization (encourage sparse activations)
    let sparsityLoss = 0;
    if (this.config.sparseRegularization > 0) {
      const targetSparsity = 0.05; // Target average activation
      const latentMean = output.latent.reduce((a, b) => a + b, 0) / output.latent.length;
      sparsityLoss = this.config.sparseRegularization * Math.abs(latentMean - targetSparsity);
    }

    return {
      total: reconstructionLoss + klLoss + sparsityLoss,
      reconstruction: reconstructionLoss,
      kl: klLoss,
      sparsity: sparsityLoss,
    };
  }

  async train(trainingData, options = {}) {
    const {
      epochs = 10,
      batchSize = 32,
      learningRate = 0.001,
      validationSplit = 0.1,
      beta = 1.0, // Beta-VAE parameter
    } = options;

    const trainingHistory = [];

    // Split data
    const splitIndex = Math.floor(trainingData.length * (1 - validationSplit));
    const trainData = trainingData.slice(0, splitIndex);
    const valData = trainingData.slice(splitIndex);

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let epochReconLoss = 0;
      let epochKLLoss = 0;
      let batchCount = 0;

      // Shuffle training data
      const shuffled = this.shuffle(trainData);

      // Process batches
      for (let i = 0; i < shuffled.length; i += batchSize) {
        const batch = shuffled.slice(i, Math.min(i + batchSize, shuffled.length));

        // Prepare batch input
        const batchInput = {
          data: batch.inputs,
          shape: [batch.inputs.length, this.config.inputSize],
        };
        batchInput.data.shape = batchInput.shape;

        // Forward pass
        const output = await this.forward(batchInput.data, true);

        // Calculate losses
        const losses = this.calculateLoss(
          batchInput.data,
          output,
          output.mu,
          output.logVar,
        );

        // Apply beta weighting for VAE
        const totalLoss = losses.reconstruction + beta * losses.kl + losses.sparsity;

        epochLoss += totalLoss;
        epochReconLoss += losses.reconstruction;
        epochKLLoss += losses.kl;

        // Backward pass
        await this.backward(totalLoss, learningRate);

        batchCount++;
      }

      // Validation
      const valLosses = await this.evaluate(valData);

      const avgTrainLoss = epochLoss / batchCount;
      const avgReconLoss = epochReconLoss / batchCount;
      const avgKLLoss = epochKLLoss / batchCount;

      const historyEntry = {
        epoch: epoch + 1,
        trainLoss: avgTrainLoss,
        reconstructionLoss: avgReconLoss,
        klLoss: avgKLLoss,
        valLoss: valLosses.total,
        valReconstructionLoss: valLosses.reconstruction,
      };

      trainingHistory.push(historyEntry);

      console.log(
        `Epoch ${epoch + 1}/${epochs} - ` +
        `Loss: ${avgTrainLoss.toFixed(4)} ` +
        `(Recon: ${avgReconLoss.toFixed(4)}, ` +
        `KL: ${avgKLLoss.toFixed(4)}) - ` +
        `Val Loss: ${valLosses.total.toFixed(4)}`,
      );

      this.updateMetrics(avgTrainLoss);
    }

    return {
      history: trainingHistory,
      finalLoss: trainingHistory[trainingHistory.length - 1].trainLoss,
      modelType: 'autoencoder',
    };
  }

  async evaluate(data) {
    let totalLoss = 0;
    let reconLoss = 0;
    let klLoss = 0;
    let batchCount = 0;

    for (const batch of data) {
      const batchInput = {
        data: batch.inputs,
        shape: [batch.inputs.length, this.config.inputSize],
      };
      batchInput.data.shape = batchInput.shape;

      const output = await this.forward(batchInput.data, false);
      const losses = this.calculateLoss(batchInput.data, output, output.mu, output.logVar);

      totalLoss += losses.total;
      reconLoss += losses.reconstruction;
      klLoss += losses.kl;
      batchCount++;
    }

    return {
      total: totalLoss / batchCount,
      reconstruction: reconLoss / batchCount,
      kl: klLoss / batchCount,
    };
  }

  // Get only the encoder part for feature extraction
  async getEncoder() {
    return {
      encode: async(input) => {
        const result = await this.encode(input, false);
        return result.latent;
      },
      config: {
        inputSize: this.config.inputSize,
        bottleneckSize: this.config.bottleneckSize,
        layers: this.config.encoderLayers,
      },
    };
  }

  // Get only the decoder part for generation
  async getDecoder() {
    return {
      decode: async(latent) => {
        return await this.decode(latent, false);
      },
      config: {
        bottleneckSize: this.config.bottleneckSize,
        outputSize: this.config.inputSize,
        layers: this.config.decoderLayers,
      },
    };
  }

  // Generate new samples (for VAE)
  async generate(numSamples = 1) {
    if (!this.config.variational) {
      throw new Error('Generation is only available for variational autoencoders');
    }

    // Sample from standard normal distribution
    const latent = new Float32Array(numSamples * this.config.bottleneckSize);

    for (let i = 0; i < latent.length; i++) {
      latent[i] = this.sampleGaussian();
    }

    latent.shape = [numSamples, this.config.bottleneckSize];

    // Decode to generate samples
    return await this.decode(latent, false);
  }

  // Interpolate between two inputs
  async interpolate(input1, input2, steps = 10) {
    // Encode both inputs
    const encoded1 = await this.encode(input1, false);
    const encoded2 = await this.encode(input2, false);

    const interpolations = [];

    for (let step = 0; step <= steps; step++) {
      const alpha = step / steps;
      const interpolatedLatent = new Float32Array(encoded1.latent.length);

      // Linear interpolation in latent space
      for (let i = 0; i < interpolatedLatent.length; i++) {
        interpolatedLatent[i] = (1 - alpha) * encoded1.latent[i] + alpha * encoded2.latent[i];
      }

      interpolatedLatent.shape = encoded1.latent.shape;

      // Decode interpolated latent vector
      const decoded = await this.decode(interpolatedLatent, false);
      interpolations.push(decoded);
    }

    return interpolations;
  }

  getConfig() {
    return {
      type: 'autoencoder',
      variant: this.config.variational ? 'variational' : 'standard',
      ...this.config,
      parameters: this.countParameters(),
    };
  }

  countParameters() {
    let count = 0;

    // Encoder parameters
    for (let i = 0; i < this.encoderWeights.length; i++) {
      count += this.encoderWeights[i].length;
      count += this.encoderBiases[i].length;
    }

    // VAE-specific parameters
    if (this.config.variational) {
      count += this.muLayer.weight.length + this.muLayer.bias.length;
      count += this.logVarLayer.weight.length + this.logVarLayer.bias.length;
    }

    // Decoder parameters
    for (let i = 0; i < this.decoderWeights.length; i++) {
      count += this.decoderWeights[i].length;
      count += this.decoderBiases[i].length;
    }

    return count;
  }
}

export { AutoencoderModel };