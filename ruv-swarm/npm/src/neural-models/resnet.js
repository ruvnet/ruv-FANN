/**
 * Residual Network (ResNet) Model
 * Implements deep neural networks with skip connections
 */

import { NeuralModel } from './base.js';

class ResNetModel extends NeuralModel {
  constructor(config = {}) {
    super('resnet');

    // ResNet configuration
    this.config = {
      inputDimensions: config.inputDimensions || 784, // Default for flattened MNIST
      numBlocks: config.numBlocks || 4,
      blockDepth: config.blockDepth || 2,
      hiddenDimensions: config.hiddenDimensions || 256,
      outputDimensions: config.outputDimensions || 10,
      activation: config.activation || 'relu',
      batchNorm: config.batchNorm !== false, // Default true
      dropoutRate: config.dropoutRate || 0.2,
      initialChannels: config.initialChannels || 64,
      ...config,
    };

    // Initialize layers
    this.blocks = [];
    this.batchNormParams = [];
    this.skipConnections = [];
    this.outputLayer = null;

    this.initializeWeights();
  }

  initializeWeights() {
    let currentDimensions = this.config.inputDimensions;

    // Initial projection layer
    this.inputProjection = {
      weight: this.createWeight([currentDimensions, this.config.initialChannels]),
      bias: new Float32Array(this.config.initialChannels).fill(0.0),
    };
    currentDimensions = this.config.initialChannels;

    // Create residual blocks
    for (let blockIdx = 0; blockIdx < this.config.numBlocks; blockIdx++) {
      const block = [];
      const blockBatchNorm = [];

      // Determine block dimensions
      const outputDim = Math.min(
        currentDimensions * 2,
        this.config.hiddenDimensions,
      );

      // Create layers within block
      for (let layerIdx = 0; layerIdx < this.config.blockDepth; layerIdx++) {
        const inputDim = layerIdx === 0 ? currentDimensions : outputDim;

        block.push({
          weight: this.createWeight([inputDim, outputDim]),
          bias: new Float32Array(outputDim).fill(0.0),
        });

        if (this.config.batchNorm) {
          blockBatchNorm.push({
            gamma: new Float32Array(outputDim).fill(1.0),
            beta: new Float32Array(outputDim).fill(0.0),
            runningMean: new Float32Array(outputDim).fill(0.0),
            runningVar: new Float32Array(outputDim).fill(1.0),
            momentum: 0.9,
          });
        }
      }

      // Skip connection projection if dimensions change
      if (currentDimensions !== outputDim) {
        this.skipConnections.push({
          weight: this.createWeight([currentDimensions, outputDim]),
          bias: new Float32Array(outputDim).fill(0.0),
        });
      } else {
        this.skipConnections.push(null); // Identity skip connection
      }

      this.blocks.push(block);
      this.batchNormParams.push(blockBatchNorm);
      currentDimensions = outputDim;
    }

    // Output layer
    this.outputLayer = {
      weight: this.createWeight([currentDimensions, this.config.outputDimensions]),
      bias: new Float32Array(this.config.outputDimensions).fill(0.0),
    };
  }

  createWeight(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    const weight = new Float32Array(size);

    // He initialization for ReLU
    const scale = Math.sqrt(2.0 / shape[0]);
    for (let i = 0; i < size; i++) {
      weight[i] = (Math.random() * 2 - 1) * scale;
    }

    weight.shape = shape;
    return weight;
  }

  async forward(input, training = false) {
    // Initial projection
    let x = this.linearTransform(input, this.inputProjection.weight, this.inputProjection.bias);
    x = this.applyActivation(x);

    // Process through residual blocks
    for (let blockIdx = 0; blockIdx < this.config.numBlocks; blockIdx++) {
      x = await this.forwardBlock(x, blockIdx, training);
    }

    // Global average pooling (if input has spatial dimensions)
    if (x.shape && x.shape.length > 2) {
      x = this.globalAveragePooling(x);
    }

    // Final classification layer
    const output = this.linearTransform(x, this.outputLayer.weight, this.outputLayer.bias);

    return output;
  }

  async forwardBlock(input, blockIdx, training = false) {
    const block = this.blocks[blockIdx];
    const batchNorm = this.batchNormParams[blockIdx];
    const skipConnection = this.skipConnections[blockIdx];

    // Save input for skip connection
    let identity = input;

    // Apply skip connection projection if needed
    if (skipConnection) {
      identity = this.linearTransform(input, skipConnection.weight, skipConnection.bias);
    }

    // Forward through block layers
    let x = input;
    for (let layerIdx = 0; layerIdx < block.length; layerIdx++) {
      const layer = block[layerIdx];

      // Linear transformation
      x = this.linearTransform(x, layer.weight, layer.bias);

      // Batch normalization
      if (this.config.batchNorm && batchNorm[layerIdx]) {
        x = this.batchNormalize(x, batchNorm[layerIdx], training);
      }

      // Activation (except for last layer in block)
      if (layerIdx < block.length - 1) {
        x = this.applyActivation(x);
      }

      // Dropout if training
      if (training && this.config.dropoutRate > 0 && layerIdx < block.length - 1) {
        x = this.dropout(x, this.config.dropoutRate);
      }
    }

    // Add skip connection
    x = this.add(x, identity);

    // Final activation
    x = this.applyActivation(x);

    return x;
  }

  linearTransform(input, weight, bias) {
    const batchSize = input.shape ? input.shape[0] : 1;
    const inputDim = weight.shape[0];
    const outputDim = weight.shape[1];

    const output = new Float32Array(batchSize * outputDim);

    for (let b = 0; b < batchSize; b++) {
      for (let out = 0; out < outputDim; out++) {
        let sum = bias[out];
        for (let inp = 0; inp < inputDim; inp++) {
          sum += input[b * inputDim + inp] * weight[inp * outputDim + out];
        }
        output[b * outputDim + out] = sum;
      }
    }

    output.shape = [batchSize, outputDim];
    return output;
  }

  batchNormalize(input, params, training = false) {
    const shape = input.shape || [input.length];
    const features = shape[shape.length - 1];
    const batchSize = input.length / features;

    const normalized = new Float32Array(input.length);

    if (training) {
      // Calculate batch statistics
      const mean = new Float32Array(features);
      const variance = new Float32Array(features);

      // Calculate mean
      for (let f = 0; f < features; f++) {
        let sum = 0;
        for (let b = 0; b < batchSize; b++) {
          sum += input[b * features + f];
        }
        mean[f] = sum / batchSize;
      }

      // Calculate variance
      for (let f = 0; f < features; f++) {
        let sum = 0;
        for (let b = 0; b < batchSize; b++) {
          const diff = input[b * features + f] - mean[f];
          sum += diff * diff;
        }
        variance[f] = sum / batchSize;
      }

      // Update running statistics
      for (let f = 0; f < features; f++) {
        params.runningMean[f] = params.momentum * params.runningMean[f] +
                               (1 - params.momentum) * mean[f];
        params.runningVar[f] = params.momentum * params.runningVar[f] +
                              (1 - params.momentum) * variance[f];
      }

      // Normalize using batch statistics
      for (let b = 0; b < batchSize; b++) {
        for (let f = 0; f < features; f++) {
          const idx = b * features + f;
          const norm = (input[idx] - mean[f]) / Math.sqrt(variance[f] + 1e-5);
          normalized[idx] = params.gamma[f] * norm + params.beta[f];
        }
      }
    } else {
      // Use running statistics for inference
      for (let b = 0; b < batchSize; b++) {
        for (let f = 0; f < features; f++) {
          const idx = b * features + f;
          const norm = (input[idx] - params.runningMean[f]) /
                      Math.sqrt(params.runningVar[f] + 1e-5);
          normalized[idx] = params.gamma[f] * norm + params.beta[f];
        }
      }
    }

    normalized.shape = input.shape;
    return normalized;
  }

  applyActivation(input) {
    switch (this.config.activation) {
    case 'relu':
      return this.relu(input);
    case 'leaky_relu':
      return this.leakyRelu(input);
    case 'elu':
      return this.elu(input);
    case 'swish':
      return this.swish(input);
    default:
      return this.relu(input);
    }
  }

  leakyRelu(input, alpha = 0.01) {
    const result = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      result[i] = input[i] > 0 ? input[i] : alpha * input[i];
    }
    result.shape = input.shape;
    return result;
  }

  elu(input, alpha = 1.0) {
    const result = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      result[i] = input[i] > 0 ? input[i] : alpha * (Math.exp(input[i]) - 1);
    }
    result.shape = input.shape;
    return result;
  }

  swish(input) {
    const result = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      result[i] = input[i] * this.sigmoid([input[i]])[0];
    }
    result.shape = input.shape;
    return result;
  }

  globalAveragePooling(input) {
    // Assumes input shape is [batch, height, width, channels]
    const { shape } = input;
    const batchSize = shape[0];
    const spatialSize = shape[1] * shape[2];
    const channels = shape[3];

    const pooled = new Float32Array(batchSize * channels);

    for (let b = 0; b < batchSize; b++) {
      for (let c = 0; c < channels; c++) {
        let sum = 0;
        for (let s = 0; s < spatialSize; s++) {
          sum += input[b * spatialSize * channels + s * channels + c];
        }
        pooled[b * channels + c] = sum / spatialSize;
      }
    }

    pooled.shape = [batchSize, channels];
    return pooled;
  }

  async train(trainingData, options = {}) {
    const {
      epochs = 20,
      batchSize = 32,
      learningRate = 0.001,
      weightDecay = 0.0001,
      validationSplit = 0.1,
    } = options;

    const trainingHistory = [];

    // Split data
    const splitIndex = Math.floor(trainingData.length * (1 - validationSplit));
    const trainData = trainingData.slice(0, splitIndex);
    const valData = trainingData.slice(splitIndex);

    // Learning rate schedule
    const lrSchedule = (epoch) => {
      if (epoch < 10) {
        return learningRate;
      }
      if (epoch < 15) {
        return learningRate * 0.1;
      }
      return learningRate * 0.01;
    };

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let correctPredictions = 0;
      let totalSamples = 0;

      const currentLR = lrSchedule(epoch);

      // Shuffle training data
      const shuffled = this.shuffle(trainData);

      // Process batches
      for (let i = 0; i < shuffled.length; i += batchSize) {
        const batch = shuffled.slice(i, Math.min(i + batchSize, shuffled.length));

        // Forward pass
        const predictions = await this.forward(batch.inputs, true);

        // Calculate loss with L2 regularization
        const loss = this.crossEntropyLoss(predictions, batch.targets);
        const l2Loss = this.calculateL2Loss() * weightDecay;
        const totalLoss = loss + l2Loss;

        epochLoss += totalLoss;

        // Calculate accuracy
        const predicted = this.argmax(predictions);
        const actual = this.argmax(batch.targets);
        for (let j = 0; j < predicted.length; j++) {
          if (predicted[j] === actual[j]) {
            correctPredictions++;
          }
        }
        totalSamples += batch.length;

        // Backward pass
        await this.backward(totalLoss, currentLR);
      }

      // Validation
      const valMetrics = await this.validateWithAccuracy(valData);

      const trainAccuracy = correctPredictions / totalSamples;
      const avgTrainLoss = epochLoss / Math.ceil(trainData.length / batchSize);

      trainingHistory.push({
        epoch: epoch + 1,
        trainLoss: avgTrainLoss,
        trainAccuracy,
        valLoss: valMetrics.loss,
        valAccuracy: valMetrics.accuracy,
        learningRate: currentLR,
      });

      console.log(
        `Epoch ${epoch + 1}/${epochs} - ` +
        `Train Loss: ${avgTrainLoss.toFixed(4)}, Train Acc: ${(trainAccuracy * 100).toFixed(2)}% - ` +
        `Val Loss: ${valMetrics.loss.toFixed(4)}, Val Acc: ${(valMetrics.accuracy * 100).toFixed(2)}%`,
      );
    }

    return {
      history: trainingHistory,
      finalLoss: trainingHistory[trainingHistory.length - 1].trainLoss,
      modelType: 'resnet',
      accuracy: trainingHistory[trainingHistory.length - 1].valAccuracy,
    };
  }

  calculateL2Loss() {
    let l2Sum = 0;
    let count = 0;

    // Add L2 norm of all weights
    for (const block of this.blocks) {
      for (const layer of block) {
        for (let i = 0; i < layer.weight.length; i++) {
          l2Sum += layer.weight[i] * layer.weight[i];
          count++;
        }
      }
    }

    return l2Sum / count;
  }

  argmax(tensor) {
    // Assumes tensor shape is [batch, classes]
    const batchSize = tensor.shape[0];
    const numClasses = tensor.shape[1];
    const result = new Int32Array(batchSize);

    for (let b = 0; b < batchSize; b++) {
      let maxIdx = 0;
      let maxVal = tensor[b * numClasses];

      for (let c = 1; c < numClasses; c++) {
        if (tensor[b * numClasses + c] > maxVal) {
          maxVal = tensor[b * numClasses + c];
          maxIdx = c;
        }
      }

      result[b] = maxIdx;
    }

    return result;
  }

  async validateWithAccuracy(validationData) {
    let totalLoss = 0;
    let correctPredictions = 0;
    let totalSamples = 0;

    for (const batch of validationData) {
      const predictions = await this.forward(batch.inputs, false);
      const loss = this.crossEntropyLoss(predictions, batch.targets);
      totalLoss += loss;

      const predicted = this.argmax(predictions);
      const actual = this.argmax(batch.targets);
      for (let i = 0; i < predicted.length; i++) {
        if (predicted[i] === actual[i]) {
          correctPredictions++;
        }
      }
      totalSamples += batch.inputs.shape[0];
    }

    return {
      loss: totalLoss / validationData.length,
      accuracy: correctPredictions / totalSamples,
    };
  }

  getConfig() {
    return {
      type: 'resnet',
      ...this.config,
      parameters: this.countParameters(),
      depth: this.config.numBlocks * this.config.blockDepth + 2, // +2 for input and output layers
    };
  }

  countParameters() {
    let count = 0;

    // Input projection
    count += this.inputProjection.weight.length + this.inputProjection.bias.length;

    // Residual blocks
    for (let blockIdx = 0; blockIdx < this.blocks.length; blockIdx++) {
      const block = this.blocks[blockIdx];

      // Block layers
      for (const layer of block) {
        count += layer.weight.length + layer.bias.length;
      }

      // Skip connection
      if (this.skipConnections[blockIdx]) {
        count += this.skipConnections[blockIdx].weight.length;
        count += this.skipConnections[blockIdx].bias.length;
      }

      // Batch norm parameters
      if (this.config.batchNorm) {
        for (const bn of this.batchNormParams[blockIdx]) {
          count += bn.gamma.length + bn.beta.length;
        }
      }
    }

    // Output layer
    count += this.outputLayer.weight.length + this.outputLayer.bias.length;

    return count;
  }
}

export { ResNetModel };