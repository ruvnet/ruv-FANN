/**
 * Convolutional Neural Network (CNN) Model
 * For pattern recognition and image processing tasks
 */

import { NeuralModel } from './base.js';

class CNNModel extends NeuralModel {
  constructor(config = {}) {
    super('cnn');

    // CNN configuration
    this.config = {
      inputShape: config.inputShape || [28, 28, 1], // [height, width, channels]
      convLayers: config.convLayers || [
        { filters: 32, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' },
        { filters: 64, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' },
        { filters: 128, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' },
      ],
      poolingSize: config.poolingSize || 2,
      denseLayers: config.denseLayers || [128, 64],
      outputSize: config.outputSize || 10,
      dropoutRate: config.dropoutRate || 0.5,
      ...config,
    };

    // Initialize layers
    this.convWeights = [];
    this.convBiases = [];
    this.denseWeights = [];
    this.denseBiases = [];

    this.initializeWeights();
  }

  initializeWeights() {
    let currentShape = [...this.config.inputShape];

    // Initialize convolutional layers
    for (const convLayer of this.config.convLayers) {
      const { filters, kernelSize } = convLayer;
      const inputChannels = currentShape[2];

      // Initialize kernel weights [kernelSize, kernelSize, inputChannels, filters]
      const kernelWeights = this.createWeight([
        kernelSize,
        kernelSize,
        inputChannels,
        filters,
      ]);

      this.convWeights.push({
        kernel: kernelWeights,
        shape: [kernelSize, kernelSize, inputChannels, filters],
      });

      // Initialize biases for each filter
      this.convBiases.push(new Float32Array(filters).fill(0));

      // Update shape for next layer
      currentShape = this.getConvOutputShape(currentShape, convLayer);

      // Apply pooling
      if (this.config.poolingSize > 1) {
        currentShape = [
          Math.floor(currentShape[0] / this.config.poolingSize),
          Math.floor(currentShape[1] / this.config.poolingSize),
          currentShape[2],
        ];
      }
    }

    // Calculate flattened size
    const flattenedSize = currentShape.reduce((a, b) => a * b, 1);

    // Initialize dense layers
    let lastSize = flattenedSize;
    for (const units of this.config.denseLayers) {
      this.denseWeights.push(this.createWeight([lastSize, units]));
      this.denseBiases.push(new Float32Array(units).fill(0));
      lastSize = units;
    }

    // Output layer
    this.denseWeights.push(this.createWeight([lastSize, this.config.outputSize]));
    this.denseBiases.push(new Float32Array(this.config.outputSize).fill(0));
  }

  createWeight(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    const weight = new Float32Array(size);

    // He initialization for ReLU activation
    const fanIn = shape.slice(0, -1).reduce((a, b) => a * b, 1);
    const scale = Math.sqrt(2.0 / fanIn);

    for (let i = 0; i < size; i++) {
      weight[i] = (Math.random() * 2 - 1) * scale;
    }

    return weight;
  }

  getConvOutputShape(inputShape, convLayer) {
    const [height, width, channels] = inputShape;
    const { filters, kernelSize, stride = 1, padding } = convLayer;

    let outputHeight, outputWidth;

    if (padding === 'same') {
      outputHeight = Math.ceil(height / stride);
      outputWidth = Math.ceil(width / stride);
    } else {
      outputHeight = Math.floor((height - kernelSize) / stride) + 1;
      outputWidth = Math.floor((width - kernelSize) / stride) + 1;
    }

    return [outputHeight, outputWidth, filters];
  }

  async forward(input, training = false) {
    let x = input;

    // Convolutional layers
    for (let i = 0; i < this.config.convLayers.length; i++) {
      x = this.conv2d(x, i);

      // Apply activation
      const { activation } = this.config.convLayers[i];
      if (activation === 'relu') {
        x = this.relu(x);
      }

      // Apply pooling
      if (this.config.poolingSize > 1) {
        x = this.maxPool2d(x, this.config.poolingSize);
      }
    }

    // Flatten
    x = this.flatten(x);

    // Dense layers
    for (let i = 0; i < this.config.denseLayers.length; i++) {
      x = this.dense(x, this.denseWeights[i], this.denseBiases[i]);
      x = this.relu(x);

      // Apply dropout if training
      if (training && this.config.dropoutRate > 0) {
        x = this.dropout(x, this.config.dropoutRate);
      }
    }

    // Output layer
    const outputIndex = this.denseWeights.length - 1;
    x = this.dense(x, this.denseWeights[outputIndex], this.denseBiases[outputIndex]);

    // Apply softmax for classification
    x = this.softmax(x);

    return x;
  }

  conv2d(input, layerIndex) {
    const convLayer = this.config.convLayers[layerIndex];
    const weights = this.convWeights[layerIndex];
    const biases = this.convBiases[layerIndex];

    const [batchSize, height, width, inputChannels] = input.shape;
    const { filters, kernelSize, stride = 1, padding } = convLayer;

    // Calculate output dimensions
    const outputShape = this.getConvOutputShape([height, width, inputChannels], convLayer);
    const [outputHeight, outputWidth, outputChannels] = outputShape;

    const output = new Float32Array(batchSize * outputHeight * outputWidth * outputChannels);

    // Apply convolution
    for (let b = 0; b < batchSize; b++) {
      for (let oh = 0; oh < outputHeight; oh++) {
        for (let ow = 0; ow < outputWidth; ow++) {
          for (let oc = 0; oc < outputChannels; oc++) {
            let sum = biases[oc];

            // Apply kernel
            for (let kh = 0; kh < kernelSize; kh++) {
              for (let kw = 0; kw < kernelSize; kw++) {
                for (let ic = 0; ic < inputChannels; ic++) {
                  let ih = oh * stride + kh;
                  let iw = ow * stride + kw;

                  // Handle padding
                  if (padding === 'same') {
                    ih -= Math.floor(kernelSize / 2);
                    iw -= Math.floor(kernelSize / 2);
                  }

                  // Check bounds
                  if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    const inputIdx = b * height * width * inputChannels +
                                   ih * width * inputChannels +
                                   iw * inputChannels + ic;

                    const weightIdx = kh * kernelSize * inputChannels * filters +
                                    kw * inputChannels * filters +
                                    ic * filters + oc;

                    sum += input[inputIdx] * weights.kernel[weightIdx];
                  }
                }
              }
            }

            const outputIdx = b * outputHeight * outputWidth * outputChannels +
                            oh * outputWidth * outputChannels +
                            ow * outputChannels + oc;

            output[outputIdx] = sum;
          }
        }
      }
    }

    output.shape = [batchSize, outputHeight, outputWidth, outputChannels];
    return output;
  }

  maxPool2d(input, poolSize) {
    const [batchSize, height, width, channels] = input.shape;
    const outputHeight = Math.floor(height / poolSize);
    const outputWidth = Math.floor(width / poolSize);

    const output = new Float32Array(batchSize * outputHeight * outputWidth * channels);

    for (let b = 0; b < batchSize; b++) {
      for (let oh = 0; oh < outputHeight; oh++) {
        for (let ow = 0; ow < outputWidth; ow++) {
          for (let c = 0; c < channels; c++) {
            let maxVal = -Infinity;

            // Find max in pool window
            for (let ph = 0; ph < poolSize; ph++) {
              for (let pw = 0; pw < poolSize; pw++) {
                const ih = oh * poolSize + ph;
                const iw = ow * poolSize + pw;

                if (ih < height && iw < width) {
                  const inputIdx = b * height * width * channels +
                                 ih * width * channels +
                                 iw * channels + c;

                  maxVal = Math.max(maxVal, input[inputIdx]);
                }
              }
            }

            const outputIdx = b * outputHeight * outputWidth * channels +
                            oh * outputWidth * channels +
                            ow * channels + c;

            output[outputIdx] = maxVal;
          }
        }
      }
    }

    output.shape = [batchSize, outputHeight, outputWidth, channels];
    return output;
  }

  flatten(input) {
    const [batchSize, ...dims] = input.shape;
    const flatSize = dims.reduce((a, b) => a * b, 1);

    const output = new Float32Array(batchSize * flatSize);

    // Copy data in flattened order
    for (let i = 0; i < output.length; i++) {
      output[i] = input[i];
    }

    output.shape = [batchSize, flatSize];
    return output;
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

  softmax(input) {
    const [batchSize, size] = input.shape;
    const output = new Float32Array(input.length);

    for (let b = 0; b < batchSize; b++) {
      const offset = b * size;
      let maxVal = -Infinity;

      // Find max for numerical stability
      for (let i = 0; i < size; i++) {
        maxVal = Math.max(maxVal, input[offset + i]);
      }

      // Compute exp and sum
      let sumExp = 0;
      for (let i = 0; i < size; i++) {
        output[offset + i] = Math.exp(input[offset + i] - maxVal);
        sumExp += output[offset + i];
      }

      // Normalize
      for (let i = 0; i < size; i++) {
        output[offset + i] /= sumExp;
      }
    }

    output.shape = input.shape;
    return output;
  }

  async train(trainingData, options = {}) {
    const {
      epochs = 10,
      batchSize = 32,
      learningRate = 0.001,
      validationSplit = 0.1,
    } = options;

    const trainingHistory = [];

    // Split data
    const splitIndex = Math.floor(trainingData.length * (1 - validationSplit));
    const trainData = trainingData.slice(0, splitIndex);
    const valData = trainingData.slice(splitIndex);

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let epochAccuracy = 0;
      let batchCount = 0;

      // Shuffle training data
      const shuffled = this.shuffle(trainData);

      // Process batches
      for (let i = 0; i < shuffled.length; i += batchSize) {
        const batch = shuffled.slice(i, Math.min(i + batchSize, shuffled.length));

        // Forward pass
        const predictions = await this.forward(batch.inputs, true);

        // Calculate loss and accuracy
        const loss = this.crossEntropyLoss(predictions, batch.targets);
        const accuracy = this.calculateAccuracy(predictions, batch.targets);

        epochLoss += loss;
        epochAccuracy += accuracy;

        // Backward pass (simplified)
        await this.backward(loss, learningRate);

        batchCount++;
      }

      // Validation
      const valMetrics = await this.evaluate(valData);

      const avgTrainLoss = epochLoss / batchCount;
      const avgTrainAccuracy = epochAccuracy / batchCount;

      trainingHistory.push({
        epoch: epoch + 1,
        trainLoss: avgTrainLoss,
        trainAccuracy: avgTrainAccuracy,
        valLoss: valMetrics.loss,
        valAccuracy: valMetrics.accuracy,
      });

      console.log(
        `Epoch ${epoch + 1}/${epochs} - ` +
        `Train Loss: ${avgTrainLoss.toFixed(4)}, ` +
        `Train Acc: ${(avgTrainAccuracy * 100).toFixed(2)}%, ` +
        `Val Loss: ${valMetrics.loss.toFixed(4)}, ` +
        `Val Acc: ${(valMetrics.accuracy * 100).toFixed(2)}%`,
      );

      this.updateMetrics(avgTrainLoss, avgTrainAccuracy);
    }

    return {
      history: trainingHistory,
      finalLoss: trainingHistory[trainingHistory.length - 1].trainLoss,
      finalAccuracy: trainingHistory[trainingHistory.length - 1].trainAccuracy,
      modelType: 'cnn',
    };
  }

  async evaluate(data) {
    let totalLoss = 0;
    let totalAccuracy = 0;
    let batchCount = 0;

    for (const batch of data) {
      const predictions = await this.forward(batch.inputs, false);
      const loss = this.crossEntropyLoss(predictions, batch.targets);
      const accuracy = this.calculateAccuracy(predictions, batch.targets);

      totalLoss += loss;
      totalAccuracy += accuracy;
      batchCount++;
    }

    return {
      loss: totalLoss / batchCount,
      accuracy: totalAccuracy / batchCount,
    };
  }

  calculateAccuracy(predictions, targets) {
    let correct = 0;
    const batchSize = predictions.shape[0];
    const numClasses = predictions.shape[1];

    for (let b = 0; b < batchSize; b++) {
      let predClass = 0;
      let maxProb = -Infinity;

      // Find predicted class
      for (let c = 0; c < numClasses; c++) {
        const prob = predictions[b * numClasses + c];
        if (prob > maxProb) {
          maxProb = prob;
          predClass = c;
        }
      }

      // Find true class
      let trueClass = 0;
      for (let c = 0; c < numClasses; c++) {
        if (targets[b * numClasses + c] === 1) {
          trueClass = c;
          break;
        }
      }

      if (predClass === trueClass) {
        correct++;
      }
    }

    return correct / batchSize;
  }

  getConfig() {
    return {
      type: 'cnn',
      ...this.config,
      parameters: this.countParameters(),
    };
  }

  countParameters() {
    let count = 0;

    // Convolutional layers
    for (let i = 0; i < this.convWeights.length; i++) {
      count += this.convWeights[i].kernel.length;
      count += this.convBiases[i].length;
    }

    // Dense layers
    for (let i = 0; i < this.denseWeights.length; i++) {
      count += this.denseWeights[i].length;
      count += this.denseBiases[i].length;
    }

    return count;
  }
}

export { CNNModel };