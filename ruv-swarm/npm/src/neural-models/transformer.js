/**
 * Transformer Neural Network Model
 * Implements multi-head attention mechanism with positional encoding
 */

import { NeuralModel } from './base.js';

class TransformerModel extends NeuralModel {
  constructor(config = {}) {
    super('transformer');

    // Transformer configuration
    this.config = {
      dimensions: config.dimensions || 512,
      heads: config.heads || 8,
      layers: config.layers || 6,
      ffDimensions: config.ffDimensions || 2048,
      maxSequenceLength: config.maxSequenceLength || 1024,
      vocabularySize: config.vocabularySize || 50000,
      dropoutRate: config.dropoutRate || 0.1,
      ...config,
    };

    // Initialize components
    this.headDimension = Math.floor(this.config.dimensions / this.config.heads);
    this.positionalEncoding = this.createPositionalEncoding();
    this.attentionWeights = new Map();
    this.layerNorms = [];
    this.feedForwardWeights = [];

    this.initializeWeights();
  }

  initializeWeights() {
    // Initialize multi-head attention weights for each layer
    for (let layer = 0; layer < this.config.layers; layer++) {
      this.attentionWeights.set(`layer_${layer}`, {
        query: this.createWeight([this.config.dimensions, this.config.dimensions]),
        key: this.createWeight([this.config.dimensions, this.config.dimensions]),
        value: this.createWeight([this.config.dimensions, this.config.dimensions]),
        output: this.createWeight([this.config.dimensions, this.config.dimensions]),
      });

      // Layer normalization parameters
      this.layerNorms.push({
        gamma: new Float32Array(this.config.dimensions).fill(1.0),
        beta: new Float32Array(this.config.dimensions).fill(0.0),
      });

      // Feed-forward network weights
      this.feedForwardWeights.push({
        w1: this.createWeight([this.config.dimensions, this.config.ffDimensions]),
        b1: new Float32Array(this.config.ffDimensions).fill(0.0),
        w2: this.createWeight([this.config.ffDimensions, this.config.dimensions]),
        b2: new Float32Array(this.config.dimensions).fill(0.0),
      });
    }

    // Output layer weights
    this.outputWeights = {
      projection: this.createWeight([this.config.dimensions, this.config.vocabularySize]),
      bias: new Float32Array(this.config.vocabularySize).fill(0.0),
    };
  }

  createWeight(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    const weight = new Float32Array(size);

    // Xavier/Glorot initialization
    const scale = Math.sqrt(2.0 / (shape[0] + shape[1]));
    for (let i = 0; i < size; i++) {
      weight[i] = (Math.random() * 2 - 1) * scale;
    }

    return weight;
  }

  createPositionalEncoding() {
    const encoding = new Float32Array(this.config.maxSequenceLength * this.config.dimensions);

    for (let pos = 0; pos < this.config.maxSequenceLength; pos++) {
      for (let i = 0; i < this.config.dimensions; i++) {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / this.config.dimensions);

        if (i % 2 === 0) {
          encoding[pos * this.config.dimensions + i] = Math.sin(angle);
        } else {
          encoding[pos * this.config.dimensions + i] = Math.cos(angle);
        }
      }
    }

    return encoding;
  }

  async forward(input, training = false) {
    // Input should be token indices [batch_size, sequence_length]
    const batchSize = input.shape[0];
    const sequenceLength = input.shape[1];

    // Token embedding (simplified - in practice would use embedding layer)
    let x = this.tokenEmbedding(input);

    // Add positional encoding
    x = this.addPositionalEncoding(x, sequenceLength);

    // Apply dropout if training
    if (training && this.config.dropoutRate > 0) {
      x = this.dropout(x, this.config.dropoutRate);
    }

    // Process through transformer layers
    for (let layer = 0; layer < this.config.layers; layer++) {
      // Multi-head self-attention
      const attentionOutput = await this.multiHeadAttention(x, layer, training);

      // Add & Norm
      x = this.layerNorm(this.add(x, attentionOutput), this.layerNorms[layer]);

      // Feed-forward network
      const ffOutput = this.feedForward(x, layer);

      // Add & Norm
      x = this.layerNorm(this.add(x, ffOutput), this.layerNorms[layer]);
    }

    // Final output projection
    const output = this.outputProjection(x);

    return output;
  }

  async multiHeadAttention(input, layerIndex, training = false) {
    const weights = this.attentionWeights.get(`layer_${layerIndex}`);
    const batchSize = input.shape[0];
    const sequenceLength = input.shape[1];

    // Linear projections for Q, K, V
    const Q = this.matmul(input, weights.query);
    const K = this.matmul(input, weights.key);
    const V = this.matmul(input, weights.value);

    // Reshape for multi-head attention
    const QHeads = this.reshapeForHeads(Q, batchSize, sequenceLength);
    const KHeads = this.reshapeForHeads(K, batchSize, sequenceLength);
    const VHeads = this.reshapeForHeads(V, batchSize, sequenceLength);

    // Scaled dot-product attention for each head
    const attentionScores = new Float32Array(batchSize * this.config.heads * sequenceLength * sequenceLength);

    for (let b = 0; b < batchSize; b++) {
      for (let h = 0; h < this.config.heads; h++) {
        for (let i = 0; i < sequenceLength; i++) {
          for (let j = 0; j < sequenceLength; j++) {
            let score = 0;

            // Compute dot product
            for (let d = 0; d < this.headDimension; d++) {
              const qIdx = b * this.config.heads * sequenceLength * this.headDimension +
                          h * sequenceLength * this.headDimension +
                          i * this.headDimension + d;
              const kIdx = b * this.config.heads * sequenceLength * this.headDimension +
                          h * sequenceLength * this.headDimension +
                          j * this.headDimension + d;

              score += QHeads[qIdx] * KHeads[kIdx];
            }

            // Scale by sqrt(d_k)
            score /= Math.sqrt(this.headDimension);

            const scoreIdx = b * this.config.heads * sequenceLength * sequenceLength +
                           h * sequenceLength * sequenceLength +
                           i * sequenceLength + j;
            attentionScores[scoreIdx] = score;
          }
        }
      }
    }

    // Apply softmax
    const attentionWeights = this.softmax(attentionScores, sequenceLength);

    // Apply attention weights to values
    const attendedValues = this.applyAttentionWeights(attentionWeights, VHeads, batchSize, sequenceLength);

    // Concatenate heads and project
    const concatenated = this.concatenateHeads(attendedValues, batchSize, sequenceLength);
    const output = this.matmul(concatenated, weights.output);

    // Apply dropout if training
    if (training && this.config.dropoutRate > 0) {
      return this.dropout(output, this.config.dropoutRate);
    }

    return output;
  }

  feedForward(input, layerIndex) {
    const weights = this.feedForwardWeights[layerIndex];

    // First linear transformation
    let hidden = this.matmul(input, weights.w1);
    hidden = this.addBias(hidden, weights.b1);

    // ReLU activation
    hidden = this.relu(hidden);

    // Second linear transformation
    let output = this.matmul(hidden, weights.w2);
    output = this.addBias(output, weights.b2);

    return output;
  }

  layerNorm(input, normParams) {
    const { shape } = input;
    const lastDim = shape[shape.length - 1];
    const normalized = new Float32Array(input.length);

    // Compute mean and variance for each position
    for (let i = 0; i < input.length / lastDim; i++) {
      let mean = 0;
      let variance = 0;

      // Calculate mean
      for (let j = 0; j < lastDim; j++) {
        mean += input[i * lastDim + j];
      }
      mean /= lastDim;

      // Calculate variance
      for (let j = 0; j < lastDim; j++) {
        const diff = input[i * lastDim + j] - mean;
        variance += diff * diff;
      }
      variance /= lastDim;

      // Normalize and apply scale/shift
      const std = Math.sqrt(variance + 1e-5);
      for (let j = 0; j < lastDim; j++) {
        const idx = i * lastDim + j;
        normalized[idx] = normParams.gamma[j] * ((input[idx] - mean) / std) + normParams.beta[j];
      }
    }

    normalized.shape = shape;
    return normalized;
  }

  async train(trainingData, options = {}) {
    const {
      epochs = 10,
      batchSize = 32,
      learningRate = 0.001,
      warmupSteps = 4000,
      validationSplit = 0.1,
    } = options;

    const trainingHistory = [];

    // Split data into training and validation
    const splitIndex = Math.floor(trainingData.length * (1 - validationSplit));
    const trainData = trainingData.slice(0, splitIndex);
    const valData = trainingData.slice(splitIndex);

    let globalStep = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let batchCount = 0;

      // Shuffle training data
      const shuffled = this.shuffle(trainData);

      // Process batches
      for (let i = 0; i < shuffled.length; i += batchSize) {
        const batch = shuffled.slice(i, Math.min(i + batchSize, shuffled.length));

        // Adaptive learning rate with warmup
        const currentLR = this.getAdaptiveLearningRate(learningRate, globalStep, warmupSteps);

        // Forward pass
        const predictions = await this.forward(batch.inputs, true);

        // Calculate loss
        const loss = this.crossEntropyLoss(predictions, batch.targets);
        epochLoss += loss;

        // Backward pass (simplified)
        await this.backward(loss, currentLR);

        globalStep++;
        batchCount++;
      }

      // Validation
      const valLoss = await this.validate(valData);

      const avgTrainLoss = epochLoss / batchCount;
      trainingHistory.push({
        epoch: epoch + 1,
        trainLoss: avgTrainLoss,
        valLoss,
        learningRate: this.getAdaptiveLearningRate(learningRate, globalStep, warmupSteps),
      });

      console.log(`Epoch ${epoch + 1}/${epochs} - Train Loss: ${avgTrainLoss.toFixed(4)}, Val Loss: ${valLoss.toFixed(4)}`);
    }

    return {
      history: trainingHistory,
      finalLoss: trainingHistory[trainingHistory.length - 1].trainLoss,
      modelType: 'transformer',
    };
  }

  getAdaptiveLearningRate(baseLR, step, warmupSteps) {
    // Learning rate schedule with warmup (as in original Transformer paper)
    const arg1 = Math.sqrt(step);
    const arg2 = step * Math.pow(warmupSteps, -1.5);
    const lr = baseLR * Math.min(arg1, arg2) * Math.sqrt(this.config.dimensions);
    return lr;
  }

  // Utility functions
  tokenEmbedding(tokenIndices) {
    // Simplified token embedding - in practice would use learned embeddings
    const embedded = new Float32Array(tokenIndices.shape[0] * tokenIndices.shape[1] * this.config.dimensions);

    for (let b = 0; b < tokenIndices.shape[0]; b++) {
      for (let s = 0; s < tokenIndices.shape[1]; s++) {
        for (let d = 0; d < this.config.dimensions; d++) {
          const idx = b * tokenIndices.shape[1] * this.config.dimensions +
                     s * this.config.dimensions + d;
          // Simple embedding based on token index
          embedded[idx] = (tokenIndices[b * tokenIndices.shape[1] + s] % this.config.vocabularySize) /
                         this.config.vocabularySize + (Math.random() - 0.5) * 0.1;
        }
      }
    }

    embedded.shape = [tokenIndices.shape[0], tokenIndices.shape[1], this.config.dimensions];
    return embedded;
  }

  addPositionalEncoding(embeddings, sequenceLength) {
    const result = new Float32Array(embeddings.length);

    for (let b = 0; b < embeddings.shape[0]; b++) {
      for (let s = 0; s < sequenceLength; s++) {
        for (let d = 0; d < this.config.dimensions; d++) {
          const embIdx = b * sequenceLength * this.config.dimensions +
                        s * this.config.dimensions + d;
          const posIdx = s * this.config.dimensions + d;

          result[embIdx] = embeddings[embIdx] + this.positionalEncoding[posIdx];
        }
      }
    }

    result.shape = embeddings.shape;
    return result;
  }

  reshapeForHeads(tensor, batchSize, sequenceLength) {
    // Reshape to [batch, heads, sequence, head_dimension]
    const reshaped = new Float32Array(tensor.length);

    for (let b = 0; b < batchSize; b++) {
      for (let s = 0; s < sequenceLength; s++) {
        for (let h = 0; h < this.config.heads; h++) {
          for (let d = 0; d < this.headDimension; d++) {
            const srcIdx = b * sequenceLength * this.config.dimensions +
                          s * this.config.dimensions +
                          h * this.headDimension + d;
            const dstIdx = b * this.config.heads * sequenceLength * this.headDimension +
                          h * sequenceLength * this.headDimension +
                          s * this.headDimension + d;

            reshaped[dstIdx] = tensor[srcIdx];
          }
        }
      }
    }

    return reshaped;
  }

  concatenateHeads(tensor, batchSize, sequenceLength) {
    // Reshape from [batch, heads, sequence, head_dimension] to [batch, sequence, dimensions]
    const concatenated = new Float32Array(batchSize * sequenceLength * this.config.dimensions);

    for (let b = 0; b < batchSize; b++) {
      for (let s = 0; s < sequenceLength; s++) {
        for (let h = 0; h < this.config.heads; h++) {
          for (let d = 0; d < this.headDimension; d++) {
            const srcIdx = b * this.config.heads * sequenceLength * this.headDimension +
                          h * sequenceLength * this.headDimension +
                          s * this.headDimension + d;
            const dstIdx = b * sequenceLength * this.config.dimensions +
                          s * this.config.dimensions +
                          h * this.headDimension + d;

            concatenated[dstIdx] = tensor[srcIdx];
          }
        }
      }
    }

    concatenated.shape = [batchSize, sequenceLength, this.config.dimensions];
    return concatenated;
  }

  softmax(scores, sequenceLength) {
    const softmaxScores = new Float32Array(scores.length);

    // Apply softmax per attention head and query position
    const stride = sequenceLength;

    for (let i = 0; i < scores.length; i += stride) {
      let maxScore = -Infinity;

      // Find max for numerical stability
      for (let j = 0; j < stride; j++) {
        maxScore = Math.max(maxScore, scores[i + j]);
      }

      // Compute exp and sum
      let sumExp = 0;
      for (let j = 0; j < stride; j++) {
        softmaxScores[i + j] = Math.exp(scores[i + j] - maxScore);
        sumExp += softmaxScores[i + j];
      }

      // Normalize
      for (let j = 0; j < stride; j++) {
        softmaxScores[i + j] /= sumExp;
      }
    }

    return softmaxScores;
  }

  applyAttentionWeights(weights, values, batchSize, sequenceLength) {
    const output = new Float32Array(batchSize * this.config.heads * sequenceLength * this.headDimension);

    for (let b = 0; b < batchSize; b++) {
      for (let h = 0; h < this.config.heads; h++) {
        for (let i = 0; i < sequenceLength; i++) {
          for (let d = 0; d < this.headDimension; d++) {
            let sum = 0;

            for (let j = 0; j < sequenceLength; j++) {
              const weightIdx = b * this.config.heads * sequenceLength * sequenceLength +
                               h * sequenceLength * sequenceLength +
                               i * sequenceLength + j;
              const valueIdx = b * this.config.heads * sequenceLength * this.headDimension +
                              h * sequenceLength * this.headDimension +
                              j * this.headDimension + d;

              sum += weights[weightIdx] * values[valueIdx];
            }

            const outIdx = b * this.config.heads * sequenceLength * this.headDimension +
                          h * sequenceLength * this.headDimension +
                          i * this.headDimension + d;
            output[outIdx] = sum;
          }
        }
      }
    }

    return output;
  }

  outputProjection(input) {
    // Project to vocabulary size
    return this.matmul(input, this.outputWeights.projection);
  }

  getConfig() {
    return {
      type: 'transformer',
      ...this.config,
      parameters: this.countParameters(),
    };
  }

  countParameters() {
    let count = 0;

    // Attention weights
    for (let layer = 0; layer < this.config.layers; layer++) {
      count += 4 * this.config.dimensions * this.config.dimensions; // Q, K, V, O projections
    }

    // Feed-forward weights
    count += this.config.layers * (
      this.config.dimensions * this.config.ffDimensions * 2 + // W1, W2
      this.config.ffDimensions + this.config.dimensions // biases
    );

    // Layer norm parameters
    count += this.config.layers * 2 * this.config.dimensions; // gamma, beta

    // Output projection
    count += this.config.dimensions * this.config.vocabularySize + this.config.vocabularySize;

    return count;
  }
}

export { TransformerModel };