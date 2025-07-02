/**
 * Base Neural Model Class
 * Abstract base class for all neural network models
 */

class NeuralModel {
  constructor(modelType) {
    this.modelType = modelType;
    this.isInitialized = false;
    this.trainingHistory = [];
    this.metrics = {
      accuracy: 0,
      loss: 1.0,
      epochsTrained: 0,
      totalSamples: 0
    };
  }

  // Abstract methods to be implemented by subclasses
  async forward(input, training = false) {
    throw new Error('forward() must be implemented by subclass');
  }

  async train(trainingData, options = {}) {
    throw new Error('train() must be implemented by subclass');
  }

  async backward(loss, learningRate) {
    // Default backward pass - can be overridden
    console.log(`Backward pass for ${this.modelType} with loss: ${loss}`);
    return true;
  }

  async validate(validationData) {
    let totalLoss = 0;
    let batchCount = 0;
    
    for (const batch of validationData) {
      const predictions = await this.forward(batch.inputs, false);
      const loss = this.crossEntropyLoss(predictions, batch.targets);
      totalLoss += loss;
      batchCount++;
    }
    
    return totalLoss / batchCount;
  }

  // Common utility methods
  matmul(a, b) {
    // Matrix multiplication helper
    // Assumes a is [m, n] and b is [n, p]
    if (!a.shape || !b.shape || a.shape.length < 2 || b.shape.length < 2) {
      throw new Error('Invalid matrix dimensions for multiplication');
    }
    
    const m = a.shape[0];
    const n = a.shape[1];
    const p = b.shape[b.shape.length - 1];
    
    const result = new Float32Array(m * p);
    
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += a[i * n + k] * b[k * p + j];
        }
        result[i * p + j] = sum;
      }
    }
    
    result.shape = [m, p];
    return result;
  }

  add(a, b) {
    // Element-wise addition
    if (a.length !== b.length) {
      throw new Error('Tensors must have same length for addition');
    }
    
    const result = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
      result[i] = a[i] + b[i];
    }
    
    result.shape = a.shape;
    return result;
  }

  addBias(input, bias) {
    // Add bias to last dimension
    const result = new Float32Array(input.length);
    const lastDim = bias.length;
    
    for (let i = 0; i < input.length; i++) {
      result[i] = input[i] + bias[i % lastDim];
    }
    
    result.shape = input.shape;
    return result;
  }

  relu(input) {
    // ReLU activation
    const result = new Float32Array(input.length);
    
    for (let i = 0; i < input.length; i++) {
      result[i] = Math.max(0, input[i]);
    }
    
    result.shape = input.shape;
    return result;
  }

  sigmoid(input) {
    // Sigmoid activation
    const result = new Float32Array(input.length);
    
    for (let i = 0; i < input.length; i++) {
      result[i] = 1 / (1 + Math.exp(-input[i]));
    }
    
    result.shape = input.shape;
    return result;
  }

  tanh(input) {
    // Tanh activation
    const result = new Float32Array(input.length);
    
    for (let i = 0; i < input.length; i++) {
      result[i] = Math.tanh(input[i]);
    }
    
    result.shape = input.shape;
    return result;
  }

  dropout(input, rate) {
    // Apply dropout during training
    if (rate <= 0) return input;
    
    const result = new Float32Array(input.length);
    const scale = 1 / (1 - rate);
    
    for (let i = 0; i < input.length; i++) {
      if (Math.random() > rate) {
        result[i] = input[i] * scale;
      } else {
        result[i] = 0;
      }
    }
    
    result.shape = input.shape;
    return result;
  }

  crossEntropyLoss(predictions, targets) {
    // Cross-entropy loss for classification
    let loss = 0;
    const epsilon = 1e-7; // For numerical stability
    
    for (let i = 0; i < predictions.length; i++) {
      const pred = Math.max(epsilon, Math.min(1 - epsilon, predictions[i]));
      if (targets[i] === 1) {
        loss -= Math.log(pred);
      } else {
        loss -= Math.log(1 - pred);
      }
    }
    
    return loss / predictions.length;
  }

  meanSquaredError(predictions, targets) {
    // MSE loss for regression
    let loss = 0;
    
    for (let i = 0; i < predictions.length; i++) {
      const diff = predictions[i] - targets[i];
      loss += diff * diff;
    }
    
    return loss / predictions.length;
  }

  shuffle(array) {
    // Fisher-Yates shuffle
    const shuffled = [...array];
    
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    
    return shuffled;
  }

  // Model persistence methods
  async save(filePath) {
    const modelData = {
      modelType: this.modelType,
      config: this.getConfig(),
      weights: this.getWeights(),
      metrics: this.metrics,
      trainingHistory: this.trainingHistory
    };
    
    // In a real implementation, save to file
    console.log(`Saving ${this.modelType} model to ${filePath}`);
    return modelData;
  }

  async load(filePath) {
    // In a real implementation, load from file
    console.log(`Loading ${this.modelType} model from ${filePath}`);
    return true;
  }

  getWeights() {
    // To be overridden by subclasses
    return {};
  }

  setWeights(weights) {
    // To be overridden by subclasses
    console.log(`Setting weights for ${this.modelType}`);
  }

  getConfig() {
    // To be overridden by subclasses
    return {
      modelType: this.modelType
    };
  }

  getMetrics() {
    return {
      ...this.metrics,
      modelType: this.modelType,
      trainingHistory: this.trainingHistory
    };
  }

  updateMetrics(loss, accuracy = null) {
    this.metrics.loss = loss;
    if (accuracy !== null) {
      this.metrics.accuracy = accuracy;
    }
    this.metrics.epochsTrained++;
  }

  reset() {
    // Reset model to initial state
    this.trainingHistory = [];
    this.metrics = {
      accuracy: 0,
      loss: 1.0,
      epochsTrained: 0,
      totalSamples: 0
    };
    this.initializeWeights();
  }
}

export { NeuralModel };