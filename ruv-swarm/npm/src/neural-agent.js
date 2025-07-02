/**
 * Neural Agent Module - Integrates ruv-FANN neural network capabilities
 * into agent processing for cognitive diversity and learning
 */

import { EventEmitter } from 'events';

// Import these after class definitions to avoid circular dependency
let MemoryOptimizer, PATTERN_MEMORY_CONFIG;

// Cognitive diversity patterns for different agent types
const COGNITIVE_PATTERNS = {
  CONVERGENT: 'convergent', // Focused problem-solving, analytical
  DIVERGENT: 'divergent', // Creative exploration, idea generation
  LATERAL: 'lateral', // Non-linear thinking, pattern breaking
  SYSTEMS: 'systems', // Holistic view, interconnections
  CRITICAL: 'critical', // Evaluation, judgment, validation
  ABSTRACT: 'abstract', // Conceptual thinking, generalization
};

// Agent type to cognitive pattern mapping
const AGENT_COGNITIVE_PROFILES = {
  researcher: {
    primary: COGNITIVE_PATTERNS.DIVERGENT,
    secondary: COGNITIVE_PATTERNS.SYSTEMS,
    learningRate: 0.7,
    momentum: 0.3,
    networkLayers: [64, 128, 64, 32],
    activationFunction: 'sigmoid',
    advancedModel: 'transformer_nlp', // Use transformer for research tasks
  },
  coder: {
    primary: COGNITIVE_PATTERNS.CONVERGENT,
    secondary: COGNITIVE_PATTERNS.LATERAL,
    learningRate: 0.5,
    momentum: 0.2,
    networkLayers: [128, 256, 128, 64],
    activationFunction: 'relu',
    advancedModel: 'gru_sequence', // Use GRU for code generation
  },
  analyst: {
    primary: COGNITIVE_PATTERNS.CRITICAL,
    secondary: COGNITIVE_PATTERNS.ABSTRACT,
    learningRate: 0.6,
    momentum: 0.25,
    networkLayers: [96, 192, 96, 48],
    activationFunction: 'tanh',
    advancedModel: 'cnn_vision', // Use CNN for pattern analysis
  },
  optimizer: {
    primary: COGNITIVE_PATTERNS.SYSTEMS,
    secondary: COGNITIVE_PATTERNS.CONVERGENT,
    learningRate: 0.4,
    momentum: 0.35,
    networkLayers: [80, 160, 80, 40],
    activationFunction: 'sigmoid',
  },
  coordinator: {
    primary: COGNITIVE_PATTERNS.SYSTEMS,
    secondary: COGNITIVE_PATTERNS.CRITICAL,
    learningRate: 0.55,
    momentum: 0.3,
    networkLayers: [112, 224, 112, 56],
    activationFunction: 'relu',
  },
};

/**
 * Neural Network wrapper for agent cognitive processing
 */
class NeuralNetwork {
  constructor(config, memoryOptimizer = null) {
    this.config = config;
    this.layers = config.networkLayers;
    this.activationFunction = config.activationFunction;
    this.learningRate = config.learningRate;
    this.momentum = config.momentum;
    this.memoryOptimizer = memoryOptimizer;

    // Memory-optimized storage
    this.weights = [];
    this.biases = [];
    this.previousWeightDeltas = [];
    this.memoryAllocations = [];

    this._initializeNetwork();
  }

  _initializeNetwork() {
    // Initialize weights and biases between layers with memory optimization
    for (let i = 0; i < this.layers.length - 1; i++) {
      const inputSize = this.layers[i];
      const outputSize = this.layers[i + 1];

      // Xavier/Glorot initialization
      const limit = Math.sqrt(6 / (inputSize + outputSize));

      // Try to allocate from memory pool if available
      if (this.memoryOptimizer && this.memoryOptimizer.isPoolInitialized()) {
        const weightSize = outputSize * inputSize * 4; // 4 bytes per float32
        const biasSize = outputSize * 4;

        const weightAlloc = this.memoryOptimizer.allocateFromPool('weights', weightSize, this.config.cognitivePattern || 'default');
        const biasAlloc = this.memoryOptimizer.allocateFromPool('weights', biasSize, this.config.cognitivePattern || 'default');

        if (weightAlloc && biasAlloc) {
          this.memoryAllocations.push(weightAlloc, biasAlloc);
        }
      }

      // Create matrices (in optimized implementation, these would use pooled memory)
      this.weights[i] = this._createMatrix(outputSize, inputSize, -limit, limit);
      this.biases[i] = this._createVector(outputSize, -0.1, 0.1);
      this.previousWeightDeltas[i] = this._createMatrix(outputSize, inputSize, 0, 0);
    }
  }

  _createMatrix(rows, cols, min, max) {
    const matrix = [];
    for (let i = 0; i < rows; i++) {
      matrix[i] = [];
      for (let j = 0; j < cols; j++) {
        matrix[i][j] = Math.random() * (max - min) + min;
      }
    }
    return matrix;
  }

  _createVector(size, min, max) {
    const vector = [];
    for (let i = 0; i < size; i++) {
      vector[i] = Math.random() * (max - min) + min;
    }
    return vector;
  }

  _activation(x, derivative = false) {
    switch (this.activationFunction) {
    case 'sigmoid':
      if (derivative) {
        const sig = 1 / (1 + Math.exp(-x));
        return sig * (1 - sig);
      }
      return 1 / (1 + Math.exp(-x));

    case 'tanh':
      if (derivative) {
        const tanh = Math.tanh(x);
        return 1 - tanh * tanh;
      }
      return Math.tanh(x);

    case 'relu':
      if (derivative) {
        return x > 0 ? 1 : 0;
      }
      return Math.max(0, x);

    default:
      return x;
    }
  }

  forward(input) {
    const activations = [input];
    let currentInput = input;

    // Forward propagation through layers
    for (let i = 0; i < this.weights.length; i++) {
      const weights = this.weights[i];
      const biases = this.biases[i];
      const output = [];

      for (let j = 0; j < weights.length; j++) {
        let sum = biases[j];
        for (let k = 0; k < currentInput.length; k++) {
          sum += weights[j][k] * currentInput[k];
        }
        output[j] = this._activation(sum);
      }

      activations.push(output);
      currentInput = output;
    }

    return {
      output: currentInput,
      activations,
    };
  }

  train(input, target, learningRate = null) {
    const lr = learningRate || this.learningRate;
    const { activations } = this.forward(input);

    // Backward propagation
    const errors = [];
    const output = activations[activations.length - 1];

    // Calculate output layer error
    const outputError = [];
    for (let i = 0; i < output.length; i++) {
      outputError[i] = (target[i] - output[i]) * this._activation(output[i], true);
    }
    errors.unshift(outputError);

    // Backpropagate errors
    for (let i = this.weights.length - 1; i > 0; i--) {
      const layerError = [];
      const weights = this.weights[i];
      const prevError = errors[0];

      for (let j = 0; j < this.weights[i - 1].length; j++) {
        let error = 0;
        for (let k = 0; k < weights.length; k++) {
          error += weights[k][j] * prevError[k];
        }
        layerError[j] = error * this._activation(activations[i][j], true);
      }
      errors.unshift(layerError);
    }

    // Update weights and biases
    for (let i = 0; i < this.weights.length; i++) {
      const weights = this.weights[i];
      const biases = this.biases[i];
      const layerError = errors[i + 1];
      const layerInput = activations[i];

      for (let j = 0; j < weights.length; j++) {
        // Update bias
        biases[j] += lr * layerError[j];

        // Update weights with momentum
        for (let k = 0; k < weights[j].length; k++) {
          const delta = lr * layerError[j] * layerInput[k];
          const momentumDelta = this.momentum * this.previousWeightDeltas[i][j][k];
          weights[j][k] += delta + momentumDelta;
          this.previousWeightDeltas[i][j][k] = delta;
        }
      }
    }

    return output;
  }

  save() {
    return {
      config: this.config,
      weights: this.weights,
      biases: this.biases,
    };
  }

  load(data) {
    this.weights = data.weights;
    this.biases = data.biases;
  }
}

/**
 * Neural Agent class that enhances base agents with neural network capabilities
 */
class NeuralAgent extends EventEmitter {
  constructor(agent, agentType, memoryOptimizer = null) {
    super();
    this.agent = agent;
    this.agentType = agentType;
    this.cognitiveProfile = AGENT_COGNITIVE_PROFILES[agentType];
    this.memoryOptimizer = memoryOptimizer || new MemoryOptimizer();

    // Add cognitive pattern to neural network config for memory optimization
    const networkConfig = {
      ...this.cognitiveProfile,
      cognitivePattern: this.cognitiveProfile.primary,
    };

    // Initialize neural network with memory optimizer
    this.neuralNetwork = new NeuralNetwork(networkConfig, this.memoryOptimizer);

    // Learning history for feedback loops
    this.learningHistory = [];
    this.taskHistory = [];
    this.performanceMetrics = {
      accuracy: 0,
      speed: 0,
      creativity: 0,
      efficiency: 0,
      memoryEfficiency: 0,
    };

    // Cognitive state
    this.cognitiveState = {
      attention: 1.0,
      fatigue: 0.0,
      confidence: 0.5,
      exploration: 0.5,
    };

    // Track memory usage
    this.memoryUsage = {
      baseline: 0,
      current: 0,
      peak: 0,
    };

    this._initializeMemoryTracking();
  }

  /**
   * Process task through neural network for intelligent routing
   */
  async analyzeTask(task) {
    // Convert task to neural input vector
    const inputVector = this._taskToVector(task);

    // Get neural network prediction
    const { output } = this.neuralNetwork.forward(inputVector);

    // Interpret output for task routing
    const analysis = {
      complexity: output[0],
      urgency: output[1],
      creativity: output[2],
      dataIntensity: output[3],
      collaborationNeeded: output[4],
      confidence: output[5],
    };

    // Apply cognitive pattern influence
    this._applyCognitivePattern(analysis);

    return analysis;
  }

  /**
   * Execute task with neural enhancement
   */
  async executeTask(task) {
    const startTime = Date.now();

    // Analyze task
    const analysis = await this.analyzeTask(task);

    // Adjust cognitive state based on task
    this._updateCognitiveState(analysis);

    // Execute base agent task
    const result = await this.agent.execute({
      ...task,
      neuralAnalysis: analysis,
      cognitiveState: this.cognitiveState,
    });

    // Calculate performance
    const executionTime = Date.now() - startTime;
    const performance = this._calculatePerformance(task, result, executionTime);

    // Learn from the experience
    await this._learnFromExecution(task, result, performance);

    // Emit events for monitoring
    this.emit('taskCompleted', {
      task,
      result,
      performance,
      cognitiveState: this.cognitiveState,
    });

    return result;
  }

  /**
   * Convert task to neural network input vector
   */
  _taskToVector(task) {
    const vector = [];

    // Task description features (simplified for example)
    const description = task.description || '';
    vector.push(
      description.length / 1000, // Length normalized
      (description.match(/\b\w+\b/g) || []).length / 100, // Word count
      (description.match(/[A-Z]/g) || []).length / description.length, // Capitalization ratio
      (description.match(/[0-9]/g) || []).length / description.length, // Numeric ratio
    );

    // Task metadata
    const priorityMap = { low: 0.2, medium: 0.5, high: 0.8, critical: 1.0 };
    vector.push(priorityMap[task.priority] || 0.5);

    // Dependencies
    vector.push(Math.min(task.dependencies?.length || 0, 10) / 10);

    // Historical performance on similar tasks
    const similarTasks = this._findSimilarTasks(task);
    if (similarTasks.length > 0) {
      const avgPerformance = similarTasks.reduce((sum, t) => sum + t.performance.overall, 0) / similarTasks.length;
      vector.push(avgPerformance);
    } else {
      vector.push(0.5); // Neutral if no history
    }

    // Current cognitive state influence
    vector.push(
      this.cognitiveState.attention,
      this.cognitiveState.fatigue,
      this.cognitiveState.confidence,
      this.cognitiveState.exploration,
    );

    // Pad or truncate to expected input size
    const inputSize = this.neuralNetwork.layers[0];
    while (vector.length < inputSize) {
      vector.push(0);
    }
    return vector.slice(0, inputSize);
  }

  /**
   * Apply cognitive pattern to analysis
   */
  _applyCognitivePattern(analysis) {
    const primary = this.cognitiveProfile.primary;
    const secondary = this.cognitiveProfile.secondary;

    switch (primary) {
    case COGNITIVE_PATTERNS.CONVERGENT:
      analysis.complexity *= 0.9; // Simplify through focus
      analysis.confidence *= 1.1; // Higher confidence in solutions
      break;

    case COGNITIVE_PATTERNS.DIVERGENT:
      analysis.creativity *= 1.2; // Boost creative requirements
      analysis.exploration = 0.8; // High exploration tendency
      break;

    case COGNITIVE_PATTERNS.LATERAL:
      analysis.creativity *= 1.15; // Enhance creative thinking
      analysis.complexity *= 1.05; // See hidden complexity
      break;

    case COGNITIVE_PATTERNS.SYSTEMS:
      analysis.collaborationNeeded *= 1.2; // See interconnections
      analysis.dataIntensity *= 1.1; // Process more context
      break;

    case COGNITIVE_PATTERNS.CRITICAL:
      analysis.confidence *= 0.9; // More cautious
      analysis.complexity *= 1.1; // See more edge cases
      break;

    case COGNITIVE_PATTERNS.ABSTRACT:
      analysis.complexity *= 0.95; // Simplify through abstraction
      analysis.creativity *= 1.05; // Abstract thinking is creative
      break;
    }

    // Apply secondary pattern with lesser influence
    this._applySecondaryPattern(analysis, secondary);
  }

  /**
   * Update cognitive state based on task execution
   */
  _updateCognitiveState(analysis) {
    // Fatigue increases with complexity
    this.cognitiveState.fatigue = Math.min(
      this.cognitiveState.fatigue + analysis.complexity * 0.1,
      1.0,
    );

    // Attention decreases with fatigue
    this.cognitiveState.attention = Math.max(
      1.0 - this.cognitiveState.fatigue * 0.5,
      0.3,
    );

    // Confidence adjusts based on recent performance
    if (this.learningHistory.length > 0) {
      const recentPerformance = this.learningHistory.slice(-5)
        .reduce((sum, h) => sum + h.performance, 0) / Math.min(this.learningHistory.length, 5);
      this.cognitiveState.confidence = 0.3 + recentPerformance * 0.7;
    }

    // Exploration vs exploitation balance
    this.cognitiveState.exploration = 0.2 + (1.0 - this.cognitiveState.confidence) * 0.6;
  }

  /**
   * Calculate performance metrics
   */
  _calculatePerformance(task, result, executionTime) {
    const performance = {
      speed: Math.max(0, 1 - (executionTime / 60000)), // Normalize to 1 minute
      accuracy: result.success ? 0.8 : 0.2,
      creativity: 0.5, // Default, should be evaluated based on result
      efficiency: 0.5,
      overall: 0.5,
    };

    // Adjust based on result quality indicators
    if (result.metrics) {
      if (result.metrics.linesOfCode) {
        performance.efficiency = Math.min(1.0, 100 / result.metrics.linesOfCode);
      }
      if (result.metrics.testsPass) {
        performance.accuracy = result.metrics.testsPass;
      }
    }

    // Calculate overall performance
    performance.overall = (
      performance.speed * 0.2 +
      performance.accuracy * 0.4 +
      performance.creativity * 0.2 +
      performance.efficiency * 0.2
    );

    return performance;
  }

  /**
   * Learn from task execution
   */
  async _learnFromExecution(task, result, performance) {
    // Prepare training data
    const input = this._taskToVector(task);
    const target = [
      performance.overall,
      performance.speed,
      performance.accuracy,
      performance.creativity,
      performance.efficiency,
      result.success ? 1.0 : 0.0,
    ];

    // Train neural network
    this.neuralNetwork.train(input, target);

    // Store in learning history
    this.learningHistory.push({
      timestamp: Date.now(),
      task: task.id,
      performance: performance.overall,
      input,
      target,
    });

    // Keep history size manageable
    if (this.learningHistory.length > 1000) {
      this.learningHistory = this.learningHistory.slice(-500);
    }

    // Update performance metrics
    this._updatePerformanceMetrics(performance);

    // Emit learning event
    this.emit('learning', {
      task: task.id,
      performance,
      networkState: this.neuralNetwork.save(),
    });
  }

  /**
   * Update overall performance metrics
   */
  _updatePerformanceMetrics(performance) {
    const alpha = 0.1; // Learning rate for exponential moving average

    this.performanceMetrics.accuracy =
      (1 - alpha) * this.performanceMetrics.accuracy + alpha * performance.accuracy;
    this.performanceMetrics.speed =
      (1 - alpha) * this.performanceMetrics.speed + alpha * performance.speed;
    this.performanceMetrics.creativity =
      (1 - alpha) * this.performanceMetrics.creativity + alpha * performance.creativity;
    this.performanceMetrics.efficiency =
      (1 - alpha) * this.performanceMetrics.efficiency + alpha * performance.efficiency;

    // Calculate memory efficiency based on task completion vs memory usage
    const memoryRatio = this.memoryUsage.baseline / this.getCurrentMemoryUsage();
    const taskEfficiency = performance.overall;
    this.performanceMetrics.memoryEfficiency =
      (1 - alpha) * this.performanceMetrics.memoryEfficiency + alpha * (memoryRatio * taskEfficiency);
  }

  /**
   * Find similar tasks from history
   */
  _findSimilarTasks(task, limit = 5) {
    if (this.taskHistory.length === 0) {
      return [];
    }

    // Simple similarity based on task properties
    const similarities = this.taskHistory.map(historicalTask => {
      let similarity = 0;

      // Priority match
      if (historicalTask.task.priority === task.priority) {
        similarity += 0.3;
      }

      // Description similarity (simple word overlap)
      const currentWords = new Set((task.description || '').toLowerCase().split(/\s+/));
      const historicalWords = new Set((historicalTask.task.description || '').toLowerCase().split(/\s+/));
      const intersection = new Set([...currentWords].filter(x => historicalWords.has(x)));
      const union = new Set([...currentWords, ...historicalWords]);
      if (union.size > 0) {
        similarity += 0.7 * (intersection.size / union.size);
      }

      return {
        task: historicalTask,
        similarity,
      };
    });

    // Return top similar tasks
    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit)
      .filter(s => s.similarity > 0.3)
      .map(s => s.task);
  }

  /**
   * Apply secondary cognitive pattern
   */
  _applySecondaryPattern(analysis, pattern) {
    const influence = 0.5; // Secondary patterns have less influence

    switch (pattern) {
    case COGNITIVE_PATTERNS.CONVERGENT:
      analysis.complexity *= (1 - influence * 0.1);
      analysis.confidence *= (1 + influence * 0.1);
      break;

    case COGNITIVE_PATTERNS.DIVERGENT:
      analysis.creativity *= (1 + influence * 0.2);
      break;

    case COGNITIVE_PATTERNS.LATERAL:
      analysis.creativity *= (1 + influence * 0.15);
      break;

    case COGNITIVE_PATTERNS.SYSTEMS:
      analysis.collaborationNeeded *= (1 + influence * 0.2);
      break;

    case COGNITIVE_PATTERNS.CRITICAL:
      analysis.confidence *= (1 - influence * 0.1);
      break;

    case COGNITIVE_PATTERNS.ABSTRACT:
      analysis.complexity *= (1 - influence * 0.05);
      break;
    }
  }

  /**
   * Rest the agent to reduce fatigue
   */
  rest(duration = 1000) {
    return new Promise((resolve) => {
      setTimeout(async() => {
        this.cognitiveState.fatigue = Math.max(0, this.cognitiveState.fatigue - 0.3);
        this.cognitiveState.attention = Math.min(1.0, this.cognitiveState.attention + 0.2);

        // Perform garbage collection on memory pools during rest
        if (this.memoryOptimizer && this.memoryOptimizer.isPoolInitialized()) {
          const collected = await this.memoryOptimizer.garbageCollect();
          if (collected > 0) {
            // Recalculate memory usage after GC
            const patternConfig = PATTERN_MEMORY_CONFIG[this.cognitiveProfile.primary];
            this.memoryUsage.current = patternConfig.baseMemory * (1 - patternConfig.poolSharing * 0.5);
          }
        }

        resolve();
      }, duration);
    });
  }

  /**
   * Initialize memory tracking for the agent
   */
  _initializeMemoryTracking() {
    const patternConfig = PATTERN_MEMORY_CONFIG[this.cognitiveProfile.primary] || PATTERN_MEMORY_CONFIG.convergent;
    this.memoryUsage.baseline = patternConfig.baseMemory;
    this.memoryUsage.current = patternConfig.baseMemory;

    // Initialize memory pools if not already done
    if (!this.memoryOptimizer.isPoolInitialized()) {
      this.memoryOptimizer.initializePools().then(() => {
        // Recalculate memory usage with pooling
        this.memoryUsage.current = patternConfig.baseMemory * (1 - patternConfig.poolSharing * 0.5);
      });
    }
  }

  /**
   * Get current memory usage for this agent
   */
  getCurrentMemoryUsage() {
    // const patternConfig = PATTERN_MEMORY_CONFIG[this.cognitiveProfile.primary] || PATTERN_MEMORY_CONFIG.convergent;
    let memoryUsage = this.memoryUsage.current;

    // Adjust based on current activity
    if (this.cognitiveState.fatigue > 0.5) {
      memoryUsage *= 1.1; // 10% more memory when fatigued
    }

    if (this.taskHistory.length > 100) {
      memoryUsage *= 1.05; // 5% more for large history
    }

    // Update peak if necessary
    if (memoryUsage > this.memoryUsage.peak) {
      this.memoryUsage.peak = memoryUsage;
    }

    return memoryUsage;
  }

  /**
   * Get agent status including neural state
   */
  getStatus() {
    return {
      ...this.agent,
      neuralState: {
        cognitiveProfile: this.cognitiveProfile,
        cognitiveState: this.cognitiveState,
        performanceMetrics: this.performanceMetrics,
        learningHistory: this.learningHistory.length,
        taskHistory: this.taskHistory.length,
        memoryUsage: {
          current: `${this.getCurrentMemoryUsage().toFixed(0) } MB`,
          baseline: `${this.memoryUsage.baseline.toFixed(0) } MB`,
          peak: `${this.memoryUsage.peak.toFixed(0) } MB`,
          efficiency: this.performanceMetrics.memoryEfficiency.toFixed(2),
        },
      },
    };
  }

  /**
   * Save neural state for persistence
   */
  saveNeuralState() {
    return {
      agentType: this.agentType,
      neuralNetwork: this.neuralNetwork.save(),
      cognitiveState: this.cognitiveState,
      performanceMetrics: this.performanceMetrics,
      learningHistory: this.learningHistory.slice(-100), // Keep recent history
      taskHistory: this.taskHistory.slice(-100),
    };
  }

  /**
   * Load neural state from saved data
   */
  loadNeuralState(data) {
    if (data.neuralNetwork) {
      this.neuralNetwork.load(data.neuralNetwork);
    }
    if (data.cognitiveState) {
      this.cognitiveState = data.cognitiveState;
    }
    if (data.performanceMetrics) {
      this.performanceMetrics = data.performanceMetrics;
    }
    if (data.learningHistory) {
      this.learningHistory = data.learningHistory;
    }
    if (data.taskHistory) {
      this.taskHistory = data.taskHistory;
    }
  }
}

/**
 * Neural Agent Factory
 */
class NeuralAgentFactory {
  static memoryOptimizer = null;

  static async initializeFactory() {
    if (!this.memoryOptimizer) {
      this.memoryOptimizer = new MemoryOptimizer();
      await this.memoryOptimizer.initializePools();
    }
  }

  static createNeuralAgent(baseAgent, agentType) {
    if (!AGENT_COGNITIVE_PROFILES[agentType]) {
      throw new Error(`Unknown agent type: ${agentType}`);
    }

    // Use shared memory optimizer for all agents
    return new NeuralAgent(baseAgent, agentType, this.memoryOptimizer);
  }

  static getCognitiveProfiles() {
    return AGENT_COGNITIVE_PROFILES;
  }

  static getCognitivePatterns() {
    return COGNITIVE_PATTERNS;
  }
}

// Lazy load to avoid circular dependency
setImmediate(() => {
  import('./neural.js').then(neural => {
    MemoryOptimizer = neural.MemoryOptimizer;
    PATTERN_MEMORY_CONFIG = neural.PATTERN_MEMORY_CONFIG;
  });
});

export {
  NeuralAgent,
  NeuralAgentFactory,
  NeuralNetwork,
  COGNITIVE_PATTERNS,
  AGENT_COGNITIVE_PROFILES,
};