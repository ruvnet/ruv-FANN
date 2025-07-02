/**
 * Unit tests for Neural Agent module
 */

import { NeuralAgent,
  NeuralAgentFactory,
  NeuralNetwork,
  COGNITIVE_PATTERNS,
  AGENT_COGNITIVE_PROFILES,
} from '../../../src/neural-agent';
import assert from 'assert';
const EventEmitter = require('events');

// Mock base agent for testing
class MockAgent {
  constructor() {
    this.id = 'mock-agent-123';
    this.type = 'researcher';
    this.status = 'idle';
    this.capabilities = ['research', 'analysis'];
  }

  async execute(task) {
    return {
      success: true,
      result: 'Mock execution result',
      metrics: {
        linesOfCode: 50,
        testsPass: 0.9,
      },
    };
  }
}

describe('NeuralNetwork Tests', () => {
  let network;
  const config = {
    networkLayers: [4, 8, 4, 2],
    activationFunction: 'sigmoid',
    learningRate: 0.5,
    momentum: 0.2,
  };

  beforeEach(() => {
    network = new NeuralNetwork(config);
  });

  describe('Initialization', () => {
    it('should initialize with correct configuration', () => {
      assert.deepStrictEqual(network.layers, config.networkLayers);
      assert.strictEqual(network.activationFunction, config.activationFunction);
      assert.strictEqual(network.learningRate, config.learningRate);
      assert.strictEqual(network.momentum, config.momentum);
    });

    it('should initialize weights and biases', () => {
      assert.strictEqual(network.weights.length, 3); // 4 layers = 3 weight matrices
      assert.strictEqual(network.biases.length, 3);
      assert.strictEqual(network.previousWeightDeltas.length, 3);

      // Check dimensions
      assert.strictEqual(network.weights[0].length, 8); // 8 neurons in layer 2
      assert.strictEqual(network.weights[0][0].length, 4); // 4 inputs from layer 1
    });
  });

  describe('Activation Functions', () => {
    it('should compute sigmoid activation', () => {
      const nn = new NeuralNetwork({ ...config, activationFunction: 'sigmoid' });
      const result = nn._activation(0);
      assert.strictEqual(result, 0.5);

      const derivative = nn._activation(0, true);
      assert.strictEqual(derivative, 0.25);
    });

    it('should compute tanh activation', () => {
      const nn = new NeuralNetwork({ ...config, activationFunction: 'tanh' });
      const result = nn._activation(0);
      assert.strictEqual(result, 0);

      const derivative = nn._activation(0, true);
      assert.strictEqual(derivative, 1);
    });

    it('should compute relu activation', () => {
      const nn = new NeuralNetwork({ ...config, activationFunction: 'relu' });
      assert.strictEqual(nn._activation(-1), 0);
      assert.strictEqual(nn._activation(1), 1);

      assert.strictEqual(nn._activation(-1, true), 0);
      assert.strictEqual(nn._activation(1, true), 1);
    });
  });

  describe('Forward Propagation', () => {
    it('should perform forward propagation', () => {
      const input = [0.5, 0.3, 0.2, 0.8];
      const result = network.forward(input);

      assert(result.output);
      assert(result.activations);
      assert.strictEqual(result.output.length, 2); // Final layer has 2 neurons
      assert.strictEqual(result.activations.length, 4); // 4 layers total
      assert.strictEqual(result.activations[0], input); // First activation is input
    });

    it('should handle different input sizes', () => {
      const input = [1, 0, 0, 1];
      const result = network.forward(input);
      assert.strictEqual(result.output.length, 2);
      result.output.forEach(val => {
        assert(val >= 0 && val <= 1); // Sigmoid outputs
      });
    });
  });

  describe('Training', () => {
    it('should train the network', () => {
      const input = [0.5, 0.3, 0.2, 0.8];
      const target = [0.7, 0.3];

      const output = network.train(input, target);
      assert(Array.isArray(output));
      assert.strictEqual(output.length, target.length);
    });

    it('should use custom learning rate', () => {
      const input = [0.5, 0.3, 0.2, 0.8];
      const target = [0.7, 0.3];

      const output = network.train(input, target, 0.1);
      assert(Array.isArray(output));
    });

    it('should update weights during training', () => {
      const input = [0.5, 0.3, 0.2, 0.8];
      const target = [0.7, 0.3];

      // Get initial weights
      const initialWeight = network.weights[0][0][0];

      // Train
      network.train(input, target);

      // Check weights changed
      assert.notStrictEqual(network.weights[0][0][0], initialWeight);
    });
  });

  describe('Save and Load', () => {
    it('should save network state', () => {
      const savedState = network.save();
      assert(savedState.config);
      assert(savedState.weights);
      assert(savedState.biases);
      assert.deepStrictEqual(savedState.config, config);
    });

    it('should load network state', () => {
      const savedState = network.save();

      // Create new network and load state
      const newNetwork = new NeuralNetwork(config);
      newNetwork.load(savedState);

      assert.deepStrictEqual(newNetwork.weights, savedState.weights);
      assert.deepStrictEqual(newNetwork.biases, savedState.biases);
    });
  });
});

describe('NeuralAgent Tests', () => {
  let mockAgent, neuralAgent;

  beforeEach(() => {
    mockAgent = new MockAgent();
    neuralAgent = new NeuralAgent(mockAgent, 'researcher');
  });

  describe('Initialization', () => {
    it('should initialize with correct properties', () => {
      assert.strictEqual(neuralAgent.agent, mockAgent);
      assert.strictEqual(neuralAgent.agentType, 'researcher');
      assert(neuralAgent.cognitiveProfile);
      assert(neuralAgent.neuralNetwork instanceof NeuralNetwork);
      assert(neuralAgent instanceof EventEmitter);
    });

    it('should initialize cognitive state', () => {
      assert.deepStrictEqual(neuralAgent.cognitiveState, {
        attention: 1.0,
        fatigue: 0.0,
        confidence: 0.5,
        exploration: 0.5,
      });
    });

    it('should initialize performance metrics', () => {
      assert.deepStrictEqual(neuralAgent.performanceMetrics, {
        accuracy: 0,
        speed: 0,
        creativity: 0,
        efficiency: 0,
      });
    });
  });

  describe('Task Analysis', () => {
    it('should analyze task', async() => {
      const task = {
        description: 'Analyze the data and create a report',
        priority: 'high',
        dependencies: ['data-collection'],
      };

      const analysis = await neuralAgent.analyzeTask(task);
      assert(analysis);
      assert('complexity' in analysis);
      assert('urgency' in analysis);
      assert('creativity' in analysis);
      assert('dataIntensity' in analysis);
      assert('collaborationNeeded' in analysis);
      assert('confidence' in analysis);
    });

    it('should apply cognitive pattern to analysis', async() => {
      const task = { description: 'Create innovative solution' };
      const analysis = await neuralAgent.analyzeTask(task);

      // Researcher has DIVERGENT primary pattern, should boost creativity
      assert(analysis.creativity > 0);
    });
  });

  describe('Task Execution', () => {
    it('should execute task with neural enhancement', async() => {
      const task = {
        id: 'task-123',
        description: 'Research new algorithms',
        priority: 'medium',
      };

      const result = await neuralAgent.executeTask(task);
      assert(result);
      assert.strictEqual(result.success, true);
    });

    it('should emit taskCompleted event', (done) => {
      const task = {
        id: 'task-123',
        description: 'Test task',
        priority: 'low',
      };

      neuralAgent.once('taskCompleted', (event) => {
        assert.deepStrictEqual(event.task, task);
        assert(event.result);
        assert(event.performance);
        assert(event.cognitiveState);
        done();
      });

      neuralAgent.executeTask(task);
    });

    it('should update cognitive state after execution', async() => {
      const task = {
        description: 'Complex research task',
        priority: 'high',
      };

      const initialFatigue = neuralAgent.cognitiveState.fatigue;
      await neuralAgent.executeTask(task);

      // Fatigue should increase after complex task
      assert(neuralAgent.cognitiveState.fatigue > initialFatigue);
    });

    it('should track learning history', async() => {
      const task = {
        id: 'task-123',
        description: 'Learn from this task',
      };

      assert.strictEqual(neuralAgent.learningHistory.length, 0);
      await neuralAgent.executeTask(task);
      assert.strictEqual(neuralAgent.learningHistory.length, 1);
    });
  });

  describe('Task Vector Conversion', () => {
    it('should convert task to vector', () => {
      const task = {
        description: 'This is a Test Task with Numbers 123',
        priority: 'high',
        dependencies: ['task1', 'task2'],
      };

      const vector = neuralAgent._taskToVector(task);
      assert(Array.isArray(vector));
      assert.strictEqual(vector.length, neuralAgent.neuralNetwork.layers[0]);
      vector.forEach(val => {
        assert(typeof val === 'number');
      });
    });

    it('should handle missing task properties', () => {
      const task = {};
      const vector = neuralAgent._taskToVector(task);
      assert(Array.isArray(vector));
      assert.strictEqual(vector.length, neuralAgent.neuralNetwork.layers[0]);
    });
  });

  describe('Performance Calculation', () => {
    it('should calculate performance metrics', () => {
      const task = { id: 'task-123' };
      const result = { success: true, metrics: { linesOfCode: 50, testsPass: 0.9 } };
      const executionTime = 30000; // 30 seconds

      const performance = neuralAgent._calculatePerformance(task, result, executionTime);
      assert(performance);
      assert(performance.speed > 0 && performance.speed <= 1);
      assert.strictEqual(performance.accuracy, 0.9);
      assert(performance.overall > 0 && performance.overall <= 1);
    });

    it('should handle failed tasks', () => {
      const task = { id: 'task-123' };
      const result = { success: false };
      const executionTime = 5000;

      const performance = neuralAgent._calculatePerformance(task, result, executionTime);
      assert.strictEqual(performance.accuracy, 0.2);
    });
  });

  describe('Similar Task Finding', () => {
    beforeEach(() => {
      // Add some task history
      neuralAgent.taskHistory = [
        {
          task: { description: 'Research machine learning', priority: 'high' },
          performance: { overall: 0.8 },
        },
        {
          task: { description: 'Analyze data patterns', priority: 'medium' },
          performance: { overall: 0.7 },
        },
        {
          task: { description: 'Code review', priority: 'low' },
          performance: { overall: 0.9 },
        },
      ];
    });

    it('should find similar tasks', () => {
      const newTask = { description: 'Research deep learning', priority: 'high' };
      const similar = neuralAgent._findSimilarTasks(newTask);

      assert(similar.length > 0);
      // Should find the machine learning task as most similar
      assert(similar[0].task.description.includes('machine learning'));
    });

    it('should limit similar tasks returned', () => {
      const newTask = { description: 'Generic task' };
      const similar = neuralAgent._findSimilarTasks(newTask, 2);
      assert(similar.length <= 2);
    });

    it('should handle empty history', () => {
      neuralAgent.taskHistory = [];
      const newTask = { description: 'New task' };
      const similar = neuralAgent._findSimilarTasks(newTask);
      assert.strictEqual(similar.length, 0);
    });
  });

  describe('Cognitive Pattern Application', () => {
    it('should apply convergent pattern', () => {
      const analysis = {
        complexity: 1.0,
        confidence: 0.5,
        creativity: 0.5,
      };

      const coderAgent = new NeuralAgent(mockAgent, 'coder'); // Has convergent primary
      coderAgent._applyCognitivePattern(analysis);

      assert(analysis.complexity < 1.0); // Should simplify
      assert(analysis.confidence > 0.5); // Should increase confidence
    });

    it('should apply divergent pattern', () => {
      const analysis = {
        creativity: 0.5,
        exploration: 0.5,
      };

      neuralAgent._applyCognitivePattern(analysis); // Researcher has divergent
      assert(analysis.creativity > 0.5); // Should boost creativity
    });

    it('should apply secondary pattern', () => {
      const analysis = {
        collaborationNeeded: 0.5,
        dataIntensity: 0.5,
      };

      // Researcher has systems as secondary
      neuralAgent._applyCognitivePattern(analysis);
      assert(analysis.collaborationNeeded > 0.5);
    });
  });

  describe('Rest Functionality', () => {
    it('should reduce fatigue when resting', async() => {
      // Increase fatigue first
      neuralAgent.cognitiveState.fatigue = 0.8;
      neuralAgent.cognitiveState.attention = 0.4;

      await neuralAgent.rest(100);

      assert(neuralAgent.cognitiveState.fatigue < 0.8);
      assert(neuralAgent.cognitiveState.attention > 0.4);
    });
  });

  describe('Status and State Management', () => {
    it('should get complete status', () => {
      const status = neuralAgent.getStatus();
      assert(status);
      assert(status.neuralState);
      assert(status.neuralState.cognitiveProfile);
      assert(status.neuralState.cognitiveState);
      assert(status.neuralState.performanceMetrics);
    });

    it('should save neural state', () => {
      // Add some history
      neuralAgent.learningHistory.push({
        timestamp: Date.now(),
        task: 'test',
        performance: 0.8,
      });

      const state = neuralAgent.saveNeuralState();
      assert.strictEqual(state.agentType, 'researcher');
      assert(state.neuralNetwork);
      assert(state.cognitiveState);
      assert(state.learningHistory.length > 0);
    });

    it('should load neural state', () => {
      const savedState = {
        neuralNetwork: neuralAgent.neuralNetwork.save(),
        cognitiveState: { attention: 0.8, fatigue: 0.2, confidence: 0.7, exploration: 0.3 },
        performanceMetrics: { accuracy: 0.9, speed: 0.8, creativity: 0.7, efficiency: 0.85 },
        learningHistory: [{ timestamp: Date.now(), task: 'old-task', performance: 0.75 }],
        taskHistory: [],
      };

      neuralAgent.loadNeuralState(savedState);
      assert.deepStrictEqual(neuralAgent.cognitiveState, savedState.cognitiveState);
      assert.deepStrictEqual(neuralAgent.performanceMetrics, savedState.performanceMetrics);
      assert.strictEqual(neuralAgent.learningHistory.length, 1);
    });
  });

  describe('Learning Event', () => {
    it('should emit learning event after task execution', (done) => {
      const task = {
        id: 'learning-task',
        description: 'Task to learn from',
      };

      neuralAgent.once('learning', (event) => {
        assert.strictEqual(event.task, task.id);
        assert(event.performance);
        assert(event.networkState);
        done();
      });

      neuralAgent.executeTask(task);
    });
  });

  describe('Performance Metrics Update', () => {
    it('should update performance metrics with exponential moving average', () => {
      neuralAgent.performanceMetrics = {
        accuracy: 0.5,
        speed: 0.5,
        creativity: 0.5,
        efficiency: 0.5,
      };

      const newPerformance = {
        accuracy: 1.0,
        speed: 1.0,
        creativity: 1.0,
        efficiency: 1.0,
      };

      neuralAgent._updatePerformanceMetrics(newPerformance);

      // Should be between old and new values
      assert(neuralAgent.performanceMetrics.accuracy > 0.5);
      assert(neuralAgent.performanceMetrics.accuracy < 1.0);
    });
  });
});

describe('NeuralAgentFactory Tests', () => {
  it('should create neural agent for valid agent type', () => {
    const mockAgent = new MockAgent();
    const neuralAgent = NeuralAgentFactory.createNeuralAgent(mockAgent, 'researcher');
    assert(neuralAgent instanceof NeuralAgent);
    assert.strictEqual(neuralAgent.agentType, 'researcher');
  });

  it('should throw error for invalid agent type', () => {
    const mockAgent = new MockAgent();
    assert.throws(() => {
      NeuralAgentFactory.createNeuralAgent(mockAgent, 'invalid-type');
    }, /Unknown agent type/);
  });

  it('should provide cognitive profiles', () => {
    const profiles = NeuralAgentFactory.getCognitiveProfiles();
    assert(profiles);
    assert(profiles.researcher);
    assert(profiles.coder);
    assert(profiles.analyst);
    assert(profiles.optimizer);
    assert(profiles.coordinator);
  });

  it('should provide cognitive patterns', () => {
    const patterns = NeuralAgentFactory.getCognitivePatterns();
    assert.deepStrictEqual(patterns, COGNITIVE_PATTERNS);
  });
});

describe('Cognitive Profiles Tests', () => {
  it('should have valid profiles for all agent types', () => {
    Object.entries(AGENT_COGNITIVE_PROFILES).forEach(([agentType, profile]) => {
      assert(profile.primary);
      assert(profile.secondary);
      assert(typeof profile.learningRate === 'number');
      assert(typeof profile.momentum === 'number');
      assert(Array.isArray(profile.networkLayers));
      assert(profile.activationFunction);
    });
  });

  it('should have valid cognitive patterns', () => {
    const validPatterns = Object.values(COGNITIVE_PATTERNS);
    Object.values(AGENT_COGNITIVE_PROFILES).forEach(profile => {
      assert(validPatterns.includes(profile.primary));
      assert(validPatterns.includes(profile.secondary));
    });
  });
});

// Run tests when this file is executed directly
if (require.main === module) {
  console.log('Running Neural Agent Unit Tests...');
  require('../../../node_modules/.bin/jest');
}