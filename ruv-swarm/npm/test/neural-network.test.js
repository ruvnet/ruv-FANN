// neural-network.test.js - Tests for neural network WASM integration

import { jest } from '@jest/globals';
import {
  createNeuralNetwork,
  createTrainer,
  createAgentNeuralManager,
  ActivationFunctions,
  initializeNeuralWasm,
  ACTIVATION_FUNCTIONS,
  TRAINING_ALGORITHMS,
  COGNITIVE_PATTERNS,
} from '../src/neural-network.js';

describe('Neural Network WASM Integration', () => {
  let wasm;

  beforeAll(async() => {
    wasm = await initializeNeuralWasm();
  });

  describe('Basic Neural Network', () => {
    test('should create a neural network', async() => {
      const network = await createNeuralNetwork({
        inputSize: 2,
        hiddenLayers: [
          { size: 3, activation: ACTIVATION_FUNCTIONS.SIGMOID },
        ],
        outputSize: 1,
        outputActivation: ACTIVATION_FUNCTIONS.SIGMOID,
      });

      expect(network).toBeDefined();

      const info = network.getInfo();
      expect(info.numInputs).toBe(2);
      expect(info.numOutputs).toBe(1);
      expect(info.numLayers).toBeGreaterThanOrEqual(3); // input, hidden, output
    });

    test('should run inference', async() => {
      const network = await createNeuralNetwork({
        inputSize: 2,
        hiddenLayers: [{ size: 3, activation: ACTIVATION_FUNCTIONS.RELU }],
        outputSize: 1,
        outputActivation: ACTIVATION_FUNCTIONS.SIGMOID,
      });

      const output = await network.run([0.5, 0.5]);
      expect(output).toHaveLength(1);
      expect(output[0]).toBeGreaterThanOrEqual(0);
      expect(output[0]).toBeLessThanOrEqual(1);
    });

    test('should get and set weights', async() => {
      const network = await createNeuralNetwork({
        inputSize: 2,
        hiddenLayers: [{ size: 2, activation: ACTIVATION_FUNCTIONS.SIGMOID }],
        outputSize: 1,
        outputActivation: ACTIVATION_FUNCTIONS.SIGMOID,
      });

      const weights = network.getWeights();
      expect(weights).toBeInstanceOf(Float32Array);
      expect(weights.length).toBeGreaterThan(0);

      // Modify weights
      const newWeights = new Float32Array(weights.length);
      for (let i = 0; i < weights.length; i++) {
        newWeights[i] = Math.random() * 2 - 1;
      }

      network.setWeights(newWeights);
      const retrievedWeights = network.getWeights();
      expect(retrievedWeights).toEqual(newWeights);
    });
  });

  describe('Training Algorithms', () => {
    test.each([
      TRAINING_ALGORITHMS.INCREMENTAL_BACKPROP,
      TRAINING_ALGORITHMS.BATCH_BACKPROP,
      TRAINING_ALGORITHMS.RPROP,
      TRAINING_ALGORITHMS.QUICKPROP,
      TRAINING_ALGORITHMS.SARPROP,
    ])('should create trainer with %s algorithm', async(algorithm) => {
      const trainer = await createTrainer({
        algorithm,
        maxEpochs: 100,
        targetError: 0.01,
      });

      expect(trainer).toBeDefined();
      const info = trainer.getAlgorithmInfo();
      expect(info.name).toBeDefined();
      expect(info.type).toBeDefined();
    });

    test('should train XOR problem', async() => {
      const network = await createNeuralNetwork({
        inputSize: 2,
        hiddenLayers: [
          { size: 4, activation: ACTIVATION_FUNCTIONS.SIGMOID },
          { size: 3, activation: ACTIVATION_FUNCTIONS.SIGMOID },
        ],
        outputSize: 1,
        outputActivation: ACTIVATION_FUNCTIONS.SIGMOID,
      });

      const trainer = await createTrainer({
        algorithm: TRAINING_ALGORITHMS.RPROP,
        maxEpochs: 500,
        targetError: 0.01,
      });

      const trainingData = {
        inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
        outputs: [[0], [1], [1], [0]],
      };

      const result = await trainer.trainUntilTarget(network, trainingData, 0.01, 500);
      expect(result.converged).toBe(true);
      expect(result.finalError).toBeLessThan(0.01);

      // Test predictions
      const predictions = await Promise.all(
        trainingData.inputs.map(input => network.run(input)),
      );

      // Check XOR logic
      expect(predictions[0][0]).toBeLessThan(0.5); // [0,0] => 0
      expect(predictions[1][0]).toBeGreaterThan(0.5); // [0,1] => 1
      expect(predictions[2][0]).toBeGreaterThan(0.5); // [1,0] => 1
      expect(predictions[3][0]).toBeLessThan(0.5); // [1,1] => 0
    });
  });

  describe('Activation Functions', () => {
    test('should list all 18 activation functions', async() => {
      const functions = await ActivationFunctions.getAll(wasm);
      expect(functions).toHaveLength(18);

      const functionNames = functions.map(([name]) => name);
      expect(functionNames).toContain('sigmoid');
      expect(functionNames).toContain('relu');
      expect(functionNames).toContain('tanh');
      expect(functionNames).toContain('gaussian');
    });

    test('should test activation functions', async() => {
      const sigmoid = await ActivationFunctions.test(wasm, 'sigmoid', 0);
      expect(sigmoid).toBeCloseTo(0.5, 5);

      const relu = await ActivationFunctions.test(wasm, 'relu', -1);
      expect(relu).toBe(0);

      const relu2 = await ActivationFunctions.test(wasm, 'relu', 1);
      expect(relu2).toBe(1);
    });

    test('should compare activation functions', async() => {
      const comparison = await ActivationFunctions.compare(wasm, 0);
      expect(comparison).toBeDefined();
      expect(comparison.sigmoid).toBeCloseTo(0.5, 5);
      expect(comparison.tanh).toBeCloseTo(0, 5);
      expect(comparison.relu).toBe(0);
    });

    test('should get activation function properties', async() => {
      const props = await ActivationFunctions.getProperties(wasm, 'sigmoid');
      expect(props.name).toBe('Sigmoid');
      expect(props.trainable).toBe(true);
      expect(props.output_range.min).toBe('0');
      expect(props.output_range.max).toBe('1');
    });
  });

  describe('Agent Neural Networks', () => {
    let manager;

    beforeEach(async() => {
      manager = await createAgentNeuralManager();
    });

    test('should create agent networks with cognitive patterns', async() => {
      const agentConfig = {
        agentId: 'test-agent-001',
        agentType: 'researcher',
        cognitivePattern: COGNITIVE_PATTERNS.DIVERGENT,
        inputSize: 10,
        outputSize: 5,
        taskSpecialization: ['pattern_recognition'],
      };

      const agentId = await manager.createAgentNetwork(agentConfig);
      expect(agentId).toBe('test-agent-001');

      const state = await manager.getAgentCognitiveState(agentId);
      expect(state.agentId).toBe('test-agent-001');
      expect(state.cognitivePattern.pattern_type).toBe('divergent');
      expect(state.neuralArchitecture.layers).toBeGreaterThan(0);
    });

    test('should support multiple agents with different patterns', async() => {
      const patterns = Object.values(COGNITIVE_PATTERNS);
      const agentIds = [];

      for (const pattern of patterns) {
        const agentId = `agent-${pattern}`;
        await manager.createAgentNetwork({
          agentId,
          agentType: 'test',
          cognitivePattern: pattern,
          inputSize: 5,
          outputSize: 3,
        });
        agentIds.push(agentId);
      }

      // Verify each agent has correct pattern
      for (let i = 0; i < patterns.length; i++) {
        const state = await manager.getAgentCognitiveState(agentIds[i]);
        expect(state.cognitivePattern.pattern_type).toBe(patterns[i]);
      }
    });

    test('should perform inference for agents', async() => {
      const agentId = await manager.createAgentNetwork({
        agentId: 'inference-test',
        agentType: 'analyst',
        cognitivePattern: COGNITIVE_PATTERNS.CRITICAL,
        inputSize: 4,
        outputSize: 2,
      });

      const input = [0.1, 0.2, 0.3, 0.4];
      const output = await manager.getAgentInference(agentId, input);

      expect(output).toHaveLength(2);
      expect(output[0]).toBeGreaterThanOrEqual(0);
      expect(output[1]).toBeGreaterThanOrEqual(0);
    });

    test('should train agent networks', async() => {
      const agentId = await manager.createAgentNetwork({
        agentId: 'training-test',
        agentType: 'coder',
        cognitivePattern: COGNITIVE_PATTERNS.CONVERGENT,
        inputSize: 3,
        outputSize: 2,
      });

      const trainingData = {
        inputs: Array(10).fill(null).map(() => [Math.random(), Math.random(), Math.random()]),
        outputs: Array(10).fill(null).map(() => [Math.random() > 0.5 ? 1 : 0, Math.random() > 0.5 ? 1 : 0]),
      };

      const result = await manager.trainAgentNetwork(agentId, trainingData);
      expect(result).toBeDefined();
      expect(result.epochs).toBeGreaterThan(0);
      expect(result.final_loss).toBeDefined();
    });

    test('should support online adaptation', async() => {
      const agentId = await manager.createAgentNetwork({
        agentId: 'adaptation-test',
        agentType: 'optimizer',
        cognitivePattern: COGNITIVE_PATTERNS.LATERAL,
        inputSize: 5,
        outputSize: 3,
      });

      const experienceData = {
        inputs: [[0.1, 0.2, 0.3, 0.4, 0.5]],
        expected_outputs: [[1, 0, 1]],
        actual_outputs: [[0.8, 0.2, 0.9]],
        rewards: [0.85],
        context: { task: 'optimization' },
      };

      const result = await manager.fineTuneDuringExecution(agentId, experienceData);
      expect(result).toBeDefined();
      expect(result.adapted).toBe(true);
    });
  });

  describe('Performance and Memory', () => {
    test('should handle 100+ simultaneous agent networks', async() => {
      const manager = await createAgentNeuralManager();
      const agentIds = [];

      // Create 100 agents
      for (let i = 0; i < 100; i++) {
        const patterns = Object.values(COGNITIVE_PATTERNS);
        const pattern = patterns[i % patterns.length];

        const agentId = await manager.createAgentNetwork({
          agentId: `perf-agent-${i}`,
          agentType: 'test',
          cognitivePattern: pattern,
          inputSize: 10,
          outputSize: 5,
        });

        agentIds.push(agentId);
      }

      expect(agentIds).toHaveLength(100);

      // Test inference on all agents
      const testInput = Array(10).fill(0.5);
      const inferencePromises = agentIds.map(id =>
        manager.getAgentInference(id, testInput),
      );

      const results = await Promise.all(inferencePromises);
      expect(results).toHaveLength(100);
      results.forEach(output => {
        expect(output).toHaveLength(5);
      });
    });

    test('should measure inference performance', async() => {
      const network = await createNeuralNetwork({
        inputSize: 100,
        hiddenLayers: [
          { size: 50, activation: ACTIVATION_FUNCTIONS.RELU },
          { size: 25, activation: ACTIVATION_FUNCTIONS.RELU },
        ],
        outputSize: 10,
        outputActivation: ACTIVATION_FUNCTIONS.SOFTMAX,
      });

      const input = Array(100).fill(0).map(() => Math.random());

      const startTime = performance.now();
      const iterations = 1000;

      for (let i = 0; i < iterations; i++) {
        await network.run(input);
      }

      const endTime = performance.now();
      const avgTime = (endTime - startTime) / iterations;

      console.log(`Average inference time: ${avgTime.toFixed(3)}ms`);
      expect(avgTime).toBeLessThan(10); // Should be fast
    });
  });
});