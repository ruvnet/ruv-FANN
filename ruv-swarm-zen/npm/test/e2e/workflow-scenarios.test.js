/**
 * End-to-End Workflow Tests
 * Tests complete user scenarios from initialization to results
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { RuvSwarm } from '../../src/index-enhanced.js';
import { EnhancedMCPTools } from '../../src/mcp-tools-enhanced.js';
import fs from 'fs/promises';
import path from 'path';

describe('E2E Workflow Scenarios', () => {
  let ruvSwarm;
  let mcpTools;
  let testDir;

  beforeAll(async() => {
    // Create test directory for outputs
    testDir = path.join(process.cwd(), 'test-outputs', `e2e-${Date.now()}`);
    await fs.mkdir(testDir, { recursive: true });

    // Initialize system
    ruvSwarm = await RuvSwarm.initialize({
      loadingStrategy: 'full',
      enablePersistence: true,
      enableNeuralNetworks: true,
      enableForecasting: true,
      useSIMD: true,
      persistencePath: path.join(testDir, 'persistence.db'),
    });

    mcpTools = new EnhancedMCPTools();
    await mcpTools.initialize();
  });

  afterAll(async() => {
    if (ruvSwarm) {
      await ruvSwarm.cleanup();
    }
    // Clean up test directory
    await fs.rm(testDir, { recursive: true, force: true });
  });

  describe('Machine Learning Pipeline Workflow', () => {
    it('should complete full ML pipeline from data to predictions', async() => {
      console.log('\n🔬 Starting ML Pipeline Workflow...');

      // Step 1: Create swarm for ML tasks
      const mlSwarm = await ruvSwarm.createSwarm({
        name: 'ml-pipeline-swarm',
        topology: 'hierarchical',
        maxAgents: 8,
      });

      // Step 2: Spawn specialized agents
      const agents = {
        dataPrep: await mlSwarm.spawn({
          type: 'researcher',
          capabilities: ['data-preprocessing', 'feature-engineering'],
        }),
        modelBuilder: await mlSwarm.spawn({
          type: 'coder',
          capabilities: ['neural-network-design', 'optimization'],
        }),
        trainer: await mlSwarm.spawn({
          type: 'analyst',
          capabilities: ['training', 'hyperparameter-tuning'],
        }),
        evaluator: await mlSwarm.spawn({
          type: 'tester',
          capabilities: ['model-evaluation', 'metrics-analysis'],
        }),
      };

      // Step 3: Generate synthetic dataset
      console.log('📊 Generating dataset...');
      const dataset = await agents.dataPrep.execute({
        task: 'generate-dataset',
        config: {
          samples: 1000,
          features: 20,
          targetType: 'classification',
          classes: 5,
          noise: 0.1,
        },
      });

      expect(dataset.inputs).toHaveLength(1000);
      expect(dataset.targets).toHaveLength(1000);
      expect(dataset.inputs[0]).toHaveLength(20);

      // Step 4: Build neural network
      console.log('🏗️ Building neural network...');
      const modelConfig = await agents.modelBuilder.execute({
        task: 'design-network',
        requirements: {
          inputSize: dataset.features,
          outputSize: dataset.classes,
          taskType: 'classification',
          complexity: 'medium',
        },
      });

      const network = await ruvSwarm.neuralManager.createNetwork(modelConfig);
      expect(network.id).toBeDefined();

      // Step 5: Train the model
      console.log('🎯 Training model...');
      const trainingResult = await agents.trainer.execute({
        task: 'train-model',
        networkId: network.id,
        dataset,
        config: {
          epochs: 50,
          batchSize: 32,
          learningRate: 0.001,
          validationSplit: 0.2,
          earlyStoppingPatience: 5,
        },
      });

      expect(trainingResult.finalLoss).toBeLessThan(trainingResult.initialLoss);
      expect(trainingResult.validationAccuracy).toBeGreaterThan(0.7);

      // Step 6: Evaluate model
      console.log('📈 Evaluating model...');
      const evaluation = await agents.evaluator.execute({
        task: 'evaluate-model',
        networkId: network.id,
        testData: dataset.test,
        metrics: ['accuracy', 'precision', 'recall', 'f1-score', 'confusion-matrix'],
      });

      expect(evaluation.accuracy).toBeGreaterThan(0.75);
      expect(evaluation.confusionMatrix).toHaveLength(5);

      // Step 7: Make predictions
      console.log('🔮 Making predictions...');
      const testSamples = Array(10).fill(null).map(() =>
        new Float32Array(20).map(() => Math.random()),
      );

      const predictions = await network.predict(testSamples);
      expect(predictions).toHaveLength(10);
      expect(predictions[0]).toHaveLength(5);

      // Step 8: Save model
      const modelPath = path.join(testDir, 'ml-model.ruv');
      await network.save(modelPath);

      const stats = await fs.stat(modelPath);
      expect(stats.size).toBeGreaterThan(0);

      console.log('✅ ML Pipeline completed successfully!');
    });
  });

  describe('Time Series Forecasting Workflow', () => {
    it('should forecast time series data using specialized models', async() => {
      console.log('\n📈 Starting Time Series Forecasting Workflow...');

      // Step 1: Create forecasting swarm
      const forecastSwarm = await ruvSwarm.createSwarm({
        name: 'forecast-swarm',
        topology: 'mesh',
        maxAgents: 6,
      });

      // Step 2: Generate time series data
      const timeSeriesData = [];
      const periods = 500;
      for (let i = 0; i < periods; i++) {
        timeSeriesData.push({
          timestamp: new Date(Date.now() - (periods - i) * 3600000),
          value: 100 + 50 * Math.sin(i * 0.1) + 20 * Math.random(),
          features: {
            dayOfWeek: i % 7,
            hour: i % 24,
            trend: i / periods,
          },
        });
      }

      // Step 3: Create forecasting pipeline
      const pipeline = await mcpTools.tools.orchestrateForecasting({
        swarmId: forecastSwarm.id,
        data: timeSeriesData,
        config: {
          models: ['lstm', 'transformer', 'nbeats'],
          horizon: 24,
          validationSplit: 0.2,
          ensembleMethod: 'weighted-average',
        },
      });

      // Step 4: Train models
      console.log('🧠 Training forecasting models...');
      const trainingResults = await pipeline.train({
        epochs: 30,
        patience: 5,
        onProgress: (model, epoch, metrics) => {
          console.log(`  ${model}: Epoch ${epoch}, Loss: ${metrics.loss.toFixed(4)}`);
        },
      });

      expect(trainingResults.lstm.finalMetrics.mae).toBeLessThan(10);
      expect(trainingResults.transformer.finalMetrics.mae).toBeLessThan(10);
      expect(trainingResults.nbeats.finalMetrics.mae).toBeLessThan(10);

      // Step 5: Generate forecasts
      console.log('🔮 Generating forecasts...');
      const forecasts = await pipeline.forecast({
        steps: 24,
        returnConfidenceIntervals: true,
        confidenceLevel: 0.95,
      });

      expect(forecasts.predictions).toHaveLength(24);
      expect(forecasts.lowerBound).toHaveLength(24);
      expect(forecasts.upperBound).toHaveLength(24);

      // Step 6: Evaluate forecast accuracy
      const evaluation = await pipeline.evaluate({
        actualValues: timeSeriesData.slice(-24).map(d => d.value),
        metrics: ['mae', 'rmse', 'mape', 'smape'],
      });

      expect(evaluation.mae).toBeLessThan(15);
      expect(evaluation.mape).toBeLessThan(0.15); // Less than 15% error

      console.log('✅ Forecasting workflow completed!');
    });
  });

  describe('Distributed Task Processing Workflow', () => {
    it('should process complex tasks across multiple agents', async() => {
      console.log('\n🚀 Starting Distributed Task Processing...');

      // Step 1: Create processing swarm
      const processingSwarm = await ruvSwarm.createSwarm({
        name: 'distributed-processing',
        topology: 'star',
        maxAgents: 10,
      });

      // Step 2: Define complex task
      const complexTask = {
        id: 'data-analysis-pipeline',
        stages: [
          {
            name: 'data-collection',
            subtasks: Array(50).fill(null).map((_, i) => ({
              id: `collect-${i}`,
              type: 'fetch',
              source: `dataset-${i}`,
              size: Math.floor(Math.random() * 1000000),
            })),
          },
          {
            name: 'data-processing',
            subtasks: Array(50).fill(null).map((_, i) => ({
              id: `process-${i}`,
              type: 'transform',
              operations: ['normalize', 'feature-extract', 'aggregate'],
              dependsOn: [`collect-${i}`],
            })),
          },
          {
            name: 'analysis',
            subtasks: Array(10).fill(null).map((_, i) => ({
              id: `analyze-${i}`,
              type: 'analyze',
              algorithms: ['statistical', 'ml-based'],
              dependsOn: Array(5).fill(null).map((_, j) => `process-${i * 5 + j}`),
            })),
          },
          {
            name: 'reporting',
            subtasks: [{
              id: 'final-report',
              type: 'aggregate',
              dependsOn: Array(10).fill(null).map((_, i) => `analyze-${i}`),
            }],
          },
        ],
      };

      // Step 3: Spawn agents dynamically based on workload
      console.log('🤖 Spawning agents...');
      const agentPool = [];
      for (let i = 0; i < 8; i++) {
        const agent = await processingSwarm.spawn({
          type: ['researcher', 'analyst', 'coder'][i % 3],
          capabilities: ['data-processing', 'parallel-execution'],
        });
        agentPool.push(agent);
      }

      // Step 4: Execute distributed processing
      console.log('⚡ Processing tasks...');
      const startTime = performance.now();

      const orchestrationResult = await processingSwarm.orchestrate({
        task: complexTask,
        strategy: 'parallel',
        monitoring: {
          interval: 100,
          onProgress: (progress) => {
            console.log(`  Progress: ${progress.completed}/${progress.total} tasks (${progress.percentage.toFixed(1)}%)`);
          },
        },
      });

      const duration = performance.now() - startTime;

      expect(orchestrationResult.completed).toBe(true);
      expect(orchestrationResult.tasksCompleted).toBe(111); // Total subtasks
      expect(orchestrationResult.duration).toBeLessThan(duration);

      // Step 5: Verify parallel execution efficiency
      const efficiency = orchestrationResult.parallelEfficiency;
      expect(efficiency).toBeGreaterThan(0.7); // At least 70% parallel efficiency

      // Step 6: Check agent utilization
      const utilization = await processingSwarm.getAgentUtilization();
      const avgUtilization = utilization.reduce((sum, u) => sum + u.utilization, 0) / utilization.length;
      expect(avgUtilization).toBeGreaterThan(0.6); // At least 60% average utilization

      console.log(`✅ Distributed processing completed in ${(duration / 1000).toFixed(2)}s`);
    });
  });

  describe('Real-time Collaboration Workflow', () => {
    it('should handle real-time collaborative editing scenario', async() => {
      console.log('\n👥 Starting Real-time Collaboration Workflow...');

      // Step 1: Create collaboration swarm
      const collabSwarm = await ruvSwarm.createSwarm({
        name: 'collab-swarm',
        topology: 'mesh',
        maxAgents: 5,
        enableRealtime: true,
      });

      // Step 2: Create shared document
      const document = {
        id: 'shared-doc-001',
        content: 'Initial document content\n',
        version: 0,
        operations: [],
      };

      // Step 3: Spawn collaborative agents
      const editors = await Promise.all([
        collabSwarm.spawn({ type: 'coder', role: 'editor-1' }),
        collabSwarm.spawn({ type: 'coder', role: 'editor-2' }),
        collabSwarm.spawn({ type: 'researcher', role: 'reviewer' }),
      ]);

      // Step 4: Simulate concurrent edits
      console.log('📝 Simulating concurrent edits...');
      const edits = [];
      const editPromises = [];

      // Editor 1 adds content
      editPromises.push(editors[0].execute({
        task: 'edit-document',
        operation: {
          type: 'insert',
          position: document.content.length,
          text: 'Section 1: Introduction\n',
        },
        documentId: document.id,
      }));

      // Editor 2 adds content concurrently
      editPromises.push(editors[1].execute({
        task: 'edit-document',
        operation: {
          type: 'insert',
          position: document.content.length,
          text: 'Section 2: Methods\n',
        },
        documentId: document.id,
      }));

      // Reviewer adds comments
      editPromises.push(editors[2].execute({
        task: 'add-comment',
        comment: {
          position: 0,
          text: 'Needs more detail in introduction',
        },
        documentId: document.id,
      }));

      const results = await Promise.all(editPromises);

      // Step 5: Verify conflict resolution
      expect(results.every(r => r.success)).toBe(true);
      expect(results.some(r => r.conflictResolved)).toBe(true);

      // Step 6: Check final document state
      const finalDoc = await collabSwarm.getSharedState(document.id);
      expect(finalDoc.content).toContain('Introduction');
      expect(finalDoc.content).toContain('Methods');
      expect(finalDoc.comments).toHaveLength(1);
      expect(finalDoc.version).toBeGreaterThan(0);

      console.log('✅ Collaboration workflow completed!');
    });
  });

  describe('Adaptive Learning Workflow', () => {
    it('should adapt agent behavior based on performance', async() => {
      console.log('\n🧬 Starting Adaptive Learning Workflow...');

      // Step 1: Create adaptive swarm
      const adaptiveSwarm = await ruvSwarm.createSwarm({
        name: 'adaptive-swarm',
        topology: 'hierarchical',
        maxAgents: 6,
        enableAdaptiveLearning: true,
      });

      // Step 2: Define performance metrics
      const performanceTracker = {
        agents: new Map(),
        taskTypes: ['optimization', 'search', 'analysis', 'synthesis'],
      };

      // Step 3: Run initial tasks and measure performance
      console.log('📊 Running baseline tasks...');
      const baselineResults = [];

      for (const taskType of performanceTracker.taskTypes) {
        const agent = await adaptiveSwarm.spawn({
          type: 'analyst',
          learningEnabled: true,
        });

        const result = await agent.execute({
          task: taskType,
          complexity: 'medium',
          measurePerformance: true,
        });

        baselineResults.push({
          agentId: agent.id,
          taskType,
          performance: result.performance,
        });

        performanceTracker.agents.set(agent.id, {
          agent,
          taskType,
          performances: [result.performance],
        });
      }

      // Step 4: Train agents through repeated tasks
      console.log('🎯 Training agents...');
      const trainingRounds = 10;

      for (let round = 0; round < trainingRounds; round++) {
        for (const [agentId, data] of performanceTracker.agents) {
          const result = await data.agent.execute({
            task: data.taskType,
            complexity: 'medium',
            learningEnabled: true,
            feedback: {
              previousPerformance: data.performances[data.performances.length - 1],
              targetImprovement: 0.05,
            },
          });

          data.performances.push(result.performance);
        }
      }

      // Step 5: Verify performance improvement
      console.log('📈 Analyzing improvements...');
      for (const [agentId, data] of performanceTracker.agents) {
        const initialPerf = data.performances[0];
        const finalPerf = data.performances[data.performances.length - 1];
        const improvement = (finalPerf.score - initialPerf.score) / initialPerf.score;

        expect(improvement).toBeGreaterThan(0.1); // At least 10% improvement
        console.log(`  Agent ${agentId}: ${(improvement * 100).toFixed(1)}% improvement`);
      }

      // Step 6: Test generalization
      console.log('🔄 Testing generalization...');
      const newTaskResults = [];

      for (const [agentId, data] of performanceTracker.agents) {
        // Test on a different task type
        const newTaskType = performanceTracker.taskTypes.find(t => t !== data.taskType);
        const result = await data.agent.execute({
          task: newTaskType,
          complexity: 'medium',
        });

        newTaskResults.push({
          agentId,
          trainedOn: data.taskType,
          testedOn: newTaskType,
          performance: result.performance.score,
        });
      }

      // Verify some knowledge transfer
      const avgNewTaskPerf = newTaskResults.reduce((sum, r) => sum + r.performance, 0) / newTaskResults.length;
      expect(avgNewTaskPerf).toBeGreaterThan(0.6); // Reasonable performance on new tasks

      console.log('✅ Adaptive learning workflow completed!');
    });
  });

  describe('Fault Tolerance Workflow', () => {
    it('should handle agent failures and recover gracefully', async() => {
      console.log('\n🛡️ Starting Fault Tolerance Workflow...');

      // Step 1: Create resilient swarm
      const resilientSwarm = await ruvSwarm.createSwarm({
        name: 'resilient-swarm',
        topology: 'mesh',
        maxAgents: 8,
        faultTolerance: {
          enabled: true,
          redundancy: 2,
          checkpointInterval: 1000,
        },
      });

      // Step 2: Create critical task with checkpoints
      const criticalTask = {
        id: 'critical-computation',
        steps: Array(20).fill(null).map((_, i) => ({
          id: `step-${i}`,
          computation: 'heavy',
          checkpointable: true,
        })),
      };

      // Step 3: Start task execution
      console.log('⚡ Starting critical task...');
      const agents = await Promise.all(
        Array(4).fill(null).map(() => resilientSwarm.spawn({ type: 'analyst' })),
      );

      let completedSteps = 0;
      const taskPromise = resilientSwarm.orchestrate({
        task: criticalTask,
        onStepComplete: (stepId) => {
          completedSteps++;
          console.log(`  Step ${stepId} completed (${completedSteps}/20)`);
        },
      });

      // Step 4: Simulate agent failures
      setTimeout(async() => {
        console.log('💥 Simulating agent failure...');
        await agents[0].simulateFailure();
      }, 2000);

      setTimeout(async() => {
        console.log('💥 Simulating another agent failure...');
        await agents[1].simulateFailure();
      }, 4000);

      // Step 5: Wait for task completion
      const result = await taskPromise;

      // Step 6: Verify task completed despite failures
      expect(result.completed).toBe(true);
      expect(result.stepsCompleted).toBe(20);
      expect(result.agentFailures).toBe(2);
      expect(result.recoveries).toBe(2);

      // Step 7: Check checkpoint usage
      const checkpointStats = await resilientSwarm.getCheckpointStats();
      expect(checkpointStats.checkpointsSaved).toBeGreaterThan(0);
      expect(checkpointStats.checkpointsRestored).toBeGreaterThan(0);

      console.log('✅ Fault tolerance workflow completed successfully!');
    });
  });
});
