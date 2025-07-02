/**
 * Integration tests for ruv-swarm API
 */

import { RuvSwarm } from '../../src/index-enhanced.js';
import { SwarmPersistence } from '../../src/persistence.js';
import { NeuralAgentFactory } from '../../src/neural-agent.js';
import assert from 'assert';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

describe('API Integration Tests', () => {
  let ruvSwarm;
  const testDbPath = path.join(__dirname, 'test-integration.db');

  beforeEach(async() => {
    // Clean up test database if exists
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }

    // Reset global state
    global._ruvSwarmInstance = null;
    global._ruvSwarmInitialized = 0;

    // Initialize with test database
    ruvSwarm = await RuvSwarm.initialize({
      enablePersistence: true,
      enableNeuralNetworks: true,
      enableForecasting: false,
      debug: false,
    });

    // Override persistence path for testing
    if (ruvSwarm.persistence) {
      ruvSwarm.persistence.close();
      ruvSwarm.persistence = new SwarmPersistence(testDbPath);
    }
  });

  afterEach(() => {
    if (ruvSwarm && ruvSwarm.persistence) {
      ruvSwarm.persistence.close();
    }
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  describe('End-to-End Swarm Workflow', () => {
    it('should complete full swarm lifecycle with persistence', async() => {
      // 1. Create swarm
      const swarm = await ruvSwarm.createSwarm({
        name: 'integration-test-swarm',
        topology: 'mesh',
        strategy: 'balanced',
        maxAgents: 10,
        enableCognitiveDiversity: true,
        enableNeuralAgents: true,
      });

      assert(swarm);
      assert(swarm.id);

      // 2. Verify swarm was persisted
      const persistedSwarms = ruvSwarm.persistence.getActiveSwarms();
      assert.strictEqual(persistedSwarms.length, 1);
      assert.strictEqual(persistedSwarms[0].name, 'integration-test-swarm');

      // 3. Spawn multiple agents
      const agents = [];
      for (const type of ['researcher', 'coder', 'analyst']) {
        const agent = await swarm.spawn({
          type,
          name: `${type}-agent`,
          capabilities: [type, 'collaboration'],
          enableNeuralNetwork: true,
        });
        agents.push(agent);
      }

      assert.strictEqual(agents.length, 3);

      // 4. Verify agents were persisted
      const persistedAgents = ruvSwarm.persistence.getSwarmAgents(swarm.id);
      assert.strictEqual(persistedAgents.length, 3);

      // 5. Orchestrate task
      const task = await swarm.orchestrate({
        description: 'Analyze codebase and implement optimization',
        priority: 'high',
        dependencies: [],
        maxAgents: 2,
        estimatedDuration: 5000,
      });

      assert(task);
      assert(task.assignedAgents.length > 0);

      // 6. Verify task was persisted
      const persistedTasks = ruvSwarm.persistence.getSwarmTasks(swarm.id);
      assert.strictEqual(persistedTasks.length, 1);
      assert.strictEqual(persistedTasks[0].description, task.description);

      // 7. Wait for task completion
      await new Promise(resolve => setTimeout(resolve, 1500));

      // 8. Verify task results
      const taskResults = await task.getResults();
      assert(taskResults);
      assert.strictEqual(taskResults.task_id, task.id);
      assert(taskResults.agent_results);

      // 9. Get swarm status
      const status = await swarm.getStatus();
      assert(status);
      assert.strictEqual(status.agents.total, 3);
      assert.strictEqual(status.tasks.total, 1);
      assert.strictEqual(status.tasks.completed, 1);

      // 10. Get global metrics
      const metrics = await ruvSwarm.getGlobalMetrics();
      assert.strictEqual(metrics.totalSwarms, 1);
      assert.strictEqual(metrics.totalAgents, 3);
      assert.strictEqual(metrics.totalTasks, 1);
    });
  });

  describe('Neural Agent Integration', () => {
    it('should create and use neural agents', async() => {
      const swarm = await ruvSwarm.createSwarm({
        name: 'neural-test-swarm',
        enableNeuralAgents: true,
      });

      // Create neural agent
      const agent = await swarm.spawn({
        type: 'researcher',
        enableNeuralNetwork: true,
      });

      // Verify neural capabilities
      assert(agent.cognitivePattern);
      assert(agent.capabilities);

      // Create neural agent wrapper
      const neuralAgent = NeuralAgentFactory.createNeuralAgent(agent, 'researcher');
      assert(neuralAgent);

      // Test neural analysis
      const analysis = await neuralAgent.analyzeTask({
        description: 'Research neural network architectures',
        priority: 'high',
      });

      assert(analysis);
      assert('complexity' in analysis);
      assert('creativity' in analysis);
      assert('confidence' in analysis);
    });
  });

  describe('Memory Persistence Integration', () => {
    it('should persist agent memory across operations', async() => {
      const swarm = await ruvSwarm.createSwarm({
        name: 'memory-test-swarm',
      });

      const agent = await swarm.spawn({
        type: 'analyst',
        name: 'memory-test-agent',
      });

      // Store agent memory
      const memoryData = {
        learned_patterns: ['pattern1', 'pattern2'],
        performance_history: [0.7, 0.8, 0.85],
        context: { domain: 'testing', version: 1 },
      };

      ruvSwarm.persistence.storeAgentMemory(
        agent.id,
        'analysis_memory',
        memoryData,
      );

      // Retrieve memory
      const retrieved = ruvSwarm.persistence.getAgentMemory(
        agent.id,
        'analysis_memory',
      );

      assert(retrieved);
      assert.deepStrictEqual(retrieved.value, memoryData);

      // Update memory
      memoryData.performance_history.push(0.9);
      ruvSwarm.persistence.storeAgentMemory(
        agent.id,
        'analysis_memory',
        memoryData,
      );

      const updated = ruvSwarm.persistence.getAgentMemory(
        agent.id,
        'analysis_memory',
      );

      assert.strictEqual(updated.value.performance_history.length, 4);
    });
  });

  describe('Multi-Swarm Coordination', () => {
    it('should coordinate multiple swarms', async() => {
      // Create multiple swarms
      const swarms = [];
      for (let i = 0; i < 3; i++) {
        const swarm = await ruvSwarm.createSwarm({
          name: `swarm-${i}`,
          topology: i === 0 ? 'mesh' : i === 1 ? 'star' : 'ring',
          maxAgents: 5,
        });

        // Spawn agents in each swarm
        for (let j = 0; j < 2; j++) {
          await swarm.spawn({
            type: j === 0 ? 'researcher' : 'coder',
            name: `agent-${i}-${j}`,
          });
        }

        swarms.push(swarm);
      }

      // Orchestrate tasks in parallel
      const tasks = await Promise.all(
        swarms.map(swarm =>
          swarm.orchestrate({
            description: `Task for ${swarm.id}`,
            priority: 'medium',
          }),
        ),
      );

      assert.strictEqual(tasks.length, 3);

      // Get all swarms status
      const allSwarms = await ruvSwarm.getAllSwarms();
      assert.strictEqual(allSwarms.length, 3);

      // Verify global metrics
      const metrics = await ruvSwarm.getGlobalMetrics();
      assert.strictEqual(metrics.totalSwarms, 3);
      assert.strictEqual(metrics.totalAgents, 6);
      assert.strictEqual(metrics.totalTasks, 3);
    });
  });

  describe('Event Logging Integration', () => {
    it('should log and retrieve events', async() => {
      const swarm = await ruvSwarm.createSwarm({
        name: 'event-test-swarm',
      });

      // Log various events
      const events = [
        { type: 'swarm_initialized', data: { topology: 'mesh' } },
        { type: 'agent_spawned', data: { agentType: 'researcher' } },
        { type: 'task_started', data: { taskId: 'task-123' } },
        { type: 'task_completed', data: { taskId: 'task-123', duration: 1000 } },
      ];

      for (const event of events) {
        ruvSwarm.persistence.logEvent(swarm.id, event.type, event.data);
      }

      // Retrieve events
      const retrievedEvents = ruvSwarm.persistence.getSwarmEvents(swarm.id);
      assert.strictEqual(retrievedEvents.length, 4);

      // Verify event order (should be DESC)
      assert.strictEqual(retrievedEvents[0].event_type, 'task_completed');
      assert.strictEqual(retrievedEvents[3].event_type, 'swarm_initialized');
    });
  });

  describe('Performance Metrics Integration', () => {
    it('should track and aggregate performance metrics', async() => {
      const swarm = await ruvSwarm.createSwarm({
        name: 'metrics-test-swarm',
      });

      const agent = await swarm.spawn({
        type: 'optimizer',
        name: 'metrics-agent',
      });

      // Record various metrics
      const metricsToRecord = [
        { name: 'task_completion_time', value: 1500 },
        { name: 'memory_usage', value: 45.5 },
        { name: 'cpu_usage', value: 32.1 },
        { name: 'accuracy_score', value: 0.92 },
      ];

      for (const metric of metricsToRecord) {
        ruvSwarm.persistence.recordMetric(
          'agent',
          agent.id,
          metric.name,
          metric.value,
        );
      }

      // Retrieve all metrics
      const allMetrics = ruvSwarm.persistence.getMetrics('agent', agent.id);
      assert.strictEqual(allMetrics.length, 4);

      // Retrieve specific metric
      const memoryMetrics = ruvSwarm.persistence.getMetrics(
        'agent',
        agent.id,
        'memory_usage',
      );
      assert.strictEqual(memoryMetrics.length, 1);
      assert.strictEqual(memoryMetrics[0].metric_value, 45.5);
    });
  });

  describe('Error Recovery Integration', () => {
    it('should handle and recover from errors gracefully', async() => {
      const swarm = await ruvSwarm.createSwarm({
        name: 'error-test-swarm',
      });

      // Try to orchestrate without agents (should fail)
      try {
        await swarm.orchestrate({
          description: 'This should fail',
        });
        assert.fail('Should have thrown error');
      } catch (error) {
        assert(error.message.includes('No agents available'));
      }

      // Spawn agent and retry
      await swarm.spawn({ type: 'researcher' });
      const task = await swarm.orchestrate({
        description: 'This should succeed',
      });

      assert(task);
      assert(task.assignedAgents.length > 0);
    });
  });

  describe('Complex Task Orchestration', () => {
    it('should handle complex task dependencies', async() => {
      const swarm = await ruvSwarm.createSwarm({
        name: 'complex-task-swarm',
        topology: 'hierarchical',
        strategy: 'specialized',
      });

      // Create specialized agents
      const researcher = await swarm.spawn({
        type: 'researcher',
        capabilities: ['research', 'documentation'],
      });

      const coder = await swarm.spawn({
        type: 'coder',
        capabilities: ['javascript', 'python', 'testing'],
      });

      const analyst = await swarm.spawn({
        type: 'analyst',
        capabilities: ['analysis', 'reporting'],
      });

      // Create tasks with dependencies
      const researchTask = await swarm.orchestrate({
        description: 'Research best practices for optimization',
        priority: 'high',
        requiredCapabilities: ['research'],
      });

      const implementationTask = await swarm.orchestrate({
        description: 'Implement optimization based on research',
        priority: 'high',
        dependencies: [researchTask.id],
        requiredCapabilities: ['javascript'],
      });

      const analysisTask = await swarm.orchestrate({
        description: 'Analyze performance improvements',
        priority: 'medium',
        dependencies: [implementationTask.id],
        requiredCapabilities: ['analysis'],
      });

      // Wait for tasks to progress
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Verify task assignments
      assert(researchTask.assignedAgents.includes(researcher.id));
      assert(implementationTask.assignedAgents.includes(coder.id));
      assert(analysisTask.assignedAgents.includes(analyst.id));
    });
  });

  describe('WASM Module Loading Integration', () => {
    it('should detect and report WASM module status', async() => {
      const metrics = await ruvSwarm.getGlobalMetrics();

      assert(metrics.features);
      assert('neural_networks' in metrics.features);
      assert('cognitive_diversity' in metrics.features);
      assert('simd_support' in metrics.features);

      assert(metrics.wasm_modules);
      // Should have at least core module status
      assert('core' in metrics.wasm_modules);
    });
  });

  describe('Swarm Monitoring Integration', () => {
    it('should monitor swarm activity', async() => {
      const swarm = await ruvSwarm.createSwarm({
        name: 'monitor-test-swarm',
      });

      await swarm.spawn({ type: 'researcher' });
      await swarm.spawn({ type: 'coder' });

      // Start monitoring
      const monitoringResult = await swarm.monitor(2000, 500);

      assert(monitoringResult);
      assert.strictEqual(monitoringResult.duration, 2000);
      assert.strictEqual(monitoringResult.interval, 500);

      // Orchestrate task during monitoring
      await swarm.orchestrate({
        description: 'Task during monitoring',
      });

      // Get final status
      const status = await swarm.getStatus(true);
      assert(status.agents.total >= 2);
      assert(status.tasks.total >= 1);
    });
  });

  describe('Cleanup and Resource Management', () => {
    it('should properly clean up resources', async() => {
      const swarm = await ruvSwarm.createSwarm({
        name: 'cleanup-test-swarm',
      });

      const agent = await swarm.spawn({ type: 'researcher' });

      // Store some data
      ruvSwarm.persistence.storeAgentMemory(agent.id, 'test-key', { data: 'test' });
      ruvSwarm.persistence.logEvent(swarm.id, 'test_event', { test: true });

      // Terminate swarm
      await swarm.terminate();

      // Verify swarm is removed from active swarms
      assert(!ruvSwarm.activeSwarms.has(swarm.id));

      // Data should still be in persistence (for historical analysis)
      const events = ruvSwarm.persistence.getSwarmEvents(swarm.id);
      assert(events.length > 0);
    });
  });
});

// Run tests
if (require.main === module) {
  console.log('Running API Integration Tests...');
  require('../../node_modules/.bin/jest');
}