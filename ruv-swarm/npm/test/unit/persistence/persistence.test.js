/**
 * Unit tests for SwarmPersistence module
 */

import { SwarmPersistence } from '../../../src/persistence';
import assert from 'assert';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

describe('SwarmPersistence Tests', () => {
  let persistence;
  const testDbPath = path.join(__dirname, 'test-persistence.db');

  beforeEach(() => {
    // Clean up test database if it exists
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
    persistence = new SwarmPersistence(testDbPath);
  });

  afterEach(() => {
    if (persistence) {
      persistence.close();
    }
    // Clean up test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  describe('Database Initialization', () => {
    it('should create database file', () => {
      assert(fs.existsSync(testDbPath));
    });

    it('should create all required tables', () => {
      const tables = persistence.db.prepare(`
        SELECT name FROM sqlite_master 
        WHERE type='table' 
        ORDER BY name
      `).all();

      const expectedTables = [
        'agent_memory',
        'agents',
        'events',
        'metrics',
        'neural_networks',
        'swarms',
        'task_results',
        'tasks',
      ];

      const tableNames = tables.map(t => t.name);
      expectedTables.forEach(table => {
        assert(tableNames.includes(table), `Missing table: ${table}`);
      });
    });

    it('should create indexes', () => {
      const indexes = persistence.db.prepare(`
        SELECT name FROM sqlite_master 
        WHERE type='index' 
        ORDER BY name
      `).all();

      assert(indexes.length > 0);
    });
  });

  describe('Swarm Operations', () => {
    const testSwarm = {
      id: 'swarm-test-123',
      name: 'Test Swarm',
      topology: 'mesh',
      maxAgents: 10,
      strategy: 'balanced',
      metadata: { test: true },
    };

    it('should create swarm', () => {
      const result = persistence.createSwarm(testSwarm);
      assert(result.changes === 1);
    });

    it('should get active swarms', () => {
      persistence.createSwarm(testSwarm);
      persistence.createSwarm({
        ...testSwarm,
        id: 'swarm-test-456',
        name: 'Another Swarm',
      });

      const swarms = persistence.getActiveSwarms();
      assert.strictEqual(swarms.length, 2);
      assert.strictEqual(swarms[0].name, 'Test Swarm');
      assert.deepStrictEqual(swarms[0].metadata, { test: true });
    });

    it('should handle swarm metadata correctly', () => {
      const swarmWithComplexMetadata = {
        ...testSwarm,
        metadata: {
          config: { enableML: true },
          tags: ['production', 'high-priority'],
          version: 2.0,
        },
      };

      persistence.createSwarm(swarmWithComplexMetadata);
      const swarms = persistence.getActiveSwarms();
      assert.deepStrictEqual(swarms[0].metadata, swarmWithComplexMetadata.metadata);
    });
  });

  describe('Agent Operations', () => {
    const testSwarmId = 'swarm-test-123';
    const testAgent = {
      id: 'agent-test-123',
      swarmId: testSwarmId,
      name: 'Test Agent',
      type: 'researcher',
      capabilities: ['research', 'analysis'],
      neuralConfig: { layers: [10, 20, 10] },
      metrics: { tasksCompleted: 0 },
    };

    beforeEach(() => {
      // Create parent swarm
      persistence.createSwarm({
        id: testSwarmId,
        name: 'Test Swarm',
        topology: 'mesh',
        maxAgents: 10,
      });
    });

    it('should create agent', () => {
      const result = persistence.createAgent(testAgent);
      assert(result.changes === 1);
    });

    it('should update agent status', () => {
      persistence.createAgent(testAgent);
      const result = persistence.updateAgentStatus(testAgent.id, 'busy');
      assert(result.changes === 1);

      const agent = persistence.getAgent(testAgent.id);
      assert.strictEqual(agent.status, 'busy');
    });

    it('should get agent by id', () => {
      persistence.createAgent(testAgent);
      const agent = persistence.getAgent(testAgent.id);

      assert.strictEqual(agent.id, testAgent.id);
      assert.strictEqual(agent.name, testAgent.name);
      assert.deepStrictEqual(agent.capabilities, testAgent.capabilities);
      assert.deepStrictEqual(agent.neural_config, testAgent.neuralConfig);
    });

    it('should get swarm agents', () => {
      persistence.createAgent(testAgent);
      persistence.createAgent({
        ...testAgent,
        id: 'agent-test-456',
        name: 'Another Agent',
        type: 'coder',
      });

      const agents = persistence.getSwarmAgents(testSwarmId);
      assert.strictEqual(agents.length, 2);
    });

    it('should filter swarm agents by status', () => {
      persistence.createAgent(testAgent);
      persistence.createAgent({
        ...testAgent,
        id: 'agent-test-456',
        name: 'Busy Agent',
      });

      persistence.updateAgentStatus('agent-test-456', 'busy');

      const busyAgents = persistence.getSwarmAgents(testSwarmId, 'busy');
      assert.strictEqual(busyAgents.length, 1);
      assert.strictEqual(busyAgents[0].id, 'agent-test-456');
    });
  });

  describe('Task Operations', () => {
    const testSwarmId = 'swarm-test-123';
    const testTask = {
      id: 'task-test-123',
      swarmId: testSwarmId,
      description: 'Test task',
      priority: 'high',
      status: 'pending',
      assignedAgents: ['agent-1', 'agent-2'],
    };

    beforeEach(() => {
      persistence.createSwarm({
        id: testSwarmId,
        name: 'Test Swarm',
        topology: 'mesh',
        maxAgents: 10,
      });
    });

    it('should create task', () => {
      const result = persistence.createTask(testTask);
      assert(result.changes === 1);
    });

    it('should update task', () => {
      persistence.createTask(testTask);

      const updates = {
        status: 'completed',
        result: { success: true, output: 'Task completed' },
        completed_at: new Date().toISOString(),
        execution_time_ms: 5000,
      };

      const result = persistence.updateTask(testTask.id, updates);
      assert(result.changes === 1);

      const task = persistence.getTask(testTask.id);
      assert.strictEqual(task.status, 'completed');
      assert.deepStrictEqual(task.result, updates.result);
      assert.strictEqual(task.execution_time_ms, 5000);
    });

    it('should get task by id', () => {
      persistence.createTask(testTask);
      const task = persistence.getTask(testTask.id);

      assert.strictEqual(task.id, testTask.id);
      assert.strictEqual(task.description, testTask.description);
      assert.deepStrictEqual(task.assigned_agents, testTask.assignedAgents);
    });

    it('should get swarm tasks', () => {
      persistence.createTask(testTask);
      persistence.createTask({
        ...testTask,
        id: 'task-test-456',
        description: 'Another task',
        status: 'in_progress',
      });

      const tasks = persistence.getSwarmTasks(testSwarmId);
      assert.strictEqual(tasks.length, 2);
    });

    it('should filter swarm tasks by status', () => {
      persistence.createTask(testTask);
      persistence.createTask({
        ...testTask,
        id: 'task-test-456',
        status: 'completed',
      });

      const pendingTasks = persistence.getSwarmTasks(testSwarmId, 'pending');
      assert.strictEqual(pendingTasks.length, 1);
      assert.strictEqual(pendingTasks[0].id, testTask.id);
    });
  });

  describe('Memory Operations', () => {
    const testAgentId = 'agent-test-123';

    beforeEach(() => {
      const swarmId = 'swarm-test-123';
      persistence.createSwarm({
        id: swarmId,
        name: 'Test Swarm',
        topology: 'mesh',
        maxAgents: 10,
      });
      persistence.createAgent({
        id: testAgentId,
        swarmId,
        name: 'Test Agent',
        type: 'researcher',
      });
    });

    it('should store agent memory', () => {
      const memoryData = { learned: 'something', count: 42 };
      const result = persistence.storeAgentMemory(testAgentId, 'test-key', memoryData);
      assert(result.changes === 1);
    });

    it('should update existing memory', () => {
      persistence.storeAgentMemory(testAgentId, 'test-key', { value: 1 });
      persistence.storeAgentMemory(testAgentId, 'test-key', { value: 2 });

      const memory = persistence.getAgentMemory(testAgentId, 'test-key');
      assert.deepStrictEqual(memory.value, { value: 2 });
    });

    it('should get agent memory by key', () => {
      const data = { test: 'data' };
      persistence.storeAgentMemory(testAgentId, 'specific-key', data);

      const memory = persistence.getAgentMemory(testAgentId, 'specific-key');
      assert.strictEqual(memory.agent_id, testAgentId);
      assert.strictEqual(memory.key, 'specific-key');
      assert.deepStrictEqual(memory.value, data);
    });

    it('should get all agent memories', () => {
      persistence.storeAgentMemory(testAgentId, 'key1', { a: 1 });
      persistence.storeAgentMemory(testAgentId, 'key2', { b: 2 });
      persistence.storeAgentMemory(testAgentId, 'key3', { c: 3 });

      const memories = persistence.getAgentMemory(testAgentId);
      assert.strictEqual(memories.length, 3);
      assert(memories.some(m => m.key === 'key1'));
      assert(memories.some(m => m.key === 'key2'));
      assert(memories.some(m => m.key === 'key3'));
    });
  });

  describe('Neural Network Operations', () => {
    const testAgentId = 'agent-test-123';
    const testNetwork = {
      agentId: testAgentId,
      architecture: {
        layers: [10, 20, 10],
        activationFunction: 'sigmoid',
      },
      weights: [[0.1, 0.2], [0.3, 0.4]],
      trainingData: { epochs: 100, loss: 0.01 },
      performanceMetrics: { accuracy: 0.95 },
    };

    beforeEach(() => {
      const swarmId = 'swarm-test-123';
      persistence.createSwarm({
        id: swarmId,
        name: 'Test Swarm',
        topology: 'mesh',
        maxAgents: 10,
      });
      persistence.createAgent({
        id: testAgentId,
        swarmId,
        name: 'Test Agent',
        type: 'researcher',
      });
    });

    it('should store neural network', () => {
      const result = persistence.storeNeuralNetwork(testNetwork);
      assert(result.changes === 1);
    });

    it('should update neural network', () => {
      const result = persistence.storeNeuralNetwork(testNetwork);
      const networkId = result.lastInsertRowid;

      const updates = {
        weights: [[0.5, 0.6], [0.7, 0.8]],
        performance_metrics: { accuracy: 0.98 },
      };

      // Get the actual ID from the insert
      const networks = persistence.getAgentNeuralNetworks(testAgentId);
      const actualId = networks[0].id;

      const updateResult = persistence.updateNeuralNetwork(actualId, updates);
      assert(updateResult.changes === 1);

      const updatedNetworks = persistence.getAgentNeuralNetworks(testAgentId);
      assert.deepStrictEqual(updatedNetworks[0].weights, updates.weights);
      assert.deepStrictEqual(updatedNetworks[0].performance_metrics, updates.performance_metrics);
    });

    it('should get agent neural networks', () => {
      persistence.storeNeuralNetwork(testNetwork);
      persistence.storeNeuralNetwork({
        ...testNetwork,
        architecture: { layers: [5, 10, 5] },
      });

      const networks = persistence.getAgentNeuralNetworks(testAgentId);
      assert.strictEqual(networks.length, 2);
      assert.strictEqual(networks[0].agent_id, testAgentId);
      assert.deepStrictEqual(networks[0].architecture, testNetwork.architecture);
    });
  });

  describe('Metrics Operations', () => {
    it('should record metric', () => {
      const result = persistence.recordMetric('agent', 'agent-123', 'task_completion_time', 1500);
      assert(result.changes === 1);
    });

    it('should get metrics', () => {
      persistence.recordMetric('swarm', 'swarm-123', 'agents_active', 5);
      persistence.recordMetric('swarm', 'swarm-123', 'agents_active', 7);
      persistence.recordMetric('swarm', 'swarm-123', 'tasks_completed', 10);

      const metrics = persistence.getMetrics('swarm', 'swarm-123');
      assert(metrics.length >= 3);
    });

    it('should filter metrics by name', () => {
      persistence.recordMetric('agent', 'agent-123', 'memory_usage', 100);
      persistence.recordMetric('agent', 'agent-123', 'memory_usage', 150);
      persistence.recordMetric('agent', 'agent-123', 'cpu_usage', 25);

      const memoryMetrics = persistence.getMetrics('agent', 'agent-123', 'memory_usage');
      assert.strictEqual(memoryMetrics.length, 2);
      assert(memoryMetrics.every(m => m.metric_name === 'memory_usage'));
    });
  });

  describe('Event Logging', () => {
    const testSwarmId = 'swarm-test-123';

    it('should log event', () => {
      const eventData = {
        action: 'agent_spawned',
        agentId: 'agent-123',
        timestamp: Date.now(),
      };

      const result = persistence.logEvent(testSwarmId, 'agent_spawn', eventData);
      assert(result.changes === 1);
    });

    it('should get swarm events', () => {
      persistence.logEvent(testSwarmId, 'swarm_created', { name: 'Test Swarm' });
      persistence.logEvent(testSwarmId, 'agent_spawn', { agentId: 'agent-1' });
      persistence.logEvent(testSwarmId, 'task_orchestrated', { taskId: 'task-1' });

      const events = persistence.getSwarmEvents(testSwarmId);
      assert.strictEqual(events.length, 3);
      assert(events[0].timestamp > events[2].timestamp); // Should be ordered desc
    });

    it('should limit events returned', () => {
      for (let i = 0; i < 10; i++) {
        persistence.logEvent(testSwarmId, 'test_event', { index: i });
      }

      const events = persistence.getSwarmEvents(testSwarmId, 5);
      assert.strictEqual(events.length, 5);
    });
  });

  describe('Cleanup Operations', () => {
    it('should cleanup old data', () => {
      const swarmId = 'swarm-test-123';

      // Insert old event (manually with old timestamp)
      const oldTimestamp = new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString();
      persistence.db.prepare(`
        INSERT INTO events (swarm_id, event_type, event_data, timestamp)
        VALUES (?, ?, ?, ?)
      `).run(swarmId, 'old_event', '{}', oldTimestamp);

      // Insert recent event
      persistence.logEvent(swarmId, 'recent_event', {});

      // Run cleanup
      persistence.cleanup();

      // Check that old event is gone
      const events = persistence.getSwarmEvents(swarmId);
      assert(events.every(e => e.event_type !== 'old_event'));
    });
  });

  describe('Error Handling', () => {
    it('should handle foreign key constraints', () => {
      const invalidAgent = {
        id: 'agent-invalid',
        swarmId: 'non-existent-swarm',
        name: 'Invalid Agent',
        type: 'researcher',
      };

      assert.throws(() => {
        persistence.createAgent(invalidAgent);
      }, /FOREIGN KEY constraint failed/);
    });

    it('should handle unique constraints', () => {
      const swarmId = 'swarm-test-123';
      const agentId = 'agent-test-123';

      persistence.createSwarm({
        id: swarmId,
        name: 'Test Swarm',
        topology: 'mesh',
        maxAgents: 10,
      });

      persistence.createAgent({
        id: agentId,
        swarmId,
        name: 'Test Agent',
        type: 'researcher',
      });

      // Store memory twice with same key (should update, not error)
      persistence.storeAgentMemory(agentId, 'duplicate-key', { value: 1 });
      const result = persistence.storeAgentMemory(agentId, 'duplicate-key', { value: 2 });
      assert(result.changes === 1);

      const memory = persistence.getAgentMemory(agentId, 'duplicate-key');
      assert.deepStrictEqual(memory.value, { value: 2 });
    });
  });
});

// Run tests when this file is executed directly
if (require.main === module) {
  console.log('Running SwarmPersistence Unit Tests...');
  require('../../../node_modules/.bin/jest');
}