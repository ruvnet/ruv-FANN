/**
 * Unit tests for RuvSwarm core class
 */

import { RuvSwarm, Swarm, Agent, Task } from '../../../src/index-enhanced.js';
import assert from 'assert';

describe('RuvSwarm Core Tests', () => {
  let ruvSwarm;

  beforeEach(async() => {
    // Reset global instance
    global._ruvSwarmInstance = null;
    global._ruvSwarmInitialized = 0;
  });

  afterEach(() => {
    if (ruvSwarm && ruvSwarm.persistence) {
      ruvSwarm.persistence.close();
    }
  });

  describe('Initialization', () => {
    it('should initialize with default options', async() => {
      ruvSwarm = await RuvSwarm.initialize();
      assert(ruvSwarm instanceof RuvSwarm);
      assert(ruvSwarm.wasmLoader);
      assert.strictEqual(typeof ruvSwarm.features, 'object');
    });

    it('should initialize with custom options', async() => {
      ruvSwarm = await RuvSwarm.initialize({
        enablePersistence: false,
        enableNeuralNetworks: false,
        useSIMD: false,
        debug: false,
      });
      assert(ruvSwarm instanceof RuvSwarm);
      assert.strictEqual(ruvSwarm.persistence, null);
      assert.strictEqual(ruvSwarm.features.neural_networks, false);
    });

    it('should return same instance on multiple initializations', async() => {
      const instance1 = await RuvSwarm.initialize();
      const instance2 = await RuvSwarm.initialize();
      assert.strictEqual(instance1, instance2);
    });

    it('should detect SIMD support', () => {
      const simdSupported = RuvSwarm.detectSIMDSupport();
      assert.strictEqual(typeof simdSupported, 'boolean');
    });

    it('should provide runtime features', () => {
      const features = RuvSwarm.getRuntimeFeatures();
      assert(features.webassembly);
      assert('simd' in features);
      assert('workers' in features);
      assert('shared_array_buffer' in features);
      assert('bigint' in features);
    });

    it('should provide version', () => {
      const version = RuvSwarm.getVersion();
      assert.strictEqual(version, '0.2.0');
    });

    it('should get memory usage', () => {
      const memoryUsage = RuvSwarm.getMemoryUsage();
      if (memoryUsage) {
        assert('used' in memoryUsage);
        assert('total' in memoryUsage);
        assert('limit' in memoryUsage);
      }
    });
  });

  describe('Swarm Creation', () => {
    beforeEach(async() => {
      ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });
    });

    it('should create swarm with default config', async() => {
      const swarm = await ruvSwarm.createSwarm({});
      assert(swarm instanceof Swarm);
      assert(swarm.id);
      assert.strictEqual(swarm.agents.size, 0);
      assert.strictEqual(swarm.tasks.size, 0);
    });

    it('should create swarm with custom config', async() => {
      const config = {
        name: 'test-swarm',
        topology: 'hierarchical',
        strategy: 'specialized',
        maxAgents: 20,
        enableCognitiveDiversity: true,
        enableNeuralAgents: true,
      };
      const swarm = await ruvSwarm.createSwarm(config);
      assert.strictEqual(swarm.id.includes('swarm-'), true);
      assert.strictEqual(ruvSwarm.activeSwarms.size, 1);
      assert.strictEqual(ruvSwarm.metrics.totalSwarms, 1);
    });

    it('should track multiple swarms', async() => {
      const swarm1 = await ruvSwarm.createSwarm({ name: 'swarm1' });
      const swarm2 = await ruvSwarm.createSwarm({ name: 'swarm2' });
      assert.strictEqual(ruvSwarm.activeSwarms.size, 2);
      assert.strictEqual(ruvSwarm.metrics.totalSwarms, 2);
      assert(ruvSwarm.activeSwarms.has(swarm1.id));
      assert(ruvSwarm.activeSwarms.has(swarm2.id));
    });
  });

  describe('Swarm Status', () => {
    let swarm;

    beforeEach(async() => {
      ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });
      swarm = await ruvSwarm.createSwarm({ name: 'status-test-swarm' });
    });

    it('should get swarm status', async() => {
      const status = await ruvSwarm.getSwarmStatus(swarm.id);
      assert(status);
      assert(status.agents);
      assert(status.tasks);
      assert.strictEqual(status.agents.total, 0);
      assert.strictEqual(status.tasks.total, 0);
    });

    it('should throw error for non-existent swarm', async() => {
      try {
        await ruvSwarm.getSwarmStatus('non-existent-id');
        assert.fail('Should have thrown error');
      } catch (error) {
        assert(error.message.includes('Swarm not found'));
      }
    });

    it('should get all swarms', async() => {
      const swarm2 = await ruvSwarm.createSwarm({ name: 'another-swarm' });
      const allSwarms = await ruvSwarm.getAllSwarms();
      assert.strictEqual(allSwarms.length, 2);
      assert(allSwarms.some(s => s.id === swarm.id));
      assert(allSwarms.some(s => s.id === swarm2.id));
    });
  });

  describe('Global Metrics', () => {
    beforeEach(async() => {
      ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });
    });

    it('should provide global metrics', async() => {
      const swarm = await ruvSwarm.createSwarm({ name: 'metrics-swarm' });
      const agent = await swarm.spawn({ type: 'researcher' });

      const metrics = await ruvSwarm.getGlobalMetrics();
      assert(metrics);
      assert.strictEqual(metrics.totalSwarms, 1);
      assert.strictEqual(metrics.totalAgents, 1);
      assert(metrics.features);
      assert(metrics.wasm_modules);
      assert(metrics.timestamp);
    });

    it('should aggregate metrics from multiple swarms', async() => {
      const swarm1 = await ruvSwarm.createSwarm({ name: 'swarm1' });
      const swarm2 = await ruvSwarm.createSwarm({ name: 'swarm2' });

      await swarm1.spawn({ type: 'coder' });
      await swarm1.spawn({ type: 'analyst' });
      await swarm2.spawn({ type: 'researcher' });

      const metrics = await ruvSwarm.getGlobalMetrics();
      assert.strictEqual(metrics.totalSwarms, 2);
      assert.strictEqual(metrics.totalAgents, 3);
    });
  });

  describe('Feature Detection', () => {
    beforeEach(async() => {
      ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });
    });

    it('should detect features', async() => {
      await ruvSwarm.detectFeatures(true);
      assert('simd_support' in ruvSwarm.features);
      assert('neural_networks' in ruvSwarm.features);
      assert('cognitive_diversity' in ruvSwarm.features);
      assert('forecasting' in ruvSwarm.features);
    });
  });
});

describe('Swarm Class Tests', () => {
  let ruvSwarm, swarm;

  beforeEach(async() => {
    ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });
    swarm = await ruvSwarm.createSwarm({ name: 'test-swarm' });
  });

  describe('Agent Spawning', () => {
    it('should spawn agent with default config', async() => {
      const agent = await swarm.spawn({});
      assert(agent instanceof Agent);
      assert(agent.id);
      assert.strictEqual(agent.type, 'researcher');
      assert.strictEqual(agent.status, 'idle');
      assert(Array.isArray(agent.capabilities));
    });

    it('should spawn agent with custom config', async() => {
      const config = {
        type: 'coder',
        name: 'test-coder',
        capabilities: ['javascript', 'python'],
        enableNeuralNetwork: true,
      };
      const agent = await swarm.spawn(config);
      assert.strictEqual(agent.type, 'coder');
      assert(agent.name.includes('test-coder'));
      assert.deepStrictEqual(agent.capabilities, ['javascript', 'python']);
    });

    it('should track spawned agents', async() => {
      const agent1 = await swarm.spawn({ type: 'researcher' });
      const agent2 = await swarm.spawn({ type: 'coder' });
      assert.strictEqual(swarm.agents.size, 2);
      assert(swarm.agents.has(agent1.id));
      assert(swarm.agents.has(agent2.id));
    });
  });

  describe('Task Orchestration', () => {
    beforeEach(async() => {
      // Spawn some agents for task assignment
      await swarm.spawn({ type: 'researcher' });
      await swarm.spawn({ type: 'coder' });
      await swarm.spawn({ type: 'analyst' });
    });

    it('should orchestrate task with default config', async() => {
      const task = await swarm.orchestrate({
        description: 'Test task',
      });
      assert(task instanceof Task);
      assert(task.id);
      assert.strictEqual(task.description, 'Test task');
      assert.strictEqual(task.status, 'orchestrated');
      assert(task.assignedAgents.length > 0);
    });

    it('should orchestrate task with custom config', async() => {
      const taskConfig = {
        description: 'Complex task',
        priority: 'high',
        dependencies: ['task-1', 'task-2'],
        maxAgents: 2,
        estimatedDuration: 5000,
        requiredCapabilities: ['analysis'],
      };
      const task = await swarm.orchestrate(taskConfig);
      assert.strictEqual(task.description, 'Complex task');
      assert(task.assignedAgents.length <= 2);
    });

    it('should throw error when no agents available', async() => {
      // Create new swarm without agents
      const emptySwarm = await ruvSwarm.createSwarm({ name: 'empty-swarm' });
      try {
        await emptySwarm.orchestrate({ description: 'Test' });
        assert.fail('Should have thrown error');
      } catch (error) {
        assert(error.message.includes('No agents available'));
      }
    });

    it('should track orchestrated tasks', async() => {
      const task1 = await swarm.orchestrate({ description: 'Task 1' });
      const task2 = await swarm.orchestrate({ description: 'Task 2' });
      assert.strictEqual(swarm.tasks.size, 2);
      assert(swarm.tasks.has(task1.id));
      assert(swarm.tasks.has(task2.id));
    });
  });

  describe('Agent Selection', () => {
    beforeEach(async() => {
      await swarm.spawn({ type: 'researcher', capabilities: ['research', 'analysis'] });
      await swarm.spawn({ type: 'coder', capabilities: ['javascript', 'python'] });
      await swarm.spawn({ type: 'analyst', capabilities: ['analysis', 'reporting'] });
    });

    it('should select available agents', () => {
      const agents = swarm.selectAvailableAgents();
      assert.strictEqual(agents.length, 3);
      agents.forEach(agent => {
        assert.notStrictEqual(agent.status, 'busy');
      });
    });

    it('should filter agents by capabilities', () => {
      const agents = swarm.selectAvailableAgents(['analysis']);
      assert.strictEqual(agents.length, 2);
      agents.forEach(agent => {
        assert(agent.capabilities.includes('analysis'));
      });
    });

    it('should limit agent selection', () => {
      const agents = swarm.selectAvailableAgents([], 2);
      assert.strictEqual(agents.length, 2);
    });
  });

  describe('Swarm Status', () => {
    beforeEach(async() => {
      await swarm.spawn({ type: 'researcher' });
      await swarm.spawn({ type: 'coder' });
      await swarm.orchestrate({ description: 'Test task' });
    });

    it('should get basic status', async() => {
      const status = await swarm.getStatus(false);
      assert(status);
      assert.strictEqual(status.id, swarm.id);
      assert.strictEqual(status.agents.total, 2);
      assert.strictEqual(status.tasks.total, 1);
    });

    it('should get detailed status', async() => {
      const status = await swarm.getStatus(true);
      assert(status);
      assert(status.agents);
      assert(status.tasks);
    });
  });

  describe('Swarm Monitoring', () => {
    it('should monitor swarm', async() => {
      const result = await swarm.monitor(1000, 100);
      assert(result);
      assert.strictEqual(result.duration, 1000);
      assert.strictEqual(result.interval, 100);
      assert(Array.isArray(result.snapshots));
    });
  });

  describe('Swarm Termination', () => {
    it('should terminate swarm', async() => {
      const swarmId = swarm.id;
      await swarm.terminate();
      assert(!ruvSwarm.activeSwarms.has(swarmId));
    });
  });
});

describe('Agent Class Tests', () => {
  let ruvSwarm, swarm, agent;

  beforeEach(async() => {
    ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });
    swarm = await ruvSwarm.createSwarm({ name: 'test-swarm' });
    agent = await swarm.spawn({ type: 'researcher', name: 'test-agent' });
  });

  describe('Agent Properties', () => {
    it('should have correct properties', () => {
      assert(agent.id);
      assert.strictEqual(agent.type, 'researcher');
      assert(agent.name.includes('test-agent'));
      assert.strictEqual(agent.cognitivePattern, 'adaptive');
      assert(Array.isArray(agent.capabilities));
      assert.strictEqual(agent.status, 'idle');
    });
  });

  describe('Task Execution', () => {
    it('should execute task', async() => {
      const result = await agent.execute({ description: 'Test task' });
      assert(result);
      assert.strictEqual(result.status, 'completed');
      assert(result.result);
      assert(result.executionTime);
    });

    it('should update status during execution', async() => {
      const promise = agent.execute({ description: 'Test task' });
      // Status should be busy during execution
      assert.strictEqual(agent.status, 'busy');
      await promise;
      // Status should be idle after execution
      assert.strictEqual(agent.status, 'idle');
    });
  });

  describe('Agent Metrics', () => {
    it('should provide metrics', async() => {
      const metrics = await agent.getMetrics();
      assert(metrics);
      assert('tasksCompleted' in metrics);
      assert('averageExecutionTime' in metrics);
      assert('successRate' in metrics);
      assert('memoryUsage' in metrics);
    });
  });

  describe('Status Updates', () => {
    it('should update status', async() => {
      await agent.updateStatus('busy');
      assert.strictEqual(agent.status, 'busy');
      await agent.updateStatus('idle');
      assert.strictEqual(agent.status, 'idle');
    });
  });
});

describe('Task Class Tests', () => {
  let ruvSwarm, swarm, task;

  beforeEach(async() => {
    ruvSwarm = await RuvSwarm.initialize({ enablePersistence: false });
    swarm = await ruvSwarm.createSwarm({ name: 'test-swarm' });

    // Create agents and orchestrate task
    await swarm.spawn({ type: 'researcher' });
    await swarm.spawn({ type: 'coder' });
    task = await swarm.orchestrate({
      description: 'Test task execution',
      priority: 'high',
    });
  });

  describe('Task Properties', () => {
    it('should have correct properties', () => {
      assert(task.id);
      assert.strictEqual(task.description, 'Test task execution');
      assert(task.assignedAgents.length > 0);
      assert.strictEqual(task.progress, 0);
    });
  });

  describe('Task Execution', () => {
    it('should execute task automatically', async() => {
      // Wait for task to complete
      await new Promise(resolve => setTimeout(resolve, 1000));

      assert.strictEqual(task.status, 'completed');
      assert.strictEqual(task.progress, 1.0);
      assert(task.result);
      assert(task.startTime);
      assert(task.endTime);
    });

    it('should track execution time', async() => {
      // Wait for task to complete
      await new Promise(resolve => setTimeout(resolve, 1000));

      const executionTime = task.endTime - task.startTime;
      assert(executionTime > 0);
      assert.strictEqual(
        task.result.execution_summary.execution_time_ms,
        executionTime,
      );
    });
  });

  describe('Task Status', () => {
    it('should get task status', async() => {
      const status = await task.getStatus();
      assert(status);
      assert.strictEqual(status.id, task.id);
      assert.strictEqual(status.status, task.status);
      assert(Array.isArray(status.assignedAgents));
      assert('progress' in status);
      assert('execution_time_ms' in status);
    });
  });

  describe('Task Results', () => {
    it('should get task results after completion', async() => {
      // Wait for task to complete
      await new Promise(resolve => setTimeout(resolve, 1000));

      const results = await task.getResults();
      assert(results);
      assert.strictEqual(results.task_id, task.id);
      assert.strictEqual(results.description, task.description);
      assert(Array.isArray(results.agent_results));
      assert(results.execution_summary);
    });
  });
});

// Run tests
console.log('Running RuvSwarm Core Unit Tests...');
import('../../../node_modules/.bin/jest');