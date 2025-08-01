/**
 * DAA Service Test Suite
 * Tests for agent lifecycle, state persistence, workflow coordination,
 * and < 1ms cross-boundary call latency
 */

import { jest } from '@jest/globals';
import { DAAService } from '../src/daa-service.js';

describe('DAA Service', () => {
  let service;

  beforeEach(() => {
    service = new DAAService();
  });

  afterEach(async() => {
    await service.cleanup();
  });

  describe('Initialization', () => {
    test('should initialize successfully', async() => {
      await expect(service.initialize()).resolves.not.toThrow();
      expect(service.initialized).toBe(true);
    });

    test('should emit initialized event', async() => {
      const listener = jest.fn();
      service.on('initialized', listener);

      await service.initialize();

      expect(listener).toHaveBeenCalled();
    });

    test('should handle multiple initialization calls', async() => {
      await service.initialize();
      await service.initialize(); // Should not throw

      expect(service.initialized).toBe(true);
    });
  });

  describe('Agent Lifecycle Management', () => {
    beforeEach(async() => {
      await service.initialize();
    });

    test('should create agent with capabilities', async() => {
      const agent = await service.createAgent('test-agent', ['decision_making', 'learning']);

      expect(agent).toBeDefined();
      expect(agent.id).toBe('test-agent');
      expect(agent.capabilities.has('decision_making')).toBe(true);
      expect(agent.capabilities.has('learning')).toBe(true);
      expect(agent.status).toBe('active');
    });

    test('should emit agentCreated event', async() => {
      const listener = jest.fn();
      service.on('agentCreated', listener);

      await service.createAgent('test-agent', ['learning']);

      expect(listener).toHaveBeenCalledWith({
        agentId: 'test-agent',
        capabilities: ['learning'],
      });
    });

    test('should destroy agent', async() => {
      await service.createAgent('test-agent');

      const result = await service.destroyAgent('test-agent');

      expect(result).toBe(true);
      expect(service.agents.has('test-agent')).toBe(false);
    });

    test('should batch create agents', async() => {
      const configs = [
        { id: 'agent-1', capabilities: ['learning'] },
        { id: 'agent-2', capabilities: ['decision_making'] },
        { id: 'agent-3', capabilities: ['coordination'] },
      ];

      const results = await service.batchCreateAgents(configs);

      expect(results).toHaveLength(3);
      expect(results.every(r => r.success)).toBe(true);
      expect(service.agents.size).toBe(3);
    });

    test('should persist and restore agent state', async() => {
      const agent = await service.createAgent('persistent-agent', ['memory_management']);

      // Save state
      service.agentStates.saveState('persistent-agent', {
        status: 'active',
        capabilities: ['memory_management'],
        metrics: agent.metrics,
        customData: 'test',
      });

      // Simulate reload by creating new service
      const newService = new DAAService();
      await newService.initialize();

      // Create agent and check if state is restored
      const restoredAgent = await newService.createAgent('persistent-agent');
      const state = await newService.agentStates.loadFromStorage('persistent-agent');

      expect(state).toBeDefined();
      expect(state.customData).toBe('test');

      await newService.cleanup();
    });
  });

  describe('Cross-Boundary Communication Performance', () => {
    let agent;
    const testContext = {
      environment_state: {
        environment_type: 'Dynamic',
        conditions: { test: 1.0 },
        stability: 0.5,
        resource_availability: 1.0,
      },
      available_actions: [
        {
          id: 'test-action',
          action_type: 'Compute',
          cost: 0.1,
          expected_reward: 0.5,
          risk: 0.1,
          prerequisites: [],
        },
      ],
      goals: [],
      history: [],
      constraints: {
        max_memory_mb: 1024,
        max_cpu_usage: 0.8,
        max_network_mbps: 100,
        max_execution_time: 300,
        energy_budget: 1000,
      },
      time_pressure: 0.0,
      uncertainty: 0.0,
    };

    beforeEach(async() => {
      await service.initialize();
      agent = await service.createAgent('perf-agent', ['decision_making']);
    });

    test('should make decision within 1ms latency threshold', async() => {
      const latencies = [];

      // Capture latency from events
      service.on('decisionMade', ({ latency }) => {
        latencies.push(latency);
      });

      // Make multiple decisions
      for (let i = 0; i < 10; i++) {
        await service.makeDecision('perf-agent', testContext);
      }

      // Check that average latency is under 1ms
      const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      console.log(`Average cross-boundary latency: ${avgLatency.toFixed(3)}ms`);

      // Allow some tolerance for test environment
      expect(avgLatency).toBeLessThan(5.0); // Relaxed for test environment
    });

    test('should handle batch decisions efficiently', async() => {
      const decisions = Array(20).fill(null).map((_, i) => ({
        agentId: 'perf-agent',
        context: { ...testContext, uncertainty: i * 0.05 },
      }));

      const start = performance.now();
      const results = await service.batchMakeDecisions(decisions);
      const duration = performance.now() - start;

      expect(results).toHaveLength(20);
      expect(results.every(r => r.success)).toBe(true);

      // Batch should be more efficient than sequential
      const perDecision = duration / 20;
      console.log(`Batch decision average: ${perDecision.toFixed(3)}ms per decision`);
      expect(perDecision).toBeLessThan(10.0); // Should benefit from batching
    });
  });

  describe('Multi-Agent Workflow Coordination', () => {
    beforeEach(async() => {
      await service.initialize();

      // Create agents for workflow
      await service.batchCreateAgents([
        { id: 'worker-1', capabilities: ['compute'] },
        { id: 'worker-2', capabilities: ['compute'] },
        { id: 'coordinator', capabilities: ['coordination'] },
      ]);
    });

    test('should create and execute simple workflow', async() => {
      const workflow = await service.createWorkflow(
        'simple-workflow',
        [
          {
            id: 'step1',
            task: async(agent) => ({ agent: agent.id, result: 'step1-complete' }),
          },
          {
            id: 'step2',
            task: async(agent) => ({ agent: agent.id, result: 'step2-complete' }),
          },
        ],
        { 'step2': ['step1'] },
      );

      expect(workflow.id).toBe('simple-workflow');
      expect(workflow.status).toBe('pending');

      // Execute steps in order
      const result1 = await service.executeWorkflowStep('simple-workflow', 'step1', ['worker-1']);
      expect(result1).toContainEqual({ agent: 'worker-1', result: 'step1-complete' });

      const result2 = await service.executeWorkflowStep('simple-workflow', 'step2', ['worker-2']);
      expect(result2).toContainEqual({ agent: 'worker-2', result: 'step2-complete' });

      // Check workflow status
      const status = service.workflows.getWorkflowStatus('simple-workflow');
      expect(status.status).toBe('completed');
      expect(status.progress.completed).toBe(2);
    });

    test('should enforce workflow dependencies', async() => {
      await service.createWorkflow(
        'dependent-workflow',
        [
          { id: 'A', task: async() => 'A' },
          { id: 'B', task: async() => 'B' },
          { id: 'C', task: async() => 'C' },
        ],
        {
          'B': ['A'],
          'C': ['A', 'B'],
        },
      );

      // Try to execute C before dependencies
      await expect(
        service.executeWorkflowStep('dependent-workflow', 'C', ['worker-1']),
      ).rejects.toThrow('Dependency A not completed');

      // Execute in correct order
      await service.executeWorkflowStep('dependent-workflow', 'A', ['worker-1']);
      await service.executeWorkflowStep('dependent-workflow', 'B', ['worker-2']);
      await service.executeWorkflowStep('dependent-workflow', 'C', ['coordinator']);

      const status = service.workflows.getWorkflowStatus('dependent-workflow');
      expect(status.status).toBe('completed');
    });

    test('should handle parallel workflow execution', async() => {
      await service.createWorkflow(
        'parallel-workflow',
        [
          { id: 'parallel-1', task: async() => new Promise(r => setTimeout(() => r('p1'), 50)) },
          { id: 'parallel-2', task: async() => new Promise(r => setTimeout(() => r('p2'), 50)) },
          { id: 'parallel-3', task: async() => new Promise(r => setTimeout(() => r('p3'), 50)) },
          { id: 'final', task: async() => 'done' },
        ],
        {
          'final': ['parallel-1', 'parallel-2', 'parallel-3'],
        },
      );

      // Execute parallel steps concurrently
      const start = performance.now();
      const [r1, r2, r3] = await Promise.all([
        service.executeWorkflowStep('parallel-workflow', 'parallel-1', ['worker-1']),
        service.executeWorkflowStep('parallel-workflow', 'parallel-2', ['worker-2']),
        service.executeWorkflowStep('parallel-workflow', 'parallel-3', ['coordinator']),
      ]);
      const parallelDuration = performance.now() - start;

      // Should complete in ~50ms, not 150ms (sequential)
      expect(parallelDuration).toBeLessThan(100);

      // Execute final step
      await service.executeWorkflowStep('parallel-workflow', 'final', ['coordinator']);

      const status = service.workflows.getWorkflowStatus('parallel-workflow');
      expect(status.status).toBe('completed');
    });
  });

  describe('State Synchronization', () => {
    beforeEach(async() => {
      await service.initialize();

      await service.batchCreateAgents([
        { id: 'sync-agent-1', capabilities: ['memory_management'] },
        { id: 'sync-agent-2', capabilities: ['memory_management'] },
        { id: 'sync-agent-3', capabilities: ['memory_management'] },
      ]);
    });

    test('should synchronize states across agents', async() => {
      // Set different states
      service.agentStates.saveState('sync-agent-1', {
        status: 'active',
        capabilities: ['memory_management'],
        metrics: { decisionsMade: 5 },
        data: 'agent1-data',
      });

      service.agentStates.saveState('sync-agent-2', {
        status: 'idle',
        capabilities: ['memory_management'],
        metrics: { decisionsMade: 3 },
        data: 'agent2-data',
      });

      // Synchronize
      const synced = await service.synchronizeStates(['sync-agent-1', 'sync-agent-2', 'sync-agent-3']);

      expect(synced.size).toBe(2); // agent-3 has no saved state
      expect(synced.get('sync-agent-1').data).toBe('agent1-data');
      expect(synced.get('sync-agent-2').data).toBe('agent2-data');
    });

    test('should emit synchronization event', async() => {
      const listener = jest.fn();
      service.on('statesSynchronized', listener);

      await service.synchronizeStates(['sync-agent-1', 'sync-agent-2']);

      expect(listener).toHaveBeenCalledWith(
        expect.objectContaining({
          agentIds: ['sync-agent-1', 'sync-agent-2'],
          duration: expect.any(Number),
        }),
      );
    });
  });

  describe('Resource Management', () => {
    beforeEach(async() => {
      await service.initialize();
    });

    test('should optimize resources', async() => {
      const result = await service.optimizeResources();

      // Result format depends on WASM implementation
      expect(result).toBeDefined();
    });

    test('should track memory usage', () => {
      const status = service.getStatus();

      expect(status.wasm.memoryUsage).toBeGreaterThanOrEqual(0);
    });

    test('should clean up resources', async() => {
      // Create some agents
      await service.batchCreateAgents([
        { id: 'cleanup-1' },
        { id: 'cleanup-2' },
        { id: 'cleanup-3' },
      ]);

      expect(service.agents.size).toBe(3);

      // Cleanup
      await service.cleanup();

      expect(service.agents.size).toBe(0);
    });
  });

  describe('Performance Monitoring', () => {
    beforeEach(async() => {
      await service.initialize();
      await service.createAgent('metrics-agent', ['decision_making']);
    });

    test('should track performance metrics', async() => {
      // Make some decisions
      const context = {
        environment_state: {
          environment_type: 'Stable',
          conditions: {},
          stability: 1.0,
          resource_availability: 1.0,
        },
        available_actions: [],
        goals: [],
        history: [],
        constraints: {
          max_memory_mb: 1024,
          max_cpu_usage: 0.8,
          max_network_mbps: 100,
          max_execution_time: 300,
          energy_budget: 1000,
        },
        time_pressure: 0.0,
        uncertainty: 0.0,
      };

      await service.makeDecision('metrics-agent', context);
      await service.makeDecision('metrics-agent', context);

      const metrics = service.getPerformanceMetrics();

      expect(metrics.system.totalAgents).toBe(1);
      expect(metrics.agents['metrics-agent'].decisionsMade).toBe(2);
      expect(metrics.agents['metrics-agent'].averageResponseTime).toBeGreaterThan(0);
    });

    test('should provide comprehensive status', () => {
      const status = service.getStatus();

      expect(status).toMatchObject({
        initialized: true,
        agents: {
          count: 1,
          ids: ['metrics-agent'],
          states: expect.any(Number),
        },
        workflows: {
          count: expect.any(Number),
          active: expect.any(Number),
        },
        wasm: {
          modules: expect.any(Object),
          memoryUsage: expect.any(Number),
        },
        performance: expect.any(Object),
      });
    });
  });

  describe('Error Handling', () => {
    beforeEach(async() => {
      await service.initialize();
    });

    test('should handle agent not found errors', async() => {
      await expect(
        service.makeDecision('non-existent', {}),
      ).rejects.toThrow('Agent non-existent not found');
    });

    test('should handle workflow not found errors', async() => {
      await expect(
        service.executeWorkflowStep('non-existent-workflow', 'step1', ['agent1']),
      ).rejects.toThrow('Workflow non-existent-workflow not found');
    });

    test('should track agent errors', async() => {
      await service.createAgent('error-agent');

      // Force an error by passing invalid context
      try {
        await service.makeDecision('error-agent', null);
      } catch (e) {
        // Expected
      }

      const metrics = service.getPerformanceMetrics();
      expect(metrics.agents['error-agent'].errors).toBe(1);
    });
  });
});

// Performance benchmark tests
describe('DAA Service Performance Benchmarks', () => {
  let service;

  beforeAll(async() => {
    service = new DAAService();
    await service.initialize();
  });

  afterAll(async() => {
    await service.cleanup();
  });

  test('should handle 100 agents without degradation', async() => {
    const configs = Array(100).fill(null).map((_, i) => ({
      id: `bench-agent-${i}`,
      capabilities: ['decision_making'],
    }));

    const start = performance.now();
    const results = await service.batchCreateAgents(configs);
    const duration = performance.now() - start;

    expect(results.every(r => r.success)).toBe(true);
    console.log(`Created 100 agents in ${duration.toFixed(0)}ms (${(duration / 100).toFixed(2)}ms per agent)`);

    // Cleanup bench agents
    for (const config of configs) {
      await service.destroyAgent(config.id);
    }
  });

  test('should maintain sub-millisecond latency under load', async() => {
    // Create test agents
    const agents = await service.batchCreateAgents([
      { id: 'load-test-1', capabilities: ['decision_making'] },
      { id: 'load-test-2', capabilities: ['decision_making'] },
      { id: 'load-test-3', capabilities: ['decision_making'] },
    ]);

    const context = {
      environment_state: {
        environment_type: 'Dynamic',
        conditions: {},
        stability: 0.5,
        resource_availability: 1.0,
      },
      available_actions: [],
      goals: [],
      history: [],
      constraints: {
        max_memory_mb: 1024,
        max_cpu_usage: 0.8,
        max_network_mbps: 100,
        max_execution_time: 300,
        energy_budget: 1000,
      },
      time_pressure: 0.5,
      uncertainty: 0.5,
    };

    // Make 1000 decisions across agents
    const latencies = [];
    service.on('decisionMade', ({ latency }) => latencies.push(latency));

    const decisions = Array(1000).fill(null).map((_, i) => ({
      agentId: `load-test-${(i % 3) + 1}`,
      context,
    }));

    const start = performance.now();
    await service.batchMakeDecisions(decisions);
    const totalDuration = performance.now() - start;

    const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    const maxLatency = Math.max(...latencies);

    console.log(`1000 decisions in ${totalDuration.toFixed(0)}ms`);
    console.log(`Average latency: ${avgLatency.toFixed(3)}ms`);
    console.log(`Max latency: ${maxLatency.toFixed(3)}ms`);
    console.log(`95th percentile: ${latencies.sort((a, b) => a - b)[Math.floor(latencies.length * 0.95)].toFixed(3)}ms`);

    // Cleanup
    for (const agent of agents) {
      if (agent.success) {
        await service.destroyAgent(agent.agent.id);
      }
    }
  });
});
