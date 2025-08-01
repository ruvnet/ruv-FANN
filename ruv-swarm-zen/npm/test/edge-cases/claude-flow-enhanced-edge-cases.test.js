/**
 * Edge Cases and E2E Tests for src/claude-flow-enhanced.js
 * Comprehensive coverage for BatchTool enforcement, parallel execution, and workflow optimization
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  ClaudeFlowEnhanced,
  BatchToolEnforcer,
  ClaudeFlowError,
  getClaudeFlow,
  createOptimizedWorkflow,
  executeWorkflow,
  getPerformanceReport,
  validateWorkflow,
} from '../../src/claude-flow-enhanced.js';

// Mock dependencies
const mockRuvSwarm = {
  initialize: vi.fn().mockResolvedValue({
    features: {
      simd_support: true,
      neural_networks: true,
      memory_limit: 2 * 1024 * 1024 * 1024,
    },
    createSwarm: vi.fn().mockResolvedValue({
      id: 'mock-swarm',
      agents: new Map(),
      maxAgents: 10,
    }),
  }),
};

const mockMcpTools = {
  initialize: vi.fn().mockResolvedValue(true),
  swarm_init: vi.fn().mockResolvedValue({
    id: 'test-swarm',
    topology: 'hierarchical',
    maxAgents: 8,
  }),
  agent_spawn: vi.fn().mockResolvedValue({
    agentId: 'test-agent',
    type: 'coordinator',
  }),
  task_orchestrate: vi.fn().mockResolvedValue({
    taskId: 'test-task',
    status: 'completed',
  }),
  memory_usage: vi.fn().mockResolvedValue({
    used: 1024 * 1024,
    total: 2 * 1024 * 1024,
  }),
  neural_status: vi.fn().mockResolvedValue({
    active: true,
    models: 5,
  }),
  benchmark_run: vi.fn().mockResolvedValue({
    duration: 1500,
    score: 95.5,
  }),
};

// Mock imports
vi.mock('../../src/index-enhanced.js', () => ({
  RuvSwarm: mockRuvSwarm,
}));

vi.mock('../../src/mcp-tools-enhanced.js', () => ({
  EnhancedMCPTools: class {
    async initialize() {
      return true;
    }
    swarm_init = mockMcpTools.swarm_init;
    agent_spawn = mockMcpTools.agent_spawn;
    task_orchestrate = mockMcpTools.task_orchestrate;
    memory_usage = mockMcpTools.memory_usage;
    neural_status = mockMcpTools.neural_status;
    benchmark_run = mockMcpTools.benchmark_run;
  },
}));

describe('Claude Flow Enhanced Edge Cases and E2E Tests', () => {
  let claudeFlow;
  let batchEnforcer;

  beforeEach(async() => {
    vi.clearAllMocks();
    claudeFlow = new ClaudeFlowEnhanced();
    batchEnforcer = new BatchToolEnforcer();

    // Reset global session ID
    global._claudeFlowSessionId = undefined;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('BatchToolEnforcer Edge Cases', () => {
    it('should track operations correctly', () => {
      batchEnforcer.trackOperation('file_operation');
      batchEnforcer.trackOperation('file_operation');
      batchEnforcer.trackOperation('mcp_tool');

      const report = batchEnforcer.getBatchingReport();
      expect(report.totalOperations).toBe(3);
      expect(report.batchableOperations).toHaveLength(1);
      expect(report.batchableOperations[0][0]).toBe('file_operation');
    });

    it('should detect batching violations', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      // Trigger batching violation
      for (let i = 0; i < 5; i++) {
        batchEnforcer.trackOperation('file_operation');
      }

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('BATCHING VIOLATION'),
      );

      const violations = batchEnforcer.violationWarnings;
      expect(violations.has('file_operation')).toBe(true);

      consoleSpy.mockRestore();
    });

    it('should generate appropriate recommendations', () => {
      // Create multiple violations
      for (let i = 0; i < 4; i++) {
        batchEnforcer.trackOperation('file_operation');
        batchEnforcer.trackOperation('mcp_tool');
      }

      const report = batchEnforcer.getBatchingReport();
      expect(report.recommendations).toContain('🔧 CRITICAL: Use BatchTool for all parallel operations');
      expect(report.recommendations).toContain('📁 File Operations: Use MultiEdit for multiple edits to same file');
      expect(report.recommendations).toContain('🤖 MCP Tools: Combine swarm operations in parallel');
    });

    it('should calculate compliance scores correctly', () => {
      // No violations = 100% compliance
      const cleanReport = batchEnforcer.getBatchingReport();
      expect(cleanReport.complianceScore).toBe(100);

      // Add violations
      for (let i = 0; i < 3; i++) {
        batchEnforcer.trackOperation('file_operation');
      }

      const violationReport = batchEnforcer.getBatchingReport();
      expect(violationReport.complianceScore).toBeLessThan(100);
    });

    it('should handle session ID generation correctly', () => {
      const sessionId1 = batchEnforcer.getCurrentSessionId();
      const sessionId2 = batchEnforcer.getCurrentSessionId();

      expect(sessionId1).toBe(sessionId2);
      expect(sessionId1).toMatch(/^\d+$/);
    });

    it('should filter recent operations correctly', () => {
      const now = Date.now();

      // Add old operation
      batchEnforcer.trackOperation('old_operation', now - 10000);

      // Add recent operations
      batchEnforcer.trackOperation('recent_operation', now - 1000);
      batchEnforcer.trackOperation('recent_operation', now - 500);

      const recentOps = batchEnforcer.getRecentOperations('recent_operation', 5000);
      expect(recentOps).toHaveLength(2);

      const oldOps = batchEnforcer.getRecentOperations('old_operation', 5000);
      expect(oldOps).toHaveLength(0);
    });
  });

  describe('ClaudeFlowEnhanced Initialization Edge Cases', () => {
    it('should initialize successfully with default options', async() => {
      const flow = await claudeFlow.initialize();

      expect(flow).toBe(claudeFlow);
      expect(mockRuvSwarm.initialize).toHaveBeenCalledWith({
        loadingStrategy: 'progressive',
        useSIMD: true,
        enableNeuralNetworks: true,
        debug: false,
      });
    });

    it('should handle custom initialization options', async() => {
      await claudeFlow.initialize({
        enforceBatching: false,
        enableSIMD: false,
        enableNeuralNetworks: false,
        debug: true,
      });

      expect(mockRuvSwarm.initialize).toHaveBeenCalledWith({
        loadingStrategy: 'progressive',
        useSIMD: false,
        enableNeuralNetworks: false,
        debug: true,
      });
    });

    it('should handle initialization failures gracefully', async() => {
      mockRuvSwarm.initialize.mockRejectedValueOnce(new Error('WASM load failed'));

      await expect(claudeFlow.initialize()).rejects.toThrow(ClaudeFlowError);
      await expect(claudeFlow.initialize()).rejects.toThrow('Initialization failed: WASM load failed');
    });

    it('should enable batch tool enforcement when requested', async() => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

      await claudeFlow.initialize({ enforceBatching: true });

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('BatchTool enforcement enabled'),
      );

      consoleSpy.mockRestore();
    });

    it('should handle MCP tools initialization failure', async() => {
      mockMcpTools.initialize.mockRejectedValueOnce(new Error('MCP init failed'));

      await expect(claudeFlow.initialize()).rejects.toThrow(ClaudeFlowError);
    });
  });

  describe('Workflow Creation Edge Cases', () => {
    beforeEach(async() => {
      await claudeFlow.initialize();
    });

    it('should create optimized workflow with full configuration', async() => {
      const workflowConfig = {
        id: 'complex-workflow',
        name: 'Complex Test Workflow',
        steps: [
          {
            id: 'step1',
            type: 'file_read',
            parallelizable: true,
            inputs: [],
            outputs: ['data1'],
          },
          {
            id: 'step2',
            type: 'mcp_tool_call',
            parallelizable: true,
            inputs: ['data1'],
            outputs: ['result1'],
          },
          {
            id: 'step3',
            type: 'neural_inference',
            parallelizable: true,
            inputs: ['result1'],
            outputs: ['prediction'],
          },
        ],
        parallelStrategy: 'aggressive',
        enableSIMD: true,
      };

      const workflow = await claudeFlow.createOptimizedWorkflow(workflowConfig);

      expect(workflow.id).toBe('complex-workflow');
      expect(workflow.name).toBe('Complex Test Workflow');
      expect(workflow.metrics.parallelizationRate).toBe(1.0); // 100% parallelizable
      expect(workflow.simdEnabled).toBe(true);
    });

    it('should handle workflows with low parallelization potential', async() => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const workflowConfig = {
        name: 'Sequential Workflow',
        steps: [
          { id: 'step1', type: 'custom', parallelizable: false },
          { id: 'step2', type: 'custom', parallelizable: false },
          { id: 'step3', type: 'file_read', parallelizable: true },
        ],
      };

      await claudeFlow.createOptimizedWorkflow(workflowConfig);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('low parallelization potential'),
      );

      consoleSpy.mockRestore();
    });

    it('should generate workflow IDs when not provided', async() => {
      const workflow = await claudeFlow.createOptimizedWorkflow({
        name: 'Auto ID Workflow',
        steps: [],
      });

      expect(workflow.id).toMatch(/^workflow_\d+$/);
    });

    it('should analyze step dependencies correctly', async() => {
      const steps = [
        {
          id: 'producer',
          type: 'file_read',
          inputs: [],
          outputs: ['file_content'],
        },
        {
          id: 'processor',
          type: 'data_processing',
          inputs: ['file_content'],
          outputs: ['processed_data'],
        },
        {
          id: 'consumer',
          type: 'file_write',
          inputs: ['processed_data'],
          outputs: [],
        },
      ];

      const workflow = await claudeFlow.createOptimizedWorkflow({
        name: 'Dependency Test',
        steps,
      });

      const processorStep = workflow.steps.find(s => s.id === 'processor');
      expect(processorStep.dependencies).toContain('producer');

      const consumerStep = workflow.steps.find(s => s.id === 'consumer');
      expect(consumerStep.dependencies).toContain('processor');
    });

    it('should handle empty or minimal workflows', async() => {
      const workflow = await claudeFlow.createOptimizedWorkflow({
        name: 'Empty Workflow',
        steps: [],
      });

      expect(workflow.metrics.totalSteps).toBe(0);
      expect(workflow.metrics.parallelizationRate).toBeNaN();
    });
  });

  describe('Workflow Execution Edge Cases', () => {
    let workflow;

    beforeEach(async() => {
      await claudeFlow.initialize();
      workflow = await claudeFlow.createOptimizedWorkflow({
        id: 'test-execution',
        name: 'Test Execution Workflow',
        steps: [
          { id: 'step1', type: 'mcp_tool_call', parallelizable: true, dependencies: [] },
          { id: 'step2', type: 'file_operation', parallelizable: true, dependencies: [] },
          { id: 'step3', type: 'neural_inference', parallelizable: true, dependencies: ['step1'] },
        ],
      });
    });

    it('should execute workflow successfully', async() => {
      const result = await claudeFlow.executeWorkflow('test-execution');

      expect(result.executionId).toMatch(/^exec_test-execution_\d+$/);
      expect(result.status).toBe('completed');
      expect(result.workflowId).toBe('test-execution');
      expect(result.results).toBeInstanceOf(Array);
      expect(result.metrics).toBeDefined();
      expect(result.batchingReport).toBeDefined();
    });

    it('should handle non-existent workflow', async() => {
      await expect(claudeFlow.executeWorkflow('non-existent')).rejects.toThrow(ClaudeFlowError);
      await expect(claudeFlow.executeWorkflow('non-existent')).rejects.toThrow('Workflow not found');
    });

    it('should create execution batches correctly', async() => {
      const steps = [
        { id: 'a', dependencies: [] },
        { id: 'b', dependencies: [] },
        { id: 'c', dependencies: ['a'] },
        { id: 'd', dependencies: ['b'] },
        { id: 'e', dependencies: ['c', 'd'] },
      ];

      const batches = claudeFlow.createExecutionBatches(steps);

      expect(batches).toHaveLength(3);
      expect(batches[0]).toHaveLength(2); // a, b
      expect(batches[1]).toHaveLength(2); // c, d
      expect(batches[2]).toHaveLength(1); // e
    });

    it('should detect circular dependencies', async() => {
      const stepsWithCircularDeps = [
        { id: 'a', dependencies: ['b'] },
        { id: 'b', dependencies: ['c'] },
        { id: 'c', dependencies: ['a'] },
      ];

      expect(() => claudeFlow.createExecutionBatches(stepsWithCircularDeps))
        .toThrow(ClaudeFlowError);
      expect(() => claudeFlow.createExecutionBatches(stepsWithCircularDeps))
        .toThrow('Circular dependency detected');
    });

    it('should handle step execution failures gracefully', async() => {
      // Mock a failing MCP tool call
      mockMcpTools.task_orchestrate.mockRejectedValueOnce(new Error('Task failed'));

      const failWorkflow = await claudeFlow.createOptimizedWorkflow({
        id: 'fail-test',
        name: 'Failure Test',
        steps: [{ id: 'fail-step', type: 'mcp_tool_call', toolName: 'task_orchestrate' }],
      });

      await expect(claudeFlow.executeWorkflow('fail-test')).rejects.toThrow(ClaudeFlowError);
    });

    it('should execute parallel steps in batches', async() => {
      const parallelWorkflow = await claudeFlow.createOptimizedWorkflow({
        id: 'parallel-test',
        name: 'Parallel Test',
        steps: [
          { id: 'p1', type: 'mcp_tool_call', dependencies: [] },
          { id: 'p2', type: 'mcp_tool_call', dependencies: [] },
          { id: 'p3', type: 'mcp_tool_call', dependencies: [] },
        ],
      });

      const result = await claudeFlow.executeWorkflow('parallel-test');

      expect(result.status).toBe('completed');
      expect(result.results).toHaveLength(3);
    });

    it('should handle mixed success and failure in parallel batches', async() => {
      // Mock one success and one failure
      mockMcpTools.agent_spawn
        .mockResolvedValueOnce({ agentId: 'success-agent' })
        .mockRejectedValueOnce(new Error('Agent spawn failed'));

      const mixedWorkflow = await claudeFlow.createOptimizedWorkflow({
        id: 'mixed-test',
        name: 'Mixed Results Test',
        steps: [
          { id: 'success', type: 'mcp_tool_call', requiresAgent: true },
          { id: 'failure', type: 'mcp_tool_call', requiresAgent: true },
        ],
      });

      const result = await claudeFlow.executeWorkflow('mixed-test');

      expect(result.results).toHaveLength(2);
      const successCount = result.results.filter(r => r.status === 'completed').length;
      const failureCount = result.results.filter(r => r.status === 'failed').length;

      expect(successCount).toBeGreaterThan(0);
      expect(failureCount).toBeGreaterThan(0);
    });
  });

  describe('Step Execution Edge Cases', () => {
    beforeEach(async() => {
      await claudeFlow.initialize();
    });

    it('should execute MCP tool steps correctly', async() => {
      const step = {
        type: 'mcp_tool_call',
        toolName: 'swarm_init',
        parameters: { topology: 'mesh' },
      };

      const result = await claudeFlow.executeStep(step, {}, null);

      expect(result.executionTime).toBeGreaterThan(0);
      expect(mockMcpTools.swarm_init).toHaveBeenCalledWith({ topology: 'mesh' });
    });

    it('should handle unknown MCP tool gracefully', async() => {
      const step = {
        type: 'mcp_tool_call',
        toolName: 'unknown_tool',
        parameters: {},
      };

      await expect(claudeFlow.executeStep(step, {}, null)).rejects.toThrow(ClaudeFlowError);
      await expect(claudeFlow.executeStep(step, {}, null)).rejects.toThrow('Unknown MCP tool');
    });

    it('should execute file operation steps', async() => {
      const step = {
        type: 'file_operation',
        operation: 'read',
        filePath: '/test/file.txt',
      };

      const result = await claudeFlow.executeStep(step, {}, null);

      expect(result.operation).toBe('read');
      expect(result.filePath).toBe('/test/file.txt');
      expect(result.success).toBe(true);
    });

    it('should execute neural inference steps with SIMD', async() => {
      const step = {
        type: 'neural_inference',
        modelConfig: { type: 'transformer' },
        inputData: { shape: [1, 512] },
        enableSIMD: true,
      };

      const result = await claudeFlow.executeStep(step, {}, null);

      expect(result.modelType).toBe('transformer');
      expect(result.simdEnabled).toBe(true);
      expect(result.performance.simdSpeedup).toBe(3.2);
    });

    it('should handle neural inference without neural networks enabled', async() => {
      // Mock features to disable neural networks
      claudeFlow.ruvSwarm = { features: { neural_networks: false } };

      const step = {
        type: 'neural_inference',
        modelConfig: { type: 'transformer' },
      };

      await expect(claudeFlow.executeStep(step, {}, null)).rejects.toThrow(ClaudeFlowError);
      await expect(claudeFlow.executeStep(step, {}, null)).rejects.toThrow('Neural networks not available');
    });

    it('should execute data processing steps with SIMD optimization', async() => {
      const step = {
        type: 'data_processing',
        operation: 'matrix_multiply',
        data: new Array(1000).fill(1),
        enableSIMD: true,
      };

      const result = await claudeFlow.executeStep(step, {}, null);

      expect(result.operation).toBe('matrix_multiply');
      expect(result.simdEnabled).toBe(true);
      expect(result.performance.simdSpeedup).toBe(4.1);
    });

    it('should execute generic steps', async() => {
      const step = {
        id: 'generic-step',
        type: 'custom_operation',
      };

      const result = await claudeFlow.executeStep(step, {}, null);

      expect(result.stepId).toBe('generic-step');
      expect(result.type).toBe('custom_operation');
      expect(result.status).toBe('completed');
    });

    it('should handle step execution timeouts', async() => {
      const step = {
        type: 'mcp_tool_call',
        toolName: 'slow_operation',
      };

      // Mock a slow operation
      claudeFlow.mcpTools.slow_operation = () => new Promise(() => {}); // Never resolves

      // This would timeout in a real scenario, but we'll mock the timeout behavior
      await expect(claudeFlow.executeStep(step, {}, null)).rejects.toThrow();
    });
  });

  describe('Performance Metrics Edge Cases', () => {
    beforeEach(async() => {
      await claudeFlow.initialize();
    });

    it('should calculate execution metrics correctly', async() => {
      const workflow = {
        steps: [
          { parallelizable: true, enableSIMD: true },
          { parallelizable: true, enableSIMD: false },
          { parallelizable: false, enableSIMD: true },
        ],
      };

      const coordination = {
        duration: 2500,
        startTime: Date.now() - 2500,
        endTime: Date.now(),
      };

      const metrics = claudeFlow.calculateExecutionMetrics(workflow, coordination);

      expect(metrics.totalSteps).toBe(3);
      expect(metrics.parallelSteps).toBe(2);
      expect(metrics.simdSteps).toBe(2);
      expect(metrics.parallelizationRate).toBeCloseTo(0.667, 2);
      expect(metrics.simdUtilization).toBeCloseTo(0.667, 2);
      expect(metrics.actualDuration).toBe(2500);
      expect(metrics.speedupFactor).toBeGreaterThan(1);
    });

    it('should generate comprehensive performance report', async() => {
      // Set up some test data
      claudeFlow.workflows.set('test-1', {
        id: 'test-1',
        name: 'Test 1',
        metrics: { parallelizationRate: 0.8, totalSteps: 5 },
      });

      claudeFlow.activeCoordinations.set('coord-1', {
        status: 'completed',
        metrics: { speedupFactor: 2.5 },
      });

      claudeFlow.activeCoordinations.set('coord-2', {
        status: 'running',
      });

      const report = claudeFlow.getPerformanceReport();

      expect(report.summary.totalWorkflows).toBe(1);
      expect(report.summary.activeCoordinations).toBe(1);
      expect(report.summary.completedCoordinations).toBe(1);
      expect(report.batching).toBeDefined();
      expect(report.features.batchingEnforced).toBe(true);
      expect(report.workflows).toHaveLength(1);
      expect(report.recommendations).toBeInstanceOf(Array);
    });

    it('should handle empty performance data', async() => {
      const report = claudeFlow.getPerformanceReport();

      expect(report.summary.totalWorkflows).toBe(0);
      expect(report.summary.averageSpeedup).toBeNaN();
    });
  });

  describe('Workflow Validation Edge Cases', () => {
    beforeEach(async() => {
      await claudeFlow.initialize();
    });

    it('should validate optimized workflows', async() => {
      const optimizedWorkflow = {
        steps: [
          { type: 'file_read', parallelizable: true },
          { type: 'neural_inference', parallelizable: true, enableSIMD: true },
          { type: 'mcp_tool_call', parallelizable: true },
        ],
      };

      const validation = claudeFlow.validateWorkflowOptimization(optimizedWorkflow);

      expect(validation.isOptimized).toBe(true);
      expect(validation.issues).toHaveLength(0);
      expect(validation.optimizationScore).toBe(100);
    });

    it('should detect optimization issues', async() => {
      const unoptimizedWorkflow = {
        steps: [
          { type: 'custom', parallelizable: false },
          { type: 'custom', parallelizable: false },
          { type: 'neural_inference', parallelizable: true, enableSIMD: false },
          { type: 'file_read', parallelizable: true },
        ],
      };

      const validation = claudeFlow.validateWorkflowOptimization(unoptimizedWorkflow);

      expect(validation.isOptimized).toBe(false);
      expect(validation.issues.length).toBeGreaterThan(0);
      expect(validation.optimizationScore).toBeLessThan(100);
      expect(validation.recommendations.length).toBeGreaterThan(0);
    });

    it('should calculate potential speedup correctly', async() => {
      const workflow = {
        steps: [
          { type: 'neural_inference', batchable: true },
          { type: 'data_processing', batchable: true },
          { type: 'mcp_tool_call', batchable: true },
          { type: 'file_read', batchable: false },
          { type: 'file_write', batchable: false },
        ],
      };

      const speedup = claudeFlow.calculatePotentialSpeedup(workflow);

      expect(speedup.parallel).toBe(2.8);
      expect(speedup.simd).toBe(3.5);
      expect(speedup.batching).toBe(1.8);
      expect(speedup.combined).toBeCloseTo(17.64, 1); // 2.8 * 3.5 * 1.8
    });

    it('should handle workflows with no optimization potential', async() => {
      const sequentialWorkflow = {
        steps: [
          { type: 'custom', batchable: false },
          { type: 'legacy', batchable: false },
        ],
      };

      const speedup = claudeFlow.calculatePotentialSpeedup(sequentialWorkflow);

      expect(speedup.parallel).toBe(1.0);
      expect(speedup.simd).toBe(1.0);
      expect(speedup.batching).toBe(1.0);
      expect(speedup.combined).toBe(1.0);
    });
  });

  describe('Global Functions Edge Cases', () => {
    it('should create and reuse claude flow instance', async() => {
      const flow1 = await getClaudeFlow();
      const flow2 = await getClaudeFlow();

      expect(flow1).toBe(flow2);
      expect(flow1).toBeInstanceOf(ClaudeFlowEnhanced);
    });

    it('should create optimized workflow through global function', async() => {
      const workflow = await createOptimizedWorkflow({
        name: 'Global Test',
        steps: [{ id: 'test', type: 'file_read' }],
      });

      expect(workflow.name).toBe('Global Test');
      expect(workflow.id).toBeDefined();
    });

    it('should execute workflow through global function', async() => {
      // First create a workflow
      const workflow = await createOptimizedWorkflow({
        id: 'global-exec-test',
        name: 'Global Execution Test',
        steps: [{ id: 'test', type: 'mcp_tool_call', toolName: 'swarm_init' }],
      });

      const result = await executeWorkflow('global-exec-test');

      expect(result.workflowId).toBe('global-exec-test');
      expect(result.status).toBe('completed');
    });

    it('should get performance report through global function', async() => {
      const report = await getPerformanceReport();

      expect(report).toBeDefined();
      expect(report.summary).toBeDefined();
      expect(report.batching).toBeDefined();
    });

    it('should validate workflow through global function', async() => {
      const testWorkflow = {
        steps: [
          { type: 'file_read', parallelizable: true },
          { type: 'neural_inference', enableSIMD: true },
        ],
      };

      const validation = await validateWorkflow(testWorkflow);

      expect(validation.isOptimized).toBeDefined();
      expect(validation.optimizationScore).toBeDefined();
    });
  });

  describe('Error Handling Edge Cases', () => {
    beforeEach(async() => {
      await claudeFlow.initialize();
    });

    it('should create ClaudeFlowError with correct properties', () => {
      const error = new ClaudeFlowError('Test error', 'TEST_CODE');

      expect(error.message).toBe('Test error');
      expect(error.name).toBe('ClaudeFlowError');
      expect(error.code).toBe('TEST_CODE');
      expect(error).toBeInstanceOf(Error);
    });

    it('should use default error code when not provided', () => {
      const error = new ClaudeFlowError('Default code test');

      expect(error.code).toBe('CLAUDE_FLOW_ERROR');
    });

    it('should handle context updates correctly', async() => {
      const context = {};
      const results = [
        { stepId: 'step1', result: { data: 'result1' } },
        { stepId: 'step2', result: { data: 'result2' } },
        { result: { data: 'no-step-id' } }, // Should be ignored
      ];

      claudeFlow.updateExecutionContext(context, results);

      expect(context.step1).toEqual({ data: 'result1' });
      expect(context.step2).toEqual({ data: 'result2' });
      expect(Object.keys(context)).toHaveLength(2);
    });

    it('should handle missing workflow dependencies gracefully', async() => {
      const workflow = await claudeFlow.createOptimizedWorkflow({
        id: 'missing-deps',
        name: 'Missing Dependencies Test',
        steps: [
          { id: 'step1', dependencies: ['non-existent'] },
        ],
      });

      await expect(claudeFlow.executeWorkflow('missing-deps')).rejects.toThrow(ClaudeFlowError);
    });
  });

  describe('End-to-End Claude Flow Tests', () => {
    it('should complete full workflow lifecycle', async() => {
      // Step 1: Initialize Claude Flow
      await claudeFlow.initialize({
        enforceBatching: true,
        enableSIMD: true,
        enableNeuralNetworks: true,
      });

      // Step 2: Create optimized workflow
      const workflow = await claudeFlow.createOptimizedWorkflow({
        id: 'e2e-test',
        name: 'End-to-End Test Workflow',
        steps: [
          {
            id: 'init',
            type: 'mcp_tool_call',
            toolName: 'swarm_init',
            parameters: { topology: 'mesh' },
            parallelizable: true,
            dependencies: [],
          },
          {
            id: 'spawn-agents',
            type: 'mcp_tool_call',
            toolName: 'agent_spawn',
            parameters: { type: 'coordinator' },
            parallelizable: true,
            dependencies: ['init'],
          },
          {
            id: 'neural-task',
            type: 'neural_inference',
            modelConfig: { type: 'transformer' },
            inputData: { shape: [1, 512] },
            enableSIMD: true,
            parallelizable: true,
            dependencies: [],
          },
          {
            id: 'orchestrate',
            type: 'mcp_tool_call',
            toolName: 'task_orchestrate',
            parameters: { task: 'final-task' },
            parallelizable: false,
            dependencies: ['spawn-agents', 'neural-task'],
          },
        ],
        parallelStrategy: 'aggressive',
      });

      expect(workflow.id).toBe('e2e-test');
      expect(workflow.metrics.totalSteps).toBe(4);

      // Step 3: Validate workflow optimization
      const validation = claudeFlow.validateWorkflowOptimization(workflow);
      expect(validation.optimizationScore).toBeGreaterThan(50);

      // Step 4: Execute workflow
      const execution = await claudeFlow.executeWorkflow('e2e-test');

      expect(execution.status).toBe('completed');
      expect(execution.workflowId).toBe('e2e-test');
      expect(execution.results).toHaveLength(4);
      expect(execution.metrics).toBeDefined();

      // Step 5: Check performance report
      const report = claudeFlow.getPerformanceReport();

      expect(report.summary.totalWorkflows).toBe(1);
      expect(report.summary.completedCoordinations).toBe(1);
      expect(report.workflows[0].id).toBe('e2e-test');
    });

    it('should handle complex parallel execution scenarios', async() => {
      await claudeFlow.initialize();

      // Create workflow with complex dependencies
      const complexWorkflow = await claudeFlow.createOptimizedWorkflow({
        id: 'complex-parallel',
        name: 'Complex Parallel Workflow',
        steps: [
          // Parallel batch 1
          { id: 'a1', type: 'file_read', dependencies: [] },
          { id: 'a2', type: 'file_read', dependencies: [] },
          { id: 'a3', type: 'file_read', dependencies: [] },

          // Parallel batch 2 (depends on batch 1)
          { id: 'b1', type: 'data_processing', dependencies: ['a1', 'a2'] },
          { id: 'b2', type: 'data_processing', dependencies: ['a2', 'a3'] },

          // Parallel batch 3 (depends on batch 2)
          { id: 'c1', type: 'neural_inference', dependencies: ['b1'] },
          { id: 'c2', type: 'neural_inference', dependencies: ['b2'] },

          // Final step (depends on all previous)
          { id: 'final', type: 'mcp_tool_call', dependencies: ['c1', 'c2'] },
        ],
      });

      const execution = await claudeFlow.executeWorkflow('complex-parallel');

      expect(execution.status).toBe('completed');
      expect(execution.results).toHaveLength(8);

      // Verify execution order through timing
      const timing = execution.results.reduce((acc, result) => {
        acc[result.stepId] = result.executionTime || 0;
        return acc;
      }, {});

      // All steps should have been executed
      expect(Object.keys(timing)).toHaveLength(8);
    });

    it('should demonstrate batch tool enforcement benefits', async() => {
      await claudeFlow.initialize({ enforceBatching: true });

      // Create workflow that would benefit from batching
      const batchWorkflow = await claudeFlow.createOptimizedWorkflow({
        id: 'batch-demo',
        name: 'Batch Demonstration',
        steps: Array.from({ length: 10 }, (_, i) => ({
          id: `parallel-${i}`,
          type: 'mcp_tool_call',
          toolName: 'agent_spawn',
          parallelizable: true,
          dependencies: [],
        })),
      });

      const execution = await claudeFlow.executeWorkflow('batch-demo');

      expect(execution.status).toBe('completed');
      expect(execution.results).toHaveLength(10);

      // Should show good batching compliance
      expect(execution.batchingReport.complianceScore).toBeGreaterThan(80);

      // Should demonstrate speedup
      expect(execution.metrics.speedupFactor).toBeGreaterThan(2);
    });

    it('should handle error recovery and partial execution', async() => {
      await claudeFlow.initialize();

      // Mock some operations to fail
      mockMcpTools.agent_spawn
        .mockResolvedValueOnce({ agentId: 'success-1' })
        .mockRejectedValueOnce(new Error('Network timeout'))
        .mockResolvedValueOnce({ agentId: 'success-2' })
        .mockRejectedValueOnce(new Error('Service unavailable'));

      const recoveryWorkflow = await claudeFlow.createOptimizedWorkflow({
        id: 'error-recovery',
        name: 'Error Recovery Test',
        steps: [
          { id: 'task1', type: 'mcp_tool_call', toolName: 'agent_spawn', requiresAgent: true },
          { id: 'task2', type: 'mcp_tool_call', toolName: 'agent_spawn', requiresAgent: true },
          { id: 'task3', type: 'mcp_tool_call', toolName: 'agent_spawn', requiresAgent: true },
          { id: 'task4', type: 'mcp_tool_call', toolName: 'agent_spawn', requiresAgent: true },
        ],
      });

      const execution = await claudeFlow.executeWorkflow('error-recovery');

      expect(execution.results).toHaveLength(4);

      const successful = execution.results.filter(r => r.status === 'completed');
      const failed = execution.results.filter(r => r.status === 'failed');

      expect(successful).toHaveLength(2);
      expect(failed).toHaveLength(2);
    });
  });
});
