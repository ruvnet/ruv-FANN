/**
 * Edge Cases and E2E Tests for src/mcp-daa-tools.js
 * Comprehensive coverage for DAA MCP Tools integration
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { DAA_MCPTools, daaMcpTools } from '../../src/mcp-daa-tools.js';

// Mock the DAA service
const mockDaaService = {
  initialize: vi.fn().mockResolvedValue(true),
  getCapabilities: vi.fn().mockReturnValue(['autonomous', 'learning', 'coordination']),
  createAgent: vi.fn().mockImplementation(async(config) => ({
    id: config.id,
    cognitivePattern: config.cognitivePattern,
    capabilities: new Set(config.capabilities || []),
    created: Date.now(),
  })),
  adaptAgent: vi.fn().mockImplementation(async(agentId, feedback) => ({
    previousPattern: 'convergent',
    newPattern: 'adaptive',
    improvement: 0.15,
    insights: ['Pattern adaptation successful', 'Performance improved'],
  })),
  createWorkflow: vi.fn().mockImplementation(async(id, steps, dependencies) => ({
    id,
    steps: steps || [],
    dependencies: dependencies || {},
    status: 'created',
  })),
  executeWorkflow: vi.fn().mockImplementation(async(workflowId, options) => ({
    complete: true,
    stepsCompleted: 5,
    totalSteps: 5,
    executionTime: 1250,
    agentsInvolved: options.agentIds || ['agent-1', 'agent-2'],
    stepResults: [
      { step: 1, result: 'success' },
      { step: 2, result: 'success' },
      { step: 3, result: 'success' },
      { step: 4, result: 'success' },
      { step: 5, result: 'success' },
    ],
  })),
  shareKnowledge: vi.fn().mockImplementation(async(sourceId, targetIds, knowledge) => ({
    updatedAgents: targetIds.length,
    transferRate: 0.95,
  })),
  getAgentLearningStatus: vi.fn().mockImplementation(async(agentId) => ({
    totalCycles: 42,
    avgProficiency: 0.87,
    domains: ['language', 'vision', 'reasoning'],
    adaptationRate: 0.23,
    neuralModelsCount: 7,
    persistentMemorySize: 1024 * 1024,
    performanceTrend: 'improving',
    detailedMetrics: {
      accuracy: 0.92,
      speed: 0.85,
      efficiency: 0.78,
    },
  })),
  getSystemLearningStatus: vi.fn().mockImplementation(async() => ({
    totalCycles: 156,
    avgProficiency: 0.83,
    domains: ['nlp', 'cv', 'rl', 'graph'],
    adaptationRate: 0.19,
    neuralModelsCount: 12,
    persistentMemorySize: 8 * 1024 * 1024,
    performanceTrend: 'stable',
  })),
  analyzeCognitivePatterns: vi.fn().mockImplementation(async(agentId) => ({
    patterns: ['convergent', 'systems', 'adaptive'],
    effectiveness: {
      convergent: 0.89,
      systems: 0.76,
      adaptive: 0.91,
    },
    recommendations: ['Increase adaptive pattern usage', 'Optimize convergent thinking'],
    optimizationScore: 0.85,
  })),
  setCognitivePattern: vi.fn().mockImplementation(async(agentId, pattern) => ({
    previousPattern: 'convergent',
    success: true,
    expectedImprovement: 0.12,
  })),
  performMetaLearning: vi.fn().mockImplementation(async(config) => ({
    knowledgeItems: 47,
    updatedAgents: 8,
    proficiencyGain: 0.18,
    insights: ['Cross-domain transfer successful', 'New patterns emerged'],
  })),
  getPerformanceMetrics: vi.fn().mockImplementation(async(config) => ({
    totalAgents: 15,
    activeAgents: 12,
    tasksCompleted: 1847,
    avgTaskTime: 2341,
    learningCycles: 89,
    successRate: 0.94,
    adaptationScore: 0.87,
    knowledgeSharingCount: 234,
    crossDomainTransfers: 45,
    tokenReduction: 0.32,
    parallelGain: 2.8,
    memoryOptimization: 0.45,
    neuralModelsActive: 9,
    avgInferenceTime: 15.6,
    totalTrainingIterations: 12847,
  })),
};

// Mock the enhanced MCP tools
const mockMcpTools = {
  recordToolMetrics: vi.fn(),
  activeSwarms: new Map([
    ['test-swarm-1', {
      agents: new Map(),
      maxAgents: 10,
    }],
    ['test-swarm-2', {
      agents: new Map(),
      maxAgents: 5,
    }],
  ]),
};

describe('MCP DAA Tools Edge Cases and E2E Tests', () => {
  let daaTools;

  beforeEach(() => {
    vi.clearAllMocks();

    // Create fresh instance for each test
    daaTools = new DAA_MCPTools(mockMcpTools);

    // Mock the daaService import
    vi.doMock('../../src/daa-service.js', () => ({
      daaService: mockDaaService,
    }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Initialization Edge Cases', () => {
    it('should handle multiple initialization calls gracefully', async() => {
      await daaTools.ensureInitialized();
      await daaTools.ensureInitialized();
      await daaTools.ensureInitialized();

      expect(mockDaaService.initialize).toHaveBeenCalledTimes(1);
      expect(daaTools.daaInitialized).toBe(true);
    });

    it('should handle initialization failure', async() => {
      mockDaaService.initialize.mockRejectedValueOnce(new Error('Init failed'));

      await expect(daaTools.ensureInitialized()).rejects.toThrow('Init failed');
      expect(daaTools.daaInitialized).toBe(false);
    });

    it('should handle concurrent initialization calls', async() => {
      const promises = Array.from({ length: 5 }, () => daaTools.ensureInitialized());

      await Promise.all(promises);

      expect(mockDaaService.initialize).toHaveBeenCalledTimes(1);
      expect(daaTools.daaInitialized).toBe(true);
    });
  });

  describe('daa_init Edge Cases', () => {
    it('should handle successful initialization with default parameters', async() => {
      const result = await daaTools.daa_init({});

      expect(result).toEqual({
        initialized: true,
        features: {
          autonomousLearning: true,
          peerCoordination: true,
          persistenceMode: 'auto',
          neuralIntegration: true,
          cognitivePatterns: 6,
        },
        capabilities: ['autonomous', 'learning', 'coordination'],
        timestamp: expect.any(String),
      });

      expect(mockMcpTools.recordToolMetrics).toHaveBeenCalledWith(
        'daa_init',
        expect.any(Number),
        'success',
      );
    });

    it('should handle custom initialization parameters', async() => {
      const params = {
        enableLearning: false,
        enableCoordination: false,
        persistenceMode: 'memory',
      };

      const result = await daaTools.daa_init(params);

      expect(result.features).toEqual({
        autonomousLearning: false,
        peerCoordination: false,
        persistenceMode: 'memory',
        neuralIntegration: true,
        cognitivePatterns: 6,
      });
    });

    it('should handle initialization errors gracefully', async() => {
      mockDaaService.initialize.mockRejectedValueOnce(new Error('Service unavailable'));

      await expect(daaTools.daa_init({})).rejects.toThrow('Service unavailable');

      expect(mockMcpTools.recordToolMetrics).toHaveBeenCalledWith(
        'daa_init',
        expect.any(Number),
        'error',
        'Service unavailable',
      );
    });

    it('should handle missing metrics recorder gracefully', async() => {
      const daaToolsNoMetrics = new DAA_MCPTools(null);

      const result = await daaToolsNoMetrics.daa_init({});

      expect(result.initialized).toBe(true);
      // Should not throw even without metrics recorder
    });
  });

  describe('daa_agent_create Edge Cases', () => {
    it('should handle successful agent creation with minimal parameters', async() => {
      const params = { id: 'test-agent-1' };

      const result = await daaTools.daa_agent_create(params);

      expect(result).toEqual({
        agent_id: 'test-agent-1',
        swarm_id: 'test-swarm-1',
        cognitive_pattern: 'adaptive',
        capabilities: [],
        learning_enabled: false,
        memory_enabled: true,
        status: 'active',
        created_at: expect.any(String),
      });
    });

    it('should handle agent creation with full configuration', async() => {
      const params = {
        id: 'full-agent',
        capabilities: ['reasoning', 'learning', 'adaptation'],
        cognitivePattern: 'divergent',
        learningRate: 0.005,
        enableMemory: false,
      };

      const result = await daaTools.daa_agent_create(params);

      expect(result).toEqual({
        agent_id: 'full-agent',
        swarm_id: 'test-swarm-1',
        cognitive_pattern: 'divergent',
        capabilities: ['reasoning', 'learning', 'adaptation'],
        learning_enabled: true,
        memory_enabled: false,
        status: 'active',
        created_at: expect.any(String),
      });
    });

    it('should handle missing agent ID error', async() => {
      const params = { capabilities: ['test'] };

      await expect(daaTools.daa_agent_create(params)).rejects.toThrow('Agent ID is required');

      expect(mockMcpTools.recordToolMetrics).toHaveBeenCalledWith(
        'daa_agent_create',
        expect.any(Number),
        'error',
        'Agent ID is required',
      );
    });

    it('should handle swarm assignment when all swarms are full', async() => {
      // Fill up all available swarms
      const swarm1 = mockMcpTools.activeSwarms.get('test-swarm-1');
      const swarm2 = mockMcpTools.activeSwarms.get('test-swarm-2');

      // Fill swarm1 to capacity
      for (let i = 0; i < 10; i++) {
        swarm1.agents.set(`agent-${i}`, { id: `agent-${i}` });
      }

      // Fill swarm2 to capacity
      for (let i = 0; i < 5; i++) {
        swarm2.agents.set(`agent-full-${i}`, { id: `agent-full-${i}` });
      }

      const result = await daaTools.daa_agent_create({ id: 'overflow-agent' });

      expect(result.swarm_id).toBe('daa-default-swarm');
    });

    it('should handle no active swarms scenario', async() => {
      const daaToolsNoSwarms = new DAA_MCPTools({ recordToolMetrics: vi.fn() });

      const result = await daaToolsNoSwarms.daa_agent_create({ id: 'no-swarm-agent' });

      expect(result.swarm_id).toBe('daa-default-swarm');
    });

    it('should handle agent creation service errors', async() => {
      mockDaaService.createAgent.mockRejectedValueOnce(new Error('Agent creation failed'));

      await expect(daaTools.daa_agent_create({ id: 'fail-agent' })).rejects.toThrow('Agent creation failed');
    });
  });

  describe('daa_agent_adapt Edge Cases', () => {
    it('should handle successful agent adaptation with minimal parameters', async() => {
      const params = { agentId: 'adapt-test' };

      const result = await daaTools.daa_agent_adapt(params);

      expect(result).toEqual({
        agent_id: 'adapt-test',
        adaptation_complete: true,
        previous_pattern: 'convergent',
        new_pattern: 'adaptive',
        performance_improvement: 0.15,
        learning_insights: ['Pattern adaptation successful', 'Performance improved'],
        timestamp: expect.any(String),
      });
    });

    it('should handle adaptation with full feedback', async() => {
      const params = {
        agentId: 'detailed-adapt',
        feedback: 'Agent performed well but could improve reasoning speed',
        performanceScore: 0.85,
        suggestions: ['Optimize inference pipeline', 'Increase attention weights'],
      };

      const result = await daaTools.daa_agent_adapt(params);

      expect(result.agent_id).toBe('detailed-adapt');
      expect(result.adaptation_complete).toBe(true);
    });

    it('should handle missing agent ID error', async() => {
      await expect(daaTools.daa_agent_adapt({})).rejects.toThrow('Agent ID is required');
    });

    it('should handle adaptation service failures', async() => {
      mockDaaService.adaptAgent.mockRejectedValueOnce(new Error('Adaptation failed'));

      await expect(daaTools.daa_agent_adapt({ agentId: 'fail-adapt' })).rejects.toThrow('Adaptation failed');
    });

    it('should handle invalid performance scores gracefully', async() => {
      const params = {
        agentId: 'invalid-score',
        performanceScore: -0.5, // Invalid negative score
      };

      // Should still work, DAA service should handle validation
      const result = await daaTools.daa_agent_adapt(params);
      expect(result.agent_id).toBe('invalid-score');
    });
  });

  describe('daa_workflow_create Edge Cases', () => {
    it('should handle successful workflow creation', async() => {
      const params = {
        id: 'test-workflow',
        name: 'Test Workflow',
        steps: [
          { id: 'step1', action: 'initialize' },
          { id: 'step2', action: 'process' },
          { id: 'step3', action: 'finalize' },
        ],
        dependencies: { step2: ['step1'], step3: ['step2'] },
      };

      const result = await daaTools.daa_workflow_create(params);

      expect(result).toEqual({
        workflow_id: 'test-workflow',
        name: 'Test Workflow',
        total_steps: 3,
        execution_strategy: 'parallel',
        dependencies_count: 2,
        status: 'created',
        created_at: expect.any(String),
      });
    });

    it('should handle missing required parameters', async() => {
      await expect(daaTools.daa_workflow_create({})).rejects.toThrow('Workflow ID and name are required');

      await expect(daaTools.daa_workflow_create({ id: 'test' })).rejects.toThrow('Workflow ID and name are required');

      await expect(daaTools.daa_workflow_create({ name: 'test' })).rejects.toThrow('Workflow ID and name are required');
    });

    it('should handle different execution strategies', async() => {
      const params = {
        id: 'sequential-workflow',
        name: 'Sequential Test',
        strategy: 'sequential',
      };

      const result = await daaTools.daa_workflow_create(params);

      expect(result.execution_strategy).toBe('sequential');
    });

    it('should handle empty steps and dependencies', async() => {
      const params = {
        id: 'minimal-workflow',
        name: 'Minimal Workflow',
      };

      const result = await daaTools.daa_workflow_create(params);

      expect(result.total_steps).toBe(0);
      expect(result.dependencies_count).toBe(0);
    });

    it('should handle workflow creation service errors', async() => {
      mockDaaService.createWorkflow.mockRejectedValueOnce(new Error('Workflow creation failed'));

      await expect(daaTools.daa_workflow_create({
        id: 'fail-workflow',
        name: 'Failing Workflow',
      })).rejects.toThrow('Workflow creation failed');
    });
  });

  describe('daa_workflow_execute Edge Cases', () => {
    it('should handle successful workflow execution', async() => {
      const params = {
        workflowId: 'execute-test',
        agentIds: ['agent-1', 'agent-2', 'agent-3'],
        parallelExecution: true,
      };

      const result = await daaTools.daa_workflow_execute(params);

      expect(result).toEqual({
        workflow_id: 'execute-test',
        execution_complete: true,
        steps_completed: 5,
        total_steps: 5,
        execution_time_ms: 1250,
        agents_involved: ['agent-1', 'agent-2', 'agent-3'],
        results: expect.any(Array),
        timestamp: expect.any(String),
      });
    });

    it('should handle missing workflow ID', async() => {
      await expect(daaTools.daa_workflow_execute({})).rejects.toThrow('Workflow ID is required');
    });

    it('should handle execution with default parameters', async() => {
      const params = { workflowId: 'default-test' };

      const result = await daaTools.daa_workflow_execute(params);

      expect(result.workflow_id).toBe('default-test');
      expect(result.agents_involved).toEqual(['agent-1', 'agent-2']); // Default from mock
    });

    it('should handle execution service failures', async() => {
      mockDaaService.executeWorkflow.mockRejectedValueOnce(new Error('Execution failed'));

      await expect(daaTools.daa_workflow_execute({
        workflowId: 'fail-execute',
      })).rejects.toThrow('Execution failed');
    });

    it('should handle sequential execution mode', async() => {
      const params = {
        workflowId: 'sequential-test',
        parallelExecution: false,
      };

      const result = await daaTools.daa_workflow_execute(params);

      expect(result.execution_complete).toBe(true);
    });
  });

  describe('daa_knowledge_share Edge Cases', () => {
    it('should handle successful knowledge sharing', async() => {
      const params = {
        sourceAgentId: 'teacher-agent',
        targetAgentIds: ['student-1', 'student-2', 'student-3'],
        knowledgeDomain: 'natural-language-processing',
        knowledgeContent: {
          concepts: ['tokenization', 'embeddings', 'attention'],
          expertise_level: 'intermediate',
          confidence: 0.92,
        },
      };

      const result = await daaTools.daa_knowledge_share(params);

      expect(result).toEqual({
        source_agent: 'teacher-agent',
        target_agents: ['student-1', 'student-2', 'student-3'],
        knowledge_domain: 'natural-language-processing',
        sharing_complete: true,
        agents_updated: 3,
        knowledge_transfer_rate: 0.95,
        timestamp: expect.any(String),
      });
    });

    it('should handle missing required parameters', async() => {
      await expect(daaTools.daa_knowledge_share({})).rejects.toThrow('Source and target agent IDs are required');

      await expect(daaTools.daa_knowledge_share({
        sourceAgentId: 'teacher',
      })).rejects.toThrow('Source and target agent IDs are required');

      await expect(daaTools.daa_knowledge_share({
        targetAgentIds: ['student'],
      })).rejects.toThrow('Source and target agent IDs are required');
    });

    it('should handle empty target agents list', async() => {
      const params = {
        sourceAgentId: 'teacher',
        targetAgentIds: [],
      };

      await expect(daaTools.daa_knowledge_share(params)).rejects.toThrow('Source and target agent IDs are required');
    });

    it('should handle knowledge sharing service failures', async() => {
      mockDaaService.shareKnowledge.mockRejectedValueOnce(new Error('Knowledge sharing failed'));

      await expect(daaTools.daa_knowledge_share({
        sourceAgentId: 'fail-teacher',
        targetAgentIds: ['student'],
      })).rejects.toThrow('Knowledge sharing failed');
    });

    it('should handle undefined knowledge content', async() => {
      const params = {
        sourceAgentId: 'teacher',
        targetAgentIds: ['student'],
        knowledgeContent: undefined,
      };

      const result = await daaTools.daa_knowledge_share(params);
      expect(result.sharing_complete).toBe(true);
    });
  });

  describe('daa_learning_status Edge Cases', () => {
    it('should handle specific agent learning status', async() => {
      const params = { agentId: 'learning-agent-1', detailed: true };

      const result = await daaTools.daa_learning_status(params);

      expect(result).toEqual({
        agent_id: 'learning-agent-1',
        total_learning_cycles: 42,
        average_proficiency: 0.87,
        knowledge_domains: ['language', 'vision', 'reasoning'],
        adaptation_rate: 0.23,
        neural_models_active: 7,
        cross_session_memory: 1048576,
        performance_trend: 'improving',
        detailed_metrics: {
          accuracy: 0.92,
          speed: 0.85,
          efficiency: 0.78,
        },
        timestamp: expect.any(String),
      });
    });

    it('should handle system-wide learning status', async() => {
      const params = {};

      const result = await daaTools.daa_learning_status(params);

      expect(result.agent_id).toBe('all');
      expect(result.total_learning_cycles).toBe(156);
      expect(result.knowledge_domains).toEqual(['nlp', 'cv', 'rl', 'graph']);
    });

    it('should handle detailed metrics only when requested', async() => {
      const params = { agentId: 'test-agent', detailed: false };

      const result = await daaTools.daa_learning_status(params);

      expect(result).not.toHaveProperty('detailed_metrics');
    });

    it('should handle learning status service failures', async() => {
      mockDaaService.getAgentLearningStatus.mockRejectedValueOnce(new Error('Status unavailable'));

      await expect(daaTools.daa_learning_status({
        agentId: 'fail-agent',
      })).rejects.toThrow('Status unavailable');
    });

    it('should handle system learning status service failures', async() => {
      mockDaaService.getSystemLearningStatus.mockRejectedValueOnce(new Error('System status unavailable'));

      await expect(daaTools.daa_learning_status({})).rejects.toThrow('System status unavailable');
    });
  });

  describe('daa_cognitive_pattern Edge Cases', () => {
    it('should handle cognitive pattern analysis', async() => {
      const params = { agentId: 'analyze-agent', analyze: true };

      const result = await daaTools.daa_cognitive_pattern(params);

      expect(result).toEqual({
        analysis_type: 'cognitive_pattern',
        agent_id: 'analyze-agent',
        current_patterns: ['convergent', 'systems', 'adaptive'],
        pattern_effectiveness: {
          convergent: 0.89,
          systems: 0.76,
          adaptive: 0.91,
        },
        recommendations: ['Increase adaptive pattern usage', 'Optimize convergent thinking'],
        optimization_potential: 0.85,
        timestamp: expect.any(String),
      });
    });

    it('should handle cognitive pattern change', async() => {
      const params = {
        agentId: 'pattern-agent',
        pattern: 'lateral',
        analyze: false,
      };

      const result = await daaTools.daa_cognitive_pattern(params);

      expect(result).toEqual({
        agent_id: 'pattern-agent',
        previous_pattern: 'convergent',
        new_pattern: 'lateral',
        adaptation_success: true,
        expected_improvement: 0.12,
        timestamp: expect.any(String),
      });
    });

    it('should handle pattern change without agent ID', async() => {
      const params = { pattern: 'divergent', analyze: false };

      await expect(daaTools.daa_cognitive_pattern(params)).rejects.toThrow('Agent ID and pattern are required for pattern change');
    });

    it('should handle pattern change without pattern', async() => {
      const params = { agentId: 'test-agent', analyze: false };

      await expect(daaTools.daa_cognitive_pattern(params)).rejects.toThrow('Agent ID and pattern are required for pattern change');
    });

    it('should handle analysis service failures', async() => {
      mockDaaService.analyzeCognitivePatterns.mockRejectedValueOnce(new Error('Analysis failed'));

      await expect(daaTools.daa_cognitive_pattern({
        agentId: 'fail-agent',
        analyze: true,
      })).rejects.toThrow('Analysis failed');
    });

    it('should handle pattern setting service failures', async() => {
      mockDaaService.setCognitivePattern.mockRejectedValueOnce(new Error('Pattern change failed'));

      await expect(daaTools.daa_cognitive_pattern({
        agentId: 'fail-agent',
        pattern: 'adaptive',
      })).rejects.toThrow('Pattern change failed');
    });
  });

  describe('daa_meta_learning Edge Cases', () => {
    it('should handle successful meta-learning transfer', async() => {
      const params = {
        sourceDomain: 'computer-vision',
        targetDomain: 'natural-language-processing',
        transferMode: 'adaptive',
        agentIds: ['agent-1', 'agent-2', 'agent-3'],
      };

      const result = await daaTools.daa_meta_learning(params);

      expect(result).toEqual({
        meta_learning_complete: true,
        source_domain: 'computer-vision',
        target_domain: 'natural-language-processing',
        transfer_mode: 'adaptive',
        knowledge_transferred: 47,
        agents_updated: 8,
        domain_proficiency_gain: 0.18,
        cross_domain_insights: ['Cross-domain transfer successful', 'New patterns emerged'],
        timestamp: expect.any(String),
      });
    });

    it('should handle meta-learning with default parameters', async() => {
      const params = {
        sourceDomain: 'reinforcement-learning',
        targetDomain: 'supervised-learning',
      };

      const result = await daaTools.daa_meta_learning(params);

      expect(result.transfer_mode).toBe('adaptive');
      expect(result.meta_learning_complete).toBe(true);
    });

    it('should handle different transfer modes', async() => {
      const params = {
        sourceDomain: 'domain-a',
        targetDomain: 'domain-b',
        transferMode: 'direct',
      };

      const result = await daaTools.daa_meta_learning(params);

      expect(result.transfer_mode).toBe('direct');
    });

    it('should handle empty agent IDs list', async() => {
      const params = {
        sourceDomain: 'domain-a',
        targetDomain: 'domain-b',
        agentIds: [],
      };

      const result = await daaTools.daa_meta_learning(params);

      expect(result.meta_learning_complete).toBe(true);
    });

    it('should handle meta-learning service failures', async() => {
      mockDaaService.performMetaLearning.mockRejectedValueOnce(new Error('Meta-learning failed'));

      await expect(daaTools.daa_meta_learning({
        sourceDomain: 'fail-source',
        targetDomain: 'fail-target',
      })).rejects.toThrow('Meta-learning failed');
    });
  });

  describe('daa_performance_metrics Edge Cases', () => {
    it('should handle comprehensive performance metrics', async() => {
      const params = { category: 'all', timeRange: '24h' };

      const result = await daaTools.daa_performance_metrics(params);

      expect(result).toEqual({
        metrics_category: 'all',
        time_range: '24h',
        system_metrics: {
          total_agents: 15,
          active_agents: 12,
          autonomous_tasks_completed: 1847,
          average_task_time_ms: 2341,
          learning_cycles_completed: 89,
        },
        performance_metrics: {
          task_success_rate: 0.94,
          adaptation_effectiveness: 0.87,
          knowledge_sharing_events: 234,
          cross_domain_transfers: 45,
        },
        efficiency_metrics: {
          token_reduction: 0.32,
          parallel_execution_gain: 2.8,
          memory_optimization: 0.45,
        },
        neural_metrics: {
          models_active: 9,
          inference_speed_ms: 15.6,
          training_iterations: 12847,
        },
        timestamp: expect.any(String),
      });
    });

    it('should handle different metric categories', async() => {
      const categories = ['system', 'performance', 'efficiency', 'neural'];

      for (const category of categories) {
        const result = await daaTools.daa_performance_metrics({ category });
        expect(result.metrics_category).toBe(category);
      }
    });

    it('should handle default parameters', async() => {
      const result = await daaTools.daa_performance_metrics({});

      expect(result.metrics_category).toBe('all');
      expect(result.time_range).toBe('1h');
    });

    it('should handle performance metrics service failures', async() => {
      mockDaaService.getPerformanceMetrics.mockRejectedValueOnce(new Error('Metrics unavailable'));

      await expect(daaTools.daa_performance_metrics({
        category: 'fail',
      })).rejects.toThrow('Metrics unavailable');
    });
  });

  describe('Tool Definitions Edge Cases', () => {
    it('should return valid tool definitions', () => {
      const definitions = daaTools.getToolDefinitions();

      expect(definitions).toBeInstanceOf(Array);
      expect(definitions).toHaveLength(10);

      const toolNames = definitions.map(def => def.name);
      expect(toolNames).toEqual([
        'daa_init',
        'daa_agent_create',
        'daa_agent_adapt',
        'daa_workflow_create',
        'daa_workflow_execute',
        'daa_knowledge_share',
        'daa_learning_status',
        'daa_cognitive_pattern',
        'daa_meta_learning',
        'daa_performance_metrics',
      ]);
    });

    it('should have valid schemas for all tools', () => {
      const definitions = daaTools.getToolDefinitions();

      definitions.forEach(definition => {
        expect(definition).toHaveProperty('name');
        expect(definition).toHaveProperty('description');
        expect(definition).toHaveProperty('inputSchema');
        expect(definition.inputSchema).toHaveProperty('type');
        expect(definition.inputSchema.type).toBe('object');
      });
    });

    it('should handle required parameters correctly', () => {
      const definitions = daaTools.getToolDefinitions();

      const agentCreateDef = definitions.find(def => def.name === 'daa_agent_create');
      expect(agentCreateDef.inputSchema.required).toEqual(['id']);

      const workflowCreateDef = definitions.find(def => def.name === 'daa_workflow_create');
      expect(workflowCreateDef.inputSchema.required).toEqual(['id', 'name']);
    });
  });

  describe('Singleton Instance Edge Cases', () => {
    it('should provide singleton instance', () => {
      expect(daaMcpTools).toBeInstanceOf(DAA_MCPTools);
      expect(daaMcpTools.mcpTools).toBe(null); // Initialized with null
    });

    it('should handle singleton operations without enhanced MCP tools', async() => {
      // This should work even without mcpTools set
      const result = await daaMcpTools.daa_init({});
      expect(result.initialized).toBe(true);
    });
  });

  describe('End-to-End DAA Workflow Tests', () => {
    it('should complete full DAA agent lifecycle', async() => {
      // Step 1: Initialize DAA service
      const initResult = await daaTools.daa_init({
        enableLearning: true,
        enableCoordination: true,
      });
      expect(initResult.initialized).toBe(true);

      // Step 2: Create multiple agents
      const agentIds = [];
      for (let i = 0; i < 3; i++) {
        const createResult = await daaTools.daa_agent_create({
          id: `e2e-agent-${i}`,
          capabilities: [`capability-${i}`, 'learning'],
          cognitivePattern: ['convergent', 'divergent', 'adaptive'][i],
          learningRate: 0.001 + (i * 0.001),
          enableMemory: true,
        });

        agentIds.push(createResult.agent_id);
        expect(createResult.status).toBe('active');
      }

      // Step 3: Create and execute workflow
      const workflowResult = await daaTools.daa_workflow_create({
        id: 'e2e-workflow',
        name: 'End-to-End Test Workflow',
        steps: [
          { id: 'analyze', action: 'data_analysis' },
          { id: 'learn', action: 'pattern_learning' },
          { id: 'adapt', action: 'behavior_adaptation' },
        ],
        strategy: 'parallel',
      });
      expect(workflowResult.workflow_id).toBe('e2e-workflow');

      const executeResult = await daaTools.daa_workflow_execute({
        workflowId: 'e2e-workflow',
        agentIds,
        parallelExecution: true,
      });
      expect(executeResult.execution_complete).toBe(true);

      // Step 4: Knowledge sharing between agents
      const shareResult = await daaTools.daa_knowledge_share({
        sourceAgentId: agentIds[0],
        targetAgentIds: agentIds.slice(1),
        knowledgeDomain: 'test-domain',
        knowledgeContent: { insights: ['pattern-1', 'pattern-2'] },
      });
      expect(shareResult.sharing_complete).toBe(true);

      // Step 5: Adapt agents based on performance
      for (const agentId of agentIds) {
        const adaptResult = await daaTools.daa_agent_adapt({
          agentId,
          feedback: 'Good performance, continue learning',
          performanceScore: 0.8 + (Math.random() * 0.2),
          suggestions: ['Increase learning rate', 'Focus on weak areas'],
        });
        expect(adaptResult.adaptation_complete).toBe(true);
      }

      // Step 6: Check learning status
      const statusResult = await daaTools.daa_learning_status({
        detailed: true,
      });
      expect(statusResult.agent_id).toBe('all');
      expect(statusResult).toHaveProperty('detailed_metrics');

      // Step 7: Get performance metrics
      const metricsResult = await daaTools.daa_performance_metrics({
        category: 'all',
        timeRange: '1h',
      });
      expect(metricsResult.system_metrics.total_agents).toBeGreaterThan(0);
    });

    it('should handle concurrent DAA operations', async() => {
      // Initialize once
      await daaTools.daa_init({});

      // Create multiple agents concurrently
      const agentPromises = Array.from({ length: 5 }, (_, i) =>
        daaTools.daa_agent_create({
          id: `concurrent-agent-${i}`,
          capabilities: [`concurrent-${i}`],
          cognitivePattern: 'adaptive',
        }),
      );

      const agents = await Promise.all(agentPromises);
      expect(agents).toHaveLength(5);

      // Perform concurrent adaptations
      const adaptPromises = agents.map(agent =>
        daaTools.daa_agent_adapt({
          agentId: agent.agent_id,
          feedback: 'Concurrent adaptation test',
          performanceScore: Math.random(),
        }),
      );

      const adaptResults = await Promise.all(adaptPromises);
      expect(adaptResults).toHaveLength(5);
      adaptResults.forEach(result => {
        expect(result.adaptation_complete).toBe(true);
      });
    });

    it('should handle error recovery in complex DAA workflows', async() => {
      await daaTools.daa_init({});

      // Create agents with some expected to fail
      const agentPromises = Array.from({ length: 5 }, async(_, i) => {
        try {
          if (i === 2) {
            // Mock a failure
            mockDaaService.createAgent.mockRejectedValueOnce(new Error('Agent creation failed'));
          }

          return await daaTools.daa_agent_create({
            id: `recovery-agent-${i}`,
            capabilities: [`recovery-${i}`],
          });
        } catch (error) {
          return { error: error.message, failed: true };
        }
      });

      const results = await Promise.allSettled(agentPromises);

      const successful = results.filter(r => r.status === 'fulfilled' && !r.value.failed);
      const failed = results.filter(r => r.status === 'rejected' || r.value?.failed);

      expect(successful.length).toBeGreaterThan(0);
      expect(failed.length).toBeGreaterThan(0);
    });
  });
});
