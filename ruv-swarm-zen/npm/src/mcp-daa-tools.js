/**
 * DAA (Decentralized Autonomous Agents) MCP Tools
 * Exposes DAA capabilities through the MCP interface
 */

import { daaService } from './daa-service.js';

export class DAA_MCPTools {
  constructor(enhancedMcpTools) {
    this.mcpTools = enhancedMcpTools;
    this.daaInitialized = false;
  }

  async ensureInitialized() {
    if (!this.daaInitialized) {
      await daaService.initialize();
      this.daaInitialized = true;
    }
  }

  /**
   * DAA MCP Tool: daa_init
   * Initialize the DAA service with autonomous agent capabilities
   */
  async daa_init(params) {
    const startTime = performance.now();
    try {
      await this.ensureInitialized();

      const {
        enableLearning = true,
        enableCoordination = true,
        persistenceMode = 'auto',
      } = params;

      const result = {
        success: true,
        initialized: true,
        features: {
          autonomousLearning: enableLearning,
          peerCoordination: enableCoordination,
          persistenceMode,
          neuralIntegration: true,
          cognitivePatterns: 6,
        },
        capabilities: daaService.getCapabilities(),
        timestamp: new Date().toISOString(),
      };

      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_init', startTime, 'success');
      }
      return result;
    } catch (error) {
      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_init', startTime, 'error', error.message);
      }
      throw error;
    }
  }

  /**
   * DAA MCP Tool: daa_agent_create
   * Create an autonomous agent with DAA capabilities
   */
  async daa_agent_create(params) {
    const startTime = performance.now();
    try {
      await this.ensureInitialized();

      const {
        id,
        capabilities = [],
        cognitivePattern = 'adaptive',
        learningRate = 0.001,
        enableMemory = true,
      } = params;

      if (!id) {
        throw new Error('Agent ID is required');
      }

      const agent = await daaService.createAgent({
        id,
        capabilities,
        cognitivePattern,
        config: {
          learningRate,
          enableMemory,
          autonomousMode: true,
        },
      });

      // Find or create a swarm for the agent
      let swarmId = null;
      if (this.mcpTools?.activeSwarms) {
        for (const [id, swarm] of this.mcpTools.activeSwarms) {
          if (swarm.agents.size < swarm.maxAgents) {
            swarmId = id;
            swarm.agents.set(agent.id, agent);
            break;
          }
        }
      } else {
        // Create a virtual swarm if none exists
        swarmId = 'daa-default-swarm';
      }

      // Persist DAA agent to database
      if (this.mcpTools?.persistence) {
        try {
          await this.mcpTools.persistence.createAgent({
            id: agent.id,
            swarmId: swarmId || 'daa-default-swarm',
            name: `DAA-${agent.id}`,
            type: 'daa',
            capabilities: Array.from(agent.capabilities || capabilities),
            neuralConfig: {
              cognitivePattern: agent.cognitivePattern || cognitivePattern,
              learningRate,
              enableMemory,
              daaEnabled: true,
            },
            metadata: {
              createdAt: new Date().toISOString(),
              autonomousMode: true,
            },
          });

          this.mcpTools.logger?.info('DAA agent persisted successfully', {
            agentId: agent.id,
            swarmId,
          });
        } catch (persistError) {
          this.mcpTools.logger?.warn('Failed to persist DAA agent', {
            agentId: agent.id,
            error: persistError.message,
          });
          // Continue execution even if persistence fails
        }
      }

      const result = {
        agent: {
          id: agent.id,
          cognitive_pattern: agent.cognitivePattern || cognitivePattern,
          capabilities: Array.from(agent.capabilities || capabilities),
          status: 'active',
          created_at: new Date().toISOString(),
        },
        swarm_id: swarmId,
        learning_enabled: learningRate > 0,
        memory_enabled: enableMemory,
      };

      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_agent_create', startTime, 'success');
      }
      return result;
    } catch (error) {
      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_agent_create', startTime, 'error', error.message);
      }
      throw error;
    }
  }

  /**
   * DAA MCP Tool: daa_agent_adapt
   * Trigger agent adaptation based on feedback
   */
  async daa_agent_adapt(params) {
    const startTime = performance.now();
    try {
      await this.ensureInitialized();

      const {
        agent_id,
        agentId,
        feedback,
        performanceScore = 0.5,
        suggestions = [],
      } = params;

      const id = agent_id || agentId;
      if (!id) {
        throw new Error('Agent ID is required. Provide either agent_id or agentId parameter.');
      }

      const adaptationResult = await daaService.adaptAgent(id, {
        feedback,
        performanceScore,
        suggestions,
        timestamp: new Date().toISOString(),
      });

      const result = {
        agent_id: id,
        adaptation_complete: true,
        previous_pattern: adaptationResult.previousPattern,
        new_pattern: adaptationResult.newPattern,
        performance_improvement: adaptationResult.improvement,
        learning_insights: adaptationResult.insights,
        timestamp: new Date().toISOString(),
      };

      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_agent_adapt', startTime, 'success');
      }
      return result;
    } catch (error) {
      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_agent_adapt', startTime, 'error', error.message);
      }
      throw error;
    }
  }

  /**
   * DAA MCP Tool: daa_workflow_create
   * Create an autonomous workflow with DAA coordination
   */
  async daa_workflow_create(params) {
    const startTime = performance.now();
    try {
      await this.ensureInitialized();

      const {
        id,
        name,
        steps = [],
        dependencies = {},
        strategy = 'parallel',
      } = params;

      if (!id || !name) {
        throw new Error('Workflow ID and name are required');
      }

      const workflow = await daaService.createWorkflow(id, steps, dependencies);

      const result = {
        workflow_id: workflow.id,
        name,
        total_steps: workflow.steps.length,
        execution_strategy: strategy,
        dependencies_count: Object.keys(workflow.dependencies).length,
        status: workflow.status,
        created_at: new Date().toISOString(),
      };

      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_workflow_create', startTime, 'success');
      }
      return result;
    } catch (error) {
      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_workflow_create', startTime, 'error', error.message);
      }
      throw error;
    }
  }

  /**
   * DAA MCP Tool: daa_workflow_execute
   * Execute a DAA workflow with autonomous agents
   */
  async daa_workflow_execute(params) {
    const startTime = performance.now();
    try {
      await this.ensureInitialized();

      const {
        workflow_id,
        workflowId,
        agentIds = [],
        parallelExecution = true,
      } = params;

      const id = workflow_id || workflowId;
      if (!id) {
        throw new Error('Workflow ID is required. Provide either workflow_id or workflowId parameter.');
      }

      const executionResult = await daaService.executeWorkflow(id, {
        agentIds,
        parallel: parallelExecution,
      });

      const result = {
        workflow_id: id,
        execution_complete: executionResult.complete,
        steps_completed: executionResult.stepsCompleted,
        total_steps: executionResult.totalSteps,
        execution_time_ms: executionResult.executionTime,
        agents_involved: executionResult.agentsInvolved,
        results: executionResult.stepResults,
        timestamp: new Date().toISOString(),
      };

      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_workflow_execute', startTime, 'success');
      }
      return result;
    } catch (error) {
      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_workflow_execute', startTime, 'error', error.message);
      }
      throw error;
    }
  }

  /**
   * DAA MCP Tool: daa_knowledge_share
   * Share knowledge between autonomous agents
   */
  async daa_knowledge_share(params) {
    const startTime = performance.now();
    try {
      await this.ensureInitialized();

      const {
        source_agent,
        sourceAgentId,
        target_agents,
        targetAgentIds,
        knowledgeDomain,
        knowledgeContent,
      } = params;

      const sourceId = source_agent || sourceAgentId;
      const targetIds = target_agents || targetAgentIds || [];

      if (!sourceId) {
        throw new Error('Source agent ID is required. Provide either source_agent or sourceAgentId parameter.');
      }
      if (!targetIds || targetIds.length === 0) {
        throw new Error('Target agent IDs are required. Provide either target_agents or targetAgentIds parameter with at least one agent ID.');
      }

      const sharingResults = await daaService.shareKnowledge(
        sourceId,
        targetIds,
        {
          domain: knowledgeDomain,
          content: knowledgeContent,
          timestamp: new Date().toISOString(),
        },
      );

      const result = {
        source_agent: sourceId,
        target_agents: targetIds,
        knowledge_domain: knowledgeDomain,
        sharing_complete: true,
        agents_updated: sharingResults.updatedAgents,
        knowledge_transfer_rate: sharingResults.transferRate,
        timestamp: new Date().toISOString(),
      };

      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_knowledge_share', startTime, 'success');
      }
      return result;
    } catch (error) {
      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_knowledge_share', startTime, 'error', error.message);
      }
      throw error;
    }
  }

  /**
   * DAA MCP Tool: daa_learning_status
   * Get learning progress and status for DAA agents
   */
  async daa_learning_status(params) {
    const startTime = performance.now();
    try {
      await this.ensureInitialized();

      const { agentId, detailed = false } = params;

      let learningStatus;
      if (agentId) {
        // Get specific agent learning status
        learningStatus = await daaService.getAgentLearningStatus(agentId);
      } else {
        // Get overall system learning status
        learningStatus = await daaService.getSystemLearningStatus();
      }

      const result = {
        agent_id: agentId || 'all',
        total_learning_cycles: learningStatus.totalCycles,
        average_proficiency: learningStatus.avgProficiency,
        knowledge_domains: learningStatus.domains,
        adaptation_rate: learningStatus.adaptationRate,
        neural_models_active: learningStatus.neuralModelsCount,
        cross_session_memory: learningStatus.persistentMemorySize,
        performance_trend: learningStatus.performanceTrend,
        timestamp: new Date().toISOString(),
      };

      if (detailed) {
        result.detailed_metrics = learningStatus.detailedMetrics;
      }

      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_learning_status', startTime, 'success');
      }
      return result;
    } catch (error) {
      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_learning_status', startTime, 'error', error.message);
      }
      throw error;
    }
  }

  /**
   * DAA MCP Tool: daa_cognitive_pattern
   * Analyze or change cognitive patterns for agents
   */
  async daa_cognitive_pattern(params) {
    const startTime = performance.now();
    try {
      await this.ensureInitialized();

      const {
        agent_id,
        agentId,
        pattern,
        action,
        analyze = false,
      } = params;

      const id = agent_id || agentId;
      const shouldAnalyze = action === 'analyze' || analyze;

      if (shouldAnalyze) {
        // Analyze current cognitive patterns
        const analysis = await daaService.analyzeCognitivePatterns(agentId);
        const result = {
          analysis_type: 'cognitive_pattern',
          agent_id: id || 'all',
          current_patterns: analysis.patterns,
          pattern_effectiveness: analysis.effectiveness,
          recommendations: analysis.recommendations,
          optimization_potential: analysis.optimizationScore,
          timestamp: new Date().toISOString(),
        };

        if (this.mcpTools?.recordToolMetrics) {
          this.mcpTools.recordToolMetrics('daa_cognitive_pattern', startTime, 'success');
        }
        return result;
      }
      // Change cognitive pattern
      if (!agentId || !pattern) {
        throw new Error('Agent ID and pattern are required for pattern change');
      }

      const changeResult = await daaService.setCognitivePattern(agentId, pattern);

      const result = {
        agent_id: agentId,
        previous_pattern: changeResult.previousPattern,
        new_pattern: pattern,
        adaptation_success: changeResult.success,
        expected_improvement: changeResult.expectedImprovement,
        timestamp: new Date().toISOString(),
      };

      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_cognitive_pattern', startTime, 'success');
      }
      return result;

    } catch (error) {
      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_cognitive_pattern', startTime, 'error', error.message);
      }
      throw error;
    }
  }

  /**
   * DAA MCP Tool: daa_meta_learning
   * Enable meta-learning capabilities across domains
   */
  async daa_meta_learning(params) {
    const startTime = performance.now();
    try {
      await this.ensureInitialized();

      const {
        sourceDomain,
        targetDomain,
        transferMode = 'adaptive',
        agentIds = [],
      } = params;

      const metaLearningResult = await daaService.performMetaLearning({
        sourceDomain,
        targetDomain,
        transferMode,
        agentIds: agentIds.length > 0 ? agentIds : undefined,
      });

      const result = {
        meta_learning_complete: true,
        source_domain: sourceDomain,
        target_domain: targetDomain,
        transfer_mode: transferMode,
        knowledge_transferred: metaLearningResult.knowledgeItems,
        agents_updated: metaLearningResult.updatedAgents,
        domain_proficiency_gain: metaLearningResult.proficiencyGain,
        cross_domain_insights: metaLearningResult.insights,
        timestamp: new Date().toISOString(),
      };

      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_meta_learning', startTime, 'success');
      }
      return result;
    } catch (error) {
      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_meta_learning', startTime, 'error', error.message);
      }
      throw error;
    }
  }

  /**
   * DAA MCP Tool: daa_performance_metrics
   * Get comprehensive DAA performance metrics
   */
  async daa_performance_metrics(params) {
    const startTime = performance.now();
    try {
      await this.ensureInitialized();

      const { category = 'all', timeRange = '1h' } = params;

      const metrics = await daaService.getPerformanceMetrics({
        category,
        timeRange,
      });

      const result = {
        metrics_category: category,
        time_range: timeRange,
        system_metrics: {
          total_agents: metrics.totalAgents,
          active_agents: metrics.activeAgents,
          autonomous_tasks_completed: metrics.tasksCompleted,
          average_task_time_ms: metrics.avgTaskTime,
          learning_cycles_completed: metrics.learningCycles,
        },
        performance_metrics: {
          task_success_rate: metrics.successRate,
          adaptation_effectiveness: metrics.adaptationScore,
          knowledge_sharing_events: metrics.knowledgeSharingCount,
          cross_domain_transfers: metrics.crossDomainTransfers,
        },
        efficiency_metrics: {
          token_reduction: metrics.tokenReduction,
          parallel_execution_gain: metrics.parallelGain,
          memory_optimization: metrics.memoryOptimization,
        },
        neural_metrics: {
          models_active: metrics.neuralModelsActive,
          inference_speed_ms: metrics.avgInferenceTime,
          training_iterations: metrics.totalTrainingIterations,
        },
        timestamp: new Date().toISOString(),
      };

      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_performance_metrics', startTime, 'success');
      }
      return result;
    } catch (error) {
      if (this.mcpTools?.recordToolMetrics) {
        this.mcpTools.recordToolMetrics('daa_performance_metrics', startTime, 'error', error.message);
      }
      throw error;
    }
  }

  /**
   * Get all DAA tool definitions for MCP
   */
  getToolDefinitions() {
    return [
      {
        name: 'daa_init',
        description: 'Initialize DAA (Decentralized Autonomous Agents) service',
        inputSchema: {
          type: 'object',
          properties: {
            enableLearning: { type: 'boolean', description: 'Enable autonomous learning' },
            enableCoordination: { type: 'boolean', description: 'Enable peer coordination' },
            persistenceMode: { type: 'string', enum: ['auto', 'memory', 'disk'], description: 'Persistence mode' },
          },
        },
      },
      {
        name: 'daa_agent_create',
        description: 'Create an autonomous agent with DAA capabilities',
        inputSchema: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Unique agent identifier' },
            capabilities: { type: 'array', items: { type: 'string' }, description: 'Agent capabilities' },
            cognitivePattern: { type: 'string', enum: ['convergent', 'divergent', 'lateral', 'systems', 'critical', 'adaptive'], description: 'Cognitive thinking pattern' },
            learningRate: { type: 'number', description: 'Learning rate (0-1)' },
            enableMemory: { type: 'boolean', description: 'Enable persistent memory' },
          },
          required: ['id'],
        },
      },
      {
        name: 'daa_agent_adapt',
        description: 'Trigger agent adaptation based on feedback',
        inputSchema: {
          type: 'object',
          properties: {
            agent_id: { type: 'string', description: 'Agent ID to adapt' },
            agentId: { type: 'string', description: 'Agent ID to adapt (legacy)' },
            feedback: { type: 'string', description: 'Feedback message' },
            performanceScore: { type: 'number', description: 'Performance score (0-1)' },
            suggestions: { type: 'array', items: { type: 'string' }, description: 'Improvement suggestions' },
          },
          required: ['agent_id'],
        },
      },
      {
        name: 'daa_workflow_create',
        description: 'Create an autonomous workflow with DAA coordination',
        inputSchema: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Workflow ID' },
            name: { type: 'string', description: 'Workflow name' },
            steps: { type: 'array', description: 'Workflow steps' },
            dependencies: { type: 'object', description: 'Step dependencies' },
            strategy: { type: 'string', enum: ['parallel', 'sequential', 'adaptive'], description: 'Execution strategy' },
          },
          required: ['id', 'name'],
        },
      },
      {
        name: 'daa_workflow_execute',
        description: 'Execute a DAA workflow with autonomous agents',
        inputSchema: {
          type: 'object',
          properties: {
            workflow_id: { type: 'string', description: 'Workflow ID to execute' },
            workflowId: { type: 'string', description: 'Workflow ID to execute (legacy)' },
            agentIds: { type: 'array', items: { type: 'string' }, description: 'Agent IDs to use' },
            parallelExecution: { type: 'boolean', description: 'Enable parallel execution' },
          },
          required: ['workflow_id'],
        },
      },
      {
        name: 'daa_knowledge_share',
        description: 'Share knowledge between autonomous agents',
        inputSchema: {
          type: 'object',
          properties: {
            source_agent: { type: 'string', description: 'Source agent ID' },
            sourceAgentId: { type: 'string', description: 'Source agent ID (legacy)' },
            target_agents: { type: 'array', items: { type: 'string' }, description: 'Target agent IDs' },
            targetAgentIds: { type: 'array', items: { type: 'string' }, description: 'Target agent IDs (legacy)' },
            knowledgeDomain: { type: 'string', description: 'Knowledge domain' },
            knowledgeContent: { type: 'object', description: 'Knowledge to share' },
          },
          required: ['source_agent', 'target_agents'],
        },
      },
      {
        name: 'daa_learning_status',
        description: 'Get learning progress and status for DAA agents',
        inputSchema: {
          type: 'object',
          properties: {
            agentId: { type: 'string', description: 'Specific agent ID (optional)' },
            detailed: { type: 'boolean', description: 'Include detailed metrics' },
          },
        },
      },
      {
        name: 'daa_cognitive_pattern',
        description: 'Analyze or change cognitive patterns for agents',
        inputSchema: {
          type: 'object',
          properties: {
            agent_id: { type: 'string', description: 'Agent ID' },
            agentId: { type: 'string', description: 'Agent ID (legacy)' },
            action: { type: 'string', enum: ['analyze', 'change'], description: 'Action to perform' },
            pattern: { type: 'string', enum: ['convergent', 'divergent', 'lateral', 'systems', 'critical', 'adaptive'], description: 'New pattern to set' },
            analyze: { type: 'boolean', description: 'Analyze patterns instead of changing' },
          },
        },
      },
      {
        name: 'daa_meta_learning',
        description: 'Enable meta-learning capabilities across domains',
        inputSchema: {
          type: 'object',
          properties: {
            sourceDomain: { type: 'string', description: 'Source knowledge domain' },
            targetDomain: { type: 'string', description: 'Target knowledge domain' },
            transferMode: { type: 'string', enum: ['adaptive', 'direct', 'gradual'], description: 'Transfer mode' },
            agentIds: { type: 'array', items: { type: 'string' }, description: 'Specific agents to update' },
          },
        },
      },
      {
        name: 'daa_performance_metrics',
        description: 'Get comprehensive DAA performance metrics',
        inputSchema: {
          type: 'object',
          properties: {
            category: { type: 'string', enum: ['all', 'system', 'performance', 'efficiency', 'neural'], description: 'Metrics category' },
            timeRange: { type: 'string', description: 'Time range (e.g., 1h, 24h, 7d)' },
          },
        },
      },
    ];
  }
}

// Export singleton instance
export const daaMcpTools = new DAA_MCPTools(null);
