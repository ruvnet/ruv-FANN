const { expect } = require('chai');
const sinon = require('sinon');
const RuvSwarm = require('../../../../src/core/ruv-swarm');
const NeuralAgentManager = require('../../../../src/neural-agent');
const MemoryManager = require('../../../../src/memory-manager');
const MCPIntegration = require('../../../../src/mcp-integration');
const HookSystem = require('../../../../src/hooks/hook-system');

describe('Cross-Feature Integration Tests', () => {
  let sandbox;
  let swarm;
  let neuralManager;
  let memoryManager;
  let mcpIntegration;
  let hookSystem;

  beforeEach(() => {
    sandbox = sinon.createSandbox();
  });

  afterEach(async() => {
    if (swarm) {
      await swarm.shutdown();
    }
    sandbox.restore();
  });

  describe('Neural + Swarm Coordination', () => {
    it('should coordinate neural agents with standard swarm operations', async() => {
      // Initialize integrated system
      swarm = new RuvSwarm();
      neuralManager = new NeuralAgentManager();

      await swarm.init({
        topology: 'mesh',
        features: ['neural', 'coordination', 'learning'],
      });

      await neuralManager.initialize({
        models: ['adaptive', 'pattern-recognition', 'optimization'],
        integration: { swarm },
      });

      // Create mixed agent ecosystem
      const standardAgents = await Promise.all([
        swarm.spawnAgent({ type: 'researcher' }),
        swarm.spawnAgent({ type: 'coder' }),
        swarm.spawnAgent({ type: 'analyst' }),
      ]);

      const neuralAgents = await Promise.all([
        swarm.spawnAgent({
          type: 'neural',
          model: 'adaptive',
          coordination: { shareWith: 'all' },
        }),
        swarm.spawnAgent({
          type: 'neural',
          model: 'pattern-recognition',
          coordination: { shareWith: ['researcher', 'analyst'] },
        }),
        swarm.spawnAgent({
          type: 'neural',
          model: 'optimization',
          coordination: { shareWith: ['coder'] },
        }),
      ]);

      // Execute coordinated task
      const task = await swarm.orchestrateTask({
        task: 'Analyze codebase and suggest neural-guided optimizations',
        requiresBoth: ['traditional-analysis', 'neural-patterns'],
        coordination: 'mixed-team',
        agents: [...standardAgents.map(a => a.id), ...neuralAgents.map(a => a.id)],
      });

      // Monitor coordination
      const coordinationEvents = [];
      swarm.on('coordination', (event) => coordinationEvents.push(event));
      neuralManager.on('insight', (insight) => coordinationEvents.push({ type: 'neural-insight', ...insight }));

      // Wait for task completion
      let taskStatus;
      for (let i = 0; i < 30; i++) {
        taskStatus = await swarm.getTaskStatus(task.id);
        if (taskStatus.status === 'completed') {
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      // Verify neural-swarm coordination
      expect(taskStatus.status).to.equal('completed');
      expect(taskStatus.neuralContributions).to.exist;
      expect(taskStatus.neuralContributions).to.have.lengthOf(3);

      // Verify knowledge sharing
      expect(coordinationEvents.filter(e => e.type === 'knowledge-share')).to.have.length.at.least(3);
      expect(coordinationEvents.filter(e => e.type === 'neural-insight')).to.have.length.at.least(2);

      // Check neural learning from swarm interactions
      const neuralMetrics = await neuralManager.getSwarmLearningMetrics();
      expect(neuralMetrics.collaborations).to.be.greaterThan(0);
      expect(neuralMetrics.patternsSaved).to.be.greaterThan(0);
      expect(neuralMetrics.accuracyImprovement).to.be.greaterThan(0);
    });

    it('should enable neural agents to learn from swarm patterns', async() => {
      swarm = new RuvSwarm();
      neuralManager = new NeuralAgentManager();

      await swarm.init({
        topology: 'hierarchical',
        learningMode: true,
        patternTracking: true,
      });

      await neuralManager.initialize({
        learningEnabled: true,
        patternRecognition: true,
      });

      // Create learning environment
      const coordinator = await swarm.spawnAgent({ type: 'coordinator' });
      const learningAgent = await swarm.spawnAgent({
        type: 'neural',
        model: 'learning',
        observeSwarmPatterns: true,
      });

      // Execute pattern-rich tasks
      const patternTasks = [
        'Implement authentication system',
        'Create user management API',
        'Build notification service',
        'Develop caching layer',
        'Add monitoring system',
      ];

      const taskResults = [];
      for (const taskDesc of patternTasks) {
        const task = await swarm.orchestrateTask({
          task: taskDesc,
          coordinatorId: coordinator.id,
          observers: [learningAgent.id],
          capturePatterns: true,
        });
        taskResults.push(task);

        // Wait for completion
        await new Promise(resolve => setTimeout(resolve, 2000));
      }

      // Analyze learned patterns
      const learnedPatterns = await neuralManager.getLearnedPatterns(learningAgent.id);
      expect(learnedPatterns).to.exist;
      expect(learnedPatterns.architecturalPatterns).to.have.length.at.least(3);
      expect(learnedPatterns.communicationPatterns).to.have.length.at.least(2);
      expect(learnedPatterns.decisionPatterns).to.have.length.at.least(2);

      // Test pattern application
      const newTask = await swarm.orchestrateTask({
        task: 'Design real-time chat system',
        useLearnedPatterns: true,
        neuralAgent: learningAgent.id,
      });

      const newTaskStatus = await swarm.getTaskStatus(newTask.id);
      expect(newTaskStatus.appliedPatterns).to.have.length.at.least(2);
      expect(newTaskStatus.confidence).to.be.greaterThan(0.8);
    });

    it('should optimize swarm topology using neural insights', async() => {
      swarm = new RuvSwarm();
      neuralManager = new NeuralAgentManager();

      await swarm.init({
        topology: 'mesh',
        adaptiveTopology: true,
        neuralOptimization: true,
      });

      await neuralManager.initialize({
        topologyOptimization: true,
        performanceAnalysis: true,
      });

      // Create neural topology optimizer
      const topologyOptimizer = await swarm.spawnAgent({
        type: 'neural',
        model: 'optimization',
        specialization: 'topology',
      });

      // Initial topology assessment
      const initialTopology = await swarm.getTopologyMetrics();
      const initialEfficiency = initialTopology.efficiency;

      // Run optimization cycles
      for (let cycle = 0; cycle < 3; cycle++) {
        // Execute representative workload
        const workloadTasks = [];
        for (let i = 0; i < 20; i++) {
          workloadTasks.push(swarm.orchestrateTask({
            task: `Workload task ${cycle}-${i}`,
            complexity: Math.random() > 0.5 ? 'high' : 'low',
          }));
        }

        await Promise.all(workloadTasks);

        // Analyze performance
        const performance = await swarm.getPerformanceMetrics();

        // Neural optimization
        const optimization = await neuralManager.optimizeTopology(topologyOptimizer.id, {
          currentTopology: await swarm.getTopology(),
          performanceData: performance,
          workloadPattern: workloadTasks.map(t => t.complexity),
        });

        if (optimization.recommendations.length > 0) {
          await swarm.applyTopologyOptimization(optimization.recommendations);
        }

        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      // Verify optimization effectiveness
      const finalTopology = await swarm.getTopologyMetrics();
      const finalEfficiency = finalTopology.efficiency;

      expect(finalEfficiency).to.be.greaterThan(initialEfficiency);

      const optimizationHistory = await neuralManager.getOptimizationHistory(topologyOptimizer.id);
      expect(optimizationHistory.improvements).to.have.length.at.least(1);
      expect(optimizationHistory.averageImprovement).to.be.greaterThan(0.05); // 5% improvement
    });
  });

  describe('Memory + Persistence Integration', () => {
    it('should persist agent memories across sessions', async() => {
      // First session
      swarm = new RuvSwarm();
      memoryManager = new MemoryManager();

      await swarm.init({
        topology: 'mesh',
        memory: { persistent: true, size: 10000 },
      });

      await memoryManager.initialize({
        persistence: true,
        storage: { type: 'sqlite', path: ':memory:' },
      });

      // Create agents with memories
      const agent1 = await swarm.spawnAgent({
        type: 'researcher',
        memory: { enabled: true, type: 'episodic' },
      });

      const agent2 = await swarm.spawnAgent({
        type: 'coder',
        memory: { enabled: true, type: 'semantic' },
      });

      // Execute tasks that create memories
      await swarm.orchestrateTask({
        task: 'Research React performance optimization',
        agents: [agent1.id],
        storeMemories: true,
      });

      await swarm.orchestrateTask({
        task: 'Implement lazy loading components',
        agents: [agent2.id],
        storeMemories: true,
      });

      // Store cross-agent shared memory
      await memoryManager.storeSharedMemory('project-context', {
        project: 'React Performance',
        insights: ['Virtual DOM optimization', 'Bundle splitting'],
        patterns: ['Component memoization', 'Code splitting'],
      });

      // Export session state
      const sessionState = await swarm.exportFullState();
      await swarm.shutdown();

      // Second session - restore from persistence
      const newSwarm = new RuvSwarm();
      const newMemoryManager = new MemoryManager();

      await newSwarm.init({
        topology: 'mesh',
        memory: { persistent: true, size: 10000 },
      });

      await newMemoryManager.initialize({
        persistence: true,
        storage: { type: 'sqlite', path: ':memory:' },
      });

      await newSwarm.importFullState(sessionState);

      // Verify memory restoration
      const restoredAgents = await newSwarm.getAgents();
      expect(restoredAgents).to.have.lengthOf(2);

      const restoredAgent1 = restoredAgents.find(a => a.type === 'researcher');
      const restoredAgent2 = restoredAgents.find(a => a.type === 'coder');

      const memory1 = await memoryManager.getAgentMemory(restoredAgent1.id);
      const memory2 = await memoryManager.getAgentMemory(restoredAgent2.id);

      expect(memory1.episodes).to.have.length.at.least(1);
      expect(memory1.episodes[0].task).to.include('React performance');

      expect(memory2.semanticMemory).to.have.property('concepts');
      expect(memory2.semanticMemory.concepts).to.include('lazy loading');

      // Verify shared memory
      const sharedMemory = await memoryManager.getSharedMemory('project-context');
      expect(sharedMemory.project).to.equal('React Performance');
      expect(sharedMemory.insights).to.have.lengthOf(2);

      await newSwarm.shutdown();
    });

    it('should enable memory-based agent coordination', async() => {
      swarm = new RuvSwarm();
      memoryManager = new MemoryManager();

      await swarm.init({
        topology: 'mesh',
        coordinationMemory: true,
      });

      await memoryManager.initialize({
        coordinationTracking: true,
        conflictResolution: true,
      });

      // Create agents with coordination memory
      const agents = await Promise.all([
        swarm.spawnAgent({
          type: 'researcher',
          coordinationMemory: true,
        }),
        swarm.spawnAgent({
          type: 'coder',
          coordinationMemory: true,
        }),
        swarm.spawnAgent({
          type: 'analyst',
          coordinationMemory: true,
        }),
      ]);

      // Execute collaborative tasks
      const collaborativeTasks = [
        {
          task: 'Research API design patterns',
          agents: [agents[0].id],
          shareWith: [agents[1].id, agents[2].id],
        },
        {
          task: 'Implement RESTful endpoints',
          agents: [agents[1].id],
          useInsights: [agents[0].id],
        },
        {
          task: 'Analyze performance metrics',
          agents: [agents[2].id],
          correlateWith: [agents[0].id, agents[1].id],
        },
      ];

      const taskResults = [];
      for (const taskConfig of collaborativeTasks) {
        const task = await swarm.orchestrateTask(taskConfig);
        taskResults.push(task);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      // Verify coordination memory
      const coordinationMemory = await memoryManager.getCoordinationMemory();
      expect(coordinationMemory.interactions).to.have.length.at.least(3);
      expect(coordinationMemory.sharedKnowledge).to.have.property('api-patterns');
      expect(coordinationMemory.sharedKnowledge).to.have.property('implementation-details');

      // Test memory-based future coordination
      const futureTask = await swarm.orchestrateTask({
        task: 'Design microservices architecture',
        useCoordinationMemory: true,
        strategy: 'memory-informed',
      });

      const futureTaskStatus = await swarm.getTaskStatus(futureTask.id);
      expect(futureTaskStatus.memoryUtilization).to.exist;
      expect(futureTaskStatus.memoryUtilization.pastInteractions).to.be.greaterThan(0);
      expect(futureTaskStatus.memoryUtilization.appliedKnowledge).to.have.length.at.least(2);
    });
  });

  describe('MCP + Agent Lifecycle Integration', () => {
    it('should manage agent lifecycle through MCP protocol', async() => {
      swarm = new RuvSwarm();
      mcpIntegration = new MCPIntegration();

      await swarm.init({
        topology: 'mesh',
        mcp: { enabled: true, protocol: 'jsonrpc' },
      });

      await mcpIntegration.initialize({
        swarmIntegration: true,
        agentManagement: true,
      });

      // MCP-controlled agent spawning
      const mcpSpawn = await mcpIntegration.call('agent_spawn', {
        type: 'researcher',
        capabilities: ['web-search', 'analysis'],
        lifecycle: 'mcp-managed',
      });

      expect(mcpSpawn.success).to.be.true;
      expect(mcpSpawn.agentId).to.exist;

      // Verify agent exists in swarm
      const agent = await swarm.getAgent(mcpSpawn.agentId);
      expect(agent).to.exist;
      expect(agent.managedBy).to.equal('mcp');

      // MCP task orchestration
      const mcpTask = await mcpIntegration.call('task_orchestrate', {
        task: 'Research latest JavaScript frameworks',
        agentId: mcpSpawn.agentId,
        priority: 'high',
      });

      expect(mcpTask.success).to.be.true;
      expect(mcpTask.taskId).to.exist;

      // Monitor through MCP
      const taskStatus = await mcpIntegration.call('task_status', {
        taskId: mcpTask.taskId,
      });

      expect(taskStatus.id).to.equal(mcpTask.taskId);
      expect(taskStatus.status).to.be.oneOf(['pending', 'running', 'completed']);

      // MCP metrics collection
      const mcpMetrics = await mcpIntegration.call('agent_metrics', {
        agentId: mcpSpawn.agentId,
      });

      expect(mcpMetrics.agentId).to.equal(mcpSpawn.agentId);
      expect(mcpMetrics.performance).to.exist;
      expect(mcpMetrics.resourceUsage).to.exist;

      // MCP-managed cleanup
      const cleanup = await mcpIntegration.call('agent_destroy', {
        agentId: mcpSpawn.agentId,
      });

      expect(cleanup.success).to.be.true;

      // Verify agent removed from swarm
      const removedAgent = await swarm.getAgent(mcpSpawn.agentId);
      expect(removedAgent).to.be.null;
    });

    it('should synchronize MCP state with swarm state', async() => {
      swarm = new RuvSwarm();
      mcpIntegration = new MCPIntegration();

      await swarm.init({
        topology: 'mesh',
        mcp: {
          enabled: true,
          stateSynchronization: true,
          syncInterval: 1000,
        },
      });

      await mcpIntegration.initialize({
        stateSynchronization: true,
        conflictResolution: 'swarm-priority',
      });

      // Create agents through both interfaces
      const swarmAgent = await swarm.spawnAgent({ type: 'coder' });
      const mcpAgent = await mcpIntegration.call('agent_spawn', { type: 'analyst' });

      // Execute tasks through both interfaces
      const swarmTask = await swarm.orchestrateTask({
        task: 'Swarm-managed task',
        agentId: swarmAgent.id,
      });

      const mcpTask = await mcpIntegration.call('task_orchestrate', {
        task: 'MCP-managed task',
        agentId: mcpAgent.agentId,
      });

      // Wait for synchronization
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Verify state synchronization
      const swarmState = await swarm.getFullState();
      const mcpState = await mcpIntegration.call('swarm_status');

      expect(swarmState.agents).to.have.lengthOf(2);
      expect(mcpState.agents).to.have.lengthOf(2);

      // Agent IDs should match
      const swarmAgentIds = swarmState.agents.map(a => a.id).sort();
      const mcpAgentIds = mcpState.agents.map(a => a.id).sort();
      expect(swarmAgentIds).to.deep.equal(mcpAgentIds);

      // Tasks should be visible in both
      expect(swarmState.tasks).to.have.lengthOf(2);
      expect(mcpState.tasks).to.have.lengthOf(2);

      // Introduce state conflict
      await swarm.updateAgent(swarmAgent.id, { status: 'busy' });

      // Wait for conflict resolution
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Verify conflict resolution
      const resolvedSwarmState = await swarm.getAgent(swarmAgent.id);
      const resolvedMcpAgent = await mcpIntegration.call('agent_status', { agentId: swarmAgent.id });

      expect(resolvedSwarmState.status).to.equal(resolvedMcpAgent.status);
    });
  });

  describe('Hooks + Event System Integration', () => {
    it('should trigger hooks throughout agent lifecycle', async() => {
      swarm = new RuvSwarm();
      hookSystem = new HookSystem();

      await swarm.init({
        topology: 'mesh',
        hooks: { enabled: true },
      });

      await hookSystem.initialize({
        swarmIntegration: true,
        eventTracking: true,
      });

      // Register lifecycle hooks
      const hookEvents = [];

      hookSystem.register('pre-agent-spawn', async(context) => {
        hookEvents.push({ type: 'pre-spawn', agentType: context.type });
        return { enhanced: true };
      });

      hookSystem.register('post-agent-spawn', async(context) => {
        hookEvents.push({ type: 'post-spawn', agentId: context.agentId });
      });

      hookSystem.register('pre-task-orchestrate', async(context) => {
        hookEvents.push({ type: 'pre-task', task: context.task });
        // Enhance task with hook data
        context.enhanced = { timestamp: Date.now() };
      });

      hookSystem.register('post-task-complete', async(context) => {
        hookEvents.push({ type: 'post-task', taskId: context.taskId, success: context.success });
      });

      // Execute operations that trigger hooks
      const agent = await swarm.spawnAgent({ type: 'researcher' });

      const task = await swarm.orchestrateTask({
        task: 'Hook-enhanced research task',
        agentId: agent.id,
      });

      // Wait for task completion
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Verify hooks were triggered
      expect(hookEvents.filter(e => e.type === 'pre-spawn')).to.have.lengthOf(1);
      expect(hookEvents.filter(e => e.type === 'post-spawn')).to.have.lengthOf(1);
      expect(hookEvents.filter(e => e.type === 'pre-task')).to.have.lengthOf(1);
      expect(hookEvents.filter(e => e.type === 'post-task')).to.have.lengthOf(1);

      // Verify hook data was applied
      const taskStatus = await swarm.getTaskStatus(task.id);
      expect(taskStatus.enhanced).to.exist;
      expect(taskStatus.enhanced.timestamp).to.be.a('number');
    });

    it('should cascade hooks across system components', async() => {
      swarm = new RuvSwarm();
      hookSystem = new HookSystem();
      neuralManager = new NeuralAgentManager();
      memoryManager = new MemoryManager();

      // Initialize all components with hooks
      await swarm.init({
        topology: 'mesh',
        hooks: { enabled: true, cascade: true },
      });

      await hookSystem.initialize({ cascading: true });
      await neuralManager.initialize({ hooks: true });
      await memoryManager.initialize({ hooks: true });

      // Register cascading hooks
      const cascadeEvents = [];

      // Swarm hooks
      hookSystem.register('swarm:agent-spawn', (ctx) => {
        cascadeEvents.push({ component: 'swarm', event: 'spawn', data: ctx });
        return hookSystem.cascade('neural:agent-available', ctx);
      });

      // Neural hooks
      hookSystem.register('neural:agent-available', (ctx) => {
        cascadeEvents.push({ component: 'neural', event: 'available', data: ctx });
        return hookSystem.cascade('memory:agent-initialize', ctx);
      });

      // Memory hooks
      hookSystem.register('memory:agent-initialize', (ctx) => {
        cascadeEvents.push({ component: 'memory', event: 'initialize', data: ctx });
      });

      // Task completion cascade
      hookSystem.register('swarm:task-complete', (ctx) => {
        cascadeEvents.push({ component: 'swarm', event: 'complete', data: ctx });
        hookSystem.cascade('neural:learn-from-task', ctx);
        hookSystem.cascade('memory:store-experience', ctx);
      });

      hookSystem.register('neural:learn-from-task', (ctx) => {
        cascadeEvents.push({ component: 'neural', event: 'learn', data: ctx });
      });

      hookSystem.register('memory:store-experience', (ctx) => {
        cascadeEvents.push({ component: 'memory', event: 'store', data: ctx });
      });

      // Trigger cascade
      const agent = await swarm.spawnAgent({ type: 'neural' });

      const task = await swarm.orchestrateTask({
        task: 'Cascade test task',
        agentId: agent.id,
      });

      // Wait for cascades
      await new Promise(resolve => setTimeout(resolve, 3000));

      // Verify cascade sequence
      const spawnCascade = cascadeEvents.filter(e => e.event === 'spawn' || e.event === 'available' || e.event === 'initialize');
      expect(spawnCascade).to.have.lengthOf(3);
      expect(spawnCascade[0].component).to.equal('swarm');
      expect(spawnCascade[1].component).to.equal('neural');
      expect(spawnCascade[2].component).to.equal('memory');

      const completeCascade = cascadeEvents.filter(e => e.event === 'complete' || e.event === 'learn' || e.event === 'store');
      expect(completeCascade).to.have.lengthOf(3);
      expect(completeCascade[0].component).to.equal('swarm');
      expect(completeCascade[1].component).to.equal('neural');
      expect(completeCascade[2].component).to.equal('memory');
    });

    it('should handle hook failures gracefully', async() => {
      swarm = new RuvSwarm();
      hookSystem = new HookSystem();

      await swarm.init({
        topology: 'mesh',
        hooks: {
          enabled: true,
          errorHandling: 'graceful',
          continueOnFailure: true,
        },
      });

      await hookSystem.initialize({
        errorRecovery: true,
        retryFailedHooks: true,
      });

      // Register failing hooks
      const hookResults = [];

      hookSystem.register('failing-hook', async(ctx) => {
        hookResults.push({ type: 'attempt', timestamp: Date.now() });
        throw new Error('Hook intentionally failed');
      });

      hookSystem.register('success-hook', async(ctx) => {
        hookResults.push({ type: 'success', timestamp: Date.now() });
      });

      // Configure hook chain
      hookSystem.chain('agent-spawn', ['failing-hook', 'success-hook']);

      // Trigger hooks
      const agent = await swarm.spawnAgent({ type: 'coder' });

      // Wait for retries
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Verify agent still created despite hook failure
      expect(agent).to.exist;
      expect(agent.id).to.exist;

      // Verify hook failure handling
      const attempts = hookResults.filter(r => r.type === 'attempt');
      const successes = hookResults.filter(r => r.type === 'success');

      expect(attempts).to.have.length.at.least(1); // Failed hook attempted
      expect(successes).to.have.lengthOf(1); // Success hook executed

      // Check error logging
      const hookErrors = await hookSystem.getErrorLog();
      expect(hookErrors).to.have.length.at.least(1);
      expect(hookErrors[0].error).to.include('Hook intentionally failed');
    });
  });

  describe('Full System Integration', () => {
    it('should demonstrate complete feature integration', async function() {
      this.timeout(45000);

      // Initialize complete system
      swarm = new RuvSwarm();
      neuralManager = new NeuralAgentManager();
      memoryManager = new MemoryManager();
      mcpIntegration = new MCPIntegration();
      hookSystem = new HookSystem();

      await swarm.init({
        topology: 'mesh',
        features: ['neural', 'memory', 'mcp', 'hooks', 'persistence'],
      });

      await neuralManager.initialize({ swarmIntegration: true });
      await memoryManager.initialize({ persistence: true });
      await mcpIntegration.initialize({ fullIntegration: true });
      await hookSystem.initialize({ allFeatures: true });

      // Create comprehensive agent ecosystem
      const ecosystem = {
        coordinator: await swarm.spawnAgent({
          type: 'coordinator',
          features: ['memory', 'mcp-managed'],
        }),
        neuralResearcher: await swarm.spawnAgent({
          type: 'neural',
          model: 'research-optimized',
          features: ['memory', 'hooks'],
        }),
        standardCoder: await swarm.spawnAgent({
          type: 'coder',
          features: ['memory', 'persistence'],
        }),
        analyst: await swarm.spawnAgent({
          type: 'analyst',
          features: ['neural-assisted', 'mcp-reporting'],
        }),
      };

      // Execute complex multi-feature workflow
      const workflow = await swarm.orchestrateTask({
        task: 'Design, implement, and optimize a microservices platform',
        coordinator: ecosystem.coordinator.id,
        strategy: 'feature-integrated',
        phases: [
          {
            phase: 'research',
            agent: ecosystem.neuralResearcher.id,
            features: ['neural-insights', 'memory-enhanced'],
          },
          {
            phase: 'implementation',
            agent: ecosystem.standardCoder.id,
            features: ['hook-driven', 'persistent-state'],
          },
          {
            phase: 'analysis',
            agent: ecosystem.analyst.id,
            features: ['neural-analysis', 'mcp-reporting'],
          },
        ],
      });

      // Monitor integration
      const integrationMetrics = {
        neuralInsights: 0,
        memoryStores: 0,
        mcpCalls: 0,
        hooksTriggered: 0,
      };

      neuralManager.on('insight', () => integrationMetrics.neuralInsights++);
      memoryManager.on('store', () => integrationMetrics.memoryStores++);
      mcpIntegration.on('call', () => integrationMetrics.mcpCalls++);
      hookSystem.on('trigger', () => integrationMetrics.hooksTriggered++);

      // Wait for workflow completion
      let workflowStatus;
      for (let i = 0; i < 30; i++) {
        workflowStatus = await swarm.getTaskStatus(workflow.id);
        if (workflowStatus.status === 'completed') {
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      // Verify comprehensive integration
      expect(workflowStatus.status).to.equal('completed');
      expect(workflowStatus.phases).to.have.lengthOf(3);
      expect(workflowStatus.phases.every(p => p.status === 'completed')).to.be.true;

      // Verify feature utilization
      expect(integrationMetrics.neuralInsights).to.be.greaterThan(0);
      expect(integrationMetrics.memoryStores).to.be.greaterThan(0);
      expect(integrationMetrics.mcpCalls).to.be.greaterThan(0);
      expect(integrationMetrics.hooksTriggered).to.be.greaterThan(0);

      // Verify cross-feature data flow
      const finalState = await swarm.getFullState();
      expect(finalState.neuralLearnings).to.exist;
      expect(finalState.memorySnapshot).to.exist;
      expect(finalState.mcpSync).to.exist;
      expect(finalState.hookHistory).to.exist;

      // Test system resilience
      const resilienceTest = await swarm.executeResilienceTest({
        duration: 10000,
        scenarios: ['component-failure', 'memory-pressure', 'network-partition'],
      });

      expect(resilienceTest.overallSuccess).to.be.true;
      expect(resilienceTest.recoveryTime).to.be.lessThan(5000);
      expect(resilienceTest.dataIntegrity).to.be.true;
    });
  });
});