const { expect } = require('chai');
const sinon = require('sinon');
const RuvSwarm = require('../../../../src/core/ruv-swarm');
const Agent = require('../../../../src/agent');
const AgentCommunicator = require('../../../../src/agent-communicator');
const NeuralAgentManager = require('../../../../src/neural-agent');
const { taskOrchestrationSimulator } = require('../../../../src/task-orchestrator');

describe('Complete Agent Workflow Integration', () => {
  let sandbox;
  let swarm;
  let communicator;

  beforeEach(() => {
    sandbox = sinon.createSandbox();
    communicator = new AgentCommunicator();
  });

  afterEach(async() => {
    if (swarm) {
      await swarm.shutdown();
    }
    sandbox.restore();
  });

  describe('Full Lifecycle Tests', () => {
    it('should handle complete workflow from spawn to completion', async() => {
      // Initialize swarm with mesh topology
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'mesh',
        maxAgents: 6,
        strategy: 'balanced',
      });

      // Verify initialization
      const initStatus = await swarm.getStatus();
      expect(initStatus.topology).to.equal('mesh');
      expect(initStatus.maxAgents).to.equal(6);
      expect(initStatus.isActive).to.be.true;

      // Spawn multiple agents in parallel
      const agentPromises = [
        swarm.spawnAgent({ type: 'researcher', name: 'Data Researcher' }),
        swarm.spawnAgent({ type: 'coder', name: 'API Developer' }),
        swarm.spawnAgent({ type: 'analyst', name: 'Performance Analyst' }),
        swarm.spawnAgent({ type: 'tester', name: 'QA Engineer' }),
      ];

      const agents = await Promise.all(agentPromises);

      // Verify all agents spawned successfully
      expect(agents).to.have.lengthOf(4);
      agents.forEach(agent => {
        expect(agent).to.have.property('id');
        expect(agent).to.have.property('type');
        expect(agent).to.have.property('status', 'idle');
      });

      // Orchestrate a complex task
      const taskResult = await swarm.orchestrateTask({
        task: 'Build and optimize REST API with authentication',
        priority: 'high',
        strategy: 'parallel',
        maxAgents: 4,
      });

      // Verify task orchestration
      expect(taskResult).to.have.property('id');
      expect(taskResult).to.have.property('status');
      expect(taskResult).to.have.property('assignedAgents');
      expect(taskResult.assignedAgents).to.have.length.at.least(1);

      // Wait for task completion with timeout
      const startTime = Date.now();
      const timeout = 30000; // 30 seconds
      let taskStatus;

      while (Date.now() - startTime < timeout) {
        taskStatus = await swarm.getTaskStatus(taskResult.id);
        if (taskStatus.status === 'completed' || taskStatus.status === 'failed') {
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      // Verify task completion
      expect(taskStatus.status).to.equal('completed');
      expect(taskStatus.results).to.exist;
      expect(taskStatus.performance).to.have.property('totalTime');
      expect(taskStatus.performance).to.have.property('efficiency');

      // Get agent metrics
      const agentMetrics = await swarm.getAgentMetrics();
      expect(agentMetrics).to.have.property('agents');
      expect(agentMetrics.agents).to.have.length.at.least(4);

      agentMetrics.agents.forEach(metric => {
        expect(metric).to.have.property('tasksCompleted');
        expect(metric).to.have.property('averagePerformance');
        expect(metric).to.have.property('resourceUsage');
      });

      // Verify proper cleanup
      await swarm.shutdown();
      const shutdownStatus = await swarm.getStatus();
      expect(shutdownStatus.isActive).to.be.false;
      expect(shutdownStatus.agents).to.have.lengthOf(0);
    });

    it('should handle agent communication throughout lifecycle', async() => {
      swarm = new RuvSwarm();
      await swarm.init({ topology: 'star' });

      // Spawn coordinator and worker agents
      const coordinator = await swarm.spawnAgent({
        type: 'coordinator',
        name: 'Central Coordinator',
      });

      const workers = await Promise.all([
        swarm.spawnAgent({ type: 'coder', name: 'Worker 1' }),
        swarm.spawnAgent({ type: 'coder', name: 'Worker 2' }),
        swarm.spawnAgent({ type: 'coder', name: 'Worker 3' }),
      ]);

      // Set up message tracking
      const messages = [];
      communicator.on('message', (msg) => messages.push(msg));

      // Orchestrate collaborative task
      const taskResult = await swarm.orchestrateTask({
        task: 'Implement microservices architecture',
        strategy: 'sequential',
        coordinatorId: coordinator.id,
      });

      // Wait for some communication
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Verify communication patterns
      expect(messages.length).to.be.greaterThan(0);

      const coordinatorMessages = messages.filter(m => m.from === coordinator.id);
      const workerMessages = messages.filter(m => workers.some(w => w.id === m.from));

      expect(coordinatorMessages.length).to.be.greaterThan(0);
      expect(workerMessages.length).to.be.greaterThan(0);

      // Verify star topology communication
      workerMessages.forEach(msg => {
        expect(msg.to).to.equal(coordinator.id);
      });
    });

    it('should persist and restore swarm state', async() => {
      // Create initial swarm
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'hierarchical',
        persistState: true,
      });

      // Spawn agents and create state
      const agents = await Promise.all([
        swarm.spawnAgent({ type: 'architect' }),
        swarm.spawnAgent({ type: 'coder' }),
        swarm.spawnAgent({ type: 'analyst' }),
      ]);

      const task = await swarm.orchestrateTask({
        task: 'Design system architecture',
        priority: 'high',
      });

      // Save current state
      const savedState = await swarm.exportState();
      expect(savedState).to.have.property('topology');
      expect(savedState).to.have.property('agents');
      expect(savedState).to.have.property('tasks');
      expect(savedState).to.have.property('memory');

      // Shutdown original swarm
      await swarm.shutdown();

      // Create new swarm and restore state
      const newSwarm = new RuvSwarm();
      await newSwarm.importState(savedState);

      // Verify restoration
      const restoredStatus = await newSwarm.getStatus();
      expect(restoredStatus.topology).to.equal('hierarchical');
      expect(restoredStatus.agents).to.have.lengthOf(3);

      const restoredTasks = await newSwarm.getActiveTasks();
      expect(restoredTasks).to.have.lengthOf(1);
      expect(restoredTasks[0].id).to.equal(task.id);

      await newSwarm.shutdown();
    });
  });

  describe('Neural Integration Lifecycle', () => {
    it('should integrate neural agents with standard workflow', async() => {
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'mesh',
        enableNeural: true,
      });

      // Create neural agent manager
      const neuralManager = new NeuralAgentManager();
      await neuralManager.initialize();

      // Spawn mixed agent types
      const agents = await Promise.all([
        swarm.spawnAgent({ type: 'neural', model: 'adaptive' }),
        swarm.spawnAgent({ type: 'researcher' }),
        swarm.spawnAgent({ type: 'neural', model: 'pattern-recognition' }),
        swarm.spawnAgent({ type: 'coder' }),
      ]);

      // Verify neural agents initialized
      const neuralAgents = agents.filter(a => a.type === 'neural');
      expect(neuralAgents).to.have.lengthOf(2);

      for (const agent of neuralAgents) {
        const status = await neuralManager.getAgentStatus(agent.id);
        expect(status).to.have.property('initialized', true);
        expect(status).to.have.property('model');
        expect(status).to.have.property('performance');
      }

      // Orchestrate task requiring neural processing
      const taskResult = await swarm.orchestrateTask({
        task: 'Analyze codebase patterns and suggest optimizations',
        requiresNeural: true,
        strategy: 'adaptive',
      });

      // Wait for neural processing
      await new Promise(resolve => setTimeout(resolve, 3000));

      // Verify neural contribution
      const taskStatus = await swarm.getTaskStatus(taskResult.id);
      expect(taskStatus.neuralContribution).to.exist;
      expect(taskStatus.neuralContribution).to.have.property('patterns');
      expect(taskStatus.neuralContribution).to.have.property('confidence');
      expect(taskStatus.neuralContribution.confidence).to.be.greaterThan(0.7);
    });

    it('should train neural patterns throughout lifecycle', async() => {
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'mesh',
        enableNeural: true,
        autoTrain: true,
      });

      const neuralManager = new NeuralAgentManager();
      await neuralManager.initialize();

      // Spawn neural agent
      const neuralAgent = await swarm.spawnAgent({
        type: 'neural',
        model: 'learning',
        trainable: true,
      });

      // Get initial performance
      const initialMetrics = await neuralManager.getAgentMetrics(neuralAgent.id);
      const initialAccuracy = initialMetrics.accuracy || 0.5;

      // Execute multiple training tasks
      const trainingTasks = [];
      for (let i = 0; i < 5; i++) {
        trainingTasks.push(swarm.orchestrateTask({
          task: `Training task ${i}: Pattern recognition`,
          agentId: neuralAgent.id,
          training: true,
        }));
      }

      await Promise.all(trainingTasks);

      // Train the neural agent
      await neuralManager.train(neuralAgent.id, { iterations: 10 });

      // Verify improvement
      const finalMetrics = await neuralManager.getAgentMetrics(neuralAgent.id);
      expect(finalMetrics.accuracy).to.be.greaterThan(initialAccuracy);
      expect(finalMetrics.trainingIterations).to.be.greaterThan(0);
    });
  });

  describe('Memory and State Management', () => {
    it('should maintain memory across agent lifecycle', async() => {
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'mesh',
        enableMemory: true,
      });

      // Create agents with memory
      const agent1 = await swarm.spawnAgent({
        type: 'researcher',
        memory: { capacity: 1000 },
      });

      const agent2 = await swarm.spawnAgent({
        type: 'analyst',
        memory: { capacity: 1000 },
      });

      // Store memories during task execution
      await swarm.orchestrateTask({
        task: 'Research and analyze market trends',
        agents: [agent1.id, agent2.id],
        storeMemory: true,
      });

      // Retrieve agent memories
      const memory1 = await swarm.getAgentMemory(agent1.id);
      const memory2 = await swarm.getAgentMemory(agent2.id);

      expect(memory1).to.have.property('experiences');
      expect(memory1).to.have.property('learnings');
      expect(memory2).to.have.property('experiences');
      expect(memory2).to.have.property('learnings');

      // Verify memory persistence
      const swarmMemory = await swarm.getSwarmMemory();
      expect(swarmMemory).to.have.property('collective');
      expect(swarmMemory.collective).to.have.property('taskHistory');
      expect(swarmMemory.collective).to.have.property('patterns');
    });

    it('should share memory between agents effectively', async() => {
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'mesh',
        enableMemory: true,
        sharedMemory: true,
      });

      // Create collaborative agents
      const agents = await Promise.all([
        swarm.spawnAgent({ type: 'researcher', sharedMemory: true }),
        swarm.spawnAgent({ type: 'coder', sharedMemory: true }),
        swarm.spawnAgent({ type: 'analyst', sharedMemory: true }),
      ]);

      // Execute collaborative task
      const task = await swarm.orchestrateTask({
        task: 'Collaborative code review and optimization',
        strategy: 'parallel',
        shareFindings: true,
      });

      // Wait for collaboration
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Verify shared memory
      const sharedMemory = await swarm.getSharedMemory();
      expect(sharedMemory).to.have.property('findings');
      expect(sharedMemory).to.have.property('decisions');
      expect(sharedMemory).to.have.property('consensus');

      // Each agent should have access to shared findings
      for (const agent of agents) {
        const agentView = await swarm.getAgentMemory(agent.id);
        expect(agentView.shared).to.deep.equal(sharedMemory);
      }
    });
  });
});