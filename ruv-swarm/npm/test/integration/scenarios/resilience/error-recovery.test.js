const { expect } = require('chai');
const sinon = require('sinon');
const RuvSwarm = require('../../../../src/core/ruv-swarm');
const Agent = require('../../../../src/agent');
const Database = require('better-sqlite3');
const EventEmitter = require('events');

describe('Error Recovery and Resilience Integration', () => {
  let sandbox;
  let swarm;
  let db;

  beforeEach(() => {
    sandbox = sinon.createSandbox();
  });

  afterEach(async() => {
    if (swarm) {
      await swarm.shutdown();
    }
    if (db && db.open) {
      db.close();
    }
    sandbox.restore();
  });

  describe('Component Failure Recovery', () => {
    it('should recover from agent failures gracefully', async() => {
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'mesh',
        resilience: { enabled: true, retryAttempts: 3 },
      });

      // Spawn multiple agents
      const agents = await Promise.all([
        swarm.spawnAgent({ type: 'coder' }),
        swarm.spawnAgent({ type: 'coder' }),
        swarm.spawnAgent({ type: 'coder' }),
      ]);

      // Simulate agent failure
      const failingAgent = agents[1];
      sandbox.stub(failingAgent, 'executeTask').rejects(new Error('Agent crashed'));

      // Orchestrate task
      const task = await swarm.orchestrateTask({
        task: 'Process data with fault tolerance',
        strategy: 'parallel',
        faultTolerant: true,
      });

      // Wait for task completion
      let taskStatus;
      for (let i = 0; i < 10; i++) {
        taskStatus = await swarm.getTaskStatus(task.id);
        if (taskStatus.status === 'completed' || taskStatus.status === 'failed') {
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      // Verify task completed despite agent failure
      expect(taskStatus.status).to.equal('completed');
      expect(taskStatus.failedAgents).to.include(failingAgent.id);
      expect(taskStatus.recoveryActions).to.have.lengthOf.at.least(1);
      expect(taskStatus.result).to.exist;
    });

    it('should handle cascading failures', async() => {
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'hierarchical',
        resilience: {
          enabled: true,
          cascadeProtection: true,
          isolationMode: 'strict',
        },
      });

      // Create hierarchy
      const coordinator = await swarm.spawnAgent({ type: 'coordinator' });
      const midLevel = await Promise.all([
        swarm.spawnAgent({ type: 'analyst', parentId: coordinator.id }),
        swarm.spawnAgent({ type: 'analyst', parentId: coordinator.id }),
      ]);
      const workers = await Promise.all([
        swarm.spawnAgent({ type: 'coder', parentId: midLevel[0].id }),
        swarm.spawnAgent({ type: 'coder', parentId: midLevel[0].id }),
        swarm.spawnAgent({ type: 'coder', parentId: midLevel[1].id }),
      ]);

      // Simulate mid-level failure
      sandbox.stub(midLevel[0], 'executeTask').rejects(new Error('Mid-level crash'));

      // Execute hierarchical task
      const task = await swarm.orchestrateTask({
        task: 'Hierarchical processing with failure',
        coordinatorId: coordinator.id,
        cascadeProtection: true,
      });

      // Monitor recovery
      const recoveryEvents = [];
      swarm.on('recovery', (event) => recoveryEvents.push(event));

      // Wait for recovery
      await new Promise(resolve => setTimeout(resolve, 3000));

      // Verify cascade protection
      expect(recoveryEvents).to.have.lengthOf.at.least(1);

      const status = await swarm.getStatus();
      const activeAgents = status.agents.filter(a => a.status !== 'failed');

      // Coordinator and other branch should survive
      expect(activeAgents).to.include.deep.members([
        { id: coordinator.id, status: 'idle' },
        { id: midLevel[1].id, status: 'idle' },
      ]);

      // Workers under failed mid-level should be reassigned
      const reassignedWorkers = recoveryEvents.filter(e => e.type === 'reassignment');
      expect(reassignedWorkers).to.have.lengthOf.at.least(2);
    });

    it('should implement circuit breaker pattern', async() => {
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'mesh',
        resilience: {
          circuitBreaker: {
            enabled: true,
            threshold: 3,
            timeout: 5000,
            halfOpenRetries: 2,
          },
        },
      });

      const agent = await swarm.spawnAgent({ type: 'coder' });

      // Create a failing service
      let callCount = 0;
      const failingService = sandbox.stub().callsFake(() => {
        callCount++;
        if (callCount <= 5) {
          throw new Error('Service unavailable');
        }
        return { success: true };
      });

      // Replace agent's service call
      agent.callService = failingService;

      // Attempt multiple calls
      const results = [];
      for (let i = 0; i < 10; i++) {
        try {
          const result = await swarm.executeAgentTask(agent.id, {
            task: 'Call external service',
            useCircuitBreaker: true,
          });
          results.push({ success: true, result });
        } catch (error) {
          results.push({ success: false, error: error.message });
        }
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      // Verify circuit breaker behavior
      const failures = results.filter(r => !r.success);
      const circuitOpenErrors = failures.filter(f => f.error.includes('Circuit breaker open'));

      expect(failures).to.have.lengthOf.at.least(3);
      expect(circuitOpenErrors).to.have.lengthOf.at.least(1);

      // Eventually should succeed after circuit closes
      const successes = results.filter(r => r.success);
      expect(successes).to.have.lengthOf.at.least(1);
    });
  });

  describe('Database and Persistence Recovery', () => {
    it('should handle database connection failures', async() => {
      swarm = new RuvSwarm();

      // Mock database with intermittent failures
      const mockDb = {
        prepare: sandbox.stub(),
        close: sandbox.stub(),
        open: true,
      };

      let failureCount = 0;
      mockDb.prepare.callsFake((query) => {
        if (failureCount++ < 2) {
          throw new Error('Database connection lost');
        }
        return {
          all: () => [],
          run: () => ({ changes: 1 }),
          get: () => ({ id: 1 }),
        };
      });

      await swarm.init({
        topology: 'mesh',
        persistence: {
          enabled: true,
          retryOnFailure: true,
          maxRetries: 5,
        },
        database: mockDb,
      });

      // Spawn agent with persistence
      const agent = await swarm.spawnAgent({
        type: 'researcher',
        persistent: true,
      });

      // Verify agent created despite initial failures
      expect(agent).to.exist;
      expect(agent.id).to.exist;

      // Verify retries occurred
      expect(mockDb.prepare.callCount).to.be.greaterThan(2);
    });

    it('should implement write-ahead logging for recovery', async() => {
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'mesh',
        persistence: {
          enabled: true,
          wal: true,
          checkpoint: 1000,
        },
      });

      // Create multiple agents and tasks
      const agents = await Promise.all([
        swarm.spawnAgent({ type: 'coder' }),
        swarm.spawnAgent({ type: 'analyst' }),
      ]);

      const tasks = [];
      for (let i = 0; i < 5; i++) {
        tasks.push(swarm.orchestrateTask({
          task: `Task ${i}`,
          persistent: true,
        }));
      }

      await Promise.all(tasks);

      // Simulate crash by forcing shutdown
      swarm._forceShutdown = true;

      // Create new swarm instance
      const recoveredSwarm = new RuvSwarm();
      await recoveredSwarm.init({
        topology: 'mesh',
        persistence: {
          enabled: true,
          recoverFromWAL: true,
        },
      });

      // Verify state recovered from WAL
      const status = await recoveredSwarm.getStatus();
      expect(status.agents).to.have.lengthOf(2);

      const recoveredTasks = await recoveredSwarm.getAllTasks();
      expect(recoveredTasks).to.have.lengthOf(5);

      await recoveredSwarm.shutdown();
    });

    it('should handle corrupted state gracefully', async() => {
      // Create corrupted database
      const corruptDb = new Database(':memory:');
      corruptDb.exec(`
        CREATE TABLE agents (
          id TEXT PRIMARY KEY,
          data TEXT
        );
        CREATE TABLE tasks (
          id TEXT PRIMARY KEY,
          data TEXT
        );
      `);

      // Insert corrupted data
      corruptDb.prepare('INSERT INTO agents VALUES (?, ?)').run('agent1', 'CORRUPTED{{{');
      corruptDb.prepare('INSERT INTO tasks VALUES (?, ?)').run('task1', '{"invalid": json}');

      swarm = new RuvSwarm();
      const initResult = await swarm.init({
        topology: 'mesh',
        persistence: {
          enabled: true,
          handleCorruption: true,
          database: corruptDb,
        },
      });

      // Should initialize despite corruption
      expect(initResult.success).to.be.true;
      expect(initResult.corruptedEntries).to.have.lengthOf(2);
      expect(initResult.recoveryMode).to.be.true;

      // Should be able to operate normally
      const agent = await swarm.spawnAgent({ type: 'coder' });
      expect(agent).to.exist;

      corruptDb.close();
    });
  });

  describe('Network and Communication Recovery', () => {
    it('should handle network partition scenarios', async() => {
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'mesh',
        network: {
          partitionTolerant: true,
          consensusProtocol: 'raft',
        },
      });

      // Create agents across partitions
      const partition1 = await Promise.all([
        swarm.spawnAgent({ type: 'coder', partition: 1 }),
        swarm.spawnAgent({ type: 'analyst', partition: 1 }),
      ]);

      const partition2 = await Promise.all([
        swarm.spawnAgent({ type: 'coder', partition: 2 }),
        swarm.spawnAgent({ type: 'researcher', partition: 2 }),
      ]);

      // Simulate network partition
      swarm.simulatePartition([1, 2]);

      // Each partition should continue operating
      const task1 = await swarm.orchestrateTask({
        task: 'Partition 1 task',
        partition: 1,
      });

      const task2 = await swarm.orchestrateTask({
        task: 'Partition 2 task',
        partition: 2,
      });

      // Wait for tasks
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Heal partition
      swarm.healPartition();

      // Wait for reconciliation
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Verify both tasks completed and state reconciled
      const allTasks = await swarm.getAllTasks();
      expect(allTasks).to.include.deep.members([
        { id: task1.id, status: 'completed' },
        { id: task2.id, status: 'completed' },
      ]);

      // Verify conflict resolution
      const conflicts = await swarm.getConflictLog();
      expect(conflicts).to.be.an('array');

      if (conflicts.length > 0) {
        conflicts.forEach(conflict => {
          expect(conflict).to.have.property('resolved', true);
          expect(conflict).to.have.property('resolution');
        });
      }
    });

    it('should implement retry with exponential backoff', async() => {
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'star',
        network: {
          retryPolicy: {
            enabled: true,
            initialDelay: 100,
            maxDelay: 5000,
            backoffMultiplier: 2,
            maxAttempts: 5,
          },
        },
      });

      const agent = await swarm.spawnAgent({ type: 'coder' });

      // Mock flaky network
      let attempts = 0;
      const flakyNetwork = sandbox.stub(swarm._network, 'send').callsFake(async() => {
        attempts++;
        if (attempts < 4) {
          throw new Error('Network timeout');
        }
        return { success: true };
      });

      const startTime = Date.now();
      const result = await swarm.sendAgentMessage(agent.id, {
        type: 'task',
        data: 'Important message',
      });

      const elapsed = Date.now() - startTime;

      // Verify retries with backoff
      expect(attempts).to.equal(4);
      expect(result.success).to.be.true;

      // Should have taken at least: 100 + 200 + 400 = 700ms
      expect(elapsed).to.be.at.least(700);
    });

    it('should handle message queue overflow', async() => {
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'mesh',
        messageQueue: {
          maxSize: 100,
          overflowStrategy: 'drop-oldest',
          persistQueue: true,
        },
      });

      const agents = await Promise.all([
        swarm.spawnAgent({ type: 'coder' }),
        swarm.spawnAgent({ type: 'analyst' }),
      ]);

      // Flood with messages
      const messages = [];
      for (let i = 0; i < 150; i++) {
        messages.push(swarm.broadcastMessage({
          type: 'update',
          priority: i < 50 ? 'low' : 'high',
          data: `Message ${i}`,
        }));
      }

      await Promise.all(messages);

      // Check queue state
      const queueStats = await swarm.getQueueStats();
      expect(queueStats.currentSize).to.equal(100);
      expect(queueStats.dropped).to.equal(50);
      expect(queueStats.overflowEvents).to.be.greaterThan(0);

      // High priority messages should be preserved
      const remainingMessages = await swarm.getQueuedMessages();
      const highPriorityCount = remainingMessages.filter(m => m.priority === 'high').length;
      expect(highPriorityCount).to.be.greaterThan(50);
    });
  });

  describe('Resource Management and Cleanup', () => {
    it('should prevent resource leaks under stress', async() => {
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'mesh',
        resourceLimits: {
          maxMemory: 512 * 1024 * 1024, // 512MB
          maxAgents: 50,
          gcInterval: 1000,
        },
      });

      // Track initial memory
      const initialMemory = process.memoryUsage().heapUsed;

      // Stress test with rapid agent creation/destruction
      for (let cycle = 0; cycle < 5; cycle++) {
        const agents = [];

        // Spawn many agents
        for (let i = 0; i < 20; i++) {
          agents.push(await swarm.spawnAgent({
            type: 'coder',
            ephemeral: true,
          }));
        }

        // Execute tasks
        const tasks = agents.map(agent =>
          swarm.executeAgentTask(agent.id, {
            task: 'Memory intensive operation',
            data: Buffer.alloc(1024 * 1024), // 1MB per task
          }),
        );

        await Promise.all(tasks);

        // Destroy agents
        await Promise.all(agents.map(a => swarm.destroyAgent(a.id)));

        // Force GC if available
        if (global.gc) {
          global.gc();
        }

        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // Check memory didn't grow excessively
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryGrowth = finalMemory - initialMemory;

      // Should not grow more than 50MB
      expect(memoryGrowth).to.be.lessThan(50 * 1024 * 1024);

      // Verify no leaked agents
      const status = await swarm.getStatus();
      expect(status.agents).to.have.lengthOf(0);
    });

    it('should handle graceful degradation under load', async() => {
      swarm = new RuvSwarm();
      await swarm.init({
        topology: 'mesh',
        loadBalancing: {
          enabled: true,
          strategy: 'least-loaded',
          degradationThreshold: 0.8,
        },
      });

      // Create limited agents
      const agents = await Promise.all([
        swarm.spawnAgent({ type: 'coder', capacity: 5 }),
        swarm.spawnAgent({ type: 'coder', capacity: 5 }),
        swarm.spawnAgent({ type: 'coder', capacity: 5 }),
      ]);

      // Generate high load
      const tasks = [];
      for (let i = 0; i < 30; i++) {
        tasks.push(swarm.orchestrateTask({
          task: `High load task ${i}`,
          priority: i < 10 ? 'high' : 'normal',
          estimatedLoad: 1,
        }));
      }

      // Monitor degradation
      const degradationEvents = [];
      swarm.on('degradation', (event) => degradationEvents.push(event));

      // Wait for processing
      await new Promise(resolve => setTimeout(resolve, 5000));

      // Verify graceful degradation occurred
      expect(degradationEvents).to.have.lengthOf.at.least(1);

      const results = await Promise.allSettled(tasks);
      const completed = results.filter(r => r.status === 'fulfilled').length;
      const rejected = results.filter(r => r.status === 'rejected').length;

      // High priority tasks should complete
      const highPriorityResults = results.slice(0, 10);
      const highPriorityCompleted = highPriorityResults.filter(r => r.status === 'fulfilled').length;
      expect(highPriorityCompleted).to.be.at.least(8);

      // Some lower priority might be rejected
      expect(rejected).to.be.greaterThan(0);
    });
  });
});