/**
 * Comprehensive test suite for Session Management System
 */

import { describe, test, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { SessionManager, SessionState, SessionConfig } from './session-manager.js';
import { SessionEnabledSwarm, SessionRecoveryService } from './session-integration.js';
import { SessionValidator, SessionSerializer, SessionRecovery, SessionStats } from './session-utils.js';
import { SwarmPersistencePooled } from './persistence-pooled.js';
import { SwarmOptions, SwarmState } from './types.js';

// Mock persistence layer
class MockPersistence extends SwarmPersistencePooled {
  private mockData: Map<string, any> = new Map();
  public initialized: boolean = false;
  
  constructor() {
    super(':memory:'); // Use in-memory SQLite for testing
  }

  override async initialize() {
    // Mock initialization
    this.initialized = true;
  }

  // Mock pool with read/write methods
  override pool = {
    read: jest.fn(async (sql: string, params?: any[]) => {
      if (sql.includes('sessions')) {
        return Array.from(this.mockData.values()).filter((item: any) => item.type === 'session');
      }
      if (sql.includes('session_checkpoints')) {
        return Array.from(this.mockData.values()).filter((item: any) => item.type === 'checkpoint');
      }
      return [];
    }),
    
    write: jest.fn(async (sql: string, params?: any[]) => {
      if (sql.includes('INSERT INTO sessions')) {
        const [id, name, status, swarmOptions, swarmState, metadata, createdAt, lastAccessedAt, version] = params || [];
        this.mockData.set(id, {
          type: 'session',
          id,
          name,
          status,
          swarm_options: swarmOptions,
          swarm_state: swarmState,
          metadata,
          created_at: createdAt,
          last_accessed_at: lastAccessedAt,
          version,
        });
      }
      if (sql.includes('INSERT INTO session_checkpoints')) {
        const [id, sessionId, timestamp, checksum, stateData, description, metadata] = params || [];
        this.mockData.set(id, {
          type: 'checkpoint',
          id,
          session_id: sessionId,
          timestamp,
          checksum,
          state_data: stateData,
          description,
          metadata,
        });
      }
      return { changes: 1, lastInsertRowid: 1 };
    }),
  };
}

describe('SessionManager', () => {
  let persistence: MockPersistence;
  let sessionManager: SessionManager;
  
  const mockSwarmOptions: SwarmOptions = {
    topology: 'mesh',
    maxAgents: 10,
    connectionDensity: 0.5,
    syncInterval: 1000,
  };

  const mockSwarmState: SwarmState = {
    agents: new Map(),
    tasks: new Map(),
    topology: 'mesh',
    connections: [],
    metrics: {
      totalTasks: 0,
      completedTasks: 0,
      failedTasks: 0,
      averageCompletionTime: 0,
      agentUtilization: new Map(),
      throughput: 0,
    },
  };

  beforeEach(async () => {
    persistence = new MockPersistence();
    await persistence.initialize();
    
    sessionManager = new SessionManager(persistence, {
      autoCheckpoint: false, // Disable for testing
      checkpointInterval: 60000,
      maxCheckpoints: 5,
    });
    
    await sessionManager.initialize();
  });

  afterEach(async () => {
    await sessionManager.shutdown();
  });

  describe('Session Lifecycle', () => {
    test('should create a new session', async () => {
      const sessionId = await sessionManager.createSession(
        'Test Session',
        mockSwarmOptions,
        mockSwarmState
      );

      expect(sessionId).toBeDefined();
      expect(sessionId).toMatch(/^session_/);
    });

    test('should load an existing session', async () => {
      const sessionId = await sessionManager.createSession(
        'Test Session',
        mockSwarmOptions,
        mockSwarmState
      );

      const loadedSession = await sessionManager.loadSession(sessionId);
      
      expect(loadedSession.id).toBe(sessionId);
      expect(loadedSession.name).toBe('Test Session');
      expect(loadedSession.status).toBe('active');
    });

    test('should save session state', async () => {
      const sessionId = await sessionManager.createSession(
        'Test Session',
        mockSwarmOptions,
        mockSwarmState
      );

      const updatedState = {
        ...mockSwarmState,
        metrics: {
          ...mockSwarmState.metrics,
          totalTasks: 5,
          completedTasks: 3,
        },
      };

      await sessionManager.saveSession(sessionId, updatedState);
      const savedSession = await sessionManager.loadSession(sessionId);
      
      expect(savedSession.swarmState.metrics.totalTasks).toBe(5);
      expect(savedSession.swarmState.metrics.completedTasks).toBe(3);
    });

    test('should pause and resume session', async () => {
      const sessionId = await sessionManager.createSession(
        'Test Session',
        mockSwarmOptions,
        mockSwarmState
      );

      await sessionManager.pauseSession(sessionId);
      let session = await sessionManager.loadSession(sessionId);
      expect(session.status).toBe('paused');

      await sessionManager.resumeSession(sessionId);
      session = await sessionManager.loadSession(sessionId);
      expect(session.status).toBe('active');
    });

    test('should hibernate and restore session', async () => {
      const sessionId = await sessionManager.createSession(
        'Test Session',
        mockSwarmOptions,
        mockSwarmState
      );

      await sessionManager.hibernateSession(sessionId);
      
      // Session should be hibernated and removed from active sessions
      const session = await sessionManager.loadSession(sessionId);
      expect(session.status).toBe('hibernated');
    });

    test('should terminate session', async () => {
      const sessionId = await sessionManager.createSession(
        'Test Session',
        mockSwarmOptions,
        mockSwarmState
      );

      await sessionManager.terminateSession(sessionId);
      
      const session = await sessionManager.loadSession(sessionId);
      expect(session.status).toBe('terminated');
    });
  });

  describe('Checkpoint System', () => {
    test('should create checkpoint', async () => {
      const sessionId = await sessionManager.createSession(
        'Test Session',
        mockSwarmOptions,
        mockSwarmState
      );

      const checkpointId = await sessionManager.createCheckpoint(
        sessionId,
        'Test checkpoint'
      );

      expect(checkpointId).toBeDefined();
      expect(checkpointId).toMatch(/^checkpoint_/);
    });

    test('should restore from checkpoint', async () => {
      const sessionId = await sessionManager.createSession(
        'Test Session',
        mockSwarmOptions,
        mockSwarmState
      );

      // Create initial checkpoint
      const checkpointId = await sessionManager.createCheckpoint(
        sessionId,
        'Initial state'
      );

      // Modify session state
      const modifiedState = {
        ...mockSwarmState,
        metrics: {
          ...mockSwarmState.metrics,
          totalTasks: 10,
        },
      };
      await sessionManager.saveSession(sessionId, modifiedState);

      // Restore from checkpoint
      await sessionManager.restoreFromCheckpoint(sessionId, checkpointId);

      const restoredSession = await sessionManager.loadSession(sessionId);
      expect(restoredSession.swarmState.metrics.totalTasks).toBe(0); // Original state
    });
  });

  describe('Session Statistics', () => {
    test('should get session statistics', async () => {
      const sessionId = await sessionManager.createSession(
        'Test Session',
        mockSwarmOptions,
        mockSwarmState
      );

      const stats = await sessionManager.getSessionStats(sessionId);
      
      expect(stats).toHaveProperty('sessionId', sessionId);
      expect(stats).toHaveProperty('name', 'Test Session');
      expect(stats).toHaveProperty('status', 'active');
      expect(stats).toHaveProperty('totalAgents');
      expect(stats).toHaveProperty('totalTasks');
    });

    test('should get global statistics', async () => {
      await sessionManager.createSession('Session 1', mockSwarmOptions, mockSwarmState);
      await sessionManager.createSession('Session 2', mockSwarmOptions, mockSwarmState);

      const globalStats = await sessionManager.getSessionStats();
      
      expect(globalStats).toHaveProperty('totalSessions');
      expect(globalStats).toHaveProperty('activeSessions');
      expect(globalStats).toHaveProperty('statusBreakdown');
    });
  });

  describe('Error Handling', () => {
    test('should handle non-existent session', async () => {
      await expect(
        sessionManager.loadSession('non-existent-session')
      ).rejects.toThrow('Session non-existent-session not found');
    });

    test('should handle invalid checkpoint restoration', async () => {
      const sessionId = await sessionManager.createSession(
        'Test Session',
        mockSwarmOptions,
        mockSwarmState
      );

      await expect(
        sessionManager.restoreFromCheckpoint(sessionId, 'invalid-checkpoint')
      ).rejects.toThrow('Checkpoint invalid-checkpoint not found');
    });
  });
});

describe('SessionEnabledSwarm', () => {
  let persistence: MockPersistence;
  let swarm: SessionEnabledSwarm;

  beforeEach(async () => {
    persistence = new MockPersistence();
    swarm = new SessionEnabledSwarm(
      {
        topology: 'mesh',
        maxAgents: 5,
      },
      {
        autoCheckpoint: false,
      },
      persistence
    );
    
    await swarm.init();
  });

  afterEach(async () => {
    await swarm.destroy();
  });

  test('should create session-enabled swarm', async () => {
    const sessionId = await swarm.createSession('Test Swarm Session');
    
    expect(sessionId).toBeDefined();
    
    const currentSession = await swarm.getCurrentSession();
    expect(currentSession).not.toBeNull();
    expect(currentSession!.name).toBe('Test Swarm Session');
  });

  test('should auto-save on agent addition', async () => {
    const sessionId = await swarm.createSession('Test Swarm Session');
    
    const agentId = swarm.addAgent({
      id: 'test-agent',
      type: 'researcher',
    });

    expect(agentId).toBeDefined();
    
    // Give some time for auto-save to complete
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const session = await swarm.getCurrentSession();
    expect(session!.swarmState.agents.size).toBeGreaterThan(0);
  });

  test('should create and restore from checkpoint', async () => {
    const sessionId = await swarm.createSession('Test Swarm Session');
    
    swarm.addAgent({
      id: 'test-agent',
      type: 'researcher',
    });

    const checkpointId = await swarm.createCheckpoint('Test checkpoint');
    expect(checkpointId).toBeDefined();

    // Add another agent
    swarm.addAgent({
      id: 'test-agent-2',
      type: 'coder',
    });

    // Restore from checkpoint
    await swarm.restoreFromCheckpoint(checkpointId);
    
    const session = await swarm.getCurrentSession();
    expect(session!.swarmState.agents.size).toBe(1); // Should have only the first agent
  });
});

describe('SessionValidator', () => {
  const validSession: SessionState = {
    id: 'test-session',
    name: 'Test Session',
    createdAt: new Date(),
    lastAccessedAt: new Date(),
    status: 'active',
    swarmState: {
      agents: new Map(),
      tasks: new Map(),
      topology: 'mesh',
      connections: [],
      metrics: {
        totalTasks: 0,
        completedTasks: 0,
        failedTasks: 0,
        averageCompletionTime: 0,
        agentUtilization: new Map(),
        throughput: 0,
      },
    },
    swarmOptions: {
      topology: 'mesh',
      maxAgents: 10,
    },
    metadata: {},
    checkpoints: [],
    version: '1.0.0',
  };

  test('should validate valid session state', () => {
    const result = SessionValidator.validateSessionState(validSession);
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  test('should detect invalid session state', () => {
    const invalidSession = {
      ...validSession,
      id: null, // Invalid ID
      status: 'invalid-status', // Invalid status
    } as any;

    const result = SessionValidator.validateSessionState(invalidSession);
    expect(result.valid).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });
});

describe('SessionSerializer', () => {
  const testSwarmState: SwarmState = {
    agents: new Map([['agent1', { id: 'agent1', type: 'researcher' } as any]]),
    tasks: new Map([['task1', { id: 'task1', description: 'Test task' } as any]]),
    topology: 'mesh',
    connections: [],
    metrics: {
      totalTasks: 1,
      completedTasks: 0,
      failedTasks: 0,
      averageCompletionTime: 0,
      agentUtilization: new Map([['agent1', 0.5]]),
      throughput: 0,
    },
  };

  test('should serialize and deserialize swarm state', () => {
    const serialized = SessionSerializer.serializeSwarmState(testSwarmState);
    expect(serialized).toBeDefined();
    expect(typeof serialized).toBe('string');

    const deserialized = SessionSerializer.deserializeSwarmState(serialized);
    expect(deserialized.agents.size).toBe(1);
    expect(deserialized.tasks.size).toBe(1);
    expect(deserialized.topology).toBe('mesh');
    expect(deserialized.metrics.totalTasks).toBe(1);
    expect(deserialized.metrics.agentUtilization.get('agent1')).toBe(0.5);
  });

  test('should export and import session', () => {
    const testSession: SessionState = {
      id: 'test-session',
      name: 'Test Session',
      createdAt: new Date('2023-01-01'),
      lastAccessedAt: new Date('2023-01-01'),
      status: 'active',
      swarmState: testSwarmState,
      swarmOptions: { topology: 'mesh' },
      metadata: { test: true },
      checkpoints: [],
      version: '1.0.0',
    };

    const exported = SessionSerializer.exportSession(testSession);
    expect(exported).toBeDefined();
    expect(typeof exported).toBe('string');

    const imported = SessionSerializer.importSession(exported);
    expect(imported.id).toBe('test-session');
    expect(imported.name).toBe('Test Session');
    expect(imported.swarmState.agents.size).toBe(1);
    expect(imported.metadata.test).toBe(true);
  });
});

describe('SessionStats', () => {
  const testSession: SessionState = {
    id: 'test-session',
    name: 'Test Session',
    createdAt: new Date(Date.now() - 86400000), // 1 day ago
    lastAccessedAt: new Date(Date.now() - 3600000), // 1 hour ago
    status: 'active',
    swarmState: {
      agents: new Map([['agent1', {} as any]]),
      tasks: new Map([['task1', {} as any]]),
      topology: 'mesh',
      connections: [],
      metrics: {
        totalTasks: 10,
        completedTasks: 8,
        failedTasks: 2,
        averageCompletionTime: 1000,
        agentUtilization: new Map(),
        throughput: 0.5,
      },
    },
    swarmOptions: { topology: 'mesh' },
    metadata: {},
    checkpoints: [
      {
        id: 'cp1',
        sessionId: 'test-session',
        timestamp: new Date(),
        checksum: 'abc123',
        state: {} as any,
        description: 'Test checkpoint',
        metadata: {},
      },
    ],
    version: '1.0.0',
  };

  test('should calculate health score', () => {
    const healthScore = SessionStats.calculateHealthScore(testSession);
    expect(healthScore).toBeGreaterThan(0);
    expect(healthScore).toBeLessThanOrEqual(100);
  });

  test('should generate session summary', () => {
    const summary = SessionStats.generateSummary(testSession);
    
    expect(summary).toHaveProperty('id', 'test-session');
    expect(summary).toHaveProperty('name', 'Test Session');
    expect(summary).toHaveProperty('status', 'active');
    expect(summary).toHaveProperty('healthScore');
    expect(summary).toHaveProperty('agents');
    expect(summary).toHaveProperty('tasks');
    expect(summary).toHaveProperty('checkpoints');
    expect(summary).toHaveProperty('performance');
    
    expect(summary.agents.total).toBe(1);
    expect(summary.tasks.total).toBe(10);
    expect(summary.tasks.completed).toBe(8);
    expect(summary.tasks.successRate).toBe(0.8);
    expect(summary.checkpoints.total).toBe(1);
  });
});

describe('SessionRecoveryService', () => {
  let persistence: MockPersistence;
  let sessionManager: SessionManager;
  let recoveryService: SessionRecoveryService;

  beforeEach(async () => {
    persistence = new MockPersistence();
    await persistence.initialize();
    
    sessionManager = new SessionManager(persistence, {
      autoCheckpoint: false,
    });
    await sessionManager.initialize();
    
    recoveryService = new SessionRecoveryService(sessionManager);
  });

  afterEach(async () => {
    await sessionManager.shutdown();
  });

  test('should run health check', async () => {
    // Create a few test sessions
    await sessionManager.createSession('Session 1', { topology: 'mesh' });
    await sessionManager.createSession('Session 2', { topology: 'hierarchical' });

    const healthReport = await recoveryService.runHealthCheck();
    
    expect(healthReport).toHaveProperty('total');
    expect(healthReport).toHaveProperty('healthy');
    expect(healthReport).toHaveProperty('corrupted');
    expect(healthReport).toHaveProperty('needsRecovery');
    expect(healthReport).toHaveProperty('recoveryRecommendations');
    
    expect(healthReport.total).toBeGreaterThanOrEqual(2);
  });
});

// Integration tests
describe('Session Management Integration', () => {
  test('should handle complete session lifecycle', async () => {
    const persistence = new MockPersistence();
    await persistence.initialize();
    
    const swarm = new SessionEnabledSwarm(
      { topology: 'mesh', maxAgents: 10 },
      { autoCheckpoint: false },
      persistence
    );
    
    await swarm.init();

    try {
      // Create session
      const sessionId = await swarm.createSession('Integration Test Session');
      expect(sessionId).toBeDefined();

      // Add agents and tasks
      const agentId = swarm.addAgent({
        id: 'test-agent',
        type: 'researcher',
      });
      
      const taskId = await swarm.submitTask({
        description: 'Test task',
        priority: 'medium',
      });

      // Create checkpoint
      const checkpointId = await swarm.createCheckpoint('Integration checkpoint');
      expect(checkpointId).toBeDefined();

      // Verify session state
      const session = await swarm.getCurrentSession();
      expect(session).not.toBeNull();
      expect(session!.swarmState.agents.size).toBeGreaterThan(0);
      expect(session!.swarmState.tasks.size).toBeGreaterThan(0);
      expect(session!.checkpoints.length).toBeGreaterThan(0);

      // Pause and resume
      await swarm.pauseSession();
      await swarm.resumeSession();

      // Get statistics
      const stats = await swarm.getSessionStats();
      expect(stats).toHaveProperty('sessionId');
      expect(stats).toHaveProperty('totalAgents');
      expect(stats).toHaveProperty('totalTasks');

      // Terminate session
      await swarm.terminateSession();
      
    } finally {
      await swarm.destroy();
    }
  });
});