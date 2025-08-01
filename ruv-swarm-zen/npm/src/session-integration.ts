/**
 * Session Integration Layer
 * 
 * Integrates the SessionManager with the existing RuvSwarm system,
 * providing seamless session persistence for swarm operations.
 */

import { EventEmitter } from 'events';
import { RuvSwarm } from './index.js';
import { SessionManager, SessionState, SessionConfig } from './session-manager.js';
import { SwarmPersistencePooled } from './persistence-pooled.js';
import { SwarmState, SwarmOptions, SwarmEvent, AgentConfig, Task } from './types.js';
import { SessionValidator, SessionSerializer, SessionRecovery } from './session-utils.js';

/**
 * Enhanced RuvSwarm with session management capabilities
 */
export class SessionEnabledSwarm extends RuvSwarm {
  private sessionManager: SessionManager;
  private currentSessionId?: string;
  private sessionIntegrationEnabled: boolean = false;

  constructor(
    options: SwarmOptions = {},
    sessionConfig: SessionConfig = {},
    persistence?: SwarmPersistencePooled
  ) {
    super(options);

    // Initialize session manager with existing or new persistence layer
    const persistenceLayer = persistence || new SwarmPersistencePooled();
    this.sessionManager = new SessionManager(persistenceLayer, sessionConfig);

    // Set up event forwarding
    this.setupEventForwarding();
  }

  /**
   * Initialize swarm with session support
   */
  override async init(): Promise<void> {
    // Initialize base swarm
    await super.init();

    // Initialize session manager
    await this.sessionManager.initialize();

    this.sessionIntegrationEnabled = true;
    this.emit('session:integration_enabled' as SwarmEvent, {});
  }

  /**
   * Create a new session and associate with this swarm
   */
  async createSession(sessionName: string): Promise<string> {
    if (!this.sessionIntegrationEnabled) {
      throw new Error('Session integration not enabled. Call init() first.');
    }

    const currentState = await this.captureCurrentState();
    const sessionId = await this.sessionManager.createSession(
      sessionName,
      this.options,
      currentState
    );

    this.currentSessionId = sessionId;
    this.emit('session:created' as SwarmEvent, { sessionId, sessionName });

    return sessionId;
  }

  /**
   * Load an existing session and restore swarm state
   */
  async loadSession(sessionId: string): Promise<void> {
    if (!this.sessionIntegrationEnabled) {
      throw new Error('Session integration not enabled. Call init() first.');
    }

    const session = await this.sessionManager.loadSession(sessionId);
    
    // Restore swarm state from session
    await this.restoreFromSessionState(session);

    this.currentSessionId = sessionId;
    this.emit('session:loaded' as SwarmEvent, { sessionId, sessionName: session.name });
  }

  /**
   * Save current swarm state to session
   */
  async saveSession(): Promise<void> {
    if (!this.currentSessionId) {
      throw new Error('No active session. Create or load a session first.');
    }

    const currentState = await this.captureCurrentState();
    await this.sessionManager.saveSession(this.currentSessionId, currentState);

    this.emit('session:saved' as SwarmEvent, { sessionId: this.currentSessionId });
  }

  /**
   * Create a checkpoint of current state
   */
  async createCheckpoint(description?: string): Promise<string> {
    if (!this.currentSessionId) {
      throw new Error('No active session. Create or load a session first.');
    }

    // Ensure current state is saved before checkpointing
    await this.saveSession();

    const checkpointId = await this.sessionManager.createCheckpoint(
      this.currentSessionId,
      description || 'Manual checkpoint'
    );

    this.emit('session:checkpoint_created' as SwarmEvent, { 
      sessionId: this.currentSessionId, 
      checkpointId,
      description 
    });

    return checkpointId;
  }

  /**
   * Restore from a specific checkpoint
   */
  async restoreFromCheckpoint(checkpointId: string): Promise<void> {
    if (!this.currentSessionId) {
      throw new Error('No active session. Create or load a session first.');
    }

    await this.sessionManager.restoreFromCheckpoint(this.currentSessionId, checkpointId);

    // Reload the session to get the restored state
    const session = await this.sessionManager.loadSession(this.currentSessionId);
    await this.restoreFromSessionState(session);

    this.emit('session:restored' as SwarmEvent, { 
      sessionId: this.currentSessionId, 
      checkpointId 
    });
  }

  /**
   * Pause the current session
   */
  async pauseSession(): Promise<void> {
    if (!this.currentSessionId) {
      throw new Error('No active session. Create or load a session first.');
    }

    // Save current state before pausing
    await this.saveSession();

    await this.sessionManager.pauseSession(this.currentSessionId);
    this.emit('session:paused' as SwarmEvent, { sessionId: this.currentSessionId });
  }

  /**
   * Resume a paused session
   */
  async resumeSession(): Promise<void> {
    if (!this.currentSessionId) {
      throw new Error('No active session. Create or load a session first.');
    }

    await this.sessionManager.resumeSession(this.currentSessionId);
    this.emit('session:resumed' as SwarmEvent, { sessionId: this.currentSessionId });
  }

  /**
   * Hibernate the current session
   */
  async hibernateSession(): Promise<void> {
    if (!this.currentSessionId) {
      throw new Error('No active session. Create or load a session first.');
    }

    // Save current state before hibernating
    await this.saveSession();

    await this.sessionManager.hibernateSession(this.currentSessionId);
    
    this.emit('session:hibernated' as SwarmEvent, { sessionId: this.currentSessionId });
    this.currentSessionId = undefined; // Session is no longer active
  }

  /**
   * Terminate the current session
   */
  async terminateSession(cleanup: boolean = false): Promise<void> {
    if (!this.currentSessionId) {
      throw new Error('No active session. Create or load a session first.');
    }

    const sessionId = this.currentSessionId;
    await this.sessionManager.terminateSession(sessionId, cleanup);
    
    this.emit('session:terminated' as SwarmEvent, { sessionId, cleanup });
    this.currentSessionId = undefined;
  }

  /**
   * List available sessions
   */
  async listSessions(filter?: any): Promise<SessionState[]> {
    if (!this.sessionIntegrationEnabled) {
      throw new Error('Session integration not enabled. Call init() first.');
    }

    return this.sessionManager.listSessions(filter);
  }

  /**
   * Get current session info
   */
  async getCurrentSession(): Promise<SessionState | null> {
    if (!this.currentSessionId) {
      return null;
    }

    return this.sessionManager.loadSession(this.currentSessionId);
  }

  /**
   * Get session statistics
   */
  async getSessionStats(sessionId?: string): Promise<Record<string, any>> {
    return this.sessionManager.getSessionStats(sessionId || this.currentSessionId);
  }

  /**
   * Enhanced agent operations with session persistence
   */
  override addAgent(config: AgentConfig): string {
    const agentId = super.addAgent(config);

    // Auto-save to session if enabled
    if (this.currentSessionId && this.sessionIntegrationEnabled) {
      setImmediate(() => this.saveSession().catch(error => {
        this.emit('session:error' as SwarmEvent, { 
          error: error.message, 
          operation: 'addAgent',
          agentId 
        });
      }));
    }

    return agentId;
  }

  /**
   * Enhanced task submission with session persistence
   */
  override async submitTask(task: Omit<Task, 'id' | 'status'>): Promise<string> {
    const taskId = await super.submitTask(task);

    // Auto-save to session if enabled
    if (this.currentSessionId && this.sessionIntegrationEnabled) {
      setImmediate(() => this.saveSession().catch(error => {
        this.emit('session:error' as SwarmEvent, { 
          error: error.message, 
          operation: 'submitTask',
          taskId 
        });
      }));
    }

    return taskId;
  }

  /**
   * Enhanced destroy with session cleanup
   */
  override async destroy(): Promise<void> {
    // Save session before destroying if there's an active session
    if (this.currentSessionId) {
      try {
        await this.saveSession();
        await this.createCheckpoint('Pre-destroy checkpoint');
      } catch (error) {
        console.error('Failed to save session before destroy:', error);
      }
    }

    // Shutdown session manager
    if (this.sessionManager) {
      await this.sessionManager.shutdown();
    }

    // Call parent destroy
    await super.destroy();
  }

  /**
   * Private helper methods
   */

  private async captureCurrentState(): Promise<SwarmState> {
    // Access the protected state from parent class
    // Note: In a real implementation, you might need to add a getter method to RuvSwarm
    return {
      agents: (this as any).state.agents,
      tasks: (this as any).state.tasks,
      topology: (this as any).state.topology,
      connections: (this as any).state.connections,
      metrics: (this as any).state.metrics,
    };
  }

  private async restoreFromSessionState(session: SessionState): Promise<void> {
    // Restore agents
    for (const [agentId, agent] of session.swarmState.agents) {
      if (!(this as any).state.agents.has(agentId)) {
        // Re-add agent if not present
        try {
          this.addAgent(agent.config);
        } catch (error) {
          console.warn(`Failed to restore agent ${agentId}:`, error);
        }
      }
    }

    // Restore tasks
    for (const [taskId, task] of session.swarmState.tasks) {
      if (!(this as any).state.tasks.has(taskId)) {
        // Re-add task if not present
        try {
          await this.submitTask({
            description: task.description,
            priority: task.priority,
            dependencies: task.dependencies,
            assignedAgents: task.assignedAgents,
          });
        } catch (error) {
          console.warn(`Failed to restore task ${taskId}:`, error);
        }
      }
    }

    // Update internal state
    (this as any).state.topology = session.swarmState.topology;
    (this as any).state.connections = session.swarmState.connections;
    (this as any).state.metrics = session.swarmState.metrics;

    this.emit('swarm:state_restored' as SwarmEvent, { 
      sessionId: session.id,
      agentCount: session.swarmState.agents.size,
      taskCount: session.swarmState.tasks.size 
    });
  }

  private setupEventForwarding(): void {
    // Forward session manager events to swarm events
    this.sessionManager.on('session:created', (data) => {
      this.emit('session:created' as SwarmEvent, data);
    });

    this.sessionManager.on('session:loaded', (data) => {
      this.emit('session:loaded' as SwarmEvent, data);
    });

    this.sessionManager.on('session:saved', (data) => {
      this.emit('session:saved' as SwarmEvent, data);
    });

    this.sessionManager.on('checkpoint:created', (data) => {
      this.emit('session:checkpoint_created' as SwarmEvent, data);
    });

    this.sessionManager.on('session:restored', (data) => {
      this.emit('session:restored' as SwarmEvent, data);
    });

    this.sessionManager.on('session:paused', (data) => {
      this.emit('session:paused' as SwarmEvent, data);
    });

    this.sessionManager.on('session:resumed', (data) => {
      this.emit('session:resumed' as SwarmEvent, data);
    });

    this.sessionManager.on('session:hibernated', (data) => {
      this.emit('session:hibernated' as SwarmEvent, data);
    });

    this.sessionManager.on('session:terminated', (data) => {
      this.emit('session:terminated' as SwarmEvent, data);
    });

    this.sessionManager.on('session:corruption_detected', (data) => {
      this.emit('session:corruption_detected' as SwarmEvent, data);
    });

    this.sessionManager.on('checkpoint:error', (data) => {
      this.emit('session:error' as SwarmEvent, { ...data, operation: 'checkpoint' });
    });
  }
}

/**
 * Session Recovery Service
 * 
 * Provides automated recovery capabilities for corrupted sessions
 */
export class SessionRecoveryService extends EventEmitter {
  private sessionManager: SessionManager;
  private recoveryInProgress: Set<string> = new Set();

  constructor(sessionManager: SessionManager) {
    super();
    this.sessionManager = sessionManager;
  }

  /**
   * Attempt to recover a corrupted session
   */
  async recoverSession(sessionId: string): Promise<boolean> {
    if (this.recoveryInProgress.has(sessionId)) {
      throw new Error(`Recovery already in progress for session ${sessionId}`);
    }

    this.recoveryInProgress.add(sessionId);
    this.emit('recovery:started', { sessionId });

    try {
      // Load the corrupted session
      const session = await this.sessionManager.loadSession(sessionId);
      
      // Validate session state
      const validation = SessionValidator.validateSessionState(session);
      if (validation.valid) {
        this.emit('recovery:not_needed', { sessionId });
        return true;
      }

      this.emit('recovery:validation_failed', { 
        sessionId, 
        errors: validation.errors 
      });

      // Attempt recovery using checkpoints
      const recoveredSession = await SessionRecovery.recoverSession(
        session,
        session.checkpoints
      );

      if (!recoveredSession) {
        this.emit('recovery:failed', { 
          sessionId, 
          reason: 'No valid checkpoints found' 
        });
        return false;
      }

      // Save recovered session
      await this.sessionManager.saveSession(sessionId, recoveredSession.swarmState);

      this.emit('recovery:completed', { 
        sessionId,
        recoveredFromCheckpoint: recoveredSession.metadata.recoveredFromCheckpoint 
      });

      return true;

    } catch (error) {
      this.emit('recovery:failed', { 
        sessionId, 
        reason: error instanceof Error ? error.message : String(error)
      });
      return false;
    } finally {
      this.recoveryInProgress.delete(sessionId);
    }
  }

  /**
   * Run health check on all sessions
   */
  async runHealthCheck(): Promise<Record<string, any>> {
    const sessions = await this.sessionManager.listSessions();
    const healthReport: Record<string, any> = {
      total: sessions.length,
      healthy: 0,
      corrupted: 0,
      needsRecovery: [],
      recoveryRecommendations: [],
    };

    for (const session of sessions) {
      const validation = SessionValidator.validateSessionState(session);
      
      if (validation.valid) {
        healthReport.healthy++;
      } else {
        healthReport.corrupted++;
        healthReport.needsRecovery.push({
          sessionId: session.id,
          name: session.name,
          errors: validation.errors,
        });

        // Generate recovery recommendation
        if (session.checkpoints.length > 0) {
          healthReport.recoveryRecommendations.push({
            sessionId: session.id,
            recommendation: 'automatic_recovery',
            availableCheckpoints: session.checkpoints.length,
          });
        } else {
          healthReport.recoveryRecommendations.push({
            sessionId: session.id,
            recommendation: 'manual_intervention',
            reason: 'No checkpoints available',
          });
        }
      }
    }

    this.emit('health_check:completed', healthReport);
    return healthReport;
  }

  /**
   * Schedule automatic recovery for corrupted sessions
   */
  async scheduleAutoRecovery(): Promise<void> {
    const healthReport = await this.runHealthCheck();
    
    const autoRecoverySessions = healthReport.recoveryRecommendations
      .filter((rec: any) => rec.recommendation === 'automatic_recovery')
      .map((rec: any) => rec.sessionId);

    this.emit('auto_recovery:scheduled', { 
      sessions: autoRecoverySessions,
      count: autoRecoverySessions.length 
    });

    for (const sessionId of autoRecoverySessions) {
      try {
        const success = await this.recoverSession(sessionId);
        this.emit('auto_recovery:session_completed', { 
          sessionId, 
          success 
        });
      } catch (error) {
        this.emit('auto_recovery:session_failed', { 
          sessionId, 
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    this.emit('auto_recovery:completed', { 
      totalSessions: autoRecoverySessions.length 
    });
  }
}

/**
 * Factory function for creating session-enabled swarms
 */
export function createSessionEnabledSwarm(
  swarmOptions?: SwarmOptions,
  sessionConfig?: SessionConfig,
  persistence?: SwarmPersistencePooled
): SessionEnabledSwarm {
  return new SessionEnabledSwarm(swarmOptions, sessionConfig, persistence);
}

// Components are already exported inline above