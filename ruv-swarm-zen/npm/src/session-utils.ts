/**
 * Session Management Utilities
 * 
 * Helper functions and utilities for session management,
 * including serialization, validation, migration, and recovery.
 */

import { SwarmState, SwarmOptions, AgentConfig, Task } from './types.js';
import { SessionState, SessionCheckpoint, SessionStatus } from './session-manager.js';
import crypto from 'crypto';

/**
 * Session validation utilities
 */
export class SessionValidator {
  /**
   * Validate session state integrity
   */
  static validateSessionState(state: SessionState): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Basic structure validation
    if (!state.id || typeof state.id !== 'string') {
      errors.push('Invalid session ID');
    }

    if (!state.name || typeof state.name !== 'string') {
      errors.push('Invalid session name');
    }

    if (!state.createdAt || !(state.createdAt instanceof Date)) {
      errors.push('Invalid created date');
    }

    if (!state.lastAccessedAt || !(state.lastAccessedAt instanceof Date)) {
      errors.push('Invalid last accessed date');
    }

    // Status validation
    const validStatuses: SessionStatus[] = ['active', 'paused', 'hibernated', 'terminated', 'corrupted'];
    if (!validStatuses.includes(state.status)) {
      errors.push(`Invalid session status: ${state.status}`);
    }

    // Swarm state validation
    if (!state.swarmState) {
      errors.push('Missing swarm state');
    } else {
      const swarmErrors = this.validateSwarmState(state.swarmState);
      errors.push(...swarmErrors);
    }

    // Swarm options validation
    if (!state.swarmOptions) {
      errors.push('Missing swarm options');
    } else {
      const optionsErrors = this.validateSwarmOptions(state.swarmOptions);
      errors.push(...optionsErrors);
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * Validate swarm state structure
   */
  static validateSwarmState(state: SwarmState): string[] {
    const errors: string[] = [];

    if (!state.agents || !(state.agents instanceof Map)) {
      errors.push('Invalid agents map');
    }

    if (!state.tasks || !(state.tasks instanceof Map)) {
      errors.push('Invalid tasks map');
    }

    if (!state.topology || typeof state.topology !== 'string') {
      errors.push('Invalid topology');
    }

    if (!Array.isArray(state.connections)) {
      errors.push('Invalid connections array');
    }

    if (!state.metrics || typeof state.metrics !== 'object') {
      errors.push('Invalid metrics object');
    }

    return errors;
  }

  /**
   * Validate swarm options
   */
  static validateSwarmOptions(options: SwarmOptions): string[] {
    const errors: string[] = [];

    if (options.maxAgents !== undefined && (typeof options.maxAgents !== 'number' || options.maxAgents <= 0)) {
      errors.push('Invalid maxAgents value');
    }

    if (options.connectionDensity !== undefined && 
        (typeof options.connectionDensity !== 'number' || options.connectionDensity < 0 || options.connectionDensity > 1)) {
      errors.push('Invalid connectionDensity value');
    }

    if (options.syncInterval !== undefined && (typeof options.syncInterval !== 'number' || options.syncInterval <= 0)) {
      errors.push('Invalid syncInterval value');
    }

    return errors;
  }

  /**
   * Validate checkpoint integrity
   */
  static validateCheckpoint(checkpoint: SessionCheckpoint): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!checkpoint.id || typeof checkpoint.id !== 'string') {
      errors.push('Invalid checkpoint ID');
    }

    if (!checkpoint.sessionId || typeof checkpoint.sessionId !== 'string') {
      errors.push('Invalid session ID');
    }

    if (!checkpoint.timestamp || !(checkpoint.timestamp instanceof Date)) {
      errors.push('Invalid timestamp');
    }

    if (!checkpoint.checksum || typeof checkpoint.checksum !== 'string') {
      errors.push('Invalid checksum');
    }

    if (!checkpoint.state) {
      errors.push('Missing checkpoint state');
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }
}

/**
 * Session serialization utilities
 */
export class SessionSerializer {
  /**
   * Serialize a SwarmState to a portable format
   */
  static serializeSwarmState(state: SwarmState): string {
    // Convert Maps to objects for JSON serialization
    const serializable = {
      agents: Object.fromEntries(state.agents.entries()),
      tasks: Object.fromEntries(state.tasks.entries()),
      topology: state.topology,
      connections: state.connections,
      metrics: {
        ...state.metrics,
        agentUtilization: Object.fromEntries(state.metrics.agentUtilization.entries()),
      },
    };

    return JSON.stringify(serializable, null, 2);
  }

  /**
   * Deserialize a SwarmState from portable format
   */
  static deserializeSwarmState(serialized: string): SwarmState {
    const data = JSON.parse(serialized);

    return {
      agents: new Map(Object.entries(data.agents)),
      tasks: new Map(Object.entries(data.tasks)),
      topology: data.topology,
      connections: data.connections,
      metrics: {
        ...data.metrics,
        agentUtilization: new Map(Object.entries(data.metrics.agentUtilization)),
      },
    };
  }

  /**
   * Export session to a portable format
   */
  static exportSession(session: SessionState): string {
    const exportData = {
      id: session.id,
      name: session.name,
      createdAt: session.createdAt.toISOString(),
      lastAccessedAt: session.lastAccessedAt.toISOString(),
      lastCheckpointAt: session.lastCheckpointAt?.toISOString(),
      status: session.status,
      swarmState: this.serializeSwarmState(session.swarmState),
      swarmOptions: session.swarmOptions,
      metadata: session.metadata,
      checkpoints: session.checkpoints.map(cp => ({
        id: cp.id,
        sessionId: cp.sessionId,
        timestamp: cp.timestamp.toISOString(),
        checksum: cp.checksum,
        state: this.serializeSwarmState(cp.state),
        description: cp.description,
        metadata: cp.metadata,
      })),
      version: session.version,
      exportedAt: new Date().toISOString(),
    };

    return JSON.stringify(exportData, null, 2);
  }

  /**
   * Import session from portable format
   */
  static importSession(exported: string): SessionState {
    const data = JSON.parse(exported);

    return {
      id: data.id,
      name: data.name,
      createdAt: new Date(data.createdAt),
      lastAccessedAt: new Date(data.lastAccessedAt),
      lastCheckpointAt: data.lastCheckpointAt ? new Date(data.lastCheckpointAt) : undefined,
      status: data.status,
      swarmState: this.deserializeSwarmState(data.swarmState),
      swarmOptions: data.swarmOptions,
      metadata: data.metadata,
      checkpoints: data.checkpoints.map((cp: any) => ({
        id: cp.id,
        sessionId: cp.sessionId,
        timestamp: new Date(cp.timestamp),
        checksum: cp.checksum,
        state: this.deserializeSwarmState(cp.state),
        description: cp.description,
        metadata: cp.metadata,
      })),
      version: data.version,
    };
  }
}

/**
 * Session migration utilities
 */
export class SessionMigrator {
  /**
   * Migrate session from older version
   */
  static migrateSession(session: any, fromVersion: string, toVersion: string): SessionState {
    const migrations = this.getMigrationPath(fromVersion, toVersion);
    
    let currentSession = session;
    for (const migration of migrations) {
      currentSession = migration(currentSession);
    }

    return currentSession;
  }

  /**
   * Get migration path between versions
   */
  private static getMigrationPath(fromVersion: string, toVersion: string): Array<(session: any) => any> {
    const migrations: Array<(session: any) => any> = [];

    // Example migrations (add as needed)
    if (fromVersion === '0.9.0' && toVersion === '1.0.0') {
      migrations.push(this.migrate_0_9_0_to_1_0_0);
    }

    return migrations;
  }

  /**
   * Example migration from 0.9.0 to 1.0.0
   */
  private static migrate_0_9_0_to_1_0_0(session: any): any {
    // Add version field if missing
    if (!session.version) {
      session.version = '1.0.0';
    }

    // Ensure metadata exists
    if (!session.metadata) {
      session.metadata = {};
    }

    // Migrate old checkpoint format
    if (session.checkpoints) {
      session.checkpoints = session.checkpoints.map((cp: any) => {
        if (!cp.metadata) {
          cp.metadata = {};
        }
        return cp;
      });
    }

    return session;
  }

  /**
   * Check if migration is needed
   */
  static needsMigration(session: any, targetVersion: string): boolean {
    return !session.version || session.version !== targetVersion;
  }
}

/**
 * Session recovery utilities
 */
export class SessionRecovery {
  /**
   * Attempt to recover a corrupted session
   */
  static async recoverSession(
    corruptedSession: any,
    checkpoints: SessionCheckpoint[]
  ): Promise<SessionState | null> {
    // Try to recover from the most recent valid checkpoint
    const sortedCheckpoints = checkpoints
      .filter(cp => this.validateCheckpointIntegrity(cp))
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    if (sortedCheckpoints.length === 0) {
      return null; // No valid checkpoints found
    }

    const latestCheckpoint = sortedCheckpoints[0];
    
    try {
      // Reconstruct session from checkpoint
      const recoveredSession: SessionState = {
        id: corruptedSession.id || latestCheckpoint.sessionId,
        name: corruptedSession.name || 'Recovered Session',
        createdAt: corruptedSession.createdAt || new Date(),
        lastAccessedAt: new Date(),
        lastCheckpointAt: latestCheckpoint.timestamp,
        status: 'active',
        swarmState: latestCheckpoint.state,
        swarmOptions: corruptedSession.swarmOptions || this.getDefaultSwarmOptions(),
        metadata: {
          ...corruptedSession.metadata,
          recovered: true,
          recoveredAt: new Date().toISOString(),
          recoveredFromCheckpoint: latestCheckpoint.id,
        },
        checkpoints: sortedCheckpoints,
        version: corruptedSession.version || '1.0.0',
      };

      return recoveredSession;
    } catch (error) {
      console.error('Failed to recover session:', error);
      return null;
    }
  }

  /**
   * Validate checkpoint integrity
   */
  private static validateCheckpointIntegrity(checkpoint: SessionCheckpoint): boolean {
    try {
      // Check basic structure
      if (!checkpoint.id || !checkpoint.sessionId || !checkpoint.state) {
        return false;
      }

      // Validate checksum if available
      if (checkpoint.checksum) {
        const stateData = JSON.stringify(checkpoint.state);
        const calculatedChecksum = crypto.createHash('sha256').update(stateData).digest('hex');
        return calculatedChecksum === checkpoint.checksum;
      }

      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get default swarm options for recovery
   */
  private static getDefaultSwarmOptions(): SwarmOptions {
    return {
      topology: 'mesh',
      maxAgents: 10,
      connectionDensity: 0.5,
      syncInterval: 1000,
    };
  }

  /**
   * Repair session state inconsistencies
   */
  static repairSessionState(session: SessionState): SessionState {
    const repairedSession = { ...session };

    // Ensure Maps are properly initialized
    if (!(repairedSession.swarmState.agents instanceof Map)) {
      repairedSession.swarmState.agents = new Map();
    }

    if (!(repairedSession.swarmState.tasks instanceof Map)) {
      repairedSession.swarmState.tasks = new Map();
    }

    if (!(repairedSession.swarmState.metrics.agentUtilization instanceof Map)) {
      repairedSession.swarmState.metrics.agentUtilization = new Map();
    }

    // Ensure arrays are properly initialized
    if (!Array.isArray(repairedSession.swarmState.connections)) {
      repairedSession.swarmState.connections = [];
    }

    if (!Array.isArray(repairedSession.checkpoints)) {
      repairedSession.checkpoints = [];
    }

    // Ensure required fields have default values
    if (!repairedSession.metadata) {
      repairedSession.metadata = {};
    }

    if (!repairedSession.version) {
      repairedSession.version = '1.0.0';
    }

    // Repair metrics if corrupted
    if (!repairedSession.swarmState.metrics) {
      repairedSession.swarmState.metrics = {
        totalTasks: 0,
        completedTasks: 0,
        failedTasks: 0,
        averageCompletionTime: 0,
        agentUtilization: new Map(),
        throughput: 0,
      };
    }

    return repairedSession;
  }
}

/**
 * Session statistics utilities
 */
export class SessionStats {
  /**
   * Calculate session health score
   */
  static calculateHealthScore(session: SessionState): number {
    let score = 100;

    // Deduct points for various issues
    if (session.status === 'corrupted') score -= 50;
    if (session.status === 'terminated') score -= 30;
    if (session.checkpoints.length === 0) score -= 20;

    // Age factor (older sessions might be less healthy)
    const ageInDays = (Date.now() - session.createdAt.getTime()) / (1000 * 60 * 60 * 24);
    if (ageInDays > 30) score -= 10;
    if (ageInDays > 90) score -= 20;

    // Access pattern (unused sessions might be less healthy)
    const daysSinceAccess = (Date.now() - session.lastAccessedAt.getTime()) / (1000 * 60 * 60 * 24);
    if (daysSinceAccess > 7) score -= 10;
    if (daysSinceAccess > 30) score -= 20;

    // Task success rate
    const metrics = session.swarmState.metrics;
    if (metrics.totalTasks > 0) {
      const successRate = metrics.completedTasks / metrics.totalTasks;
      if (successRate < 0.5) score -= 20;
      if (successRate < 0.2) score -= 30;
    }

    return Math.max(0, Math.min(100, score));
  }

  /**
   * Generate session summary
   */
  static generateSummary(session: SessionState): Record<string, any> {
    const metrics = session.swarmState.metrics;
    const healthScore = this.calculateHealthScore(session);

    return {
      id: session.id,
      name: session.name,
      status: session.status,
      healthScore,
      createdAt: session.createdAt,
      lastAccessedAt: session.lastAccessedAt,
      lastCheckpointAt: session.lastCheckpointAt,
      ageInDays: Math.floor((Date.now() - session.createdAt.getTime()) / (1000 * 60 * 60 * 24)),
      daysSinceAccess: Math.floor((Date.now() - session.lastAccessedAt.getTime()) / (1000 * 60 * 60 * 24)),
      agents: {
        total: session.swarmState.agents.size,
        topology: session.swarmState.topology,
      },
      tasks: {
        total: metrics.totalTasks,
        completed: metrics.completedTasks,
        failed: metrics.failedTasks,
        successRate: metrics.totalTasks > 0 ? metrics.completedTasks / metrics.totalTasks : 0,
        averageCompletionTime: metrics.averageCompletionTime,
      },
      checkpoints: {
        total: session.checkpoints.length,
        latest: session.checkpoints[0]?.timestamp,
      },
      performance: {
        throughput: metrics.throughput,
        agentUtilization: Array.from(metrics.agentUtilization.entries()).reduce((acc, [id, util]) => {
          acc[id] = util;
          return acc;
        }, {} as Record<string, number>),
      },
      version: session.version,
    };
  }
}

/**
 * Export all utilities
 */
// Classes are already exported inline above