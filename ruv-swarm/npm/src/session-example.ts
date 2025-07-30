/**
 * Session Management System - Comprehensive Examples
 * 
 * This file demonstrates how to use the session management system
 * for persistent swarm orchestration across multiple executions.
 */

import { 
  SessionEnabledSwarm, 
  SessionRecoveryService
} from './session-integration.js';
import { SessionManager, SessionState } from './session-manager.js';
import { SwarmPersistencePooled } from './persistence-pooled.js';
import { SessionStats, SessionValidator } from './session-utils.js';

/**
 * Example 1: Basic Session Usage
 */
async function basicSessionExample() {
  console.log('=== Basic Session Example ===');

  // Create a session-enabled swarm
  const swarm = new SessionEnabledSwarm({
    topology: 'hierarchical',
    maxAgents: 8,
    connectionDensity: 0.7,
    syncInterval: 2000,
  }, {
    autoCheckpoint: true,
    checkpointInterval: 60000, // 1 minute
    maxCheckpoints: 10,
  });

  try {
    // Initialize the swarm
    await swarm.init();

    // Create a new session
    const sessionId = await swarm.createSession('ML Training Pipeline');
    console.log(`Created session: ${sessionId}`);

    // Add agents with different capabilities
    const dataAgent = swarm.addAgent({
      id: 'data-processor',
      type: 'analyst',
      capabilities: ['data-preprocessing', 'feature-extraction'],
    });

    const modelAgent = swarm.addAgent({
      id: 'model-trainer',
      type: 'researcher',
      capabilities: ['neural-networks', 'optimization'],
    });

    const evaluatorAgent = swarm.addAgent({
      id: 'model-evaluator',
      type: 'tester',
      capabilities: ['validation', 'metrics-calculation'],
    });

    console.log(`Added agents: ${dataAgent}, ${modelAgent}, ${evaluatorAgent}`);

    // Submit tasks
    const dataTask = await swarm.submitTask({
      description: 'Preprocess training dataset',
      priority: 'high',
      assignedAgents: [dataAgent],
    });

    const trainingTask = await swarm.submitTask({
      description: 'Train neural network model',
      priority: 'high',
      dependencies: [dataTask],
      assignedAgents: [modelAgent],
    });

    const evaluationTask = await swarm.submitTask({
      description: 'Evaluate model performance',
      priority: 'medium',
      dependencies: [trainingTask],
      assignedAgents: [evaluatorAgent],
    });

    console.log(`Submitted tasks: ${dataTask}, ${trainingTask}, ${evaluationTask}`);

    // Create a manual checkpoint
    const checkpointId = await swarm.createCheckpoint('Initial pipeline setup');
    console.log(`Created checkpoint: ${checkpointId}`);

    // Get session statistics
    const stats = await swarm.getSessionStats();
    console.log('Session Statistics:', {
      sessionId: stats.sessionId,
      name: stats.name,
      totalAgents: stats.totalAgents,
      totalTasks: stats.totalTasks,
      checkpointCount: stats.checkpointCount,
    });

    // Save the session
    await swarm.saveSession();
    console.log('Session saved successfully');

    return sessionId;

  } finally {
    await swarm.destroy();
  }
}

/**
 * Example 2: Session Recovery and Restoration
 */
async function sessionRecoveryExample(existingSessionId: string) {
  console.log('=== Session Recovery Example ===');

  const swarm = new SessionEnabledSwarm({
    topology: 'mesh',
    maxAgents: 10,
  });

  try {
    await swarm.init();

    // Load the existing session
    console.log(`Loading session: ${existingSessionId}`);
    await swarm.loadSession(existingSessionId);

    const currentSession = await swarm.getCurrentSession();
    if (currentSession) {
      console.log(`Loaded session: ${currentSession.name}`);
      console.log(`Agents: ${currentSession.swarmState.agents.size}`);
      console.log(`Tasks: ${currentSession.swarmState.tasks.size}`);
      console.log(`Checkpoints: ${currentSession.checkpoints.length}`);

      // Add more work to the restored session
      const optimizerAgent = swarm.addAgent({
        id: 'hyperparameter-optimizer',
        type: 'optimizer',
        capabilities: ['grid-search', 'bayesian-optimization'],
      });

      const optimizationTask = await swarm.submitTask({
        description: 'Optimize hyperparameters',
        priority: 'medium',
        assignedAgents: [optimizerAgent],
      });

      console.log(`Added optimizer agent: ${optimizerAgent}`);
      console.log(`Added optimization task: ${optimizationTask}`);

      // Create another checkpoint
      await swarm.createCheckpoint('Added hyperparameter optimization');

      // Demonstrate checkpoint restoration
      if (currentSession.checkpoints.length > 0) {
        const firstCheckpoint = currentSession.checkpoints[0];
        console.log(`Restoring from checkpoint: ${firstCheckpoint.id}`);
        
        await swarm.restoreFromCheckpoint(firstCheckpoint.id);
        console.log('Successfully restored from checkpoint');
      }
    }

  } finally {
    await swarm.destroy();
  }
}

/**
 * Example 3: Session Lifecycle Management
 */
async function sessionLifecycleExample() {
  console.log('=== Session Lifecycle Example ===');

  const persistence = new SwarmPersistencePooled();
  await persistence.initialize();

  const sessionManager = new SessionManager(persistence, {
    autoCheckpoint: true,
    checkpointInterval: 30000, // 30 seconds
    maxCheckpoints: 5,
    compressionEnabled: true,
    encryptionEnabled: false,
  });

  try {
    await sessionManager.initialize();

    // Create a session
    const sessionId = await sessionManager.createSession(
      'Long Running Analysis',
      { topology: 'distributed', maxAgents: 15 }
    );

    console.log(`Created session: ${sessionId}`);

    // Simulate some work
    await sessionManager.saveSession(sessionId, {
      agents: new Map([
        ['analyst-1', { id: 'analyst-1', type: 'analyst' } as any],
        ['researcher-1', { id: 'researcher-1', type: 'researcher' } as any],
      ]),
      tasks: new Map([
        ['task-1', { id: 'task-1', description: 'Analyze data' } as any],
      ]),
      topology: 'distributed',
      connections: [],
      metrics: {
        totalTasks: 1,
        completedTasks: 0,
        failedTasks: 0,
        averageCompletionTime: 0,
        agentUtilization: new Map(),
        throughput: 0,
      },
    });

    // Create checkpoints
    const checkpoint1 = await sessionManager.createCheckpoint(sessionId, 'Work started');
    console.log(`Created checkpoint: ${checkpoint1}`);

    // Pause the session
    await sessionManager.pauseSession(sessionId);
    console.log('Session paused');

    // Resume the session
    await sessionManager.resumeSession(sessionId);
    console.log('Session resumed');

    // Get session statistics
    const stats = await sessionManager.getSessionStats(sessionId);
    console.log('Detailed Statistics:', stats);

    // List all sessions
    const allSessions = await sessionManager.listSessions();
    console.log(`Total sessions: ${allSessions.length}`);

    // Hibernate the session
    await sessionManager.hibernateSession(sessionId);
    console.log('Session hibernated');

    // Load hibernated session
    const hibernatedSession = await sessionManager.loadSession(sessionId);
    console.log(`Hibernated session status: ${hibernatedSession.status}`);

    return sessionId;

  } finally {
    await sessionManager.shutdown();
    await persistence.close();
  }
}

/**
 * Example 4: Session Health Monitoring and Recovery
 */
async function sessionHealthExample() {
  console.log('=== Session Health and Recovery Example ===');

  const persistence = new SwarmPersistencePooled();
  await persistence.initialize();

  const sessionManager = new SessionManager(persistence);
  await sessionManager.initialize();

  const recoveryService = new SessionRecoveryService(sessionManager);

  try {
    // Create some test sessions
    const session1 = await sessionManager.createSession('Healthy Session', { topology: 'mesh' });
    const session2 = await sessionManager.createSession('Test Session', { topology: 'hierarchical' });

    // Add some checkpoints
    await sessionManager.createCheckpoint(session1, 'Healthy checkpoint');
    await sessionManager.createCheckpoint(session2, 'Test checkpoint');

    // Run health check
    console.log('Running health check...');
    const healthReport = await recoveryService.runHealthCheck();
    
    console.log('Health Report:');
    console.log(`- Total sessions: ${healthReport.total}`);
    console.log(`- Healthy sessions: ${healthReport.healthy}`);
    console.log(`- Corrupted sessions: ${healthReport.corrupted}`);
    console.log(`- Sessions needing recovery: ${healthReport.needsRecovery.length}`);

    // Get detailed statistics for each session
    for (const sessionId of [session1, session2]) {
      const session = await sessionManager.loadSession(sessionId);
      const healthScore = SessionStats.calculateHealthScore(session);
      const summary = SessionStats.generateSummary(session);
      
      console.log(`\nSession ${sessionId}:`);
      console.log(`- Health Score: ${healthScore}/100`);
      console.log(`- Age: ${summary.ageInDays} days`);
      console.log(`- Last Access: ${summary.daysSinceAccess} days ago`);
      console.log(`- Status: ${summary.status}`);
    }

    // Schedule automatic recovery if needed
    if (healthReport.needsRecovery.length > 0) {
      console.log('\nScheduling automatic recovery...');
      await recoveryService.scheduleAutoRecovery();
    }

  } finally {
    await sessionManager.shutdown();
    await persistence.close();
  }
}

/**
 * Example 5: Advanced Session Operations
 */
async function advancedSessionExample() {
  console.log('=== Advanced Session Operations Example ===');

  const swarm = new SessionEnabledSwarm({
    topology: 'hybrid',
    maxAgents: 20,
    connectionDensity: 0.8,
  }, {
    autoCheckpoint: true,
    checkpointInterval: 45000,
    maxCheckpoints: 8,
    compressionEnabled: true,
  });

  try {
    await swarm.init();

    // Create session with event monitoring
    swarm.on('session:created', (data: any) => {
      console.log(`Session created event: ${data.sessionId}`);
    });

    swarm.on('session:checkpoint_created', (data: any) => {
      console.log(`Checkpoint created: ${data.checkpointId} - ${data.description}`);
    });

    swarm.on('session:error', (data: any) => {
      console.error(`Session error: ${data.error} during ${data.operation}`);
    });

    const sessionId = await swarm.createSession('Advanced ML Pipeline');

    // Create multiple specialized agents
    const agents = [
      { id: 'data-collector', type: 'researcher', capabilities: ['web-scraping', 'api-integration'] },
      { id: 'data-cleaner', type: 'analyst', capabilities: ['data-validation', 'outlier-detection'] },
      { id: 'feature-engineer', type: 'architect', capabilities: ['feature-selection', 'dimensionality-reduction'] },
      { id: 'model-builder', type: 'coder', capabilities: ['deep-learning', 'ensemble-methods'] },
      { id: 'model-validator', type: 'reviewer', capabilities: ['cross-validation', 'performance-metrics'] },
      { id: 'deployment-manager', type: 'architect', capabilities: ['containerization', 'monitoring'] },
    ];

    const agentIds = agents.map(agent => swarm.addAgent(agent));
    console.log(`Created ${agentIds.length} specialized agents`);

    // Create complex task dependencies
    const tasks = [
      {
        description: 'Collect training data from multiple sources',
        priority: 'critical' as const,
        assignedAgents: [agentIds[0]],
        dependencies: [] as string[],
      },
      {
        description: 'Clean and validate collected data',
        priority: 'high' as const,
        assignedAgents: [agentIds[1]],
        dependencies: [] as string[],
      },
      {
        description: 'Engineer features from cleaned data',
        priority: 'high' as const,
        assignedAgents: [agentIds[2]],
        dependencies: [] as string[],
      },
      {
        description: 'Build and train multiple models',
        priority: 'high' as const,
        assignedAgents: [agentIds[3]],
        dependencies: [] as string[],
      },
      {
        description: 'Validate model performance',
        priority: 'medium' as const,
        assignedAgents: [agentIds[4]],
        dependencies: [] as string[],
      },
      {
        description: 'Deploy best performing model',
        priority: 'medium' as const,
        assignedAgents: [agentIds[5]],
        dependencies: [] as string[],
      },
    ];

    const taskIds = [];
    for (let i = 0; i < tasks.length; i++) {
      const task = tasks[i];
      if (i > 0) {
        task.dependencies = [taskIds[i - 1]]; // Sequential dependencies
      }
      const taskId = await swarm.submitTask(task);
      taskIds.push(taskId);
    }

    console.log(`Created ${taskIds.length} interconnected tasks`);

    // Create milestone checkpoints
    await swarm.createCheckpoint('Pipeline setup complete');

    // Simulate some progress
    await new Promise(resolve => setTimeout(resolve, 2000));

    await swarm.createCheckpoint('Data collection phase');

    // Get comprehensive session information
    const session = await swarm.getCurrentSession();
    if (session) {
      console.log('\nSession Overview:');
      console.log(`- ID: ${session.id}`);
      console.log(`- Name: ${session.name}`);
      console.log(`- Status: ${session.status}`);
      console.log(`- Agents: ${session.swarmState.agents.size}`);
      console.log(`- Tasks: ${session.swarmState.tasks.size}`);
      console.log(`- Checkpoints: ${session.checkpoints.length}`);
      console.log(`- Topology: ${session.swarmState.topology}`);
      console.log(`- Version: ${session.version}`);

      // Validate session integrity
      const validation = SessionValidator.validateSessionState(session);
      console.log(`- Integrity: ${validation.valid ? 'Valid' : 'Invalid'}`);
      if (!validation.valid) {
        console.log(`- Errors: ${validation.errors.join(', ')}`);
      }

      // Calculate health metrics
      const healthScore = SessionStats.calculateHealthScore(session);
      console.log(`- Health Score: ${healthScore}/100`);

      const summary = SessionStats.generateSummary(session);
      console.log(`- Success Rate: ${(summary.tasks.successRate * 100).toFixed(1)}%`);
      console.log(`- Throughput: ${summary.performance.throughput} tasks/sec`);
    }

    // Demonstrate session export/import capability
    if (session) {
      console.log('\nTesting session export/import...');
      const exportedData = await swarm.exportSession();
      console.log(`Exported session data: ${exportedData.length} characters`);
      
      // In a real scenario, you would save this to a file
      // and import it in another application instance
    }

  } finally {
    await swarm.destroy();
  }
}

/**
 * Main execution function
 */
async function runAllExamples() {
  console.log('🚀 Starting Session Management System Examples\n');

  try {
    // Run basic session example
    const sessionId = await basicSessionExample();
    console.log('\n' + '='.repeat(50) + '\n');

    // Run recovery example using the created session
    await sessionRecoveryExample(sessionId);
    console.log('\n' + '='.repeat(50) + '\n');

    // Run lifecycle management example
    const lifecycleSessionId = await sessionLifecycleExample();
    console.log('\n' + '='.repeat(50) + '\n');

    // Run health monitoring example
    await sessionHealthExample();
    console.log('\n' + '='.repeat(50) + '\n');

    // Run advanced operations example
    await advancedSessionExample();

    console.log('\n✅ All examples completed successfully!');

  } catch (error) {
    console.error('❌ Error running examples:', error);
    process.exit(1);
  }
}

/**
 * Extension method for SessionEnabledSwarm to export session data
 */
declare module './session-integration.js' {
  interface SessionEnabledSwarm {
    exportSession(): Promise<string>;
  }
}

// Add the export method to the prototype (for demonstration)
if (typeof module !== 'undefined' && module.exports) {
  const { SessionEnabledSwarm } = require('./session-integration');
  const { SessionSerializer } = require('./session-utils');
  
  SessionEnabledSwarm.prototype.exportSession = async function(): Promise<string> {
    const session = await this.getCurrentSession();
    if (!session) {
      throw new Error('No active session to export');
    }
    return SessionSerializer.exportSession(session);
  };
}

// Run examples if this file is executed directly
if (require.main === module) {
  runAllExamples().catch(console.error);
}

export {
  basicSessionExample,
  sessionRecoveryExample,
  sessionLifecycleExample,
  sessionHealthExample,
  advancedSessionExample,
  runAllExamples,
};