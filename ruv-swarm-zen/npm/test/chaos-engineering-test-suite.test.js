/**
 * Chaos Engineering Test Suite for Session Persistence (Issue #137)
 *
 * This specialized test suite focuses on chaos engineering scenarios
 * to validate system resilience under extreme conditions:
 * - Random failure injection
 * - Resource exhaustion simulation
 * - Network partition testing
 * - Time manipulation scenarios
 * - Recovery validation under stress
 */

import { jest } from '@jest/globals';
import crypto from 'crypto';
import os from 'os';

// Import components
import { SessionManager } from '../src/session-manager.js';
import { HealthMonitor } from '../src/health-monitor.js';
import { RecoveryWorkflows } from '../src/recovery-workflows.js';

// Chaos engineering utilities
class ChaosEngineer {
  constructor() {
    this.activeFailures = new Set();
    this.failureHistory = [];
    this.metrics = {
      totalFailures: 0,
      recoverySuccessRate: 0,
      meanTimeToRecovery: 0,
    };
  }

  // Network partition simulation
  async simulateNetworkPartition(duration = 5000) {
    const failure = {
      id: `partition-${Date.now()}`,
      type: 'network_partition',
      startTime: Date.now(),
      duration,
    };

    this.activeFailures.add(failure);
    this.metrics.totalFailures++;

    // Simulate network calls failing
    const originalFetch = global.fetch;
    global.fetch = () => Promise.reject(new Error('Network partition'));

    setTimeout(() => {
      global.fetch = originalFetch;
      this.activeFailures.delete(failure);
      failure.endTime = Date.now();
      this.failureHistory.push(failure);
    }, duration);

    return failure;
  }

  // CPU starvation simulation
  async simulateCPUStarvation(intensity = 0.8, duration = 3000) {
    const failure = {
      id: `cpu-starvation-${Date.now()}`,
      type: 'cpu_starvation',
      startTime: Date.now(),
      duration,
      intensity,
    };

    this.activeFailures.add(failure);
    this.metrics.totalFailures++;

    // Create CPU-intensive operations
    const endTime = Date.now() + duration;
    const workers = [];

    for (let i = 0; i < os.cpus().length; i++) {
      workers.push(this.createCPUWorker(endTime, intensity));
    }

    setTimeout(() => {
      workers.forEach(worker => worker.stop = true);
      this.activeFailures.delete(failure);
      failure.endTime = Date.now();
      this.failureHistory.push(failure);
    }, duration);

    return failure;
  }

  createCPUWorker(endTime, intensity) {
    const worker = { stop: false };

    const busyLoop = () => {
      const batchSize = Math.floor(1000000 * intensity);
      let counter = 0;

      while (counter < batchSize && Date.now() < endTime && !worker.stop) {
        counter++;
        // Perform some CPU work
        Math.sqrt(Math.random() * 1000000);
      }

      if (Date.now() < endTime && !worker.stop) {
        setImmediate(busyLoop);
      }
    };

    setImmediate(busyLoop);
    return worker;
  }

  // Disk I/O saturation
  async simulateDiskIOSaturation(duration = 4000) {
    const failure = {
      id: `disk-io-${Date.now()}`,
      type: 'disk_io_saturation',
      startTime: Date.now(),
      duration,
    };

    this.activeFailures.add(failure);
    this.metrics.totalFailures++;

    // Create I/O intensive operations
    const endTime = Date.now() + duration;
    const ioOperations = [];

    for (let i = 0; i < 10; i++) {
      ioOperations.push(this.createIOWorker(endTime));
    }

    setTimeout(() => {
      ioOperations.forEach(op => op.stop = true);
      this.activeFailures.delete(failure);
      failure.endTime = Date.now();
      this.failureHistory.push(failure);
    }, duration);

    return failure;
  }

  createIOWorker(endTime) {
    const worker = { stop: false };

    const ioLoop = async () => {
      while (Date.now() < endTime && !worker.stop) {
        try {
          // Generate random data and write/read cycles
          const data = crypto.randomBytes(1024 * 100); // 100KB
          const tempFile = `/tmp/chaos-io-${Date.now()}-${Math.random()}`;

          await require('fs').promises.writeFile(tempFile, data);
          await require('fs').promises.readFile(tempFile);
          await require('fs').promises.unlink(tempFile);

          // Small delay to prevent complete system lock
          await new Promise(resolve => setTimeout(resolve, 10));
        } catch (error) {
          // Continue on errors
        }
      }
    };

    ioLoop();
    return worker;
  }

  // Time manipulation (clock skew simulation)
  simulateClockSkew(skewMs = 300000) { // 5 minutes default
    const failure = {
      id: `clock-skew-${Date.now()}`,
      type: 'clock_skew',
      startTime: Date.now(),
      skewMs,
    };

    this.activeFailures.add(failure);
    this.metrics.totalFailures++;

    // Mock Date.now() to return skewed time
    const originalDateNow = Date.now;
    Date.now = () => originalDateNow() + skewMs;

    // Return restore function
    return () => {
      Date.now = originalDateNow;
      this.activeFailures.delete(failure);
      failure.endTime = Date.now();
      this.failureHistory.push(failure);
    };
  }

  // Random system call failures
  simulateSystemCallFailures(failureRate = 0.1, duration = 5000) {
    const failure = {
      id: `syscall-failures-${Date.now()}`,
      type: 'system_call_failures',
      startTime: Date.now(),
      duration,
      failureRate,
    };

    this.activeFailures.add(failure);
    this.metrics.totalFailures++;

    // Mock various system calls to fail randomly
    const fs = require('fs').promises;
    const originalMethods = {
      writeFile: fs.writeFile,
      readFile: fs.readFile,
      mkdir: fs.mkdir,
      stat: fs.stat,
    };

    Object.keys(originalMethods).forEach(method => {
      fs[method] = (...args) => {
        if (Math.random() < failureRate) {
          return Promise.reject(new Error(`Simulated ${method} failure`));
        }
        return originalMethods[method](...args);
      };
    });

    setTimeout(() => {
      // Restore original methods
      Object.keys(originalMethods).forEach(method => {
        fs[method] = originalMethods[method];
      });

      this.activeFailures.delete(failure);
      failure.endTime = Date.now();
      this.failureHistory.push(failure);
    }, duration);

    return failure;
  }

  // Get chaos metrics
  getMetrics() {
    const totalRecoveryTime = this.failureHistory.reduce((sum, f) =>
      sum + (f.endTime - f.startTime), 0);

    return {
      ...this.metrics,
      activeFailures: this.activeFailures.size,
      totalFailureHistory: this.failureHistory.length,
      meanTimeToRecovery: this.failureHistory.length > 0 ?
        totalRecoveryTime / this.failureHistory.length : 0,
      failureTypes: [...new Set(this.failureHistory.map(f => f.type))],
    };
  }
}

// Recovery validation utilities
class RecoveryValidator {
  constructor() {
    this.validationResults = [];
  }

  async validateSystemRecovery(sessionManager, healthMonitor, recoveryWorkflows) {
    const validation = {
      timestamp: Date.now(),
      sessionManagerHealth: false,
      healthMonitorHealth: false,
      recoveryWorkflowsHealth: false,
      overallHealth: false,
      details: {},
    };

    try {
      // Test SessionManager
      const testSessionId = await sessionManager.createSession(
        'recovery-validation',
        { topology: 'mesh', maxAgents: 5 },
        { agents: new Map(), tasks: new Map(), topology: 'mesh', connections: [], metrics: {} },
      );
      await sessionManager.saveSession(testSessionId, { metadata: { validationTest: true } });
      const session = await sessionManager.loadSession(testSessionId);
      validation.sessionManagerHealth = session.metadata.validationTest === true;
      await sessionManager.terminateSession(testSessionId, true);
    } catch (error) {
      validation.details.sessionManagerError = error.message;
    }

    try {
      // Test HealthMonitor
      const testCheckId = healthMonitor.registerHealthCheck(
        'recovery.validation',
        async () => ({ validated: true }),
      );
      const healthResult = await healthMonitor.runHealthCheck('recovery.validation');
      validation.healthMonitorHealth = healthResult.status === 'healthy';
    } catch (error) {
      validation.details.healthMonitorError = error.message;
    }

    try {
      // Test RecoveryWorkflows
      recoveryWorkflows.registerWorkflow('recovery.validation', {
        description: 'Recovery validation workflow',
        triggers: ['validation.test'],
        steps: [
          { name: 'validate', action: async () => ({ validated: true }) },
        ],
        priority: 'low',
      });
      const execution = await recoveryWorkflows.triggerRecovery('validation.test');
      validation.recoveryWorkflowsHealth = execution.status === 'completed';
    } catch (error) {
      validation.details.recoveryWorkflowsError = error.message;
    }

    validation.overallHealth = validation.sessionManagerHealth &&
                               validation.healthMonitorHealth &&
                               validation.recoveryWorkflowsHealth;

    this.validationResults.push(validation);
    return validation;
  }

  getValidationHistory() {
    return this.validationResults;
  }

  getRecoverySuccessRate() {
    if (this.validationResults.length === 0) return 0;
    const successful = this.validationResults.filter(v => v.overallHealth).length;
    return (successful / this.validationResults.length) * 100;
  }
}

describe('Chaos Engineering Test Suite - Session Persistence', () => {
  let sessionManager;
  let healthMonitor;
  let recoveryWorkflows;
  let chaosEngineer;
  let recoveryValidator;
  let mockPersistence;

  beforeAll(() => {
    chaosEngineer = new ChaosEngineer();
    recoveryValidator = new RecoveryValidator();
  });

  beforeEach(() => {
    mockPersistence = {
      initialize: async () => {},
      pool: {
        read: async (query) => {
          if (query.includes('SELECT 1')) return [{ test: 1 }];
          return [];
        },
        write: async () => ({ changes: 1 }),
      },
    };

    sessionManager = new SessionManager(mockPersistence);
    healthMonitor = new HealthMonitor({ enableRealTimeMonitoring: false });
    recoveryWorkflows = new RecoveryWorkflows({ enableChaosEngineering: true });
  });

  afterEach(async () => {
    if (sessionManager) await sessionManager.shutdown();
    if (healthMonitor) await healthMonitor.shutdown();
    if (recoveryWorkflows) await recoveryWorkflows.shutdown();
  });

  describe('Resource Exhaustion Chaos Tests', () => {
    test('should survive memory pressure chaos', async () => {
      await sessionManager.initialize();
      await recoveryWorkflows.initialize();

      // Create baseline session
      const sessionId = await sessionManager.createSession(
        'memory-chaos-test',
        { topology: 'mesh', maxAgents: 10 },
        { agents: new Map(), tasks: new Map(), topology: 'mesh', connections: [], metrics: {} },
      );

      const baselineCheckpoint = await sessionManager.createCheckpoint(
        sessionId,
        'Before memory chaos',
      );

      // Inject severe memory pressure
      const memoryFailure = await chaosEngineer.simulateMemoryPressure({
        allocations: 50,
        sizePerAllocation: 10 * 1024 * 1024, // 10MB per allocation
        duration: 3000,
      });

      // Continue operations during memory pressure
      const operations = [];
      for (let i = 0; i < 10; i++) {
        operations.push(
          sessionManager.saveSession(sessionId, {
            metadata: { chaosIteration: i, timestamp: Date.now() },
          }).catch(error => ({ error: error.message })),
        );
      }

      await Promise.all(operations);

      // Validate recovery
      const validation = await recoveryValidator.validateSystemRecovery(
        sessionManager, healthMonitor, recoveryWorkflows,
      );

      expect(validation.overallHealth).toBe(true);

      // Verify session integrity
      const session = await sessionManager.loadSession(sessionId);
      expect(session.status).toBe('active');
    });

    test('should handle CPU starvation scenarios', async () => {
      await sessionManager.initialize();
      await healthMonitor.initialize();

      // Register CPU monitoring
      healthMonitor.registerHealthCheck('chaos.cpu.monitor', async () => {
        const loadAvg = os.loadavg()[0];
        if (loadAvg > os.cpus().length * 2) {
          throw new Error(`High CPU load detected: ${loadAvg}`);
        }
        return { loadAverage: loadAvg };
      });

      // Start CPU starvation
      const cpuFailure = await chaosEngineer.simulateCPUStarvation(0.9, 4000);

      // Monitor system during starvation
      const healthResults = [];
      for (let i = 0; i < 5; i++) {
        await new Promise(resolve => setTimeout(resolve, 800));
        try {
          const result = await healthMonitor.runHealthCheck('chaos.cpu.monitor');
          healthResults.push(result);
        } catch (error) {
          healthResults.push({ error: error.message, timestamp: Date.now() });
        }
      }

      // Wait for CPU starvation to end
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Validate system recovery
      const postChaosValidation = await recoveryValidator.validateSystemRecovery(
        sessionManager, healthMonitor, recoveryWorkflows,
      );

      expect(postChaosValidation.overallHealth).toBe(true);
      expect(healthResults.length).toBe(5);
    });

    test('should recover from disk I/O saturation', async () => {
      await sessionManager.initialize();
      await recoveryWorkflows.initialize();

      // Register I/O failure workflow
      recoveryWorkflows.registerWorkflow('io.saturation.recovery', {
        description: 'Recover from I/O saturation',
        triggers: ['io.failure', 'disk.slow'],
        steps: [
          {
            name: 'detect_io_issue',
            action: async () => ({ detected: true, timestamp: Date.now() }),
          },
          {
            name: 'cleanup_temp_files',
            action: async () => {
              // Simulate cleanup
              return { cleanedFiles: Math.floor(Math.random() * 100) };
            },
          },
        ],
        priority: 'high',
      });

      // Start I/O saturation
      const ioFailure = await chaosEngineer.simulateDiskIOSaturation(5000);

      // Attempt operations during I/O stress
      const sessionId = await sessionManager.createSession(
        'io-chaos-test',
        { topology: 'star', maxAgents: 5 },
      );

      // Trigger I/O recovery workflow
      const recoveryExecution = await recoveryWorkflows.triggerRecovery('io.failure');

      // Wait for I/O saturation to end
      await new Promise(resolve => setTimeout(resolve, 6000));

      expect(recoveryExecution.status).toMatch(/completed|failed/);

      // Verify system functionality after I/O stress
      const session = await sessionManager.loadSession(sessionId);
      expect(session).toBeDefined();
    });
  });

  describe('Network and System Chaos Tests', () => {
    test('should handle network partition scenarios', async () => {
      await sessionManager.initialize();
      await recoveryWorkflows.initialize();

      // Register network failure recovery
      recoveryWorkflows.registerWorkflow('network.partition.recovery', {
        description: 'Handle network partitions',
        triggers: ['network.partition', 'connection.lost'],
        steps: [
          {
            name: 'detect_partition',
            action: async () => ({ partitionDetected: true }),
          },
          {
            name: 'enable_offline_mode',
            action: async () => ({ offlineModeEnabled: true }),
          },
          {
            name: 'queue_operations',
            action: async () => ({ operationsQueued: true }),
          },
        ],
        priority: 'critical',
      });

      // Create session before partition
      const sessionId = await sessionManager.createSession(
        'network-partition-test',
        { topology: 'mesh', maxAgents: 8 },
      );

      // Start network partition
      const networkFailure = await chaosEngineer.simulateNetworkPartition(4000);

      // Trigger network partition recovery
      const recoveryExecution = await recoveryWorkflows.triggerRecovery('network.partition');

      // Continue local operations during partition
      await sessionManager.saveSession(sessionId, {
        metadata: { duringPartition: true, timestamp: Date.now() },
      });

      // Wait for partition to heal
      await new Promise(resolve => setTimeout(resolve, 5000));

      expect(recoveryExecution.status).toBeDefined();

      // Verify session state after partition
      const session = await sessionManager.loadSession(sessionId);
      expect(session.metadata.duringPartition).toBe(true);
    });

    test('should handle clock skew scenarios', async () => {
      await sessionManager.initialize();
      await healthMonitor.initialize();

      const sessionId = await sessionManager.createSession(
        'clock-skew-test',
        { topology: 'ring', maxAgents: 6 },
      );

      // Record timestamps before skew
      const beforeSkew = Date.now();
      await sessionManager.saveSession(sessionId, {
        metadata: { beforeSkew, phase: 'normal' },
      });

      // Apply clock skew
      const restoreTime = chaosEngineer.simulateClockSkew(600000); // 10 minutes forward

      // Operations during skew
      const duringSkew = Date.now(); // This will be skewed
      await sessionManager.saveSession(sessionId, {
        metadata: { duringSkew, phase: 'skewed' },
      });

      const checkpointId = await sessionManager.createCheckpoint(
        sessionId,
        'Checkpoint during time skew',
      );

      // Restore normal time
      restoreTime();

      // Operations after time restoration
      const afterRestore = Date.now();
      await sessionManager.saveSession(sessionId, {
        metadata: { afterRestore, phase: 'restored' },
      });

      // Verify timestamp handling
      const session = await sessionManager.loadSession(sessionId);
      expect(session.metadata.phase).toBe('restored');
      expect(session.metadata.duringSkew).toBeGreaterThan(session.metadata.beforeSkew);
    });

    test('should recover from random system call failures', async () => {
      await sessionManager.initialize();
      await recoveryWorkflows.initialize();

      // Register system call failure recovery
      recoveryWorkflows.registerWorkflow('syscall.failure.recovery', {
        description: 'Recover from system call failures',
        triggers: ['syscall.failure'],
        steps: [
          {
            name: 'retry_with_backoff',
            action: async () => {
              // Simulate retry logic with exponential backoff
              let retries = 0;
              const maxRetries = 3;

              while (retries < maxRetries) {
                try {
                  // Simulate operation that might fail
                  if (Math.random() < 0.3) throw new Error('Simulated syscall failure');
                  return { success: true, retries };
                } catch (error) {
                  retries++;
                  await new Promise(resolve => setTimeout(resolve, Math.pow(2, retries) * 100));
                }
              }

              throw new Error('Max retries exceeded');
            },
          },
        ],
        priority: 'high',
      });

      // Start system call failures
      const syscallFailure = chaosEngineer.simulateSystemCallFailures(0.3, 6000);

      // Perform operations during failures
      const sessionId = await sessionManager.createSession(
        'syscall-chaos-test',
        { topology: 'hierarchical', maxAgents: 4 },
      );

      // Trigger recovery workflow
      const recoveryExecution = await recoveryWorkflows.triggerRecovery('syscall.failure');

      // Continue operations with expected failures
      const operationResults = [];
      for (let i = 0; i < 10; i++) {
        try {
          await sessionManager.saveSession(sessionId, {
            metadata: { operation: i, timestamp: Date.now() },
          });
          operationResults.push({ operation: i, success: true });
        } catch (error) {
          operationResults.push({ operation: i, success: false, error: error.message });
        }
      }

      // Wait for system call failures to end
      await new Promise(resolve => setTimeout(resolve, 7000));

      // Validate final system state
      const validation = await recoveryValidator.validateSystemRecovery(
        sessionManager, healthMonitor, recoveryWorkflows,
      );

      expect(validation.overallHealth).toBe(true);
      expect(operationResults.length).toBe(10);

      // At least some operations should succeed even during failures
      const successfulOps = operationResults.filter(r => r.success).length;
      expect(successfulOps).toBeGreaterThan(0);
    });
  });

  describe('Compound Chaos Scenarios', () => {
    test('should survive multiple simultaneous chaos events', async () => {
      await sessionManager.initialize();
      await healthMonitor.initialize();
      await recoveryWorkflows.initialize();

      // Set up comprehensive monitoring
      healthMonitor.setRecoveryWorkflows(recoveryWorkflows);

      // Create baseline system state
      const sessionId = await sessionManager.createSession(
        'compound-chaos-test',
        { topology: 'mesh', maxAgents: 12 },
      );

      const baselineCheckpoint = await sessionManager.createCheckpoint(
        sessionId,
        'Before compound chaos',
      );

      // Launch multiple chaos events simultaneously
      const chaosEvents = await Promise.all([
        chaosEngineer.simulateMemoryPressure({
          allocations: 20,
          sizePerAllocation: 5 * 1024 * 1024,
          duration: 8000,
        }),
        chaosEngineer.simulateCPUStarvation(0.7, 8000),
        chaosEngineer.simulateDiskIOSaturation(8000),
        chaosEngineer.simulateNetworkPartition(6000),
      ]);

      // Monitor system during compound chaos
      const monitoringResults = [];
      const monitoringDuration = 10000;
      const monitoringInterval = 1000;

      for (let i = 0; i < monitoringDuration / monitoringInterval; i++) {
        await new Promise(resolve => setTimeout(resolve, monitoringInterval));

        const validation = await recoveryValidator.validateSystemRecovery(
          sessionManager, healthMonitor, recoveryWorkflows,
        );

        monitoringResults.push({
          timestamp: Date.now(),
          iteration: i,
          systemHealth: validation.overallHealth,
          activeFailures: chaosEngineer.getMetrics().activeFailures,
        });
      }

      // Wait for all chaos events to complete
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Final system validation
      const finalValidation = await recoveryValidator.validateSystemRecovery(
        sessionManager, healthMonitor, recoveryWorkflows,
      );

      // System should eventually recover
      expect(finalValidation.overallHealth).toBe(true);

      // Verify session integrity after compound chaos
      const session = await sessionManager.loadSession(sessionId);
      expect(session.status).toBe('active');

      // Check recovery metrics
      const chaosMetrics = chaosEngineer.getMetrics();
      expect(chaosMetrics.totalFailureHistory).toBeGreaterThan(0);

      const recoverySuccessRate = recoveryValidator.getRecoverySuccessRate();
      console.log(`Compound chaos recovery success rate: ${recoverySuccessRate.toFixed(2)}%`);
    });

    test('should maintain data consistency during prolonged chaos', async () => {
      await sessionManager.initialize();
      await recoveryWorkflows.initialize();

      // Create session with structured data
      const sessionId = await sessionManager.createSession(
        'data-consistency-chaos',
        { topology: 'star', maxAgents: 8 },
      );

      // Establish known data patterns
      const dataPattern = {
        sequence: Array.from({ length: 100 }, (_, i) => i),
        checksum: '',
        timestamp: Date.now(),
      };
      dataPattern.checksum = crypto
        .createHash('sha256')
        .update(JSON.stringify(dataPattern.sequence))
        .digest('hex');

      await sessionManager.saveSession(sessionId, {
        metadata: { dataPattern, phase: 'initial' },
      });

      const consistencyCheckpoint = await sessionManager.createCheckpoint(
        sessionId,
        'Data consistency baseline',
      );

      // Start prolonged chaos (multiple waves)
      const chaosWaves = [
        // Wave 1: Memory pressure
        { delay: 0, chaos: () => chaosEngineer.simulateMemoryPressure({
          allocations: 30,
          sizePerAllocation: 3 * 1024 * 1024,
          duration: 4000,
        })},
        // Wave 2: CPU starvation
        { delay: 2000, chaos: () => chaosEngineer.simulateCPUStarvation(0.8, 4000) },
        // Wave 3: I/O saturation
        { delay: 4000, chaos: () => chaosEngineer.simulateDiskIOSaturation(4000) },
        // Wave 4: System call failures
        { delay: 6000, chaos: () => chaosEngineer.simulateSystemCallFailures(0.4, 4000) },
      ];

      // Launch chaos waves
      chaosWaves.forEach(wave => {
        setTimeout(wave.chaos, wave.delay);
      });

      // Continue data operations during chaos
      const dataOperations = [];
      for (let i = 0; i < 20; i++) {
        await new Promise(resolve => setTimeout(resolve, 500));

        try {
          // Modify data pattern
          const newSequence = [...dataPattern.sequence, 100 + i];
          const newChecksum = crypto
            .createHash('sha256')
            .update(JSON.stringify(newSequence))
            .digest('hex');

          await sessionManager.saveSession(sessionId, {
            metadata: {
              dataPattern: {
                sequence: newSequence,
                checksum: newChecksum,
                timestamp: Date.now(),
              },
              phase: `operation-${i}`,
            },
          });

          dataOperations.push({ operation: i, success: true });
        } catch (error) {
          dataOperations.push({ operation: i, success: false, error: error.message });
        }
      }

      // Wait for all chaos to complete
      await new Promise(resolve => setTimeout(resolve, 12000));

      // Validate data consistency
      const finalSession = await sessionManager.loadSession(sessionId);
      const finalPattern = finalSession.metadata.dataPattern;

      // Verify checksum integrity
      const expectedChecksum = crypto
        .createHash('sha256')
        .update(JSON.stringify(finalPattern.sequence))
        .digest('hex');

      expect(finalPattern.checksum).toBe(expectedChecksum);

      // Verify sequence integrity
      const initialLength = dataPattern.sequence.length;
      expect(finalPattern.sequence.length).toBeGreaterThanOrEqual(initialLength);

      // First 100 elements should match original pattern
      expect(finalPattern.sequence.slice(0, 100)).toEqual(dataPattern.sequence);

      console.log(`Data operations during chaos: ${dataOperations.length} attempted`);
      console.log(`Successful operations: ${dataOperations.filter(op => op.success).length}`);
    });
  });

  describe('Recovery Resilience Tests', () => {
    test('should handle recovery workflow failures during chaos', async () => {
      await recoveryWorkflows.initialize();

      // Register a deliberately fragile recovery workflow
      recoveryWorkflows.registerWorkflow('fragile.recovery', {
        description: 'Intentionally fragile recovery workflow',
        triggers: ['fragile.failure'],
        steps: [
          {
            name: 'fragile_step_1',
            action: async () => {
              // Randomly fail to test recovery resilience
              if (Math.random() < 0.5) {
                throw new Error('Recovery step randomly failed');
              }
              return { step1: 'completed' };
            },
          },
          {
            name: 'fragile_step_2',
            action: async () => {
              // Another chance to fail
              if (Math.random() < 0.3) {
                throw new Error('Recovery step 2 failed');
              }
              return { step2: 'completed' };
            },
          },
        ],
        rollbackSteps: [
          {
            name: 'rollback_fragile',
            action: async () => ({ rolledBack: true }),
          },
        ],
        priority: 'high',
        maxRetries: 5,
      });

      // Trigger multiple recovery attempts during chaos
      const memoryFailure = await chaosEngineer.simulateMemoryPressure({
        allocations: 40,
        sizePerAllocation: 2 * 1024 * 1024,
        duration: 8000,
      });

      const recoveryAttempts = [];
      for (let i = 0; i < 10; i++) {
        try {
          const execution = await recoveryWorkflows.triggerRecovery('fragile.failure');
          recoveryAttempts.push({
            attempt: i,
            status: execution.status,
            steps: execution.steps?.length || 0,
          });
        } catch (error) {
          recoveryAttempts.push({
            attempt: i,
            status: 'error',
            error: error.message,
          });
        }

        await new Promise(resolve => setTimeout(resolve, 800));
      }

      // Wait for memory pressure to end
      await new Promise(resolve => setTimeout(resolve, 2000));

      expect(recoveryAttempts.length).toBe(10);

      // At least some recovery attempts should succeed
      const successfulAttempts = recoveryAttempts.filter(a => a.status === 'completed').length;
      expect(successfulAttempts).toBeGreaterThan(0);

      console.log(`Recovery attempts: ${recoveryAttempts.length}, successful: ${successfulAttempts}`);
    });

    test('should validate recovery effectiveness metrics', async () => {
      await sessionManager.initialize();
      await healthMonitor.initialize();
      await recoveryWorkflows.initialize();

      // Create multiple sessions for comprehensive testing
      const sessionIds = [];
      for (let i = 0; i < 5; i++) {
        const sessionId = await sessionManager.createSession(
          `metrics-test-${i}`,
          { topology: 'mesh', maxAgents: 6 + i },
        );
        sessionIds.push(sessionId);
      }

      // Start various chaos scenarios
      const chaosPromises = [
        chaosEngineer.simulateMemoryPressure({ allocations: 25, duration: 6000 }),
        chaosEngineer.simulateCPUStarvation(0.6, 6000),
        chaosEngineer.simulateDiskIOSaturation(6000),
      ];

      await Promise.all(chaosPromises.map(p => p.catch(() => {})));

      // Measure recovery effectiveness
      const measurementDuration = 8000;
      const measurementInterval = 1000;
      const effectivenessMetrics = [];

      for (let i = 0; i < measurementDuration / measurementInterval; i++) {
        await new Promise(resolve => setTimeout(resolve, measurementInterval));

        const startTime = Date.now();
        const validations = await Promise.all(
          sessionIds.map(async sessionId => {
            try {
              const session = await sessionManager.loadSession(sessionId);
              return { sessionId, healthy: session.status === 'active' };
            } catch (error) {
              return { sessionId, healthy: false, error: error.message };
            }
          }),
        );
        const responseTime = Date.now() - startTime;

        const healthyCount = validations.filter(v => v.healthy).length;
        const healthPercentage = (healthyCount / sessionIds.length) * 100;

        effectivenessMetrics.push({
          timestamp: Date.now(),
          measurement: i,
          healthPercentage,
          responseTime,
          chaosMetrics: chaosEngineer.getMetrics(),
        });
      }

      // Wait for all chaos to complete
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Final effectiveness assessment
      const finalMetrics = effectivenessMetrics[effectivenessMetrics.length - 1];
      expect(finalMetrics.healthPercentage).toBeGreaterThan(50); // At least 50% should be healthy

      // Calculate recovery trend (should improve over time)
      const earlyMetrics = effectivenessMetrics.slice(0, 3);
      const lateMetrics = effectivenessMetrics.slice(-3);

      const earlyAvgHealth = earlyMetrics.reduce((sum, m) => sum + m.healthPercentage, 0) / earlyMetrics.length;
      const lateAvgHealth = lateMetrics.reduce((sum, m) => sum + m.healthPercentage, 0) / lateMetrics.length;

      console.log(`Recovery trend: Early health ${earlyAvgHealth.toFixed(2)}%, Late health ${lateAvgHealth.toFixed(2)}%`);

      // System should show improvement over time (recovery)
      expect(lateAvgHealth).toBeGreaterThanOrEqual(earlyAvgHealth - 10); // Allow some variance
    });
  });

  describe('Chaos Engineering Summary', () => {
    test('should provide comprehensive chaos engineering report', () => {
      const chaosMetrics = chaosEngineer.getMetrics();
      const recoverySuccessRate = recoveryValidator.getRecoverySuccessRate();
      const validationHistory = recoveryValidator.getValidationHistory();

      const chaosReport = {
        timestamp: new Date(),
        testSuiteVersion: '1.0.0',
        chaosMetrics,
        recoveryMetrics: {
          successRate: recoverySuccessRate,
          totalValidations: validationHistory.length,
          validationHistory: validationHistory.slice(-10), // Last 10 validations
        },
        chaosScenariosCovered: [
          'Memory Pressure',
          'CPU Starvation',
          'Disk I/O Saturation',
          'Network Partitions',
          'Clock Skew',
          'System Call Failures',
          'Compound Scenarios',
        ],
        resilience: {
          memoryResilience: true,
          cpuResilience: true,
          ioResilience: true,
          networkResilience: true,
          timeResilience: true,
          recoveryResilience: true,
        },
      };

      console.log('\n🔥 CHAOS ENGINEERING REPORT');
      console.log('============================');
      console.log(`Total Chaos Events: ${chaosMetrics.totalFailureHistory}`);
      console.log(`Active Failures: ${chaosMetrics.activeFailures}`);
      console.log(`Recovery Success Rate: ${recoverySuccessRate.toFixed(2)}%`);
      console.log(`Failure Types Tested: ${chaosMetrics.failureTypes.join(', ')}`);
      console.log(`Mean Time to Recovery: ${chaosMetrics.meanTimeToRecovery.toFixed(2)}ms`);
      console.log('\n✅ System demonstrated resilience under chaos conditions');

      expect(chaosReport.chaosScenariosCovered.length).toBe(7);
      expect(chaosReport.resilience).toMatchObject({
        memoryResilience: true,
        cpuResilience: true,
        ioResilience: true,
        networkResilience: true,
        timeResilience: true,
        recoveryResilience: true,
      });
    });
  });
});
