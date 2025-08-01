/**
 * Comprehensive Test Suite for Recovery Workflows System
 *
 * Tests all recovery system components including:
 * - Health monitoring and alerting
 * - Recovery workflow execution
 * - Connection state management
 * - Monitoring dashboard integration
 * - Chaos engineering validation
 * - End-to-end recovery scenarios
 */

import { jest } from '@jest/globals';
import HealthMonitor from '../src/health-monitor.js';
import RecoveryWorkflows from '../src/recovery-workflows.js';
import ConnectionStateManager from '../src/connection-state-manager.js';
import MonitoringDashboard from '../src/monitoring-dashboard.js';
import ChaosEngineering from '../src/chaos-engineering.js';
import RecoveryIntegration from '../src/recovery-integration.js';

// Mock external dependencies
jest.mock('../src/logger.js', () => ({
  Logger: jest.fn().mockImplementation(() => ({
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  })),
}));

jest.mock('../src/utils.js', () => ({
  generateId: jest.fn().mockImplementation((prefix) => `${prefix}-${Date.now()}-${Math.random()}`),
}));

// Mock process methods for testing
const originalExit = process.exit;
const originalMemoryUsage = process.memoryUsage;
const originalCpuUsage = process.cpuUsage;

beforeAll(() => {
  process.exit = jest.fn();
  process.memoryUsage = jest.fn().mockReturnValue({
    rss: 50000000,
    heapTotal: 30000000,
    heapUsed: 20000000,
    external: 5000000,
  });
  process.cpuUsage = jest.fn().mockReturnValue({
    user: 1000000,
    system: 500000,
  });
});

afterAll(() => {
  process.exit = originalExit;
  process.memoryUsage = originalMemoryUsage;
  process.cpuUsage = originalCpuUsage;
});

describe('Health Monitor', () => {
  let healthMonitor;

  beforeEach(async () => {
    healthMonitor = new HealthMonitor({
      checkInterval: 1000, // Fast for testing
      systemCheckInterval: 2000,
      enableRealTimeMonitoring: false, // Disable for testing
    });
  });

  afterEach(async () => {
    if (healthMonitor) {
      await healthMonitor.shutdown();
    }
  });

  test('should initialize with built-in health checks', async () => {
    expect(healthMonitor.healthChecks.size).toBeGreaterThan(0);
    expect(healthMonitor.healthChecks.has('system.memory')).toBe(true);
    expect(healthMonitor.healthChecks.has('system.cpu')).toBe(true);
    expect(healthMonitor.healthChecks.has('system.eventloop')).toBe(true);
  });

  test('should register custom health check', () => {
    const checkFunction = jest.fn().mockResolvedValue({ status: 'ok' });

    const checkId = healthMonitor.registerHealthCheck('custom.test', checkFunction, {
      category: 'test',
      priority: 'low',
    });

    expect(checkId).toBeDefined();
    expect(healthMonitor.healthChecks.has('custom.test')).toBe(true);

    const healthCheck = healthMonitor.healthChecks.get('custom.test');
    expect(healthCheck.category).toBe('test');
    expect(healthCheck.priority).toBe('low');
  });

  test('should run health check successfully', async () => {
    const checkFunction = jest.fn().mockResolvedValue({ value: 42 });
    healthMonitor.registerHealthCheck('test.success', checkFunction);

    const result = await healthMonitor.runHealthCheck('test.success');

    expect(result.status).toBe('healthy');
    expect(result.result.value).toBe(42);
    expect(result.duration).toBeGreaterThan(0);
    expect(checkFunction).toHaveBeenCalled();
  });

  test('should handle health check failure', async () => {
    const checkFunction = jest.fn().mockRejectedValue(new Error('Test failure'));
    healthMonitor.registerHealthCheck('test.failure', checkFunction);

    const result = await healthMonitor.runHealthCheck('test.failure');

    expect(result.status).toBe('unhealthy');
    expect(result.error).toBe('Test failure');
    expect(result.failureCount).toBe(1);
  });

  test('should run all health checks', async () => {
    const check1 = jest.fn().mockResolvedValue({ check: 1 });
    const check2 = jest.fn().mockResolvedValue({ check: 2 });

    healthMonitor.registerHealthCheck('test.1', check1);
    healthMonitor.registerHealthCheck('test.2', check2);

    const { results, summary } = await healthMonitor.runAllHealthChecks();

    expect(results.length).toBeGreaterThanOrEqual(2);
    expect(summary.total).toBeGreaterThanOrEqual(2);
    expect(check1).toHaveBeenCalled();
    expect(check2).toHaveBeenCalled();
  });

  test('should get health status', () => {
    const status = healthMonitor.getHealthStatus();

    expect(status).toHaveProperty('total');
    expect(status).toHaveProperty('healthy');
    expect(status).toHaveProperty('unhealthy');
    expect(status).toHaveProperty('overallStatus');
  });

  test('should export health data', () => {
    const data = healthMonitor.exportHealthData();

    expect(data).toHaveProperty('timestamp');
    expect(data).toHaveProperty('stats');
    expect(data).toHaveProperty('healthChecks');
    expect(data).toHaveProperty('alerts');
  });
});

describe('Recovery Workflows', () => {
  let recoveryWorkflows;

  beforeEach(async () => {
    recoveryWorkflows = new RecoveryWorkflows({
      maxRetries: 2,
      retryDelay: 100, // Fast for testing
      maxConcurrentRecoveries: 2,
    });
  });

  afterEach(async () => {
    if (recoveryWorkflows) {
      await recoveryWorkflows.shutdown();
    }
  });

  test('should initialize with built-in workflows', async () => {
    expect(recoveryWorkflows.workflows.size).toBeGreaterThan(0);
    expect(recoveryWorkflows.workflows.has('swarm_init_failure')).toBe(true);
    expect(recoveryWorkflows.workflows.has('agent_failure')).toBe(true);
    expect(recoveryWorkflows.workflows.has('memory_pressure')).toBe(true);
  });

  test('should register custom workflow', () => {
    const workflowId = recoveryWorkflows.registerWorkflow('test.workflow', {
      description: 'Test workflow',
      failureType: 'test',
      steps: [
        {
          name: 'test_step',
          action: async () => ({ result: 'success' }),
        },
      ],
      priority: 'normal',
    });

    expect(workflowId).toBeDefined();
    expect(recoveryWorkflows.workflows.has('test.workflow')).toBe(true);
  });

  test('should execute workflow successfully', async () => {
    // Register a simple test workflow
    recoveryWorkflows.registerWorkflow('test.simple', {
      description: 'Simple test workflow',
      triggers: ['test.trigger'],
      steps: [
        {
          name: 'step1',
          action: async () => ({ step: 1, result: 'ok' }),
        },
        {
          name: 'step2',
          action: async () => ({ step: 2, result: 'ok' }),
        },
      ],
    });

    const execution = await recoveryWorkflows.triggerRecovery('test.trigger');

    expect(execution.status).toBe('completed');
    expect(execution.steps).toHaveLength(2);
    expect(execution.steps[0].status).toBe('completed');
    expect(execution.steps[1].status).toBe('completed');
  });

  test('should handle workflow step failure', async () => {
    recoveryWorkflows.registerWorkflow('test.failure', {
      description: 'Failure test workflow',
      triggers: ['test.failure'],
      steps: [
        {
          name: 'step1',
          action: async () => ({ result: 'ok' }),
        },
        {
          name: 'failing_step',
          action: async () => {
            throw new Error('Step failed');
          },
        },
      ],
      rollbackSteps: [
        {
          name: 'rollback',
          action: async () => ({ rollback: 'completed' }),
        },
      ],
    });

    const execution = await recoveryWorkflows.triggerRecovery('test.failure');

    expect(execution.status).toBe('failed');
    expect(execution.error).toContain('Step failed');
    expect(execution.rollbackSteps).toBeDefined();
  });

  test('should get recovery status', () => {
    const status = recoveryWorkflows.getRecoveryStatus();
    expect(Array.isArray(status)).toBe(true);
  });

  test('should export recovery data', () => {
    const data = recoveryWorkflows.exportRecoveryData();

    expect(data).toHaveProperty('timestamp');
    expect(data).toHaveProperty('stats');
    expect(data).toHaveProperty('workflows');
    expect(data).toHaveProperty('activeRecoveries');
  });
});

describe('Connection State Manager', () => {
  let connectionManager;

  beforeEach(async () => {
    connectionManager = new ConnectionStateManager({
      maxConnections: 5,
      connectionTimeout: 1000, // Fast for testing
      persistenceEnabled: false, // Disable for testing
    });
  });

  afterEach(async () => {
    if (connectionManager) {
      await connectionManager.shutdown();
    }
  });

  test('should register connection configuration', async () => {
    const connectionId = await connectionManager.registerConnection({
      type: 'stdio',
      command: 'echo',
      args: ['test'],
      autoReconnect: false,
    });

    expect(connectionId).toBeDefined();
    expect(connectionManager.connections.has(connectionId)).toBe(true);
  });

  test('should get connection status', () => {
    const status = connectionManager.getConnectionStatus();

    expect(status).toHaveProperty('connections');
    expect(status).toHaveProperty('stats');
    expect(status).toHaveProperty('summary');
  });

  test('should get connection statistics', () => {
    const stats = connectionManager.getConnectionStats();

    expect(stats).toHaveProperty('connectionCount');
    expect(stats).toHaveProperty('activeConnections');
    expect(stats).toHaveProperty('healthyConnections');
  });

  test('should export connection data', () => {
    const data = connectionManager.exportConnectionData();

    expect(data).toHaveProperty('timestamp');
    expect(data).toHaveProperty('stats');
    expect(data).toHaveProperty('connections');
  });
});

describe('Monitoring Dashboard', () => {
  let dashboard;

  beforeEach(async () => {
    dashboard = new MonitoringDashboard({
      aggregationInterval: 100, // Fast for testing
      enableRealTimeStreaming: false, // Disable for testing
    });
  });

  afterEach(async () => {
    if (dashboard) {
      await dashboard.shutdown();
    }
  });

  test('should initialize dashboard', async () => {
    expect(dashboard.metrics).toBeDefined();
    expect(dashboard.aggregatedMetrics).toBeDefined();
    expect(dashboard.alerts).toBeDefined();
  });

  test('should record health metric', () => {
    const healthResult = {
      name: 'test.check',
      status: 'healthy',
      duration: 100,
      metadata: { category: 'test', priority: 'normal' },
    };

    dashboard.recordHealthMetric(healthResult);

    expect(dashboard.healthStatus.has('test.check')).toBe(true);
    const status = dashboard.healthStatus.get('test.check');
    expect(status.status).toBe('healthy');
  });

  test('should record alert', () => {
    const alert = {
      id: 'alert-123',
      name: 'test.alert',
      severity: 'warning',
      healthCheck: { category: 'test', priority: 'normal' },
      acknowledged: false,
    };

    dashboard.recordAlert(alert);

    expect(dashboard.alerts.has('alert-123')).toBe(true);
  });

  test('should export dashboard data', () => {
    const data = dashboard.exportDashboardData();

    expect(data).toHaveProperty('timestamp');
    expect(data).toHaveProperty('summary');
    expect(data).toHaveProperty('health');
    expect(data).toHaveProperty('recovery');
    expect(data).toHaveProperty('connections');
  });

  test('should format data for Prometheus', () => {
    const data = dashboard.exportDashboardData();
    const prometheusData = dashboard.formatForPrometheus(data);

    expect(typeof prometheusData).toBe('string');
    expect(prometheusData).toContain('# HELP');
    expect(prometheusData).toContain('# TYPE');
  });
});

describe('Chaos Engineering', () => {
  let chaosEngineering;

  beforeEach(async () => {
    chaosEngineering = new ChaosEngineering({
      enableChaos: true,
      safetyEnabled: false, // Disable for testing
      maxConcurrentExperiments: 2,
    });
  });

  afterEach(async () => {
    if (chaosEngineering) {
      await chaosEngineering.shutdown();
    }
  });

  test('should initialize with built-in experiments', async () => {
    expect(chaosEngineering.experiments.size).toBeGreaterThan(0);
    expect(chaosEngineering.experiments.has('memory_pressure_recovery')).toBe(true);
    expect(chaosEngineering.failureInjectors.size).toBeGreaterThan(0);
  });

  test('should register custom experiment', () => {
    const experimentId = chaosEngineering.registerExperiment('test.experiment', {
      description: 'Test experiment',
      failureType: 'memory_pressure',
      duration: 1000,
      blastRadius: 0.1,
    });

    expect(experimentId).toBeDefined();
    expect(chaosEngineering.experiments.has('test.experiment')).toBe(true);
  });

  test('should validate blast radius limit', () => {
    expect(() => {
      chaosEngineering.registerExperiment('test.large', {
        description: 'Large blast radius test',
        failureType: 'memory_pressure',
        blastRadius: 0.8, // Exceeds default limit
      });
    }).toThrow();
  });

  test('should get experiment status', () => {
    const status = chaosEngineering.getExperimentStatus();
    expect(Array.isArray(status)).toBe(true);
  });

  test('should get chaos statistics', () => {
    const stats = chaosEngineering.getChaosStats();

    expect(stats).toHaveProperty('totalExperiments');
    expect(stats).toHaveProperty('activeExperiments');
    expect(stats).toHaveProperty('registeredExperiments');
    expect(stats).toHaveProperty('failureInjectors');
  });

  test('should export chaos data', () => {
    const data = chaosEngineering.exportChaosData();

    expect(data).toHaveProperty('timestamp');
    expect(data).toHaveProperty('stats');
    expect(data).toHaveProperty('experiments');
    expect(data).toHaveProperty('failureInjectors');
  });
});

describe('Recovery Integration', () => {
  let integration;

  beforeEach(async () => {
    integration = new RecoveryIntegration({
      enableHealthMonitoring: true,
      enableRecoveryWorkflows: true,
      enableConnectionManagement: true,
      enableMonitoringDashboard: true,
      enableChaosEngineering: false, // Disable for testing
      autoIntegrate: true,
      configValidation: false, // Disable for testing
    });
  });

  afterEach(async () => {
    if (integration) {
      await integration.shutdown();
    }
  });

  test('should initialize all enabled components', async () => {
    expect(integration.healthMonitor).toBeDefined();
    expect(integration.recoveryWorkflows).toBeDefined();
    expect(integration.connectionManager).toBeDefined();
    expect(integration.monitoringDashboard).toBeDefined();
    expect(integration.chaosEngineering).toBeNull(); // Disabled
  });

  test('should set up component integrations', async () => {
    expect(integration.integrationStatus.size).toBeGreaterThan(0);

    // Check for successful integrations
    let successfulIntegrations = 0;
    for (const [key, status] of integration.integrationStatus) {
      if (status.status === 'success') {
        successfulIntegrations++;
      }
    }

    expect(successfulIntegrations).toBeGreaterThan(0);
  });

  test('should get system status', () => {
    const status = integration.getSystemStatus();

    expect(status).toHaveProperty('isInitialized');
    expect(status).toHaveProperty('components');
    expect(status).toHaveProperty('integrations');
    expect(status).toHaveProperty('performance');
    expect(status.isInitialized).toBe(true);
  });

  test('should get performance metrics', () => {
    const metrics = integration.getPerformanceMetrics();

    expect(metrics).toHaveProperty('initializationTime');
    expect(metrics).toHaveProperty('componentStartupTimes');
    expect(metrics).toHaveProperty('integrationLatency');
    expect(metrics).toHaveProperty('currentMemoryUsage');
  });

  test('should start and stop system', async () => {
    await integration.start();
    expect(integration.isRunning).toBe(true);

    await integration.stop();
    expect(integration.isRunning).toBe(false);
  });

  test('should run system health check', async () => {
    const healthResults = await integration.runSystemHealthCheck();

    expect(healthResults).toHaveProperty('overall');
    expect(healthResults).toHaveProperty('components');
    expect(healthResults).toHaveProperty('issues');
    expect(['healthy', 'degraded', 'error']).toContain(healthResults.overall);
  });

  test('should export comprehensive system data', () => {
    const data = integration.exportSystemData();

    expect(data).toHaveProperty('timestamp');
    expect(data).toHaveProperty('status');
    expect(data).toHaveProperty('health');
    expect(data).toHaveProperty('recovery');
    expect(data).toHaveProperty('connections');
    expect(data).toHaveProperty('monitoring');
  });
});

describe('End-to-End Recovery Scenarios', () => {
  let integration;
  let mockMCPTools;

  beforeEach(async () => {
    // Mock MCP Tools
    mockMCPTools = {
      swarm_init: jest.fn().mockResolvedValue({ swarmId: 'test-swarm' }),
      swarm_status: jest.fn().mockResolvedValue({
        agents: [],
        options: { topology: 'mesh' },
      }),
      agent_spawn: jest.fn().mockResolvedValue({ id: 'test-agent' }),
      agent_list: jest.fn().mockResolvedValue({ agents: [] }),
    };

    integration = new RecoveryIntegration({
      enableHealthMonitoring: true,
      enableRecoveryWorkflows: true,
      enableConnectionManagement: true,
      enableMonitoringDashboard: true,
      enableChaosEngineering: true,
      autoIntegrate: true,
      configValidation: false,
    });

    integration.setMCPTools(mockMCPTools);
    await integration.start();
  });

  afterEach(async () => {
    if (integration) {
      await integration.shutdown();
    }
  });

  test('should handle health check failure and trigger recovery', async () => {
    const healthMonitor = integration.healthMonitor;
    const recoveryWorkflows = integration.recoveryWorkflows;

    // Register a failing health check
    healthMonitor.registerHealthCheck('test.failing', async () => {
      throw new Error('Simulated failure');
    });

    // Register a recovery workflow for this failure
    recoveryWorkflows.registerWorkflow('test.recovery', {
      description: 'Test recovery workflow',
      triggers: ['test.failing'],
      steps: [
        {
          name: 'diagnose',
          action: async () => ({ diagnosed: true }),
        },
        {
          name: 'fix',
          action: async () => ({ fixed: true }),
        },
      ],
    });

    // Run the failing health check multiple times to trigger alert
    await healthMonitor.runHealthCheck('test.failing');
    await healthMonitor.runHealthCheck('test.failing');
    await healthMonitor.runHealthCheck('test.failing');

    // Wait a bit for alert processing
    await new Promise(resolve => setTimeout(resolve, 100));

    // Check that alert was recorded
    const dashboardData = integration.monitoringDashboard.exportDashboardData();
    expect(dashboardData.alerts.recent.length).toBeGreaterThan(0);
  });

  test('should integrate connection failure with recovery', async () => {
    const connectionManager = integration.connectionManager;
    const recoveryWorkflows = integration.recoveryWorkflows;

    // Mock a connection that will fail
    const connectionId = await connectionManager.registerConnection({
      type: 'stdio',
      command: 'nonexistent-command',
      args: [],
      autoReconnect: false,
    });

    // Wait for connection attempt to fail
    await new Promise(resolve => setTimeout(resolve, 100));

    // Check connection status
    const connectionStatus = connectionManager.getConnectionStatus(connectionId);
    expect(connectionStatus.status).toBe('failed');

    // Verify recovery stats are updated
    const recoveryStats = recoveryWorkflows.getRecoveryStats();
    expect(recoveryStats).toHaveProperty('totalRecoveries');
  });

  test('should provide comprehensive system monitoring', () => {
    const systemData = integration.exportSystemData();

    // Verify all components are reporting data
    expect(systemData.health).toBeDefined();
    expect(systemData.recovery).toBeDefined();
    expect(systemData.connections).toBeDefined();
    expect(systemData.monitoring).toBeDefined();
    expect(systemData.chaos).toBeDefined();

    // Verify system status
    expect(systemData.status.isInitialized).toBe(true);
    expect(systemData.status.isRunning).toBe(true);
    expect(Object.keys(systemData.status.components).length).toBeGreaterThan(0);
  });
});

describe('Performance and Load Testing', () => {
  let integration;

  beforeEach(async () => {
    integration = new RecoveryIntegration({
      enableHealthMonitoring: true,
      enableRecoveryWorkflows: true,
      enableConnectionManagement: false, // Simplify for performance test
      enableMonitoringDashboard: true,
      enableChaosEngineering: false,
      performanceOptimization: true,
    });
  });

  afterEach(async () => {
    if (integration) {
      await integration.shutdown();
    }
  });

  test('should handle multiple concurrent health checks', async () => {
    const healthMonitor = integration.healthMonitor;

    // Register multiple health checks
    for (let i = 0; i < 10; i++) {
      healthMonitor.registerHealthCheck(`test.check.${i}`, async () => {
        await new Promise(resolve => setTimeout(resolve, Math.random() * 10));
        return { checkId: i };
      });
    }

    const startTime = Date.now();
    const { results } = await healthMonitor.runAllHealthChecks();
    const duration = Date.now() - startTime;

    expect(results.length).toBeGreaterThanOrEqual(10);
    expect(duration).toBeLessThan(1000); // Should complete within 1 second
  });

  test('should handle multiple concurrent recovery workflows', async () => {
    const recoveryWorkflows = integration.recoveryWorkflows;

    // Register multiple recovery workflows
    for (let i = 0; i < 5; i++) {
      recoveryWorkflows.registerWorkflow(`test.workflow.${i}`, {
        description: `Test workflow ${i}`,
        triggers: [`test.trigger.${i}`],
        steps: [
          {
            name: 'step1',
            action: async () => {
              await new Promise(resolve => setTimeout(resolve, 50));
              return { workflowId: i };
            },
          },
        ],
      });
    }

    // Trigger multiple recoveries concurrently
    const recoveryPromises = [];
    for (let i = 0; i < 3; i++) { // Within concurrent limit
      recoveryPromises.push(
        recoveryWorkflows.triggerRecovery(`test.trigger.${i}`),
      );
    }

    const results = await Promise.all(recoveryPromises);

    expect(results).toHaveLength(3);
    results.forEach(result => {
      expect(result.status).toBe('completed');
    });
  });

  test('should track performance metrics accurately', () => {
    const metrics = integration.getPerformanceMetrics();

    expect(metrics.initializationTime).toBeGreaterThan(0);
    expect(metrics.componentStartupTimes).toBeDefined();
    expect(metrics.currentMemoryUsage).toBeDefined();
    expect(metrics.currentMemoryUsage.heapUsed).toBeGreaterThan(0);
  });
});

describe('Error Handling and Edge Cases', () => {
  test('should handle component initialization failure gracefully', async () => {
    // Mock a component class that fails to initialize
    const FailingComponent = class {
      constructor() {}
      async initialize() {
        throw new Error('Initialization failed');
      }
    };

    const integration = new RecoveryIntegration({
      enableHealthMonitoring: false,
      enableRecoveryWorkflows: false,
      enableConnectionManagement: false,
      enableMonitoringDashboard: false,
      enableChaosEngineering: false,
    });

    // Manually try to initialize a failing component
    await expect(
      integration.initializeComponent('failing', FailingComponent),
    ).rejects.toThrow('Initialization failed');

    await integration.shutdown();
  });

  test('should handle missing dependencies gracefully', async () => {
    const integration = new RecoveryIntegration({
      enableHealthMonitoring: true,
      enableRecoveryWorkflows: true,
      autoIntegrate: true,
    });

    // Don't set MCP tools - integrations should be skipped
    const systemStatus = integration.getSystemStatus();

    expect(systemStatus.isInitialized).toBe(true);

    // Check that some integrations were skipped due to missing dependencies
    let skippedIntegrations = 0;
    for (const [key, status] of integration.integrationStatus) {
      if (status.status === 'skipped') {
        skippedIntegrations++;
      }
    }

    expect(skippedIntegrations).toBeGreaterThan(0);

    await integration.shutdown();
  });

  test('should handle emergency shutdown correctly', async () => {
    const integration = new RecoveryIntegration({
      enableChaosEngineering: true,
      safetyEnabled: false,
    });

    const shutdownSpy = jest.fn();
    integration.on('emergency:shutdown', shutdownSpy);

    await integration.emergencyShutdown('Test emergency');

    expect(shutdownSpy).toHaveBeenCalledWith({ reason: 'Test emergency' });
    expect(integration.isRunning).toBe(false);
  });
});
