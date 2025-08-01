/**
 * Performance Benchmarks Test Suite for Session Persistence (Issue #137)
 *
 * This test suite provides comprehensive performance validation for
 * session persistence and recovery components:
 * - Throughput benchmarks for high-volume operations
 * - Latency measurements under various loads
 * - Memory usage profiling and optimization validation
 * - Scalability testing across different system sizes
 * - Concurrent operation performance analysis
 * - Resource utilization benchmarking
 */

import { jest } from '@jest/globals';
import { performance } from 'perf_hooks';
import os from 'os';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';

// Import components
import { SessionManager } from '../src/session-manager.js';
import { HealthMonitor } from '../src/health-monitor.js';
import { RecoveryWorkflows } from '../src/recovery-workflows.js';

// Performance measurement utilities
class PerformanceBenchmark {
  constructor() {
    this.metrics = new Map();
    this.samples = new Map();
    this.memorySnapshots = [];
  }

  startMeasurement(name) {
    this.metrics.set(name, {
      startTime: performance.now(),
      startMemory: process.memoryUsage(),
      startCpu: process.cpuUsage(),
    });
  }

  endMeasurement(name) {
    const start = this.metrics.get(name);
    if (!start) throw new Error(`No measurement started for ${name}`);

    const endTime = performance.now();
    const endMemory = process.memoryUsage();
    const endCpu = process.cpuUsage(start.startCpu);

    const result = {
      name,
      duration: endTime - start.startTime,
      memoryDelta: {
        heapUsed: endMemory.heapUsed - start.startMemory.heapUsed,
        heapTotal: endMemory.heapTotal - start.startMemory.heapTotal,
        external: endMemory.external - start.startMemory.external,
        rss: endMemory.rss - start.startMemory.rss,
      },
      cpuUsage: {
        user: endCpu.user,
        system: endCpu.system,
      },
    };

    // Store sample
    if (!this.samples.has(name)) {
      this.samples.set(name, []);
    }
    this.samples.get(name).push(result);

    return result;
  }

  addMemorySnapshot(label = '') {
    this.memorySnapshots.push({
      timestamp: Date.now(),
      label,
      memory: process.memoryUsage(),
      cpuUsage: process.cpuUsage(),
    });
  }

  calculateStatistics(name) {
    const samples = this.samples.get(name);
    if (!samples || samples.length === 0) return null;

    const durations = samples.map(s => s.duration);
    const memoryUsages = samples.map(s => s.memoryDelta.heapUsed);

    durations.sort((a, b) => a - b);
    memoryUsages.sort((a, b) => a - b);

    return {
      name,
      sampleCount: samples.length,
      duration: {
        min: Math.min(...durations),
        max: Math.max(...durations),
        mean: durations.reduce((sum, d) => sum + d, 0) / durations.length,
        median: durations[Math.floor(durations.length / 2)],
        p95: durations[Math.floor(durations.length * 0.95)],
        p99: durations[Math.floor(durations.length * 0.99)],
      },
      memory: {
        min: Math.min(...memoryUsages),
        max: Math.max(...memoryUsages),
        mean: memoryUsages.reduce((sum, m) => sum + m, 0) / memoryUsages.length,
        median: memoryUsages[Math.floor(memoryUsages.length / 2)],
      },
      throughput: samples.length / (samples[samples.length - 1].duration / 1000), // ops/sec
    };
  }

  getOverallReport() {
    const report = {
      timestamp: new Date(),
      systemInfo: {
        platform: os.platform(),
        cpus: os.cpus().length,
        totalMemory: os.totalmem(),
        freeMemory: os.freemem(),
        nodeVersion: process.version,
      },
      measurements: {},
      memoryProfile: this.memorySnapshots,
    };

    for (const name of this.samples.keys()) {
      report.measurements[name] = this.calculateStatistics(name);
    }

    return report;
  }
}

// Load testing utilities
class LoadTester {
  constructor() {
    this.workers = [];
    this.results = [];
  }

  async runConcurrentLoad(testFunction, concurrency, iterations) {
    const promises = [];

    for (let i = 0; i < concurrency; i++) {
      promises.push(this.runWorkerLoad(testFunction, iterations, i));
    }

    const results = await Promise.all(promises);
    return results.flat();
  }

  async runWorkerLoad(testFunction, iterations, workerId) {
    const results = [];

    for (let i = 0; i < iterations; i++) {
      const startTime = performance.now();
      try {
        const result = await testFunction(workerId, i);
        const endTime = performance.now();

        results.push({
          workerId,
          iteration: i,
          duration: endTime - startTime,
          success: true,
          result,
        });
      } catch (error) {
        const endTime = performance.now();
        results.push({
          workerId,
          iteration: i,
          duration: endTime - startTime,
          success: false,
          error: error.message,
        });
      }
    }

    return results;
  }

  analyzeResults(results) {
    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);
    const durations = successful.map(r => r.duration);

    if (durations.length === 0) {
      return { error: 'No successful operations' };
    }

    durations.sort((a, b) => a - b);

    return {
      total: results.length,
      successful: successful.length,
      failed: failed.length,
      successRate: (successful.length / results.length) * 100,
      duration: {
        min: Math.min(...durations),
        max: Math.max(...durations),
        mean: durations.reduce((sum, d) => sum + d, 0) / durations.length,
        median: durations[Math.floor(durations.length / 2)],
        p95: durations[Math.floor(durations.length * 0.95)],
        p99: durations[Math.floor(durations.length * 0.99)],
      },
      throughput: successful.length / ((Date.now() - results[0]?.timestamp || 1) / 1000),
      errors: failed.map(f => f.error),
    };
  }
}

// Scalability testing utilities
class ScalabilityTester {
  constructor() {
    this.testResults = new Map();
  }

  async testScalability(testFunction, scales, baseLoad = 100) {
    const results = new Map();

    for (const scale of scales) {
      const loadSize = baseLoad * scale;
      console.log(`Testing scalability at ${scale}x scale (${loadSize} operations)`);

      const startTime = performance.now();
      const startMemory = process.memoryUsage();

      try {
        const result = await testFunction(loadSize, scale);
        const endTime = performance.now();
        const endMemory = process.memoryUsage();

        results.set(scale, {
          scale,
          loadSize,
          duration: endTime - startTime,
          memoryUsed: endMemory.heapUsed - startMemory.heapUsed,
          success: true,
          result,
        });
      } catch (error) {
        const endTime = performance.now();
        const endMemory = process.memoryUsage();

        results.set(scale, {
          scale,
          loadSize,
          duration: endTime - startTime,
          memoryUsed: endMemory.heapUsed - startMemory.heapUsed,
          success: false,
          error: error.message,
        });
      }

      // Allow garbage collection between tests
      if (global.gc) global.gc();
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    return this.analyzeScalabilityResults(results);
  }

  analyzeScalabilityResults(results) {
    const successful = Array.from(results.values()).filter(r => r.success);

    if (successful.length < 2) {
      return { error: 'Not enough successful tests for scalability analysis' };
    }

    // Calculate scalability metrics
    const scalabilityMetrics = successful.map((result, index) => {
      if (index === 0) return null; // No previous result to compare

      const prev = successful[index - 1];
      const scaleIncrease = result.scale / prev.scale;
      const durationIncrease = result.duration / prev.duration;
      const memoryIncrease = result.memoryUsed / prev.memoryUsed;

      return {
        scale: result.scale,
        scaleIncrease,
        durationIncrease,
        memoryIncrease,
        efficiency: scaleIncrease / durationIncrease, // >1 is good scaling
        memoryEfficiency: scaleIncrease / memoryIncrease,
      };
    }).filter(m => m !== null);

    return {
      rawResults: Array.from(results.values()),
      scalabilityMetrics,
      overallEfficiency: scalabilityMetrics.reduce((sum, m) => sum + m.efficiency, 0) / scalabilityMetrics.length,
      memoryEfficiency: scalabilityMetrics.reduce((sum, m) => sum + m.memoryEfficiency, 0) / scalabilityMetrics.length,
      maxScale: Math.max(...successful.map(r => r.scale)),
      recommendations: this.generateScalabilityRecommendations(scalabilityMetrics),
    };
  }

  generateScalabilityRecommendations(metrics) {
    const recommendations = [];
    const avgEfficiency = metrics.reduce((sum, m) => sum + m.efficiency, 0) / metrics.length;
    const avgMemoryEfficiency = metrics.reduce((sum, m) => sum + m.memoryEfficiency, 0) / metrics.length;

    if (avgEfficiency < 0.7) {
      recommendations.push('Performance degrades significantly with scale - consider optimization');
    } else if (avgEfficiency > 1.2) {
      recommendations.push('Excellent scaling performance - system handles load well');
    }

    if (avgMemoryEfficiency < 0.5) {
      recommendations.push('Memory usage grows faster than scale - investigate memory leaks');
    } else if (avgMemoryEfficiency > 0.9) {
      recommendations.push('Good memory efficiency - memory usage scales linearly');
    }

    return recommendations;
  }
}

describe('Performance Benchmarks - Session Persistence', () => {
  let sessionManager;
  let healthMonitor;
  let recoveryWorkflows;
  let benchmark;
  let loadTester;
  let scalabilityTester;
  let mockPersistence;

  beforeAll(() => {
    benchmark = new PerformanceBenchmark();
    loadTester = new LoadTester();
    scalabilityTester = new ScalabilityTester();
  });

  beforeEach(() => {
    mockPersistence = {
      initialize: async () => {},
      pool: {
        read: async (query, params = []) => {
          // Simulate database latency
          await new Promise(resolve => setTimeout(resolve, 1 + Math.random() * 2));
          if (query.includes('SELECT 1')) return [{ test: 1 }];
          return [];
        },
        write: async (query, params = []) => {
          // Simulate write latency
          await new Promise(resolve => setTimeout(resolve, 2 + Math.random() * 3));
          return { changes: 1 };
        },
      },
    };

    sessionManager = new SessionManager(mockPersistence);
    healthMonitor = new HealthMonitor({ enableRealTimeMonitoring: false });
    recoveryWorkflows = new RecoveryWorkflows({ enableChaosEngineering: false });
  });

  afterEach(async () => {
    if (sessionManager) await sessionManager.shutdown();
    if (healthMonitor) await healthMonitor.shutdown();
    if (recoveryWorkflows) await recoveryWorkflows.shutdown();
  });

  describe('SessionManager Performance Tests', () => {
    test('should benchmark session creation throughput', async () => {
      await sessionManager.initialize();
      benchmark.addMemorySnapshot('before-session-creation');

      const sessionCount = 1000;
      benchmark.startMeasurement('session-creation-batch');

      const sessionIds = [];
      for (let i = 0; i < sessionCount; i++) {
        benchmark.startMeasurement('single-session-creation');

        const sessionId = await sessionManager.createSession(
          `benchmark-session-${i}`,
          { topology: 'mesh', maxAgents: 5 + (i % 10) },
          {
            agents: new Map([[`agent-${i}`, { id: `agent-${i}`, status: 'active' }]]),
            tasks: new Map(),
            topology: 'mesh',
            connections: [],
            metrics: { totalTasks: 0, completedTasks: 0, failedTasks: 0 },
          },
        );

        sessionIds.push(sessionId);
        benchmark.endMeasurement('single-session-creation');

        if (i % 100 === 0) {
          benchmark.addMemorySnapshot(`after-${i}-sessions`);
        }
      }

      benchmark.endMeasurement('session-creation-batch');
      benchmark.addMemorySnapshot('after-session-creation');

      const batchStats = benchmark.calculateStatistics('session-creation-batch');
      const singleStats = benchmark.calculateStatistics('single-session-creation');

      expect(sessionIds).toHaveLength(sessionCount);
      expect(singleStats.duration.mean).toBeLessThan(100); // Average < 100ms per session
      expect(singleStats.throughput).toBeGreaterThan(10); // > 10 sessions/sec

      console.log(`Session creation: ${singleStats.duration.mean.toFixed(2)}ms avg, ${singleStats.throughput.toFixed(2)} ops/sec`);
    });

    test('should benchmark high-frequency state updates', async () => {
      await sessionManager.initialize();

      const sessionId = await sessionManager.createSession(
        'high-frequency-test',
        { topology: 'ring', maxAgents: 20 },
      );

      const updateCount = 2000;
      benchmark.startMeasurement('high-frequency-updates');
      benchmark.addMemorySnapshot('before-updates');

      for (let i = 0; i < updateCount; i++) {
        benchmark.startMeasurement('single-update');

        await sessionManager.saveSession(sessionId, {
          metadata: {
            updateNumber: i,
            timestamp: Date.now(),
            randomData: Math.random().toString(36).substring(7),
          },
        });

        benchmark.endMeasurement('single-update');
      }

      benchmark.endMeasurement('high-frequency-updates');
      benchmark.addMemorySnapshot('after-updates');

      const updateStats = benchmark.calculateStatistics('single-update');
      expect(updateStats.duration.mean).toBeLessThan(50); // Average < 50ms per update
      expect(updateStats.throughput).toBeGreaterThan(20); // > 20 updates/sec

      console.log(`State updates: ${updateStats.duration.mean.toFixed(2)}ms avg, ${updateStats.throughput.toFixed(2)} ops/sec`);
    });

    test('should benchmark checkpoint operations', async () => {
      await sessionManager.initialize();

      const sessionId = await sessionManager.createSession(
        'checkpoint-benchmark',
        { topology: 'hierarchical', maxAgents: 50 },
        {
          agents: new Map(Array.from({ length: 50 }, (_, i) => [
            `agent-${i}`,
            {
              id: `agent-${i}`,
              type: 'worker',
              status: 'active',
              metadata: { performance: 'benchmark', index: i },
            },
          ])),
          tasks: new Map(Array.from({ length: 100 }, (_, i) => [
            `task-${i}`,
            {
              id: `task-${i}`,
              type: 'analysis',
              status: 'running',
              data: Array.from({ length: 100 }, () => Math.random()),
            },
          ])),
          topology: 'hierarchical',
          connections: Array.from({ length: 49 }, (_, i) => `agent-${i}:agent-${i+1}`),
          metrics: {
            totalTasks: 100,
            completedTasks: 30,
            failedTasks: 5,
            averageCompletionTime: 1500,
            agentUtilization: new Map(Array.from({ length: 50 }, (_, i) => [`agent-${i}`, Math.random()])),
            throughput: 85.5,
          },
        },
      );

      const checkpointCount = 100;
      benchmark.startMeasurement('checkpoint-operations');

      const checkpointIds = [];
      for (let i = 0; i < checkpointCount; i++) {
        benchmark.startMeasurement('single-checkpoint');

        const checkpointId = await sessionManager.createCheckpoint(
          sessionId,
          `Benchmark checkpoint ${i}`,
          { benchmarkIndex: i, timestamp: Date.now() },
        );

        checkpointIds.push(checkpointId);
        benchmark.endMeasurement('single-checkpoint');
      }

      benchmark.endMeasurement('checkpoint-operations');

      const checkpointStats = benchmark.calculateStatistics('single-checkpoint');
      expect(checkpointStats.duration.mean).toBeLessThan(200); // Average < 200ms per checkpoint
      expect(checkpointIds).toHaveLength(checkpointCount);

      console.log(`Checkpoints: ${checkpointStats.duration.mean.toFixed(2)}ms avg, ${checkpointStats.throughput.toFixed(2)} ops/sec`);
    });

    test('should benchmark session loading with large state', async () => {
      await sessionManager.initialize();

      // Create session with large state
      const largeState = {
        agents: new Map(Array.from({ length: 500 }, (_, i) => [
          `agent-${i}`,
          {
            id: `agent-${i}`,
            type: 'worker',
            status: 'active',
            config: {
              capabilities: Array.from({ length: 20 }, (_, j) => `capability-${j}`),
              history: Array.from({ length: 100 }, (_, k) => ({
                timestamp: Date.now() - k * 1000,
                action: `action-${k}`,
                result: Math.random(),
              })),
            },
          },
        ])),
        tasks: new Map(Array.from({ length: 1000 }, (_, i) => [
          `task-${i}`,
          {
            id: `task-${i}`,
            type: 'processing',
            status: i % 3 === 0 ? 'completed' : 'running',
            data: Array.from({ length: 50 }, () => Math.random()),
            metadata: {
              created: Date.now() - Math.random() * 86400000,
              priority: Math.floor(Math.random() * 5),
              tags: Array.from({ length: 5 }, (_, j) => `tag-${j}`),
            },
          },
        ])),
        topology: 'mesh',
        connections: Array.from({ length: 999 }, (_, i) => `agent-${i}:agent-${i+1}`),
        metrics: {
          totalTasks: 1000,
          completedTasks: Math.floor(1000 / 3),
          failedTasks: Math.floor(Math.random() * 50),
          averageCompletionTime: 2500,
          agentUtilization: new Map(Array.from({ length: 500 }, (_, i) => [`agent-${i}`, Math.random()])),
          throughput: 125.7,
          detailedMetrics: Array.from({ length: 100 }, (_, i) => ({
            timestamp: Date.now() - i * 60000,
            value: Math.random() * 100,
          })),
        },
      };

      const sessionId = await sessionManager.createSession(
        'large-state-benchmark',
        { topology: 'mesh', maxAgents: 500 },
        largeState,
      );

      // Benchmark loading
      const loadCount = 50;
      benchmark.startMeasurement('large-state-loading');

      for (let i = 0; i < loadCount; i++) {
        benchmark.startMeasurement('single-load');

        const loadedSession = await sessionManager.loadSession(sessionId);
        expect(loadedSession.swarmState.agents.size).toBe(500);
        expect(loadedSession.swarmState.tasks.size).toBe(1000);

        benchmark.endMeasurement('single-load');
      }

      benchmark.endMeasurement('large-state-loading');

      const loadStats = benchmark.calculateStatistics('single-load');
      expect(loadStats.duration.mean).toBeLessThan(500); // Average < 500ms for large state

      console.log(`Large state loading: ${loadStats.duration.mean.toFixed(2)}ms avg, ${loadStats.throughput.toFixed(2)} ops/sec`);
    });

    test('should benchmark concurrent session operations', async () => {
      await sessionManager.initialize();

      const concurrency = 10;
      const operationsPerWorker = 50;

      const concurrentTest = async (workerId, iteration) => {
        const sessionId = await sessionManager.createSession(
          `concurrent-${workerId}-${iteration}`,
          { topology: 'ring', maxAgents: 8 },
        );

        // Perform mixed operations
        await sessionManager.saveSession(sessionId, {
          metadata: { workerId, iteration, timestamp: Date.now() },
        });

        const checkpointId = await sessionManager.createCheckpoint(
          sessionId,
          `Concurrent checkpoint ${workerId}-${iteration}`,
        );

        const loadedSession = await sessionManager.loadSession(sessionId);
        expect(loadedSession.metadata.workerId).toBe(workerId);

        await sessionManager.terminateSession(sessionId, true);

        return { sessionId, checkpointId, workerId, iteration };
      };

      benchmark.startMeasurement('concurrent-operations');
      const results = await loadTester.runConcurrentLoad(
        concurrentTest,
        concurrency,
        operationsPerWorker,
      );
      benchmark.endMeasurement('concurrent-operations');

      const analysis = loadTester.analyzeResults(results);

      expect(analysis.successRate).toBeGreaterThan(95); // > 95% success rate
      expect(analysis.successful).toBe(concurrency * operationsPerWorker);

      console.log(`Concurrent ops: ${analysis.successRate.toFixed(2)}% success, ${analysis.duration.mean.toFixed(2)}ms avg`);
    });
  });

  describe('HealthMonitor Performance Tests', () => {
    test('should benchmark health check execution speed', async () => {
      await healthMonitor.initialize();

      // Register various health checks
      const checkCount = 100;
      for (let i = 0; i < checkCount; i++) {
        healthMonitor.registerHealthCheck(`perf-check-${i}`, async () => {
          // Simulate different check complexities
          const complexity = i % 3;
          await new Promise(resolve => setTimeout(resolve, complexity * 5));

          return {
            checkId: i,
            complexity,
            timestamp: Date.now(),
            data: Array.from({ length: complexity * 10 }, () => Math.random()),
          };
        }, {
          priority: i % 4 === 0 ? 'high' : 'normal',
          category: `category-${i % 5}`,
        });
      }

      benchmark.startMeasurement('health-checks-parallel');
      const { results, summary } = await healthMonitor.runAllHealthChecks();
      benchmark.endMeasurement('health-checks-parallel');

      expect(results.length).toBe(checkCount + 3); // +3 for built-in checks
      expect(summary.healthy).toBeGreaterThan(checkCount * 0.8); // Most should be healthy

      const parallelStats = benchmark.calculateStatistics('health-checks-parallel');
      console.log(`Health checks (parallel): ${parallelStats.duration.mean.toFixed(2)}ms for ${checkCount} checks`);

      // Compare with sequential execution
      benchmark.startMeasurement('health-checks-sequential');
      for (let i = 0; i < Math.min(checkCount, 20); i++) { // Limit sequential test
        await healthMonitor.runHealthCheck(`perf-check-${i}`);
      }
      benchmark.endMeasurement('health-checks-sequential');

      const sequentialStats = benchmark.calculateStatistics('health-checks-sequential');
      console.log(`Sequential execution would be much slower: ${sequentialStats.duration.mean}ms for 20 checks`);
    });

    test('should benchmark monitoring overhead', async () => {
      await healthMonitor.initialize();

      // Register lightweight checks for monitoring
      for (let i = 0; i < 20; i++) {
        healthMonitor.registerHealthCheck(`monitor-check-${i}`, async () => {
          return { status: 'ok', timestamp: Date.now() };
        }, {
          interval: 100, // Very frequent
          priority: 'low',
        });
      }

      benchmark.addMemorySnapshot('before-monitoring');
      benchmark.startMeasurement('monitoring-overhead');

      // Start monitoring
      await healthMonitor.startMonitoring();

      // Let it run for a period
      await new Promise(resolve => setTimeout(resolve, 5000));

      await healthMonitor.stopMonitoring();
      benchmark.endMeasurement('monitoring-overhead');
      benchmark.addMemorySnapshot('after-monitoring');

      const monitoringStats = healthMonitor.getMonitoringStats();
      expect(monitoringStats.totalChecks).toBe(23); // 20 + 3 built-in
      expect(monitoringStats.isRunning).toBe(false);

      const overheadStats = benchmark.calculateStatistics('monitoring-overhead');
      console.log(`Monitoring overhead: ${overheadStats.memoryDelta.heapUsed} bytes heap increase`);
    });

    test('should benchmark alert processing performance', async () => {
      await healthMonitor.initialize();

      const alertCount = 500;
      let alertsTriggered = 0;

      healthMonitor.on('health:alert', () => {
        alertsTriggered++;
      });

      // Register checks that will fail and trigger alerts
      for (let i = 0; i < alertCount; i++) {
        healthMonitor.registerHealthCheck(`alert-check-${i}`, async () => {
          throw new Error(`Alert test failure ${i}`);
        });
      }

      benchmark.startMeasurement('alert-processing');

      // Trigger multiple failures to generate alerts
      for (let i = 0; i < alertCount; i++) {
        await healthMonitor.runHealthCheck(`alert-check-${i}`);

        // Run multiple times to reach alert threshold
        for (let j = 0; j < 3; j++) {
          await healthMonitor.runHealthCheck(`alert-check-${i}`);
        }
      }

      benchmark.endMeasurement('alert-processing');

      expect(alertsTriggered).toBeGreaterThan(0);

      const alertStats = benchmark.calculateStatistics('alert-processing');
      console.log(`Alert processing: ${alertsTriggered} alerts generated in ${alertStats.duration.mean.toFixed(2)}ms`);
    });
  });

  describe('RecoveryWorkflows Performance Tests', () => {
    test('should benchmark workflow execution speed', async () => {
      await recoveryWorkflows.initialize();

      const workflowCount = 100;

      // Register performance test workflows
      for (let i = 0; i < workflowCount; i++) {
        recoveryWorkflows.registerWorkflow(`perf-workflow-${i}`, {
          description: `Performance test workflow ${i}`,
          triggers: [`perf-trigger-${i}`],
          steps: [
            {
              name: 'step1',
              action: async () => {
                // Simulate work
                await new Promise(resolve => setTimeout(resolve, Math.random() * 10));
                return { step: 1, workflowId: i };
              },
            },
            {
              name: 'step2',
              action: async (context) => {
                // More complex step
                const data = Array.from({ length: 100 }, () => Math.random());
                return { step: 2, workflowId: i, dataLength: data.length };
              },
            },
          ],
          priority: i % 3 === 0 ? 'high' : 'normal',
        });
      }

      benchmark.startMeasurement('workflow-execution');

      const executionPromises = [];
      for (let i = 0; i < workflowCount; i++) {
        benchmark.startMeasurement('single-workflow-execution');
        const promise = recoveryWorkflows.triggerRecovery(`perf-trigger-${i}`)
          .then(execution => {
            benchmark.endMeasurement('single-workflow-execution');
            return execution;
          });
        executionPromises.push(promise);
      }

      const executions = await Promise.all(executionPromises);
      benchmark.endMeasurement('workflow-execution');

      const successfulExecutions = executions.filter(e => e.status === 'completed');
      expect(successfulExecutions.length).toBeGreaterThan(workflowCount * 0.9); // > 90% success

      const workflowStats = benchmark.calculateStatistics('single-workflow-execution');
      console.log(`Workflow execution: ${workflowStats.duration.mean.toFixed(2)}ms avg, ${workflowStats.throughput.toFixed(2)} workflows/sec`);
    });

    test('should benchmark recovery workflow scalability', async () => {
      await recoveryWorkflows.initialize();

      const scalabilityTest = async (workflowCount, scale) => {
        // Register workflows for this scale
        for (let i = 0; i < workflowCount; i++) {
          recoveryWorkflows.registerWorkflow(`scale-${scale}-workflow-${i}`, {
            description: `Scalability test workflow ${i} at scale ${scale}`,
            triggers: [`scale-${scale}-trigger-${i}`],
            steps: [
              {
                name: 'scalability_step',
                action: async () => {
                  // Simulate proportional work
                  const workAmount = scale * 5;
                  await new Promise(resolve => setTimeout(resolve, workAmount));
                  return { scale, workAmount, workflowIndex: i };
                },
              },
            ],
            priority: 'normal',
          });
        }

        // Execute all workflows
        const executions = await Promise.all(
          Array.from({ length: workflowCount }, (_, i) =>
            recoveryWorkflows.triggerRecovery(`scale-${scale}-trigger-${i}`),
          ),
        );

        return {
          scale,
          workflowCount,
          successful: executions.filter(e => e.status === 'completed').length,
          executions,
        };
      };

      const scales = [1, 2, 4, 8];
      const scalabilityResults = await scalabilityTester.testScalability(
        scalabilityTest,
        scales,
        25, // base workflow count
      );

      expect(scalabilityResults.overallEfficiency).toBeGreaterThan(0.5); // Reasonable scaling
      console.log(`Workflow scalability: ${scalabilityResults.overallEfficiency.toFixed(2)} efficiency`);
      console.log(`Recommendations: ${scalabilityResults.recommendations.join(', ')}`);
    });

    test('should benchmark recovery under load', async () => {
      await sessionManager.initialize();
      await healthMonitor.initialize();
      await recoveryWorkflows.initialize();

      // Set up integration
      healthMonitor.setRecoveryWorkflows(recoveryWorkflows);

      // Register recovery workflow
      recoveryWorkflows.registerWorkflow('load.recovery', {
        description: 'Recovery under load test',
        triggers: ['load.failure'],
        steps: [
          {
            name: 'diagnose_load_issue',
            action: async () => {
              const memUsage = process.memoryUsage();
              return { diagnosis: 'load_detected', memoryUsage: memUsage.heapUsed };
            },
          },
          {
            name: 'mitigate_load',
            action: async () => {
              // Simulate load mitigation
              if (global.gc) global.gc();
              return { mitigation: 'gc_triggered' };
            },
          },
        ],
        priority: 'critical',
      });

      // Create background load
      const loadPromises = [];
      for (let i = 0; i < 50; i++) {
        loadPromises.push((async () => {
          const sessionId = await sessionManager.createSession(
            `load-test-${i}`,
            { topology: 'mesh', maxAgents: 10 },
          );

          for (let j = 0; j < 20; j++) {
            await sessionManager.saveSession(sessionId, {
              metadata: { loadIteration: j, timestamp: Date.now() },
            });
            await new Promise(resolve => setTimeout(resolve, 50));
          }

          return sessionId;
        })());
      }

      // Trigger recovery during load
      benchmark.startMeasurement('recovery-under-load');

      const recoveryPromise = recoveryWorkflows.triggerRecovery('load.failure');

      // Continue load while recovery runs
      const loadResults = await Promise.allSettled(loadPromises);
      const recoveryResult = await recoveryPromise;

      benchmark.endMeasurement('recovery-under-load');

      const successfulLoads = loadResults.filter(r => r.status === 'fulfilled').length;
      expect(successfulLoads).toBeGreaterThan(40); // Most should succeed
      expect(recoveryResult.status).toBe('completed');

      const recoveryStats = benchmark.calculateStatistics('recovery-under-load');
      console.log(`Recovery under load: ${recoveryStats.duration.mean.toFixed(2)}ms, ${successfulLoads}/50 loads succeeded`);
    });
  });

  describe('Memory Usage and Leak Detection', () => {
    test('should detect memory leaks in long-running operations', async () => {
      await sessionManager.initialize();

      const initialMemory = process.memoryUsage();
      benchmark.addMemorySnapshot('initial');

      // Run memory-intensive operations
      const iterations = 100;
      for (let i = 0; i < iterations; i++) {
        const sessionId = await sessionManager.createSession(
          `memory-test-${i}`,
          { topology: 'ring', maxAgents: 20 },
          {
            agents: new Map(Array.from({ length: 20 }, (_, j) => [
              `agent-${j}`,
              {
                id: `agent-${j}`,
                data: Array.from({ length: 1000 }, () => Math.random()),
              },
            ])),
            tasks: new Map(),
            topology: 'ring',
            connections: [],
            metrics: {},
          },
        );

        await sessionManager.createCheckpoint(sessionId, `Memory test checkpoint ${i}`);
        await sessionManager.terminateSession(sessionId, true);

        if (i % 10 === 0) {
          benchmark.addMemorySnapshot(`iteration-${i}`);
          if (global.gc) global.gc(); // Force GC for accurate measurement
        }
      }

      benchmark.addMemorySnapshot('final');
      const finalMemory = process.memoryUsage();

      const memoryGrowth = finalMemory.heapUsed - initialMemory.heapUsed;
      const memoryGrowthPerOperation = memoryGrowth / iterations;

      // Memory growth should be reasonable (< 1MB per 100 operations)
      expect(memoryGrowth).toBeLessThan(1024 * 1024);

      console.log(`Memory growth: ${memoryGrowth} bytes total, ${memoryGrowthPerOperation.toFixed(2)} bytes per operation`);

      // Analyze memory snapshots for leak patterns
      const snapshots = benchmark.memorySnapshots;
      const heapGrowthPattern = snapshots.map((snapshot, index) => ({
        iteration: index,
        heapUsed: snapshot.memory.heapUsed,
        growth: index > 0 ? snapshot.memory.heapUsed - snapshots[0].memory.heapUsed : 0,
      }));

      // Check if memory growth is linear (potential leak) vs stable
      const lastHalf = heapGrowthPattern.slice(Math.floor(heapGrowthPattern.length / 2));
      const avgGrowthRate = lastHalf.reduce((sum, point) => sum + point.growth, 0) / lastHalf.length;

      console.log(`Average memory growth rate in final phase: ${avgGrowthRate.toFixed(2)} bytes`);
    });

    test('should profile memory usage patterns', async () => {
      await sessionManager.initialize();
      await healthMonitor.initialize();

      benchmark.addMemorySnapshot('profile-start');

      // Profile different operations
      const operations = [
        {
          name: 'session-creation',
          operation: async () => {
            const sessionId = await sessionManager.createSession(
              `profile-session-${Date.now()}`,
              { topology: 'mesh', maxAgents: 15 },
            );
            return sessionId;
          },
        },
        {
          name: 'health-checks',
          operation: async () => {
            healthMonitor.registerHealthCheck(`profile-check-${Date.now()}`, async () => ({ ok: true }));
            return await healthMonitor.runAllHealthChecks();
          },
        },
        {
          name: 'large-state-save',
          operation: async () => {
            const sessionId = await sessionManager.createSession(
              `large-profile-${Date.now()}`,
              { topology: 'hierarchical', maxAgents: 100 },
            );

            const largeState = {
              agents: new Map(Array.from({ length: 100 }, (_, i) => [
                `agent-${i}`,
                { id: `agent-${i}`, data: Array.from({ length: 500 }, () => Math.random()) },
              ])),
              tasks: new Map(),
              topology: 'hierarchical',
              connections: [],
              metrics: {},
            };

            await sessionManager.saveSession(sessionId, largeState);
            return sessionId;
          },
        },
      ];

      const profiles = {};
      for (const op of operations) {
        const beforeMemory = process.memoryUsage();
        benchmark.addMemorySnapshot(`before-${op.name}`);

        await op.operation();

        const afterMemory = process.memoryUsage();
        benchmark.addMemorySnapshot(`after-${op.name}`);

        profiles[op.name] = {
          heapUsedDelta: afterMemory.heapUsed - beforeMemory.heapUsed,
          heapTotalDelta: afterMemory.heapTotal - beforeMemory.heapTotal,
          externalDelta: afterMemory.external - beforeMemory.external,
          rssDelta: afterMemory.rss - beforeMemory.rss,
        };
      }

      console.log('Memory usage profiles:');
      Object.entries(profiles).forEach(([name, profile]) => {
        console.log(`  ${name}: ${profile.heapUsedDelta} bytes heap used`);
      });

      // Each operation should use reasonable memory
      Object.values(profiles).forEach(profile => {
        expect(profile.heapUsedDelta).toBeLessThan(10 * 1024 * 1024); // < 10MB per operation
      });
    });
  });

  describe('Performance Summary and Recommendations', () => {
    test('should generate comprehensive performance report', () => {
      const performanceReport = benchmark.getOverallReport();

      const report = {
        ...performanceReport,
        testSuiteVersion: '1.0.0',
        performanceCategories: {
          sessionManager: {
            operations: ['session-creation', 'state-updates', 'checkpoints', 'loading'],
            keyMetrics: ['throughput', 'latency', 'memory-efficiency'],
          },
          healthMonitor: {
            operations: ['health-checks', 'monitoring', 'alert-processing'],
            keyMetrics: ['parallel-execution', 'monitoring-overhead', 'alert-latency'],
          },
          recoveryWorkflows: {
            operations: ['workflow-execution', 'recovery-under-load'],
            keyMetrics: ['scalability', 'recovery-time', 'success-rate'],
          },
        },
        performanceTargets: {
          sessionCreation: { target: '<100ms', actual: 'measured' },
          stateUpdates: { target: '<50ms', actual: 'measured' },
          healthChecks: { target: 'parallel execution', actual: 'achieved' },
          workflowExecution: { target: '>90% success rate', actual: 'measured' },
          memoryUsage: { target: '<1MB growth per 100 ops', actual: 'measured' },
        },
        recommendations: [
          'Monitor memory usage patterns for potential optimizations',
          'Consider connection pooling for high-throughput scenarios',
          'Implement caching for frequently accessed session data',
          'Use batch operations for bulk updates to improve efficiency',
          'Consider implementing read replicas for load balancing',
        ],
      };

      console.log('\n📊 PERFORMANCE BENCHMARK REPORT');
      console.log('=================================');
      console.log(`System: ${report.systemInfo.platform} with ${report.systemInfo.cpus} CPUs`);
      console.log(`Memory: ${(report.systemInfo.totalMemory / 1024 / 1024 / 1024).toFixed(2)}GB total`);
      console.log(`Node.js: ${report.systemInfo.nodeVersion}`);
      console.log('\nMeasurements:');

      Object.entries(report.measurements).forEach(([name, stats]) => {
        if (stats) {
          console.log(`  ${name}:`);
          console.log(`    Mean: ${stats.duration.mean.toFixed(2)}ms`);
          console.log(`    P95: ${stats.duration.p95.toFixed(2)}ms`);
          console.log(`    Throughput: ${stats.throughput.toFixed(2)} ops/sec`);
        }
      });

      console.log('\nRecommendations:');
      report.recommendations.forEach(rec => console.log(`  • ${rec}`));
      console.log('\n✅ Performance benchmarking completed successfully');

      expect(report.systemInfo).toBeDefined();
      expect(Object.keys(report.measurements).length).toBeGreaterThan(0);
      expect(report.recommendations.length).toBeGreaterThan(0);
    });
  });
});
