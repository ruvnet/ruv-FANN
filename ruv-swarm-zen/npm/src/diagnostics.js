/**
 * Diagnostic utilities for ruv-swarm
 * Helps debug connection issues and performance problems
 */

import { Logger } from './logger.js';
import { loggingConfig } from './logging-config.js';
import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';

/**
 * Connection diagnostics
 */
export class ConnectionDiagnostics {
  constructor(logger = null) {
    this.logger = logger || loggingConfig.getLogger('diagnostics', { level: 'DEBUG' });
    this.connectionHistory = [];
    this.maxHistorySize = 100;
    this.activeConnections = new Map();
  }

  /**
   * Record connection event
   */
  recordEvent(connectionId, event, details = {}) {
    const timestamp = new Date().toISOString();
    const entry = {
      connectionId,
      event,
      timestamp,
      details,
      memoryUsage: process.memoryUsage(),
      cpuUsage: process.cpuUsage(),
    };

    this.connectionHistory.push(entry);
    if (this.connectionHistory.length > this.maxHistorySize) {
      this.connectionHistory.shift();
    }

    // Track active connections
    if (event === 'established') {
      this.activeConnections.set(connectionId, {
        startTime: Date.now(),
        ...details,
      });
    } else if (event === 'closed' || event === 'failed') {
      const conn = this.activeConnections.get(connectionId);
      if (conn) {
        entry.duration = Date.now() - conn.startTime;
        this.activeConnections.delete(connectionId);
      }
    }

    this.logger.debug('Connection event recorded', entry);
    return entry;
  }

  /**
   * Get connection summary
   */
  getConnectionSummary() {
    const events = this.connectionHistory.reduce((acc, event) => {
      acc[event.event] = (acc[event.event] || 0) + 1;
      return acc;
    }, {});

    const failures = this.connectionHistory.filter(e => e.event === 'failed');
    const recentFailures = failures.slice(-10);

    return {
      totalEvents: this.connectionHistory.length,
      eventCounts: events,
      activeConnections: this.activeConnections.size,
      recentFailures,
      failureRate: failures.length / this.connectionHistory.length,
    };
  }

  /**
   * Analyze connection patterns
   */
  analyzePatterns() {
    const failures = this.connectionHistory.filter(e => e.event === 'failed');

    // Group failures by error type
    const errorTypes = failures.reduce((acc, failure) => {
      const error = failure.details.error?.message || 'Unknown';
      acc[error] = (acc[error] || 0) + 1;
      return acc;
    }, {});

    // Find time patterns
    const hourlyFailures = new Array(24).fill(0);
    failures.forEach(failure => {
      const hour = new Date(failure.timestamp).getHours();
      hourlyFailures[hour]++;
    });

    // Memory patterns at failure time
    const memoryAtFailure = failures.map(f => ({
      timestamp: f.timestamp,
      heapUsed: f.memoryUsage.heapUsed / (1024 * 1024), // MB
      external: f.memoryUsage.external / (1024 * 1024), // MB
    }));

    return {
      errorTypes,
      hourlyFailures,
      memoryAtFailure,
      avgMemoryAtFailure: memoryAtFailure.reduce((sum, m) => sum + m.heapUsed, 0) / memoryAtFailure.length,
    };
  }

  /**
   * Generate diagnostic report
   */
  generateReport() {
    const summary = this.getConnectionSummary();
    const patterns = this.analyzePatterns();
    const systemInfo = {
      platform: process.platform,
      nodeVersion: process.version,
      uptime: process.uptime(),
      memoryUsage: process.memoryUsage(),
      cpuUsage: process.cpuUsage(),
    };

    const report = {
      timestamp: new Date().toISOString(),
      system: systemInfo,
      connections: summary,
      patterns,
      recommendations: this.generateRecommendations(summary, patterns),
    };

    this.logger.info('Diagnostic report generated', {
      failureRate: summary.failureRate,
      activeConnections: summary.activeConnections,
    });

    return report;
  }

  /**
   * Generate recommendations based on patterns
   */
  generateRecommendations(summary, patterns) {
    const recommendations = [];

    // High failure rate
    if (summary.failureRate > 0.1) {
      recommendations.push({
        severity: 'high',
        issue: 'High connection failure rate',
        suggestion: 'Check network stability and MCP server configuration',
      });
    }

    // Memory issues
    if (patterns.avgMemoryAtFailure > 500) {
      recommendations.push({
        severity: 'medium',
        issue: 'High memory usage during failures',
        suggestion: 'Consider increasing memory limits or optimizing memory usage',
      });
    }

    // Specific error patterns
    Object.entries(patterns.errorTypes).forEach(([error, count]) => {
      if (count > 5) {
        recommendations.push({
          severity: 'medium',
          issue: `Recurring error: ${error}`,
          suggestion: `Investigate root cause of: ${error}`,
        });
      }
    });

    return recommendations;
  }
}

/**
 * Performance diagnostics
 */
export class PerformanceDiagnostics {
  constructor(logger = null) {
    this.logger = logger || loggingConfig.getLogger('diagnostics', { level: 'DEBUG' });
    this.operations = new Map();
    this.thresholds = {
      'swarm_init': 1000, // 1 second
      'agent_spawn': 500, // 500ms
      'task_orchestrate': 2000, // 2 seconds
      'neural_train': 5000, // 5 seconds
    };
  }

  /**
   * Start tracking an operation
   */
  startOperation(name, metadata = {}) {
    const id = `${name}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    this.operations.set(id, {
      name,
      startTime: performance.now(),
      startMemory: process.memoryUsage(),
      metadata,
    });
    return id;
  }

  /**
   * End tracking an operation
   */
  endOperation(id, success = true) {
    const operation = this.operations.get(id);
    if (!operation) {
      return null;
    }

    const endTime = performance.now();
    const duration = endTime - operation.startTime;
    const endMemory = process.memoryUsage();

    const result = {
      ...operation,
      endTime,
      duration,
      success,
      memoryDelta: {
        heapUsed: endMemory.heapUsed - operation.startMemory.heapUsed,
        external: endMemory.external - operation.startMemory.external,
      },
      aboveThreshold: duration > (this.thresholds[operation.name] || 1000),
    };

    this.operations.delete(id);

    if (result.aboveThreshold) {
      this.logger.warn('Operation exceeded threshold', {
        operation: operation.name,
        duration,
        threshold: this.thresholds[operation.name],
      });
    }

    return result;
  }

  /**
   * Get slow operations
   */
  getSlowOperations(limit = 10) {
    const completed = [];

    // Get completed operations from logger's performance tracker
    // This would need to be implemented to store historical data

    return completed
      .filter(op => op.aboveThreshold)
      .sort((a, b) => b.duration - a.duration)
      .slice(0, limit);
  }
}

/**
 * System diagnostics
 */
export class SystemDiagnostics {
  constructor(logger = null) {
    this.logger = logger || loggingConfig.getLogger('diagnostics', { level: 'DEBUG' });
    this.samples = [];
    this.maxSamples = 60; // 1 minute of samples at 1Hz
  }

  /**
   * Collect system sample
   */
  collectSample() {
    const sample = {
      timestamp: Date.now(),
      memory: process.memoryUsage(),
      cpu: process.cpuUsage(),
      handles: process._getActiveHandles?.().length || 0,
      requests: process._getActiveRequests?.().length || 0,
    };

    this.samples.push(sample);
    if (this.samples.length > this.maxSamples) {
      this.samples.shift();
    }

    return sample;
  }

  /**
   * Start monitoring
   */
  startMonitoring(interval = 1000) {
    if (this.monitorInterval) {
      this.stopMonitoring();
    }

    this.monitorInterval = setInterval(() => {
      const sample = this.collectSample();

      // Check for anomalies
      if (sample.memory.heapUsed > 500 * 1024 * 1024) { // 500MB
        this.logger.warn('High memory usage detected', {
          heapUsed: `${(sample.memory.heapUsed / 1024 / 1024).toFixed(2)} MB`,
        });
      }

      if (sample.handles > 100) {
        this.logger.warn('High number of active handles', {
          handles: sample.handles,
        });
      }
    }, interval);

    this.logger.info('System monitoring started', { interval });
  }

  /**
   * Stop monitoring
   */
  stopMonitoring() {
    if (this.monitorInterval) {
      clearInterval(this.monitorInterval);
      this.monitorInterval = null;
      this.logger.info('System monitoring stopped');
    }
  }

  /**
   * Get system health
   */
  getSystemHealth() {
    if (this.samples.length === 0) {
      return { status: 'unknown', message: 'No samples collected' };
    }

    const latest = this.samples[this.samples.length - 1];
    const avgMemory = this.samples.reduce((sum, s) => sum + s.memory.heapUsed, 0) / this.samples.length;

    let status = 'healthy';
    const issues = [];

    if (latest.memory.heapUsed > 400 * 1024 * 1024) {
      status = 'warning';
      issues.push('High memory usage');
    }

    if (latest.handles > 50) {
      status = 'warning';
      issues.push('Many active handles');
    }

    if (avgMemory > 300 * 1024 * 1024) {
      status = 'warning';
      issues.push('Sustained high memory usage');
    }

    return {
      status,
      issues,
      metrics: {
        currentMemory: `${(latest.memory.heapUsed / 1024 / 1024).toFixed(2)} MB`,
        avgMemory: `${(avgMemory / 1024 / 1024).toFixed(2)} MB`,
        handles: latest.handles,
        requests: latest.requests,
      },
    };
  }
}

/**
 * Main diagnostics manager
 */
export class DiagnosticsManager {
  constructor() {
    this.logger = loggingConfig.getLogger('diagnostics', { level: 'DEBUG' });
    this.connection = new ConnectionDiagnostics(this.logger);
    this.performance = new PerformanceDiagnostics(this.logger);
    this.system = new SystemDiagnostics(this.logger);
  }

  /**
   * Enable all diagnostics
   */
  enableAll() {
    this.system.startMonitoring();
    this.logger.info('All diagnostics enabled');
  }

  /**
   * Disable all diagnostics
   */
  disableAll() {
    this.system.stopMonitoring();
    this.logger.info('All diagnostics disabled');
  }

  /**
   * Generate full diagnostic report
   */
  async generateFullReport(outputPath = null) {
    const report = {
      timestamp: new Date().toISOString(),
      connection: this.connection.generateReport(),
      performance: {
        slowOperations: this.performance.getSlowOperations(),
      },
      system: this.system.getSystemHealth(),
      logs: await this.collectRecentLogs(),
    };

    if (outputPath) {
      const reportPath = path.resolve(outputPath);
      fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
      this.logger.info('Diagnostic report saved', { path: reportPath });
    }

    return report;
  }

  /**
   * Collect recent logs
   */
  async collectRecentLogs() {
    // This would read from log files if file logging is enabled
    // For now, return a placeholder
    return {
      message: 'Log collection would read from log files',
      logsEnabled: process.env.LOG_TO_FILE === 'true',
    };
  }

  /**
   * Run diagnostic tests
   */
  async runDiagnosticTests() {
    const tests = [];

    // Test 1: Memory allocation
    tests.push(await this.testMemoryAllocation());

    // Test 2: File system access
    tests.push(await this.testFileSystem());

    // Test 3: WASM loading
    tests.push(await this.testWasmLoading());

    return {
      timestamp: new Date().toISOString(),
      tests,
      summary: {
        total: tests.length,
        passed: tests.filter(t => t.success).length,
        failed: tests.filter(t => !t.success).length,
      },
    };
  }

  async testMemoryAllocation() {
    try {
      const start = process.memoryUsage().heapUsed;
      const testArray = new Array(1000000).fill(0);
      const end = process.memoryUsage().heapUsed;

      return {
        name: 'Memory Allocation',
        success: true,
        allocated: `${((end - start) / 1024 / 1024).toFixed(2)} MB`,
      };
    } catch (error) {
      return {
        name: 'Memory Allocation',
        success: false,
        error: error.message,
      };
    }
  }

  async testFileSystem() {
    try {
      const testPath = path.join(process.cwd(), 'logs', '.diagnostic-test');
      fs.mkdirSync(path.dirname(testPath), { recursive: true });
      fs.writeFileSync(testPath, 'test');
      fs.unlinkSync(testPath);

      return {
        name: 'File System Access',
        success: true,
        path: testPath,
      };
    } catch (error) {
      return {
        name: 'File System Access',
        success: false,
        error: error.message,
      };
    }
  }

  async testWasmLoading() {
    try {
      // Test if WASM module can be loaded
      const wasmPath = path.join(process.cwd(), 'wasm', 'ruv_swarm_wasm_bg.wasm');
      const exists = fs.existsSync(wasmPath);

      return {
        name: 'WASM Module Check',
        success: exists,
        path: wasmPath,
        exists,
      };
    } catch (error) {
      return {
        name: 'WASM Module Check',
        success: false,
        error: error.message,
      };
    }
  }
}

// Singleton instance
export const diagnostics = new DiagnosticsManager();

export default diagnostics;
