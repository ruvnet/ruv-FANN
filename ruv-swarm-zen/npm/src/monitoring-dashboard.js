/**
 * Monitoring Dashboard Integration for RuvSwarm
 *
 * Provides dashboard-ready data export, real-time metrics streaming,
 * and integration points for monitoring systems like Grafana, Prometheus, etc.
 *
 * Features:
 * - Real-time metrics streaming
 * - Dashboard-ready data formatting
 * - Integration with external monitoring systems
 * - Custom metric aggregation and visualization
 * - Alert correlation and analysis
 * - Performance trend analysis
 */

import { EventEmitter } from 'events';
import { Logger } from './logger.js';
import { ErrorFactory } from './errors.js';
import { generateId } from './utils.js';

export class MonitoringDashboard extends EventEmitter {
  constructor(options = {}) {
    super();

    this.options = {
      metricsRetentionPeriod: options.metricsRetentionPeriod || 86400000, // 24 hours
      aggregationInterval: options.aggregationInterval || 60000, // 1 minute
      enableRealTimeStreaming: options.enableRealTimeStreaming !== false,
      enableTrendAnalysis: options.enableTrendAnalysis !== false,
      maxDataPoints: options.maxDataPoints || 1440, // 24 hours at 1-minute intervals
      exportFormats: options.exportFormats || ['prometheus', 'json', 'grafana'],
      ...options,
    };

    this.logger = new Logger({
      name: 'monitoring-dashboard',
      level: process.env.LOG_LEVEL || 'INFO',
      metadata: { component: 'monitoring-dashboard' },
    });

    // Data storage
    this.metrics = new Map();
    this.aggregatedMetrics = new Map();
    this.alerts = new Map();
    this.trends = new Map();
    this.healthStatus = new Map();

    // Real-time streaming
    this.streamingClients = new Set();
    this.lastUpdate = new Date();

    // Integration points
    this.healthMonitor = null;
    this.recoveryWorkflows = null;
    this.connectionManager = null;
    this.mcpTools = null;

    // Aggregation timer
    this.aggregationTimer = null;

    this.initialize();
  }

  /**
   * Initialize monitoring dashboard
   */
  async initialize() {
    try {
      this.logger.info('Initializing Monitoring Dashboard');

      // Start metric aggregation
      this.startMetricAggregation();

      // Set up data collection
      this.setupDataCollection();

      this.logger.info('Monitoring Dashboard initialized successfully');
      this.emit('dashboard:initialized');

    } catch (error) {
      const dashboardError = ErrorFactory.createError('resource',
        'Failed to initialize monitoring dashboard', {
          error: error.message,
          component: 'monitoring-dashboard',
        });
      this.logger.error('Monitoring Dashboard initialization failed', dashboardError);
      throw dashboardError;
    }
  }

  /**
   * Set integration points
   */
  setHealthMonitor(healthMonitor) {
    this.healthMonitor = healthMonitor;

    // Subscribe to health monitor events
    healthMonitor.on('health:check', (result) => {
      this.recordHealthMetric(result);
    });

    healthMonitor.on('health:alert', (alert) => {
      this.recordAlert(alert);
    });

    this.logger.info('Health Monitor integration configured');
  }

  setRecoveryWorkflows(recoveryWorkflows) {
    this.recoveryWorkflows = recoveryWorkflows;

    // Subscribe to recovery events
    recoveryWorkflows.on('recovery:started', (event) => {
      this.recordRecoveryMetric('started', event);
    });

    recoveryWorkflows.on('recovery:completed', (event) => {
      this.recordRecoveryMetric('completed', event);
    });

    recoveryWorkflows.on('recovery:failed', (event) => {
      this.recordRecoveryMetric('failed', event);
    });

    this.logger.info('Recovery Workflows integration configured');
  }

  setConnectionManager(connectionManager) {
    this.connectionManager = connectionManager;

    // Subscribe to connection events
    connectionManager.on('connection:established', (event) => {
      this.recordConnectionMetric('established', event);
    });

    connectionManager.on('connection:failed', (event) => {
      this.recordConnectionMetric('failed', event);
    });

    connectionManager.on('connection:closed', (event) => {
      this.recordConnectionMetric('closed', event);
    });

    this.logger.info('Connection Manager integration configured');
  }

  setMCPTools(mcpTools) {
    this.mcpTools = mcpTools;
    this.logger.info('MCP Tools integration configured');
  }

  /**
   * Record health metric
   */
  recordHealthMetric(healthResult) {
    const timestamp = new Date();
    const metricKey = `health.${healthResult.name}`;

    const metric = {
      timestamp,
      name: healthResult.name,
      status: healthResult.status,
      duration: healthResult.duration,
      category: healthResult.metadata?.category || 'unknown',
      priority: healthResult.metadata?.priority || 'normal',
      failureCount: healthResult.failureCount || 0,
    };

    this.addMetric(metricKey, metric);

    // Update health status map
    this.healthStatus.set(healthResult.name, {
      status: healthResult.status,
      lastUpdate: timestamp,
      failureCount: healthResult.failureCount || 0,
    });

    // Stream to real-time clients
    if (this.options.enableRealTimeStreaming) {
      this.streamUpdate('health', metric);
    }
  }

  /**
   * Record alert
   */
  recordAlert(alert) {
    const timestamp = new Date();
    const alertKey = `alert.${alert.id}`;

    const alertMetric = {
      timestamp,
      id: alert.id,
      name: alert.name,
      severity: alert.severity,
      category: alert.healthCheck?.category || 'unknown',
      priority: alert.healthCheck?.priority || 'normal',
      acknowledged: alert.acknowledged,
    };

    this.addMetric(alertKey, alertMetric);
    this.alerts.set(alert.id, alertMetric);

    // Stream to real-time clients
    if (this.options.enableRealTimeStreaming) {
      this.streamUpdate('alert', alertMetric);
    }
  }

  /**
   * Record recovery metric
   */
  recordRecoveryMetric(eventType, event) {
    const timestamp = new Date();
    const metricKey = `recovery.${eventType}`;

    const metric = {
      timestamp,
      eventType,
      executionId: event.executionId,
      workflowName: event.workflow?.name || event.execution?.workflowName,
      duration: event.execution?.duration,
      status: event.execution?.status,
      stepCount: event.execution?.steps?.length || 0,
    };

    this.addMetric(metricKey, metric);

    // Stream to real-time clients
    if (this.options.enableRealTimeStreaming) {
      this.streamUpdate('recovery', metric);
    }
  }

  /**
   * Record connection metric
   */
  recordConnectionMetric(eventType, event) {
    const timestamp = new Date();
    const metricKey = `connection.${eventType}`;

    const metric = {
      timestamp,
      eventType,
      connectionId: event.connectionId,
      connectionType: event.connection?.type,
      reconnectAttempts: event.connection?.reconnectAttempts || 0,
    };

    this.addMetric(metricKey, metric);

    // Stream to real-time clients
    if (this.options.enableRealTimeStreaming) {
      this.streamUpdate('connection', metric);
    }
  }

  /**
   * Add metric to storage
   */
  addMetric(key, metric) {
    if (!this.metrics.has(key)) {
      this.metrics.set(key, []);
    }

    const metrics = this.metrics.get(key);
    metrics.push(metric);

    // Trim old metrics based on retention period and max data points
    const cutoffTime = Date.now() - this.options.metricsRetentionPeriod;
    const filtered = metrics
      .filter(m => m.timestamp.getTime() > cutoffTime)
      .slice(-this.options.maxDataPoints);

    this.metrics.set(key, filtered);
  }

  /**
   * Start metric aggregation
   */
  startMetricAggregation() {
    this.aggregationTimer = setInterval(() => {
      try {
        this.aggregateMetrics();
      } catch (error) {
        this.logger.error('Error in metric aggregation', {
          error: error.message,
        });
      }
    }, this.options.aggregationInterval);

    this.logger.debug('Metric aggregation started');
  }

  /**
   * Aggregate metrics for dashboard display
   */
  aggregateMetrics() {
    const timestamp = new Date();
    const aggregations = new Map();

    // Aggregate health metrics
    this.aggregateHealthMetrics(aggregations, timestamp);

    // Aggregate recovery metrics
    this.aggregateRecoveryMetrics(aggregations, timestamp);

    // Aggregate connection metrics
    this.aggregateConnectionMetrics(aggregations, timestamp);

    // Aggregate system metrics
    this.aggregateSystemMetrics(aggregations, timestamp);

    // Store aggregations
    this.aggregatedMetrics.set(timestamp.getTime(), aggregations);

    // Trim old aggregations
    const cutoffTime = Date.now() - this.options.metricsRetentionPeriod;
    for (const [ts, _] of this.aggregatedMetrics) {
      if (ts < cutoffTime) {
        this.aggregatedMetrics.delete(ts);
      }
    }

    // Update trends if enabled
    if (this.options.enableTrendAnalysis) {
      this.updateTrends(aggregations, timestamp);
    }

    // Stream aggregated data
    if (this.options.enableRealTimeStreaming) {
      this.streamUpdate('aggregation', Object.fromEntries(aggregations));
    }

    this.lastUpdate = timestamp;
  }

  /**
   * Aggregate health metrics
   */
  aggregateHealthMetrics(aggregations, timestamp) {
    const healthMetrics = {
      totalChecks: 0,
      healthyChecks: 0,
      unhealthyChecks: 0,
      averageDuration: 0,
      totalDuration: 0,
      categories: {},
      priorities: {},
    };

    // Get recent health metrics (last aggregation interval)
    const since = timestamp.getTime() - this.options.aggregationInterval;

    for (const [key, metrics] of this.metrics) {
      if (key.startsWith('health.')) {
        const recentMetrics = metrics.filter(m => m.timestamp.getTime() > since);

        recentMetrics.forEach(metric => {
          healthMetrics.totalChecks++;
          healthMetrics.totalDuration += metric.duration || 0;

          if (metric.status === 'healthy') {
            healthMetrics.healthyChecks++;
          } else {
            healthMetrics.unhealthyChecks++;
          }

          // Aggregate by category
          const category = metric.category || 'unknown';
          healthMetrics.categories[category] = (healthMetrics.categories[category] || 0) + 1;

          // Aggregate by priority
          const priority = metric.priority || 'normal';
          healthMetrics.priorities[priority] = (healthMetrics.priorities[priority] || 0) + 1;
        });
      }
    }

    if (healthMetrics.totalChecks > 0) {
      healthMetrics.averageDuration = healthMetrics.totalDuration / healthMetrics.totalChecks;
    }

    aggregations.set('health', healthMetrics);
  }

  /**
   * Aggregate recovery metrics
   */
  aggregateRecoveryMetrics(aggregations, timestamp) {
    const recoveryMetrics = {
      totalRecoveries: 0,
      startedRecoveries: 0,
      completedRecoveries: 0,
      failedRecoveries: 0,
      averageDuration: 0,
      totalDuration: 0,
      workflows: {},
    };

    // Get recent recovery metrics
    const since = timestamp.getTime() - this.options.aggregationInterval;

    for (const [key, metrics] of this.metrics) {
      if (key.startsWith('recovery.')) {
        const recentMetrics = metrics.filter(m => m.timestamp.getTime() > since);

        recentMetrics.forEach(metric => {
          if (metric.eventType === 'started') {
            recoveryMetrics.startedRecoveries++;
          } else if (metric.eventType === 'completed') {
            recoveryMetrics.completedRecoveries++;
            if (metric.duration) {
              recoveryMetrics.totalDuration += metric.duration;
              recoveryMetrics.totalRecoveries++;
            }
          } else if (metric.eventType === 'failed') {
            recoveryMetrics.failedRecoveries++;
            if (metric.duration) {
              recoveryMetrics.totalDuration += metric.duration;
              recoveryMetrics.totalRecoveries++;
            }
          }

          // Aggregate by workflow
          if (metric.workflowName) {
            recoveryMetrics.workflows[metric.workflowName] =
              (recoveryMetrics.workflows[metric.workflowName] || 0) + 1;
          }
        });
      }
    }

    if (recoveryMetrics.totalRecoveries > 0) {
      recoveryMetrics.averageDuration = recoveryMetrics.totalDuration / recoveryMetrics.totalRecoveries;
    }

    aggregations.set('recovery', recoveryMetrics);
  }

  /**
   * Aggregate connection metrics
   */
  aggregateConnectionMetrics(aggregations, timestamp) {
    const connectionMetrics = {
      establishedConnections: 0,
      failedConnections: 0,
      closedConnections: 0,
      connectionTypes: {},
      totalReconnectAttempts: 0,
    };

    // Get recent connection metrics
    const since = timestamp.getTime() - this.options.aggregationInterval;

    for (const [key, metrics] of this.metrics) {
      if (key.startsWith('connection.')) {
        const recentMetrics = metrics.filter(m => m.timestamp.getTime() > since);

        recentMetrics.forEach(metric => {
          if (metric.eventType === 'established') {
            connectionMetrics.establishedConnections++;
          } else if (metric.eventType === 'failed') {
            connectionMetrics.failedConnections++;
          } else if (metric.eventType === 'closed') {
            connectionMetrics.closedConnections++;
          }

          // Aggregate by connection type
          if (metric.connectionType) {
            connectionMetrics.connectionTypes[metric.connectionType] =
              (connectionMetrics.connectionTypes[metric.connectionType] || 0) + 1;
          }

          // Sum reconnect attempts
          connectionMetrics.totalReconnectAttempts += metric.reconnectAttempts || 0;
        });
      }
    }

    aggregations.set('connection', connectionMetrics);
  }

  /**
   * Aggregate system metrics
   */
  aggregateSystemMetrics(aggregations, timestamp) {
    const systemMetrics = {
      memoryUsage: process.memoryUsage(),
      cpuUsage: process.cpuUsage(),
      uptime: process.uptime(),
      timestamp: timestamp.getTime(),
    };

    // Add system load information
    try {
      const os = require('os');
      systemMetrics.loadAverage = os.loadavg();
      systemMetrics.totalMemory = os.totalmem();
      systemMetrics.freeMemory = os.freemem();
      systemMetrics.cpuCount = os.cpus().length;
    } catch (error) {
      this.logger.warn('Could not collect system metrics', {
        error: error.message,
      });
    }

    aggregations.set('system', systemMetrics);
  }

  /**
   * Update trend analysis
   */
  updateTrends(aggregations, timestamp) {
    for (const [category, data] of aggregations) {
      if (!this.trends.has(category)) {
        this.trends.set(category, []);
      }

      const trend = this.trends.get(category);
      trend.push({
        timestamp,
        data,
      });

      // Keep only recent trend data
      const cutoffTime = Date.now() - this.options.metricsRetentionPeriod;
      this.trends.set(category,
        trend.filter(t => t.timestamp.getTime() > cutoffTime));
    }
  }

  /**
   * Set up data collection
   */
  setupDataCollection() {
    // Collect initial system state
    this.collectSystemState();

    // Set up periodic system data collection
    setInterval(() => {
      this.collectSystemState();
    }, this.options.aggregationInterval);
  }

  /**
   * Collect current system state
   */
  collectSystemState() {
    try {
      // Collect health status if available
      if (this.healthMonitor) {
        const healthData = this.healthMonitor.exportHealthData();
        this.recordSystemMetric('health_summary', {
          totalChecks: healthData.healthChecks.length,
          activeAlerts: healthData.alerts.length,
          isMonitoring: healthData.stats.isRunning,
        });
      }

      // Collect recovery status if available
      if (this.recoveryWorkflows) {
        const recoveryData = this.recoveryWorkflows.exportRecoveryData();
        this.recordSystemMetric('recovery_summary', {
          activeRecoveries: recoveryData.activeRecoveries.length,
          totalWorkflows: recoveryData.workflows.length,
          stats: recoveryData.stats,
        });
      }

      // Collect connection status if available
      if (this.connectionManager) {
        const connectionData = this.connectionManager.exportConnectionData();
        this.recordSystemMetric('connection_summary', {
          totalConnections: Object.keys(connectionData.connections).length,
          activeConnections: connectionData.stats.activeConnections,
          stats: connectionData.stats,
        });
      }

    } catch (error) {
      this.logger.error('Error collecting system state', {
        error: error.message,
      });
    }
  }

  /**
   * Record system metric
   */
  recordSystemMetric(name, data) {
    const timestamp = new Date();
    const metricKey = `system.${name}`;

    const metric = {
      timestamp,
      name,
      data,
    };

    this.addMetric(metricKey, metric);
  }

  /**
   * Stream update to real-time clients
   */
  streamUpdate(type, data) {
    const update = {
      type,
      timestamp: new Date(),
      data,
    };

    // Emit to streaming clients
    this.emit('stream:update', update);

    // Send to WebSocket clients if any
    for (const client of this.streamingClients) {
      try {
        if (client.readyState === 1) { // WebSocket.OPEN
          client.send(JSON.stringify(update));
        }
      } catch (error) {
        this.logger.warn('Error sending stream update to client', {
          error: error.message,
        });
        this.streamingClients.delete(client);
      }
    }
  }

  /**
   * Add streaming client
   */
  addStreamingClient(client) {
    this.streamingClients.add(client);

    // Send initial data
    const initialData = this.exportDashboardData();
    try {
      client.send(JSON.stringify({
        type: 'initial',
        timestamp: new Date(),
        data: initialData,
      }));
    } catch (error) {
      this.logger.warn('Error sending initial data to streaming client', {
        error: error.message,
      });
    }

    // Clean up on close
    client.on('close', () => {
      this.streamingClients.delete(client);
    });

    this.logger.debug('Added streaming client', {
      totalClients: this.streamingClients.size,
    });
  }

  /**
   * Get dashboard data in specified format
   */
  exportDashboardData(format = 'json') {
    const data = {
      timestamp: new Date(),
      lastUpdate: this.lastUpdate,
      summary: this.generateSummary(),
      health: this.exportHealthData(),
      recovery: this.exportRecoveryData(),
      connections: this.exportConnectionData(),
      system: this.exportSystemData(),
      alerts: this.exportAlertData(),
      trends: this.exportTrendData(),
    };

    switch (format.toLowerCase()) {
      case 'prometheus':
        return this.formatForPrometheus(data);
      case 'grafana':
        return this.formatForGrafana(data);
      case 'json':
      default:
        return data;
    }
  }

  /**
   * Generate summary statistics
   */
  generateSummary() {
    const now = Date.now();
    const recentWindow = now - this.options.aggregationInterval;

    let healthySystems = 0;
    let totalSystems = 0;
    let activeAlerts = 0;
    let activeRecoveries = 0;
    let activeConnections = 0;

    // Count healthy systems
    for (const [name, status] of this.healthStatus) {
      totalSystems++;
      if (status.status === 'healthy') {
        healthySystems++;
      }
    }

    // Count active alerts
    activeAlerts = Array.from(this.alerts.values())
      .filter(alert => !alert.acknowledged).length;

    // Get active recoveries from recovery workflows
    if (this.recoveryWorkflows) {
      const recoveryData = this.recoveryWorkflows.exportRecoveryData();
      activeRecoveries = recoveryData.activeRecoveries.length;
    }

    // Get active connections
    if (this.connectionManager) {
      const connectionData = this.connectionManager.exportConnectionData();
      activeConnections = connectionData.stats.activeConnections;
    }

    return {
      overallHealth: totalSystems > 0 ? (healthySystems / totalSystems) * 100 : 100,
      totalSystems,
      healthySystems,
      activeAlerts,
      activeRecoveries,
      activeConnections,
      lastUpdate: this.lastUpdate,
    };
  }

  /**
   * Export health data for dashboard
   */
  exportHealthData() {
    return {
      currentStatus: Object.fromEntries(this.healthStatus),
      recentMetrics: this.getRecentMetrics('health'),
      categories: this.getCategoryBreakdown('health'),
      priorities: this.getPriorityBreakdown('health'),
    };
  }

  /**
   * Export recovery data for dashboard
   */
  exportRecoveryData() {
    return {
      recentMetrics: this.getRecentMetrics('recovery'),
      workflowBreakdown: this.getWorkflowBreakdown(),
      successRate: this.getRecoverySuccessRate(),
    };
  }

  /**
   * Export connection data for dashboard
   */
  exportConnectionData() {
    return {
      recentMetrics: this.getRecentMetrics('connection'),
      typeBreakdown: this.getConnectionTypeBreakdown(),
      healthStatus: this.getConnectionHealthStatus(),
    };
  }

  /**
   * Export system data for dashboard
   */
  exportSystemData() {
    return {
      recentMetrics: this.getRecentMetrics('system'),
      currentState: this.getCurrentSystemState(),
    };
  }

  /**
   * Export alert data for dashboard
   */
  exportAlertData() {
    const recentAlerts = Array.from(this.alerts.values())
      .filter(alert => {
        const alertAge = Date.now() - alert.timestamp.getTime();
        return alertAge < this.options.metricsRetentionPeriod;
      })
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    return {
      recent: recentAlerts.slice(0, 50), // Last 50 alerts
      breakdown: this.getAlertBreakdown(recentAlerts),
      acknowledged: recentAlerts.filter(a => a.acknowledged).length,
      unacknowledged: recentAlerts.filter(a => !a.acknowledged).length,
    };
  }

  /**
   * Export trend data for dashboard
   */
  exportTrendData() {
    const trends = {};

    for (const [category, trendData] of this.trends) {
      trends[category] = trendData.slice(-100); // Last 100 data points
    }

    return trends;
  }

  /**
   * Helper methods for data processing
   */

  getRecentMetrics(category, limit = 100) {
    const recentMetrics = [];
    const since = Date.now() - this.options.aggregationInterval * 5; // Last 5 intervals

    for (const [key, metrics] of this.metrics) {
      if (key.startsWith(`${category}.`)) {
        const recent = metrics
          .filter(m => m.timestamp.getTime() > since)
          .slice(-limit);
        recentMetrics.push(...recent);
      }
    }

    return recentMetrics.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  getCategoryBreakdown(category) {
    const breakdown = {};
    const recentMetrics = this.getRecentMetrics(category);

    recentMetrics.forEach(metric => {
      if (metric.category) {
        breakdown[metric.category] = (breakdown[metric.category] || 0) + 1;
      }
    });

    return breakdown;
  }

  getPriorityBreakdown(category) {
    const breakdown = {};
    const recentMetrics = this.getRecentMetrics(category);

    recentMetrics.forEach(metric => {
      if (metric.priority) {
        breakdown[metric.priority] = (breakdown[metric.priority] || 0) + 1;
      }
    });

    return breakdown;
  }

  getWorkflowBreakdown() {
    const breakdown = {};
    const recoveryMetrics = this.getRecentMetrics('recovery');

    recoveryMetrics.forEach(metric => {
      if (metric.workflowName) {
        breakdown[metric.workflowName] = (breakdown[metric.workflowName] || 0) + 1;
      }
    });

    return breakdown;
  }

  getRecoverySuccessRate() {
    const recoveryMetrics = this.getRecentMetrics('recovery');
    const completed = recoveryMetrics.filter(m => m.eventType === 'completed').length;
    const failed = recoveryMetrics.filter(m => m.eventType === 'failed').length;
    const total = completed + failed;

    return total > 0 ? (completed / total) * 100 : 0;
  }

  getConnectionTypeBreakdown() {
    const breakdown = {};
    const connectionMetrics = this.getRecentMetrics('connection');

    connectionMetrics.forEach(metric => {
      if (metric.connectionType) {
        breakdown[metric.connectionType] = (breakdown[metric.connectionType] || 0) + 1;
      }
    });

    return breakdown;
  }

  getConnectionHealthStatus() {
    if (!this.connectionManager) return {};

    const connectionData = this.connectionManager.exportConnectionData();
    const healthStatus = {};

    connectionData.connections.forEach(connection => {
      healthStatus[connection.id] = {
        status: connection.health?.status || 'unknown',
        latency: connection.health?.latency,
        lastCheck: connection.health?.lastCheck,
      };
    });

    return healthStatus;
  }

  getCurrentSystemState() {
    return {
      memory: process.memoryUsage(),
      cpu: process.cpuUsage(),
      uptime: process.uptime(),
      nodeVersion: process.version,
      platform: process.platform,
      pid: process.pid,
    };
  }

  getAlertBreakdown(alerts) {
    const breakdown = {
      severity: {},
      category: {},
      priority: {},
    };

    alerts.forEach(alert => {
      breakdown.severity[alert.severity] = (breakdown.severity[alert.severity] || 0) + 1;
      breakdown.category[alert.category] = (breakdown.category[alert.category] || 0) + 1;
      breakdown.priority[alert.priority] = (breakdown.priority[alert.priority] || 0) + 1;
    });

    return breakdown;
  }

  /**
   * Format data for Prometheus
   */
  formatForPrometheus(data) {
    const metrics = [];

    // Health metrics
    metrics.push(`# HELP ruv_swarm_health_checks_total Total number of health checks`);
    metrics.push(`# TYPE ruv_swarm_health_checks_total counter`);
    metrics.push(`ruv_swarm_health_checks_total ${data.health.recentMetrics.length}`);

    // Recovery metrics
    metrics.push(`# HELP ruv_swarm_recoveries_total Total number of recoveries`);
    metrics.push(`# TYPE ruv_swarm_recoveries_total counter`);
    const recoveryTotal = data.recovery.recentMetrics.length;
    metrics.push(`ruv_swarm_recoveries_total ${recoveryTotal}`);

    // Connection metrics
    metrics.push(`# HELP ruv_swarm_connections_active Active connections`);
    metrics.push(`# TYPE ruv_swarm_connections_active gauge`);
    metrics.push(`ruv_swarm_connections_active ${data.summary.activeConnections}`);

    // Alert metrics
    metrics.push(`# HELP ruv_swarm_alerts_active Active alerts`);
    metrics.push(`# TYPE ruv_swarm_alerts_active gauge`);
    metrics.push(`ruv_swarm_alerts_active ${data.summary.activeAlerts}`);

    return metrics.join('\n');
  }

  /**
   * Format data for Grafana
   */
  formatForGrafana(data) {
    return {
      ...data,
      panels: [
        {
          title: 'System Health Overview',
          type: 'stat',
          targets: [
            { expr: 'ruv_swarm_health_checks_total', legendFormat: 'Health Checks' },
          ],
        },
        {
          title: 'Recovery Success Rate',
          type: 'stat',
          targets: [
            { expr: 'ruv_swarm_recovery_success_rate', legendFormat: 'Success Rate' },
          ],
        },
        {
          title: 'Active Connections',
          type: 'graph',
          targets: [
            { expr: 'ruv_swarm_connections_active', legendFormat: 'Active Connections' },
          ],
        },
        {
          title: 'Alert Distribution',
          type: 'piechart',
          targets: [
            { expr: 'ruv_swarm_alerts_by_severity', legendFormat: '{{severity}}' },
          ],
        },
      ],
    };
  }

  /**
   * Acknowledge alert
   */
  acknowledgeAlert(alertId, acknowledgedBy = 'system') {
    const alert = this.alerts.get(alertId);
    if (!alert) {
      throw new Error(`Alert ${alertId} not found`);
    }

    alert.acknowledged = true;
    alert.acknowledgedBy = acknowledgedBy;
    alert.acknowledgedAt = new Date();

    this.logger.info(`Alert acknowledged: ${alertId}`, {
      acknowledgedBy,
      alertName: alert.name,
    });

    // Stream update
    if (this.options.enableRealTimeStreaming) {
      this.streamUpdate('alert_acknowledged', alert);
    }
  }

  /**
   * Get monitoring statistics
   */
  getMonitoringStats() {
    return {
      metricsCount: this.metrics.size,
      totalDataPoints: Array.from(this.metrics.values())
        .reduce((sum, metrics) => sum + metrics.length, 0),
      aggregationsCount: this.aggregatedMetrics.size,
      activeAlerts: this.alerts.size,
      streamingClients: this.streamingClients.size,
      trendsCount: this.trends.size,
      lastUpdate: this.lastUpdate,
      retentionPeriod: this.options.metricsRetentionPeriod,
      aggregationInterval: this.options.aggregationInterval,
    };
  }

  /**
   * Cleanup and shutdown
   */
  async shutdown() {
    this.logger.info('Shutting down Monitoring Dashboard');

    // Clear aggregation timer
    if (this.aggregationTimer) {
      clearInterval(this.aggregationTimer);
    }

    // Close all streaming clients
    for (const client of this.streamingClients) {
      try {
        client.close();
      } catch (error) {
        this.logger.warn('Error closing streaming client', {
          error: error.message,
        });
      }
    }
    this.streamingClients.clear();

    // Clear all data
    this.metrics.clear();
    this.aggregatedMetrics.clear();
    this.alerts.clear();
    this.trends.clear();
    this.healthStatus.clear();

    this.emit('dashboard:shutdown');
  }
}

export default MonitoringDashboard;
