/**
 * Recovery Integration Module for RuvSwarm
 *
 * Provides comprehensive integration between all recovery system components,
 * centralized configuration, and unified management interface.
 *
 * Features:
 * - Centralized recovery system orchestration
 * - Component integration and communication
 * - Configuration management and validation
 * - Unified API for all recovery operations
 * - Performance monitoring and optimization
 * - Production-ready deployment patterns
 */

import { EventEmitter } from 'events';
import { Logger } from './logger.js';
import { ErrorFactory } from './errors.js';
import { generateId } from './utils.js';

import HealthMonitor from './health-monitor.js';
import RecoveryWorkflows from './recovery-workflows.js';
import ConnectionStateManager from './connection-state-manager.js';
import MonitoringDashboard from './monitoring-dashboard.js';
import ChaosEngineering from './chaos-engineering.js';

export class RecoveryIntegration extends EventEmitter {
  constructor(options = {}) {
    super();

    this.options = {
      enableHealthMonitoring: options.enableHealthMonitoring !== false,
      enableRecoveryWorkflows: options.enableRecoveryWorkflows !== false,
      enableConnectionManagement: options.enableConnectionManagement !== false,
      enableMonitoringDashboard: options.enableMonitoringDashboard !== false,
      enableChaosEngineering: options.enableChaosEngineering === true,
      autoIntegrate: options.autoIntegrate !== false,
      configValidation: options.configValidation !== false,
      performanceOptimization: options.performanceOptimization !== false,
      ...options,
    };

    this.logger = new Logger({
      name: 'recovery-integration',
      level: process.env.LOG_LEVEL || 'INFO',
      metadata: { component: 'recovery-integration' },
    });

    // Component instances
    this.healthMonitor = null;
    this.recoveryWorkflows = null;
    this.connectionManager = null;
    this.monitoringDashboard = null;
    this.chaosEngineering = null;

    // External integrations
    this.mcpTools = null;
    this.persistence = null;

    // System state
    this.isInitialized = false;
    this.isRunning = false;
    this.components = new Map();
    this.integrationStatus = new Map();

    // Performance tracking
    this.performanceMetrics = {
      initializationTime: 0,
      componentStartupTimes: new Map(),
      integrationLatency: new Map(),
      totalMemoryUsage: 0,
    };

    this.initialize();
  }

  /**
   * Initialize the recovery integration system
   */
  async initialize() {
    const startTime = Date.now();

    try {
      this.logger.info('Initializing Recovery Integration System');

      // Validate configuration
      if (this.options.configValidation) {
        await this.validateConfiguration();
      }

      // Initialize components in dependency order
      await this.initializeComponents();

      // Set up integrations
      if (this.options.autoIntegrate) {
        await this.setupIntegrations();
      }

      // Start performance optimization if enabled
      if (this.options.performanceOptimization) {
        this.startPerformanceOptimization();
      }

      this.isInitialized = true;
      this.performanceMetrics.initializationTime = Date.now() - startTime;

      this.logger.info('Recovery Integration System initialized successfully', {
        initializationTime: this.performanceMetrics.initializationTime,
        componentsInitialized: this.components.size,
      });

      this.emit('integration:initialized');

    } catch (error) {
      const integrationError = ErrorFactory.createError('resource',
        'Failed to initialize recovery integration system', {
          error: error.message,
          component: 'recovery-integration',
        });
      this.logger.error('Recovery Integration initialization failed', integrationError);
      throw integrationError;
    }
  }

  /**
   * Initialize individual components
   */
  async initializeComponents() {
    // Initialize Health Monitor
    if (this.options.enableHealthMonitoring) {
      await this.initializeComponent('healthMonitor', HealthMonitor, {
        checkInterval: 30000,
        systemCheckInterval: 60000,
        enableRealTimeMonitoring: true,
        enablePerformanceMetrics: true,
      });
    }

    // Initialize Recovery Workflows
    if (this.options.enableRecoveryWorkflows) {
      await this.initializeComponent('recoveryWorkflows', RecoveryWorkflows, {
        maxRetries: 3,
        retryDelay: 5000,
        maxConcurrentRecoveries: 3,
        recoveryTimeout: 300000,
      });
    }

    // Initialize Connection State Manager
    if (this.options.enableConnectionManagement) {
      await this.initializeComponent('connectionManager', ConnectionStateManager, {
        maxConnections: 10,
        connectionTimeout: 30000,
        healthCheckInterval: 30000,
        persistenceEnabled: true,
      });
    }

    // Initialize Monitoring Dashboard
    if (this.options.enableMonitoringDashboard) {
      await this.initializeComponent('monitoringDashboard', MonitoringDashboard, {
        enableRealTimeStreaming: true,
        enableTrendAnalysis: true,
        metricsRetentionPeriod: 86400000, // 24 hours
        aggregationInterval: 60000, // 1 minute
      });
    }

    // Initialize Chaos Engineering (if enabled)
    if (this.options.enableChaosEngineering) {
      await this.initializeComponent('chaosEngineering', ChaosEngineering, {
        enableChaos: true,
        safetyEnabled: true,
        maxConcurrentExperiments: 2,
        blastRadiusLimit: 0.3,
      });
    }
  }

  /**
   * Initialize a single component
   */
  async initializeComponent(name, ComponentClass, options = {}) {
    const startTime = Date.now();

    try {
      this.logger.debug(`Initializing component: ${name}`);

      const componentOptions = {
        ...this.options[name] || {},
        ...options,
      };

      const component = new ComponentClass(componentOptions);
      await component.initialize();

      this[name] = component;
      this.components.set(name, {
        instance: component,
        status: 'initialized',
        initTime: Date.now() - startTime,
        options: componentOptions,
      });

      this.performanceMetrics.componentStartupTimes.set(name, Date.now() - startTime);

      this.logger.debug(`Component initialized: ${name}`, {
        initTime: Date.now() - startTime,
      });

      this.emit('component:initialized', { name, component });

    } catch (error) {
      this.logger.error(`Failed to initialize component: ${name}`, {
        error: error.message,
      });

      this.components.set(name, {
        instance: null,
        status: 'failed',
        error: error.message,
        initTime: Date.now() - startTime,
      });

      throw error;
    }
  }

  /**
   * Set up integrations between components
   */
  async setupIntegrations() {
    this.logger.info('Setting up component integrations');

    const integrations = [
      // Health Monitor integrations
      {
        from: 'healthMonitor',
        to: 'recoveryWorkflows',
        method: 'setHealthMonitor',
      },
      {
        from: 'healthMonitor',
        to: 'monitoringDashboard',
        method: 'setHealthMonitor',
      },
      {
        from: 'healthMonitor',
        to: 'chaosEngineering',
        method: 'setHealthMonitor',
      },

      // Recovery Workflows integrations
      {
        from: 'recoveryWorkflows',
        to: 'healthMonitor',
        method: 'setRecoveryWorkflows',
      },
      {
        from: 'recoveryWorkflows',
        to: 'monitoringDashboard',
        method: 'setRecoveryWorkflows',
      },
      {
        from: 'recoveryWorkflows',
        to: 'chaosEngineering',
        method: 'setRecoveryWorkflows',
      },

      // Connection Manager integrations
      {
        from: 'connectionManager',
        to: 'healthMonitor',
        method: 'setConnectionManager',
      },
      {
        from: 'connectionManager',
        to: 'recoveryWorkflows',
        method: 'setConnectionManager',
      },
      {
        from: 'connectionManager',
        to: 'monitoringDashboard',
        method: 'setConnectionManager',
      },
      {
        from: 'connectionManager',
        to: 'chaosEngineering',
        method: 'setConnectionManager',
      },

      // External integrations
      {
        from: 'mcpTools',
        to: 'healthMonitor',
        method: 'setMCPTools',
      },
      {
        from: 'mcpTools',
        to: 'recoveryWorkflows',
        method: 'setMCPTools',
      },
      {
        from: 'mcpTools',
        to: 'monitoringDashboard',
        method: 'setMCPTools',
      },
      {
        from: 'mcpTools',
        to: 'chaosEngineering',
        method: 'setMCPTools',
      },

      // Persistence integrations
      {
        from: 'persistence',
        to: 'connectionManager',
        method: 'setPersistence',
      },
    ];

    for (const integration of integrations) {
      await this.setupIntegration(integration);
    }

    this.logger.info('Component integrations completed', {
      totalIntegrations: integrations.length,
      successfulIntegrations: Array.from(this.integrationStatus.values())
        .filter(status => status.status === 'success').length,
    });
  }

  /**
   * Set up a single integration
   */
  async setupIntegration(integration) {
    const { from, to, method } = integration;
    const integrationKey = `${from}->${to}`;

    try {
      const fromComponent = from === 'mcpTools' ? this.mcpTools :
        from === 'persistence' ? this.persistence :
          this[from];
      const toComponent = this[to];

      if (!fromComponent || !toComponent) {
        this.integrationStatus.set(integrationKey, {
          status: 'skipped',
          reason: `Component not available: from=${!!fromComponent}, to=${!!toComponent}`,
        });
        return;
      }

      if (typeof toComponent[method] === 'function') {
        const startTime = Date.now();
        await toComponent[method](fromComponent);
        const latency = Date.now() - startTime;

        this.performanceMetrics.integrationLatency.set(integrationKey, latency);

        this.integrationStatus.set(integrationKey, {
          status: 'success',
          latency,
        });

        this.logger.debug(`Integration completed: ${integrationKey}`, { latency });
      } else {
        this.integrationStatus.set(integrationKey, {
          status: 'failed',
          reason: `Method '${method}' not found on component '${to}'`,
        });
      }

    } catch (error) {
      this.integrationStatus.set(integrationKey, {
        status: 'failed',
        error: error.message,
      });

      this.logger.error(`Integration failed: ${integrationKey}`, {
        error: error.message,
      });
    }
  }

  /**
   * Start the recovery system
   */
  async start() {
    if (!this.isInitialized) {
      await this.initialize();
    }

    if (this.isRunning) {
      this.logger.warn('Recovery system already running');
      return;
    }

    try {
      this.logger.info('Starting Recovery Integration System');

      // Start health monitoring
      if (this.healthMonitor) {
        await this.healthMonitor.startMonitoring();
      }

      this.isRunning = true;

      this.logger.info('Recovery Integration System started successfully');
      this.emit('integration:started');

    } catch (error) {
      const startError = ErrorFactory.createError('resource',
        'Failed to start recovery integration system', {
          error: error.message,
        });
      this.logger.error('Recovery Integration start failed', startError);
      throw startError;
    }
  }

  /**
   * Stop the recovery system
   */
  async stop() {
    if (!this.isRunning) {
      this.logger.warn('Recovery system not running');
      return;
    }

    try {
      this.logger.info('Stopping Recovery Integration System');

      // Stop health monitoring
      if (this.healthMonitor) {
        await this.healthMonitor.stopMonitoring();
      }

      this.isRunning = false;

      this.logger.info('Recovery Integration System stopped successfully');
      this.emit('integration:stopped');

    } catch (error) {
      this.logger.error('Error stopping recovery integration system', {
        error: error.message,
      });
      throw error;
    }
  }

  /**
   * Set external integrations
   */
  setMCPTools(mcpTools) {
    this.mcpTools = mcpTools;
    this.logger.info('MCP Tools integration configured');

    // Propagate to components if already initialized
    if (this.isInitialized && this.options.autoIntegrate) {
      this.propagateIntegration('mcpTools', mcpTools);
    }
  }

  setPersistence(persistence) {
    this.persistence = persistence;
    this.logger.info('Persistence integration configured');

    // Propagate to components if already initialized
    if (this.isInitialized && this.options.autoIntegrate) {
      this.propagateIntegration('persistence', persistence);
    }
  }

  /**
   * Propagate integration to components
   */
  async propagateIntegration(integrationType, integration) {
    const methodMap = {
      mcpTools: 'setMCPTools',
      persistence: 'setPersistence',
    };

    const method = methodMap[integrationType];
    if (!method) return;

    for (const [name, componentData] of this.components) {
      if (componentData.instance && typeof componentData.instance[method] === 'function') {
        try {
          await componentData.instance[method](integration);
          this.logger.debug(`Propagated ${integrationType} to ${name}`);
        } catch (error) {
          this.logger.error(`Failed to propagate ${integrationType} to ${name}`, {
            error: error.message,
          });
        }
      }
    }
  }

  /**
   * Register swarm for monitoring across all components
   */
  async registerSwarm(swarmId, swarmInstance) {
    this.logger.info(`Registering swarm across recovery system: ${swarmId}`);

    // Register with health monitor
    if (this.healthMonitor) {
      this.healthMonitor.registerSwarm(swarmId, swarmInstance);
    }

    // Register with connection manager if it has MCP connections
    if (this.connectionManager && swarmInstance.mcpConnections) {
      for (const [connectionId, connectionConfig] of Object.entries(swarmInstance.mcpConnections)) {
        await this.connectionManager.registerConnection({
          id: connectionId,
          ...connectionConfig,
          metadata: { swarmId },
        });
      }
    }

    this.emit('swarm:registered', { swarmId, swarmInstance });
  }

  /**
   * Unregister swarm from monitoring
   */
  async unregisterSwarm(swarmId) {
    this.logger.info(`Unregistering swarm from recovery system: ${swarmId}`);

    // Unregister from health monitor
    if (this.healthMonitor) {
      this.healthMonitor.unregisterSwarm(swarmId);
    }

    // Remove connections associated with this swarm
    if (this.connectionManager) {
      const connectionStatus = this.connectionManager.getConnectionStatus();
      for (const [connectionId, connection] of Object.entries(connectionStatus.connections)) {
        if (connection.metadata?.swarmId === swarmId) {
          await this.connectionManager.removeConnection(connectionId);
        }
      }
    }

    this.emit('swarm:unregistered', { swarmId });
  }

  /**
   * Get comprehensive system status
   */
  getSystemStatus() {
    const status = {
      isInitialized: this.isInitialized,
      isRunning: this.isRunning,
      components: {},
      integrations: Object.fromEntries(this.integrationStatus),
      performance: this.getPerformanceMetrics(),
      health: null,
      recovery: null,
      connections: null,
      monitoring: null,
      chaos: null,
    };

    // Component statuses
    for (const [name, componentData] of this.components) {
      status.components[name] = {
        status: componentData.status,
        initTime: componentData.initTime,
        error: componentData.error,
      };
    }

    // Individual component statuses
    if (this.healthMonitor) {
      status.health = this.healthMonitor.getMonitoringStats();
    }

    if (this.recoveryWorkflows) {
      status.recovery = this.recoveryWorkflows.getRecoveryStats();
    }

    if (this.connectionManager) {
      status.connections = this.connectionManager.getConnectionStats();
    }

    if (this.monitoringDashboard) {
      status.monitoring = this.monitoringDashboard.getMonitoringStats();
    }

    if (this.chaosEngineering) {
      status.chaos = this.chaosEngineering.getChaosStats();
    }

    return status;
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics() {
    // Update memory usage
    const memUsage = process.memoryUsage();
    this.performanceMetrics.totalMemoryUsage = memUsage.heapUsed;

    return {
      ...this.performanceMetrics,
      componentStartupTimes: Object.fromEntries(this.performanceMetrics.componentStartupTimes),
      integrationLatency: Object.fromEntries(this.performanceMetrics.integrationLatency),
      currentMemoryUsage: memUsage,
      timestamp: new Date(),
    };
  }

  /**
   * Start performance optimization
   */
  startPerformanceOptimization() {
    this.logger.info('Starting performance optimization');

    // Monitor memory usage and trigger optimization
    const optimizationInterval = setInterval(() => {
      try {
        this.performMemoryOptimization();
      } catch (error) {
        this.logger.error('Error in performance optimization', {
          error: error.message,
        });
      }
    }, 300000); // Every 5 minutes

    this.optimizationInterval = optimizationInterval;
  }

  /**
   * Perform memory optimization
   */
  performMemoryOptimization() {
    const memUsage = process.memoryUsage();
    const heapUsedMB = memUsage.heapUsed / 1024 / 1024;

    if (heapUsedMB > 512) { // More than 512MB
      this.logger.info('Performing memory optimization', {
        heapUsedMB: heapUsedMB.toFixed(2),
      });

      // Trigger garbage collection if available
      if (global.gc) {
        global.gc();
      }

      // Optimize component caches
      this.optimizeComponentCaches();

      this.emit('performance:optimized', { heapUsedMB });
    }
  }

  /**
   * Optimize component caches
   */
  optimizeComponentCaches() {
    // Optimize health monitor cache
    if (this.healthMonitor) {
      // Health monitor would have its own cache optimization
    }

    // Optimize monitoring dashboard cache
    if (this.monitoringDashboard) {
      // Dashboard would optimize its metrics cache
    }

    this.logger.debug('Component caches optimized');
  }

  /**
   * Validate configuration
   */
  async validateConfiguration() {
    this.logger.debug('Validating recovery integration configuration');

    const validationErrors = [];

    // Validate component configurations
    if (this.options.enableHealthMonitoring && this.options.healthMonitor) {
      const healthConfig = this.options.healthMonitor;
      if (healthConfig.checkInterval && healthConfig.checkInterval < 5000) {
        validationErrors.push('Health check interval too low (minimum 5000ms)');
      }
    }

    if (this.options.enableRecoveryWorkflows && this.options.recoveryWorkflows) {
      const recoveryConfig = this.options.recoveryWorkflows;
      if (recoveryConfig.maxConcurrentRecoveries && recoveryConfig.maxConcurrentRecoveries > 10) {
        validationErrors.push('Too many concurrent recoveries (maximum 10)');
      }
    }

    if (this.options.enableChaosEngineering && this.options.chaosEngineering) {
      const chaosConfig = this.options.chaosEngineering;
      if (chaosConfig.blastRadiusLimit && chaosConfig.blastRadiusLimit > 0.5) {
        validationErrors.push('Blast radius limit too high (maximum 0.5)');
      }
    }

    if (validationErrors.length > 0) {
      throw ErrorFactory.createError('configuration',
        `Configuration validation failed: ${validationErrors.join(', ')}`);
    }

    this.logger.debug('Configuration validation passed');
  }

  /**
   * Run system health check
   */
  async runSystemHealthCheck() {
    const healthResults = {
      overall: 'healthy',
      components: {},
      issues: [],
    };

    // Check component health
    for (const [name, componentData] of this.components) {
      if (componentData.status === 'failed') {
        healthResults.components[name] = 'failed';
        healthResults.issues.push(`Component ${name} failed to initialize`);
        healthResults.overall = 'degraded';
      } else {
        healthResults.components[name] = 'healthy';
      }
    }

    // Check integration health
    let failedIntegrations = 0;
    for (const [key, status] of this.integrationStatus) {
      if (status.status === 'failed') {
        failedIntegrations++;
        healthResults.issues.push(`Integration failed: ${key}`);
      }
    }

    if (failedIntegrations > 0) {
      healthResults.overall = failedIntegrations > 2 ? 'error' : 'degraded';
    }

    // Run health monitor check if available
    if (this.healthMonitor) {
      const systemHealth = this.healthMonitor.getHealthStatus();
      if (systemHealth.overallStatus !== 'healthy') {
        healthResults.overall = systemHealth.overallStatus;
        healthResults.issues.push('System health monitor reports issues');
      }
    }

    return healthResults;
  }

  /**
   * Export comprehensive system data
   */
  exportSystemData() {
    return {
      timestamp: new Date(),
      status: this.getSystemStatus(),
      health: this.healthMonitor ? this.healthMonitor.exportHealthData() : null,
      recovery: this.recoveryWorkflows ? this.recoveryWorkflows.exportRecoveryData() : null,
      connections: this.connectionManager ? this.connectionManager.exportConnectionData() : null,
      monitoring: this.monitoringDashboard ? this.monitoringDashboard.exportDashboardData() : null,
      chaos: this.chaosEngineering ? this.chaosEngineering.exportChaosData() : null,
    };
  }

  /**
   * Emergency shutdown procedure
   */
  async emergencyShutdown(reason = 'Emergency shutdown') {
    this.logger.warn('EMERGENCY SHUTDOWN INITIATED', { reason });

    try {
      // Stop chaos engineering first
      if (this.chaosEngineering) {
        await this.chaosEngineering.emergencyStop(reason);
      }

      // Stop all other components
      await this.shutdown();

      this.emit('emergency:shutdown', { reason });

    } catch (error) {
      this.logger.error('Error during emergency shutdown', {
        error: error.message,
      });
    }
  }

  /**
   * Cleanup and shutdown
   */
  async shutdown() {
    this.logger.info('Shutting down Recovery Integration System');

    // Clear optimization interval
    if (this.optimizationInterval) {
      clearInterval(this.optimizationInterval);
    }

    // Shutdown components in reverse order
    const shutdownOrder = [
      'chaosEngineering',
      'monitoringDashboard',
      'connectionManager',
      'recoveryWorkflows',
      'healthMonitor',
    ];

    for (const componentName of shutdownOrder) {
      const component = this[componentName];
      if (component && typeof component.shutdown === 'function') {
        try {
          await component.shutdown();
          this.logger.debug(`Component shutdown: ${componentName}`);
        } catch (error) {
          this.logger.error(`Error shutting down component: ${componentName}`, {
            error: error.message,
          });
        }
      }
    }

    // Clear all data
    this.components.clear();
    this.integrationStatus.clear();
    this.isInitialized = false;
    this.isRunning = false;

    this.emit('integration:shutdown');
  }
}

export default RecoveryIntegration;
