/**
 * Integration Test Configuration for ruv-swarm
 * Centralized configuration for all integration test scenarios
 */

const path = require('path');
const os = require('os');

const config = {
  // Test Environment Configuration
  environment: {
    nodeEnv: 'test',
    testMode: true,
    logLevel: process.env.CI ? 'error' : 'info',
    timeout: {
      suite: 60000, // 60 seconds per suite
      test: 30000, // 30 seconds per test
      hook: 5000, // 5 seconds for hooks
      lifecycle: 120000, // 2 minutes for full lifecycle tests
    },
    retries: {
      flaky: 2, // Retry flaky tests twice
      critical: 1, // Retry critical tests once
      performance: 0, // No retries for performance tests
    },
  },

  // Database Configuration
  database: {
    testDb: ':memory:',
    persistentTestDb: path.join(__dirname, '../../data/integration-test.db'),
    backupOnFailure: true,
    cleanupAfterTests: true,
  },

  // Swarm Configuration
  swarm: {
    topologies: ['mesh', 'hierarchical', 'ring', 'star'],
    maxAgents: {
      small: 10,
      medium: 50,
      large: 100,
      stress: 200,
    },
    strategies: ['parallel', 'sequential', 'adaptive', 'balanced'],
    performance: {
      targetResponseTime: 100, // 100ms
      maxMemoryUsage: 512 * 1024 * 1024, // 512MB
      minThroughput: 10, // 10 ops/second
      maxCpuUsage: 80, // 80%
    },
  },

  // Neural Agent Configuration
  neural: {
    models: [
      'adaptive',
      'pattern-recognition',
      'optimization',
      'learning',
      'research-optimized',
    ],
    training: {
      iterations: 10,
      batchSize: 5,
      learningRate: 0.01,
      convergenceThreshold: 0.95,
    },
    performance: {
      minAccuracy: 0.8,
      maxTrainingTime: 5000,
      memoryEfficiency: 0.9,
    },
  },

  // Memory Management Configuration
  memory: {
    types: ['episodic', 'semantic', 'procedural', 'collective'],
    capacity: {
      agent: 1000, // 1000 memories per agent
      swarm: 10000, // 10000 collective memories
      session: 5000, // 5000 session memories
    },
    persistence: {
      enabled: true,
      interval: 1000, // Save every second
      compression: true,
      encryption: false, // Disabled for tests
    },
  },

  // MCP Integration Configuration
  mcp: {
    protocol: 'jsonrpc',
    timeout: 5000,
    retries: 3,
    batchSize: 10,
    features: [
      'agent-management',
      'task-orchestration',
      'metrics-collection',
      'state-synchronization',
    ],
  },

  // Hook System Configuration
  hooks: {
    enabled: true,
    async: true,
    errorHandling: 'graceful',
    cascading: true,
    lifecycle: [
      'pre-agent-spawn',
      'post-agent-spawn',
      'pre-task-orchestrate',
      'post-task-complete',
      'pre-system-shutdown',
      'post-system-cleanup',
    ],
  },

  // Load Testing Configuration
  load: {
    profiles: {
      light: {
        agents: 10,
        tasks: 50,
        duration: 30000, // 30 seconds
        concurrency: 5,
      },
      medium: {
        agents: 50,
        tasks: 200,
        duration: 60000, // 60 seconds
        concurrency: 20,
      },
      heavy: {
        agents: 100,
        tasks: 500,
        duration: 120000, // 2 minutes
        concurrency: 50,
      },
      stress: {
        agents: 200,
        tasks: 1000,
        duration: 300000, // 5 minutes
        concurrency: 100,
      },
    },
    thresholds: {
      responseTime: {
        p50: 100, // 50th percentile under 100ms
        p95: 500, // 95th percentile under 500ms
        p99: 1000, // 99th percentile under 1000ms
      },
      throughput: {
        min: 10, // At least 10 ops/second
        target: 50, // Target 50 ops/second
        max: 100, // Maximum expected 100 ops/second
      },
      resources: {
        memory: 1024 * 1024 * 1024, // 1GB max
        cpu: 80, // 80% max CPU
        handles: 1000, // Max 1000 handles
      },
    },
  },

  // Resilience Testing Configuration
  resilience: {
    scenarios: {
      'component-failure': {
        type: 'random-agent-crash',
        frequency: 0.1, // 10% chance
        recovery: 'auto',
      },
      'network-partition': {
        type: 'split-brain',
        duration: 5000, // 5 seconds
        healing: 'auto',
      },
      'memory-pressure': {
        type: 'leak-simulation',
        intensity: 'medium',
        duration: 10000, // 10 seconds
      },
      'database-corruption': {
        type: 'random-corruption',
        severity: 'recoverable',
        backup: true,
      },
    },
    recovery: {
      maxTime: 5000, // 5 seconds max recovery
      retries: 3,
      backoff: 'exponential',
    },
  },

  // Performance Monitoring Configuration
  monitoring: {
    metrics: [
      'response-time',
      'throughput',
      'memory-usage',
      'cpu-usage',
      'error-rate',
      'agent-utilization',
      'task-completion-rate',
    ],
    sampling: {
      interval: 100, // Sample every 100ms
      window: 10000, // 10 second windows
      retention: 300000, // Keep 5 minutes of data
    },
    alerts: {
      responseTime: 1000, // Alert if > 1 second
      errorRate: 0.05, // Alert if > 5% errors
      memoryUsage: 0.9, // Alert if > 90% memory
      cpuUsage: 0.9, // Alert if > 90% CPU
    },
  },

  // CI/CD Configuration
  ci: {
    parallel: true,
    coverage: {
      threshold: 80, // 80% coverage required
      reporters: ['text', 'html', 'json'],
      exclude: ['test/**', 'examples/**'],
    },
    reporting: {
      formats: ['junit', 'json', 'tap'],
      artifacts: ['screenshots', 'logs', 'metrics'],
      retention: 30, // Keep 30 days of results
    },
  },

  // Platform-Specific Configuration
  platform: {
    linux: {
      maxAgents: 200,
      maxMemory: 2048 * 1024 * 1024, // 2GB
      scheduler: 'cfs',
    },
    darwin: {
      maxAgents: 100,
      maxMemory: 1024 * 1024 * 1024, // 1GB
      scheduler: 'default',
    },
    win32: {
      maxAgents: 50,
      maxMemory: 512 * 1024 * 1024, // 512MB
      scheduler: 'default',
    },
  },
};

// Apply platform-specific overrides
const platformConfig = config.platform[os.platform()];
if (platformConfig) {
  config.swarm.maxAgents.stress = Math.min(config.swarm.maxAgents.stress, platformConfig.maxAgents);
  config.swarm.performance.maxMemoryUsage = Math.min(
    config.swarm.performance.maxMemoryUsage,
    platformConfig.maxMemory,
  );
}

// Apply CI overrides
if (process.env.CI) {
  config.environment.timeout.suite *= 2; // Double timeouts in CI
  config.environment.timeout.test *= 2;
  config.environment.retries.flaky = 0; // No retries in CI
  config.swarm.maxAgents.stress = Math.min(config.swarm.maxAgents.stress, 100); // Limit agents in CI
  config.monitoring.sampling.interval = 1000; // Less frequent sampling in CI
}

// Helper functions
config.getLoadProfile = (profileName) => {
  return config.load.profiles[profileName] || config.load.profiles.light;
};

config.getTimeout = (type) => {
  return config.environment.timeout[type] || config.environment.timeout.test;
};

config.shouldRetry = (testType, attempt) => {
  const maxRetries = config.environment.retries[testType] || 0;
  return attempt < maxRetries;
};

config.getResourceLimits = () => {
  const platformLimits = config.platform[os.platform()] || {};
  return {
    maxAgents: platformLimits.maxAgents || config.swarm.maxAgents.large,
    maxMemory: platformLimits.maxMemory || config.swarm.performance.maxMemoryUsage,
    maxCpu: config.swarm.performance.maxCpuUsage,
  };
};

module.exports = config;