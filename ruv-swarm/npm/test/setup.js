/**
 * Test setup and configuration
 */

// Set test environment
process.env.NODE_ENV = 'test';

// Suppress console logs during tests unless DEBUG is set
if (!process.env.DEBUG) {
  global.console = {
    ...console,
    log: jest.fn(),
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    // Keep error for debugging failed tests
    error: console.error,
  };
}

// Global test utilities
global.testUtils = {
  // Wait for async operations
  wait: (ms) => new Promise(resolve => setTimeout(resolve, ms)),

  // Create mock data
  createMockSwarmConfig: (overrides = {}) => ({
    name: 'test-swarm',
    topology: 'mesh',
    strategy: 'balanced',
    maxAgents: 10,
    enableCognitiveDiversity: true,
    enableNeuralAgents: true,
    ...overrides,
  }),

  createMockAgentConfig: (overrides = {}) => ({
    type: 'researcher',
    name: 'test-agent',
    capabilities: ['research', 'analysis'],
    enableNeuralNetwork: true,
    ...overrides,
  }),

  createMockTaskConfig: (overrides = {}) => ({
    description: 'Test task',
    priority: 'medium',
    dependencies: [],
    maxAgents: null,
    estimatedDuration: null,
    requiredCapabilities: [],
    ...overrides,
  }),
};

// Mock WebAssembly if not available
if (typeof WebAssembly === 'undefined') {
  global.WebAssembly = {
    validate: jest.fn(() => false),
    instantiate: jest.fn(() => Promise.resolve({
      module: {},
      instance: {
        exports: {},
      },
    })),
    Module: jest.fn(),
    Instance: jest.fn(),
    Memory: jest.fn(() => ({ buffer: new ArrayBuffer(1024) })),
    Table: jest.fn(),
  };
}

// Mock performance API if not available
if (typeof performance === 'undefined') {
  global.performance = {
    now: jest.fn(() => Date.now()),
    memory: {
      usedJSHeapSize: 1000000,
      totalJSHeapSize: 2000000,
      jsHeapSizeLimit: 4000000,
    },
  };
}

// Clean up function for tests
global.cleanupTest = async() => {
  // Reset global state
  if (global._ruvSwarmInstance) {
    if (global._ruvSwarmInstance.persistence) {
      global._ruvSwarmInstance.persistence.close();
    }
    global._ruvSwarmInstance = null;
  }
  global._ruvSwarmInitialized = 0;

  // Clear all mocks
  jest.clearAllMocks();
};

// Add custom matchers
expect.extend({
  toBeWithinRange(received, floor, ceiling) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () => `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    }
    return {
      message: () => `expected ${received} to be within range ${floor} - ${ceiling}`,
      pass: false,
    };

  },

  toHaveValidId(received) {
    const pass = typeof received === 'string' &&
                 received.length > 0 &&
                 (received.includes('-') || received.includes('_'));
    if (pass) {
      return {
        message: () => `expected ${received} not to be a valid ID`,
        pass: true,
      };
    }
    return {
      message: () => `expected ${received} to be a valid ID (string with separator)`,
      pass: false,
    };

  },
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (error) => {
  console.error('Unhandled promise rejection in test:', error);
});

// Increase timeout for CI environments
if (process.env.CI) {
  jest.setTimeout(60000);
}