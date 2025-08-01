/**
 * Jest Test Setup
 * Global configuration and mocks for all tests
 */

// Global test configuration
global.console = {
  ...console,
  // Uncomment to suppress console output during tests
  // log: jest.fn(),
  // debug: jest.fn(),
  // info: jest.fn(),
  // warn: jest.fn(),
  error: console.error, // Keep errors visible
};

// WebAssembly polyfill for Node.js environments that might not have it
if (typeof global.WebAssembly === 'undefined') {
  global.WebAssembly = {
    Memory: class MockMemory {
      constructor(descriptor) {
        this.descriptor = descriptor;
        this.buffer = new ArrayBuffer(descriptor.initial * 65536);
      }
    },
    Module: class MockModule {
      constructor() {}
    },
    Instance: class MockInstance {
      constructor(module, imports) {
        this.exports = {
          memory: new global.WebAssembly.Memory({ initial: 1 }),
          ...(imports?.env || {}),
        };
      }
    },
    instantiate: jest.fn().mockResolvedValue({
      instance: new global.WebAssembly.Instance(),
      module: new global.WebAssembly.Module(),
    }),
    instantiateStreaming: jest.fn().mockResolvedValue({
      instance: new global.WebAssembly.Instance(),
      module: new global.WebAssembly.Module(),
    }),
  };
}

// Mock worker_threads for environments that don't support it
jest.mock('worker_threads', () => ({
  Worker: jest.fn(),
  isMainThread: true,
  parentPort: null,
  workerData: null,
  MessageChannel: jest.fn(),
  MessagePort: jest.fn(),
  moveMessagePortToContext: jest.fn(),
  receiveMessageOnPort: jest.fn(),
  threadId: 0,
}), { virtual: true });

// Mock fs/promises for Node.js compatibility
jest.mock('fs/promises', () => ({
  readFile: jest.fn(),
  writeFile: jest.fn(),
  mkdir: jest.fn(),
  readdir: jest.fn(),
  stat: jest.fn(),
  access: jest.fn(),
}), { virtual: true });

// Mock better-sqlite3 for tests that don't need real database
jest.mock('better-sqlite3', () => {
  return jest.fn().mockImplementation(() => ({
    prepare: jest.fn().mockReturnValue({
      run: jest.fn(),
      get: jest.fn(),
      all: jest.fn().mockReturnValue([]),
      iterate: jest.fn(),
    }),
    exec: jest.fn(),
    close: jest.fn(),
    transaction: jest.fn().mockReturnValue(() => {}),
    pragma: jest.fn(),
  }));
}, { virtual: true });

// Mock UUID generation for consistent test results
jest.mock('uuid', () => ({
  v4: jest.fn(() => `mock-uuid-${ Date.now() }-${ Math.random().toString(36).substr(2, 9)}`),
}), { virtual: true });

// Mock WebSocket for MCP tests
jest.mock('ws', () => ({
  WebSocket: jest.fn().mockImplementation(() => ({
    on: jest.fn(),
    send: jest.fn(),
    close: jest.fn(),
    readyState: 1, // OPEN
    CONNECTING: 0,
    OPEN: 1,
    CLOSING: 2,
    CLOSED: 3,
  })),
  WebSocketServer: jest.fn().mockImplementation(() => ({
    on: jest.fn(),
    close: jest.fn(),
  })),
}), { virtual: true });

// Performance polyfill for older Node.js versions
if (typeof global.performance === 'undefined') {
  global.performance = {
    now: () => Date.now(),
    mark: () => {},
    measure: () => {},
    getEntries: () => [],
    getEntriesByName: () => [],
    getEntriesByType: () => [],
    clearMarks: () => {},
    clearMeasures: () => {},
  };
}

// Set up default timeouts
jest.setTimeout(30000);

// Global test utilities
global.testUtils = {
  /**
   * Wait for a specific amount of time
   */
  wait: (ms) => new Promise(resolve => setTimeout(resolve, ms)),

  /**
   * Create a mock agent configuration
   */
  createMockAgent: (overrides = {}) => ({
    id: `mock-agent-${ Date.now()}`,
    type: 'researcher',
    name: 'test-agent',
    capabilities: ['research', 'analysis'],
    status: 'idle',
    ...overrides,
  }),

  /**
   * Create a mock swarm configuration
   */
  createMockSwarm: (overrides = {}) => ({
    id: `mock-swarm-${ Date.now()}`,
    name: 'test-swarm',
    topology: 'mesh',
    maxAgents: 10,
    strategy: 'balanced',
    ...overrides,
  }),

  /**
   * Create a mock task configuration
   */
  createMockTask: (overrides = {}) => ({
    id: `mock-task-${ Date.now()}`,
    description: 'Test task',
    priority: 'medium',
    status: 'pending',
    dependencies: [],
    ...overrides,
  }),

  /**
   * Mock WASM module loader
   */
  createMockWasmModule: () => ({
    exports: {
      initialize: jest.fn().mockReturnValue(true),
      getVersion: jest.fn().mockReturnValue('0.2.0'),
      createAgent: jest.fn().mockReturnValue('mock-agent'),
      createSwarm: jest.fn().mockReturnValue('mock-swarm'),
      getMemoryUsage: jest.fn().mockReturnValue({ heapUsed: 1024, heapTotal: 2048 }),
    },
  }),
};

// Environment detection
global.testEnv = {
  isCI: process.env.CI === 'true',
  isGitHub: process.env.GITHUB_ACTIONS === 'true',
  nodeVersion: process.version,
  platform: process.platform,
  arch: process.arch,
};

// Suppress deprecation warnings in tests
process.on('warning', (warning) => {
  if (warning.name === 'DeprecationWarning') {
    return; // Suppress deprecation warnings
  }
  console.warn(warning);
});

// Handle unhandled promise rejections in tests
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  // Don't exit the process in tests, just log
});

// Cleanup after each test
afterEach(() => {
  // Clear all timers
  jest.clearAllTimers();

  // Clear all mocks
  jest.clearAllMocks();

  // Reset modules
  jest.resetModules();
});

// Global cleanup
afterAll(() => {
  // Final cleanup
  jest.restoreAllMocks();
});

console.log('Jest setup completed successfully');
console.log('Test environment:', global.testEnv);
console.log('WebAssembly support:', typeof global.WebAssembly !== 'undefined');
