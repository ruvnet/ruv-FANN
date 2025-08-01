/**
 * Vitest Test Setup
 * Global configuration and mocks for Vitest tests
 */

import { vi } from 'vitest';

// WebAssembly polyfill for Node.js environments that might not have it
if (typeof globalThis.WebAssembly === 'undefined') {
  globalThis.WebAssembly = {
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
          memory: new globalThis.WebAssembly.Memory({ initial: 1 }),
          ...(imports?.env || {}),
        };
      }
    },
    instantiate: vi.fn().mockResolvedValue({
      instance: new globalThis.WebAssembly.Instance(),
      module: new globalThis.WebAssembly.Module(),
    }),
    instantiateStreaming: vi.fn().mockResolvedValue({
      instance: new globalThis.WebAssembly.Instance(),
      module: new globalThis.WebAssembly.Module(),
    }),
  };
}

// Mock worker_threads for environments that don't support it
vi.mock('worker_threads', () => ({
  Worker: vi.fn(),
  isMainThread: true,
  parentPort: null,
  workerData: null,
  MessageChannel: vi.fn(),
  MessagePort: vi.fn(),
  moveMessagePortToContext: vi.fn(),
  receiveMessageOnPort: vi.fn(),
  threadId: 0,
}), { virtual: true });

// Mock better-sqlite3 for tests that don't need real database
vi.mock('better-sqlite3', () => {
  return vi.fn().mockImplementation(() => ({
    prepare: vi.fn().mockReturnValue({
      run: vi.fn(),
      get: vi.fn(),
      all: vi.fn().mockReturnValue([]),
      iterate: vi.fn(),
    }),
    exec: vi.fn(),
    close: vi.fn(),
    transaction: vi.fn().mockReturnValue(() => {}),
    pragma: vi.fn(),
  }));
}, { virtual: true });

// Mock UUID generation for consistent test results
vi.mock('uuid', () => ({
  v4: vi.fn(() => `mock-uuid-${ Date.now() }-${ Math.random().toString(36).substr(2, 9)}`),
}), { virtual: true });

// Mock WebSocket for MCP tests
vi.mock('ws', () => ({
  default: vi.fn().mockImplementation(() => ({
    on: vi.fn(),
    send: vi.fn(),
    close: vi.fn(),
    readyState: 1, // OPEN
    CONNECTING: 0,
    OPEN: 1,
    CLOSING: 2,
    CLOSED: 3,
  })),
  WebSocketServer: vi.fn().mockImplementation(() => ({
    on: vi.fn(),
    close: vi.fn(),
  })),
}), { virtual: true });

// Performance polyfill for older Node.js versions
if (typeof globalThis.performance === 'undefined') {
  globalThis.performance = {
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

// Global test utilities
globalThis.testUtils = {
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
      initialize: vi.fn().mockReturnValue(true),
      getVersion: vi.fn().mockReturnValue('0.2.0'),
      createAgent: vi.fn().mockReturnValue('mock-agent'),
      createSwarm: vi.fn().mockReturnValue('mock-swarm'),
      getMemoryUsage: vi.fn().mockReturnValue({ heapUsed: 1024, heapTotal: 2048 }),
    },
  }),
};

// Environment detection
globalThis.testEnv = {
  isCI: process.env.CI === 'true',
  isGitHub: process.env.GITHUB_ACTIONS === 'true',
  nodeVersion: process.version,
  platform: process.platform,
  arch: process.arch,
};

console.log('Vitest setup completed successfully');
console.log('Test environment:', globalThis.testEnv);
console.log('WebAssembly support:', typeof globalThis.WebAssembly !== 'undefined');
