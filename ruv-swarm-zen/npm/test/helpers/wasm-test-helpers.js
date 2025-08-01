/**
 * WASM Test Helpers - TDD London Style
 * Provides inline mocks for WASM functionality following TDD London principles
 */

/**
 * Create a mock WASM module following TDD London patterns
 * @param {Object} overrides - Custom mock implementations
 * @returns {Object} Mock WASM module
 */
export function createMockWasmModule(overrides = {}) {
  const defaultMocks = {
    init: jest.fn().mockResolvedValue(undefined),
    createSwarm: jest.fn().mockReturnValue(1),
    addAgent: jest.fn().mockReturnValue(1),
    assignTask: jest.fn(),
    getState: jest.fn().mockReturnValue({
      agents: new Map(),
      tasks: new Map(),
      topology: 'mesh',
      connections: [],
      metrics: {
        totalTasks: 0,
        completedTasks: 0,
        failedTasks: 0,
        averageCompletionTime: 0,
        agentUtilization: new Map(),
        throughput: 0,
      },
    }),
    destroy: jest.fn(),
    detectSIMDSupport: jest.fn().mockReturnValue(true),
    getVersion: jest.fn().mockReturnValue('1.0.0'),
    getMemoryUsage: jest.fn().mockReturnValue({
      heapUsed: 1024,
      heapTotal: 4096,
    }),
  };

  return {
    ...defaultMocks,
    ...overrides,
  };
}

/**
 * Create a mock WASM loader for testing
 * @param {Object} moduleOverrides - Custom module implementations
 * @returns {Object} Mock WASM loader
 */
export function createMockWasmLoader(moduleOverrides = {}) {
  const mockModule = createMockWasmModule(moduleOverrides);

  return {
    loadModule: jest.fn().mockResolvedValue({
      exports: mockModule,
    }),
    cleanup: jest.fn().mockResolvedValue(undefined),
    isLoaded: jest.fn().mockReturnValue(true),
  };
}

/**
 * Create a spy for WASM module interactions
 * Follows TDD London principle of behavior verification
 * @returns {Object} Spy object with verification methods
 */
export function createWasmModuleSpy() {
  const interactions = [];

  const recordInteraction = (method, args, result) => {
    interactions.push({
      method,
      args,
      result,
      timestamp: Date.now(),
    });
  };

  const mockModule = {
    init: jest.fn((...args) => {
      const result = Promise.resolve(undefined);
      recordInteraction('init', args, result);
      return result;
    }),
    createSwarm: jest.fn((...args) => {
      const result = 1;
      recordInteraction('createSwarm', args, result);
      return result;
    }),
    addAgent: jest.fn((...args) => {
      const result = 1;
      recordInteraction('addAgent', args, result);
      return result;
    }),
    assignTask: jest.fn((...args) => {
      recordInteraction('assignTask', args, undefined);
    }),
    getState: jest.fn((...args) => {
      const result = {
        agents: new Map(),
        tasks: new Map(),
        topology: 'mesh',
        connections: [],
        metrics: {
          totalTasks: 0,
          completedTasks: 0,
          failedTasks: 0,
          averageCompletionTime: 0,
          agentUtilization: new Map(),
          throughput: 0,
        },
      };
      recordInteraction('getState', args, result);
      return result;
    }),
    destroy: jest.fn((...args) => {
      recordInteraction('destroy', args, undefined);
    }),
  };

  return {
    module: mockModule,
    getInteractions: () => [...interactions],
    verifyInteractionSequence: (expectedSequence) => {
      const actualSequence = interactions.map(i => i.method);
      return expectedSequence.every((method, index) =>
        actualSequence[index] === method,
      );
    },
    verifyNoUnexpectedInteractions: (allowedMethods) => {
      return interactions.every(i => allowedMethods.includes(i.method));
    },
    reset: () => {
      interactions.length = 0;
      Object.values(mockModule).forEach(fn => {
        if (fn.mockClear) fn.mockClear();
      });
    },
  };
}

/**
 * Mock WASM instantiation for tests
 * @param {string} wasmPath - Path to WASM file (ignored in mock)
 * @param {Object} imports - Import object (ignored in mock)
 * @returns {Promise<Object>} Mock WASM instance
 */
export async function mockInstantiateWasm(wasmPath, imports) {
  return {
    instance: {
      exports: createMockWasmModule(),
    },
    module: {},
  };
}

/**
 * Create a test double for WASM module with verification
 * Following TDD London's emphasis on behavior verification
 */
export class WasmModuleTestDouble {
  constructor() {
    this.expectations = [];
    this.actualCalls = [];
    this.stubs = new Map();
  }

  expect(method, ...args) {
    this.expectations.push({ method, args });
    return this;
  }

  returns(value) {
    const lastExpectation = this.expectations[this.expectations.length - 1];
    this.stubs.set(
      `${lastExpectation.method}-${JSON.stringify(lastExpectation.args)}`,
      value,
    );
    return this;
  }

  createModule() {
    const testDouble = this;
    const methods = ['init', 'createSwarm', 'addAgent', 'assignTask', 'getState', 'destroy'];

    const module = {};
    methods.forEach(method => {
      module[method] = jest.fn((...args) => {
        testDouble.actualCalls.push({ method, args });
        const key = `${method}-${JSON.stringify(args)}`;
        return testDouble.stubs.get(key) ?? undefined;
      });
    });

    return module;
  }

  verify() {
    const unmetExpectations = this.expectations.filter(expectation => {
      return !this.actualCalls.some(call =>
        call.method === expectation.method &&
        JSON.stringify(call.args) === JSON.stringify(expectation.args),
      );
    });

    if (unmetExpectations.length > 0) {
      throw new Error(
        `Unmet expectations:\n${unmetExpectations.map(e =>
          `  - ${e.method}(${e.args.join(', ')})`,
        ).join('\n')}`,
      );
    }
  }

  reset() {
    this.expectations = [];
    this.actualCalls = [];
    this.stubs.clear();
  }
}
