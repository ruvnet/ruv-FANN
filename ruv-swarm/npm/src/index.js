/**
 * @ruv/swarm - High-performance neural network swarm orchestration in WebAssembly
 * Enhanced version with progressive WASM loading and full feature set
 */

// Re-export the enhanced implementation
export * from './index-enhanced.js';

/* Legacy exports for backward compatibility */
import path from 'path';
import { promises as fs } from 'fs';

// Lazy-loaded WASM module
let wasmModule = null;
let wasmInstance = null;

/**
 * WASM loader with feature detection and caching
 */
class WASMLoader {
  constructor(options = {}) {
    this.useSIMD = options.useSIMD && this.detectSIMDSupport();
    this.wasmPath = options.wasmPath || path.join(new URL('.', import.meta.url).pathname, '..', 'wasm');
    this.debug = options.debug || false;
  }

  detectSIMDSupport() {
    try {
      // WebAssembly SIMD feature detection
      if (typeof WebAssembly !== 'undefined' && WebAssembly.validate) {
        // Test SIMD instruction
        const simdTest = new Uint8Array([
          0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
          0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
          0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
          0x41, 0x00, 0xfd, 0x0f, 0x26, 0x0b,
        ]);
        return WebAssembly.validate(simdTest);
      }
    } catch (e) {
      if (this.debug) {
        console.warn('SIMD detection failed:', e);
      }
    }
    return false;
  }

  async loadModule() {
    if (wasmModule) {
      return wasmModule;
    }

    // Use the generated WASM bindings directly (ES module import)
    try {
      const wasmJsPath = path.join(this.wasmPath, 'ruv_swarm_wasm.js');
      const wasmBindings = await import(path.resolve(wasmJsPath));
      wasmModule = wasmBindings;
      return wasmModule;
    } catch (error) {
      if (this.debug) {
        console.error('Failed to load WASM bindings:', error);
      }
    }

    // Fallback to manual loading
    const moduleFile = this.useSIMD ? 'ruv_swarm_simd.wasm' : 'ruv_swarm_wasm_bg.wasm';
    const wasmFilePath = path.join(this.wasmPath, moduleFile);

    try {
      let wasmBuffer;

      if (typeof window !== 'undefined') {
        // Browser environment
        const response = await fetch(wasmFilePath);
        wasmBuffer = await response.arrayBuffer();
      } else {
        // Node.js environment
        wasmBuffer = await fs.readFile(wasmFilePath);
      }

      const imports = {
        // Add any required imports here
        env: {
          memory: new WebAssembly.Memory({ initial: 256, maximum: 4096 }),
        },
      };

      const result = await WebAssembly.instantiate(wasmBuffer, imports);
      wasmModule = result.module;
      wasmInstance = result.instance;

      if (this.debug) {
        console.log(`Loaded WASM module: ${moduleFile}`);
      }

      return result;
    } catch (error) {
      throw new Error(`Failed to load WASM module: ${error.message}`);
    }
  }
}

/**
 * Worker pool for parallel execution
 */
class WorkerPool {
  constructor(size = 4) {
    this.size = size;
    this.workers = [];
    this.queue = [];
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) {
      return;
    }

    // In Node.js, use worker_threads
    if (typeof window === 'undefined') {
      // const { Worker } = require('worker_threads');
      for (let i = 0; i < this.size; i++) {
        // TODO: Create worker thread
      }
    } else {
      // In browser, use Web Workers
      for (let i = 0; i < this.size; i++) {
        // TODO: Create web worker
      }
    }

    this.initialized = true;
  }

  async execute(task) {
    // TODO: Implement worker pool execution
    return task;
  }

  terminate() {
    this.workers.forEach(worker => {
      if (worker.terminate) {
        worker.terminate();
      }
    });
    this.workers = [];
    this.initialized = false;
  }
}

/**
 * Main RuvSwarm class
 */
class RuvSwarm {
  constructor(wasmSwarm, options = {}) {
    this._wasmSwarm = wasmSwarm;
    this._options = options;
    this._workerPool = null;

    if (options.parallel) {
      this._workerPool = new WorkerPool(options.workerPoolSize || 4);
    }
  }

  static async initialize(options = {}) {
    const loader = new WASMLoader(options);
    // const wasmResult = await loader.loadModule();

    // Load the WASM bindings (ES module import with proper file URL)
    const wasmJsPath = path.join(loader.wasmPath, 'ruv_swarm_wasm.js');
    const bindings = await import(path.resolve(wasmJsPath));

    // Initialize WASM module with file buffer for Node.js
    if (bindings.default) {
      const wasmPath = path.join(loader.wasmPath, 'ruv_swarm_wasm_bg.wasm');
      const wasmBuffer = await fs.readFile(wasmPath);
      await bindings.default({ module_or_path: wasmBuffer });
    }

    // Get runtime features
    const features = new bindings.RuntimeFeatures();
    if (options.debug) {
      console.log('Runtime features:', {
        simd: features.simd_available,
        threads: features.threads_available,
        memoryLimit: features.memory_limit,
      });
    }

    return new RuvSwarm(bindings, options);
  }

  static detectSIMDSupport() {
    const loader = new WASMLoader();
    return loader.detectSIMDSupport();
  }

  static getRuntimeFeatures() {
    if (!wasmInstance) {
      throw new Error('RuvSwarm not initialized. Call RuvSwarm.initialize() first.');
    }

    const features = new wasmInstance.exports.RuntimeFeatures();
    return {
      simdAvailable: features.simd_available,
      threadsAvailable: features.threads_available,
      memoryLimit: features.memory_limit,
    };
  }

  static getVersion() {
    if (!wasmInstance) {
      return require('../package.json').version;
    }
    return wasmInstance.exports.get_version();
  }

  static getMemoryUsage() {
    if (!wasmInstance) {
      return 0;
    }
    return wasmInstance.exports.get_wasm_memory_usage();
  }

  async createSwarm(config) {
    try {
      const swarm = new this._wasmSwarm.RuvSwarm(config);
      return new SwarmWrapper(swarm, this._options);
    } catch (error) {
      throw new Error(`Failed to create swarm: ${error.message}`);
    }
  }
}

/**
 * Swarm wrapper class
 */
class SwarmWrapper {
  constructor(wasmSwarm, options = {}) {
    this._swarm = wasmSwarm;
    this._options = options;
    this._retryAttempts = options.retryAttempts || 3;
    this._retryDelay = options.retryDelay || 1000;
  }

  get name() {
    return this._swarm.name;
  }

  get agentCount() {
    return this._swarm.agent_count;
  }

  get maxAgents() {
    return this._swarm.max_agents;
  }

  async spawn(config) {
    return await this._retryOperation(async() => {
      const agent = await this._swarm.spawn(config);
      return new AgentWrapper(agent, this._options);
    });
  }

  async orchestrate(task) {
    return await this._retryOperation(async() => {
      return await this._swarm.orchestrate(task);
    });
  }

  getAgents() {
    return this._swarm.get_agents();
  }

  getStatus() {
    return this._swarm.get_status();
  }

  async _retryOperation(operation) {
    let lastError;

    for (let attempt = 0; attempt < this._retryAttempts; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        if (attempt < this._retryAttempts - 1) {
          await new Promise(resolve => setTimeout(resolve, this._retryDelay));
        }
      }
    }

    throw lastError;
  }
}

/**
 * Agent wrapper class
 */
class AgentWrapper {
  constructor(wasmAgent, options = {}) {
    this._agent = wasmAgent;
    this._options = options;
  }

  get id() {
    return this._agent.id;
  }

  get agentType() {
    return this._agent.agent_type;
  }

  get status() {
    return this._agent.status;
  }

  get tasksCompleted() {
    return this._agent.tasks_completed;
  }

  async execute(task) {
    return await this._agent.execute(task);
  }

  getMetrics() {
    return this._agent.get_metrics();
  }

  getCapabilities() {
    return this._agent.get_capabilities();
  }

  reset() {
    this._agent.reset();
  }
}

// Re-export utility functions
const consoleLog = (message) => {
  if (wasmInstance && wasmInstance.exports.console_log) {
    wasmInstance.exports.console_log(message);
  } else {
    console.log(message);
  }
};

const consoleError = (message) => {
  if (wasmInstance && wasmInstance.exports.console_error) {
    wasmInstance.exports.console_error(message);
  } else {
    console.error(message);
  }
};

const consoleWarn = (message) => {
  if (wasmInstance && wasmInstance.exports.console_warn) {
    wasmInstance.exports.console_warn(message);
  } else {
    console.warn(message);
  }
};

const formatJsError = (error) => {
  if (wasmInstance && wasmInstance.exports.format_js_error) {
    return wasmInstance.exports.format_js_error(error);
  }
  return error.toString();
};

// Import neural agent capabilities
import {
  NeuralAgent,
  NeuralAgentFactory,
  NeuralNetwork,
  COGNITIVE_PATTERNS,
  AGENT_COGNITIVE_PROFILES,
} from './neural-agent.js';

// Legacy exports - these are now provided by index-enhanced.js
// Export all the legacy functions and classes directly
export {
  RuvSwarm,
  consoleLog,
  consoleError,
  consoleWarn,
  formatJsError,
  // Neural agent exports
  NeuralAgent,
  NeuralAgentFactory,
  NeuralNetwork,
  COGNITIVE_PATTERNS,
  AGENT_COGNITIVE_PROFILES,
};