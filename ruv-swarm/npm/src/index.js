/** Node.js facade for the generated WebAssembly bindings. */

import path from 'node:path';
import { promises as fs, readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
const packageInfo = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));

let wasmModule = null;
let wasmInstance = null;
const loadedBindings = new Map();

function detectSIMDSupport() {
  return WebAssembly.validate(new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123,
    3, 2, 1, 0, 10, 22, 1, 20, 0, 253, 12, ...new Array(16).fill(0), 11,
  ]));
}

class RuvSwarm {
  constructor(bindings, options = {}) {
    if (typeof bindings?.WasmSwarmOrchestrator !== 'function') {
      throw new Error('Use await RuvSwarm.initialize() before creating a swarm');
    }
    this._bindings = bindings;
    this._options = options;
  }

  static async initialize(options = {}) {
    const wasmPath = path.resolve(options.wasmPath || fileURLToPath(new URL('../wasm/', import.meta.url)));
    const simd = Boolean(options.useSIMD && detectSIMDSupport());
    const moduleUrl = pathToFileURL(path.join(wasmPath, 'ruv_swarm_wasm.js'));
    moduleUrl.searchParams.set('variant', simd ? 'simd' : 'core');
    const key = moduleUrl.href;
    if (!loadedBindings.has(key)) {
      const loading = (async() => {
        const bindings = await import(key);
        const bytes = await fs.readFile(path.join(wasmPath, simd ? 'ruv_swarm_simd.wasm' : 'ruv_swarm_wasm_bg.wasm'));
        const instance = await bindings.default({ module_or_path: bytes });
        return { bindings, instance };
      })();
      loadedBindings.set(key, loading);
      loading.catch(() => loadedBindings.delete(key));
    }
    const loaded = await loadedBindings.get(key);
    wasmModule = loaded.bindings;
    wasmInstance = loaded.instance;
    return new RuvSwarm(loaded.bindings, { ...options, useSIMD: simd });
  }

  static detectSIMDSupport() {
    return detectSIMDSupport();
  }
  detectSIMDSupport() {
    return detectSIMDSupport();
  }
  static getVersion() {
    return packageInfo.version;
  }

  static getRuntimeFeatures() {
    if (!wasmModule) {
      throw new Error('Call RuvSwarm.initialize() first');
    }
    const features = new wasmModule.RuntimeFeatures();
    try {
      return { simdAvailable: features.simd_available, threadsAvailable: features.threads_available,
        memoryLimit: Number(features.memory_limit) };
    } finally {
      features.free();
    }
  }

  static getMemoryUsage() {
    return wasmInstance ? wasmInstance.memory.buffer.byteLength : 0;
  }

  async createSwarm(config = {}) {
    if (!config || typeof config !== 'object' || Array.isArray(config)) {
      throw new TypeError('Swarm configuration must be an object');
    }
    const maxAgents = config.maxAgents ?? 10;
    if (!Number.isInteger(maxAgents) || maxAgents < 1 || maxAgents > 1000) {
      throw new RangeError('maxAgents must be an integer between 1 and 1000');
    }
    const topology = config.topology || ({ centralized: 'star', distributed: 'mesh', hybrid: 'mesh' }[config.mode] || config.mode) || 'mesh';
    const swarm = new this._bindings.WasmSwarmOrchestrator(topology);
    return new SwarmWrapper(swarm, { ...config, topology, maxAgents, name: config.name || 'default-swarm' });
  }

  createNeuralNetwork(layers, activation = 'Sigmoid') {
    if (!Array.isArray(layers) && !(layers instanceof Uint32Array)) {
      throw new TypeError('layers must be an array of positive integers');
    }
    if (!Array.from(layers).every(layer => Number.isSafeInteger(layer) && layer > 0 && layer <= 1000000)) {
      throw new RangeError('layers must contain positive bounded integers');
    }
    const value = this._bindings.ActivationFunction[activation];
    if (typeof value !== 'number') {
      throw new RangeError('Unknown activation function');
    }
    return new this._bindings.WasmNeuralNetwork(Uint32Array.from(layers), value);
  }

  createForecastingModel(modelType) {
    return new this._bindings.WasmForecastingModel(modelType);
  }
}

class SwarmWrapper {
  constructor(swarm, config) {
    this._swarm = swarm;
    this._config = config;
  }
  get name() {
    return this._config.name;
  }
  get maxAgents() {
    return this._config.maxAgents;
  }
  get agentCount() {
    return this._swarm.get_agent_count();
  }
  _details() {
    return JSON.parse(this._swarm.get_status(true));
  }

  async spawn(config) {
    if (this.agentCount >= this.maxAgents) {
      throw new RangeError('Swarm agent limit reached');
    }
    const info = JSON.parse(this._swarm.spawn(JSON.stringify(config)));
    return new AgentWrapper(this, info);
  }

  async orchestrate(task) {
    if (!task || typeof task !== 'object' || Array.isArray(task)) {
      throw new TypeError('Task must be an object');
    }
    if (task.dependencies?.length) {
      throw new Error('Dependency scheduling is not supported by the WASM metadata orchestrator');
    }
    const config = { description: task.description, priority: task.priority || 'medium' };
    for (const key of ['maxAgents', 'requiredCapabilities']) {
      if (task[key] !== undefined) {
        config[key] = task[key];
      }
    }
    const result = this._swarm.orchestrate(JSON.stringify(config));
    try {
      return { taskId: result.task_id, description: result.description, priority: result.priority,
        status: result.status, assignedAgents: Array.from(result.assigned_agents) };
    } finally {
      result.free();
    }
  }

  getAgents() {
    return this._details().agent_details.map(agent => agent.id);
  }
  getStatus() {
    const details = this._details();
    return { ...details, name: this.name, agentCount: this.agentCount, maxAgents: this.maxAgents,
      strategy: this._config.strategy || 'metadata', mode: this._config.topology };
  }
  free() {
    this._swarm.free();
  }
}

class AgentWrapper {
  constructor(swarm, info) {
    this._swarm = swarm; this._info = info;
  }
  get id() {
    return this._info.agent_id;
  }
  get agentType() {
    return this._info.type;
  }
  get status() {
    return this._swarm._details().agent_details.find(agent => agent.id === this.id)?.status;
  }
  get tasksCompleted() {
    return 0;
  }
  async execute() {
    throw new Error('WASM agents track assignments; external task execution is not implemented');
  }
  getMetrics() {
    throw new Error('Execution metrics require an external task executor');
  }
  getCapabilities() {
    return [...this._info.capabilities];
  }
  reset() {
    this._swarm._swarm.update_agent_status(this.id, 'idle');
  }
}

const consoleLog = message => console.log(message);
const consoleError = message => console.error(message);
const consoleWarn = message => console.warn(message);
const formatJsError = error => String(error);

// Reserved autonomous service APIs fail explicitly until implemented.
import { DAAService, daaService } from './daa-service.js';
export { RuvSwarm, SwarmWrapper as WasmSwarm, AgentWrapper as JsAgent,
  consoleLog, consoleError, consoleWarn, formatJsError, DAAService, daaService };
