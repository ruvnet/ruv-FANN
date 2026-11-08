/** MCP operations backed by actual WASM inference and SQLite records. */
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { RuvSwarm } from './index.js';
import { SwarmPersistence } from './persistence.js';
import { infer, forecast, denseBenchmark } from './neural-runtime.js';
import { unsupported } from './unsupported.js';

const string = { type: 'string', minLength: 1, maxLength: 4096 };
const numericArray = { type: 'array', minItems: 1, maxItems: 1000000, items: { type: 'number' } };
const definitions = [
  ['swarm_init', 'Create a local swarm metadata record', { topology: string, maxAgents: { type: 'integer', minimum: 1, maximum: 1000 }, name: string }],
  ['swarm_status', 'Read actual swarm and agent metadata', { swarmId: string }],
  ['swarm_monitor', 'Sample local metadata and process memory for up to 60 seconds', { duration: { type: 'integer', minimum: 1, maximum: 60000 }, interval: { type: 'integer', minimum: 10, maximum: 10000 } }],
  ['agent_spawn', 'Register a local agent and its capabilities', { swarmId: string, type: string, name: string, capabilities: { type: 'array', maxItems: 64, items: string } }],
  ['agent_list', 'List registered agents', { swarmId: string }],
  ['agent_metrics', 'Read recorded agent metrics; no inferred scores', { agentId: string }],
  ['agent_update_status', 'Set local agent state after application handling', { agentId: string, status: { enum: ['idle', 'busy', 'offline', 'error'] } }],
  ['task_orchestrate', 'Assign matching idle agents; does not execute an external model', { swarmId: string, task: string, priority: string, maxAgents: { type: 'integer', minimum: 1, maximum: 1000 }, requiredCapabilities: { type: 'array', maxItems: 64, items: string } }],
  ['task_status', 'Read a persisted task record', { taskId: string }],
  ['task_results', 'Read recorded results; missing tasks are errors', { taskId: string }],
  ['neural_status', 'Report actual local inference and forecasting support', {}],
  ['neural_infer', 'Run dense WASM inference with explicit weights', { layers: { ...numericArray, maxItems: 64 }, weights: numericArray, input: numericArray, activation: string }],
  ['forecast_predict', 'Compute a one-step statistical forecast', { model: { enum: ['linear', 'mean', 'naive'] }, input: numericArray }],
  ['benchmark_run', 'Measure local dense WASM inference; no LLM or distributed benchmark', { iterations: { type: 'integer', minimum: 1, maximum: 10000 }, type: { enum: ['wasm', 'neural', 'dense-inference'] } }],
  ['features_detect', 'Report capabilities exposed by the loaded implementation', {}],
  ['memory_usage', 'Read actual Node process and WASM memory', {}],
  ['persistence_stats', 'Count actual SQLite swarm, agent and task records', {}],
];

export function getSupportedToolDefinitions() {
  return definitions.map(([name, description, properties]) => ({ name, description,
    inputSchema: { type: 'object', properties, additionalProperties: false } }));
}

function text(value, name, maximum = 4096) {
  if (typeof value !== 'string' || !value.trim() || value.length > maximum) {
    throw new TypeError(`${name} must be nonempty text of at most ${maximum} characters`);
  }
  return value;
}
function integer(value, name, maximum = 1000) {
  if (!Number.isSafeInteger(value) || value < 1 || value > maximum) {
    throw new RangeError(`${name} must be an integer between 1 and ${maximum}`);
  }
  return value;
}
function normalizeError(error) {
  return error instanceof Error ? error : new Error(String(error));
}

export class EnhancedMCPTools {
  constructor(_legacyRuntime = null, options = {}) {
    this.options = options;
    this.activeSwarms = new Map();
    this.toolMetrics = new Map();
    this.persistence = null;
    this.ruvSwarm = null;
    this.initialization = null;
    this.destroyed = false;
    this.limits = { swarms: integer(options.maxSwarms ?? 64, 'maxSwarms', 64),
      agents: integer(options.maxAgents ?? 1000, 'maxAgents'), tasks: integer(options.maxTasks ?? 10000, 'maxTasks', 10000) };
    this.tools = Object.fromEntries(definitions.map(([name, _description, properties]) => [name, async(params = {}) => {
      try {
        await this.initialize();
        if (!params || typeof params !== 'object' || Array.isArray(params)) {
          throw new TypeError('Tool arguments must be an object');
        }
        for (const key of Object.keys(params)) {
          if (!Object.hasOwn(properties, key)) {
            throw new TypeError(`Unsupported argument for ${name}: ${key}`);
          }
        }
        const result = await this[name](params);
        this.toolMetrics.set(name, (this.toolMetrics.get(name) || 0) + 1);
        return result;
      } catch (error) {
        throw normalizeError(error);
      }
    }]));
    for (const name of ['neural_train', 'neural_patterns', 'pool_health', 'pool_stats']) {
      this.tools[name] = async() => unsupported(name);
    }
  }

  async initialize() {
    if (this.destroyed) {
      throw new Error('MCP tools have been closed');
    }
    if (!this.initialization) {
      this.initialization = (async() => {
        this.ruvSwarm = await RuvSwarm.initialize();
        this.persistence = new SwarmPersistence(this.options.dbPath || process.env.RUV_SWARM_DB_PATH || path.resolve('.ruv-swarm/state.sqlite'));
        if (this.count('swarms') > this.limits.swarms || this.count('agents') > this.limits.agents || this.count('tasks') > this.limits.tasks) {
          throw new RangeError('Stored metadata exceeds the configured resource limits');
        }
        for (const row of this.persistence.getActiveSwarms()) {
          const instance = await this.ruvSwarm.createSwarm({ name: row.name, topology: row.topology, maxAgents: row.max_agents });
          this.activeSwarms.set(row.id, { ...row, instance });
          for (const agent of this.persistence.getSwarmAgents(row.id)) {
            await instance.spawn({ id: agent.id, name: agent.name, type: agent.type, capabilities: agent.capabilities });
            instance._swarm.update_agent_status(agent.id, ['idle', 'busy', 'offline', 'error'].includes(agent.status) ? agent.status : 'offline');
          }
        }
        return this;
      })();
    }
    return this.initialization;
  }

  swarm(params = {}) {
    const id = params.swarmId || params.swarm_id || this.activeSwarms.keys().next().value;
    const swarm = this.activeSwarms.get(id);
    if (!swarm) {
      throw new Error(`Swarm not found: ${id || '(none initialized)'}`);
    }
    return swarm;
  }

  count(table) {
    if (!['swarms', 'agents', 'tasks'].includes(table)) {
      throw new TypeError('Unknown count table');
    }
    return this.persistence.db.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get().count;
  }

  async swarm_init(params = {}) {
    const maxAgents = integer(params.maxAgents ?? params.max_agents ?? 10, 'maxAgents');
    const topology = params.topology || 'mesh';
    const name = text(params.name || 'local-swarm', 'name');
    const id = randomUUID();
    const instance = await this.ruvSwarm.createSwarm({ name, topology, maxAgents });
    try {
      this.persistence.db.transaction(() => {
        if (this.count('swarms') >= this.limits.swarms) {
          throw new RangeError('Swarm resource limit reached');
        }
        this.persistence.createSwarm({ id, name, topology, maxAgents, strategy: params.strategy || 'metadata' });
      })();
      this.activeSwarms.set(id, { id, name, topology, max_agents: maxAgents, instance });
      return { id, swarm_id: id, name, topology, maxAgents, status: 'active', scope: 'local-metadata' };
    } catch (error) {
      instance.free(); throw error;
    }
  }

  async swarm_status(params = {}) {
    const swarms = (params.swarmId || params.swarm_id) ? [this.swarm(params)] : [...this.activeSwarms.values()];
    return { swarms: swarms.map(s => ({ id: s.id, ...s.instance.getStatus() })),
      active_swarms: swarms.length, total_agents: swarms.reduce((n, s) => n + s.instance.agentCount, 0),
      scope: 'local-metadata', execution_available: false };
  }

  async agent_spawn(params = {}) {
    const swarm = this.swarm(params);
    const info = { id: randomUUID(), name: params.name || 'Local agent', type: params.type || 'generic', capabilities: params.capabilities || [] };
    // Persist first so a failed SQL write cannot leave an unrecorded WASM agent.
    this.persistence.db.transaction(() => {
      if (this.count('agents') >= this.limits.agents) {
        throw new RangeError('Agent resource limit reached');
      }
      this.persistence.createAgent({ ...info, swarmId: swarm.id });
    })();
    try {
      const agent = await swarm.instance.spawn(info);
      return { id: agent.id, agent_id: agent.id, name: info.name, type: agent.agentType, capabilities: agent.getCapabilities(), status: agent.status, swarm_id: swarm.id };
    } catch (error) {
      this.persistence.db.prepare('DELETE FROM agents WHERE id = ?').run(info.id);
      throw error;
    }
  }

  async agent_list(params = {}) {
    const agents = (params.swarmId || params.swarm_id) ? this.persistence.getSwarmAgents(this.swarm(params).id) :
      [...this.activeSwarms.keys()].flatMap(id => this.persistence.getSwarmAgents(id));
    return { agents, count: agents.length };
  }

  async agent_metrics(params = {}) {
    const id = text(params.agentId || params.agent_id, 'agentId');
    const agent = this.persistence.getAgent(id);
    if (!agent) {
      throw new Error(`Agent not found: ${id}`);
    }
    return { agent_id: id, status: agent.status, recorded_metrics: this.persistence.getMetrics('agent', id),
      execution_available: false };
  }

  async agent_update_status(params = {}) {
    const id = text(params.agentId || params.agent_id, 'agentId');
    const agent = this.persistence.getAgent(id);
    if (!agent) {
      throw new Error(`Agent not found: ${id}`);
    }
    if (!['idle', 'busy', 'offline', 'error'].includes(params.status)) {
      throw new TypeError('Invalid agent status');
    }
    const swarm = this.swarm({ swarmId: agent.swarm_id });
    // Roll back the persisted state if WASM rejects the matching transition.
    this.persistence.db.transaction(() => {
      this.persistence.updateAgentStatus(id, params.status);
      swarm.instance._swarm.update_agent_status(id, params.status);
    })();
    return { agent_id: id, status: params.status };
  }

  async task_orchestrate(params = {}) {
    if (params.dependencies?.length) {
      unsupported('Dependency scheduling');
    }
    const swarm = this.swarm(params);
    const description = text(params.task || params.description, 'task');
    const assignment = await swarm.instance.orchestrate({ description, priority: params.priority || 'medium',
      maxAgents: integer(params.maxAgents ?? 3, 'maxAgents'), requiredCapabilities: params.requiredCapabilities || [] });
    const id = randomUUID();
    try {
      this.persistence.db.transaction(() => {
        if (this.count('tasks') >= this.limits.tasks) {
          throw new RangeError('Task resource limit reached');
        }
        this.persistence.createTask({ id, swarmId: swarm.id, description, status: assignment.status,
          priority: assignment.priority, assignedAgents: assignment.assignedAgents });
        for (const agent of assignment.assignedAgents) {
          const changed = this.persistence.db.prepare("UPDATE agents SET status = 'busy' WHERE id = ? AND status = 'idle'").run(agent);
          if (changed.changes !== 1) {
            throw new Error('Agent state changed during assignment; retry with current metadata');
          }
        }
      })();
    } catch (error) {
      for (const agent of assignment.assignedAgents) {
        const status = this.persistence.getAgent(agent)?.status || 'offline';
        swarm.instance._swarm.update_agent_status(agent, status);
      }
      throw error;
    }
    return { ...assignment, id, task_id: id, taskId: id, execution_available: false };
  }

  async task_status(params = {}) {
    const id = text(params.taskId || params.task_id, 'taskId');
    const task = this.persistence.getTask(id);
    if (!task) {
      throw new Error(`Task not found: ${id}`);
    }
    return { ...task, execution_available: false };
  }

  async task_results(params = {}) {
    const task = await this.task_status(params);
    const rows = this.persistence.db.prepare('SELECT * FROM task_results WHERE task_id = ? ORDER BY created_at').all(task.id);
    return { task_id: task.id, status: task.status, result: task.result, recorded_results: rows,
      source: 'persisted-records', execution_available: false };
  }

  async neural_status() {
    return this.features_detect();
  }
  async neural_infer(params) {
    return infer(this.ruvSwarm, params);
  }
  async forecast_predict(params) {
    return forecast(this.ruvSwarm, params);
  }
  async benchmark_run(params) {
    return denseBenchmark(this.ruvSwarm, params);
  }
  async features_detect() {
    return { wasm_version: this.ruvSwarm._bindings.get_version(), runtime: RuvSwarm.getRuntimeFeatures(),
      dense_inference: true, statistical_forecasts: ['linear', 'mean', 'naive'],
      neural_training: false, external_execution: false, autonomous_learning: false, distributed_consensus: false };
  }
  async memory_usage() {
    return { process: process.memoryUsage(), wasm_bytes: RuvSwarm.getMemoryUsage(), source: 'runtime-measurements' };
  }
  async persistence_stats() {
    const counts = {};
    for (const table of ['swarms', 'agents', 'tasks', 'task_results']) {
      counts[table] = this.persistence.db.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get().count;
    }
    return { backend: 'SQLite', counts };
  }
  async neural_train() {
    unsupported('Neural training');
  }
  async neural_patterns() {
    unsupported('Learned cognitive-pattern analysis');
  }
  async swarm_monitor(params = {}) {
    const duration = integer(params.duration ?? 1000, 'duration', 60000);
    const interval = integer(params.interval ?? 1000, 'interval', 10000);
    if (interval < 10) {
      throw new RangeError('interval must be at least 10 milliseconds');
    }
    const samplingInterval = Math.max(interval, Math.ceil(duration / 99));
    const deadline = performance.now() + duration;
    const snapshots = [];
    do {
      const agentCounts = this.persistence.db.prepare('SELECT status, COUNT(*) AS count FROM agents GROUP BY status').all();
      snapshots.push({ timestamp: new Date().toISOString(), agent_counts: Object.fromEntries(agentCounts.map(row => [row.status, row.count])),
        swarms: this.count('swarms'), memory: process.memoryUsage() });
      const remaining = deadline - performance.now();
      if (remaining <= 0) {
        break;
      }
      await delay(Math.min(samplingInterval, remaining));
    } while (performance.now() <= deadline && snapshots.length < 100);
    return { scope: 'local-metadata-and-process', sampling_interval_ms: samplingInterval, snapshots };
  }
  async pool_health() {
    unsupported('SQLite connection pool telemetry');
  }
  async pool_stats() {
    unsupported('SQLite connection pool telemetry');
  }
  getAllToolDefinitions() {
    return getSupportedToolDefinitions();
  }
  recordToolMetrics(name) {
    this.toolMetrics.set(name, (this.toolMetrics.get(name) || 0) + 1);
  }
  handleError(error) {
    throw normalizeError(error);
  }
  async destroy() {
    if (this.destroyed) {
      return;
    }
    if (this.initialization) {
      try {
        await this.initialization;
      } catch { /* Close resources allocated before initialization failed. */ }
    }
    for (const swarm of this.activeSwarms.values()) {
      swarm.instance.free();
    }
    this.activeSwarms.clear();
    this.persistence?.close();
    this.destroyed = true;
  }
}

let instance;
export function getEnhancedMCPToolsInstance() {
  return instance || (instance = new EnhancedMCPTools());
}
export default new Proxy({}, { get(_target, key) {
  const tools = getEnhancedMCPToolsInstance();
  return typeof tools[key] === 'function' ? tools[key].bind(tools) : tools[key];
} });
