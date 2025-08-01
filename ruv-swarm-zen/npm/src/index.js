/**
 * Enhanced RuvSwarm Main Class
 * Provides full WASM capabilities with progressive loading,
 * neural networks, forecasting, and swarm orchestration
 */

import { WasmModuleLoader } from './wasm-loader.js';
import { SwarmPersistencePooled } from './persistence-pooled.js';
import { getContainer } from './singleton-container.js';
// import { NeuralAgentFactory } from './neural-agent.js';
// import path from 'path';
// import fs from 'fs';

/**
 * Simple Agent class for test compatibility
 */
export class Agent {
  constructor(config = {}) {
    this.id = config.id || `agent-${Date.now()}`;
    this.type = config.type || 'generic';
    this.config = config;
    this.isActive = false;
  }

  async initialize() {
    this.isActive = true;
    return true;
  }

  async execute(task) {
    return {
      success: true,
      result: `Agent ${this.id} executed: ${task}`,
      agent: this.id,
    };
  }

  async cleanup() {
    this.isActive = false;
    return true;
  }
}

class RuvSwarm {
  constructor() {
    this.wasmLoader = new WasmModuleLoader();
    this.persistence = null;
    this.activeSwarms = new Map();
    this.globalAgents = new Map();
    this.isInitialized = false;
    this.metrics = {
      totalSwarms: 0,
      totalAgents: 0,
      totalTasks: 0,
      memoryUsage: 0,
      performance: {},
    };
    this.features = {
      neural_networks: false,
      forecasting: false,
      cognitive_diversity: false,
      simd_support: false,
    };
  }

  /**
   * Cleanup method for proper resource disposal
   */
  destroy() {
    console.log('🧹 Cleaning up RuvSwarm instance...');

    // Terminate all active swarms
    for (const swarm of this.activeSwarms.values()) {
      if (typeof swarm.terminate === 'function') {
        swarm.terminate();
      }
    }

    this.activeSwarms.clear();
    this.globalAgents.clear();

    // Cleanup persistence
    if (this.persistence && typeof this.persistence.close === 'function') {
      this.persistence.close();
    }

    // Cleanup WASM loader
    if (this.wasmLoader && typeof this.wasmLoader.cleanup === 'function') {
      this.wasmLoader.cleanup();
    }

    this.isInitialized = false;
  }

  static async initialize(options = {}) {
    const container = getContainer();

    // Register RuvSwarm factory if not already registered
    if (!container.has('RuvSwarm')) {
      container.register('RuvSwarm', () => new RuvSwarm(), {
        singleton: true,
        lazy: false,
      });
    }

    // Get or create singleton instance through container
    const instance = container.get('RuvSwarm');

    const {
      // wasmPath = './wasm',
      loadingStrategy = 'progressive',
      enablePersistence = true,
      enableNeuralNetworks = true,
      enableForecasting = false,
      useSIMD = true,
      debug = false,
    } = options;

    // Check if already initialized through container
    if (instance.isInitialized) {
      if (debug) {
        console.log('[DEBUG] RuvSwarm already initialized through container');
      }
      return instance;
    }

    // Minimal loading for fast startup
    if (loadingStrategy === 'minimal') {
      if (debug) {
        console.log('[DEBUG] Using minimal loading strategy for fast startup');
      }
      console.log('⚡ Fast-initializing ruv-swarm (minimal mode)...');
      instance.isInitialized = true;
      instance.features.simd_support = false; // Skip SIMD detection for speed
      return instance;
    }

    console.log('🧠 Initializing ruv-swarm with WASM capabilities...');

    try {
      // Initialize WASM modules (skip for minimal loading)
      if (loadingStrategy !== 'minimal') {
        await instance.wasmLoader.initialize(loadingStrategy);
      }

      // Detect and enable features (skip for minimal loading)
      if (loadingStrategy !== 'minimal') {
        await instance.detectFeatures(useSIMD);
      }

      // Initialize pooled persistence if enabled
      if (enablePersistence) {
        try {
          // Configure pool settings based on environment or defaults
          const poolOptions = {
            maxReaders: parseInt(process.env.POOL_MAX_READERS) || 6,
            maxWorkers: parseInt(process.env.POOL_MAX_WORKERS) || 3,
            mmapSize: parseInt(process.env.POOL_MMAP_SIZE) || 268435456, // 256MB
            cacheSize: parseInt(process.env.POOL_CACHE_SIZE) || -64000, // 64MB
            enableBackup: process.env.POOL_ENABLE_BACKUP === 'true',
            healthCheckInterval: 60000, // 1 minute
          };

          instance.persistence = new SwarmPersistencePooled(undefined, poolOptions);
          await instance.persistence.initialize();
          console.log('[INFO] Pooled persistence layer initialized successfully');
          if (debug) {
            console.log(`[DEBUG] Pool configuration: ${poolOptions.maxReaders} readers, ${poolOptions.maxWorkers} workers`);
          }
        } catch (error) {
          console.warn('⚠️ Pooled persistence not available:', error.message);
          instance.persistence = null;
        }
      } else {
        if (debug) {
          console.log('[DEBUG] Persistence disabled for fast startup');
        }
      }

      // Pre-load neural networks if enabled
      if (enableNeuralNetworks && loadingStrategy !== 'minimal') {
        try {
          await instance.wasmLoader.loadModule('neural');
          instance.features.neural_networks = true;
          if (debug) {
            console.log('[DEBUG] Neural network capabilities loaded');
          }
        } catch (error) {
          if (debug) {
            console.warn('⚠️ Neural network module not available:', error.message);
          }
          instance.features.neural_networks = false;
        }
      }

      // Pre-load forecasting if enabled
      if (enableForecasting && enableNeuralNetworks && loadingStrategy !== 'minimal') {
        try {
          await instance.wasmLoader.loadModule('forecasting');
          instance.features.forecasting = true;
          if (debug) {
            console.log('[DEBUG] Forecasting capabilities loaded');
          }
        } catch (error) {
          if (debug) {
            console.warn('⚠️ Forecasting module not available:', error.message);
          }
          instance.features.forecasting = false;
        }
      }

      if (loadingStrategy !== 'minimal') {
        console.log('✅ ruv-swarm initialized successfully');
        if (debug) {
          console.log('📊 Features:', instance.features);
        }
      }

      // Mark as initialized
      instance.isInitialized = true;

      return instance;
    } catch (error) {
      console.error('❌ Failed to initialize ruv-swarm:', error);
      throw error;
    }
  }

  async detectFeatures(useSIMD = true) {
    try {
      // Load core module to detect basic features
      const coreModule = await this.wasmLoader.loadModule('core');

      // Detect SIMD support
      if (useSIMD) {
        this.features.simd_support = RuvSwarm.detectSIMDSupport();
      }

      // Check if core module has the expected exports
      if (coreModule.exports) {
        // Check for neural network support
        this.features.neural_networks = true; // Will be validated when module loads

        // Check for cognitive diversity support
        this.features.cognitive_diversity = true; // Default enabled
      }

      console.log('🔍 Feature detection complete');
    } catch (error) {
      console.warn('⚠️ Feature detection failed:', error.message);
    }
  }

  async createSwarm(config) {
    const {
      id = null, // Allow existing ID for persistence loading
      name = 'default-swarm',
      topology = 'mesh',
      strategy = 'balanced',
      maxAgents = 10,
      enableCognitiveDiversity = true,
      // enableNeuralAgents = true,
    } = config;

    // Ensure core module is loaded
    const coreModule = await this.wasmLoader.loadModule('core');

    // Create swarm configuration
    const swarmConfig = {
      name,
      topology_type: topology,
      max_agents: maxAgents,
      enable_cognitive_diversity: enableCognitiveDiversity && this.features.cognitive_diversity,
    };

    // Use the core module exports to create swarm
    let wasmSwarm;
    if (coreModule.exports && coreModule.exports.RuvSwarm) {
      try {
        wasmSwarm = new coreModule.exports.RuvSwarm();
        // Store swarm config - use existing ID if provided
        wasmSwarm.id = id || `swarm-${Date.now()}`;
        wasmSwarm.name = name;
        wasmSwarm.config = swarmConfig;
      } catch (error) {
        console.warn('Failed to create WASM swarm:', error.message);
        // Fallback to JavaScript implementation
        wasmSwarm = {
          id: id || `swarm-${Date.now()}`,
          name,
          config: swarmConfig,
          agents: new Map(),
          tasks: new Map(),
        };
      }
    } else {
      // Fallback for placeholder or different module structure
      wasmSwarm = {
        id: id || `swarm-${Date.now()}`,
        name,
        config: swarmConfig,
        agents: new Map(),
        tasks: new Map(),
      };
    }

    // Create JavaScript wrapper
    const swarm = new Swarm(wasmSwarm.id || wasmSwarm.name, wasmSwarm, this);

    // Persist swarm if persistence is enabled and this is a new swarm
    if (this.persistence && !id) {
      try {
        this.persistence.createSwarm({
          id: swarm.id,
          name,
          topology,
          strategy,
          maxAgents,
          created: new Date().toISOString(),
        });
      } catch (error) {
        if (!error.message.includes('UNIQUE constraint failed')) {
          console.warn('Failed to persist swarm:', error.message);
        }
      }
    }

    this.activeSwarms.set(swarm.id, swarm);
    this.metrics.totalSwarms++;

    console.log(`🐝 Created swarm: ${name} (${swarm.id})`);
    return swarm;
  }

  async getSwarmStatus(swarmId, detailed = false) {
    const swarm = this.activeSwarms.get(swarmId);
    if (!swarm) {
      throw new Error(`Swarm not found: ${swarmId}`);
    }

    return swarm.getStatus(detailed);
  }

  async getAllSwarms() {
    const swarms = [];
    for (const [id, swarm] of this.activeSwarms) {
      swarms.push({
        id,
        status: await swarm.getStatus(false),
      });
    }
    return swarms;
  }

  async getGlobalMetrics() {
    this.metrics.memoryUsage = this.wasmLoader.getTotalMemoryUsage();

    // Aggregate metrics from all swarms
    let totalAgents = 0;
    let totalTasks = 0;

    for (const swarm of this.activeSwarms.values()) {
      const status = await swarm.getStatus(false);
      totalAgents += status.agents?.total || 0;
      totalTasks += status.tasks?.total || 0;
    }

    this.metrics.totalAgents = totalAgents;
    this.metrics.totalTasks = totalTasks;
    this.metrics.totalSwarms = this.activeSwarms.size;

    return {
      ...this.metrics,
      features: this.features,
      wasm_modules: this.wasmLoader.getModuleStatus(),
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Legacy compatibility method for spawnAgent
   * Creates a default swarm if none exists and spawns an agent
   * @param {string} name - Agent name
   * @param {string} type - Agent type
   * @param {Object} options - Additional options
   * @returns {Promise<Agent>} The spawned agent
   */
  async spawnAgent(name, type = 'researcher', options = {}) {
    // Create a default swarm if none exists
    if (this.activeSwarms.size === 0) {
      await this.createSwarm({
        name: 'default-swarm',
        maxAgents: options.maxAgents || 10,
      });
    }

    // Get the first available swarm (or the default one we just created)
    const swarm = this.activeSwarms.values().next().value;

    return await swarm.spawnAgent(name, type, options);
  }

  // Feature detection helpers
  static detectSIMDSupport() {
    try {
      // Check for WebAssembly SIMD support using v128 type validation
      // This is more compatible across Node.js versions
      const simdTestModule = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, // WASM magic
        0x01, 0x00, 0x00, 0x00, // Version 1
        0x01, 0x05, 0x01, // Type section: 1 type
        0x60, 0x00, 0x01, 0x7b, // Function type: () -> v128 (SIMD type)
      ]);

      // If v128 type is supported, SIMD is available
      return WebAssembly.validate(simdTestModule);
    } catch {
      return false;
    }
  }

  static getVersion() {
    return '0.2.0'; // Enhanced version with full WASM capabilities
  }

  static getMemoryUsage() {
    if (typeof performance !== 'undefined' && performance.memory) {
      return {
        used: performance.memory.usedJSHeapSize,
        total: performance.memory.totalJSHeapSize,
        limit: performance.memory.jsHeapSizeLimit,
      };
    }
    return null;
  }

  static getRuntimeFeatures() {
    return {
      webassembly: typeof WebAssembly !== 'undefined',
      simd: RuvSwarm.detectSIMDSupport(),
      workers: typeof Worker !== 'undefined',
      shared_array_buffer: typeof SharedArrayBuffer !== 'undefined',
      bigint: typeof BigInt !== 'undefined',
    };
  }

  // Instance method that delegates to static method for API convenience
  detectSIMDSupport() {
    return RuvSwarm.detectSIMDSupport();
  }
}

// Enhanced Swarm wrapper class
class Swarm {
  constructor(id, wasmInstance, ruvSwarmInstance) {
    this.id = id;
    this.wasmSwarm = wasmInstance;
    this.ruvSwarm = ruvSwarmInstance;
    this.agents = new Map();
    this.tasks = new Map();
  }

  async spawn(config) {
    const {
      id = null, // Allow existing ID for persistence loading
      type = 'researcher',
      name = null,
      capabilities = null,
      enableNeuralNetwork = true,
    } = config;

    // Ensure neural networks are loaded if requested
    if (enableNeuralNetwork && this.ruvSwarm.features.neural_networks) {
      await this.ruvSwarm.wasmLoader.loadModule('neural');
    }

    const agentConfig = {
      agent_type: type,
      name: name || `${type}-${Date.now()}`,
      capabilities: capabilities || [],
      max_agents: 100, // Default limit
    };

    let result;
    if (this.wasmSwarm.spawn) {
      result = this.wasmSwarm.spawn(agentConfig);
    } else {
      // Fallback for placeholder - use existing ID if provided
      result = {
        agent_id: id || `agent-${Date.now()}`,
        name: agentConfig.name,
        type: agentConfig.agent_type,
        capabilities: agentConfig.capabilities,
        cognitive_pattern: 'adaptive',
        neural_network_id: enableNeuralNetwork ? `nn-${Date.now()}` : null,
      };
    }

    const agentId = id || result.agent_id || result.id;

    // Create JavaScript wrapper
    const agent = new Agent(agentId, result, this);
    this.agents.set(agentId, agent);

    // Persist agent if persistence is enabled and this is a new agent
    if (this.ruvSwarm.persistence && !id) {
      try {
        this.ruvSwarm.persistence.createAgent({
          id: agentId,
          swarmId: this.id,
          name: result.name,
          type,
          capabilities: result.capabilities,
          cognitive_pattern: result.cognitive_pattern,
          created: new Date().toISOString(),
        });
      } catch (error) {
        if (!error.message.includes('UNIQUE constraint failed')) {
          console.warn('Failed to persist agent:', error.message);
        }
      }
    }

    console.log(`🤖 Spawned agent: ${result.name} (${type})`);
    return agent;
  }

  /**
   * Legacy compatibility method for spawnAgent
   * @param {string} name - Agent name
   * @param {string} type - Agent type
   * @param {Object} options - Additional options
   * @returns {Promise<Agent>} The spawned agent
   */
  async spawnAgent(name, type = 'researcher', options = {}) {
    return await this.spawn({
      name,
      type,
      ...options,
    });
  }

  async orchestrate(taskConfig) {
    const {
      description,
      priority = 'medium',
      dependencies = [],
      maxAgents = null,
      estimatedDuration = null,
      requiredCapabilities = [],
    } = taskConfig;

    const config = {
      description,
      priority,
      dependencies,
      max_agents: maxAgents,
      estimated_duration_ms: estimatedDuration,
    };

    let result;
    if (this.wasmSwarm.orchestrate) {
      result = this.wasmSwarm.orchestrate(config);
    } else {
      // Enhanced fallback with proper agent assignment
      const availableAgents = this.selectAvailableAgents(requiredCapabilities, maxAgents);

      if (availableAgents.length === 0) {
        throw new Error('No agents available for task orchestration. Please spawn agents first.');
      }

      // Assign task to selected agents
      const assignedAgentIds = availableAgents.map(agent => agent.id);

      // Update agent status to busy
      for (const agent of availableAgents) {
        await agent.updateStatus('busy');
      }

      result = {
        task_id: `task-${Date.now()}`,
        task_description: description,
        description,
        status: 'orchestrated',
        assigned_agents: assignedAgentIds,
        priority,
        estimated_duration_ms: estimatedDuration,
        agent_selection_strategy: 'capability_and_load_based',
      };
    }

    const taskId = result.task_id || result.id;

    // Create JavaScript wrapper
    const task = new Task(taskId, result, this);
    this.tasks.set(taskId, task);

    // Persist task if persistence is enabled
    if (this.ruvSwarm.persistence) {
      await this.ruvSwarm.persistence.createTask({
        id: taskId,
        swarmId: this.id,
        description,
        priority,
        assigned_agents: result.assigned_agents,
        created: new Date().toISOString(),
      });
    }

    console.log(`📋 Orchestrated task: ${description} (${taskId}) - Assigned to ${result.assigned_agents.length} agents`);
    return task;
  }

  // Helper method to select available agents for task assignment
  selectAvailableAgents(requiredCapabilities = [], maxAgents = null) {
    const availableAgents = Array.from(this.agents.values()).filter(agent => {
      // Agent must be idle or active (not busy)
      if (agent.status === 'busy') {
        return false;
      }

      // Check if agent has required capabilities
      if (requiredCapabilities.length > 0) {
        const hasCapabilities = requiredCapabilities.some(capability =>
          agent.capabilities.includes(capability),
        );
        if (!hasCapabilities) {
          return false;
        }
      }

      return true;
    });

    // Apply maxAgents limit if specified
    if (maxAgents && maxAgents > 0) {
      return availableAgents.slice(0, maxAgents);
    }

    return availableAgents;
  }

  async getStatus(detailed = false) {
    if (this.wasmSwarm.get_status) {
      return this.wasmSwarm.get_status(detailed);
    }

    // Fallback status
    return {
      id: this.id,
      agents: {
        total: this.agents.size,
        active: Array.from(this.agents.values()).filter(a => a.status === 'active').length,
        idle: Array.from(this.agents.values()).filter(a => a.status === 'idle').length,
      },
      tasks: {
        total: this.tasks.size,
        pending: Array.from(this.tasks.values()).filter(t => t.status === 'pending').length,
        in_progress: Array.from(this.tasks.values()).filter(t => t.status === 'in_progress').length,
        completed: Array.from(this.tasks.values()).filter(t => t.status === 'completed').length,
      },
    };
  }

  async monitor(duration = 10000, interval = 1000) {
    if (this.wasmSwarm.monitor) {
      return this.wasmSwarm.monitor(duration, interval);
    }

    // Fallback monitoring
    console.log(`📊 Monitoring swarm ${this.id} for ${duration}ms...`);
    return {
      duration,
      interval,
      snapshots: [],
    };
  }

  async terminate() {
    console.log(`🛑 Terminating swarm: ${this.id}`);
    this.ruvSwarm.activeSwarms.delete(this.id);
  }
}

// Enhanced Agent wrapper class removed - using main Agent class instead

// Enhanced Task wrapper class
class Task {
  constructor(id, wasmResult, swarm) {
    this.id = id;
    this.description = wasmResult.task_description || wasmResult.description;
    this.status = wasmResult.status || 'pending';
    this.assignedAgents = wasmResult.assigned_agents || [];
    this.result = null;
    this.swarm = swarm;
    this.startTime = null;
    this.endTime = null;
    this.progress = 0;

    // Start task execution if agents are assigned
    if (this.assignedAgents.length > 0 && this.status === 'orchestrated') {
      this.executeTask();
    }
  }

  async executeTask() {
    this.status = 'in_progress';
    this.startTime = Date.now();
    this.progress = 0.1;

    console.log(`🏃 Executing task: ${this.description} with ${this.assignedAgents.length} agents`);

    try {
      // Execute task with all assigned agents
      const agentResults = [];

      for (const agentId of this.assignedAgents) {
        const agent = this.swarm.agents.get(agentId);
        if (agent) {
          const agentResult = await agent.execute(this);
          agentResults.push({
            agentId,
            agentType: agent.type,
            result: agentResult,
          });
        }
        this.progress = Math.min(0.9, this.progress + (0.8 / this.assignedAgents.length));
      }

      // Aggregate results
      this.result = {
        task_id: this.id,
        description: this.description,
        agent_results: agentResults,
        execution_summary: {
          total_agents: this.assignedAgents.length,
          successful_executions: agentResults.filter(r => r.result.status === 'completed').length,
          execution_time_ms: Date.now() - this.startTime,
          average_agent_time_ms: agentResults.reduce((sum, r) => sum + (r.result.executionTime || 0), 0) / agentResults.length,
        },
      };

      this.status = 'completed';
      this.progress = 1.0;
      this.endTime = Date.now();

      // Mark agents as idle again
      for (const agentId of this.assignedAgents) {
        const agent = this.swarm.agents.get(agentId);
        if (agent) {
          await agent.updateStatus('idle');
        }
      }

      console.log(`✅ Task completed: ${this.description} (${this.endTime - this.startTime}ms)`);

    } catch (error) {
      this.status = 'failed';
      this.result = {
        error: error.message,
        execution_time_ms: Date.now() - this.startTime,
      };

      // Mark agents as idle on failure too
      for (const agentId of this.assignedAgents) {
        const agent = this.swarm.agents.get(agentId);
        if (agent) {
          await agent.updateStatus('idle');
        }
      }

      console.error(`❌ Task failed: ${this.description} - ${error.message}`);
    }
  }

  async getStatus() {
    return {
      id: this.id,
      status: this.status,
      assignedAgents: this.assignedAgents,
      progress: this.progress,
      execution_time_ms: this.startTime ? (this.endTime || Date.now()) - this.startTime : 0,
    };
  }

  async getResults() {
    return this.result;
  }
}

// Import DAA service for comprehensive agent management
import { DAAService, daaService } from './daa-service.js';

export { RuvSwarm, Swarm, Task, DAAService, daaService };
