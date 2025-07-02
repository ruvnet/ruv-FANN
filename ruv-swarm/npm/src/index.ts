/**
 * RuvSwarm - WASM-based swarm orchestration with cognitive diversity
 */

import {
  SwarmOptions,
  SwarmState,
  SwarmTopology,
  SwarmEventEmitter,
  SwarmEvent,
  AgentConfig,
  Task,
  TaskStatus,
  Message,
  MessageType,
  SwarmMetrics,
  Connection,
  WasmModule,
} from './types';

import {
  generateId,
  validateSwarmOptions,
  formatMetrics,
  recommendTopology,
  priorityToNumber,
} from './utils';

import {
  BaseAgent,
  createAgent,
  AgentPool,
} from './agent';

export * from './types';
export * from './utils';
export * from './agent';
export * from './neural-network';

/**
 * Main RuvSwarm class for orchestrating swarm operations
 */
export class RuvSwarm implements SwarmEventEmitter {
  private options: Required<SwarmOptions>;
  private state: SwarmState;
  private agentPool: AgentPool;
  private eventHandlers: Map<SwarmEvent, Set<(data: any) => void>>;
  private wasmModule?: WasmModule;
  private swarmId?: number;
  private isInitialized: boolean = false;

  constructor(options: SwarmOptions = {}) {
    const errors = validateSwarmOptions(options);
    if (errors.length > 0) {
      throw new Error(`Invalid swarm options: ${errors.join(', ')}`);
    }

    this.options = {
      topology: options.topology || 'mesh',
      maxAgents: options.maxAgents || 10,
      connectionDensity: options.connectionDensity || 0.5,
      syncInterval: options.syncInterval || 1000,
      wasmPath: options.wasmPath || './wasm/ruv_swarm_wasm.js',
    };

    this.agentPool = new AgentPool();
    this.eventHandlers = new Map();

    this.state = {
      agents: new Map(),
      tasks: new Map(),
      topology: this.options.topology,
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
  }

  /**
   * Initialize the swarm with WASM module
   */
  async init(): Promise<void> {
    if (this.isInitialized) {
      throw new Error('Swarm is already initialized');
    }

    try {
      // Load WASM module
      if (typeof window !== 'undefined') {
        // Browser environment
        const wasmModule = await import(this.options.wasmPath);
        await wasmModule.default();
        this.wasmModule = wasmModule as any;
      } else {
        // Node.js environment
        console.log('WASM support in Node.js - using pure JS implementation');
      }

      // Initialize swarm in WASM if available
      if (this.wasmModule) {
        await this.wasmModule.init();
        this.swarmId = this.wasmModule.createSwarm(this.options);
      }

      this.isInitialized = true;
      this.startSyncLoop();
      
      this.emit('swarm:initialized', { options: this.options });
    } catch (error) {
      throw new Error(`Failed to initialize swarm: ${error}`);
    }
  }

  /**
   * Static factory method for easy initialization
   */
  static async create(options?: SwarmOptions): Promise<RuvSwarm> {
    const swarm = new RuvSwarm(options);
    await swarm.init();
    return swarm;
  }

  /**
   * Add an agent to the swarm
   */
  addAgent(config: AgentConfig): string {
    if (!this.isInitialized) {
      throw new Error('Swarm must be initialized before adding agents');
    }

    if (this.state.agents.size >= this.options.maxAgents) {
      throw new Error(`Maximum agent limit (${this.options.maxAgents}) reached`);
    }

    const agent = createAgent(config);
    this.state.agents.set(agent.id, agent);
    this.agentPool.addAgent(agent);

    // Add to WASM if available
    if (this.wasmModule && this.swarmId !== undefined) {
      const wasmAgentId = this.wasmModule.addAgent(this.swarmId, config);
      (agent as BaseAgent).setWasmAgentId(wasmAgentId);
    }

    // Create connections based on topology
    this.updateConnections(agent.id);

    this.emit('agent:added', { agentId: agent.id, config });
    
    return agent.id;
  }

  /**
   * Remove an agent from the swarm
   */
  removeAgent(agentId: string): void {
    const agent = this.state.agents.get(agentId);
    if (!agent) {
      throw new Error(`Agent ${agentId} not found`);
    }

    if (agent.state.status === 'busy') {
      throw new Error(`Cannot remove busy agent ${agentId}`);
    }

    this.state.agents.delete(agentId);
    this.agentPool.removeAgent(agentId);
    
    // Remove connections
    this.state.connections = this.state.connections.filter(
      conn => conn.from !== agentId && conn.to !== agentId,
    );

    this.emit('agent:removed', { agentId });
  }

  /**
   * Submit a task to the swarm
   */
  async submitTask(task: Omit<Task, 'id' | 'status'>): Promise<string> {
    if (!this.isInitialized) {
      throw new Error('Swarm must be initialized before submitting tasks');
    }

    const fullTask: Task = {
      ...task,
      id: generateId('task'),
      status: 'pending',
    };

    this.state.tasks.set(fullTask.id, fullTask);
    this.state.metrics.totalTasks++;

    this.emit('task:created', { task: fullTask });

    // Assign to WASM if available
    if (this.wasmModule && this.swarmId !== undefined) {
      this.wasmModule.assignTask(this.swarmId, fullTask);
    } else {
      // Use JS implementation
      await this.assignTask(fullTask);
    }

    return fullTask.id;
  }

  /**
   * Get the current state of a task
   */
  getTaskStatus(taskId: string): Task | undefined {
    return this.state.tasks.get(taskId);
  }

  /**
   * Get all tasks with a specific status
   */
  getTasksByStatus(status: TaskStatus): Task[] {
    return Array.from(this.state.tasks.values()).filter(task => task.status === status);
  }

  /**
   * Get current swarm metrics
   */
  getMetrics(): SwarmMetrics {
    return { ...this.state.metrics };
  }

  /**
   * Get formatted metrics string
   */
  getFormattedMetrics(): string {
    return formatMetrics(this.state.metrics);
  }

  /**
   * Event emitter implementation
   */
  on(event: SwarmEvent, handler: (data: any) => void): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(handler);
  }

  off(event: SwarmEvent, handler: (data: any) => void): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  emit(event: SwarmEvent, data: any): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in event handler for ${event}:`, error);
        }
      });
    }
  }

  /**
   * Shutdown the swarm
   */
  async destroy(): Promise<void> {
    if (!this.isInitialized) {
      return;
    }

    // Cancel any pending tasks
    for (const task of this.state.tasks.values()) {
      if (task.status === 'pending' || task.status === 'in_progress') {
        task.status = 'failed';
        task.error = new Error('Swarm shutdown');
      }
    }

    // Destroy WASM resources
    if (this.wasmModule && this.swarmId !== undefined) {
      this.wasmModule.destroy(this.swarmId);
    }

    this.isInitialized = false;
    this.emit('swarm:destroyed', {});
  }

  /**
   * Private methods
   */

  private async assignTask(task: Task): Promise<void> {
    // Find suitable agent based on task requirements
    const agent = this.agentPool.getAvailableAgent();
    
    if (!agent) {
      // No available agents, queue the task
      console.log(`No available agents for task ${task.id}, queuing...`);
      return;
    }

    task.status = 'assigned';
    task.assignedAgents = [agent.id];
    
    this.emit('task:assigned', { taskId: task.id, agentId: agent.id });

    // Send task assignment message
    const message: Message = {
      id: generateId('msg'),
      from: 'swarm',
      to: agent.id,
      type: 'task_assignment',
      payload: task,
      timestamp: Date.now(),
    };

    await agent.communicate(message);

    // Execute task
    try {
      task.status = 'in_progress';
      const startTime = Date.now();
      
      const result = await agent.execute(task);
      
      task.status = 'completed';
      task.result = result;
      
      const executionTime = Date.now() - startTime;
      this.updateMetrics(true, executionTime);
      
      this.emit('task:completed', { taskId: task.id, result });
      
    } catch (error) {
      task.status = 'failed';
      task.error = error as Error;
      
      this.updateMetrics(false, 0);
      
      this.emit('task:failed', { taskId: task.id, error });
    } finally {
      this.agentPool.releaseAgent(agent.id);
      
      // Check for pending tasks
      const pendingTasks = this.getTasksByStatus('pending');
      if (pendingTasks.length > 0) {
        // Sort by priority
        pendingTasks.sort((a, b) => priorityToNumber(b.priority) - priorityToNumber(a.priority));
        await this.assignTask(pendingTasks[0]);
      }
    }
  }

  private updateConnections(newAgentId: string): void {
    const agents = Array.from(this.state.agents.keys());
    
    switch (this.options.topology) {
    case 'mesh':
      // Connect to all other agents
      for (const agentId of agents) {
        if (agentId !== newAgentId) {
          this.state.connections.push({
            from: newAgentId,
            to: agentId,
            weight: 1,
            type: 'coordination',
          });
        }
      }
      break;
        
    case 'hierarchical':
      // Connect to parent/children based on position
      if (agents.length > 1) {
        const parentIndex = Math.floor((agents.indexOf(newAgentId) - 1) / 2);
        if (parentIndex >= 0) {
          this.state.connections.push({
            from: newAgentId,
            to: agents[parentIndex],
            weight: 1,
            type: 'control',
          });
        }
      }
      break;
        
    case 'distributed':
      // Random connections based on density
      const numConnections = Math.floor(agents.length * this.options.connectionDensity);
      const shuffled = agents.filter(id => id !== newAgentId).sort(() => Math.random() - 0.5);
      for (let i = 0; i < Math.min(numConnections, shuffled.length); i++) {
        this.state.connections.push({
          from: newAgentId,
          to: shuffled[i],
          weight: Math.random(),
          type: 'data',
        });
      }
      break;
    }
  }

  private updateMetrics(success: boolean, executionTime: number): void {
    if (success) {
      this.state.metrics.completedTasks++;
    } else {
      this.state.metrics.failedTasks++;
    }

    // Update average completion time
    if (success && executionTime > 0) {
      const totalCompleted = this.state.metrics.completedTasks;
      const currentAvg = this.state.metrics.averageCompletionTime;
      this.state.metrics.averageCompletionTime = 
        (currentAvg * (totalCompleted - 1) + executionTime) / totalCompleted;
    }

    // Update throughput (tasks per second)
    // This is a simplified calculation - in production, use a sliding window
    const totalProcessed = this.state.metrics.completedTasks + this.state.metrics.failedTasks;
    const elapsedSeconds = (Date.now() - this.startTime) / 1000;
    this.state.metrics.throughput = totalProcessed / elapsedSeconds;
  }

  private startTime: number = Date.now();

  private startSyncLoop(): void {
    setInterval(() => {
      if (this.wasmModule && this.swarmId !== undefined) {
        // Sync state with WASM
        const wasmState = this.wasmModule.getState(this.swarmId);
        // Update local state as needed
      }

      // Update agent utilization metrics
      for (const agent of this.state.agents.values()) {
        this.state.metrics.agentUtilization.set(
          agent.id,
          agent.state.status === 'busy' ? 1 : 0,
        );
      }
    }, this.options.syncInterval);
  }
}

/**
 * Default export for convenience
 */
export default RuvSwarm;