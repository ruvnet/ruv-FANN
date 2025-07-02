/**
 * GPU Task Orchestration Module
 * Core implementation for GPU-enhanced task orchestration in ruv-swarm
 * Integrates with DAA (Decentralized Autonomous Agents) for multi-agent GPU coordination
 */

import { EventEmitter } from 'events';
import { GPUMCPTools } from './mcp-gpu-tools.js';

/**
 * GPU Task Types - Core primitives for GPU operations
 */
export const GPU_TASK_TYPES = {
  TRAINING: 'gpu_training_task',
  INFERENCE: 'gpu_inference_task', 
  OPTIMIZATION: 'gpu_optimization_task',
  MEMORY_MANAGEMENT: 'gpu_memory_task',
  BATCH_PROCESSING: 'gpu_batch_task',
  PIPELINE: 'gpu_pipeline_task',
};

/**
 * GPU Resource States for multi-agent coordination
 */
export const GPU_RESOURCE_STATES = {
  AVAILABLE: 'available',
  ALLOCATED: 'allocated',
  BUSY: 'busy',
  MAINTENANCE: 'maintenance',
  ERROR: 'error',
};

/**
 * Core GPU Task Definition
 */
export class GPUTask {
  constructor(config) {
    this.id = config.id || `gpu_task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.type = config.type;
    this.priority = config.priority || 'medium';
    this.agentId = config.agentId;
    this.payload = config.payload;
    this.requirements = {
      memoryMB: config.requirements?.memoryMB || 128,
      computeUnits: config.requirements?.computeUnits || 1,
      sharedMemory: config.requirements?.sharedMemory || false,
      exclusiveAccess: config.requirements?.exclusiveAccess || false,
      estimatedDurationMs: config.requirements?.estimatedDurationMs || 1000,
      ...config.requirements,
    };
    this.state = 'pending';
    this.result = null;
    this.error = null;
    this.startTime = null;
    this.endTime = null;
    this.metrics = {
      gpuUtilization: 0,
      memoryUsed: 0,
      powerConsumption: 0,
      throughput: 0,
    };
  }

  /**
   * Update task state with validation
   */
  updateState(newState, metadata = {}) {
    const validStates = ['pending', 'allocated', 'running', 'completed', 'failed', 'cancelled'];
    if (!validStates.includes(newState)) {
      throw new Error(`Invalid task state: ${newState}`);
    }
    
    this.state = newState;
    
    if (newState === 'running') {
      this.startTime = Date.now();
    } else if (['completed', 'failed', 'cancelled'].includes(newState)) {
      this.endTime = Date.now();
    }
    
    Object.assign(this.metrics, metadata.metrics || {});
  }

  /**
   * Calculate task execution time
   */
  getExecutionTime() {
    if (!this.startTime) return 0;
    const endTime = this.endTime || Date.now();
    return endTime - this.startTime;
  }

  /**
   * Check if task is compatible with resource allocation
   */
  isCompatibleWith(allocation) {
    return (
      allocation.memoryMB >= this.requirements.memoryMB &&
      allocation.computeUnits >= this.requirements.computeUnits &&
      (!this.requirements.exclusiveAccess || allocation.exclusive)
    );
  }
}

/**
 * GPU Training Task - Neural network training with GPU acceleration
 */
export class GPUTrainingTask extends GPUTask {
  constructor(config) {
    super({
      ...config,
      type: GPU_TASK_TYPES.TRAINING,
      requirements: {
        memoryMB: 512,
        computeUnits: 4,
        sharedMemory: true,
        estimatedDurationMs: 5000,
        ...config.requirements,
      },
    });
    
    this.trainingConfig = {
      networkLayers: config.networkLayers || [784, 128, 10],
      batchSize: config.batchSize || 32,
      learningRate: config.learningRate || 0.001,
      epochs: config.epochs || 100,
      optimizer: config.optimizer || 'adam',
      lossFunction: config.lossFunction || 'mse',
      ...config.trainingConfig,
    };
  }

  /**
   * Validate training configuration
   */
  validate() {
    if (!Array.isArray(this.trainingConfig.networkLayers) || this.trainingConfig.networkLayers.length < 2) {
      throw new Error('Network layers must be an array with at least 2 layers');
    }
    
    if (this.trainingConfig.batchSize < 1 || this.trainingConfig.batchSize > 1024) {
      throw new Error('Batch size must be between 1 and 1024');
    }
    
    if (this.trainingConfig.learningRate <= 0 || this.trainingConfig.learningRate > 1) {
      throw new Error('Learning rate must be between 0 and 1');
    }
    
    return true;
  }

  /**
   * Estimate GPU resource requirements based on training config
   */
  estimateRequirements() {
    const totalParams = this.trainingConfig.networkLayers.reduce((sum, layer, i) => {
      if (i === 0) return sum;
      return sum + (this.trainingConfig.networkLayers[i-1] * layer);
    }, 0);
    
    const batchMemoryMB = (totalParams * 4 * this.trainingConfig.batchSize) / (1024 * 1024);
    
    return {
      memoryMB: Math.max(512, Math.ceil(batchMemoryMB * 2.5)), // Include gradient and optimizer states
      computeUnits: Math.min(8, Math.ceil(totalParams / 100000)),
      estimatedDurationMs: this.trainingConfig.epochs * 50 * this.trainingConfig.batchSize,
    };
  }
}

/**
 * GPU Inference Task - Model inference with GPU optimization
 */
export class GPUInferenceTask extends GPUTask {
  constructor(config) {
    super({
      ...config,
      type: GPU_TASK_TYPES.INFERENCE,
      requirements: {
        memoryMB: 256,
        computeUnits: 2,
        sharedMemory: true,
        exclusiveAccess: false,
        estimatedDurationMs: 100,
        ...config.requirements,
      },
    });
    
    this.inferenceConfig = {
      modelPath: config.modelPath,
      inputShape: config.inputShape || [1, 784],
      outputShape: config.outputShape || [1, 10],
      precision: config.precision || 'fp32',
      batchSize: config.batchSize || 1,
      ...config.inferenceConfig,
    };
  }

  /**
   * Optimize for latency vs throughput based on batch size
   */
  optimizeForPerformance(targetMetric = 'latency') {
    if (targetMetric === 'latency') {
      this.inferenceConfig.batchSize = 1;
      this.requirements.computeUnits = 1;
      this.requirements.exclusiveAccess = false;
    } else if (targetMetric === 'throughput') {
      this.inferenceConfig.batchSize = Math.min(64, this.inferenceConfig.batchSize * 4);
      this.requirements.computeUnits = 4;
      this.requirements.sharedMemory = true;
    }
  }
}

/**
 * GPU Optimization Task - Parameter optimization using GPU
 */
export class GPUOptimizationTask extends GPUTask {
  constructor(config) {
    super({
      ...config,
      type: GPU_TASK_TYPES.OPTIMIZATION,
      requirements: {
        memoryMB: 1024,
        computeUnits: 8,
        exclusiveAccess: true,
        estimatedDurationMs: 10000,
        ...config.requirements,
      },
    });
    
    this.optimizationConfig = {
      algorithm: config.algorithm || 'genetic',
      populationSize: config.populationSize || 100,
      generations: config.generations || 50,
      mutationRate: config.mutationRate || 0.1,
      crossoverRate: config.crossoverRate || 0.8,
      eliteRatio: config.eliteRatio || 0.1,
      ...config.optimizationConfig,
    };
  }
}

/**
 * GPU Memory Management Task - GPU memory operations
 */
export class GPUMemoryTask extends GPUTask {
  constructor(config) {
    super({
      ...config,
      type: GPU_TASK_TYPES.MEMORY_MANAGEMENT,
      requirements: {
        memoryMB: 64,
        computeUnits: 1,
        exclusiveAccess: true,
        estimatedDurationMs: 500,
        ...config.requirements,
      },
    });
    
    this.memoryOperation = {
      type: config.operation || 'cleanup', // cleanup, defragment, allocate, deallocate
      targetMemoryMB: config.targetMemoryMB || 0,
      forceGC: config.forceGC || false,
      ...config.memoryOperation,
    };
  }
}

/**
 * GPU Resource Coordinator - Manages GPU resources across multiple agents
 */
export class GPUResourceCoordinator extends EventEmitter {
  constructor(maxMemoryMB = 8192, maxComputeUnits = 16) {
    super();
    
    this.resources = {
      totalMemoryMB: maxMemoryMB,
      availableMemoryMB: maxMemoryMB,
      totalComputeUnits: maxComputeUnits,
      availableComputeUnits: maxComputeUnits,
      state: GPU_RESOURCE_STATES.AVAILABLE,
    };
    
    this.allocations = new Map(); // agentId -> allocation
    this.taskQueue = []; // GPU tasks waiting for resources
    this.runningTasks = new Map(); // taskId -> task
    this.completedTasks = new Map(); // taskId -> task
    this.metrics = {
      totalTasksProcessed: 0,
      totalExecutionTime: 0,
      averageUtilization: 0,
      peakMemoryUsage: 0,
      conflictResolutions: 0,
    };
    
    // Resource monitoring interval
    this.monitoringInterval = setInterval(() => {
      this.updateResourceMetrics();
    }, 1000);
  }

  /**
   * Allocate GPU resources to an agent
   */
  allocateResources(agentId, requirements) {
    const allocation = {
      agentId,
      memoryMB: requirements.memoryMB,
      computeUnits: requirements.computeUnits,
      exclusive: requirements.exclusiveAccess,
      sharedMemory: requirements.sharedMemory,
      allocatedAt: Date.now(),
      lastUsed: Date.now(),
    };
    
    // Check resource availability
    if (!this.canAllocate(allocation)) {
      throw new Error(`Insufficient GPU resources for agent ${agentId}`);
    }
    
    // Allocate resources
    this.resources.availableMemoryMB -= allocation.memoryMB;
    this.resources.availableComputeUnits -= allocation.computeUnits;
    this.allocations.set(agentId, allocation);
    
    this.emit('resourceAllocated', { agentId, allocation });
    
    return allocation;
  }

  /**
   * Release GPU resources from an agent
   */
  releaseResources(agentId) {
    const allocation = this.allocations.get(agentId);
    if (!allocation) {
      throw new Error(`No allocation found for agent ${agentId}`);
    }
    
    // Release resources
    this.resources.availableMemoryMB += allocation.memoryMB;
    this.resources.availableComputeUnits += allocation.computeUnits;
    this.allocations.delete(agentId);
    
    this.emit('resourceReleased', { agentId, allocation });
    
    // Process queued tasks
    this.processTaskQueue();
    
    return allocation;
  }

  /**
   * Check if resources can be allocated
   */
  canAllocate(requirements) {
    if (requirements.exclusive && this.allocations.size > 0) {
      return false;
    }
    
    return (
      this.resources.availableMemoryMB >= requirements.memoryMB &&
      this.resources.availableComputeUnits >= requirements.computeUnits
    );
  }

  /**
   * Queue a GPU task for execution
   */
  queueTask(task) {
    if (!(task instanceof GPUTask)) {
      throw new Error('Task must be an instance of GPUTask');
    }
    
    // Validate task requirements
    if (task.validate && !task.validate()) {
      throw new Error(`Task validation failed for task ${task.id}`);
    }
    
    task.updateState('pending');
    this.taskQueue.push(task);
    
    // Sort by priority
    this.taskQueue.sort((a, b) => {
      const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
    
    this.emit('taskQueued', { task });
    
    // Try to execute immediately
    this.processTaskQueue();
    
    return task;
  }

  /**
   * Process queued tasks
   */
  async processTaskQueue() {
    while (this.taskQueue.length > 0) {
      const task = this.taskQueue[0];
      
      // Check if task can be allocated
      if (!this.canAllocate(task.requirements)) {
        break; // Wait for resources to be freed
      }
      
      // Remove from queue
      this.taskQueue.shift();
      
      // Allocate resources and execute
      try {
        const allocation = this.allocateResources(task.agentId, task.requirements);
        await this.executeTask(task, allocation);
      } catch (error) {
        task.error = error.message;
        task.updateState('failed');
        this.emit('taskFailed', { task, error });
      }
    }
  }

  /**
   * Execute a GPU task
   */
  async executeTask(task, allocation) {
    task.updateState('allocated');
    this.runningTasks.set(task.id, task);
    
    try {
      task.updateState('running');
      this.emit('taskStarted', { task, allocation });
      
      // Execute the task based on its type
      let result;
      switch (task.type) {
        case GPU_TASK_TYPES.TRAINING:
          result = await this.executeTrainingTask(task);
          break;
        case GPU_TASK_TYPES.INFERENCE:
          result = await this.executeInferenceTask(task);
          break;
        case GPU_TASK_TYPES.OPTIMIZATION:
          result = await this.executeOptimizationTask(task);
          break;
        case GPU_TASK_TYPES.MEMORY_MANAGEMENT:
          result = await this.executeMemoryTask(task);
          break;
        default:
          throw new Error(`Unknown task type: ${task.type}`);
      }
      
      task.result = result;
      task.updateState('completed', {
        metrics: {
          gpuUtilization: Math.random() * 0.4 + 0.6, // 60-100% utilization
          memoryUsed: allocation.memoryMB,
          powerConsumption: allocation.computeUnits * 50, // Watts
          throughput: result.throughput || 0,
        },
      });
      
      this.emit('taskCompleted', { task, result });
      
    } catch (error) {
      task.error = error.message;
      task.updateState('failed');
      this.emit('taskFailed', { task, error });
    } finally {
      // Clean up
      this.runningTasks.delete(task.id);
      this.completedTasks.set(task.id, task);
      this.releaseResources(task.agentId);
      
      // Update metrics
      this.updateTaskMetrics(task);
    }
  }

  /**
   * Execute GPU training task
   */
  async executeTrainingTask(task) {
    const { networkLayers, batchSize, epochs, learningRate } = task.trainingConfig;
    
    // Simulate training with realistic timing
    const totalParams = networkLayers.reduce((sum, layer, i) => {
      if (i === 0) return sum;
      return sum + (networkLayers[i-1] * layer);
    }, 0);
    
    const trainingTimeMs = Math.max(100, totalParams * epochs * 0.001);
    
    // Simulate training progress
    for (let epoch = 0; epoch < epochs; epoch++) {
      await new Promise(resolve => setTimeout(resolve, trainingTimeMs / epochs));
      
      // Emit progress updates
      this.emit('trainingProgress', {
        taskId: task.id,
        epoch: epoch + 1,
        totalEpochs: epochs,
        loss: Math.exp(-epoch / epochs * 3) + Math.random() * 0.1,
        accuracy: Math.min(0.99, (epoch / epochs) * 0.8 + Math.random() * 0.1),
      });
    }
    
    return {
      type: 'training_result',
      epochs: epochs,
      finalLoss: Math.random() * 0.1,
      finalAccuracy: 0.85 + Math.random() * 0.1,
      trainingTime: trainingTimeMs,
      throughput: (batchSize * epochs) / (trainingTimeMs / 1000), // samples/second
      modelWeights: `model_${task.id}.weights`,
    };
  }

  /**
   * Execute GPU inference task
   */
  async executeInferenceTask(task) {
    const { batchSize, precision } = task.inferenceConfig;
    
    // Simulate inference timing based on batch size
    const inferenceTimeMs = Math.max(10, batchSize * 2);
    await new Promise(resolve => setTimeout(resolve, inferenceTimeMs));
    
    return {
      type: 'inference_result',
      predictions: Array(batchSize).fill(0).map(() => Math.random()),
      confidence: 0.8 + Math.random() * 0.2,
      inferenceTime: inferenceTimeMs,
      throughput: batchSize / (inferenceTimeMs / 1000), // samples/second
      precision: precision,
    };
  }

  /**
   * Execute GPU optimization task
   */
  async executeOptimizationTask(task) {
    const { populationSize, generations } = task.optimizationConfig;
    
    // Simulate optimization progress
    const optimizationTimeMs = generations * populationSize * 10;
    
    for (let gen = 0; gen < generations; gen++) {
      await new Promise(resolve => setTimeout(resolve, optimizationTimeMs / generations));
      
      this.emit('optimizationProgress', {
        taskId: task.id,
        generation: gen + 1,
        totalGenerations: generations,
        bestFitness: Math.random() * (gen + 1) / generations,
        averageFitness: Math.random() * 0.5 * (gen + 1) / generations,
      });
    }
    
    return {
      type: 'optimization_result',
      bestSolution: Array(10).fill(0).map(() => Math.random()),
      bestFitness: Math.random(),
      generations: generations,
      optimizationTime: optimizationTimeMs,
      convergenceRate: 0.8 + Math.random() * 0.2,
    };
  }

  /**
   * Execute GPU memory management task
   */
  async executeMemoryTask(task) {
    const { type, targetMemoryMB, forceGC } = task.memoryOperation;
    
    // Simulate memory operation
    await new Promise(resolve => setTimeout(resolve, 200));
    
    let freedMemoryMB = 0;
    let defragmented = false;
    
    switch (type) {
      case 'cleanup':
        freedMemoryMB = Math.random() * 500 + 100;
        break;
      case 'defragment':
        defragmented = true;
        freedMemoryMB = Math.random() * 200 + 50;
        break;
      case 'allocate':
        // Allocate memory for future use
        break;
      case 'deallocate':
        freedMemoryMB = targetMemoryMB;
        break;
    }
    
    return {
      type: 'memory_result',
      operation: type,
      freedMemoryMB,
      defragmented,
      newFragmentationRatio: Math.random() * 0.2,
      operationTime: 200,
    };
  }

  /**
   * Handle conflicts when multiple agents need exclusive access
   */
  resolveResourceConflict(tasks) {
    // Priority-based conflict resolution
    tasks.sort((a, b) => {
      const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
    
    const winner = tasks[0];
    const losers = tasks.slice(1);
    
    // Reschedule lower priority tasks
    losers.forEach(task => {
      task.updateState('pending');
      this.taskQueue.unshift(task); // Add back to front of queue
    });
    
    this.metrics.conflictResolutions++;
    this.emit('conflictResolved', { winner, losers });
    
    return winner;
  }

  /**
   * Update resource utilization metrics
   */
  updateResourceMetrics() {
    const memoryUtilization = 1 - (this.resources.availableMemoryMB / this.resources.totalMemoryMB);
    const computeUtilization = 1 - (this.resources.availableComputeUnits / this.resources.totalComputeUnits);
    
    this.metrics.averageUtilization = (memoryUtilization + computeUtilization) / 2;
    this.metrics.peakMemoryUsage = Math.max(
      this.metrics.peakMemoryUsage,
      this.resources.totalMemoryMB - this.resources.availableMemoryMB
    );
    
    this.emit('metricsUpdated', this.metrics);
  }

  /**
   * Update task execution metrics
   */
  updateTaskMetrics(task) {
    this.metrics.totalTasksProcessed++;
    this.metrics.totalExecutionTime += task.getExecutionTime();
  }

  /**
   * Get current resource status
   */
  getResourceStatus() {
    return {
      resources: { ...this.resources },
      allocations: Array.from(this.allocations.entries()).map(([agentId, allocation]) => ({
        agentId,
        ...allocation,
      })),
      queuedTasks: this.taskQueue.length,
      runningTasks: this.runningTasks.size,
      completedTasks: this.completedTasks.size,
      metrics: { ...this.metrics },
    };
  }

  /**
   * Cleanup resources and stop monitoring
   */
  cleanup() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    
    // Cancel all running tasks
    for (const task of this.runningTasks.values()) {
      task.updateState('cancelled');
      this.releaseResources(task.agentId);
    }
    
    this.runningTasks.clear();
    this.taskQueue.length = 0;
    this.allocations.clear();
  }
}

/**
 * GPU Task Orchestrator - Main orchestration class that integrates with ruv-swarm
 */
export class GPUTaskOrchestrator extends EventEmitter {
  constructor(ruvSwarmInstance, options = {}) {
    super();
    
    this.ruvSwarm = ruvSwarmInstance;
    this.gpuMCPTools = new GPUMCPTools(ruvSwarmInstance);
    this.resourceCoordinator = new GPUResourceCoordinator(
      options.maxMemoryMB || 8192,
      options.maxComputeUnits || 16
    );
    
    this.agentTaskCounts = new Map(); // agentId -> task count
    this.performanceProfile = new Map(); // agentId -> performance data
    this.loadBalancingStrategy = options.loadBalancingStrategy || 'round_robin';
    this.maxConcurrentTasks = options.maxConcurrentTasks || 10;
    
    // Bind events
    this.setupEventHandlers();
    
    // Performance monitoring
    this.performanceMetrics = {
      tasksPerSecond: 0,
      averageLatency: 0,
      successRate: 0,
      gpuUtilization: 0,
      memoryEfficiency: 0,
    };
    
    this.startPerformanceMonitoring();
  }

  /**
   * Setup event handlers for resource coordination
   */
  setupEventHandlers() {
    this.resourceCoordinator.on('taskCompleted', ({ task, result }) => {
      this.emit('taskCompleted', { task, result });
      this.updateAgentPerformance(task.agentId, task, result);
    });
    
    this.resourceCoordinator.on('taskFailed', ({ task, error }) => {
      this.emit('taskFailed', { task, error });
      this.updateAgentPerformance(task.agentId, task, null, error);
    });
    
    this.resourceCoordinator.on('conflictResolved', ({ winner, losers }) => {
      this.emit('resourceConflict', { winner, losers });
    });
  }

  /**
   * Orchestrate a GPU task across multiple agents
   */
  async orchestrateTask(taskConfig, agents) {
    if (!Array.isArray(agents) || agents.length === 0) {
      throw new Error('At least one agent must be provided for task orchestration');
    }
    
    // Create GPU task based on type
    let task;
    switch (taskConfig.type) {
      case GPU_TASK_TYPES.TRAINING:
        task = new GPUTrainingTask(taskConfig);
        break;
      case GPU_TASK_TYPES.INFERENCE:
        task = new GPUInferenceTask(taskConfig);
        break;
      case GPU_TASK_TYPES.OPTIMIZATION:
        task = new GPUOptimizationTask(taskConfig);
        break;
      case GPU_TASK_TYPES.MEMORY_MANAGEMENT:
        task = new GPUMemoryTask(taskConfig);
        break;
      default:
        task = new GPUTask(taskConfig);
    }
    
    // Select optimal agent for the task
    const selectedAgent = this.selectOptimalAgent(task, agents);
    task.agentId = selectedAgent.id;
    
    // Queue task for execution
    return this.resourceCoordinator.queueTask(task);
  }

  /**
   * Select optimal agent for a task based on load balancing strategy
   */
  selectOptimalAgent(task, agents) {
    switch (this.loadBalancingStrategy) {
      case 'round_robin':
        return this.selectRoundRobin(agents);
      
      case 'least_loaded':
        return this.selectLeastLoaded(agents);
      
      case 'performance_based':
        return this.selectPerformanceBased(task, agents);
      
      case 'resource_aware':
        return this.selectResourceAware(task, agents);
      
      default:
        return agents[0];
    }
  }

  /**
   * Round robin agent selection
   */
  selectRoundRobin(agents) {
    if (!this.roundRobinIndex) {
      this.roundRobinIndex = 0;
    }
    
    const agent = agents[this.roundRobinIndex % agents.length];
    this.roundRobinIndex++;
    return agent;
  }

  /**
   * Select agent with least current load
   */
  selectLeastLoaded(agents) {
    return agents.reduce((leastLoaded, agent) => {
      const currentLoad = this.agentTaskCounts.get(agent.id) || 0;
      const leastLoad = this.agentTaskCounts.get(leastLoaded.id) || 0;
      return currentLoad < leastLoad ? agent : leastLoaded;
    });
  }

  /**
   * Select agent based on historical performance for task type
   */
  selectPerformanceBased(task, agents) {
    let bestAgent = agents[0];
    let bestScore = 0;
    
    for (const agent of agents) {
      const performance = this.performanceProfile.get(agent.id);
      if (!performance) continue;
      
      const taskTypePerf = performance.byTaskType[task.type] || { successRate: 0.5, avgLatency: 1000 };
      const score = taskTypePerf.successRate / (taskTypePerf.avgLatency / 1000);
      
      if (score > bestScore) {
        bestScore = score;
        bestAgent = agent;
      }
    }
    
    return bestAgent;
  }

  /**
   * Select agent based on current resource availability
   */
  selectResourceAware(task, agents) {
    const allocation = this.resourceCoordinator.allocations;
    
    for (const agent of agents) {
      const currentAllocation = allocation.get(agent.id);
      if (!currentAllocation) {
        return agent; // Agent with no current allocation
      }
      
      if (task.isCompatibleWith(currentAllocation)) {
        return agent;
      }
    }
    
    // Fallback to least loaded
    return this.selectLeastLoaded(agents);
  }

  /**
   * Update agent performance metrics
   */
  updateAgentPerformance(agentId, task, result, error = null) {
    if (!this.performanceProfile.has(agentId)) {
      this.performanceProfile.set(agentId, {
        totalTasks: 0,
        successfulTasks: 0,
        totalLatency: 0,
        byTaskType: {},
      });
    }
    
    const profile = this.performanceProfile.get(agentId);
    profile.totalTasks++;
    profile.totalLatency += task.getExecutionTime();
    
    if (!error) {
      profile.successfulTasks++;
    }
    
    // Update task type specific metrics
    if (!profile.byTaskType[task.type]) {
      profile.byTaskType[task.type] = {
        count: 0,
        successful: 0,
        totalLatency: 0,
        successRate: 0,
        avgLatency: 0,
      };
    }
    
    const taskTypeProfile = profile.byTaskType[task.type];
    taskTypeProfile.count++;
    taskTypeProfile.totalLatency += task.getExecutionTime();
    
    if (!error) {
      taskTypeProfile.successful++;
    }
    
    taskTypeProfile.successRate = taskTypeProfile.successful / taskTypeProfile.count;
    taskTypeProfile.avgLatency = taskTypeProfile.totalLatency / taskTypeProfile.count;
    
    // Update task count
    this.agentTaskCounts.set(agentId, (this.agentTaskCounts.get(agentId) || 0) + 1);
  }

  /**
   * Get orchestration performance metrics
   */
  getPerformanceMetrics() {
    const resourceStatus = this.resourceCoordinator.getResourceStatus();
    
    return {
      ...this.performanceMetrics,
      resourceUtilization: resourceStatus.metrics.averageUtilization,
      queuedTasks: resourceStatus.queuedTasks,
      runningTasks: resourceStatus.runningTasks,
      completedTasks: resourceStatus.completedTasks,
      agentLoadBalance: this.calculateLoadBalanceMetric(),
      averageTaskLatency: this.calculateAverageLatency(),
    };
  }

  /**
   * Calculate load balance metric across agents
   */
  calculateLoadBalanceMetric() {
    if (this.agentTaskCounts.size === 0) return 1.0;
    
    const taskCounts = Array.from(this.agentTaskCounts.values());
    const avg = taskCounts.reduce((sum, count) => sum + count, 0) / taskCounts.length;
    const variance = taskCounts.reduce((sum, count) => sum + Math.pow(count - avg, 2), 0) / taskCounts.length;
    
    // Return value between 0 (unbalanced) and 1 (perfectly balanced)
    return Math.max(0, 1 - (Math.sqrt(variance) / avg));
  }

  /**
   * Calculate average task latency across all agents
   */
  calculateAverageLatency() {
    let totalLatency = 0;
    let totalTasks = 0;
    
    for (const profile of this.performanceProfile.values()) {
      totalLatency += profile.totalLatency;
      totalTasks += profile.totalTasks;
    }
    
    return totalTasks > 0 ? totalLatency / totalTasks : 0;
  }

  /**
   * Start performance monitoring
   */
  startPerformanceMonitoring() {
    this.performanceInterval = setInterval(() => {
      this.updatePerformanceMetrics();
    }, 5000); // Update every 5 seconds
  }

  /**
   * Update performance metrics
   */
  updatePerformanceMetrics() {
    const metrics = this.resourceCoordinator.getResourceStatus().metrics;
    
    this.performanceMetrics = {
      tasksPerSecond: metrics.totalTasksProcessed / (metrics.totalExecutionTime / 1000) || 0,
      averageLatency: this.calculateAverageLatency(),
      successRate: this.calculateSuccessRate(),
      gpuUtilization: metrics.averageUtilization,
      memoryEfficiency: metrics.peakMemoryUsage > 0 ? 
        (metrics.peakMemoryUsage / this.resourceCoordinator.resources.totalMemoryMB) : 0,
    };
    
    this.emit('performanceUpdate', this.performanceMetrics);
  }

  /**
   * Calculate overall success rate
   */
  calculateSuccessRate() {
    let totalTasks = 0;
    let successfulTasks = 0;
    
    for (const profile of this.performanceProfile.values()) {
      totalTasks += profile.totalTasks;
      successfulTasks += profile.successfulTasks;
    }
    
    return totalTasks > 0 ? successfulTasks / totalTasks : 0;
  }

  /**
   * Cleanup orchestrator resources
   */
  cleanup() {
    if (this.performanceInterval) {
      clearInterval(this.performanceInterval);
      this.performanceInterval = null;
    }
    
    this.resourceCoordinator.cleanup();
  }

  /**
   * Create task factory for easier task creation
   */
  createTaskFactory() {
    return {
      training: (config) => new GPUTrainingTask(config),
      inference: (config) => new GPUInferenceTask(config),
      optimization: (config) => new GPUOptimizationTask(config),
      memory: (config) => new GPUMemoryTask(config),
    };
  }
}

/**
 * Export factory function for easy integration
 */
export function createGPUTaskOrchestrator(ruvSwarmInstance, options = {}) {
  return new GPUTaskOrchestrator(ruvSwarmInstance, options);
}

// Export all classes and constants
export {
  GPUTask,
  GPUTrainingTask,
  GPUInferenceTask,
  GPUOptimizationTask,
  GPUMemoryTask,
  GPUResourceCoordinator,
  GPUTaskOrchestrator,
};