/**
 * @ruv/swarm - High-performance neural network swarm orchestration in WebAssembly
 * Enhanced version with progressive WASM loading and full feature set
 */

// Re-export all types from the enhanced definitions
export * from './index-enhanced';

export interface InitOptions {
  /** Path to WASM files */
  wasmPath?: string;
  /** Use SIMD optimizations if available */
  useSIMD?: boolean;
  /** Enable debug logging */
  debug?: boolean;
}

export interface SwarmConfig {
  /** Name of the swarm */
  name: string;
  /** Swarm strategy */
  strategy: 'research' | 'development' | 'analysis' | 'testing' | 'optimization' | 'maintenance';
  /** Coordination mode */
  mode: 'centralized' | 'distributed' | 'hierarchical' | 'mesh' | 'hybrid';
  /** Maximum number of agents */
  maxAgents?: number;
  /** Enable parallel execution */
  parallel?: boolean;
  /** Enable monitoring */
  monitor?: boolean;
}

export interface AgentConfig {
  /** Agent name */
  name: string;
  /** Agent type */
  type: 'researcher' | 'coder' | 'analyst' | 'optimizer' | 'coordinator';
  /** Agent capabilities */
  capabilities?: string[];
  /** Maximum concurrent tasks */
  maxConcurrentTasks?: number;
  /** Memory limit in bytes */
  memoryLimit?: number;
}

export interface TaskConfig {
  /** Task ID */
  id: string;
  /** Task description */
  description: string;
  /** Task priority */
  priority: 'low' | 'medium' | 'high';
  /** Task dependencies */
  dependencies: string[];
  /** Additional metadata */
  metadata?: Record<string, any>;
}

export interface TaskRequest {
  /** Task ID */
  id: string;
  /** Task description */
  description: string;
  /** Task parameters */
  parameters?: any;
  /** Timeout in milliseconds */
  timeout?: number;
}

export interface TaskResponse {
  /** Task ID */
  taskId: string;
  /** Execution status */
  status: 'completed' | 'failed' | 'timeout';
  /** Task result */
  result: any;
  /** Execution time in seconds */
  executionTime: number;
  /** Error message if failed */
  error?: string;
}

export interface OrchestrationResult {
  /** Task ID */
  taskId: string;
  /** Overall status */
  status: string;
  /** Results from each agent */
  results: AgentResult[];
  /** Orchestration metrics */
  metrics: OrchestrationMetrics;
}

export interface AgentResult {
  /** Agent ID */
  agentId: string;
  /** Agent type */
  agentType: string;
  /** Agent output */
  output: any;
  /** Execution time in seconds */
  executionTime: number;
}

export interface OrchestrationMetrics {
  /** Total orchestration time in seconds */
  totalTime: number;
  /** Number of agents spawned */
  agentsSpawned: number;
  /** Number of tasks completed */
  tasksCompleted: number;
  /** Memory usage in MB */
  memoryUsage: number;
}

export interface AgentMetrics {
  /** Number of tasks completed */
  tasksCompleted: number;
  /** Number of tasks failed */
  tasksFailed: number;
  /** Average execution time in seconds */
  averageExecutionTime: number;
  /** Memory usage in bytes */
  memoryUsage: number;
  /** CPU usage percentage */
  cpuUsage: number;
}

export interface RuntimeFeatures {
  /** SIMD support available */
  simdAvailable: boolean;
  /** Threading support available */
  threadsAvailable: boolean;
  /** Memory limit in bytes */
  memoryLimit: number;
}

/**
 * JavaScript Agent interface for swarm orchestration
 */
export declare class JsAgent {
  /** Agent ID */
  readonly id: string;
  /** Agent type */
  readonly agentType: string;
  /** Current status */
  readonly status: string;
  /** Number of completed tasks */
  readonly tasksCompleted: number;

  /**
   * Execute a task
   * @param task Task request configuration
   * @returns Task response with results
   */
  execute(task: TaskRequest): Promise<TaskResponse>;

  /**
   * Get agent metrics
   * @returns Current agent metrics
   */
  getMetrics(): AgentMetrics;

  /**
   * Get agent capabilities
   * @returns List of agent capabilities
   */
  getCapabilities(): string[];

  /**
   * Reset agent state
   */
  reset(): void;
}

/**
 * Main RuvSwarm class for neural network swarm orchestration
 */
export declare class RuvSwarm {
  /** Swarm name */
  readonly name: string;
  /** Number of active agents */
  readonly agentCount: number;
  /** Maximum number of agents */
  readonly maxAgents: number;

  /**
   * Create a new RuvSwarm instance
   * @param config Swarm configuration
   */
  constructor(config: SwarmConfig);

  /**
   * Initialize RuvSwarm with WASM module
   * @param options Initialization options
   * @returns Initialized RuvSwarm instance
   */
  static initialize(options?: InitOptions): Promise<RuvSwarm>;

  /**
   * Detect SIMD support in the current environment
   * @returns True if SIMD is supported
   */
  static detectSIMDSupport(): boolean;

  /**
   * Get runtime features
   * @returns Runtime feature detection results
   */
  static getRuntimeFeatures(): RuntimeFeatures;

  /**
   * Get library version
   * @returns Version string
   */
  static getVersion(): string;

  /**
   * Spawn a new agent
   * @param config Agent configuration
   * @returns Spawned agent instance
   */
  spawn(config: AgentConfig): Promise<JsAgent>;

  /**
   * Orchestrate a task across the swarm
   * @param task Task configuration
   * @returns Orchestration results
   */
  orchestrate(task: TaskConfig): Promise<OrchestrationResult>;

  /**
   * Get list of active agents
   * @returns Array of agent IDs
   */
  getAgents(): string[];

  /**
   * Get swarm status
   * @returns Current swarm status
   */
  getStatus(): {
    name: string;
    strategy: string;
    mode: string;
    agents: string[];
    agentCount: number;
    maxAgents: number;
  };

  /**
   * Get WASM memory usage
   * @returns Memory usage in bytes
   */
  static getMemoryUsage(): number;
}

/**
 * Performance timer utility
 */
export declare class PerformanceTimer {
  /**
   * Create a new performance timer
   * @param name Timer name
   */
  constructor(name: string);

  /**
   * Get elapsed time in milliseconds
   * @returns Elapsed time
   */
  elapsed(): number;

  /**
   * Log elapsed time to console
   */
  log(): void;
}

// Re-export utility functions
export declare function consoleLog(message: string): void;
export declare function consoleError(message: string): void;
export declare function consoleWarn(message: string): void;
export declare function formatJsError(error: any): string;

// Export DAA service types and interfaces
export * from './daa-service';