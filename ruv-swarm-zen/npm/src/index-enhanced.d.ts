/**
 * TypeScript definitions for enhanced ruv-swarm NPX package
 * Complete WASM integration with neural networks and forecasting
 */

declare module 'ruv-swarm' {
  // Main RuvSwarm class
  export class RuvSwarm {
    constructor();
    
    static initialize(options?: RuvSwarmOptions): Promise<RuvSwarm>;
    static detectSIMDSupport(): boolean;
    static getVersion(): string;
    static getMemoryUsage(): MemoryUsage | null;
    static getRuntimeFeatures(): RuntimeFeatures;
    
    wasmLoader: WasmModuleLoader;
    persistence: SwarmPersistence | null;
    activeSwarms: Map<string, Swarm>;
    globalAgents: Map<string, Agent>;
    metrics: GlobalMetrics;
    features: SwarmFeatures;
    
    detectFeatures(useSIMD?: boolean): Promise<void>;
    createSwarm(config: SwarmConfig): Promise<Swarm>;
    getSwarmStatus(swarmId: string, detailed?: boolean): Promise<SwarmStatus>;
    getAllSwarms(): Promise<SwarmInfo[]>;
    getGlobalMetrics(): Promise<GlobalMetrics>;
    spawnAgent(name: string, type?: AgentType, options?: any): Promise<Agent>;
  }

  // Configuration options
  export interface RuvSwarmOptions {
    wasmPath?: string;
    loadingStrategy?: 'eager' | 'on-demand' | 'progressive';
    enablePersistence?: boolean;
    enableNeuralNetworks?: boolean;
    enableForecasting?: boolean;
    useSIMD?: boolean;
    debug?: boolean;
  }

  // Swarm configuration
  export interface SwarmConfig {
    name?: string;
    topology?: 'mesh' | 'star' | 'hierarchical' | 'ring';
    strategy?: 'balanced' | 'specialized' | 'adaptive';
    maxAgents?: number;
    enableCognitiveDiversity?: boolean;
    enableNeuralAgents?: boolean;
  }

  // Swarm class
  export class Swarm {
    id: string;
    agents: Map<string, Agent>;
    tasks: Map<string, Task>;
    
    spawn(config: AgentConfig): Promise<Agent>;
    spawnAgent(name: string, type?: AgentType, options?: any): Promise<Agent>;
    orchestrate(taskConfig: TaskConfig): Promise<Task>;
    getStatus(detailed?: boolean): Promise<SwarmStatus>;
    monitor(duration?: number, interval?: number): Promise<MonitorResult>;
    terminate(): Promise<void>;
  }

  // Agent types
  export type AgentType = 'researcher' | 'coder' | 'analyst' | 'optimizer' | 'coordinator';

  // Agent configuration
  export interface AgentConfig {
    type?: AgentType;
    name?: string;
    capabilities?: string[];
    enableNeuralNetwork?: boolean;
  }

  // Agent class
  export class Agent {
    id: string;
    name: string;
    type: string;
    cognitivePattern: CognitivePattern;
    capabilities: string[];
    neuralNetworkId: string | null;
    status: 'idle' | 'busy' | 'offline';
    
    execute(task: any): Promise<TaskResult>;
    getMetrics(): Promise<AgentMetrics>;
    updateStatus(status: string): Promise<void>;
  }

  // Task configuration
  export interface TaskConfig {
    description: string;
    priority?: 'low' | 'medium' | 'high' | 'critical';
    dependencies?: string[];
    maxAgents?: number;
    estimatedDuration?: number;
  }

  // Task class
  export class Task {
    id: string;
    description: string;
    status: 'pending' | 'in_progress' | 'completed' | 'failed';
    assignedAgents: string[];
    result: any;
    
    getStatus(): Promise<TaskStatus>;
    getResults(): Promise<any>;
  }

  // WASM Module Loader
  export class WasmModuleLoader {
    modules: Map<string, WasmModule>;
    loadingPromises: Map<string, Promise<WasmModule>>;
    loadingStrategy: 'eager' | 'on-demand' | 'progressive';
    moduleManifest: ModuleManifest;
    
    initialize(strategy?: string): Promise<boolean>;
    loadModule(moduleName: string): Promise<WasmModule>;
    getModuleStatus(): ModuleStatusMap;
    getTotalMemoryUsage(): number;
  }

  // Enhanced MCP Tools
  export class EnhancedMCPTools {
    constructor();
    
    initialize(): Promise<RuvSwarm>;
    
    // Core MCP tools
    swarm_init(params: SwarmInitParams): Promise<SwarmInitResult>;
    agent_spawn(params: AgentSpawnParams): Promise<AgentSpawnResult>;
    task_orchestrate(params: TaskOrchestrateParams): Promise<TaskOrchestrateResult>;
    swarm_status(params: SwarmStatusParams): Promise<SwarmStatusResult>;
    benchmark_run(params: BenchmarkParams): Promise<BenchmarkResult>;
    features_detect(params: FeaturesParams): Promise<FeaturesResult>;
    memory_usage(params: MemoryParams): Promise<MemoryResult>;
    
    // Neural network tools
    neural_status(params: NeuralStatusParams): Promise<NeuralStatusResult>;
    neural_train(params: NeuralTrainParams): Promise<NeuralTrainResult>;
    neural_patterns(params: NeuralPatternsParams): Promise<NeuralPatternsResult>;
  }

  // Neural Network Manager
  export class NeuralNetworkManager {
    constructor(wasmLoader: WasmModuleLoader);
    
    neuralNetworks: Map<string, NeuralNetwork>;
    templates: NeuralTemplates;
    
    createAgentNeuralNetwork(agentId: string, config?: NeuralConfig): Promise<NeuralNetwork>;
    fineTuneNetwork(agentId: string, trainingData: TrainingData, options?: TrainingOptions): Promise<NeuralMetrics>;
    enableCollaborativeLearning(agentIds: string[], options?: CollaborativeOptions): Promise<CollaborativeSession>;
    getNetworkMetrics(agentId: string): NeuralMetrics | null;
    saveNetworkState(agentId: string, filePath: string): Promise<boolean>;
    loadNetworkState(agentId: string, filePath: string): Promise<boolean>;
  }

  // Types and interfaces
  export interface SwarmFeatures {
    neural_networks: boolean;
    forecasting: boolean;
    cognitive_diversity: boolean;
    simd_support: boolean;
  }

  export interface GlobalMetrics {
    totalSwarms: number;
    totalAgents: number;
    totalTasks: number;
    memoryUsage: number;
    performance: Record<string, any>;
    features: SwarmFeatures;
    wasm_modules: ModuleStatusMap;
    timestamp: string;
  }

  export interface SwarmStatus {
    id: string;
    agents: {
      total: number;
      active: number;
      idle: number;
    };
    tasks: {
      total: number;
      pending: number;
      in_progress: number;
      completed: number;
    };
    wasm_metrics?: WasmMetrics;
  }

  export interface AgentMetrics {
    tasksCompleted: number;
    averageExecutionTime: number;
    successRate: number;
    memoryUsage: number;
  }

  export interface NeuralConfig {
    template?: string;
    layers?: number[];
    activation?: string;
    learningRate?: number;
    optimizer?: string;
  }

  export interface TrainingData {
    samples: Array<{
      input: number[];
      target: number[];
    }>;
  }

  export interface TrainingOptions {
    epochs?: number;
    batchSize?: number;
    learningRate?: number;
    freezeLayers?: number[];
  }

  export interface NeuralMetrics {
    accuracy: number;
    loss: number;
    epochs_trained: number;
    total_samples: number;
    training_history?: Array<{ epoch: number; loss: number }>;
    network_info?: {
      layers: number[];
      parameters: number;
    };
  }

  export type CognitivePattern = 'convergent' | 'divergent' | 'lateral' | 'systems' | 'critical' | 'abstract' | 'adaptive';

  export interface RuntimeFeatures {
    webassembly: boolean;
    simd: boolean;
    workers: boolean;
    shared_array_buffer: boolean;
    bigint: boolean;
  }

  export interface MemoryUsage {
    used: number;
    total: number;
    limit: number;
  }

  export interface ModuleManifest {
    [moduleName: string]: {
      path: string;
      size: number;
      priority: 'high' | 'medium' | 'low';
      dependencies: string[];
    };
  }

  export interface ModuleStatusMap {
    [moduleName: string]: {
      loaded: boolean;
      loading: boolean;
      size: number;
      priority: string;
      dependencies: string[];
      isPlaceholder?: boolean;
    };
  }

  export interface WasmModule {
    instance: WebAssembly.Instance;
    module: WebAssembly.Module;
    exports: any;
    memory: WebAssembly.Memory;
    isPlaceholder?: boolean;
  }

  // MCP Tool Parameters and Results
  export interface SwarmInitParams {
    topology?: 'mesh' | 'star' | 'hierarchical' | 'ring';
    maxAgents?: number;
    strategy?: 'balanced' | 'specialized' | 'adaptive';
    enableCognitiveDiversity?: boolean;
    enableNeuralAgents?: boolean;
    enableForecasting?: boolean;
  }

  export interface SwarmInitResult {
    id: string;
    message: string;
    topology: string;
    strategy: string;
    maxAgents: number;
    features: Record<string, boolean>;
    created: string;
    performance: {
      initialization_time_ms: number;
      memory_usage_mb: number;
    };
  }

  export interface AgentSpawnParams {
    type?: string;
    name?: string;
    capabilities?: string[];
    cognitivePattern?: string;
    neuralConfig?: any;
    swarmId?: string;
  }

  export interface AgentSpawnResult {
    agent: {
      id: string;
      name: string;
      type: string;
      cognitive_pattern: string;
      capabilities: string[];
      neural_network_id: string | null;
      status: string;
    };
    swarm_info: {
      id: string;
      agent_count: number;
      capacity: string;
    };
    message: string;
    performance: {
      spawn_time_ms: number;
      memory_overhead_mb: number;
    };
  }

  export interface TaskOrchestrateParams {
    task: string;
    priority?: string;
    strategy?: string;
    maxAgents?: number;
    swarmId?: string;
    requiredCapabilities?: string[];
    estimatedDuration?: number;
  }

  export interface TaskOrchestrateResult {
    taskId: string;
    status: string;
    description: string;
    priority: string;
    strategy: string;
    assigned_agents: string[];
    swarm_info: any;
    orchestration: any;
    performance: any;
    message: string;
  }

  // Utility exports
  export function consoleLog(message: string): void;
  export function consoleError(message: string): void;
  export function consoleWarn(message: string): void;
  export function formatJsError(error: Error): string;

  // Neural agent exports
  export { NeuralAgent, NeuralAgentFactory, NeuralNetwork } from './neural-agent';
  export const COGNITIVE_PATTERNS: Record<string, CognitivePattern>;
  export const AGENT_COGNITIVE_PROFILES: Record<string, any>;

  // Template exports
  export const NeuralNetworkTemplates: {
    getTemplate(templateName: string): NeuralConfig;
  };
}