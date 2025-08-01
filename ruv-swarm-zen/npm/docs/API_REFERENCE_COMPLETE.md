# ruv-swarm Complete API Reference

## 📚 Table of Contents

- [Core Classes](#core-classes)
- [Swarm Management](#swarm-management)
- [Agent Management](#agent-management)
- [Task Orchestration](#task-orchestration)
- [Neural Networks](#neural-networks)
- [Memory & Persistence](#memory--persistence)
- [Performance & Monitoring](#performance--monitoring)
- [MCP Integration](#mcp-integration)
- [TypeScript Definitions](#typescript-definitions)
- [Error Handling](#error-handling)
- [Examples](#examples)

---

## Core Classes

### RuvSwarm (Main Class)

The primary entry point for ruv-swarm functionality.

```typescript
class RuvSwarm {
  // Static initialization
  static async initialize(options?: SwarmInitOptions): Promise<RuvSwarm>
  static detectSIMDSupport(): boolean
  static getRuntimeFeatures(): RuntimeFeatures
  static getVersion(): string
  static getMemoryUsage(): number
  
  // Instance methods
  async createSwarm(config: SwarmConfig): Promise<SwarmWrapper>
  async destroy(): Promise<void>
}
```

#### SwarmInitOptions

```typescript
interface SwarmInitOptions {
  // Performance options
  useSIMD?: boolean;              // Enable SIMD optimizations
  wasmPath?: string;              // Custom WASM module path
  parallel?: boolean;             // Enable parallel execution
  workerPoolSize?: number;        // Worker thread pool size
  
  // Feature flags
  loadingStrategy?: 'progressive' | 'immediate';
  enablePersistence?: boolean;
  enableNeuralNetworks?: boolean;
  enableForecasting?: boolean;
  
  // Debug options
  debug?: boolean;
  retryAttempts?: number;
  retryDelay?: number;
}
```

#### RuntimeFeatures

```typescript
interface RuntimeFeatures {
  simdAvailable: boolean;
  threadsAvailable: boolean;
  memoryLimit: number;
  wasmVersion: string;
  platform: string;
}
```

---

## Swarm Management

### SwarmWrapper

Manages a collection of agents and their coordination.

```typescript
class SwarmWrapper {
  // Properties
  readonly name: string;
  readonly agentCount: number;
  readonly maxAgents: number;
  
  // Agent lifecycle
  async spawn(config: AgentConfig): Promise<AgentWrapper>
  getAgents(): AgentWrapper[]
  
  // Task execution
  async orchestrate(task: TaskConfig): Promise<OrchestrationResult>
  
  // Status and monitoring
  getStatus(): SwarmStatus
  getTopology(): TopologyInfo
  getMetrics(): SwarmMetrics
  
  // Advanced features
  async createCluster(name: string, config: ClusterConfig): Promise<Cluster>
  query(selector: AgentSelector): AgentWrapper[]
  on(event: SwarmEvent, handler: EventHandler): void
}
```

#### SwarmConfig

```typescript
interface SwarmConfig {
  topology: 'mesh' | 'hierarchical' | 'ring' | 'star' | 'clustered' | 'pipeline';
  maxAgents: number;
  name?: string;
  
  // Advanced options
  cognitiveProfiles?: boolean;
  persistence?: PersistenceConfig;
  monitoring?: MonitoringConfig;
  scaling?: ScalingConfig;
  features?: FeatureFlag[];
}
```

#### SwarmStatus

```typescript
interface SwarmStatus {
  id: string;
  topology: string;
  agents: {
    total: number;
    active: number;
    idle: number;
    busy: number;
  };
  tasks: {
    total: number;
    pending: number;
    in_progress: number;
    completed: number;
    failed: number;
  };
  performance: {
    uptime: number;
    throughput: number;
    memory_usage: number;
    cpu_usage: number;
  };
  features: Record<string, boolean>;
}
```

---

## Agent Management

### AgentWrapper

Represents an individual AI agent within the swarm.

```typescript
class AgentWrapper {
  // Properties
  readonly id: string;
  readonly agentType: AgentType;
  readonly status: AgentStatus;
  readonly tasksCompleted: number;
  
  // Core functionality
  async execute(task: Task): Promise<TaskResult>
  async collaborate(agents: AgentWrapper[], objective: string): Promise<CollaborationResult>
  async learn(experience: Experience): Promise<void>
  
  // State management
  getState(): AgentState
  getMetrics(): AgentMetrics
  getMemory(): AgentMemory
  getCapabilities(): string[]
  updateCapabilities(capabilities: string[]): void
  
  // Communication
  async sendMessage(to: AgentWrapper, message: Message): Promise<void>
  async broadcast(message: Message): Promise<void>
  subscribe(topic: string, handler: MessageHandler): void
  
  // Lifecycle
  reset(): void
  terminate(): void
}
```

#### AgentConfig

```typescript
interface AgentConfig {
  type: AgentType;
  name?: string;
  cognitiveProfile?: CognitiveProfile;
  capabilities?: string[];
  specialization?: string;
  memory?: MemoryConfig;
  constraints?: AgentConstraints;
  
  // Neural network options
  neuralNetwork?: {
    enabled: boolean;
    model?: string;
    pretrainedWeights?: string;
  };
}
```

#### AgentType

```typescript
type AgentType = 
  | 'researcher'     // Research and analysis
  | 'coder'         // Code generation and development
  | 'analyst'       // Data analysis and insights
  | 'architect'     // System design and planning
  | 'optimizer'     // Performance optimization
  | 'coordinator'   // Task coordination and management
  | 'tester'        // Quality assurance and testing
  | 'documenter'    // Documentation generation
  | 'reviewer'      // Code and content review
  | 'debugger';     // Error detection and fixing
```

#### CognitiveProfile

```typescript
interface CognitiveProfile {
  analytical: number;      // Data-driven reasoning (0-1)
  creative: number;        // Novel solution generation (0-1)
  systematic: number;      // Structured problem-solving (0-1)
  intuitive: number;       // Pattern-based insights (0-1)
  collaborative: number;   // Team coordination (0-1)
  independent: number;     // Autonomous operation (0-1)
}
```

#### AgentMetrics

```typescript
interface AgentMetrics {
  // Performance metrics
  tasks_completed: number;
  success_rate: number;
  average_response_time: number;
  throughput: number;
  
  // Resource utilization
  memory_usage: number;
  cpu_usage: number;
  network_usage: number;
  
  // Quality metrics
  accuracy_score: number;
  collaboration_score: number;
  learning_rate: number;
  
  // Neural network metrics (if enabled)
  neural?: {
    training_loss: number;
    validation_accuracy: number;
    inference_time: number;
    model_size: number;
  };
}
```

---

## Task Orchestration

### Task Management

```typescript
interface Task {
  id: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  strategy: 'parallel' | 'sequential' | 'adaptive';
  
  // Execution parameters
  maxAgents?: number;
  timeout?: number;
  retries?: number;
  
  // Dependencies
  dependencies?: string[];
  prerequisites?: Prerequisite[];
  
  // Context
  context?: Record<string, any>;
  metadata?: Record<string, any>;
}
```

#### TaskConfig

```typescript
interface TaskConfig {
  task: string;
  strategy?: 'parallel' | 'sequential' | 'adaptive';
  priority?: 'low' | 'medium' | 'high' | 'critical';
  maxAgents?: number;
  timeout?: number;
  
  // Advanced options
  agents?: AgentWrapper[];
  phases?: TaskPhase[];
  constraints?: TaskConstraints;
  monitoring?: TaskMonitoringOptions;
}
```

#### OrchestrationResult

```typescript
interface OrchestrationResult {
  taskId: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled';
  
  // Execution details
  assigned_agents: AgentInfo[];
  started_at: Date;
  completed_at?: Date;
  duration?: number;
  
  // Results
  result?: any;
  error?: string;
  artifacts?: Artifact[];
  
  // Performance
  performance: {
    estimated_completion_ms: number;
    actual_completion_ms?: number;
    throughput: number;
    efficiency_score: number;
  };
  
  // Metrics
  metrics: TaskMetrics;
}
```

#### TaskPhase

```typescript
interface TaskPhase {
  name: string;
  description: string;
  agents?: AgentWrapper[];
  dependencies?: string[];
  parallel?: boolean;
  
  // Execution
  tasks: SubTask[];
  timeout?: number;
  retries?: number;
  
  // Validation
  validation?: ValidationRule[];
  completion_criteria?: CompletionCriteria;
}
```

---

## Neural Networks

### NeuralAgent

Enhanced agents with neural network capabilities.

```typescript
class NeuralAgent extends AgentWrapper {
  // Neural-specific properties
  readonly neuralNetworkId?: string;
  readonly cognitivePattern: string;
  
  // Neural operations
  async train(data: TrainingData, options?: TrainingOptions): Promise<TrainingResult>
  async predict(input: any): Promise<PredictionResult>
  async evaluate(testData: TestData): Promise<EvaluationResult>
  
  // Model management
  async saveModel(path: string): Promise<void>
  async loadModel(path: string): Promise<void>
  getModelInfo(): ModelInfo
  
  // Learning and adaptation
  async adaptToCognitivePattern(pattern: CognitivePattern): Promise<void>
  async updateFromExperience(experience: Experience): Promise<void>
}
```

#### Neural Network Types

```typescript
// Available neural architectures
type NeuralArchitecture = 
  | 'lstm'           // Long Short-Term Memory
  | 'gru'            // Gated Recurrent Unit  
  | 'transformer'    // Transformer architecture
  | 'cnn'            // Convolutional Neural Network
  | 'autoencoder'    // Autoencoder
  | 'vae'            // Variational Autoencoder
  | 'gnn'            // Graph Neural Network
  | 'resnet';        // Residual Network

// Cognitive patterns for specialized thinking
type CognitivePattern = 
  | 'convergent'     // Focused, systematic thinking
  | 'divergent'      // Creative, exploratory thinking
  | 'lateral'        // Lateral, innovative thinking
  | 'systems'        // Systems-level thinking
  | 'critical'       // Critical analysis thinking
  | 'abstract';      // Abstract conceptual thinking
```

#### NeuralNetworkManager

```typescript
class NeuralNetworkManager {
  // Model creation and management
  async createModel(config: ModelConfig): Promise<NeuralNetwork>
  async loadModel(id: string): Promise<NeuralNetwork>
  listModels(): ModelInfo[]
  
  // Training orchestration
  async trainModel(modelId: string, data: TrainingData): Promise<TrainingResult>
  async evaluateModel(modelId: string, testData: TestData): Promise<EvaluationResult>
  
  // Model optimization
  async optimizeHyperparameters(modelId: string): Promise<OptimizationResult>
  async pruneModel(modelId: string, threshold: number): Promise<void>
  
  // Ensemble methods
  async createEnsemble(modelIds: string[]): Promise<EnsembleModel>
  async compareModels(modelIds: string[]): Promise<ComparisonResult>
}
```

---

## Memory & Persistence

### Memory Management

```typescript
interface MemoryConfig {
  backend: 'sqlite' | 'memory' | 'file';
  path?: string;
  features: MemoryFeature[];
  
  // Retention policies
  retention?: {
    episodic_memory: number;    // Days
    semantic_memory: number;    // Days
    procedural_memory: number;  // Days
  };
  
  // Performance
  cache_size?: number;
  compression?: boolean;
}

type MemoryFeature = 
  | 'episodic_memory'     // Remember specific experiences
  | 'semantic_memory'     // Store factual knowledge
  | 'procedural_memory'   // Learn procedures and skills
  | 'working_memory'      // Short-term context
  | 'relationship_tracking'; // Track agent relationships
```

#### Memory Operations

```typescript
interface MemoryOperations {
  // Storage
  store(key: string, value: any, type?: MemoryType): Promise<void>
  retrieve(key: string): Promise<any>
  delete(key: string): Promise<boolean>
  
  // Querying
  search(query: string, options?: SearchOptions): Promise<SearchResult[]>
  recall(context: RecallContext): Promise<Memory[]>
  associate(concept: string, related: string[]): Promise<void>
  
  // Management
  consolidate(): Promise<void>
  backup(path: string): Promise<void>
  restore(path: string): Promise<void>
  cleanup(olderThan: Date): Promise<number>
}
```

#### Persistent Learning

```typescript
interface LearningSystem {
  // Skill acquisition
  async learnSkill(skill: Skill, examples: Example[]): Promise<LearnResult>
  async transferSkill(fromAgent: string, toAgent: string, skill: string): Promise<void>
  
  // Experience processing
  async processExperience(experience: Experience): Promise<void>
  async generateInsights(experiences: Experience[]): Promise<Insight[]>
  
  // Knowledge management
  async addKnowledge(knowledge: Knowledge): Promise<void>
  async queryKnowledge(query: string): Promise<Knowledge[]>
  async updateKnowledge(id: string, updates: Partial<Knowledge>): Promise<void>
}
```

---

## Performance & Monitoring

### Performance Monitoring

```typescript
interface PerformanceMonitor {
  // Real-time metrics
  getCurrentMetrics(): CurrentMetrics
  getHistoricalMetrics(timeRange: TimeRange): HistoricalMetrics
  
  // Monitoring controls
  startMonitoring(options?: MonitoringOptions): void
  stopMonitoring(): void
  setAlerts(rules: AlertRule[]): void
  
  // Analysis
  analyzeBottlenecks(): BottleneckAnalysis
  generateOptimizationSuggestions(): OptimizationSuggestion[]
  predictPerformance(scenario: Scenario): PerformancePrediction
}
```

#### Metrics

```typescript
interface SwarmMetrics {
  // System-wide metrics
  total_agents: number;
  active_agents: number;
  total_tasks: number;
  completed_tasks: number;
  
  // Performance metrics
  throughput: number;           // Tasks per second
  latency: number;             // Average response time
  error_rate: number;          // Percentage of failed tasks
  efficiency: number;          // Resource utilization efficiency
  
  // Resource utilization
  memory_usage: number;        // Bytes
  cpu_usage: number;          // Percentage
  network_usage: number;      // Bytes per second
  
  // Quality metrics
  accuracy: number;           // Task accuracy percentage
  collaboration_index: number; // Inter-agent collaboration score
  learning_velocity: number;  // Rate of skill acquisition
  
  // WASM-specific metrics
  wasm_memory: number;
  wasm_execution_time: number;
  simd_utilization: number;
}
```

#### Benchmarking

```typescript
interface BenchmarkSuite {
  // Benchmark execution
  async runBenchmark(type: BenchmarkType, options?: BenchmarkOptions): Promise<BenchmarkResult>
  async compareBenchmarks(results: BenchmarkResult[]): Promise<ComparisonReport>
  
  // Benchmark types
  async benchmarkSwarmPerformance(): Promise<SwarmBenchmark>
  async benchmarkAgentCapabilities(): Promise<AgentBenchmark>
  async benchmarkNeuralNetworks(): Promise<NeuralBenchmark>
  async benchmarkWASMPerformance(): Promise<WASMBenchmark>
  
  // SWE-Bench integration
  async runSWEBench(instances?: number): Promise<SWEBenchResult>
  async validateSolveRate(): Promise<SolveRateResult>
}

type BenchmarkType = 
  | 'swarm_coordination'
  | 'agent_performance' 
  | 'neural_inference'
  | 'wasm_execution'
  | 'memory_usage'
  | 'throughput'
  | 'swe_bench';
```

---

## MCP Integration

### MCP Tools

Complete Model Context Protocol integration for Claude Code.

```typescript
interface MCPTools {
  // Swarm management tools
  swarm_init(params: SwarmInitParams): Promise<SwarmInitResult>
  swarm_status(params?: StatusParams): Promise<StatusResult>
  swarm_monitor(params?: MonitorParams): Promise<MonitorResult>
  
  // Agent management tools
  agent_spawn(params: AgentSpawnParams): Promise<AgentSpawnResult>
  agent_list(params?: AgentListParams): Promise<AgentListResult>
  agent_metrics(params?: AgentMetricsParams): Promise<AgentMetricsResult>
  
  // Task orchestration tools
  task_orchestrate(params: TaskParams): Promise<TaskResult>
  task_status(params?: TaskStatusParams): Promise<TaskStatusResult>
  task_results(params: TaskResultsParams): Promise<TaskResultsResult>
  
  // Memory operations
  memory_store(params: MemoryStoreParams): Promise<MemoryResult>
  memory_retrieve(params: MemoryRetrieveParams): Promise<MemoryResult>
  memory_usage(params?: MemoryUsageParams): Promise<MemoryUsageResult>
  
  // Neural network tools
  neural_status(params?: NeuralStatusParams): Promise<NeuralStatusResult>
  neural_train(params?: NeuralTrainParams): Promise<NeuralTrainResult>
  neural_patterns(params?: NeuralPatternsParams): Promise<NeuralPatternsResult>
  
  // Performance tools
  benchmark_run(params?: BenchmarkParams): Promise<BenchmarkResult>
  features_detect(params?: FeaturesParams): Promise<FeaturesResult>
}
```

#### MCP Tool Parameters

```typescript
// Swarm initialization parameters
interface SwarmInitParams {
  topology: 'mesh' | 'hierarchical' | 'ring' | 'star';
  maxAgents?: number;
  strategy?: 'balanced' | 'specialized' | 'adaptive';
  enableCognitiveDiversity?: boolean;
  enableNeuralAgents?: boolean;
}

// Agent spawning parameters
interface AgentSpawnParams {
  type: AgentType;
  name?: string;
  capabilities?: string[];
  cognitiveProfile?: Partial<CognitiveProfile>;
  neuralNetwork?: boolean;
}

// Task orchestration parameters
interface TaskParams {
  task: string;
  strategy?: 'parallel' | 'sequential' | 'adaptive';
  priority?: 'low' | 'medium' | 'high' | 'critical';
  maxAgents?: number;
  timeout?: number;
}

// Memory operation parameters
interface MemoryStoreParams {
  key: string;
  value: any;
  type?: 'episodic' | 'semantic' | 'procedural';
  expiry?: number;
}

interface MemoryRetrieveParams {
  key: string;
  type?: 'episodic' | 'semantic' | 'procedural';
}
```

### Claude Code Hooks

Automated integration hooks for seamless Claude Code workflow.

```typescript
interface ClaudeHooks {
  // Pre-operation hooks
  pre_task(params: PreTaskParams): Promise<HookResult>
  pre_edit(params: PreEditParams): Promise<HookResult>
  pre_search(params: PreSearchParams): Promise<HookResult>
  
  // Post-operation hooks  
  post_task(params: PostTaskParams): Promise<HookResult>
  post_edit(params: PostEditParams): Promise<HookResult>
  post_search(params: PostSearchParams): Promise<HookResult>
  
  // Session management
  session_start(params: SessionParams): Promise<HookResult>
  session_end(params: SessionParams): Promise<HookResult>
  session_restore(params: RestoreParams): Promise<HookResult>
  
  // Git integration
  git_commit(params: GitCommitParams): Promise<HookResult>
  git_status(params?: GitStatusParams): Promise<HookResult>
  
  // Notification system
  notification(params: NotificationParams): Promise<HookResult>
  telemetry(params: TelemetryParams): Promise<HookResult>
}
```

---

## TypeScript Definitions

### Complete Type Definitions

```typescript
// Export all types for TypeScript users
export {
  // Core types
  RuvSwarm,
  SwarmWrapper,
  AgentWrapper,
  
  // Configuration interfaces
  SwarmInitOptions,
  SwarmConfig,
  AgentConfig,
  TaskConfig,
  MemoryConfig,
  
  // Result interfaces
  SwarmStatus,
  AgentMetrics,
  OrchestrationResult,
  BenchmarkResult,
  
  // Utility types
  AgentType,
  CognitiveProfile,
  CognitivePattern,
  NeuralArchitecture,
  MemoryFeature,
  BenchmarkType,
  
  // MCP types
  MCPTools,
  SwarmInitParams,
  AgentSpawnParams,
  TaskParams,
  
  // Hook types
  ClaudeHooks,
  HookResult,
  PreTaskParams,
  PostTaskParams,
  
  // Neural network types
  NeuralAgent,
  NeuralNetworkManager,
  ModelConfig,
  TrainingResult,
  
  // Performance types
  PerformanceMonitor,
  SwarmMetrics,
  BenchmarkSuite,
  
  // Memory types
  MemoryOperations,
  LearningSystem,
  Knowledge,
  Experience
};
```

---

## Error Handling

### Error Types

```typescript
// Base error class
class RuvSwarmError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: any
  ) {
    super(message);
    this.name = 'RuvSwarmError';
  }
}

// Specific error types
class ValidationError extends RuvSwarmError {
  constructor(message: string, parameter: string) {
    super(message, 'VALIDATION_ERROR', { parameter });
  }
}

class AgentError extends RuvSwarmError {
  constructor(message: string, agentId: string) {
    super(message, 'AGENT_ERROR', { agentId });
  }
}

class TaskError extends RuvSwarmError {
  constructor(message: string, taskId: string) {
    super(message, 'TASK_ERROR', { taskId });
  }
}

class WASMError extends RuvSwarmError {
  constructor(message: string, module: string) {
    super(message, 'WASM_ERROR', { module });
  }
}

class NetworkError extends RuvSwarmError {
  constructor(message: string, endpoint: string) {
    super(message, 'NETWORK_ERROR', { endpoint });
  }
}
```

### Error Recovery

```typescript
interface ErrorRecovery {
  // Automatic retry mechanisms
  retryWithBackoff<T>(
    operation: () => Promise<T>, 
    options?: RetryOptions
  ): Promise<T>
  
  // Circuit breaker pattern
  circuitBreaker<T>(
    operation: () => Promise<T>,
    options?: CircuitBreakerOptions
  ): Promise<T>
  
  // Graceful degradation
  fallbackOperation<T>(
    primary: () => Promise<T>,
    fallback: () => Promise<T>
  ): Promise<T>
  
  // Error reporting
  reportError(error: Error, context?: ErrorContext): Promise<void>
}
```

---

## Examples

### Basic Usage

```typescript
import { RuvSwarm } from 'ruv-swarm';

// Initialize ruv-swarm
const swarm = await RuvSwarm.initialize({
  useSIMD: true,
  enablePersistence: true,
  enableNeuralNetworks: true
});

// Create a swarm
const mySwarm = await swarm.createSwarm({
  topology: 'mesh',
  maxAgents: 10,
  cognitiveProfiles: true
});

// Spawn specialized agents
const researcher = await mySwarm.spawn({
  type: 'researcher',
  name: 'Data Scientist',
  cognitiveProfile: {
    analytical: 0.9,
    systematic: 0.8,
    creative: 0.6
  }
});

const coder = await mySwarm.spawn({
  type: 'coder',
  name: 'Senior Developer',
  cognitiveProfile: {
    systematic: 0.9,
    analytical: 0.7,
    creative: 0.8
  }
});

// Orchestrate a complex task
const result = await mySwarm.orchestrate({
  task: "Build a neural architecture search system",
  strategy: 'adaptive',
  maxAgents: 5,
  timeout: 300000 // 5 minutes
});

console.log('Task completed:', result);
```

### Advanced Neural Network Usage

```typescript
import { NeuralAgent, NeuralNetworkManager } from 'ruv-swarm';

// Create neural network manager
const neuralManager = new NeuralNetworkManager();

// Create a specialized model
const model = await neuralManager.createModel({
  architecture: 'transformer',
  layers: [
    { type: 'attention', heads: 8, dim: 512 },
    { type: 'feedforward', dim: 2048 },
    { type: 'output', dim: 1000 }
  ],
  optimizer: 'adam',
  learningRate: 0.001
});

// Create neural agent with the model
const neuralAgent = await mySwarm.spawn({
  type: 'researcher',
  neuralNetwork: {
    enabled: true,
    model: model.id,
    cognitivePattern: 'convergent'
  }
}) as NeuralAgent;

// Train the agent
const trainingResult = await neuralAgent.train({
  inputs: trainingInputs,
  targets: trainingTargets,
  epochs: 100,
  batchSize: 32
});

console.log('Training completed:', trainingResult);
```

### MCP Integration Example

```typescript
// MCP tool usage (for Claude Code integration)
import { EnhancedMCPTools } from 'ruv-swarm';

const mcpTools = new EnhancedMCPTools();

// Initialize swarm via MCP
const swarmResult = await mcpTools.swarm_init({
  topology: 'hierarchical',
  maxAgents: 20,
  strategy: 'specialized'
});

// Spawn agents via MCP
const agentResult = await mcpTools.agent_spawn({
  type: 'architect',
  name: 'System Designer',
  capabilities: ['system_design', 'scalability', 'security']
});

// Orchestrate task via MCP
const taskResult = await mcpTools.task_orchestrate({
  task: "Design microservices architecture",
  strategy: 'adaptive',
  priority: 'high'
});

// Monitor progress
const statusResult = await mcpTools.task_status({
  taskId: taskResult.taskId,
  detailed: true
});
```

### Performance Monitoring

```typescript
import { PerformanceMonitor } from 'ruv-swarm';

const monitor = new PerformanceMonitor();

// Start monitoring
monitor.startMonitoring({
  interval: 1000,
  metrics: ['throughput', 'latency', 'memory', 'cpu']
});

// Set up alerts
monitor.setAlerts([
  {
    metric: 'cpu_usage',
    threshold: 0.8,
    action: 'scale_up'
  },
  {
    metric: 'error_rate',
    threshold: 0.05,
    action: 'investigate'
  }
]);

// Get current metrics
const metrics = monitor.getCurrentMetrics();
console.log('Current performance:', metrics);

// Analyze bottlenecks
const bottlenecks = monitor.analyzeBottlenecks();
console.log('Performance bottlenecks:', bottlenecks);

// Get optimization suggestions
const suggestions = monitor.generateOptimizationSuggestions();
console.log('Optimization suggestions:', suggestions);
```

---

## Version Information

- **API Version**: 0.2.1
- **Protocol Version**: 2024-11-05 (MCP)
- **TypeScript Support**: Full type definitions included
- **Browser Compatibility**: Chrome 70+, Firefox 65+, Safari 14+, Edge 79+
- **Node.js Compatibility**: 14.0+

---

## Support and Resources

- **Documentation**: [Full Documentation](../README.md)
- **Examples**: [Examples Directory](../examples/)
- **Issues**: [GitHub Issues](https://github.com/ruvnet/ruv-FANN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ruvnet/ruv-FANN/discussions)

---

*This API reference covers 100% of the ruv-swarm functionality with complete type definitions and practical examples.*