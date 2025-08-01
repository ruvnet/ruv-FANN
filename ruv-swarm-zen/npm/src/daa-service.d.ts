/**
 * DAA Service Layer TypeScript Definitions
 * Comprehensive type definitions for the DAA service layer
 */

import { EventEmitter } from 'events';

export interface PerformanceMetric {
  operation: string;
  duration: number;
  withinThreshold: boolean;
}

export interface AgentState {
  status: 'active' | 'idle' | 'suspended' | 'error';
  capabilities: string[];
  metrics: AgentMetrics;
  timestamp: number;
  version: number;
  lastDecision?: any;
  lastContext?: DecisionContext;
}

export interface AgentMetrics {
  decisionsMade: number;
  tasksCompleted: number;
  errors: number;
  averageResponseTime: number;
}

export interface Agent {
  id: string;
  wasmAgent: WasmAutonomousAgent;
  capabilities: Set<string>;
  status: string;
  createdAt: number;
  lastActivity: number;
  metrics: AgentMetrics;
  state?: AgentState;
}

export interface DecisionContext {
  environment_state: EnvironmentState;
  available_actions: Action[];
  goals: Goal[];
  history: HistoricalEvent[];
  constraints: ResourceConstraints;
  time_pressure: number;
  uncertainty: number;
}

export interface EnvironmentState {
  environment_type: 'Stable' | 'Dynamic' | 'Hostile' | 'Constrained' | 'Unknown';
  conditions: Map<string, number>;
  stability: number;
  resource_availability: number;
}

export interface Action {
  id: string;
  action_type: 'Compute' | 'Communicate' | 'Allocate' | 'Learn' | 'Monitor' | 'Heal' | 'Coordinate' | 'Plan';
  cost: number;
  expected_reward: number;
  risk: number;
  prerequisites: string[];
  duration?: number;
}

export interface Goal {
  id: string;
  description: string;
  goal_type: 'Performance' | 'Learning' | 'Efficiency' | 'Task' | 'Collaboration' | 'Survival';
  priority: number;
  progress: number;
  target_completion?: number;
  success_criteria: string[];
}

export interface HistoricalEvent {
  timestamp: number;
  event_type: 'Decision' | 'Action' | 'Problem' | 'Success' | 'Failure' | 'Learning';
  description: string;
  outcome: 'Positive' | 'Negative' | 'Neutral' | 'Mixed';
  lessons: string[];
}

export interface ResourceConstraints {
  max_memory_mb: number;
  max_cpu_usage: number;
  max_network_mbps: number;
  max_execution_time: number;
  energy_budget: number;
}

export interface WorkflowStep {
  id: string;
  name?: string;
  description?: string;
  task?: Function | { method: string; args?: any[] };
  action?: Function | { method: string; args?: any[] };
  agentFilter?: (agent: Agent) => boolean;
  timeout?: number;
}

export interface Workflow {
  id: string;
  steps: Map<string, WorkflowStep>;
  dependencies: { [stepId: string]: string[] };
  status: 'pending' | 'running' | 'completed' | 'failed';
  createdAt: number;
  completedSteps: Set<string>;
  activeSteps: Set<string>;
  pendingSteps: Set<string>;
}

export interface WorkflowStatus {
  id: string;
  status: string;
  progress: {
    total: number;
    completed: number;
    active: number;
    pending: number;
  };
  completedSteps: string[];
  activeSteps: string[];
  pendingSteps: string[];
}

export interface WasmModuleStatus {
  [moduleName: string]: {
    loaded: boolean;
    loading: boolean;
    placeholder: boolean;
    size: number;
    priority: string;
    deps: string[];
  };
}

export interface PerformanceMetrics {
  agents: { [agentId: string]: AgentMetrics & { uptime: number; status: string } };
  workflows: { [workflowId: string]: WorkflowStatus };
  system: {
    totalAgents: number;
    activeWorkflows: number;
    averageLatencies: {
      crossBoundaryCall: number;
      agentSpawn: number;
      stateSync: number;
      workflowStep: number;
    };
  };
}

export interface ServiceStatus {
  initialized: boolean;
  agents: {
    count: number;
    ids: string[];
    states: number;
  };
  workflows: {
    count: number;
    active: number;
  };
  wasm: {
    modules: WasmModuleStatus;
    memoryUsage: number;
  };
  performance: PerformanceMetrics;
}

export interface BatchCreateConfig {
  id: string;
  capabilities?: string[];
}

export interface BatchDecisionConfig {
  agentId: string;
  context: DecisionContext;
}

export interface BatchResult<T> {
  success: boolean;
  agent?: Agent;
  decision?: any;
  error?: string;
  config?: T;
}

// WASM Type Definitions
export interface WasmAutonomousAgent {
  new(id: string): WasmAutonomousAgent;
  id: string;
  autonomy_level: number;
  learning_rate: number;
  add_capability(capability: string): boolean;
  remove_capability(capability: string): boolean;
  has_capability(capability: string): boolean;
  get_capabilities(): string[];
  make_decision(context_json: string): Promise<string>;
  adapt(feedback_json: string): Promise<string>;
  is_autonomous(): boolean;
  get_status(): string;
  optimize_resources(): Promise<string>;
}

export interface WasmCoordinator {
  new(): WasmCoordinator;
  add_agent(agent: WasmAutonomousAgent): void;
  remove_agent(agent_id: string): boolean;
  agent_count(): number;
  set_strategy(strategy: string): void;
  get_strategy(): string;
  set_frequency(frequency_ms: number): void;
  get_frequency(): number;
  coordinate(): Promise<string>;
  get_status(): string;
}

export interface WasmResourceManager {
  new(max_memory_mb: number): WasmResourceManager;
  allocate_memory(size_mb: number): boolean;
  deallocate_memory(size_mb: number): boolean;
  get_memory_usage(): number;
  get_allocated_memory(): number;
  get_max_memory(): number;
  set_cpu_usage(usage: number): void;
  get_cpu_usage(): number;
  enable_optimization(): void;
  disable_optimization(): void;
  is_optimization_enabled(): boolean;
  optimize(): Promise<string>;
  get_status(): string;
}

export interface WasmUtils {
  init(): void;
  get_system_capabilities(): string[];
  check_wasm_support(): boolean;
  get_performance_info(): string;
  log(message: string): void;
  create_context(js_object: any): string;
  create_feedback(performance: number, efficiency: number): string;
}

// Event definitions
export interface DAAServiceEvents {
  initialized: () => void;
  agentCreated: (data: { agentId: string; capabilities: string[] }) => void;
  agentDestroyed: (data: { agentId: string }) => void;
  decisionMade: (data: { agentId: string; decision: any; latency: number; withinThreshold: boolean }) => void;
  workflowCreated: (data: { workflowId: string; steps: string[]; dependencies: any }) => void;
  workflowStepCompleted: (data: { workflowId: string; stepId: string; agentIds: string[]; duration: number; result: any }) => void;
  statesSynchronized: (data: { agentIds: string[]; duration: number }) => void;
  resourcesOptimized: (data: { result: any }) => void;
  cleanup: (data: any) => void;
}

export declare class DAAService extends EventEmitter {
  constructor();
  
  // Lifecycle
  initialize(): Promise<void>;
  cleanup(): Promise<void>;
  
  // Agent management
  createAgent(id: string, capabilities?: string[]): Promise<Agent>;
  destroyAgent(id: string): Promise<boolean>;
  batchCreateAgents(configs: BatchCreateConfig[]): Promise<BatchResult<BatchCreateConfig>[]>;
  
  // Decision making
  makeDecision(agentId: string, context: DecisionContext): Promise<any>;
  batchMakeDecisions(decisions: BatchDecisionConfig[]): Promise<BatchResult<BatchDecisionConfig>[]>;
  
  // Workflow management
  createWorkflow(workflowId: string, steps: WorkflowStep[], dependencies?: { [stepId: string]: string[] }): Promise<Workflow>;
  executeWorkflowStep(workflowId: string, stepId: string, agentIds: string[]): Promise<any>;
  
  // State management
  synchronizeStates(agentIds: string[]): Promise<Map<string, AgentState>>;
  
  // Resource management
  optimizeResources(): Promise<any>;
  
  // Monitoring
  getPerformanceMetrics(): PerformanceMetrics;
  getStatus(): ServiceStatus;
  
  // Event handling
  on<K extends keyof DAAServiceEvents>(event: K, listener: DAAServiceEvents[K]): this;
  emit<K extends keyof DAAServiceEvents>(event: K, ...args: Parameters<DAAServiceEvents[K]>): boolean;
}

export declare const daaService: DAAService;

export default DAAService;