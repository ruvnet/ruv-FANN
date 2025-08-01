/**
 * Core types and interfaces for RuvSwarm
 */

export interface SwarmOptions {
  topology?: SwarmTopology;
  maxAgents?: number;
  connectionDensity?: number;
  syncInterval?: number;
  wasmPath?: string;
}

export type SwarmTopology = 'mesh' | 'hierarchical' | 'distributed' | 'centralized' | 'hybrid';

export interface AgentConfig {
  id: string;
  type: AgentType;
  cognitiveProfile?: CognitiveProfile;
  capabilities?: string[];
  memory?: AgentMemory;
}

export type AgentType = 
  | 'researcher'
  | 'coder'
  | 'analyst'
  | 'architect'
  | 'reviewer'
  | 'debugger'
  | 'tester'
  | 'documenter'
  | 'optimizer'
  | 'custom';

export interface CognitiveProfile {
  analytical: number;
  creative: number;
  systematic: number;
  intuitive: number;
  collaborative: number;
  independent: number;
}

export interface AgentMemory {
  shortTerm: Map<string, any>;
  longTerm: Map<string, any>;
  episodic: EpisodicMemory[];
}

export interface EpisodicMemory {
  timestamp: number;
  context: string;
  data: any;
  importance: number;
}

export interface Task {
  id: string;
  description: string;
  priority: TaskPriority;
  dependencies?: string[];
  assignedAgents?: string[];
  status: TaskStatus;
  result?: any;
  error?: Error;
}

export type TaskPriority = 'low' | 'medium' | 'high' | 'critical';
export type TaskStatus = 'pending' | 'assigned' | 'in_progress' | 'completed' | 'failed';

export interface SwarmState {
  agents: Map<string, Agent>;
  tasks: Map<string, Task>;
  topology: SwarmTopology;
  connections: Connection[];
  metrics: SwarmMetrics;
}

export interface Connection {
  from: string;
  to: string;
  weight: number;
  type: ConnectionType;
}

export type ConnectionType = 'data' | 'control' | 'feedback' | 'coordination';

export interface SwarmMetrics {
  totalTasks: number;
  completedTasks: number;
  failedTasks: number;
  averageCompletionTime: number;
  agentUtilization: Map<string, number>;
  throughput: number;
}

export interface Agent {
  id: string;
  config: AgentConfig;
  state: AgentState;
  connections: string[];
  execute(task: Task): Promise<any>;
  communicate(message: Message): Promise<void>;
  update(state: Partial<AgentState>): void;
}

export interface AgentState {
  status: AgentStatus;
  currentTask?: string;
  load: number;
  performance: AgentPerformance;
}

export type AgentStatus = 'idle' | 'busy' | 'error' | 'offline';

export interface AgentPerformance {
  tasksCompleted: number;
  tasksFailed: number;
  averageExecutionTime: number;
  successRate: number;
}

export interface Message {
  id: string;
  from: string;
  to: string | string[];
  type: MessageType;
  payload: any;
  timestamp: number;
}

export type MessageType = 
  | 'task_assignment'
  | 'task_result'
  | 'status_update'
  | 'coordination'
  | 'knowledge_share'
  | 'error';

export interface SwarmEventEmitter {
  on(event: SwarmEvent, handler: (data: any) => void): void;
  off(event: SwarmEvent, handler: (data: any) => void): void;
  emit(event: SwarmEvent, data: any): void;
}

export type SwarmEvent =
  | 'agent:added'
  | 'agent:removed'
  | 'agent:status_changed'
  | 'task:created'
  | 'task:assigned'
  | 'task:completed'
  | 'task:failed'
  | 'swarm:topology_changed'
  | 'swarm:error'
  | 'swarm:state_restored'
  | 'swarm:initialized'
  | 'swarm:destroyed'
  | 'session:created'
  | 'session:loaded'
  | 'session:saved'
  | 'session:paused'
  | 'session:resumed'
  | 'session:hibernated'
  | 'session:terminated'
  | 'session:restored'
  | 'session:checkpoint_created'
  | 'session:corruption_detected'
  | 'session:error'
  | 'session:integration_enabled';

export interface WasmModule {
  init(): Promise<void>;
  createSwarm(options: SwarmOptions): number;
  addAgent(swarmId: number, config: AgentConfig): number;
  assignTask(swarmId: number, task: Task): void;
  getState(swarmId: number): SwarmState;
  destroy(swarmId: number): void;
}