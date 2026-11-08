/** Supported Node.js facade for the generated WASM APIs. */
import type { WasmNeuralNetwork, WasmForecastingModel } from '../wasm/ruv_swarm_wasm.js';
export { DAAService, daaService } from './daa-service.js';

export interface InitOptions {
  wasmPath?: string;
  useSIMD?: boolean;
  debug?: boolean;
}
export interface RuntimeFeatures {
  simdAvailable: boolean;
  threadsAvailable: boolean;
  memoryLimit: number;
}
export interface SwarmConfig {
  name?: string;
  topology?: 'mesh' | 'ring' | 'star' | 'hierarchical';
  mode?: 'centralized' | 'distributed' | 'hierarchical' | 'mesh' | 'hybrid';
  strategy?: string;
  maxAgents?: number;
}
export interface AgentConfig {
  id?: string;
  name?: string;
  type?: string;
  capabilities?: string[];
}
export interface TaskConfig {
  id?: string;
  description: string;
  priority?: 'low' | 'normal' | 'medium' | 'high' | 'critical';
  maxAgents?: number;
  requiredCapabilities?: string[];
  /** Only empty dependencies are supported by the metadata orchestrator. */
  dependencies?: string[];
}
export interface OrchestrationResult {
  taskId: string;
  description: string;
  priority: string;
  status: 'assigned' | 'unassigned';
  assignedAgents: string[];
}
export interface SwarmStatus {
  name: string;
  agentCount: number;
  maxAgents: number;
  strategy: string;
  mode: string;
  topology: string;
  agents: { total: number; idle: number; busy: number };
  agent_details: Array<{ id: string; name: string; type: string; status: string; capabilities: string[] }>;
}
export declare class JsAgent {
  private constructor();
  readonly id: string;
  readonly agentType: string;
  readonly status: string | undefined;
  readonly tasksCompleted: 0;
  /** Local WASM agents track assignments; external execution is unsupported. */
  execute(task?: unknown): Promise<never>;
  getMetrics(): never;
  getCapabilities(): string[];
  reset(): void;
}
export declare class WasmSwarm {
  private constructor();
  readonly name: string;
  readonly maxAgents: number;
  readonly agentCount: number;
  spawn(config: AgentConfig): Promise<JsAgent>;
  orchestrate(task: TaskConfig): Promise<OrchestrationResult>;
  getAgents(): string[];
  getStatus(): SwarmStatus;
  free(): void;
}
export declare class RuvSwarm {
  private constructor();
  static initialize(options?: InitOptions): Promise<RuvSwarm>;
  static detectSIMDSupport(): boolean;
  static getVersion(): string;
  static getRuntimeFeatures(): RuntimeFeatures;
  static getMemoryUsage(): number;
  detectSIMDSupport(): boolean;
  createSwarm(config?: SwarmConfig): Promise<WasmSwarm>;
  createNeuralNetwork(layers: number[] | Uint32Array, activation?: string): WasmNeuralNetwork;
  createForecastingModel(modelType: 'linear' | 'mean' | 'naive'): WasmForecastingModel;
}
export declare function consoleLog(message: string): void;
export declare function consoleError(message: string): void;
export declare function consoleWarn(message: string): void;
export declare function formatJsError(error: unknown): string;
