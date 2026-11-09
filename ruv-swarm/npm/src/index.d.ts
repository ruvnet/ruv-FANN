/** Supported Node.js facade for the generated WASM APIs. */
import type { WasmNeuralNetwork, WasmForecastingModel, WasmPreparedNetwork, WasmTrainableNetwork } from '../wasm/ruv_swarm_wasm.js';
export type { WasmPreparedNetwork, WasmPreparedBatch, WasmTrainableNetwork } from '../wasm/ruv_swarm_wasm.js';
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
/** Time fields are safe-integer Unix milliseconds, not seconds or ISO strings. */
export interface CalibrationRecord {
  series: string;
  model: string;
  cutoff: number;
  timestamp: number;
  actual: number;
  prediction: number;
}
export interface ConformalConfig {
  as_of: number;
  horizon: number;
  coverages: number[];
  records: CalibrationRecord[];
}
export interface ConformalForecastRecord {
  series: string;
  model: string;
  origin: number;
  timestamp: number;
  prediction: number;
}
export interface ConformalGroup {
  series: string;
  model: string;
  horizon: number;
  samples: number;
  first_outcome: number;
  last_outcome: number;
  /** Radii correspond to ascending sorted configuration coverages. */
  radii: number[];
}
export interface ConformalInterval {
  series: string;
  model: string;
  horizon: number;
  coverage: number;
  point: number;
  lower: number;
  upper: number;
  calibration_samples: number;
  forecast_origin: number;
  calibration_as_of: number;
  timestamp: number;
}
/** Historical absolute-error intervals; no temporal coverage guarantee. */
export declare class ConformalCalibrator {
  private constructor();
  groups(): ConformalGroup[];
  apply(request: { records: ConformalForecastRecord[] }): ConformalInterval[];
  /** Idempotent. Other methods throw after release. */
  free(): void;
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
  /** Import actual ruv_fann::Network<f64> serde JSON, limited to 4 MiB UTF-8. */
  createPreparedNetwork(networkJson: string): WasmPreparedNetwork;
  /** Canonical [outputs, inputs] dense-layer schema; 4 MiB UTF-8 maximum. */
  createDensePreparedNetwork(denseJson: string): WasmPreparedNetwork;
  /** Actual MSE/autograd, SGD/AdamW. maxRows must be an integer in 1..4096. */
  createTrainableNetwork(denseJson: string, maxRows?: number): WasmTrainableNetwork;
  createConformalCalibrator(config: ConformalConfig): ConformalCalibrator;
}
export declare function consoleLog(message: string): void;
export declare function consoleError(message: string): void;
export declare function consoleWarn(message: string): void;
export declare function formatJsError(error: unknown): string;
