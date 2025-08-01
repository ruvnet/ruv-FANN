/* tslint:disable */
 
export function js_array_to_vec_f32(array: any): Float32Array;
export function vec_f32_to_js_array(vec: Float32Array): Float32Array;
export function get_wasm_memory_usage(): bigint;
export function console_log(message: string): void;
export function console_error(message: string): void;
export function console_warn(message: string): void;
export function format_js_error(error: any): string;
/**
 * SIMD feature detection and runtime capabilities
 */
export function detect_simd_capabilities(): string;
export function run_simd_verification_suite(): string;
export function simd_performance_report(size: number, iterations: number): string;
/**
 * Comprehensive SIMD feature validation
 */
export function validate_simd_implementation(): boolean;
export function init(): void;
export function create_neural_network(layers: Uint32Array, activation: ActivationFunction): WasmNeuralNetwork;
export function create_swarm_orchestrator(topology: string): WasmSwarmOrchestrator;
export function create_forecasting_model(model_type: string): WasmForecastingModel;
export function get_version(): string;
export function get_features(): string;
export enum ActivationFunction {
  Linear = 0,
  Sigmoid = 1,
  SymmetricSigmoid = 2,
  Tanh = 3,
  ReLU = 4,
  LeakyReLU = 5,
  Swish = 6,
  Gaussian = 7,
  Elliot = 8,
  SymmetricElliot = 9,
  Sine = 10,
  Cosine = 11,
  SinSymmetric = 12,
  CosSymmetric = 13,
  ThresholdSymmetric = 14,
  Threshold = 15,
  StepSymmetric = 16,
  Step = 17,
}
/**
 * Agent memory pool specifically optimized for neural network agents
 */
export class AgentMemoryPool {
  free(): void;
  constructor();
  /**
   * Allocate memory for an agent based on complexity
   */
  allocate_for_agent(complexity: string): Uint8Array | undefined;
  /**
   * Return agent memory to the appropriate pool
   */
  deallocate_agent_memory(memory: Uint8Array): void;
  /**
   * Get total memory usage across all pools
   */
  total_memory_usage_mb(): number;
  /**
   * Check if memory usage is within target (< 50MB for 10 agents)
   */
  is_within_memory_target(): boolean;
}
/**
 * Memory pool for efficient memory management
 */
export class MemoryPool {
  free(): void;
  /**
   * Create a new memory pool with specified block size and maximum blocks
   */
  constructor(block_size: number, max_blocks: number);
  /**
   * Allocate a memory block from the pool
   */
  allocate(): Uint8Array | undefined;
  /**
   * Return a memory block to the pool for reuse
   */
  deallocate(block: Uint8Array): void;
  /**
   * Get the number of available blocks in the pool
   */
  available_blocks(): number;
  /**
   * Get total memory usage in bytes
   */
  memory_usage(): number;
  /**
   * Get pool efficiency metrics
   */
  get_metrics(): PoolMetrics;
}
export class OptimizedAgent {
  private constructor();
  free(): void;
}
export class OptimizedAgentSpawner {
  free(): void;
  constructor();
  spawn_agent(agent_type: string, complexity: string): string;
  release_agent(agent_id: string): void;
  get_performance_report(): string;
  get_active_agent_count(): number;
  is_within_memory_target(): boolean;
}
export class PerformanceMonitor {
  free(): void;
  constructor();
  record_load_time(time: number): void;
  record_spawn_time(time: number): void;
  update_memory_usage(bytes: number): void;
  get_average_spawn_time(): number;
  get_memory_usage_mb(): number;
  meets_performance_targets(): boolean;
  get_report(): string;
}
export class PerformanceTimer {
  free(): void;
  constructor(name: string);
  elapsed(): number;
  log(): void;
}
/**
 * Pool metrics for monitoring
 */
export class PoolMetrics {
  private constructor();
  free(): void;
  total_blocks: number;
  free_blocks: number;
  block_size: number;
  reuse_count: number;
  memory_usage_mb: number;
}
export class RuntimeFeatures {
  free(): void;
  constructor();
  get_features_json(): string;
  readonly simd_available: boolean;
  readonly threads_available: boolean;
  readonly memory_limit: bigint;
}
/**
 * Performance benchmarking utilities
 */
export class SimdBenchmark {
  free(): void;
  constructor();
  /**
   * Benchmark SIMD vs scalar dot product
   */
  benchmark_dot_product(size: number, iterations: number): string;
  /**
   * Benchmark SIMD vs scalar activation functions
   */
  benchmark_activation(size: number, iterations: number, activation: string): string;
}
/**
 * SIMD-accelerated matrix operations
 */
export class SimdMatrixOps {
  free(): void;
  constructor();
  /**
   * SIMD-optimized matrix-vector multiplication
   */
  matrix_vector_multiply(matrix: Float32Array, vector: Float32Array, rows: number, cols: number): Float32Array;
  /**
   * SIMD-optimized matrix-matrix multiplication (small matrices)
   */
  matrix_multiply(a: Float32Array, b: Float32Array, a_rows: number, a_cols: number, b_cols: number): Float32Array;
}
/**
 * SIMD-accelerated vector operations
 */
export class SimdVectorOps {
  free(): void;
  constructor();
  /**
   * SIMD-optimized vector dot product
   */
  dot_product(a: Float32Array, b: Float32Array): number;
  /**
   * SIMD-optimized vector addition
   */
  vector_add(a: Float32Array, b: Float32Array): Float32Array;
  /**
   * SIMD-optimized vector scaling
   */
  vector_scale(vec: Float32Array, scalar: number): Float32Array;
  /**
   * SIMD-optimized activation function application
   */
  apply_activation(vec: Float32Array, activation: string): Float32Array;
}
export class WasmAgent {
  free(): void;
  constructor(id: string, agent_type: string);
  set_status(status: string): void;
  add_capability(capability: string): void;
  has_capability(capability: string): boolean;
  readonly id: string;
  readonly agent_type: string;
  readonly status: string;
}
export class WasmForecastingModel {
  free(): void;
  constructor(model_type: string);
  predict(input: Float64Array): Float64Array;
  get_model_type(): string;
}
export class WasmNeuralNetwork {
  free(): void;
  constructor(layers: Uint32Array, activation: ActivationFunction);
  randomize_weights(min: number, max: number): void;
  set_weights(weights: Float64Array): void;
  get_weights(): Float64Array;
  run(inputs: Float64Array): Float64Array;
}
export class WasmSwarmOrchestrator {
  free(): void;
  constructor(topology: string);
  spawn(config: string): string;
  orchestrate(config: string): WasmTaskResult;
  add_agent(agent_id: string): void;
  get_agent_count(): number;
  get_topology(): string;
  get_status(detailed: boolean): string;
}
export class WasmTaskResult {
  private constructor();
  free(): void;
  readonly task_id: string;
  readonly description: string;
  readonly status: string;
  readonly assigned_agents: string[];
  readonly priority: string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_runtimefeatures_free: (a: number, b: number) => void;
  readonly runtimefeatures_new: () => number;
  readonly runtimefeatures_simd_available: (a: number) => number;
  readonly runtimefeatures_threads_available: (a: number) => number;
  readonly runtimefeatures_memory_limit: (a: number) => bigint;
  readonly runtimefeatures_get_features_json: (a: number, b: number) => void;
  readonly js_array_to_vec_f32: (a: number, b: number) => void;
  readonly vec_f32_to_js_array: (a: number, b: number) => number;
  readonly __wbg_performancetimer_free: (a: number, b: number) => void;
  readonly performancetimer_new: (a: number, b: number) => number;
  readonly performancetimer_elapsed: (a: number) => number;
  readonly performancetimer_log: (a: number) => void;
  readonly get_wasm_memory_usage: () => bigint;
  readonly console_log: (a: number, b: number) => void;
  readonly console_error: (a: number, b: number) => void;
  readonly console_warn: (a: number, b: number) => void;
  readonly format_js_error: (a: number, b: number) => void;
  readonly simdvectorops_dot_product: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly simdvectorops_vector_add: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly simdvectorops_vector_scale: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly simdvectorops_apply_activation: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly simdmatrixops_matrix_vector_multiply: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly simdmatrixops_matrix_multiply: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
  readonly __wbg_simdbenchmark_free: (a: number, b: number) => void;
  readonly simdbenchmark_new: () => number;
  readonly simdbenchmark_benchmark_dot_product: (a: number, b: number, c: number, d: number) => void;
  readonly simdbenchmark_benchmark_activation: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly detect_simd_capabilities: (a: number) => void;
  readonly run_simd_verification_suite: (a: number) => void;
  readonly simd_performance_report: (a: number, b: number, c: number) => void;
  readonly validate_simd_implementation: () => number;
  readonly __wbg_memorypool_free: (a: number, b: number) => void;
  readonly memorypool_new: (a: number, b: number) => number;
  readonly memorypool_allocate: (a: number, b: number) => void;
  readonly memorypool_deallocate: (a: number, b: number, c: number) => void;
  readonly memorypool_available_blocks: (a: number) => number;
  readonly memorypool_memory_usage: (a: number) => number;
  readonly memorypool_get_metrics: (a: number) => number;
  readonly __wbg_poolmetrics_free: (a: number, b: number) => void;
  readonly __wbg_get_poolmetrics_total_blocks: (a: number) => number;
  readonly __wbg_set_poolmetrics_total_blocks: (a: number, b: number) => void;
  readonly __wbg_get_poolmetrics_free_blocks: (a: number) => number;
  readonly __wbg_set_poolmetrics_free_blocks: (a: number, b: number) => void;
  readonly __wbg_get_poolmetrics_block_size: (a: number) => number;
  readonly __wbg_set_poolmetrics_block_size: (a: number, b: number) => void;
  readonly __wbg_get_poolmetrics_reuse_count: (a: number) => number;
  readonly __wbg_set_poolmetrics_reuse_count: (a: number, b: number) => void;
  readonly __wbg_get_poolmetrics_memory_usage_mb: (a: number) => number;
  readonly __wbg_set_poolmetrics_memory_usage_mb: (a: number, b: number) => void;
  readonly __wbg_agentmemorypool_free: (a: number, b: number) => void;
  readonly agentmemorypool_new: () => number;
  readonly agentmemorypool_allocate_for_agent: (a: number, b: number, c: number, d: number) => void;
  readonly agentmemorypool_deallocate_agent_memory: (a: number, b: number, c: number) => void;
  readonly agentmemorypool_total_memory_usage_mb: (a: number) => number;
  readonly agentmemorypool_is_within_memory_target: (a: number) => number;
  readonly init: () => void;
  readonly __wbg_wasmneuralnetwork_free: (a: number, b: number) => void;
  readonly wasmneuralnetwork_new: (a: number, b: number, c: number) => number;
  readonly wasmneuralnetwork_randomize_weights: (a: number, b: number, c: number) => void;
  readonly wasmneuralnetwork_set_weights: (a: number, b: number, c: number) => void;
  readonly wasmneuralnetwork_get_weights: (a: number, b: number) => void;
  readonly wasmneuralnetwork_run: (a: number, b: number, c: number, d: number) => void;
  readonly __wbg_wasmswarmorchestrator_free: (a: number, b: number) => void;
  readonly __wbg_wasmagent_free: (a: number, b: number) => void;
  readonly wasmagent_new: (a: number, b: number, c: number, d: number) => number;
  readonly wasmagent_id: (a: number, b: number) => void;
  readonly wasmagent_agent_type: (a: number, b: number) => void;
  readonly wasmagent_status: (a: number, b: number) => void;
  readonly wasmagent_set_status: (a: number, b: number, c: number) => void;
  readonly wasmagent_add_capability: (a: number, b: number, c: number) => void;
  readonly wasmagent_has_capability: (a: number, b: number, c: number) => number;
  readonly __wbg_wasmtaskresult_free: (a: number, b: number) => void;
  readonly wasmtaskresult_task_id: (a: number, b: number) => void;
  readonly wasmtaskresult_description: (a: number, b: number) => void;
  readonly wasmtaskresult_status: (a: number, b: number) => void;
  readonly wasmtaskresult_assigned_agents: (a: number, b: number) => void;
  readonly wasmtaskresult_priority: (a: number, b: number) => void;
  readonly wasmswarmorchestrator_spawn: (a: number, b: number, c: number, d: number) => void;
  readonly wasmswarmorchestrator_orchestrate: (a: number, b: number, c: number) => number;
  readonly wasmswarmorchestrator_add_agent: (a: number, b: number, c: number) => void;
  readonly wasmswarmorchestrator_get_agent_count: (a: number) => number;
  readonly wasmswarmorchestrator_get_topology: (a: number, b: number) => void;
  readonly wasmswarmorchestrator_get_status: (a: number, b: number, c: number) => void;
  readonly __wbg_wasmforecastingmodel_free: (a: number, b: number) => void;
  readonly wasmforecastingmodel_predict: (a: number, b: number, c: number, d: number) => void;
  readonly wasmforecastingmodel_get_model_type: (a: number, b: number) => void;
  readonly create_neural_network: (a: number, b: number, c: number) => number;
  readonly create_swarm_orchestrator: (a: number, b: number) => number;
  readonly create_forecasting_model: (a: number, b: number) => number;
  readonly get_version: (a: number) => void;
  readonly get_features: (a: number) => void;
  readonly __wbg_performancemonitor_free: (a: number, b: number) => void;
  readonly performancemonitor_new: () => number;
  readonly performancemonitor_record_load_time: (a: number, b: number) => void;
  readonly performancemonitor_record_spawn_time: (a: number, b: number) => void;
  readonly performancemonitor_update_memory_usage: (a: number, b: number) => void;
  readonly performancemonitor_get_average_spawn_time: (a: number) => number;
  readonly performancemonitor_get_memory_usage_mb: (a: number) => number;
  readonly performancemonitor_meets_performance_targets: (a: number) => number;
  readonly performancemonitor_get_report: (a: number, b: number) => void;
  readonly __wbg_optimizedagentspawner_free: (a: number, b: number) => void;
  readonly __wbg_optimizedagent_free: (a: number, b: number) => void;
  readonly optimizedagentspawner_new: () => number;
  readonly optimizedagentspawner_spawn_agent: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly optimizedagentspawner_release_agent: (a: number, b: number, c: number, d: number) => void;
  readonly optimizedagentspawner_get_performance_report: (a: number, b: number) => void;
  readonly optimizedagentspawner_get_active_agent_count: (a: number) => number;
  readonly optimizedagentspawner_is_within_memory_target: (a: number) => number;
  readonly simdvectorops_new: () => number;
  readonly simdmatrixops_new: () => number;
  readonly wasmswarmorchestrator_new: (a: number, b: number) => number;
  readonly wasmforecastingmodel_new: (a: number, b: number) => number;
  readonly __wbg_simdmatrixops_free: (a: number, b: number) => void;
  readonly __wbg_simdvectorops_free: (a: number, b: number) => void;
  readonly __wbindgen_export_0: (a: number) => void;
  readonly __wbindgen_export_1: (a: number, b: number) => number;
  readonly __wbindgen_export_2: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_export_3: (a: number, b: number, c: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
