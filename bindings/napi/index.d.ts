/** Native Node-API classes. Inputs are intrinsically checked again inside the addon. */
export class NativePreparedNetwork {
  private constructor();
  static from_network_json(json: string): NativePreparedNetwork;
  static from_dense_layers_json(json: string): NativePreparedNetwork;
  readonly input_size: number;
  readonly output_size: number;
  readonly estimated_bytes: number;
  run(input: Float64Array): Float64Array;
  run_into(input: Float64Array, output: Float64Array): void;
  create_batch(maxRows: number): NativePreparedBatch;
  /** Idempotent; existing batches retain their own shared immutable snapshot. */
  free(): void;
}
export class NativePreparedBatch {
  private constructor();
  readonly input_size: number;
  readonly output_size: number;
  readonly estimated_bytes: number;
  readonly max_rows: number;
  run_into(input: Float64Array, output: Float64Array, rows: number): void;
  free(): void;
}
/** Actual fixed dense reverse-mode MSE with revision-checked SGD and AdamW. */
export class NativeTrainableNetwork {
  private constructor();
  static from_dense_layers_json(json: string, maxRows: number): NativeTrainableNetwork;
  readonly input_size: number;
  readonly output_size: number;
  readonly parameter_count: number;
  readonly max_rows: number;
  readonly estimated_bytes: number;
  mse_backward(input: Float64Array, targets: Float64Array, rows: number): number;
  parameter_gradients(): Float64Array;
  input_gradients(): Float64Array;
  predictions(): Float64Array;
  sgd_step(learningRate: number): void;
  configure_adamw(learningRate: number, beta1: number, beta2: number, epsilon: number, weightDecay: number): void;
  adamw_step(): void;
  prepare(): NativePreparedNetwork;
  to_dense_layers_json(): string;
  free(): void;
}
