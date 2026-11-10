export type DenseActivation = 'Linear' | 'Threshold' | 'ThresholdSymmetric' | 'Sigmoid' |
  'SigmoidSymmetric' | 'Tanh' | 'Gaussian' | 'GaussianSymmetric' | 'Elliot' |
  'ElliotSymmetric' | 'LinearPiece' | 'LinearPieceSymmetric' | 'ReLU' | 'ReLULeaky' |
  'Sin' | 'Cos' | 'SinSymmetric' | 'CosSymmetric' | 'Gelu' | 'Swish' | 'LeakyRelu';
export interface DenseLayer {
  readonly inputs: number;
  readonly outputs: number;
  /** Row-major [outputs, inputs]. */
  readonly weights: readonly number[];
  readonly bias?: readonly number[] | null;
  readonly activation: DenseActivation;
  readonly steepness: number;
}
export interface DenseModel {
  readonly schema: 'ruv-fann.dense-layers/1';
  readonly layers: readonly DenseLayer[];
}
export interface PreparedModel {
  readonly inputSize: number;
  readonly outputSize: number;
  readonly maxRows: number;
  readonly estimatedBytes: number;
  predict(inputs: Float64Array, rows?: number): Float64Array;
  /** Exact output length required. Validation/numerical errors preserve output. Aliasing is supported. */
  predictInto(inputs: Float64Array, outputs: Float64Array, rows?: number): void;
  /** Idempotent; subsequent predictions throw. */
  dispose(): void;
}
export interface TrainOptions {
  inputs: Float64Array;
  targets: Float64Array;
  rows: number;
  /** Full-batch updates, integer from 1 to 1000, subject to total work budget. */
  epochs: number;
  learningRate: number;
  optimizer?: 'sgd' | 'adamw';
}
export interface TrainResult {
  model: DenseModel;
  initialLoss: number;
  finalLoss: number;
  epochs: number;
  parameterCount: number;
}
export interface FannRuntime {
  readonly backend: 'wasm' | 'wasm-simd';
  prepare(model: DenseModel | string, options?: { maxRows?: number }): PreparedModel;
  train(model: DenseModel | string, options: TrainOptions): TrainResult;
}
export function loadRuntime(options?: { simd?: boolean | 'auto' }): Promise<FannRuntime>;
export const LIMITS: Readonly<{ maxRows: number; maxDimension: number; maxLayers: number;
  maxParameters: number; maxModelBytes: number; maxBytes: number; maxEpochs: number; maxTrainingWork: number }>;
