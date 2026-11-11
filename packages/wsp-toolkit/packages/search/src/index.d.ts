import type { DenseModel, EvaluationDataset, EvaluationPolicy, EvaluationOptions, PatchLimits, WeightPatch, WeightEvaluation, WeightMetrics, WeightBatch, WeightLocation } from 'ruv-fann-wsp';

export interface DiscreteWeightLocation extends WeightLocation { values: number[] }
export type Immutable<T> = { readonly [K in keyof T]: T[K] extends object ? Immutable<T[K]> : T[K] };
export interface ObjectiveInput {
  readonly predictions: readonly number[];
  readonly metrics: Readonly<WeightMetrics>;
  /** A separate recursively frozen snapshot. Callback code is trusted and is not a sandbox. */
  readonly model: Immutable<DenseModel>;
}
export interface SearchOptions extends EvaluationOptions {
  locations: DiscreteWeightLocation[];
  /** Candidate inference screens, excluding the baseline and final two-pass authoritative evaluation; default 64, maximum 256. */
  maxEvaluations?: number;
  /** Default 4; maximum 16. */
  maxPasses?: number;
  /** May tighten the 10,000,000 aggregate parameter-row work ceiling. */
  maxWork?: number;
  /** Synchronous trusted local callback; lower finite scores win. Default is MSE. No provider or generated code is executed. */
  objective?: (input: ObjectiveInput) => number;
}
export interface SearchResult {
  schema: 'ruv-fann.weight-search/1';
  method: 'discrete-coordinate-search';
  baseHash: string;
  objective: 'custom' | 'mse';
  baselineScore: number;
  bestScore: number;
  /** Null if no strictly better nonempty patch was found. */
  patch: WeightPatch | null;
  /** Real-WASM evaluation of the winner against the original baseline and supplied policy. May reject the custom objective's preferred candidate. */
  evaluation: WeightEvaluation | null;
  evaluations: number;
  inferencePasses: number;
  passes: number;
  budgetExhausted: boolean;
  reservedWork: number;
  backend: 'wasm' | 'wasm-simd';
}
export function searchWeightPatch(model: DenseModel | string, dataset: EvaluationDataset, policy: EvaluationPolicy, options: SearchOptions): Promise<SearchResult>;
export const SEARCH_LIMITS: Readonly<{ maxLocations: 16; maxValues: 32; maxEvaluations: 256; maxPasses: 16; maxWork: number }>;
export interface BooleanTable {
  /** 1..6 binary inputs. Rows are ordered as 00..0, 00..1, ... 11..1, most significant bit first. */
  inputs: number;
  /** 1..16 binary outputs. */
  outputs: number;
  /** Complete flattened row-major table of length (2 ** inputs) * outputs, containing only numeric 0 and 1. */
  targets: number[];
}
/** Experimental direct compilation. Exact on the complete binary domain; unconstrained behavior for nonbinary inputs. */
export function compileBooleanTable(table: BooleanTable): DenseModel;
export interface HebbianOptions {
  layer: number;
  /** Explicit flattened row-major weight indices; at most 128, no bias edits. */
  indices: number[];
  /** In (0,1]. Correlation updates are clipped to maxAbsDelta and maxAbsValue. */
  learningRate: number;
  limits?: PatchLimits;
  simd?: boolean | 'auto';
}
/** Experimental correlation proposal using real-WASM activations. Always evaluate independently before promotion. */
export function createHebbianPatch(model: DenseModel | string, batch: WeightBatch, options: HebbianOptions): Promise<WeightPatch>;
