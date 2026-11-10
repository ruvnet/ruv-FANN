import type { DenseModel } from 'ruv-fann';
export type { DenseModel, DenseLayer, DenseActivation } from 'ruv-fann';
export interface WeightEdit {
  layer: number;
  parameter: 'weights' | 'bias';
  index: number;
  value: number;
}
export interface WeightPatch {
  schema: 'ruv-fann.weight-patch/1';
  baseHash: string;
  edits: WeightEdit[];
}
/** Limits may only tighten the defaults: 128 edits, abs(value)<=1e6, abs(delta)<=1000. */
export interface PatchLimits { maxEdits?: number; maxAbsValue?: number; maxAbsDelta?: number }
export interface EvaluationDataset { inputs: number[]; targets: number[]; rows: number }
export interface EvaluationPolicy { maxMSE: number; maxRegressionMSE: number; maxAbsoluteError?: number }
export interface EvaluationOptions { simd?: boolean | 'auto'; limits?: PatchLimits }
export interface WeightMetrics { mse: number; maxAbsoluteError: number }
/** Hash-bound evidence, not a signature or an authorization receipt. Caller owns this result. */
export interface WeightEvaluation {
  schema: 'ruv-fann.weight-evaluation/1';
  baseHash: string;
  candidateHash: string;
  datasetHash: string;
  policyHash: string;
  accepted: boolean;
  baseline: WeightMetrics;
  candidate: WeightMetrics;
  model: DenseModel;
  patch: WeightPatch;
  policy: EvaluationPolicy;
  rows: number;
  backend: 'wasm' | 'wasm-simd';
  reasons: string[];
}
export interface WeightSnapshot { revision: number; hash: string; model: DenseModel }
export interface WeightProposal { id: string; revision: number; evaluation: WeightEvaluation }
export interface WeightHistoryEntry {
  revision: number;
  hash: string;
  action: 'initial' | 'promote' | 'rollback';
  previousRevision?: number;
  proposalId?: string;
  targetRevision?: number;
}
export interface WeightWorkspace {
  snapshot(): WeightSnapshot;
  propose(patch: WeightPatch, dataset: EvaluationDataset, policy: EvaluationPolicy): Promise<WeightProposal>;
  promote(id: string, expectedRevision: number): WeightSnapshot;
  rollback(targetRevision: number, expectedRevision: number): WeightSnapshot;
  history(): WeightHistoryEntry[];
}
export function fingerprintModel(model: DenseModel | string): Promise<string>;
export function applyWeightPatch(model: DenseModel | string, patch: WeightPatch, limits?: PatchLimits): Promise<DenseModel>;
export function evaluateWeightPatch(model: DenseModel | string, patch: WeightPatch, dataset: EvaluationDataset,
  policy: EvaluationPolicy, options?: EvaluationOptions): Promise<WeightEvaluation>;
/** Produces a bounded prompt string only. Browsers require secure-context WebCrypto. */
export function createWeightEditPrompt(model: DenseModel | string, specification: string, limits?: PatchLimits): Promise<string>;
export function createWeightWorkspace(model: DenseModel | string, options?: EvaluationOptions): Promise<WeightWorkspace>;
export const WSP_LIMITS: Readonly<{ maxEdits: number; maxAbsValue: number; maxAbsDelta: number;
  maxRows: number; maxElements: number; maxParameters: number; maxModelBytes: number; maxWork: number;
  maxLayers: number; maxDimension: number; maxSpecificationChars: number; maxPromptModelBytes: number;
  maxPromptBytes: number; maxProposals: number; maxHistory: number }>;

export interface WeightLayerInspection {
  layer: number;
  inputs: number;
  outputs: number;
  activation: import('ruv-fann').DenseActivation;
  steepness: number;
  weightCount: number;
  biasCount: number;
  parameterCount: number;
}
export interface WeightModelInspection {
  schema: 'ruv-fann.weight-model-inspection/1';
  baseHash: string;
  parameterCount: number;
  inputSize: number;
  outputSize: number;
  layers: WeightLayerInspection[];
}
export interface WeightBatch { inputs: number[]; rows: number }
export interface WeightTraceOptions { simd?: boolean | 'auto' }
export interface WeightLayerTrace {
  layer: number;
  inputs: number;
  outputs: number;
  /** Raw affine sums, before activation steepness. Flattened row-major. */
  preactivations: number[];
  /** Actual WASM activation outputs. Flattened row-major. */
  activations: number[];
}
export interface WeightTrace {
  schema: 'ruv-fann.weight-trace/1';
  baseHash: string;
  inputHash: string;
  backend: 'wasm' | 'wasm-simd';
  rows: number;
  layers: WeightLayerTrace[];
}
export interface WeightLocation { layer: number; parameter: 'weights' | 'bias'; index: number }
/** At most 16 unique locations. Epsilon defaults to 1e-4 and must be in [1e-8,1]. */
export interface WeightSensitivityOptions extends WeightTraceOptions { epsilon?: number }
export interface WeightSensitivityEntry extends WeightLocation {
  minus: WeightMetrics;
  plus: WeightMetrics;
  /** Central difference of MSE divided by the actual represented parameter difference. Not autograd. */
  derivative: number;
}
export interface WeightSensitivity {
  schema: 'ruv-fann.weight-sensitivity/1';
  method: 'central-finite-difference';
  baseHash: string;
  datasetHash: string;
  backend: 'wasm' | 'wasm-simd';
  epsilon: number;
  baseline: WeightMetrics;
  entries: WeightSensitivityEntry[];
}
export function inspectWeightModel(model: DenseModel | string): Promise<WeightModelInspection>;
/** Two real-WASM passes per layer; aggregate work and returned trace elements are bounded. */
export function traceWeightModel(model: DenseModel | string, batch: WeightBatch, options?: WeightTraceOptions): Promise<WeightTrace>;
/** Bounded numerical MSE sensitivity; no gradients, provider calls or model updates. */
export function analyzeWeightSensitivity(model: DenseModel | string, dataset: EvaluationDataset,
  locations: WeightLocation[], options?: WeightSensitivityOptions): Promise<WeightSensitivity>;
/** Disconnect one hidden neuron by zeroing its outgoing weights. Does not alter topology. */
export function createNeuronAblationPatch(model: DenseModel | string, layer: number, neuron: number,
  limits?: PatchLimits): Promise<WeightPatch>;
