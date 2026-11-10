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
