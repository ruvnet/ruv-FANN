import type { DenseModel, WeightPatch, EvaluationDataset, EvaluationPolicy, EvaluationOptions, WeightEvaluation } from 'ruv-fann-wsp';
export type GateEvaluation = Omit<WeightEvaluation, 'model' | 'patch'>;
export interface RegressionSuite { name: string; dataset: EvaluationDataset; policy: EvaluationPolicy }
export interface RegressionGate { schema: 'ruv-fann.regression-gate/1'; baseHash: string; candidateHash: string; accepted: boolean; suites: { name: string; evaluation: GateEvaluation }[] }
export function runRegressionGate(bundle: {model: DenseModel | string; patch: WeightPatch; suites: RegressionSuite[]}, options?: EvaluationOptions): Promise<RegressionGate>;
export function evaluatePatchBranches(model: DenseModel | string, patches: WeightPatch[], dataset: EvaluationDataset, policy: EvaluationPolicy, options?: EvaluationOptions): Promise<{schema: 'ruv-fann.patch-branches/1'; baseHash: string; branches: {index: number; evaluation: WeightEvaluation}[]}>;
export const LIFECYCLE_LIMITS: Readonly<{maxSuites: number; maxBranches: number; maxBytes: number; maxTotalWork: number; maxTotalElements: number}>;
