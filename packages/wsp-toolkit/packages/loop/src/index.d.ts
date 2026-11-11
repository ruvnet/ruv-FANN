import type { DenseModel, WeightPatch, WeightEdit, EvaluationDataset, EvaluationPolicy, PatchLimits, WeightEvaluation } from 'ruv-fann-wsp';
import type { SearchOptions } from 'ruv-fann-wsp-search';
import type { InvariantSuite } from 'ruv-fann-wsp-invariants';
export interface LoopProgram { revision: number; hash: string; model: DenseModel }
export interface LoopRegistry {
  get(id: string): Promise<LoopProgram | null>;
  propose(id: string, patch: WeightPatch, dataset: EvaluationDataset, policy: EvaluationPolicy): Promise<{revision: number; proposalId: string; evaluation: WeightEvaluation}>;
  promote(id: string, proposalId: string, expectedRevision: number): Promise<LoopProgram>;
}
export interface LoopSuite { name: string; dataset: EvaluationDataset; policy: EvaluationPolicy }
export type LoopStrategy = { name: string; kind: 'search'; options: Omit<SearchOptions, 'objective'> } | {name: string; kind: 'patch'; edits: WeightEdit[]};
export interface EvolutionModule {
  id: string;
  development: {dataset: EvaluationDataset; policy: EvaluationPolicy};
  /** Trusted coordinator inputs; these datasets are never dispatched to proposers. */
  holdouts: LoopSuite[];
  invariants: InvariantSuite;
  strategies: LoopStrategy[];
  minImprovement?: number;
  goalMSE?: number;
  limits?: PatchLimits;
}
export interface EvolutionLoopOptions {
  registry: LoopRegistry;
  modules: EvolutionModule[];
  maxRounds?: number;
  maxProposals?: number;
  maxWork?: number;
  concurrency?: number;
  jobTimeoutMs?: number;
  maxDurationMs?: number;
  simd?: boolean | 'auto';
  signal?: AbortSignal;
}
export interface EvolutionEvent {
  round: number; moduleId: string; revision: number; strategy: string;
  type: 'worker-error' | 'no-candidate' | 'proposed' | 'duplicate' | 'rejected' | 'validated' | 'stale' | 'promoted';
  reason?: string; candidateHash?: string; baselineMSE?: number; candidateMSE?: number; nextRevision?: number;
  failedSuites?: string[]; invariantSuiteHash?: string;
  suiteHashes?: {name: string; datasetHash: string; policyHash: string}[];
}
export interface EvolutionLoopReport {
  schema: 'ruv-fann.evolution-loop/1';
  stopReason: 'round-limit' | 'proposal-limit' | 'work-limit' | 'deadline' | 'aborted' | 'goals-met' | 'no-progress';
  rounds: number; promotions: number;
  budget: { maxRounds: number; maxProposals: number; maxWork: number; concurrency: number; jobTimeoutMs: number; maxDurationMs: number; proposals: number; reservedWork: number; elapsedMs: number };
  workerThreadIds: number[];
  modules: { id: string; initialRevision: number; initialHash: string; finalRevision: number; finalHash: string; promotions: number; goalReached: boolean }[];
  events: EvolutionEvent[];
}
export const LOOP_LIMITS: Readonly<{maxModules: number; maxRounds: number; maxProposals: number; maxWork: number; maxConcurrency: number; maxDurationMs: number}>;
export function runEvolutionLoop(options: EvolutionLoopOptions): Promise<EvolutionLoopReport>;
