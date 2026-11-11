import type { DenseModel, EvaluationDataset, EvaluationPolicy, WeightPatch } from 'ruv-fann-wsp';
import type { SearchOptions } from 'ruv-fann-wsp-search';
export type WorkerSearchOptions = Omit<SearchOptions, 'objective'>;
export interface SearchProposalJob { id: string; kind: 'search'; model: DenseModel | string; dataset: EvaluationDataset; policy: EvaluationPolicy; searchOptions: WorkerSearchOptions }
export interface PatchProposalJob { id: string; kind: 'patch'; patch: WeightPatch }
export type ProposalJob = SearchProposalJob | PatchProposalJob;
export interface ProposalSwarmOptions {
  /** 1..4 workers; default 2. */
  concurrency?: number;
  /** Per-job lifetime including startup; 1..60000 ms, default 5000. */
  timeoutMs?: number;
  /** May tighten the 100 million aggregate parameter-row reservation. */
  maxTotalWork?: number;
  signal?: AbortSignal;
}
export interface ProposalSwarmResult {
  id: string;
  status: 'ok' | 'error' | 'timeout' | 'cancelled';
  patch?: WeightPatch | null;
  workerThreadId?: number;
  search?: { evaluations: number; inferencePasses: number; reservedWork: number; backend: 'wasm' | 'wasm-simd' };
  error?: string;
  /** Monotonic parent-process performance.now() timestamps. Aborted queued jobs omit them. */
  startedAtMs?: number;
  finishedAtMs?: number;
}
export const SWARM_LIMITS: Readonly<{ maxJobs: 64; maxConcurrency: 4; maxBytes: number; maxTimeoutMs: 60000; maxTotalWork: number }>;
/** Validates data and returns conservative search work; patch transport reserves zero inference work. */
export function estimateSwarmWork(jobs: ProposalJob[]): number;
/** Resolves in input order only after every started worker has terminated. No checkpoint authority. */
export function runProposalSwarm(jobs: ProposalJob[], options?: ProposalSwarmOptions): Promise<ProposalSwarmResult[]>;
