import type { DenseModel, WeightPatch, EvaluationDataset, EvaluationPolicy, WeightEvaluation } from 'ruv-fann-wsp';
export type StoredWeightEvaluation = Omit<WeightEvaluation, 'model'>;
export interface WeightProgramMetadata { id: string; revision: number; hash: string; createdAt: string; updatedAt: string }
export interface WeightProgramHistory {
  revision: number; hash: string; operation: 'create' | 'promote' | 'rollback'; sourceRevision: number | null;
  proposalId: string | null; createdAt: string; evaluation: StoredWeightEvaluation | null;
}
export interface WeightProgramProposal {
  id: string; revision: number; baseHash: string; candidateHash: string; accepted: boolean;
  status: 'pending' | 'rejected' | 'stale' | 'promoted' | 'expired'; createdAt: string; expiresAt: string;
  evaluation: StoredWeightEvaluation;
}
export interface WeightProgramRecord extends WeightProgramMetadata { model: DenseModel; history: WeightProgramHistory[]; proposals: WeightProgramProposal[] }
export interface WeightProgramProposed { id: string; proposalId: string; revision: number; evaluation: WeightEvaluation; createdAt: string; expiresAt: string }
export interface WeightProgramsCapability {
  create(id: string, model: DenseModel | string): Promise<WeightProgramRecord>;
  get(id: string): WeightProgramRecord | null | Promise<WeightProgramRecord | null>;
  list(limit?: number): WeightProgramMetadata[] | Promise<WeightProgramMetadata[]>;
  propose(id: string, patch: WeightPatch, dataset: EvaluationDataset, policy: EvaluationPolicy): Promise<WeightProgramProposed>;
  promote(id: string, proposalId: string, expectedRevision: number): Promise<WeightProgramRecord>;
  rollback(id: string, targetRevision: number, expectedRevision: number): WeightProgramRecord | Promise<WeightProgramRecord>;
}
export interface RegistryOptions {
  /** Trusted persistence hook. Omitting this yields explicitly memory-only acknowledgements. */
  flush?: () => void | Promise<void>;
  /** Called once after queued work settles, including after failed persistence. */
  close?: (state: Readonly<{poisoned: boolean}>) => void | Promise<void>;
  /** May tighten the hard 16 queued-operation limit. */
  maxPending?: number;
}
export interface AgentDBRegistry {
  create(id: string, model: DenseModel | string): Promise<WeightProgramRecord>;
  get(id: string): Promise<WeightProgramRecord | null>;
  list(limit?: number): Promise<WeightProgramMetadata[]>;
  propose(id: string, patch: WeightPatch, dataset: EvaluationDataset, policy: EvaluationPolicy): Promise<WeightProgramProposed>;
  promote(id: string, proposalId: string, expectedRevision: number): Promise<WeightProgramRecord>;
  rollback(id: string, targetRevision: number, expectedRevision: number): Promise<WeightProgramRecord>;
  close(): Promise<void>;
  status(): Readonly<{pending: number; closing: boolean; closed: boolean; poisoned: boolean; durability: 'caller-flush' | 'memory-only'}>;
}
export function createAgentDBRegistry(programs: WeightProgramsCapability, options?: RegistryOptions): Readonly<AgentDBRegistry>;
export const REGISTRY_LIMITS: Readonly<{maxPending: number; maxBytes: number; maxNodes: number; maxDepth: number}>;
export class RegistryPersistenceError extends Error { readonly code: 'WSP_REGISTRY_PERSISTENCE'; constructor(cause: unknown) }
export class RegistryStaleError extends Error { readonly code: 'WSP_REGISTRY_STALE'; constructor(cause: unknown) }
