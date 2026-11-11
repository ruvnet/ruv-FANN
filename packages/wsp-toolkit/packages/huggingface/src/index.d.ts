import type { DenseModel, WeightModelInspection } from 'ruv-fann-wsp';

export interface HubModelSource {
  /** Public model repository in owner/name form. */
  repo: string;
  /** Immutable lowercase 40-character commit SHA; branches and tags are rejected. */
  revision: string;
  /** Relative .json file path. */
  path: string;
  /** SHA-256 of the raw downloaded bytes, distinct from the canonical model hash. */
  sha256: string;
  /** May tighten the 1 MiB maximum. */
  maxBytes?: number;
}
export interface HubLoadOptions {
  /** Trusted fetch capability; defaults to globalThis.fetch. */
  fetch?: typeof globalThis.fetch;
  signal?: AbortSignal;
  /** Download deadline, 1..60000 ms; default 15000. */
  timeoutMs?: number;
}
export interface HubModel {
  schema: 'ruv-fann.hub-model/1';
  model: DenseModel;
  inspection: WeightModelInspection;
  provenance: { repo: string; revision: string; path: string; sha256: string; bytes: number; url: string };
}
/** Public dense-layers JSON only. At most three same-origin redirects, no cookies or tokens. */
export function loadHubModel(source: HubModelSource, options?: HubLoadOptions): Promise<HubModel>;

export interface HubLinearHead extends Omit<HubModel, 'schema'> {
  schema: 'ruv-fann.hub-linear-head/1';
  interpretation: 'affine-projection';
}
/** Strict {weights:number[],bias:number} → one Linear output with steepness 1.
 * This affine projection does not reconstruct an originating model's activations or preprocessing.
 */
export function loadHubLinearHead(source: HubModelSource, options?: HubLoadOptions): Promise<HubLinearHead>;
