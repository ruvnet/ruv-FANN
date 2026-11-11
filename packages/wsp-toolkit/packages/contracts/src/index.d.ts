export type ConstellationRole = 'nucleus' | 'primitive' | 'satellite';
export interface CatalogPackage {
  name: string; role: ConstellationRole; family: string; status: 'published' | 'candidate'; version: string;
  repository: string; description: string; directory?: string; contracts?: string[];
  links?: { npm?: string; source?: string }; proposedScope?: string;
}
export interface CatalogProfile { id: string; description: string; packages: string[] }
export interface ConstellationCatalog { schema: 'ruvnet.constellation-catalog/1'; repository: string; packages: CatalogPackage[]; profiles: CatalogProfile[] }
export interface ConstellationModuleManifest { schema: 'ruvnet.constellation-module/1'; name: string; version: string; role: ConstellationRole; contracts: string[]; integrations: string[] }
export const CONTRACTS: Readonly<{
  model: 'ruv-fann.dense-layers/1'; patch: 'ruv-fann.weight-patch/1'; evaluation: 'ruv-fann.weight-evaluation/1';
  dataset: 'ruv-fann.weight-dataset/1'; policy: 'ruv-fann.weight-policy/1'; invariants: 'ruv-fann.invariants/1';
  invariantCheck: 'ruv-fann.invariant-check/1'; loop: 'ruv-fann.evolution-loop/1';
  module: 'ruvnet.constellation-module/1'; catalog: 'ruvnet.constellation-catalog/1';
}>;
export const CONTRACT_LIMITS: Readonly<{maxPackages: number; maxProfiles: number; maxContracts: number; maxIntegrations: number; maxBytes: number}>;
export function validateCatalog(value: unknown): ConstellationCatalog;
export function validateModuleManifest(value: unknown): ConstellationModuleManifest;
