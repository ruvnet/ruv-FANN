export interface CatalogPackage {
  name: string; role: 'nucleus' | 'primitive' | 'satellite'; family: string; status: 'published' | 'candidate';
  version: string; repository: string; description: string; directory?: string;
  contracts?: string[]; links?: {npm?: string; source?: string}; proposedScope?: string;
}
export interface ConstellationCatalog {
  schema: 'ruvnet.constellation-catalog/1'; repository: string; packages: CatalogPackage[];
  profiles: {id: string; description: string; packages: string[]}[];
}
export interface DownloadDay { day: string; downloads: number }
export interface PackageObservation {
  name: string; status: 'ok' | 'unavailable' | 'candidate'; days: DownloadDay[];
  version?: string; dependencies?: string[]; peerDependencies?: string[]; optionalDependencies?: string[];
  metadataStatus?: 'ok' | 'unavailable'; metadataError?: string; error?: string; repository?: string;
  sources?: {downloads: string; metadata: string};
}
export interface RepositoryObservation {
  repository: string; status: 'ok' | 'unavailable'; stars?: number; openIssuesAndPRs?: number; source?: string; error?: string;
}
export interface ConstellationObservations {
  schema: 'ruvnet.constellation-observations/1'; observedAt: string; start: string; end: string;
  packages: PackageObservation[]; repositories: RepositoryObservation[];
}
export interface CollectionOptions {
  start: string; end: string;
  /** Trusted fetch capability; defaults to global fetch. Must honor AbortSignal for transport cleanup. */
  fetch?: typeof globalThis.fetch;
  /** Whole-collection deadline, default 20000 ms, maximum 60000. */
  timeoutMs?: number;
  /** 1..8 task runners; default 4. */
  concurrency?: number;
}
export interface AdoptionSample {
  /** Caller-supplied pseudonymous SHA-256 project identifier, never copied into the report. */
  sampleId: string;
  observedAt: string;
  packages: string[];
}
export interface ConstellationAdoption {
  schema: 'ruvnet.constellation-adoption/1'; consent: true; samples: AdoptionSample[];
}
export interface DeclaredLink { source: string; target: string; kind: 'runtime' | 'peer' | 'optional' }
export interface PackageMetrics {
  name: string; role: CatalogPackage['role']; family: string; catalogStatus: CatalogPackage['status']; status: PackageObservation['status'];
  catalogVersion: string; version: string | null; description: string; repository: string; npmURL: string | null; sourceURL: string;
  latestDayDownloads: number | null; last7Downloads: number | null; last30Downloads: number | null;
  weekdayBaseline: number | null; weekdayDeltaPercent: number | null; weekdayBaselineDays: number;
  days: DownloadDay[]; coverage: {observedDays: number; expectedDays: number}; metadataStatus: 'ok' | 'unavailable';
  dependencies: string[]; peerDependencies: string[]; optionalDependencies: string[];
  runtimeDependencyCount: number | null; peerDependencyCount: number | null; optionalDependencyCount: number | null; declaredLinks: DeclaredLink[]; error?: string;
}
export interface AttachMetrics {
  status: 'available' | 'unavailable'; unit: 'consenting-project-snapshots'; denominator: number; numerator: number; percentage: number | null; projects: number;
  perPrimitive: {name: string; numerator: number; denominator: number; percentage: number | null}[];
  source: {provided: boolean; samplesReceived: number; snapshotsInWindow: number; duplicateSnapshotsRemoved: number}; note: string;
}
export interface ConstellationReport {
  schema: 'ruvnet.constellation-report/1'; observedAt: string; start: string; end: string; packages: PackageMetrics[];
  totals: {
    latestDayDownloads: number | null; last7Downloads: number | null; last30Downloads: number | null;
    observedLast7Downloads: number; observedLast30Downloads: number;
    coverage: {catalogPackages: number; publishedPackages: number; candidatePackages: number; packagesWithLatestDay: number; packagesWithLast7: number; packagesWithLast30: number; unavailablePackages: number};
  };
  dailySeries: {day: string; downloads: number | null; observedDownloads: number; observedPackages: number; expectedPackages: number}[];
  repositories: {repository: string; status: 'ok' | 'unavailable'; stars: number | null; openIssuesAndPRs: number | null; source: string; error?: string}[];
  weekdaySummary: {eligiblePackages: number; aboveBaseline: number; netChange: number; families: {family: string; netChange: number; shareOfNetChangePercent: number | null}[]; note: string};
  correlations: {kind: 'daily-download-correlation'; consideredPackages: number; eligiblePairs: number; pairs: {left: string; right: string; r: number; samples: number}[]; note: string};
  declaredLinks: DeclaredLink[]; declaredLinksCoverage: {observedManifests: number; expectedManifests: number}; attach: AttachMetrics; definitions: {downloads: string; totals: string; weekday: string; repositories: string; adoption: string};
}
/** Read-only explicit collection; candidates are not queried and unknown metrics are never zero-filled. */
export function collectPublicMetrics(catalog: ConstellationCatalog, options: CollectionOptions): Promise<ConstellationObservations>;
/** Synchronous, owned report. Aggregate downloads cannot identify install attach or distinct users. */
export function analyzeConstellation(catalog: ConstellationCatalog, observations: ConstellationObservations, options?: {adoption?: ConstellationAdoption}): ConstellationReport;
/** Standalone HTML with local controls, safe DOM construction and no trackers or remote assets. */
export function renderConstellationDashboard(report: ConstellationReport): string;
export const TELEMETRY_LIMITS: Readonly<{maxPackages: number; maxDays: number; maxBytes: number; maxInputBytes: number; maxConcurrency: number; maxTimeoutMs: number; maxSamples: number; maxSamplePackages: number}>;
