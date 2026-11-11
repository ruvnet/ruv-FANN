import type {ConstellationCatalog,CatalogPackage} from 'ruvnet-constellation-contracts';
export interface ConstellationPlan {
  schema:'ruvnet.constellation-plan/1'; profile:string; description:string; packages:CatalogPackage[];
  registrySpecs:string[]; localCandidates:{name:string;directory?:string;repository:string}[];
  publicInstallAvailable:boolean; automaticInstall:false; telemetry:'disabled';
}
export interface SatelliteScaffoldOptions {directory:string;name:string;description?:string;integrations?:string[]}
export interface SatelliteScaffold {schema:'ruvnet.constellation-scaffold/1';directory:string;name:string;status:'candidate';files:string[];automaticInstall:false;telemetry:'disabled';published:false}
export function readCatalog(path?:string|URL):Promise<ConstellationCatalog>;
export function planProfile(catalog:ConstellationCatalog,id:string):ConstellationPlan;
/** Scaffolding requires Linux directory descriptors and /proc/self/fd; fails closed elsewhere. */
export function scaffoldSatellite(options:SatelliteScaffoldOptions):Promise<SatelliteScaffold>;
