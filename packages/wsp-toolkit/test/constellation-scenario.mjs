import {mkdtemp,readFile,access,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {execFileSync} from 'node:child_process';
import {readCatalog,planProfile,scaffoldSatellite} from 'ruvnet-constellation';
import {validateModuleManifest} from 'ruvnet-constellation-contracts';
import {constellationFixture} from './constellation-fixture.mjs';
const check=(ok,message)=>{if(!ok)throw new Error(`constellation scenario: ${message}`);};
export async function runConstellationScenario() {
  const catalog=await readCatalog(),plan=planProfile(catalog,'weights'),local=planProfile(catalog,'local-loop');
  check(plan.registrySpecs.length===1&&plan.registrySpecs[0]==='ruv-fann-wsp@0.2.0','declared primitive profile');
  check(local.publicInstallAvailable===true&&local.registrySpecs.length===4&&local.localCandidates.length===0,'declared release profile supplies explicit versioned install specs');
  const candidateCatalog=structuredClone(catalog),localNames=new Set(local.packages.map(pkg=>pkg.name));
  for(const entry of candidateCatalog.packages)if(localNames.has(entry.name))entry.status='candidate';
  const candidatePlan=planProfile(candidateCatalog,'local-loop');
  check(candidatePlan.publicInstallAvailable===false&&candidatePlan.registrySpecs.length===0&&candidatePlan.localCandidates.length===4,'independent source-only fixture cannot masquerade as registry installs');
  const directory=await mkdtemp(join(tmpdir(),'constellation-scenario-'));
  try {
    const scaffold=await scaffoldSatellite({directory:join(directory,'satellite'),name:'constellation-review-fixture',integrations:['ruv-fann-wsp']});
    check(scaffold.telemetry==='disabled'&&!scaffold.published,'scaffold remains an explicit local candidate');
    const module=validateModuleManifest(JSON.parse(await readFile(join(scaffold.directory,'constellation.json'),'utf8')));
    check(module.integrations.includes('ruv-fann-wsp'),'explicit integration contract');
    const manifest=JSON.parse(await readFile(join(scaffold.directory,'package.json'),'utf8'));
    check(!manifest.dependencies&&!manifest.peerDependencies,'no forced nucleus dependencies');
    const env={...process.env};delete env.NODE_TEST_CONTEXT;
    const output=execFileSync('npm',['test'],{cwd:scaffold.directory,encoding:'utf8',env});
    check(/(?:# pass 3|ℹ pass 3)/.test(output)&&!/(?:# fail [1-9]|ℹ fail [1-9]|# skipped [1-9]|ℹ skipped [1-9])/.test(output),'generated satellite tests actually execute');
    let installed=false;try{await access(join(scaffold.directory,'node_modules'));installed=true;}catch{}
    check(!installed,'scaffold works without install');
    const fixture=await constellationFixture({adoption:true}),unsampled=await constellationFixture();
    check(fixture.report.attach.denominator===3&&fixture.report.attach.numerator===2,'attach measures sampled projects');
    check(unsampled.report.attach.status==='unavailable'&&unsampled.report.attach.percentage===null,'downloads do not invent attach');
    check(!JSON.stringify(fixture.report).includes('1'.repeat(64)),'sample IDs are not published');
    return {schema:'ruvnet.constellation-acceptance/1',catalogEntries:catalog.packages.length,published:catalog.packages.filter(p=>p.status==='published').length,
      releaseProfileInstallSpecs:local.registrySpecs.length,candidateProfileBlocked:true,scaffoldTests:3,scaffoldNoInstall:true,attachUnit:fixture.report.attach.unit,attachDenominator:3,attachNumerator:2,unsampledAttach:'unavailable',sampleIDsExcluded:true};
  } finally {await rm(directory,{recursive:true,force:true});}
}
