// Exercise canonical/public SEO archives independently of source/workspace links.
import assert from 'node:assert/strict';
import {readFile,cp,readdir,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {verifyToolkitInstallation} from '../packages/wsp-toolkit/scripts/packed-consumer.mjs';

const target=resolve(process.argv[2]??''),sourceSha=process.argv[3];
assert.ok(process.argv[2]);assert.match(sourceSha??'',/^[a-f0-9]{40}$/);
const root=fileURLToPath(new URL('..',import.meta.url));
const config=JSON.parse(await readFile(join(root,'metadata-release.json'),'utf8'));
const checks=[];
const record=(name,tests)=>{assert.ok(tests>0);checks.push({name,tests,failed:0,skipped:0,todo:0});};
const run=(args,cwd=root)=>execFileSync(process.execPath,args,{cwd,encoding:'utf8',maxBuffer:12*1024*1024});
for(const spec of config.packages){
  const installed=JSON.parse(await readFile(join(target,'node_modules',spec.name,'package.json'),'utf8'));
  const source=JSON.parse(await readFile(join(root,spec.directory,'package.json'),'utf8'));
  assert.equal(installed.version,spec.toVersion);assert.deepEqual(installed.keywords,source.keywords);
}
record('exact installed metadata',config.packages.length);

// The source suites include real Chromium, workerd, worker threads and AgentDB.
run(['packages/wsp-toolkit/scripts/validate.mjs']);
const sourceReport=JSON.parse(await readFile(join(root,'packages/wsp-toolkit/.artifacts/validation.json'),'utf8'));
assert.ok(sourceReport.tests.tests>0);
assert.equal(sourceReport.tests.pass,sourceReport.tests.tests);
for(const key of ['fail','cancelled','skipped','todo'])assert.equal(sourceReport.tests[key],0);
assert.equal(sourceReport.audit.vulnerabilities.total,0);
assert.equal(sourceReport.agentdbIntegrationAudit.vulnerabilities.total,0);
record('source Chromium, workerd and integration regression suite',sourceReport.tests.tests);

// Copy only private test fixtures; every import below resolves the installed archives.
for(const [directory,name] of [['packages/wasm','ruv-fann-wasm'],['packages/wsp','ruv-fann-wsp']]){
  const tests=join(target,'node_modules',name,'test');
  await cp(join(root,directory,'test'),tests,{recursive:true});
  const files=(await readdir(tests)).filter(file=>file.endsWith('.test.mjs')).sort().map(file=>join(tests,file));
  const tap=run(['--test','--test-reporter=tap',...files],target);
  const count=key=>Number(tap.match(new RegExp(`^# ${key} (\\d+)$`,'m'))?.[1]);
  assert.ok(count('tests')>0);assert.equal(count('pass'),count('tests'));
  for(const key of ['fail','cancelled','skipped','todo'])assert.equal(count(key),0);
  record(`${name} actual installed archive tests`,count('tests'));
}

const imports=join(target,'metadata-imports.mjs');
await writeFile(imports,`for(const name of ${JSON.stringify(config.packages.map(p=>p.name))}){const module=await import(name);if(Object.keys(module).length===0)throw Error('No exports: '+name);}console.log('all imports passed');`);
run([imports],target);record('all package public entrypoints',config.packages.length);
const proof=await verifyToolkitInstallation(target);
assert.deepEqual(proof.results.map(x=>x.backend),['wasm','wasm-simd']);
for(const row of proof.results){assert.equal(row.candidateMSE,0);assert.equal(row.regressionRejected,true);assert.equal(row.rollbackRevision,3);}
assert.equal(proof.evolution.report.promotions,2);
assert.ok(new Set(proof.evolution.report.workerThreadIds).size>=2);
assert.deepEqual(proof.evolution.durability,{reopened:true,rollbackPersisted:true});
assert.equal(proof.constellation.scaffoldTests,3);
record('installed scalar and SIMD weight programming',proof.results.length);
record('installed concurrent workers and durable AgentDB rollback',proof.evolution.report.modules.length);
record('installed constellation generated contract tests',proof.constellation.scaffoldTests);
console.log(JSON.stringify({status:'passed',sourceSha,checks,runtime:process.version,
  sourceAudit:sourceReport.audit.vulnerabilities,agentdbAudit:sourceReport.agentdbIntegrationAudit.vulnerabilities,
  proof:{backends:proof.results.map(x=>x.backend),durability:proof.evolution.durability,workerThreads:proof.evolution.report.workerThreadIds.length}}));
