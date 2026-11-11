import { execFileSync } from 'node:child_process';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const output=join(root,'.artifacts');await mkdir(output,{recursive:true});
const run=(command,args)=>execFileSync(command,args,{cwd:root,encoding:'utf8',maxBuffer:8*1024*1024});
run(process.execPath,['node_modules/typescript/bin/tsc','--noEmit','-p','tsconfig.json']);
run(process.execPath,['scripts/constellation-links.mjs','--check']);
const tests=[];
for(const pkg of (await readdir(join(root,'packages'))).sort()) {
  const dir=`packages/${pkg}/test`;
  for(const file of (await readdir(join(root,dir))).sort())if(file.endsWith('.test.mjs'))tests.push(`${dir}/${file}`);
}
for(const file of (await readdir(join(root,'test'))).sort())if(file.endsWith('.test.mjs'))tests.push(`test/${file}`);
tests.push('../../scripts/constellation-oidc.test.mjs','../../scripts/wsp-release-evidence.test.mjs');
const tap=run(process.execPath,['--test','--test-reporter=tap',...tests]);
await writeFile(join(output,'tests.tap'),tap);
const count=name=>Number(tap.match(new RegExp(`^# ${name} (\\d+)$`,'m'))?.[1]??NaN);
const summary=Object.fromEntries(['tests','pass','fail','cancelled','skipped','todo'].map(name=>[name,count(name)]));
if(!Number.isInteger(summary.tests)||summary.tests<1||summary.pass!==summary.tests||['fail','cancelled','skipped','todo'].some(k=>summary[k]!==0))throw Error('tests must pass with zero cancellations, skips or todos');
const benchmark=JSON.parse(run(process.execPath,['scripts/benchmark.mjs']));
const audit=JSON.parse(run('npm',['audit','--json']));
if(audit.metadata?.vulnerabilities?.total!==0)throw Error('dependency audit must report zero known vulnerabilities');
const agentdbAudit=JSON.parse(run('npm',['audit','--json','--prefix','tooling/agentdb','--omit=optional']));
if(agentdbAudit.metadata?.vulnerabilities?.total!==0)throw Error('AgentDB integration dependency audit must report zero known vulnerabilities');
const files=[];
async function walk(relative='') {
  for(const item of (await readdir(join(root,relative),{withFileTypes:true})).sort((a,b)=>a.name.localeCompare(b.name))) {
    if(['node_modules','.artifacts','evidence'].includes(item.name))continue;
    const name=relative?`${relative}/${item.name}`:item.name;
    if(item.isDirectory())await walk(name);
    else if(item.isFile())files.push({path:name,sha256:createHash('sha256').update(await readFile(join(root,name))).digest('hex')});
  }
}
await walk();
for(const external of ['../../scripts/constellation-oidc.mjs','../../scripts/constellation-oidc.test.mjs','../../scripts/wsp-release-lane.mjs','../../.github/workflows/protected-wsp-release.yml','../../.github/workflows/wsp-modules.yml','../../.github/workflows/wsp-toolkit.yml','../../.github/workflows/constellation-observatory.yml','../../docs/constellation-design.md','../../docs/constellation-release.md','../../docs/constellation.md'])files.push({path:external,sha256:createHash('sha256').update(await readFile(join(root,external))).digest('hex')});
const report={schema:'ruv-fann.toolkit-validation/1',createdAt:new Date().toISOString(),node:process.version,
  typecheck:'passed',tests:summary,benchmark,audit:{vulnerabilities:audit.metadata.vulnerabilities,scope:'all production and development dependencies',lockfileSHA256:createHash('sha256').update(await readFile(join(root,'package-lock.json'))).digest('hex')},
  agentdbIntegrationAudit:{vulnerabilities:agentdbAudit.metadata.vulnerabilities,scope:'installed integration tree, optional dependencies omitted',lockfileSHA256:createHash('sha256').update(await readFile(join(root,'tooling/agentdb/package-lock.json'))).digest('hex')},
  sourceTreeSHA256:createHash('sha256').update(JSON.stringify(files)).digest('hex'),sourceFiles:files,
  limits:'Local validation only. Source digest excludes generated .artifacts and docs/evidence. This is not a signature or deployment authorization.'};
await writeFile(join(output,'validation.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({typecheck:report.typecheck,tests:summary,audit:audit.metadata.vulnerabilities,agentdbIntegrationAudit:agentdbAudit.metadata.vulnerabilities,
  inferencePassReductionPercent:benchmark.inferencePassReductionPercent,sourceTreeSHA256:report.sourceTreeSHA256,report:'.artifacts/validation.json'},null,2));
