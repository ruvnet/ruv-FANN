import { mkdtemp, readFile, writeFile, copyFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
export const toolkitRoot=resolve(dirname(fileURLToPath(import.meta.url)),'..');
export const names=['lifecycle','search','huggingface','cloudflare','ruos','invariants','swarm','registry','loop'].map(p=>`ruv-fann-wsp-${p}`).concat(['ruvnet-constellation-contracts','ruvnet-constellation','ruvnet-constellation-telemetry']);
export async function packToolkit(directory) {
  const output=JSON.parse(execFileSync('npm',['pack','--ignore-scripts','--json','--pack-destination',directory,...names.flatMap(n=>['--workspace',n])],{cwd:toolkitRoot,encoding:'utf8'}));
  return Promise.all(output.map(async p=>({name:p.name,version:p.version,filename:p.filename,bytes:p.size,
    sha256:createHash('sha256').update(await readFile(join(directory,p.filename))).digest('hex')})));
}
export async function verifyPackedConsumer() {
  const directory=await mkdtemp(join(tmpdir(),'wsp-consumer-'));
  try {
    const archives=await packToolkit(directory);
    await writeFile(join(directory,'package.json'),JSON.stringify({private:true,type:'module'}));
    execFileSync('npm',['install','--ignore-scripts','--prefer-offline','--no-audit','--no-fund',...archives.map(p=>join(directory,p.filename))],{cwd:directory,encoding:'utf8'});
    await copyFile(join(toolkitRoot,'test','scenario.mjs'),join(directory,'scenario.mjs'));
    await copyFile(join(toolkitRoot,'test','types.ts'),join(directory,'types.ts'));
    await copyFile(join(toolkitRoot,'test','evolution-scenario.mjs'),join(directory,'evolution-scenario.mjs'));
    for(const file of ['constellation-scenario.mjs','constellation-fixture.mjs'])await copyFile(join(toolkitRoot,'test',file),join(directory,file));
    execFileSync(process.execPath,[join(toolkitRoot,'node_modules/typescript/bin/tsc'),'--noEmit','--strict','--module','nodenext','--target','es2023','--lib','es2023,dom','types.ts'],{cwd:directory,encoding:'utf8'});
    const stdout=execFileSync(process.execPath,['--input-type=module','-e',
      "import {pathToFileURL} from 'node:url'; import {runScenario} from './scenario.mjs'; import {runEvolutionScenario} from './evolution-scenario.mjs'; import {runConstellationScenario} from './constellation-scenario.mjs'; import {createAgentDBRegistry} from 'ruv-fann-wsp-registry'; import {buildWorker} from 'ruv-fann-wsp-cloudflare/build'; const {openDemoRegistry}=await import(pathToFileURL(process.argv[1]).href); await buildWorker('./worker'); const evolution=await runEvolutionScenario(path=>openDemoRegistry(path,createAgentDBRegistry),'.'); console.log('WSP_CONSUMER_RESULT '+JSON.stringify({results:[await runScenario(false),await runScenario(true)],evolution,constellation:await runConstellationScenario()}));",join(toolkitRoot,'tooling/agentdb/runtime.mjs')],{cwd:directory,encoding:'utf8'});
    const line=stdout.split('\n').find(line=>line.startsWith('WSP_CONSUMER_RESULT '));
    if(!line)throw new Error('packed consumer result is missing');
    const result=JSON.parse(line.slice('WSP_CONSUMER_RESULT '.length));
    const light=await mkdtemp(join(tmpdir(),'constellation-light-'));
    let lightweight;
    try {
      await writeFile(join(light,'package.json'),JSON.stringify({private:true,type:'module'}));
      const small=archives.filter(p=>['ruvnet-constellation','ruvnet-constellation-contracts'].includes(p.name));
      execFileSync('npm',['install','--ignore-scripts','--prefer-offline','--no-audit','--no-fund',...small.map(p=>join(directory,p.filename))],{cwd:light,encoding:'utf8'});
      const tree=JSON.parse(execFileSync('npm',['ls','--all','--json'],{cwd:light,encoding:'utf8'}));
      const found=new Set();const walk=tree=>{for(const [name,value]of Object.entries(tree.dependencies??{})){found.add(name);walk(value);}};walk(tree);
      if(found.size!==2||!found.has('ruvnet-constellation')||!found.has('ruvnet-constellation-contracts'))throw Error('lightweight onramp installed unexpected dependencies');
      execFileSync(process.execPath,['--input-type=module','-e',"import {readCatalog,planProfile} from 'ruvnet-constellation'; if(planProfile(await readCatalog(),'weights').registrySpecs[0]!=='ruv-fann-wsp@0.2.0')throw Error('light profile');"],{cwd:light,encoding:'utf8'});
      lightweight={packageCount:found.size,mlStackInstalled:false};
    }finally{await rm(light,{recursive:true,force:true});}
    return {schema:'ruv-fann.packed-consumer/1',archives,...result,lightweight};
  }finally{await rm(directory,{recursive:true,force:true});}
}
