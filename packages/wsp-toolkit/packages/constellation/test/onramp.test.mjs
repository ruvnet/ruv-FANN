import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm,mkdir,symlink,readdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {execFileSync,spawnSync} from 'node:child_process';
import {Worker} from 'node:worker_threads';
import {readCatalog,planProfile,scaffoldSatellite} from '../src/index.js';
const cli=new URL('../src/cli.js',import.meta.url).pathname;
async function temp(t){const root=await mkdtemp(join(tmpdir(),'constellation-onramp-'));t.after(()=>rm(root,{recursive:true,force:true}));return root;}

test('default release catalog plans explicit published pins without installing packages',async()=>{
  const catalog=await readCatalog(),weights=planProfile(catalog,'weights'),local=planProfile(catalog,'local-loop');
  assert.deepEqual(weights.registrySpecs,['ruv-fann-wsp@0.2.0']);assert.equal(weights.publicInstallAvailable,true);
  assert.deepEqual(local.registrySpecs,['ruv-fann-wsp-loop@0.1.0','ruv-fann-wsp-registry@0.1.0','ruv-fann-wsp-swarm@0.1.0','ruv-fann-wsp-invariants@0.1.0']);
  assert.equal(local.localCandidates.length,0);assert.equal(local.publicInstallAvailable,true);
  assert.equal(catalog.packages.length,24);assert.ok(catalog.packages.every(pkg=>pkg.status==='published'));
  assert.equal(catalog.packages.find(pkg=>pkg.name==='ruv-fann').version,'0.2.0');
  assert.equal(local.automaticInstall,false);assert.equal(local.telemetry,'disabled');
  catalog.profiles[0].description='changed';assert.notEqual(weights.description,'changed');
  assert.throws(()=>planProfile(catalog,'absent'));
});
test('independent source-candidate catalog remains distinct from registry install specifications',()=>{
  const catalog={schema:'ruvnet.constellation-catalog/1',repository:'example/adapters',packages:[
    {name:'published-adapter',version:'1.2.3',role:'primitive',family:'example',status:'published',repository:'example/adapters',description:'Published fixture'},
    {name:'source-adapter',version:'0.1.0',role:'satellite',family:'example',status:'candidate',repository:'example/adapters',directory:'packages/source',description:'Source-only fixture'}
  ],profiles:[{id:'mixed',description:'Explicit fixture status distinction',packages:['published-adapter','source-adapter']}]};
  const plan=planProfile(catalog,'mixed');
  assert.deepEqual(plan.registrySpecs,['published-adapter@1.2.3']);
  assert.equal(plan.publicInstallAvailable,false);assert.equal(plan.localCandidates[0].name,'source-adapter');
  assert.equal(plan.automaticInstall,false);assert.equal(plan.telemetry,'disabled');
});
test('local catalog reader rejects oversized, nonregular, invalid UTF-8 and HTTP inputs',async t=>{
  const root=await temp(t),file=join(root,'catalog.json');
  await writeFile(file,JSON.stringify(await readCatalog()));assert.ok((await readCatalog(pathToFileURL(file))).packages.length>0);
  await writeFile(file,'x'.repeat(2*1024*1024+1));await assert.rejects(readCatalog(file),/byte/);
  await writeFile(file,Uint8Array.of(0xc3,0x28));await assert.rejects(readCatalog(file));
  await assert.rejects(readCatalog(root),/regular/);await assert.rejects(readCatalog(new URL('https://example.test/catalog.json')));
  const pipe=join(root,'fifo');execFileSync('mkfifo',[pipe]);
  const attempt=spawnSync(process.execPath,['--input-type=module','-e',`import{readCatalog}from ${JSON.stringify(new URL('../src/index.js',import.meta.url).href)}; await readCatalog(${JSON.stringify(pipe)});`],{timeout:2000,encoding:'utf8'});
  assert.notEqual(attempt.status,0);assert.equal(attempt.signal,null);assert.match(attempt.stderr,/regular file/);
});
test('CLI help/catalog/plan produce explicit local output and refuse unknown commands',()=>{
  const help=execFileSync(process.execPath,[cli,'help'],{encoding:'utf8'});
  assert.match(help,/https:\/\/www.npmjs.com\/package\/agentdb/);assert.match(help,/No command/);
  const output=JSON.parse(execFileSync(process.execPath,[cli,'catalog','--json'],{encoding:'utf8'}));assert.ok(output.packages.length>0);
  const plan=JSON.parse(execFileSync(process.execPath,[cli,'plan','local-loop'],{encoding:'utf8'}));assert.equal(plan.publicInstallAvailable,true);assert.equal(plan.registrySpecs.length,4);
  for(const args of [['publish'],['plan'],['catalog','--token','example'],['scaffold','x','--name','ok','--exec','id']]) {
    const rejected=spawnSync(process.execPath,[cli,...args],{encoding:'utf8'});assert.equal(rejected.status,2);
  }
});
test('fresh scoped scaffold runs its independent checks and benchmark without installing a stack',async t=>{
  const root=await temp(t),directory=join(root,'adapter');
  const result=await scaffoldSatellite({directory,name:'@example/adapter',description:'Fixture adapter',integrations:['ruv-fann-wsp']});
  assert.equal(result.published,false);assert.equal(result.automaticInstall,false);assert.equal(result.files.length,8);
  const manifest=JSON.parse(await readFile(join(directory,'package.json'),'utf8'));
  assert.equal(manifest.name,'@example/adapter');assert.equal(manifest.dependencies,undefined);assert.equal(manifest.peerDependencies,undefined);
  const module=JSON.parse(await readFile(join(directory,'constellation.json'),'utf8'));assert.deepEqual(module.integrations,['ruv-fann-wsp']);
  const childEnv={...process.env};delete childEnv.NODE_TEST_CONTEXT;
  const tests=execFileSync('npm',['test'],{cwd:directory,encoding:'utf8',env:childEnv});assert.match(tests,/(?:pass 3|✔)/);
  const benchmark=execFileSync(process.execPath,['benchmark.mjs'],{cwd:directory,encoding:'utf8'});assert.equal(JSON.parse(benchmark).iterations,1000);
  const {adaptRecord}=await import(pathToFileURL(join(directory,'src/index.js')).href);
  const source={x:[1,2]};const owned=adaptRecord(source);source.x[0]=9;assert.equal(owned.x[0],1);
  assert.equal((await readdir(directory)).includes('node_modules'),false);
});
test('existing targets, symlinks and traversal cannot overwrite sentinels or escape explicit target',async t=>{
  const root=await temp(t),existing=join(root,'existing'),outside=join(root,'outside');await mkdir(existing);await mkdir(outside);
  await writeFile(join(existing,'sentinel'),'keep');await writeFile(join(outside,'sentinel'),'outside');
  await assert.rejects(scaffoldSatellite({directory:existing,name:'test-adapter'}));
  const link=join(root,'link');await symlink(outside,link,'dir');
  await assert.rejects(scaffoldSatellite({directory:link,name:'test-adapter'}));
  await assert.rejects(scaffoldSatellite({directory:join(link,'child'),name:'test-adapter'}),/parent/);
  await assert.rejects(scaffoldSatellite({directory:existing+'/../escape',name:'test-adapter'}),/traversal/);
  assert.equal(await readFile(join(existing,'sentinel'),'utf8'),'keep');assert.equal(await readFile(join(outside,'sentinel'),'utf8'),'outside');
  assert.deepEqual((await readdir(outside)).sort(),['sentinel']);
});
test('configuration is snapshotted before filesystem awaits and accessors never execute',async t=>{
  const root=await temp(t),options={directory:join(root,'snapshot'),name:'original-adapter',integrations:['agentdb']};
  const pending=scaffoldSatellite(options);options.name='changed-adapter';options.integrations.push('ruvector');
  const result=await pending;assert.equal(result.name,'original-adapter');
  const manifest=JSON.parse(await readFile(join(result.directory,'constellation.json'),'utf8'));assert.deepEqual(manifest.integrations,['agentdb']);
  let calls=0;const accessor={directory:join(root,'getter'),get name(){calls++;return 'bad';}};
  await assert.rejects(scaffoldSatellite(accessor));assert.equal(calls,0);
  for(const name of ['../escape','BadName','a;$(id)'])await assert.rejects(scaffoldSatellite({directory:join(root,'bad'),name}));
  await assert.rejects(scaffoldSatellite({directory:join(root,'bad'),name:'safe',exec:()=>{calls++;}}));assert.equal(calls,0);
});
test('CLI scaffold handles literal shell metacharacters as path data without shell execution',async t=>{
  const root=await temp(t),directory=join(root,"literal;$(touch escaped)");
  const result=JSON.parse(execFileSync(process.execPath,[cli,'scaffold',directory,'--name','literal-adapter','--json'],{encoding:'utf8'}));
  assert.equal(result.directory,directory);assert.deepEqual((await readdir(root)).sort(),["literal;$(touch escaped)"]);
});
test('concurrent child directory symlink substitution cannot redirect writes outside the new target',{timeout:12000},async t=>{
  const root=await temp(t);
  for(const child of ['src','test']) {
    const target=join(root,`raced-${child}`),outside=join(root,`outside-${child}`);await mkdir(outside);
    await writeFile(join(outside,'sentinel'),'keep');
    let readyResolve,resultResolve,rejectReady,rejectResult;
    const ready=new Promise((resolve,reject)=>{readyResolve=resolve;rejectReady=reject;});
    const result=new Promise((resolve,reject)=>{resultResolve=resolve;rejectResult=reject;});
    const worker=new Worker(`
      const {parentPort,workerData}=require('node:worker_threads');
      const fs=require('node:fs');const target=workerData.target+'/'+workerData.child;
      parentPort.postMessage({ready:true});const deadline=Date.now()+5000;let swapped=false;
      while(Date.now()<deadline){try{fs.rmdirSync(target);fs.symlinkSync(workerData.outside,target,'dir');swapped=true;break;}
        catch(error){if(!['ENOENT','ENOTEMPTY'].includes(error.code))throw error;}}
      parentPort.postMessage({swapped});
    `,{eval:true,execArgv:[],workerData:{target,outside,child}});
    worker.on('message',message=>message.ready?readyResolve():resultResolve(message));
    worker.on('error',error=>{rejectReady(error);rejectResult(error);});
    try {
      await ready;
      const scaffold=scaffoldSatellite({directory:target,name:'raced-adapter'}).then(value=>({ok:true,value}),error=>({ok:false,error}));
      const attack=await result,outcome=await scaffold;
      assert.equal(attack.swapped,true,'the worker actually replaced the created child directory');
      assert.equal(outcome.ok,false,'namespace substitution must fail scaffold acknowledgement');
      assert.deepEqual((await readdir(outside)).sort(),['sentinel']);
      assert.equal(await readFile(join(outside,'sentinel'),'utf8'),'keep');
    } finally {await worker.terminate();}
  }
});
test('scaffolding fails before creating output when directory-descriptor support is unavailable',async t=>{
  const root=await temp(t),directory=join(root,'unsupported');
  const source=new URL('../src/index.js',import.meta.url).href;
  const program=`Object.defineProperty(process,'platform',{value:'unsupported'}); const {readCatalog,scaffoldSatellite}=await import(${JSON.stringify(source)});await readCatalog(); try{await scaffoldSatellite({directory:${JSON.stringify(directory)},name:'adapter'});process.exitCode=2;}catch(error){if(!/requires Linux/.test(error.message))throw error;}`;
  const output=spawnSync(process.execPath,['--input-type=module','-e',program],{encoding:'utf8'});
  assert.equal(output.status,0,output.stderr);assert.deepEqual(await readdir(root),[]);
});
