#!/usr/bin/env node
// One canonical pack, two installed-runtime qualifications, protected publication, public-byte proof.
import { execFileSync } from 'node:child_process';
import { readFile,writeFile,readdir,mkdir,mkdtemp,rm,copyFile,lstat } from 'node:fs/promises';
import { join,resolve,dirname,basename } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import {isDeepStrictEqual} from 'node:util';
import { schema,releaseTag,packageSpecs,sha256,integrity,tarballURL,safeFilename,assertSourceSha,topologicalOrder,validateRows,validateCoreReceipt,validateQualification,validateToolkitProof,publicLock } from './toolkit-release-lib.mjs';
import {verifyToolkitInstallation} from '../packages/wsp-toolkit/scripts/packed-consumer.mjs';
const root=resolve(fileURLToPath(new URL('..',import.meta.url))),toolkit=join(root,'packages/wsp-toolkit');
const args=process.argv.slice(2),mode=args.shift(),options=Object.create(null);
for(let i=0;i<args.length;i+=2){if(!/^--[a-z-]+$/.test(args[i])||!args[i+1]||Object.hasOwn(options,args[i].slice(2)))throw Error('Invalid argument');options[args[i].slice(2)]=args[i+1];}
const need=name=>{if(!options[name])throw Error(`Missing --${name}`);return options[name];};
const run=(command,argv,cwd=root,env=process.env)=>execFileSync(command,argv,{cwd,env,encoding:'utf8',stdio:['ignore','pipe','inherit'],maxBuffer:32*1024*1024,timeout:40*60_000}).trim();
const json=async path=>JSON.parse(await readFile(path,'utf8'));
const save=async(path,data)=>{await mkdir(dirname(path),{recursive:true});await writeFile(path,JSON.stringify(data,null,2)+'\n',{flag:'wx'});};
const gitSha=()=>run('git',['rev-parse','HEAD']);
const clean=()=>{if(run('git',['status','--porcelain']))throw Error('Release source must be clean');};
const source=()=>assertSourceSha(gitSha());
const cleanEnv=()=>Object.fromEntries(Object.entries(process.env).filter(([key])=>!/^(?:NPM_TOKEN|NODE_AUTH_TOKEN|GH_TOKEN|GITHUB_TOKEN|CARGO_REGISTRY_TOKEN|ACTIONS_ID_TOKEN_REQUEST_TOKEN|ACTIONS_ID_TOKEN_REQUEST_URL|npm_config_.*auth.*)$/i.test(key)));
async function isolatedEnv(target){
  const env=cleanEnv(),config=join(target,'consumer.npmrc'),globalConfig=join(target,'global.npmrc');
  await writeFile(config,'registry=https://registry.npmjs.org\n');await writeFile(globalConfig,'');
  for(const key of Object.keys(env))if(/^npm_config_(userconfig|globalconfig|cache|registry)$/i.test(key))delete env[key];
  return {...env,NPM_CONFIG_USERCONFIG:config,NPM_CONFIG_GLOBALCONFIG:globalConfig,NPM_CONFIG_REGISTRY:'https://registry.npmjs.org'};
}
async function metadata(path){
  const names=run('tar',['-tzf',path]).split('\n');
  if(names.length>2000||new Set(names).size!==names.length||names.some(name=>!name.startsWith('package/')||name.includes('..')||name.includes('\\')||name.startsWith('/')||name.endsWith('/')))throw Error('Unsafe or duplicate package archive paths');
  if(run('tar',['-tvzf',path]).split('\n').some(line=>!/^[-d]/.test(line)))throw Error('Archive links and special files forbidden');
  return JSON.parse(run('tar',['-xOzf',path,'package/package.json']));
}
function assertTrackedPayload(spec,relative,bytes){
  if(typeof relative!=='string'||!relative||relative.startsWith('/')||relative.includes('..')||relative.includes('\\'))throw Error('Unsafe source payload path');
  const path=`${spec.directory}/${relative}`;
  let tracked;try{tracked=execFileSync('git',['show',`HEAD:${path}`],{cwd:root,stdio:['ignore','pipe','pipe'],maxBuffer:20*1024*1024});}catch{throw Error(`Package payload is not tracked at source HEAD: ${path}`);}
  if(relative==='package.json'){
    // npm can normalize harmless JSON formatting; the entire parsed manifest must still match HEAD.
    if(!isDeepStrictEqual(JSON.parse(bytes),JSON.parse(tracked)))throw Error(`Packed package manifest differs from tracked HEAD: ${path}`);
  }else if(!tracked.equals(bytes))throw Error(`Package payload bytes differ from tracked HEAD: ${path}`);
}
async function preflightPackage(spec){
  const listing=JSON.parse(run('npm',['pack','--dry-run','--json','--ignore-scripts'],join(root,spec.directory),cleanEnv()));
  if(listing.length!==1||!listing[0].files?.length)throw Error('Package preflight returned no files');
  const names=new Set();for(const file of listing[0].files){if(names.has(file.path))throw Error('Duplicate package preflight path');names.add(file.path);const path=join(root,spec.directory,file.path);if(!(await lstat(path)).isFile())throw Error('Source payload must be a regular tracked file');assertTrackedPayload(spec,file.path,await readFile(path));}
}
async function preflight(){clean();const spec=packageSpecs.find(row=>row.name===need('package'));if(!spec)throw Error('Unknown package');await preflightPackage(spec);}
async function inspectRows(directory,rows){
  validateRows(rows);
  for(const row of rows){
    const path=join(directory,safeFilename(row.tarball)),bytes=await readFile(path),pkg=await metadata(path);
    if(sha256(bytes)!==row.sha256||integrity(bytes)!==row.integrity||pkg.name!==row.name||pkg.version!==row.version||pkg.private||JSON.stringify(pkg.dependencies??{})!==JSON.stringify(row.dependencies))throw Error(`Archive identity, dependency or digest mismatch: ${row.name}`);
    const spec=packageSpecs.find(spec=>spec.name===row.name),tracked=await json(join(root,spec.directory,'package.json'));
    if(pkg.name!==tracked.name||pkg.version!==tracked.version||JSON.stringify(pkg.dependencies??{})!==JSON.stringify(tracked.dependencies??{}))throw Error('Archive differs from source package identity/dependencies');
    for(const filename of run('tar',['-tzf',path]).split('\n')){
      const bytes=execFileSync('tar',['-xOzf',path,filename],{maxBuffer:20*1024*1024});
      assertTrackedPayload(spec,filename.slice('package/'.length),bytes);
    }
  }
}
async function pack(){
  clean();const directory=resolve(need('directory'));await mkdir(directory,{recursive:true});
  if((await readdir(directory)).length)throw Error('Canonical pack directory must be empty');
  // Check the complete pack list before creating any archive. git status alone ignores ignored code.
  for(const spec of packageSpecs)await preflightPackage(spec);
  const rows=[];
  for(const spec of packageSpecs){
    const pkg=await json(join(root,spec.directory,'package.json'));if(pkg.name!==spec.name||pkg.version!==spec.version||pkg.private)throw Error(`Unexpected source package: ${spec.name}`);
    const result=JSON.parse(run('npm',['pack','--json','--ignore-scripts','--pack-destination',directory],join(root,spec.directory),cleanEnv()));
    if(result.length!==1)throw Error('Pack must emit exactly one archive');
    const tarball=safeFilename(result[0].filename),bytes=await readFile(join(directory,tarball));
    rows.push({name:pkg.name,version:pkg.version,tarball,sha256:sha256(bytes),integrity:integrity(bytes),dependencies:pkg.dependencies??{}});
  }
  const manifest={schema:`${schema}.candidate/1`,sourceSha:source(),sourceTree:run('git',['rev-parse','HEAD^{tree}']),releaseTag,packages:topologicalOrder(rows)};
  await inspectRows(directory,manifest.packages);await save(join(directory,'package-manifest.json'),manifest);
}
async function candidate(directory){
  const manifest=await json(join(directory,'package-manifest.json'));
  if(manifest.schema!==`${schema}.candidate/1`||manifest.sourceSha!==source()||manifest.sourceTree!==run('git',['rev-parse','HEAD^{tree}'])||manifest.releaseTag!==releaseTag)throw Error('Candidate source/identity mismatch');
  await inspectRows(directory,manifest.packages);return manifest;
}
async function installCandidate(directory,target,env){
  const manifest=await candidate(directory);
  await writeFile(join(target,'package.json'),JSON.stringify({name:'toolkit-release-consumer',version:'1.0.0',private:true,type:'module'}));
  run('npm',['install','--ignore-scripts','--no-audit','--no-fund','--cache',join(target,'npm-cache'),...manifest.packages.map(row=>join(directory,row.tarball))],target,env);
  for(const row of manifest.packages){const pkg=await json(join(target,'node_modules',row.name,'package.json'));if(pkg.version!==row.version)throw Error('Wrong installed candidate version');}
  return manifest;
}
async function coreConsumer(target,output,sourceSha,env){
  run(process.execPath,[join(root,'packages/core/test/installed-consumer.mjs'),target,output],root,{...env,EXPECTED_SHA:sourceSha,RUV_FANN_PLAYWRIGHT:join(toolkit,'node_modules/playwright'),RUV_FANN_TSC:join(toolkit,'node_modules/typescript/bin/tsc')});
  const report=await json(output);validateCoreReceipt(report,sourceSha);return report;
}
async function qualify(){
  clean();const directory=resolve(need('directory')),manifest=await candidate(directory),major=Number(process.versions.node.split('.')[0]);
  if(![22,24].includes(major))throw Error('Qualification requires Node22 or24');
  const env={...cleanEnv(),EXPECTED_SHA:manifest.sourceSha,WSP_TOOLKIT_CANDIDATE:join(directory,'package-manifest.json')};
  run(process.execPath,['scripts/validate.mjs'],toolkit,env);
  const validation=await json(join(toolkit,'.artifacts/validation.json'));
  const target=await mkdtemp(join(tmpdir(),'toolkit-exact-consumer-'));
  try{
    const consumerEnv={...await isolatedEnv(target),EXPECTED_SHA:manifest.sourceSha};
    await installCandidate(directory,target,consumerEnv);
    const lock=publicLock(await json(join(target,'package-lock.json')),manifest.packages),lockPath=join(directory,'public-package-lock.json');
    if(major===22)await save(lockPath,lock);
    else {
      // Node24 must accept the Node22 resolution as well, rather than silently selecting a different dependency tree.
      const sealedLock=await json(lockPath);await writeFile(join(target,'package-lock.json'),JSON.stringify(sealedLock));
      // Candidate package entries continue to use the canonical archives until public publication.
      const local=structuredClone(sealedLock);for(const row of manifest.packages)local.packages[`node_modules/${row.name}`].resolved=`file:${join(directory,row.tarball)}`;
      await writeFile(join(target,'package-lock.json'),JSON.stringify(local));
      await writeFile(join(target,'package.json'),JSON.stringify({name:'toolkit-release-consumer',version:'1.0.0',private:true,type:'module',dependencies:sealedLock.packages[''].dependencies}));
      run('npm',['ci','--ignore-scripts','--no-audit','--no-fund','--cache',join(target,'ci-cache')],target,consumerEnv);
    }
    const core=await coreConsumer(target,join(target,'core-receipt.json'),manifest.sourceSha,consumerEnv);
    const toolkitProof=await verifyToolkitInstallation(target,{env:consumerEnv});validateToolkitProof(toolkitProof);
    const receipt={schema:`${schema}.qualification/1`,status:'passed',sourceSha:manifest.sourceSha,manifestSha256:sha256(await readFile(join(directory,'package-manifest.json'))),runtime:process.version,validation,core,toolkit:toolkitProof,publicLockSha256:sha256(await readFile(lockPath))};
    validateQualification(receipt,manifest.sourceSha,receipt.manifestSha256);
    await save(join(directory,`node-${major}-qualification.json`),receipt);
  }finally{await rm(target,{recursive:true,force:true});}
  clean();
}
async function seal(){
  clean();const directory=resolve(need('directory')),manifest=await candidate(directory),manifestHash=sha256(await readFile(join(directory,'package-manifest.json'))),lockHash=sha256(await readFile(join(directory,'public-package-lock.json')));
  if(assertSourceSha(need('expected-sha'))!==manifest.sourceSha)throw Error('Seal requested wrong source');
  for(const major of [22,24]){
    const receipt=await json(join(directory,`node-${major}-qualification.json`));validateQualification(receipt,manifest.sourceSha,manifestHash);
    if(Number(receipt.runtime.split('.')[0].slice(1))!==major||receipt.publicLockSha256!==lockHash)throw Error('Qualification runtime/lock binding mismatch');
  }
  const files=[];for(const filename of ['package-manifest.json','public-package-lock.json','node-22-qualification.json','node-24-qualification.json',...manifest.packages.map(row=>row.tarball)])files.push({filename,sha256:sha256(await readFile(join(directory,filename)))});
  await save(join(directory,'candidate-seal.json'),{schema:`${schema}.seal/1`,status:'passed',sourceSha:manifest.sourceSha,sourceTree:manifest.sourceTree,releaseTag,files});
  await verify(directory);
}
async function verify(directory){
  const seal=await json(join(directory,'candidate-seal.json'));
  if(seal.schema!==`${schema}.seal/1`||seal.status!=='passed'||seal.sourceSha!==source()||seal.releaseTag!==releaseTag||seal.sourceTree!==run('git',['rev-parse','HEAD^{tree}']))throw Error('Seal source/generation mismatch');
  const manifest=await candidate(directory),expected=['package-manifest.json','public-package-lock.json','node-22-qualification.json','node-24-qualification.json',...manifest.packages.map(row=>row.tarball)];
  const actual=await readdir(directory);
  if(JSON.stringify(actual.sort())!==JSON.stringify([...expected,'candidate-seal.json'].sort()))throw Error('Candidate directory contains unsealed files');
  for(const filename of actual)if(!(await lstat(join(directory,filename))).isFile())throw Error('Candidate contains links or non-files');
  if(!Array.isArray(seal.files)||seal.files.length!==expected.length||new Set(seal.files.map(row=>row.filename)).size!==expected.length)throw Error('Seal file set incomplete');
  for(const filename of expected){const row=seal.files.find(row=>row.filename===filename);if(!row||row.sha256!==sha256(await readFile(join(directory,safeFilename(filename)))))throw Error(`Sealed bytes changed: ${filename}`);}
  const manifestHash=sha256(await readFile(join(directory,'package-manifest.json'))),lockHash=sha256(await readFile(join(directory,'public-package-lock.json')));
  const lock=await json(join(directory,'public-package-lock.json'));publicLock(lock,manifest.packages);
  for(const row of manifest.packages)if(lock.packages[`node_modules/${row.name}`]?.integrity!==row.integrity||lock.packages[`node_modules/${row.name}`]?.resolved!==tarballURL(row))throw Error('Sealed public lock SRI mismatch');
  for(const major of [22,24]){const receipt=await json(join(directory,`node-${major}-qualification.json`));validateQualification(receipt,source(),manifestHash);if(receipt.publicLockSha256!==lockHash||Number(receipt.runtime.split('.')[0].slice(1))!==major)throw Error('Sealed runtime/lock receipt mismatch');}
  return {seal,manifest};
}
function protectedContext(sha){
  const expected={GITHUB_ACTIONS:'true',GITHUB_REPOSITORY:'ruvnet/ruv-FANN',GITHUB_REF:'refs/heads/main',GITHUB_SHA:sha,GITHUB_WORKFLOW:'protected-toolkit-release',GITHUB_WORKFLOW_REF:'ruvnet/ruv-FANN/.github/workflows/protected-toolkit-release.yml@refs/heads/main',GITHUB_EVENT_NAME:'workflow_dispatch',GITHUB_JOB:'publish'};
  for(const [key,value]of Object.entries(expected))if(process.env[key]!==value)throw Error('Publication requires exact upstream protected main workflow');
  if(source()!==sha||run('gh',['api','repos/ruvnet/ruv-FANN/commits/main','--jq','.sha'])!==sha)throw Error('Main advanced beyond sealed source');
  if(run('gh',['api','repos/ruvnet/ruv-FANN/issues?labels=release-blocker&state=open&per_page=1','--jq','length'])!=='0')throw Error('Release blockers prevent publication');
  clean();
}
async function publicBytes(row){
  const response=await fetch(tarballURL(row),{redirect:'error',signal:AbortSignal.timeout(30_000)});
  if(response.status===404)return null;if(!response.ok)throw Error(`Registry HTTP${response.status}`);
  const maximum=20*1024*1024;if(Number(response.headers.get('content-length'))>maximum){await response.body?.cancel();throw Error('Oversized registry tarball');}
  const chunks=[];let size=0;for await(const chunk of response.body){size+=chunk.length;if(size>maximum)throw Error('Oversized registry tarball');chunks.push(Buffer.from(chunk));}return Buffer.concat(chunks);
}
async function awaitPublic(row,timeout=180_000){const deadline=Date.now()+timeout;while(Date.now()<deadline){const bytes=await publicBytes(row);if(bytes){if(sha256(bytes)!==row.sha256||integrity(bytes)!==row.integrity)throw Error(`Public bytes differ: ${row.name}@${row.version}`);return bytes;}await new Promise(resolve=>setTimeout(resolve,5000));}throw Error(`Registry propagation timeout: ${row.name}`);}
async function publish(){
  const directory=resolve(need('directory')),{seal,manifest}=await verify(directory);protectedContext(seal.sourceSha);
  if(!process.env.NODE_AUTH_TOKEN)throw Error('Explicit protected NPM_TOKEN required; OIDC-only registration has not been verified');
  // Preflight every name before making any upload, so known collisions do not leave a partial release.
  for(const row of manifest.packages){const bytes=await publicBytes(row);if(bytes&&sha256(bytes)!==row.sha256)throw Error(`Existing version has different bytes: ${row.name}`);}
  const results=[];
  for(const row of manifest.packages){
    protectedContext(seal.sourceSha);const previous=await publicBytes(row);
    if(!previous){try{run('npm',['publish',join(directory,row.tarball),'--ignore-scripts','--access','public','--provenance','--registry=https://registry.npmjs.org']);}catch(error){try{await awaitPublic(row);}catch{throw error;}}}
    await awaitPublic(row);results.push({name:row.name,version:row.version,sha256:row.sha256,integrity:row.integrity,status:previous?'identical-existing-version':'published-and-byte-verified'});
    // Append-only progress survives a later package failure.
    await save(join(resolve(need('output')),`${row.name}.json`),{schema:`${schema}.package-publication/1`,sourceSha:seal.sourceSha,...results.at(-1)});
  }
  await save(join(resolve(need('output')),'publication.json'),{schema:`${schema}.publication/1`,status:'passed',sourceSha:seal.sourceSha,releaseTag,sealSha256:sha256(await readFile(join(directory,'candidate-seal.json'))),packages:results.map(row=>({...row,status:'public-bytes-verified'}))});
}
async function publicConsumer(){
  const directory=resolve(need('directory')),{seal,manifest}=await verify(directory),publication=await json(need('publication'));
  if(publication.status!=='passed'||publication.sourceSha!==seal.sourceSha||publication.sealSha256!==sha256(await readFile(join(directory,'candidate-seal.json')))||publication.packages.length!==manifest.packages.length)throw Error('Publication receipt does not match seal');
  for(const row of manifest.packages)if(!publication.packages.some(item=>item.name===row.name&&item.version===row.version&&item.sha256===row.sha256&&item.integrity===row.integrity))throw Error('Publication package digest mismatch');
  for(const row of manifest.packages)await awaitPublic(row);
  const target=await mkdtemp(join(tmpdir(),'toolkit-public-consumer-')),env=await isolatedEnv(target);
  try{
    const lock=await json(join(directory,'public-package-lock.json'));
    await copyFile(join(directory,'public-package-lock.json'),join(target,'package-lock.json'));
    await writeFile(join(target,'package.json'),JSON.stringify({name:lock.name,version:lock.version,private:true,type:'module',dependencies:lock.packages[''].dependencies}));
    run('npm',['ci','--ignore-scripts','--no-audit','--no-fund','--registry=https://registry.npmjs.org','--cache',join(target,'fresh-cache')],target,env);
    for(const row of manifest.packages)if((await json(join(target,'node_modules',row.name,'package.json'))).version!==row.version)throw Error('Wrong public installed version');
    const core=await coreConsumer(target,join(target,'core-public.json'),seal.sourceSha,env);
    const toolkitProof=await verifyToolkitInstallation(target,{env});validateToolkitProof(toolkitProof);
    // Public acceptance is repeatable across recovery runs. Timing observations remain
    // in the sealed qualification reports; this receipt records behavior and work.
    delete toolkitProof.evolution.report.budget.elapsedMs;
    await save(need('output'),{schema:`${schema}.public-consumer/1`,status:'passed',sourceSha:seal.sourceSha,releaseTag,sealSha256:sha256(await readFile(join(directory,'candidate-seal.json'))),publicLockSha256:sha256(await readFile(join(directory,'public-package-lock.json'))),packages:manifest.packages.map(({name,version,sha256,integrity})=>({name,version,sha256,integrity})),core,toolkit:toolkitProof});
  }finally{await rm(target,{recursive:true,force:true});}
}
function tagCommit(tag){let item=JSON.parse(run('gh',['api',`repos/ruvnet/ruv-FANN/git/ref/tags/${tag}`])).object;for(let i=0;i<5;i++){if(item.type==='commit')return item.sha;if(item.type!=='tag')throw Error('Tag does not resolve to commit');item=JSON.parse(run('gh',['api',`repos/ruvnet/ruv-FANN/git/tags/${item.sha}`])).object;}throw Error('Tag nesting exceeds limit');}
async function githubRelease(){
  const directory=resolve(need('directory')),{seal,manifest}=await verify(directory);protectedContext(seal.sourceSha);
  const publication=await json(need('publication')),consumer=await json(need('public-consumer'));
  if(publication.status!=='passed'||consumer.status!=='passed'||publication.sourceSha!==seal.sourceSha||consumer.sourceSha!==seal.sourceSha||consumer.sealSha256!==sha256(await readFile(join(directory,'candidate-seal.json')))||publication.sealSha256!==consumer.sealSha256)throw Error('Public release evidence mismatch');
  for(const row of manifest.packages)for(const receipt of [publication,consumer])if(receipt.packages?.length!==manifest.packages.length||!receipt.packages.some(item=>item.name===row.name&&item.version===row.version&&item.sha256===row.sha256&&item.integrity===row.integrity))throw Error('Release public package evidence differs from seal');
  validateCoreReceipt(consumer.core,seal.sourceSha);validateToolkitProof(consumer.toolkit);for(const row of manifest.packages)await awaitPublic(row);
  const receipt={schema:`${schema}.release/1`,status:'passed',sourceSha:seal.sourceSha,releaseTag,sealSha256:sha256(await readFile(join(directory,'candidate-seal.json'))),publicationSha256:sha256(await readFile(need('publication'))),publicConsumerSha256:sha256(await readFile(need('public-consumer'))),packages:manifest.packages};
  const paths=[...seal.files.map(row=>join(directory,row.filename)),join(directory,'candidate-seal.json'),resolve(need('publication')),resolve(need('public-consumer'))];
  const receiptName=safeFilename(basename(need('output'))),receiptBytes=Buffer.from(JSON.stringify(receipt,null,2)+'\n');
  const expected=new Map(await Promise.all(paths.map(async path=>[basename(path),await readFile(path)])));expected.set(receiptName,receiptBytes);
  let existing;try{existing=JSON.parse(run('gh',['release','view',releaseTag,'--json','assets']));}catch{
    run('gh',['release','create',releaseTag,...paths,'--target',seal.sourceSha,'--title','ruv-fann core 0.2.0 and WSP toolkit 0.1.0','--notes',`Thirteen exact npm archives from source ${seal.sourceSha}. Core subpaths expose bounded WSP lifecycle, search, invariants, workers, registry and constellation capabilities. Node22/24 installed consumers, real Chromium/WASM, strict types, public dependency lock and exact-byte receipts attached. No generalization or SOTA claim.`]);
    existing=JSON.parse(run('gh',['release','view',releaseTag,'--json','assets']));
  }
  if(tagCommit(releaseTag)!==seal.sourceSha)throw Error('GitHub tag source mismatch');
  const target=await mkdtemp(join(tmpdir(),'toolkit-github-public-'));
  try{
    async function verifyExisting(assets,requireAll=false){
      const names=assets.map(row=>row.name);
      if(new Set(names).size!==names.length||names.some(name=>!expected.has(name))||requireAll&&(names.length!==expected.size))throw Error('GitHub release asset set differs');
      const download=await mkdtemp(join(target,'download-'));
      if(names.length)run('gh',['release','download',releaseTag,'--dir',download]);
      for(const name of names)if(!expected.get(name).equals(await readFile(join(download,name))))throw Error(`Existing GitHub asset bytes differ: ${name}`);
    }
    // Resume a partial asset set only after every existing asset and tag passes. Never clobber.
    await verifyExisting(existing.assets);
    for(const path of paths)if(!existing.assets.some(row=>row.name===basename(path)))run('gh',['release','upload',releaseTag,path]);
    existing=JSON.parse(run('gh',['release','view',releaseTag,'--json','assets']));
    for(const path of paths)if(!existing.assets.some(row=>row.name===basename(path)))throw Error('GitHub base asset upload incomplete');
    await verifyExisting(existing.assets);
    // The passed receipt is staged only after the public base assets have been downloaded and verified.
    const staged=join(target,receiptName);await writeFile(staged,receiptBytes,{flag:'wx'});
    if(!existing.assets.some(row=>row.name===receiptName))run('gh',['release','upload',releaseTag,staged]);
    const final=JSON.parse(run('gh',['release','view',releaseTag,'--json','assets']));
    await verifyExisting(final.assets,true);
    // No successful receipt is exposed until the complete public asset set, including this receipt, matches.
    await save(need('output'),receipt);
  }finally{await rm(target,{recursive:true,force:true});}
}
const modes={preflight,pack,qualify,seal,verify:()=>verify(resolve(need('directory'))),publish,'public-consumer':publicConsumer,'github-release':githubRelease};
if(!modes[mode])throw Error('Unsupported toolkit release mode');await modes[mode]();
