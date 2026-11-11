#!/usr/bin/env node
// Build once from public archives; qualify, seal and publish exactly those bytes.
import {execFileSync} from 'node:child_process';
import {readFile,writeFile,readdir,mkdir,mkdtemp,rm,lstat,rename} from 'node:fs/promises';
import {join,resolve,dirname,basename} from 'node:path';
import {tmpdir} from 'node:os';
import {fileURLToPath} from 'node:url';
import {isDeepStrictEqual as equal} from 'node:util';
import {schema,assert,sha256,integrity,filename,sourceSha,tarballURL,archiveName,validateConfig,readArchive,writeArchive,validatePayload,validateConsumer,publicLock} from './metadata-release-lib.mjs';
import {readPublicArchive} from './metadata-registry.mjs';

const root=resolve(fileURLToPath(new URL('..',import.meta.url))),args=process.argv.slice(2),mode=args.shift(),options=Object.create(null);
for(let i=0;i<args.length;i+=2){assert(/^--[a-z-]+$/.test(args[i])&&args[i+1]&&!Object.hasOwn(options,args[i].slice(2)),'Invalid arguments');options[args[i].slice(2)]=args[i+1];}
const need=key=>{assert(options[key],`Missing --${key}`);return options[key];};
const run=(bin,argv,cwd=root,env=process.env)=>execFileSync(bin,argv,{cwd,env,encoding:'utf8',stdio:['ignore','pipe','inherit'],timeout:40*60_000,maxBuffer:40*1024*1024}).trim();
const json=async path=>JSON.parse(await readFile(path,'utf8'));
const save=async(path,value)=>{await mkdir(dirname(path),{recursive:true});await writeFile(path,JSON.stringify(value,null,2)+'\n',{flag:'wx'});};
const head=()=>sourceSha(run('git',['rev-parse','HEAD']));
const tree=()=>run('git',['rev-parse','HEAD^{tree}']);
const clean=()=>assert(run('git',['status','--porcelain'])==='','Release source must be clean');
function tracked(path){
  const record=run('git',['ls-files','--stage','--',path]);assert(/^100(?:644|755) /.test(record)&&record.split('\n').length===1,`Source must be a tracked regular file: ${path}`);
  return execFileSync('git',['show',`HEAD:${path}`],{cwd:root,maxBuffer:40*1024*1024,stdio:['ignore','pipe','pipe']});
}
async function config(){const path='metadata-release.json',bytes=tracked(path);assert(bytes.equals(await readFile(join(root,path))),'Config differs from HEAD');tracked('scripts/metadata-consumer.mjs');return validateConfig(JSON.parse(bytes));}
const configHash=()=>sha256(tracked('metadata-release.json'));
function cleanEnv(){return Object.fromEntries(Object.entries(process.env).filter(([key])=>!/token|secret|password|^npm_config_/i.test(key)));}
async function isolatedEnv(target){
  const user=join(target,'consumer.npmrc'),global=join(target,'global.npmrc');await writeFile(user,'registry=https://registry.npmjs.org\n');await writeFile(global,'');
  return {...cleanEnv(),NPM_CONFIG_USERCONFIG:user,NPM_CONFIG_GLOBALCONFIG:global,NPM_CONFIG_REGISTRY:'https://registry.npmjs.org'};
}
async function overlay(spec){const result={};for(const name of ['package.json',...spec.overlays]){const path=spec.directory==='.'?name:`${spec.directory}/${name}`,bytes=tracked(path);assert((await lstat(join(root,path))).isFile()&&bytes.equals(await readFile(join(root,path))),`Source overlay differs from HEAD: ${path}`);result[name]=bytes;}return result;}
const hashes=files=>[...files].map(([path,item])=>({path,mode:item.mode,sha256:sha256(item.bytes)})).sort((a,b)=>a.path<b.path?-1:a.path>b.path?1:0);
async function empty(path){await mkdir(path,{recursive:true});assert((await readdir(path)).length===0,'Output directory must be empty');}
async function bootstrap(){clean();const cfg=await config();for(const [bin,...argv]of cfg.bootstrap)run(bin,argv,root,cleanEnv());clean();}
async function pack(){
  clean();const cfg=await config(),directory=resolve(need('directory'));await empty(directory);const packages=[];
  for(const spec of cfg.packages){
    const baseline={name:spec.name,version:spec.fromVersion},response=await fetch(`https://registry.npmjs.org/${encodeURIComponent(spec.name)}/${spec.fromVersion}`,{redirect:'error',signal:AbortSignal.timeout(30_000)});
    assert(response.ok,`Published baseline metadata unavailable: ${spec.name}`);const metadata=await response.json();
    assert(metadata.name===spec.name&&metadata.version===spec.fromVersion&&metadata.dist?.tarball===tarballURL(baseline)&&/^sha512-[A-Za-z0-9+/]{86}==$/.test(metadata.dist.integrity),'Baseline registry identity/SRI missing');
    const bytes=await readPublicArchive(baseline);assert(bytes&&integrity(bytes)===metadata.dist.integrity,'Published baseline SRI mismatch');
    const before=readArchive(bytes),overlays=await overlay(spec),after=new Map([...before].map(([name,item])=>[name,{...item,bytes:overlays[name.slice(8)]??item.bytes}]));
    validatePayload(before,after,spec,overlays);const packed=writeArchive(after),tarball=archiveName(spec.name,spec.toVersion),baselineTarball=`baseline-${archiveName(spec.name,spec.fromVersion)}`;
    // This writer runs once. Every later stage validates and retains these canonical bytes.
    await writeFile(join(directory,baselineTarball),bytes,{flag:'wx'});await writeFile(join(directory,tarball),packed,{flag:'wx'});
    const pkg=JSON.parse(overlays['package.json']);packages.push({name:spec.name,version:spec.toVersion,tarball,sha256:sha256(packed),integrity:integrity(packed),dependencies:pkg.dependencies??{},baseline:{version:spec.fromVersion,tarball:baselineTarball,sha256:sha256(bytes),integrity:integrity(bytes)},overlays:Object.fromEntries(Object.entries(overlays).map(([name,bytes])=>[name,sha256(bytes)])),files:hashes(after)});
  }
  await save(join(directory,'package-manifest.json'),{schema,kind:'candidate',sourceSha:head(),sourceTree:tree(),repository:cfg.repository,releaseTag:cfg.releaseTag,configSha256:configHash(),packages});
  await candidate(directory);clean();
}
async function candidate(directory){
  const cfg=await config(),manifest=await json(join(directory,'package-manifest.json'));
  assert(manifest.schema===schema&&manifest.kind==='candidate'&&manifest.sourceSha===head()&&manifest.sourceTree===tree()&&manifest.repository===cfg.repository&&manifest.releaseTag===cfg.releaseTag&&manifest.configSha256===configHash(),'Candidate source/config mismatch');
  assert(Array.isArray(manifest.packages)&&manifest.packages.length===cfg.packages.length,'Candidate package count mismatch');
  for(let i=0;i<cfg.packages.length;i++){
    const spec=cfg.packages[i],row=manifest.packages[i];assert(row.name===spec.name&&row.version===spec.toVersion&&row.tarball===archiveName(spec.name,spec.toVersion)&&row.baseline?.version===spec.fromVersion&&row.baseline.tarball===`baseline-${archiveName(spec.name,spec.fromVersion)}`,'Candidate package/baseline identity mismatch');
    const bytes=await readFile(join(directory,filename(row.tarball))),baseline=await readFile(join(directory,filename(row.baseline.tarball)));
    assert(sha256(bytes)===row.sha256&&integrity(bytes)===row.integrity&&sha256(baseline)===row.baseline.sha256&&integrity(baseline)===row.baseline.integrity,'Candidate/baseline bytes changed');
    const before=readArchive(baseline),after=readArchive(bytes),overlays=await overlay(spec);validatePayload(before,after,spec,overlays);
    assert(equal(row.files,hashes(after))&&equal(row.overlays,Object.fromEntries(Object.entries(overlays).map(([name,bytes])=>[name,sha256(bytes)])))&&equal(row.dependencies,JSON.parse(overlays['package.json']).dependencies??{}),'Candidate file/overlay/dependency manifest changed');
  }
  return manifest;
}
async function validateInstallation(target,directory,manifest){
  for(const row of manifest.packages){
    const installed=join(target,'node_modules',row.name),files=readArchive(await readFile(join(directory,row.tarball)));
    async function walk(path,relative=''){const result=[];for(const item of await readdir(path,{withFileTypes:true})){if(!relative&&item.name==='node_modules')continue;const name=relative?`${relative}/${item.name}`:item.name;if(item.isDirectory())result.push(...await walk(join(path,item.name),name));else{assert(item.isFile(),'Installed package contains non-regular file');result.push(name);}}return result;}
    assert(equal((await walk(installed)).sort(),[...files.keys()].map(name=>name.slice(8)).sort()),'Installed candidate file set differs');
    for(const [path,file]of files)assert(file.bytes.equals(await readFile(join(installed,path.slice(8)))),`Installed candidate bytes differ: ${row.name}/${path}`);
  }
}
async function consumer(target,sha,env){
  const stdout=run(process.execPath,[join(root,'scripts/metadata-consumer.mjs'),resolve(target),sha],root,{...env,EXPECTED_SHA:sha});
  const receipt=JSON.parse(stdout.split('\n').at(-1));return validateConsumer(receipt,sha);
}
async function install(target,directory,manifest,major,env,publicInstall=false){
  let lock;
  if(major===22&&!publicInstall){
    await writeFile(join(target,'package.json'),JSON.stringify({name:'metadata-release-consumer',version:'1.0.0',private:true,type:'module'}));
    run('npm',['install','--ignore-scripts','--no-audit','--no-fund','--cache',join(target,'fresh-cache'),...manifest.packages.map(row=>join(directory,row.tarball))],target,env);
    lock=publicLock(await json(join(target,'package-lock.json')),manifest.packages);await save(join(directory,'public-package-lock.json'),lock);
  }else{
    lock=await json(join(directory,'public-package-lock.json'));assert(equal(lock,publicLock(lock,manifest.packages)),'Public lock is not canonical');const local=structuredClone(lock);
    if(!publicInstall)for(const row of manifest.packages)local.packages[`node_modules/${row.name}`].resolved=`file:${join(directory,row.tarball)}`;
    await writeFile(join(target,'package.json'),JSON.stringify({name:lock.name,version:lock.version,private:true,type:'module',dependencies:lock.packages[''].dependencies}));
    await writeFile(join(target,'package-lock.json'),JSON.stringify(local));
    run('npm',['ci','--ignore-scripts','--no-audit','--no-fund','--cache',join(target,'fresh-cache')],target,env);
  }
  await validateInstallation(target,directory,manifest);return lock;
}
async function qualify(){
  clean();const directory=resolve(need('directory')),manifest=await candidate(directory),major=Number(process.versions.node.split('.')[0]);assert([22,24].includes(major),'Node22/24 required');
  const target=await mkdtemp(join(tmpdir(),'metadata-qualify-'));
  try{const env=await isolatedEnv(target);await install(target,directory,manifest,major,env);const proof=await consumer(target,manifest.sourceSha,env);
    await save(join(directory,`node-${major}-qualification.json`),{schema,kind:'qualification',status:'passed',sourceSha:head(),runtime:process.version,manifestSha256:sha256(await readFile(join(directory,'package-manifest.json'))),publicLockSha256:sha256(await readFile(join(directory,'public-package-lock.json'))),consumer:proof});
  }finally{await rm(target,{recursive:true,force:true});}clean();
}
async function qualifications(directory,manifest){
  const manifestHash=sha256(await readFile(join(directory,'package-manifest.json'))),lockHash=sha256(await readFile(join(directory,'public-package-lock.json'))),lock=await json(join(directory,'public-package-lock.json'));
  assert(equal(lock,publicLock(lock,manifest.packages)),'Public lock mismatch');
  for(const major of [22,24]){const receipt=await json(join(directory,`node-${major}-qualification.json`));assert(receipt.schema===schema&&receipt.kind==='qualification'&&receipt.status==='passed'&&receipt.sourceSha===head()&&receipt.manifestSha256===manifestHash&&receipt.publicLockSha256===lockHash&&new RegExp(`^v${major}\\.`).test(receipt.runtime),'Qualification source/runtime/digest mismatch');validateConsumer(receipt.consumer,manifest.sourceSha);}
}
const evidenceFiles=manifest=>['package-manifest.json','public-package-lock.json','node-22-qualification.json','node-24-qualification.json',...manifest.packages.flatMap(row=>[row.tarball,row.baseline.tarball])];
async function seal(){
  clean();const directory=resolve(need('directory')),manifest=await candidate(directory);assert(sourceSha(need('expected-sha'))===head(),'Seal source mismatch');await qualifications(directory,manifest);
  const files=await Promise.all(evidenceFiles(manifest).map(async name=>({filename:name,sha256:sha256(await readFile(join(directory,name)))})));
  await save(join(directory,'candidate-seal.json'),{schema,kind:'seal',status:'passed',sourceSha:head(),sourceTree:tree(),repository:manifest.repository,releaseTag:manifest.releaseTag,configSha256:configHash(),files});await verify(directory);
}
async function verify(directory){
  clean();const manifest=await candidate(directory),seal=await json(join(directory,'candidate-seal.json'));
  assert(seal.schema===schema&&seal.kind==='seal'&&seal.status==='passed'&&seal.sourceSha===head()&&seal.sourceTree===tree()&&seal.configSha256===configHash()&&seal.repository===manifest.repository&&seal.releaseTag===manifest.releaseTag,'Seal source/config mismatch');
  const expected=evidenceFiles(manifest),actual=await readdir(directory);assert(equal(actual.sort(),[...expected,'candidate-seal.json'].sort()),'Candidate contains extra/missing files');
  assert(Array.isArray(seal.files)&&seal.files.length===expected.length&&new Set(seal.files.map(row=>row.filename)).size===expected.length,'Incomplete/duplicate seal');
  for(const name of actual)assert((await lstat(join(directory,name))).isFile(),'Candidate contains non-regular files');
  for(const name of expected)assert(seal.files.find(row=>row.filename===name)?.sha256===sha256(await readFile(join(directory,name))),`Sealed file changed: ${name}`);
  await qualifications(directory,manifest);return {seal,manifest};
}
async function protectedContext(sha){
  const cfg=await config(),expected={GITHUB_ACTIONS:'true',GITHUB_REPOSITORY:cfg.repository,GITHUB_REF:'refs/heads/main',GITHUB_SHA:sha,GITHUB_WORKFLOW:'protected-metadata-release',GITHUB_WORKFLOW_REF:`${cfg.repository}/.github/workflows/protected-metadata-release.yml@refs/heads/main`,GITHUB_EVENT_NAME:'workflow_dispatch',GITHUB_JOB:'publish'};
  for(const [key,value]of Object.entries(expected))assert(process.env[key]===value,'Publication requires exact upstream protected main workflow');
  assert(head()===sha&&run('gh',['api',`repos/${cfg.repository}/commits/main`,'--jq','.sha'])===sha,'Current main differs from sealed source');
  assert(run('gh',['api',`repos/${cfg.repository}/issues?labels=release-blocker&state=open&per_page=1`,'--jq','length'])==='0','Open release blockers');clean();return cfg;
}
async function awaitPublic(row,{canonical=false}={}){
  const deadline=Date.now()+600_000;while(Date.now()<deadline){const bytes=await readPublicArchive(row,{canonical});if(bytes){assert(sha256(bytes)===row.sha256&&integrity(bytes)===row.integrity,`Public bytes differ: ${row.name}`);return bytes;}await new Promise(resolve=>setTimeout(resolve,5000));}throw Error(`Registry propagation timeout: ${row.name}`);
}
async function download(){
  const sha=sourceSha(need('expected-sha')),cfg=await protectedContext(sha),repo=cfg.repository,api=path=>JSON.parse(run('gh',['api',path]));
  const runs=api(`repos/${repo}/actions/workflows/metadata-qualification.yml/runs?head_sha=${sha}&status=completed&per_page=100`).workflow_runs.filter(row=>row.head_sha===sha&&row.head_repository?.full_name===repo&&row.conclusion==='success'&&['pull_request','workflow_dispatch','push','workflow_call'].includes(row.event)).sort((a,b)=>b.run_number-a.run_number),name=`metadata-candidate-${sha}`;
  let selected;for(const row of runs){const artifacts=api(`repos/${repo}/actions/runs/${row.id}/artifacts?per_page=100`).artifacts.filter(item=>item.name===name&&!item.expired);if(artifacts.length===1){selected=row;break;}}
  assert(selected,'No successful upstream exact-SHA qualification artifact');const target=await mkdtemp(join(tmpdir(),'metadata-download-'));
  try{run('gh',['run','download',String(selected.id),'-R',repo,'-n',name,'-D',target]);await verify(target);const output=resolve(need('output'));await empty(output);for(const name of await readdir(target))await rename(join(target,filename(name)),join(output,name));console.log(JSON.stringify({status:'verified',sourceSha:sha,runId:selected.id,artifact:name}));}finally{await rm(target,{recursive:true,force:true});}
}
async function publish(){
  const directory=resolve(need('directory')),{seal,manifest}=await verify(directory);await protectedContext(seal.sourceSha);assert(process.env.NODE_AUTH_TOKEN,'Explicit protected NPM_TOKEN secret required');const output=resolve(need('output'));await empty(output);
  for(const row of manifest.packages){const bytes=await readPublicArchive(row);assert(!bytes||sha256(bytes)===row.sha256&&integrity(bytes)===row.integrity,`Existing version differs: ${row.name}`);}
  const results=[];for(const row of manifest.packages){await protectedContext(seal.sourceSha);const previous=await readPublicArchive(row);if(!previous){try{run('npm',['publish',join(directory,row.tarball),'--ignore-scripts','--access','public','--provenance','--registry=https://registry.npmjs.org']);}catch(error){try{await awaitPublic(row);}catch{throw error;}}}await awaitPublic(row);const result={name:row.name,version:row.version,sha256:row.sha256,integrity:row.integrity,status:'public-bytes-verified'};results.push(result);await save(join(output,`${archiveName(row.name,row.version)}.json`),{schema,kind:'package-publication',sourceSha:seal.sourceSha,...result});}
  await save(join(output,'publication.json'),{schema,kind:'publication',status:'passed',sourceSha:seal.sourceSha,sealSha256:sha256(await readFile(join(directory,'candidate-seal.json'))),packages:results});
}
async function publicEvidence(path,kind,directory,seal,manifest){const receipt=await json(path);assert(receipt.schema===schema&&receipt.kind===kind&&receipt.status==='passed'&&receipt.sourceSha===seal.sourceSha&&receipt.sealSha256===sha256(await readFile(join(directory,'candidate-seal.json')))&&Array.isArray(receipt.packages)&&receipt.packages.length===manifest.packages.length,'Public evidence source/seal mismatch');for(const row of manifest.packages)assert(receipt.packages.filter(item=>item.name===row.name&&item.version===row.version&&item.sha256===row.sha256&&item.integrity===row.integrity).length===1,'Public evidence package mismatch');if(kind==='public-consumer'){assert(receipt.publicLockSha256===sha256(await readFile(join(directory,'public-package-lock.json'))),'Public consumer lock mismatch');validateConsumer(receipt.consumer,seal.sourceSha);}return receipt;}
async function publicConsumer(){
  const directory=resolve(need('directory')),{seal,manifest}=await verify(directory);await publicEvidence(need('publication'),'publication',directory,seal,manifest);
  // Fresh probes after upload; canonical URL must work before ordinary npm ci.
  await Promise.all(manifest.packages.map(row=>awaitPublic(row,{canonical:true})));const target=await mkdtemp(join(tmpdir(),'metadata-public-consumer-'));
  try{const env=await isolatedEnv(target);await install(target,directory,manifest,24,env,true);const proof=await consumer(target,seal.sourceSha,env);await save(need('output'),{schema,kind:'public-consumer',status:'passed',sourceSha:seal.sourceSha,sealSha256:sha256(await readFile(join(directory,'candidate-seal.json'))),publicLockSha256:sha256(await readFile(join(directory,'public-package-lock.json'))),packages:manifest.packages.map(({name,version,sha256,integrity})=>({name,version,sha256,integrity})),consumer:proof});}finally{await rm(target,{recursive:true,force:true});}
}
function tagCommit(repo,tag){let item=JSON.parse(run('gh',['api',`repos/${repo}/git/ref/tags/${tag}`])).object;for(let depth=0;depth<5;depth++){if(item.type==='commit')return item.sha;assert(item.type==='tag','Tag does not resolve to commit');item=JSON.parse(run('gh',['api',`repos/${repo}/git/tags/${item.sha}`])).object;}throw Error('Tag nesting too deep');}
async function githubRelease(){
  const directory=resolve(need('directory')),{seal,manifest}=await verify(directory),cfg=await protectedContext(seal.sourceSha);await publicEvidence(need('publication'),'publication',directory,seal,manifest);await publicEvidence(need('public-consumer'),'public-consumer',directory,seal,manifest);for(const row of manifest.packages)await awaitPublic(row);
  const receipt={schema,kind:'release',status:'passed',sourceSha:seal.sourceSha,releaseTag:cfg.releaseTag,sealSha256:sha256(await readFile(join(directory,'candidate-seal.json'))),publicationSha256:sha256(await readFile(need('publication'))),publicConsumerSha256:sha256(await readFile(need('public-consumer'))),packages:manifest.packages};
  const paths=[...evidenceFiles(manifest).map(name=>join(directory,name)),join(directory,'candidate-seal.json'),resolve(need('publication')),resolve(need('public-consumer'))],receiptName=filename(basename(need('output'))),receiptBytes=Buffer.from(JSON.stringify(receipt,null,2)+'\n');
  const expected=new Map(await Promise.all(paths.map(async path=>[basename(path),await readFile(path)])));assert(expected.size===paths.length&&!expected.has(receiptName),'Release asset names collide');expected.set(receiptName,receiptBytes);
  let existing;try{existing=JSON.parse(run('gh',['release','view',cfg.releaseTag,'-R',cfg.repository,'--json','assets']));}catch{await protectedContext(seal.sourceSha);run('gh',['release','create',cfg.releaseTag,...paths,'-R',cfg.repository,'--target',seal.sourceSha,'--title',cfg.releaseTag,'--notes',`Metadata-only npm release from ${seal.sourceSha}. Published runtime/binary files and dependency pins are unchanged. Exact baseline/candidate archives, Node22/24 installed consumer qualifications, sealed public lock and public byte proofs are attached.`]);existing=JSON.parse(run('gh',['release','view',cfg.releaseTag,'-R',cfg.repository,'--json','assets']));}
  assert(tagCommit(cfg.repository,cfg.releaseTag)===seal.sourceSha,'Release tag differs from sealed source');const target=await mkdtemp(join(tmpdir(),'metadata-github-'));
  try{
    const verifyAssets=async(assets,complete=false)=>{const names=assets.map(row=>row.name);assert(new Set(names).size===names.length&&names.every(name=>expected.has(name))&&(!complete||names.length===expected.size),'GitHub release asset set differs');const dir=await mkdtemp(join(target,'download-'));if(names.length)run('gh',['release','download',cfg.releaseTag,'-R',cfg.repository,'--dir',dir]);for(const name of names)assert(expected.get(name).equals(await readFile(join(dir,name))),`GitHub asset bytes differ: ${name}`);};
    await verifyAssets(existing.assets);for(const path of paths)if(!existing.assets.some(row=>row.name===basename(path)))run('gh',['release','upload',cfg.releaseTag,path,'-R',cfg.repository]);existing=JSON.parse(run('gh',['release','view',cfg.releaseTag,'-R',cfg.repository,'--json','assets']));assert(paths.every(path=>existing.assets.some(row=>row.name===basename(path))),'GitHub base asset upload incomplete');await verifyAssets(existing.assets);
    const staged=join(target,receiptName);await writeFile(staged,receiptBytes,{flag:'wx'});if(!existing.assets.some(row=>row.name===receiptName))run('gh',['release','upload',cfg.releaseTag,staged,'-R',cfg.repository]);const final=JSON.parse(run('gh',['release','view',cfg.releaseTag,'-R',cfg.repository,'--json','assets']));await verifyAssets(final.assets,true);await save(need('output'),receipt);
  }finally{await rm(target,{recursive:true,force:true});}
}
const modes={bootstrap,pack,qualify,seal,verify:()=>verify(resolve(need('directory'))),download,publish,'public-consumer':publicConsumer,'github-release':githubRelease};assert(modes[mode],'Unsupported metadata release mode');await modes[mode]();
