#!/usr/bin/env node
// Never accept a caller-supplied run id: select successful upstream exact-head qualification.
import {execFileSync} from 'node:child_process';
import {mkdtemp,mkdir,readdir,rename,rm,readFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {assertSourceSha,safeFilename} from './toolkit-release-lib.mjs';
const args=process.argv.slice(2),options=Object.create(null);
for(let i=0;i<args.length;i+=2){if(!/^--[a-z-]+$/.test(args[i])||!args[i+1]||Object.hasOwn(options,args[i].slice(2)))throw Error('Invalid argument');options[args[i].slice(2)]=args[i+1];}
const sha=assertSourceSha(options['expected-sha']),output=resolve(options.output??'');
if(!options.output||process.env.GITHUB_ACTIONS!=='true'||process.env.GITHUB_REPOSITORY!=='ruvnet/ruv-FANN'||process.env.GITHUB_WORKFLOW!=='protected-toolkit-release'||process.env.GITHUB_REF!=='refs/heads/main'||process.env.GITHUB_SHA!==sha||process.env.GITHUB_EVENT_NAME!=='workflow_dispatch')throw Error('Exact protected upstream main context required');
const run=(bin,args)=>execFileSync(bin,args,{encoding:'utf8',stdio:['ignore','pipe','inherit'],timeout:5*60_000,maxBuffer:10*1024*1024}).trim();
const api=path=>JSON.parse(run('gh',['api',path])),repo='ruvnet/ruv-FANN';
const runs=api(`repos/${repo}/actions/workflows/wsp-toolkit.yml/runs?head_sha=${sha}&status=completed&per_page=100`).workflow_runs
  .filter(row=>row.head_sha===sha&&row.head_repository?.full_name===repo&&row.conclusion==='success'&&['pull_request','workflow_dispatch'].includes(row.event))
  .sort((a,b)=>b.run_number-a.run_number);
let selected;const name=`toolkit-candidate-${sha}`;
for(const item of runs){const matches=api(`repos/${repo}/actions/runs/${item.id}/artifacts?per_page=100`).artifacts.filter(row=>row.name===name&&!row.expired);if(matches.length===1){selected=item;break;}}
if(!selected)throw Error('No successful upstream exact-source sealed candidate artifact');
const temp=await mkdtemp(join(tmpdir(),'toolkit-candidate-download-'));
try{
  run('gh',['run','download',String(selected.id),'-R',repo,'-n',name,'-D',temp]);
  const files=await readdir(temp);files.forEach(safeFilename);
  const seal=JSON.parse(await readFile(join(temp,'candidate-seal.json'),'utf8'));
  if(seal.sourceSha!==sha)throw Error('Downloaded seal source mismatch');
  run(process.execPath,['scripts/toolkit-release-lane.mjs','verify','--directory',temp]);
  await mkdir(output,{recursive:true});if((await readdir(output)).length)throw Error('Candidate output must be empty');
  for(const file of files)await rename(join(temp,file),join(output,file));
  console.log(JSON.stringify({status:'verified',sourceSha:sha,runId:selected.id,artifact:name}));
}finally{await rm(temp,{recursive:true,force:true});}
