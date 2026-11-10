import { promises as fs } from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
const expected=process.env.EXPECTED_SHA;
const repo=process.env.GITHUB_REPOSITORY;
if(process.env.GITHUB_ACTIONS!=='true'||process.env.GITHUB_WORKFLOW!=='Pages story'||process.env.GITHUB_EVENT_NAME!=='workflow_dispatch'||! /^[a-f0-9]{40}$/.test(expected??'')||!repo)throw Error('Protected Pages workflow invocation required');
const output=path.resolve(process.argv[2]??'site-candidate');await fs.mkdir(output,{recursive:true});
const api=endpoint=>JSON.parse(execFileSync('gh',['api',endpoint],{encoding:'utf8',maxBuffer:16*1024*1024}));
const runs=api(`repos/${repo}/actions/workflows/pages-story.yml/runs?event=pull_request&status=success&per_page=100`).workflow_runs.filter(r=>r.head_sha===expected&&r.conclusion==='success'&&r.event==='pull_request');
if(!runs.length)throw Error('No successful exact-SHA Pages qualification');
let chosen;
for(const run of runs){const artifacts=api(`repos/${repo}/actions/runs/${run.id}/artifacts?per_page=100`).artifacts;const found=artifacts.find(a=>a.name===`pages-candidate-${expected}`&&!a.expired);if(found){chosen=found;break;}}
if(!chosen)throw Error('Exact qualified site artifact is absent or expired');
const zip=path.join(output,'candidate.zip');await fs.writeFile(zip,execFileSync('gh',['api',`repos/${repo}/actions/artifacts/${chosen.id}/zip`],{maxBuffer:64*1024*1024}));
const names=execFileSync('unzip',['-Z1',zip],{encoding:'utf8'}).trim().split('\n');
if(names.some(n=>path.isAbsolute(n)||n.split('/').includes('..')||n.includes('\\')))throw Error('Unsafe candidate archive');
execFileSync('unzip',['-q',zip,'-d',output]);await fs.rm(zip);
execFileSync(process.execPath,['scripts/site-seal.mjs','verify',output],{stdio:'inherit',env:process.env});
