import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm,open,mkdir,symlink,lstat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawnSync,execFileSync} from 'node:child_process';
import {runConstellationScenario} from './constellation-scenario.mjs';
test('catalog planning, satellite quality checks and privacy-aware attach compose end-to-end',async()=>{
  const result=await runConstellationScenario();assert.equal(result.catalogEntries,24);assert.equal(result.published,12);assert.equal(result.scaffoldNoInstall,true);
});
test('public report command rejects malformed, oversized and nonregular catalog input before network',async()=>{
  const directory=await mkdtemp(join(tmpdir(),'constellation-cli-')),script=new URL('../scripts/constellation-report.mjs',import.meta.url).pathname;
  try {
    const malformed=join(directory,'bad.json');await writeFile(malformed,'{}');
    assert.notEqual(spawnSync(process.execPath,[script,'--catalog',malformed],{timeout:3000}).status,0);
    const oversized=join(directory,'big.json'),file=await open(oversized,'w');await file.truncate(3*1024*1024);await file.close();
    const big=spawnSync(process.execPath,[script,'--catalog',oversized],{timeout:3000,encoding:'utf8'});assert.notEqual(big.status,0);assert.match(big.stderr,/byte limit/);
    const fifo=join(directory,'pipe');execFileSync('mkfifo',[fifo]);const nonregular=spawnSync(process.execPath,[script,'--catalog',fifo],{timeout:3000,encoding:'utf8'});
    assert.notEqual(nonregular.status,0);assert.match(nonregular.stderr,/regular file/);
    const help=spawnSync(process.execPath,[script,'--help'],{timeout:3000,encoding:'utf8'});assert.equal(help.status,0);assert.match(help.stdout,/no identifiers|no scheduled|no.*upload/i);
  } finally {await rm(directory,{recursive:true,force:true});}
});
test('report output atomically replaces leaf symlinks without overwriting their destinations',async()=>{
  const directory=await mkdtemp(join(tmpdir(),'constellation-output-')),output=join(directory,'report');await mkdir(output);
  const preload=join(directory,'public-fixture.mjs'),script=new URL('../scripts/constellation-report.mjs',import.meta.url).pathname;
  try {
    await writeFile(preload,`globalThis.fetch=async input=>{const u=new URL(String(input));let data;if(u.hostname==='api.npmjs.org'){const parts=u.pathname.split('/');const [start,end]=parts[3].split(':');data={package:decodeURIComponent(parts[4]),start,end,downloads:[]};}else if(u.hostname==='registry.npmjs.org'){data={name:decodeURIComponent(u.pathname.split('/')[1]),version:'0.1.0'};}else if(u.hostname==='api.github.com'){data={full_name:u.pathname.slice('/repos/'.length),stargazers_count:1,open_issues_count:0};}else throw Error('unexpected host');return new Response(JSON.stringify(data));};`);
    for(const name of ['report.json','observations.json','dashboard.html']){const sentinel=join(directory,'sentinel-'+name);await writeFile(sentinel,'preserve-me');await symlink(sentinel,join(output,name));}
    const child=spawnSync(process.execPath,['--import',preload,script,'--start','2026-10-01','--end','2026-11-10','--output',output],{timeout:10000,encoding:'utf8'});
    assert.equal(child.status,0,child.stderr);
    for(const name of ['report.json','observations.json','dashboard.html']){assert.equal(await readFile(join(directory,'sentinel-'+name),'utf8'),'preserve-me');assert.equal((await lstat(join(output,name))).isSymbolicLink(),false);}
  }finally{await rm(directory,{recursive:true,force:true});}
});
