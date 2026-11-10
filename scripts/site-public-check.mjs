import { promises as fs } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
const root=path.resolve(process.argv[2]??'_site'), output=path.resolve(process.argv[3]??'_site-public-evidence');
const url=process.env.SITE_PUBLIC_URL??'https://ruvnet.github.io/ruv-FANN/';
if(new URL(url).origin!=='https://ruvnet.github.io'||new URL(url).pathname!=='/ruv-FANN/')throw Error('Unexpected Pages publication URL');
const manifestBytes=await fs.readFile(path.join(root,'site-manifest.json'));const manifest=JSON.parse(manifestBytes);
if(process.env.EXPECTED_SHA&&process.env.EXPECTED_SHA!==manifest.sourceSha)throw Error('Expected source differs from candidate');
const hash=b=>createHash('sha256').update(b).digest('hex');
await fs.mkdir(output,{recursive:true});
let last;
for(let attempt=0;attempt<20;attempt++){
  try{
    const deployed=await fetch(new URL(`site-manifest.json?source=${manifest.sourceSha}&attempt=${attempt}`,url),{signal:AbortSignal.timeout(30000),cache:'no-store'});
    if(!deployed.ok||hash(Buffer.from(await deployed.arrayBuffer()))!==hash(manifestBytes))throw Error('Public source manifest has not converged');
    last=null;break;
  }catch(error){last=error;if(attempt<19)await new Promise(r=>setTimeout(r,15000));}
}
if(last)throw last;
for(const [name,digest] of Object.entries(manifest.files)){
  const r=await fetch(new URL(name+`?source=${manifest.sourceSha}`,url),{signal:AbortSignal.timeout(60000),cache:'no-store'});
  if(!r.ok||hash(Buffer.from(await r.arrayBuffer()))!==digest)throw Error(`Public site bytes differ: ${name}`);
}
execFileSync(process.execPath,['scripts/site-test.mjs',root,output],{stdio:'inherit',env:{...process.env,SITE_PUBLIC_URL:url}});
await fs.writeFile(path.join(output,'publication-receipt.json'),JSON.stringify({schema:'ruv-fann.pages-publication/1',status:'verified',sourceSha:manifest.sourceSha,url,manifestSha256:hash(manifestBytes),filesVerified:Object.keys(manifest.files).length,browserReceiptSha256:hash(await fs.readFile(path.join(output,'browser-receipt.json')))},null,2)+'\n');
