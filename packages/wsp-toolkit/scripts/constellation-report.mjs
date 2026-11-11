import { constants } from 'node:fs';
import { open,mkdir,rename,unlink,lstat } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { resolve,join,dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateCatalog } from 'ruvnet-constellation-contracts';
import { collectPublicMetrics,analyzeConstellation,renderConstellationDashboard } from 'ruvnet-constellation-telemetry';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2),options={};
for(let i=0;i<args.length;i++) {
  const key=args[i];
  if(key==='--help'){console.log('Usage: node scripts/constellation-report.mjs [--catalog file] [--adoption explicit-consenting-project-samples.json] [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--output directory]\nReads public npm/GitHub metrics. Adoption input stays local; no identifiers enter reports. No scheduled collection or upload.');process.exit(0);}
  if(!['--catalog','--adoption','--start','--end','--output'].includes(key)||options[key]!==undefined||!args[i+1]||args[i+1].startsWith('--'))throw new Error('unknown, duplicate or incomplete report option');
  options[key]=args[++i];
}
async function readJSON(path,maxBytes) {
  const file=await open(path,constants.O_RDONLY|constants.O_NONBLOCK);
  try {
    if(!(await file.stat()).isFile())throw new TypeError('input must be a regular file');
    const bytes=Buffer.alloc(maxBytes+1);let length=0;
    while(length<bytes.length){const {bytesRead}=await file.read(bytes,length,bytes.length-length,null);if(!bytesRead)break;length+=bytesRead;}
    if(length>maxBytes)throw new RangeError('input file exceeds byte limit');
    try{return JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes.subarray(0,length)));}catch{throw new TypeError('input must contain valid UTF-8 JSON');}
  }finally{await file.close();}
}
const configuredEnd=options['--end']??(process.env.CONSTELLATION_REPORT_END||undefined);
if(configuredEnd!==undefined&&(!/^\d{4}-\d{2}-\d{2}$/.test(configuredEnd)||new Date(configuredEnd+'T00:00:00Z').toISOString().slice(0,10)!==configuredEnd))throw new TypeError('end must be a valid UTC calendar day');
const end=configuredEnd??new Date(Date.UTC(new Date().getUTCFullYear(),new Date().getUTCMonth(),new Date().getUTCDate()-1)).toISOString().slice(0,10);
const start=options['--start']??new Date(Date.parse(end+'T00:00:00Z')-41*86400000).toISOString().slice(0,10);
const catalog=validateCatalog(await readJSON(resolve(options['--catalog']??join(root,'packages/constellation/catalog.json')),2*1024*1024));
const adoption=options['--adoption']===undefined?undefined:await readJSON(resolve(options['--adoption']),8*1024*1024);
const observations=await collectPublicMetrics(catalog,{start,end,timeoutMs:60000,concurrency:4});
const report=analyzeConstellation(catalog,observations,adoption===undefined?undefined:{adoption});
const output=resolve(options['--output']??join(root,'.artifacts/constellation'));
await mkdir(output,{recursive:true});
if((await lstat(output)).isSymbolicLink())throw new TypeError('output must be a real directory');
async function atomicWrite(name,content) {
  const temporary=join(output,'.'+name+'.'+randomUUID()+'.tmp');
  let created=false;
  try{
    const file=await open(temporary,constants.O_WRONLY|constants.O_CREAT|constants.O_EXCL|(constants.O_NOFOLLOW??0),0o600);created=true;
    try{await file.writeFile(content);}finally{await file.close();}
    await rename(temporary,join(output,name));
  }finally{if(created)await unlink(temporary).catch(error=>{if(error.code!=='ENOENT')throw error;});}
}
await atomicWrite('observations.json',JSON.stringify(observations,null,2)+'\n');
await atomicWrite('report.json',JSON.stringify(report,null,2)+'\n');
await atomicWrite('dashboard.html',renderConstellationDashboard(report));
console.log(JSON.stringify({catalogEntries:catalog.packages.length,published:catalog.packages.filter(p=>p.status==='published').length,
  packageCoverage:observations.packages.reduce((counts,p)=>({...counts,[p.status]:(counts[p.status]??0)+1}),{}),
  repositoryCoverage:observations.repositories.reduce((counts,r)=>({...counts,[r.status]:(counts[r.status]??0)+1}),{}),
  start,end,attachStatus:report.attach.status,unit:report.attach.unit,output},null,2));
