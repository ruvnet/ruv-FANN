// Shared metadata-only release invariants. No package runtime or repository-specific imports.
import {createHash} from 'node:crypto';
import {gzipSync, gunzipSync} from 'node:zlib';
import {isDeepStrictEqual as equal} from 'node:util';

export const schema='npm.metadata-release/1';
export const sha256=bytes=>createHash('sha256').update(bytes).digest('hex');
export const integrity=bytes=>`sha512-${createHash('sha512').update(bytes).digest('base64')}`;
export const assert=(condition,message)=>{if(!condition)throw Error(message);};
export function sourceSha(value){assert(typeof value==='string'&&/^[a-f0-9]{40}$/.test(value),'Full source SHA required');return value;}
export function filename(value){assert(typeof value==='string'&&/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,180}$/.test(value)&&!value.includes('..'),'Unsafe filename');return value;}
export function relative(value){assert(typeof value==='string'&&value.length>0&&!value.startsWith('/')&&!value.includes('\\')&&!value.includes('\0')&&value.split('/').every(p=>p&&p!=='.'&&p!=='..'),'Unsafe relative path');return value;}
export const tarballURL=row=>`https://registry.npmjs.org/${row.name}/-/${row.name.split('/').at(-1)}-${row.version}.tgz`;
export const archiveName=(name,version)=>filename(`${name.replace(/^@/,'').replace('/','-')}-${version}.tgz`);
export function validateBaselineMetadata(spec,baseline,metadata){
  assert(baseline.version===spec.fromVersion&&metadata?.name===spec.name&&metadata.version===spec.fromVersion,'Historical baseline registry identity mismatch');
  assert(metadata.dist?.tarball===tarballURL({name:spec.name,version:spec.fromVersion})&&/^sha512-[A-Za-z0-9+/]{86}==$/.test(metadata.dist.integrity)&&metadata.dist.integrity===baseline.integrity,'Historical baseline registry URL/SRI mismatch');
}
export function validateProtectedEnvironment(repository,sha,env){
  sourceSha(sha);
  const expected={GITHUB_ACTIONS:'true',GITHUB_REPOSITORY:repository,GITHUB_REF:'refs/heads/main',GITHUB_SHA:sha,GITHUB_WORKFLOW:'protected-metadata-release',GITHUB_WORKFLOW_REF:`${repository}/.github/workflows/protected-metadata-release.yml@refs/heads/main`,GITHUB_EVENT_NAME:'workflow_dispatch',GITHUB_JOB:'publish'};
  for(const [key,value]of Object.entries(expected))assert(env[key]===value,`Publication requires exact upstream protected main workflow: ${key}`);
}
export function validateConfig(config){
  assert(config.schema===schema&&/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(config.repository),'Invalid release configuration repository/schema');
  filename(config.releaseTag);
  assert(Array.isArray(config.packages)&&config.packages.length>0&&config.packages.length<=50,'Expected bounded package list');
  const names=new Set(),directories=new Set();
  for(const spec of config.packages){
    assert(typeof spec.name==='string'&&/^(?:@[a-z0-9_-]+\/)?[a-z0-9][a-z0-9._-]*$/.test(spec.name)&&!names.has(spec.name),'Invalid/duplicate package name');names.add(spec.name);
    if(spec.directory!=='.')relative(spec.directory);
    assert(!directories.has(spec.directory),'Duplicate package directory');directories.add(spec.directory);
    assert(/^\d+\.\d+\.\d+$/.test(spec.fromVersion)&&/^\d+\.\d+\.\d+$/.test(spec.toVersion)&&spec.fromVersion!==spec.toVersion,'Invalid release versions');
    assert(Array.isArray(spec.overlays)&&spec.overlays.includes('README.md')&&new Set(spec.overlays).size===spec.overlays.length&&spec.overlays.every(p=>['README.md','constellation.json'].includes(p)),'Only README and constellation version overlays allowed');
  }
  assert(Array.isArray(config.bootstrap)&&config.bootstrap.every(row=>Array.isArray(row)&&row.length>0&&row.every(arg=>typeof arg==='string'&&arg.length>0&&!arg.includes('\0'))),'Bootstrap must contain literal argv arrays');
  return config;
}

const text=(block,start,length)=>{const bytes=block.subarray(start,start+length),end=bytes.indexOf(0);return bytes.subarray(0,end<0?length:end).toString('utf8');};
function octal(block,start,length){const value=text(block,start,length).trim();assert(/^[0-7]*$/.test(value),'Unsupported tar numeric field');return value?Number.parseInt(value,8):0;}
export function readArchive(bytes){
  assert(Buffer.isBuffer(bytes)&&bytes.length<=40*1024*1024,'Oversized compressed archive');
  const raw=gunzipSync(bytes,{maxOutputLength:100*1024*1024}),files=new Map();
  assert(raw.length%512===0,'Truncated tar archive');
  let offset=0,terminated=false;
  while(offset<raw.length){
    const header=raw.subarray(offset,offset+512);offset+=512;
    if(header.every(byte=>byte===0)){assert(raw.subarray(offset).every(byte=>byte===0),'Data after tar terminator');terminated=true;break;}
    let checksum=0;for(let i=0;i<512;i++)checksum+=i>=148&&i<156?32:header[i];
    assert(octal(header,148,8)===checksum,'Tar checksum mismatch');
    assert(text(header,257,6)==='ustar'&&text(header,263,2)==='00','Only POSIX USTAR archives allowed');
    const type=text(header,156,1);assert(type===''||type==='0','Only regular archive files allowed (no links, directories or extension headers)');
    assert(text(header,157,100)==='','Archive links forbidden');
    const prefix=text(header,345,155),name=(prefix?prefix+'/':'')+text(header,0,100);relative(name);
    assert(name.startsWith('package/')&&name!=='package/'&&!files.has(name)&&Buffer.byteLength(name)===name.length,'Unsafe/duplicate/non-ASCII npm archive path');
    assert(![...files.keys()].some(existing=>existing.startsWith(name+'/')||name.startsWith(existing+'/')),'Archive file/directory path collision');
    const mode=octal(header,100,8),size=octal(header,124,12);assert([0o644,0o755].includes(mode),'Unsupported archive file mode');
    assert(Number.isSafeInteger(size)&&size>=0&&offset+size<=raw.length,'Truncated archive entry');
    files.set(name,{bytes:Buffer.from(raw.subarray(offset,offset+size)),mode});offset+=Math.ceil(size/512)*512;
    assert(files.size<=10000,'Too many package files');
  }
  assert(terminated&&files.has('package/package.json'),'Missing tar terminator/package manifest');return files;
}

export function writeArchive(files){
  const chunks=[];
  for(const name of [...files.keys()].sort()){
    relative(name);assert(name.startsWith('package/'),'Invalid npm archive root');
    const entry=files.get(name),header=Buffer.alloc(512);let base=name,prefix='';
    if(Buffer.byteLength(base)>100){const split=base.lastIndexOf('/');prefix=base.slice(0,split);base=base.slice(split+1);}
    assert(Buffer.byteLength(base)<=100&&Buffer.byteLength(prefix)<=155,'Archive path too long');
    const put=(value,start,length)=>{assert(Buffer.byteLength(value)<=length,'Tar field overflow');header.write(value,start,length,'ascii');};
    const number=(value,start,length)=>put(value.toString(8).padStart(length-1,'0')+'\0',start,length);
    put(base,0,100);number(entry.mode,100,8);number(0,108,8);number(0,116,8);number(entry.bytes.length,124,12);number(0,136,12);
    put('        ',148,8);put('0',156,1);put('ustar\0',257,6);put('00',263,2);put(prefix,345,155);
    const sum=header.reduce((total,byte)=>total+byte,0);put(sum.toString(8).padStart(6,'0')+'\0 ',148,8);
    chunks.push(header,entry.bytes,Buffer.alloc((512-entry.bytes.length%512)%512));
  }
  chunks.push(Buffer.alloc(1024));return gzipSync(Buffer.concat(chunks),{level:9,mtime:0});
}

export function validateOverlay(path,before,after,spec){
  if(path==='package.json'){
    const a=JSON.parse(before),b=JSON.parse(after);
    assert(a.name===spec.name&&b.name===spec.name&&a.version===spec.fromVersion&&b.version===spec.toVersion&&!b.private,'Package identity/version mismatch');
    assert(Array.isArray(b.keywords)&&b.keywords.length>0&&b.keywords.every(v=>typeof v==='string'&&v.trim().length>0)&&new Set(b.keywords).size===b.keywords.length,'Invalid package keywords');
    for(const item of [a,b]){delete item.version;delete item.keywords;}
    assert(equal(a,b),'Metadata release changed package behavior outside version/keywords');
  }else if(path==='README.md'){
    assert(after.subarray(0,before.length).equals(before),'README must preserve every published byte');
    const suffix=after.subarray(before.length).toString('utf8');
    assert(/^\n+## Discoverability topics\n\n?\S/.test(suffix)&&!/[`<>\r\0]/.test(suffix)&&suffix.length<=5000,'README must append a bounded plain-text Discoverability topics section');
  }else if(path==='constellation.json'){
    const a=JSON.parse(before),b=JSON.parse(after);assert(a.version===spec.fromVersion&&b.version===spec.toVersion,'Constellation identity version mismatch');
    delete a.version;delete b.version;assert(equal(a,b),'Constellation behavior changed outside root version');
  }else throw Error('Unexpected metadata overlay');
}
export function validatePayload(baseline,candidate,spec,overlays){
  assert(equal([...baseline.keys()].sort(),[...candidate.keys()].sort()),'Metadata release changed package file set');
  const changed=new Set(['package.json',...spec.overlays]);
  for(const [name,old]of baseline){
    const next=candidate.get(name),path=name.slice(8);assert(old.mode===next.mode,'Metadata release changed executable mode');
    if(changed.has(path)){assert(Buffer.isBuffer(overlays[path])&&next.bytes.equals(overlays[path]),'Overlay differs from source HEAD');validateOverlay(path,old.bytes,next.bytes,spec);}
    else assert(old.bytes.equals(next.bytes),`Runtime/binary bytes changed: ${name}`);
  }
  for(const path of changed)assert(baseline.has(`package/${path}`),'Overlay absent from published baseline');
}
export function validateConsumer(receipt,sha){
  assert(receipt?.status==='passed'&&receipt.sourceSha===sourceSha(sha)&&Array.isArray(receipt.checks)&&receipt.checks.length>0,'Consumer must pass nonempty source-bound checks');
  for(const check of receipt.checks)assert(typeof check.name==='string'&&check.name.length>0&&Number.isSafeInteger(check.tests)&&check.tests>0&&check.failed===0&&check.skipped===0&&check.todo===0,'Consumer checks must have positive tests, zero failed/skipped/todo');
  return receipt;
}
export function publicLock(lock,rows){
  assert(lock.lockfileVersion===3&&lock.packages&&lock.packages[''],'Missing npm v3 dependency lock');const result=structuredClone(lock);
  for(const row of rows){const entry=result.packages[`node_modules/${row.name}`];assert(entry&&entry.version===row.version&&!entry.link,'Wrong candidate lock identity');entry.resolved=tarballURL(row);entry.integrity=row.integrity;result.packages[''].dependencies[row.name]=row.version;}
  for(const [name,entry]of Object.entries(result.packages)){if(name==='')continue;assert(!entry.link&&entry.resolved?.startsWith('https://registry.npmjs.org/')&&/^sha512-[A-Za-z0-9+/]{86}==$/.test(entry.integrity),'Non-public/unsealed dependency in lock');}
  return result;
}
