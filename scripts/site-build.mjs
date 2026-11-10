import { createHash } from 'node:crypto';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.resolve(repo, process.argv[2] ?? '_site');
const source = path.join(repo, 'site');
if (path.dirname(output)!==repo||!/^_site(?:-[a-z0-9-]+)?$/.test(path.basename(output))) throw Error('Build output must use a reserved _site or _site-name directory in the repository root');
const packages = [
  ['ruv-fann-wsp','0.2.0','sha512-vCHGNiMUuxvCycd58gIOdlXTy1e2Tw6J1MD3FV9ri2ljkS534sk90c3h2mS4bBJozpebQu/fB6iDwKvdp09xIQ=='],
  ['ruv-fann','0.1.2','sha512-gdHjT8EUz4tSPpBYkCjbpISwW7bcB5XvtDPeRqZps8M9DcDF+I3deiIBt9aw6hxqKphrnbzQSltiFYLUYc0vjQ=='],
  ['ruv-fann-wasm','0.1.2','sha512-wV0xT+yMoidjs6ndeTkTO0rNUiDAZ4H7aP6slbVbdM1vG3xS6iCS1/jDwknNBo+m37GBXN4kgFXHZ0yRmMP7oQ==']
];
const sha = (bytes, algorithm='sha256', format='hex') => createHash(algorithm).update(bytes).digest(format);
async function safeFiles(dir, prefix='') {
  const result=[];
  for (const item of (await fs.readdir(dir,{withFileTypes:true})).sort((a,b)=>a.name.localeCompare(b.name))) {
    if (['node_modules','.git'].includes(item.name)) continue;
    const relative=path.posix.join(prefix,item.name);
    if (item.isSymbolicLink()) throw Error(`Symlink forbidden: ${relative}`);
    if (item.isDirectory()) result.push(...await safeFiles(path.join(dir,item.name),relative));
    else if (item.isFile()) result.push(relative);
    else throw Error(`Nonregular file forbidden: ${relative}`);
  }
  return result;
}
await fs.rm(output,{recursive:true,force:true}); await fs.mkdir(output,{recursive:true});
const allowed=/\.(?:html|css|js|svg|png|webp|woff2|ico|txt)$/;
const tracked=new Set(execFileSync('git',['ls-files','-z','--','site'],{cwd:repo,encoding:'utf8'}).split('\0').filter(Boolean));
let sourceTreeClean=!execFileSync('git',['status','--porcelain','--untracked-files=normal'],{cwd:repo,encoding:'utf8'}).trim();
for (const file of await safeFiles(source)) {
  if (!allowed.test(file) || file === 'README.md') continue;
  if (!tracked.has('site/'+file)) sourceTreeClean=false;
  await fs.mkdir(path.dirname(path.join(output,file)),{recursive:true});
  await fs.copyFile(path.join(source,file),path.join(output,file));
}
const temp=await fs.mkdtemp(path.join(os.tmpdir(),'fann-pages-'));
try {
  for (const [name,version,integrity] of packages) {
    const response=await fetch(`https://registry.npmjs.org/${name}/-/${name}-${version}.tgz`,{signal:AbortSignal.timeout(120000)});
    if(!response.ok) throw Error(`Public package download failed: ${name} ${response.status}`);
    const bytes=Buffer.from(await response.arrayBuffer());
    if ('sha512-'+sha(bytes,'sha512','base64')!==integrity) throw Error(`Pinned package integrity mismatch: ${name}`);
    const archive=path.join(temp,name+'.tgz'); await fs.writeFile(archive,bytes);
    const names=execFileSync('tar',['-tzf',archive],{encoding:'utf8'}).trim().split('\n');
    const types=execFileSync('tar',['-tvzf',archive],{encoding:'utf8'}).trim().split('\n');
    if (new Set(names).size!==names.length || names.some(n=>!n.startsWith('package/')||n.includes('..')||n.includes('\\')) || types.some(n=>!['-','d'].includes(n[0]))) throw Error(`Unsafe package archive: ${name}`);
    const unpack=path.join(temp,name); await fs.mkdir(unpack);
    execFileSync('tar',['-xzf',archive,'-C',unpack,'--no-same-owner','--no-same-permissions']);
    const meta=JSON.parse(await fs.readFile(path.join(unpack,'package/package.json'),'utf8'));
    if(meta.name!==name||meta.version!==version) throw Error('Package identity mismatch');
    for(const file of await safeFiles(path.join(unpack,'package'))) {
      if (!(file==='LICENSE'||file==='package.json'||/^(src|dist)\/.*\.(js|wasm|json)$/.test(file))) continue;
      const target=path.join(output,'vendor',name,file); await fs.mkdir(path.dirname(target),{recursive:true});
      await fs.copyFile(path.join(unpack,'package',file),target);
    }
  }
} finally { await fs.rm(temp,{recursive:true,force:true}); }
const sourceSha=execFileSync('git',['rev-parse','HEAD'],{cwd:repo,encoding:'utf8'}).trim();
const importmap=JSON.stringify({imports:Object.fromEntries(packages.map(([name])=>[name,`./vendor/${name}/src/index.js`]))});
const csp=`default-src 'none'; script-src 'self' 'wasm-unsafe-eval' 'sha256-${sha(importmap,'sha256','base64')}'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; upgrade-insecure-requests`;
const index=path.join(output,'index.html'); let html=await fs.readFile(index,'utf8');
if (!html.includes('<!-- SITE_CSP -->')||!html.includes('<!-- SITE_IMPORTMAP -->')) throw Error('Missing site security placeholders');
html=html.replace('<!-- SITE_CSP -->',`<meta http-equiv="Content-Security-Policy" content="${csp}">`).replace('<!-- SITE_IMPORTMAP -->',`<script type="importmap">${importmap}</script>`).replaceAll('__SOURCE_SHA__',sourceSha);
for(const tag of html.matchAll(/<(?:script|link|img)\b[^>]*>/gi)) {
  if (/^<link\b/i.test(tag[0])&&/\brel=["']canonical["']/i.test(tag[0])) continue;
  if(/\b(?:src|href)=["'](?:https?:)?\/\//i.test(tag[0])) throw Error('External runtime assets forbidden');
}
if (/\son\w+\s*=/i.test(html)) throw Error('Inline event handlers forbidden');
await fs.writeFile(index,html); await fs.writeFile(path.join(output,'.nojekyll'),'');
const files={}; for(const file of await safeFiles(output)) files[file]=sha(await fs.readFile(path.join(output,file)));
const manifest={schema:'ruv-fann.pages-manifest/1',sourceSha,sourceTreeClean,packages:packages.map(([name,version,integrity])=>({name,version,integrity})),files};
await fs.writeFile(path.join(output,'site-manifest.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify({output,sourceSha,files:Object.keys(files).length}));
