import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { verifySite } from './site-seal.mjs';
import { serveSite } from './site-serve.mjs';
const root=path.resolve(process.argv[2]??'_site');const temp=await fs.mkdtemp(path.join(os.tmpdir(),'fann-pages-hostile-'));
let checks=0;const test=async(name,fn)=>{await fn();checks++;console.log('PASS '+name);};
try {
  await fs.cp(root,temp,{recursive:true});const original=await fs.readFile(path.join(temp,'site-manifest.json'),'utf8');const m=JSON.parse(original);
  await test('unchanged manifest verifies',()=>verifySite(temp,m.sourceSha));
  await test('wrong expected source fails',()=>assert.rejects(verifySite(temp,'0'.repeat(40)),/mismatch/));
  const target=Object.keys(m.files).find(n=>n.endsWith('.js'));const bytes=await fs.readFile(path.join(temp,target));
  await fs.appendFile(path.join(temp,target),'\n// tamper');
  await test('runtime byte tampering fails',()=>assert.rejects(verifySite(temp),/digest mismatch/));await fs.writeFile(path.join(temp,target),bytes);
  await fs.writeFile(path.join(temp,'unlisted.js'),'alert(1)');await test('unlisted files fail',()=>assert.rejects(verifySite(temp),/file set mismatch/));await fs.rm(path.join(temp,'unlisted.js'));
  await fs.symlink('/etc/passwd',path.join(temp,'unsafe'));await test('symlinks fail',()=>assert.rejects(verifySite(temp),/Symlink/));await fs.rm(path.join(temp,'unsafe'));
  m.files['../escape']='0'.repeat(64);await fs.writeFile(path.join(temp,'site-manifest.json'),JSON.stringify(m));await test('manifest traversal fails',()=>assert.rejects(verifySite(temp)));await fs.writeFile(path.join(temp,'site-manifest.json'),original);
  const {server,url}=await serveSite(temp);
  try{
    await test('outside base path rejected',async()=>assert.equal((await fetch(new URL('/index.html',url))).status,404));
    await test('encoded traversal rejected',async()=>assert.notEqual((await fetch(url+'..%2f..%2fetc%2fpasswd')).status,200));
    await fs.symlink(path.dirname(temp),path.join(temp,'outside'));
    await test('symlinked parent rejected',async()=>assert.notEqual((await fetch(url+'outside/'+path.basename(temp)+'/index.html')).status,200));
    await fs.rm(path.join(temp,'outside'));
    await test('MIME supports real WASM',async()=>{const wasm=Object.keys(m.files).find(n=>n.endsWith('.wasm'));assert.equal((await fetch(url+wasm)).headers.get('content-type'),'application/wasm');});
  }finally{await new Promise(r=>server.close(r));}
  console.log(JSON.stringify({schema:'ruv-fann.pages-security/1',status:'passed',tests:checks,failed:0,skipped:0,sourceSha:m.sourceSha}));
}finally{await fs.rm(temp,{recursive:true,force:true});}
