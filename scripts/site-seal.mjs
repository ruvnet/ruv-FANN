import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const requiredChecks=['sourceBinding','allStorySections','desktopNoOverflow','localLinksResolve','scrollNarrative','motionPause','keyboardControls','copyInstall','benchmarkTabs','demoExplicitStart','demoRejectedPatch','demoAcceptedPatch','demoExplicitPromotion','demoReceiptInvalidation','demoRollback','demoReset','accessibilityDesktop','mobileNoOverflow','reducedMotion','accessibilityMobile','narrowNoOverflow','noJavaScriptReadable','strictBasePath','runtimeSelfOrigin','noBrowserErrors'];
function validBrowser(r,sha,manifest) {
  return r?.schema==='ruv-fann.pages-browser/1'&&r.status==='passed'&&r.sourceSha===sha&&r.manifestSha256===manifest&&r.failed===0&&r.skipped===0&&r.tests===Object.keys(r.checks??{}).length&&requiredChecks.every(k=>r.checks?.[k]===true)&&Object.values(r.checks).every(v=>v===true);
}
export async function verifySite(root,expectedSha) {
  const bytes=await fs.readFile(path.join(root,'site-manifest.json')); const m=JSON.parse(bytes);
  if(m.schema!=='ruv-fann.pages-manifest/1'||! /^[a-f0-9]{40}$/.test(m.sourceSha)|| (expectedSha&&m.sourceSha!==expectedSha)) throw Error('Source-bound manifest mismatch');
  const actual=[];
  async function walk(dir,prefix='') {for(const e of await fs.readdir(dir,{withFileTypes:true})){const name=path.posix.join(prefix,e.name);if(e.isSymbolicLink())throw Error('Symlink forbidden');if(e.isDirectory())await walk(path.join(dir,e.name),name);else if(e.isFile())actual.push(name);else throw Error('Nonregular file');}}
  await walk(root);
  const expected=Object.keys(m.files).concat('site-manifest.json').sort();if(JSON.stringify(actual.sort())!==JSON.stringify(expected))throw Error('Manifest file set mismatch');
  for(const [file,digest] of Object.entries(m.files)){if(path.isAbsolute(file)||file.split('/').includes('..')||file.includes('\\'))throw Error('Unsafe manifest path');if(hash(await fs.readFile(path.join(root,file)))!==digest)throw Error(`Site digest mismatch: ${file}`);}
  return {manifest:m,manifestSha256:hash(bytes)};
}
const [mode,first='_site',evidence='_site-evidence',out='site-candidate']=process.argv.slice(2);
if (mode==='seal') {
  const root=path.resolve(repo,first), proof=path.resolve(repo,evidence), candidate=path.resolve(repo,out);
  const sourceSha=execFileSync('git',['rev-parse','HEAD'],{cwd:repo,encoding:'utf8'}).trim();
  if(execFileSync('git',['status','--porcelain','--untracked-files=normal'],{cwd:repo,encoding:'utf8'}).trim())throw Error('Clean committed source required for sealing');
  const verified=await verifySite(root,sourceSha);
  if(verified.manifest.sourceTreeClean!==true)throw Error('Site was built from uncommitted or untracked source');
  const receipt=JSON.parse(await fs.readFile(path.join(proof,'browser-receipt.json'),'utf8'));
  if(!validBrowser(receipt,sourceSha,verified.manifestSha256))throw Error('Nonzero complete source-bound browser qualification required');
  await fs.mkdir(candidate,{recursive:true});const archive=path.join(candidate,'site.tar.gz');
  execFileSync('tar',['--sort=name','--mtime=@0','--owner=0','--group=0','--numeric-owner','-czf',archive,'-C',root,'.']);
  await fs.cp(proof,path.join(candidate,'evidence'),{recursive:true});
  const seal={schema:'ruv-fann.pages-candidate/1',sourceSha,archive:'site.tar.gz',archiveSha256:hash(await fs.readFile(archive)),manifestSha256:verified.manifestSha256,browserReceiptSha256:hash(await fs.readFile(path.join(candidate,'evidence/browser-receipt.json')))};
  await fs.writeFile(path.join(candidate,'candidate-seal.json'),JSON.stringify(seal,null,2)+'\n');console.log(JSON.stringify(seal));
} else if (mode==='verify') {
  const candidate=path.resolve(repo,first),seal=JSON.parse(await fs.readFile(path.join(candidate,'candidate-seal.json'),'utf8'));
  if(seal.schema!=='ruv-fann.pages-candidate/1'||(process.env.EXPECTED_SHA&&seal.sourceSha!==process.env.EXPECTED_SHA))throw Error('Candidate source mismatch');
  const archive=path.join(candidate,'site.tar.gz');if(hash(await fs.readFile(archive))!==seal.archiveSha256)throw Error('Archive seal mismatch');
  const names=execFileSync('tar',['-tzf',archive],{encoding:'utf8'}).trim().split('\n');const types=execFileSync('tar',['-tvzf',archive],{encoding:'utf8'}).trim().split('\n');
  if(names.some(n=>!n.startsWith('./')||n.split('/').includes('..')||n.includes('\\'))||types.some(n=>!['-','d'].includes(n[0])))throw Error('Unsafe sealed archive');
  const temp=await fs.mkdtemp(path.join(os.tmpdir(),'fann-pages-verify-'));
  try {execFileSync('tar',['-xzf',archive,'-C',temp,'--no-same-owner','--no-same-permissions']);const v=await verifySite(temp,seal.sourceSha);if(v.manifestSha256!==seal.manifestSha256||v.manifest.sourceTreeClean!==true)throw Error('Manifest seal mismatch or uncommitted source');}finally{await fs.rm(temp,{recursive:true,force:true});}
  const bytes=await fs.readFile(path.join(candidate,'evidence/browser-receipt.json'));const r=JSON.parse(bytes);
  if(hash(bytes)!==seal.browserReceiptSha256||!validBrowser(r,seal.sourceSha,seal.manifestSha256))throw Error('Browser seal mismatch');
  console.log(JSON.stringify({status:'verified',sourceSha:seal.sourceSha,archiveSha256:seal.archiveSha256}));
} else if(process.argv[1]===fileURLToPath(import.meta.url)) throw Error('Usage: site-seal.mjs seal [site] [evidence] [candidate] | verify [candidate]');
