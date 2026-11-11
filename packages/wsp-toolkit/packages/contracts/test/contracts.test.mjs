import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {CONTRACTS,CONTRACT_LIMITS,validateCatalog,validateModuleManifest} from '../src/index.js';
const module=()=>({schema:CONTRACTS.module,name:'@example/adapter',version:'0.1.0',role:'satellite',contracts:[CONTRACTS.model,CONTRACTS.patch],integrations:['agentdb']});
const catalog=()=>({schema:CONTRACTS.catalog,repository:'example/repository',packages:[{name:'anchor',version:'1.2.3',role:'nucleus',family:'example',status:'published',repository:'example/repository',description:'Example anchor',links:{npm:'https://www.npmjs.com/package/anchor',source:'https://github.com/example/repository'},proposedScope:'@example'},{name:'candidate',version:'0.1.0',role:'satellite',family:'example',status:'candidate',repository:'example/repository',directory:'packages/candidate',description:'Local candidate',contracts:[CONTRACTS.module]}],profiles:[{id:'example',description:'Explicit selection',packages:['anchor','candidate']}]});

test('catalog/module contracts own the data and preserve status without claiming namespace authority',()=>{
  const input=catalog(),result=validateCatalog(input);input.packages[0].description='changed';input.profiles[0].packages.pop();
  assert.equal(result.packages[0].description,'Example anchor');assert.equal(result.profiles[0].packages.length,2);
  assert.equal(result.packages[1].status,'candidate');assert.equal(result.packages[0].proposedScope,'@example');
  const source=module(),manifest=validateModuleManifest(source);source.contracts[0]='changed';source.integrations.push('other');
  assert.equal(manifest.contracts[0],CONTRACTS.model);assert.deepEqual(manifest.integrations,['agentdb']);
});
test('the actual default catalog uses versioned contracts and known profile members',async()=>{
  const value=JSON.parse(await readFile(new URL('../../constellation/catalog.json',import.meta.url),'utf8'));
  const result=validateCatalog(value);
  assert.ok(result.packages.some(pkg=>pkg.name==='ruv-fann-wsp'&&pkg.status==='published'));
  assert.ok(result.packages.some(pkg=>pkg.name==='ruvnet-constellation'&&pkg.status==='published'));
  assert.ok(result.profiles.some(profile=>profile.id==='local-loop'));
  assert.equal(result.packages.length,24);assert.ok(result.packages.every(pkg=>pkg.status==='published'));
  assert.equal(result.packages.find(pkg=>pkg.name==='ruv-fann').version,'0.2.0');
});
test('unknown fields, classes, getters, symbols and hostile arrays reject without invocation',()=>{
  let calls=0;
  for(const mutate of [x=>x.unknown=true,x=>Object.defineProperty(x,'name',{get(){calls++;return 'bad';}}),x=>x[Symbol('extra')]=true,
    x=>x.contracts=new Array(2),x=>Object.defineProperty(x.contracts,0,{get(){calls++;return CONTRACTS.model;}}),
    x=>x.integrations[Symbol.iterator]=function*(){calls++;yield 'agentdb';}]) {
    const input=module();mutate(input);assert.throws(()=>validateModuleManifest(input));
  }
  class Custom{constructor(){Object.assign(this,module());}}
  assert.throws(()=>validateModuleManifest(new Custom()));assert.equal(calls,0);
});
test('safe lowercase names, SemVer, schema IDs and required fields fail closed',()=>{
  for(const name of ['BadName','../escape','@bad/../escape','a b','a;$(id)','_hidden','node_modules','favicon.ico','@/missing','@x/','a'.repeat(215)])assert.throws(()=>validateModuleManifest({...module(),name}));
  for(const version of ['latest','1.0','01.2.3','1.2.3-alpha.01','1.2.3;cmd'])assert.throws(()=>validateModuleManifest({...module(),version}));
  assert.equal(validateModuleManifest({...module(),version:'1.2.3-alpha.1+build.20'}).version,'1.2.3-alpha.1+build.20');
  for(const contracts of [[],['unversioned'],['javascript:bad/1'],[CONTRACTS.model,CONTRACTS.model]])assert.throws(()=>validateModuleManifest({...module(),contracts}));
  for(const role of ['admin','unknown',null])assert.throws(()=>validateModuleManifest({...module(),role}));
  const missing=module();delete missing.integrations;assert.throws(()=>validateModuleManifest(missing));
});
test('relative repository paths and approved HTTPS links reject traversal and credential-bearing URLs',()=>{
  for(const path of ['/etc/passwd','a/../b','../b','a/./b','a\\b','a//b','a/%2e%2e/b','https://other/path']) {
    const input=catalog();input.packages[1].directory=path;assert.throws(()=>validateCatalog(input));
  }
  for(const source of ['http://github.com/example/repository','https://evil.test','javascript:alert(1)','https://user:secret@github.com/example/repository','https://github.com:8443/example/repository','https://github.com/example/repository?token=example']) {
    const input=catalog();input.packages[0].links.source=source;assert.throws(()=>validateCatalog(input));
  }
  const input=catalog();input.packages[0].links.npm='https://github.com/elsewhere';assert.throws(()=>validateCatalog(input));
});
test('duplicate package/profile/member identities, invalid roles and unknown references reject',()=>{
  for(const mutate of [x=>x.packages.push(x.packages[0]),x=>x.profiles.push(x.profiles[0]),x=>x.profiles[0].packages.push('anchor'),
    x=>x.profiles[0].packages.push('absent'),x=>x.packages[0].role='hidden',x=>x.packages[0].status='verified',
    x=>x.profiles[0].id='bad:id',x=>x.packages[0].proposedScope='@example/name']) {
    const input=catalog();mutate(input);assert.throws(()=>validateCatalog(input));
  }
});
test('aggregate bounds reject before reading oversized array elements and limit text bytes',()=>{
  let calls=0;const value=catalog();value.packages=new Array(CONTRACT_LIMITS.maxPackages+1);
  Object.defineProperty(value.packages,0,{get(){calls++;return catalog().packages[0];}});
  assert.throws(()=>validateCatalog(value));assert.equal(calls,0);
  const large=catalog();large.packages=Array.from({length:1000},(_,index)=>({...large.packages[0],name:`pkg-${index}`,description:'界'.repeat(1000)}));
  large.profiles=[];assert.throws(()=>validateCatalog(large),/byte/);
});
test('schema references name canonical formats without claiming arbitrary model validation',async()=>{
  const moduleSchema=JSON.parse(await readFile(new URL('../schemas/module.schema.json',import.meta.url),'utf8'));
  const catalogSchema=JSON.parse(await readFile(new URL('../schemas/catalog.schema.json',import.meta.url),'utf8'));
  const refs=JSON.parse(await readFile(new URL('../schemas/wsp-references.schema.json',import.meta.url),'utf8'));
  assert.equal(moduleSchema.properties.schema.const,CONTRACTS.module);assert.equal(catalogSchema.properties.schema.const,CONTRACTS.catalog);
  assert.equal(moduleSchema.additionalProperties,false);assert.equal(catalogSchema.additionalProperties,false);
  assert.ok(refs.enum.includes(CONTRACTS.model));assert.match(refs.description,/never a neural model/);
});
