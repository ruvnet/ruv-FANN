import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir,mkdtemp,symlink,rm} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {readCatalog} from '../src/index.js';

test('link preview covers published local packages and keeps npm README links absolute without writing files',async()=>{
  const catalog=await readCatalog(),entries=catalog.packages.filter(pkg=>pkg.directory?.startsWith('packages/wsp-toolkit/packages/'));
  assert.equal(entries.length,12);assert.ok(entries.every(pkg=>pkg.status==='published'));
  const files=entries.flatMap(pkg=>['README.md','package.json','constellation.json'].map(file=>new URL(`../../../../../${pkg.directory}/${file}`,import.meta.url)));
  const before=await Promise.all(files.map(file=>readFile(file,'utf8')));
  const script=new URL('../../../scripts/constellation-links.mjs',import.meta.url);
  const preview=JSON.parse(execFileSync(process.execPath,[script.pathname,'--print'],{encoding:'utf8'}));
  assert.equal(preview.mode,'print');assert.equal(preview.packages,12);assert.equal(preview.blocks.length,12);
  assert.deepEqual(new Set(preview.blocks.map(block=>block.name)),new Set(entries.map(pkg=>pkg.name)));
  for(const block of preview.blocks){
    assert.match(block.readmeBlock,/https:\/\/www.npmjs.com\/package\/ruvnet-constellation/);
    assert.match(block.readmeBlock,/https:\/\/ruvnet.github.io\/ruv-FANN\//);
    assert.match(block.readmeBlock,/independent verification/);
    assert.doesNotMatch(block.readmeBlock,/\]\((?:\.\.?\/)/);
  }
  assert.deepEqual(await Promise.all(files.map(file=>readFile(file,'utf8'))),before);
});

test('metadata refresh preserves published catalog pins and README bytes while checking the exact local module version',async t=>{
  const directory=await mkdtemp(join(tmpdir(),'constellation-links-'));
  t.after(()=>rm(directory,{recursive:true,force:true}));
  const workspace=join(directory,'packages/wsp-toolkit'),scripts=join(workspace,'scripts');
  const local=join(workspace,'packages/contracts'),catalogDir=join(workspace,'packages/constellation');
  await Promise.all([mkdir(scripts,{recursive:true}),mkdir(local,{recursive:true}),mkdir(catalogDir,{recursive:true}),mkdir(join(workspace,'node_modules'),{recursive:true})]);
  await symlink(new URL('../../contracts',import.meta.url),join(workspace,'node_modules/ruvnet-constellation-contracts'),'dir');
  const source=new URL('../../../scripts/constellation-links.mjs',import.meta.url),script=join(scripts,'constellation-links.mjs');
  await writeFile(script,await readFile(source));
  const catalog=await readCatalog(),entry=catalog.packages.find(pkg=>pkg.name==='ruvnet-constellation-contracts');
  const catalogBytes=JSON.stringify({...catalog,packages:[entry],profiles:[]});
  await writeFile(join(catalogDir,'catalog.json'),catalogBytes);
  const originalReadme=await readFile(new URL('../../contracts/README.md',import.meta.url),'utf8');
  const manifest=JSON.parse(await readFile(new URL('../../contracts/package.json',import.meta.url),'utf8'));
  const module=JSON.parse(await readFile(new URL('../../contracts/constellation.json',import.meta.url),'utf8'));
  const run=(...args)=>JSON.parse(execFileSync(process.execPath,[script,...args],{encoding:'utf8',stdio:['ignore','pipe','pipe']}));
  await writeFile(join(local,'README.md'),originalReadme);
  for(const version of ['0.1.0','0.1.1']) {
    await writeFile(join(local,'package.json'),JSON.stringify({...manifest,version}));
    await writeFile(join(local,'constellation.json'),JSON.stringify({...module,version},null,2)+'\n');
    assert.equal(run('--check').changes,0);
    assert.equal(await readFile(join(catalogDir,'catalog.json'),'utf8'),catalogBytes);
    assert.equal(await readFile(join(local,'README.md'),'utf8'),originalReadme);
  }
  await writeFile(join(local,'constellation.json'),JSON.stringify({...module,version:'0.1.0'},null,2)+'\n');
  assert.throws(()=>run('--check'),/metadata\/README blocks need regeneration/);
  assert.equal(run().changes,1);
  assert.deepEqual(JSON.parse(await readFile(join(local,'constellation.json'),'utf8')),{...module,version:'0.1.1'});
  assert.equal(await readFile(join(local,'README.md'),'utf8'),originalReadme);
  assert.equal(await readFile(join(catalogDir,'catalog.json'),'utf8'),catalogBytes);
  assert.equal(run('--check').changes,0);
  await writeFile(join(local,'package.json'),JSON.stringify({...manifest,version:'0.1.2'}));
  assert.throws(()=>run('--print'),/catalog package identity differs from source/);
  await writeFile(join(local,'package.json'),JSON.stringify({...manifest,name:'wrong-name',version:'0.1.1'}));
  assert.throws(()=>run('--print'),/catalog package identity differs from source/);
});
