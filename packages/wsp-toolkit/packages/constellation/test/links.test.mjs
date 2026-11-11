import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
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
