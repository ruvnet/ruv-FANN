import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import { fingerprintModel } from 'ruv-fann-wsp';
import { runScenario, xorModel, xorData, strictPolicy } from './scenario.mjs';
for (const simd of [false,true]) test(`complete platform-neutral WSP scenario with real WASM simd=${simd}`,async()=>{
  const result=await runScenario(simd);
  assert.equal(result.backend,simd?'wasm-simd':'wasm');
  assert.equal(result.baselineMSE,0.5);
  assert.equal(result.candidateMSE,0);
});
test('checkpoint CLI returns distinct acceptance, rejection and malformed statuses', async()=>{
  const directory=await mkdtemp(join(tmpdir(),'wsp-gate-'));
  try {
    const model=xorModel(), baseHash=await fingerprintModel(model);
    const bundle={model,patch:{schema:'ruv-fann.weight-patch/1',baseHash,edits:[0,1].map(index=>({layer:1,parameter:'weights',index,value:1}))},suites:[{name:'xor',dataset:xorData(),policy:strictPolicy()}]};
    const file=join(directory,'gate.json'), cli=new URL('../packages/lifecycle/src/cli.js',import.meta.url);
    await writeFile(file,JSON.stringify(bundle));
    const accepted=JSON.parse(execFileSync(process.execPath,[cli.pathname,file],{encoding:'utf8'}));
    assert.equal(accepted.accepted,true);
    bundle.patch.edits[1].value=0;
    await writeFile(file,JSON.stringify(bundle));
    const rejected=spawnSync(process.execPath,[cli.pathname,file],{encoding:'utf8'});
    assert.equal(rejected.status,1);assert.equal(JSON.parse(rejected.stdout).accepted,false);
    const fifo=join(directory,'pipe');
    execFileSync('mkfifo',[fifo]);
    const nonregular=spawnSync(process.execPath,[cli.pathname,fifo],{timeout:2000,encoding:'utf8'});
    assert.equal(nonregular.status,2);assert.match(nonregular.stderr,/regular file/);
    await writeFile(file,'{');
    assert.equal(spawnSync(process.execPath,[cli.pathname,file]).status,2);
  } finally { await rm(directory,{recursive:true,force:true}); }
});
