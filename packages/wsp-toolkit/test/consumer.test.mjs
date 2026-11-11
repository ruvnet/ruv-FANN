import test from 'node:test';
import assert from 'node:assert/strict';
import { verifyPackedConsumer } from '../scripts/packed-consumer.mjs';
test('canonical core or twelve development archives install, typecheck and run in a fresh consumer', {timeout:60000},async()=>{
  const result=await verifyPackedConsumer();
  assert.equal(result.archives.length,process.env.WSP_TOOLKIT_CANDIDATE?1:12);
  if(process.env.WSP_TOOLKIT_CANDIDATE)assert.deepEqual(result.archives.map(({name,version})=>({name,version})),[{name:'ruv-fann',version:'0.3.0'}]);
  assert.deepEqual(result.results.map(r=>r.backend),['wasm','wasm-simd']);
  assert.ok(result.results.every(r=>r.candidateMSE===0 && r.regressionRejected));
  assert.equal(result.evolution.report.promotions,2);
  assert.equal(result.constellation.catalogEntries,24);assert.equal(result.constellation.scaffoldTests,3);
  assert.equal(result.lightweight.mlStackInstalled,false);assert.equal(result.lightweight.packageCount,2);
  assert.equal(result.evolution.durability.rollbackPersisted,true);
  assert.ok(result.archives.every(a=>/^[a-f0-9]{64}$/.test(a.sha256)));
});
