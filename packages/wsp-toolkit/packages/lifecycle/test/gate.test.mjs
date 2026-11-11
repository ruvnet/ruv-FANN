import test from 'node:test';
import assert from 'node:assert/strict';
import { fingerprintModel } from 'ruv-fann-wsp';
import { runRegressionGate, evaluatePatchBranches } from '../src/index.js';
const model = () => ({ schema:'ruv-fann.dense-layers/1',layers:[{ inputs:1,outputs:1,weights:[0],bias:[0],activation:'Linear',steepness:1 }] });
const data = () => ({inputs:[1,2],targets:[2,4],rows:2});
const policy = {maxMSE:0,maxRegressionMSE:0};
async function bundle(value=2) { const m=model(); return { model:m,patch:{schema:'ruv-fann.weight-patch/1',baseHash:await fingerprintModel(m),edits:[{layer:0,parameter:'weights',index:0,value}]},
  suites:[{name:'repair',dataset:data(),policy}] }; }

test('all suites gate one hash-bound candidate and reject an independent regression', async () => {
  const input=await bundle();
  const good=await runRegressionGate(input,{simd:false});
  assert.equal(good.accepted,true); assert.equal(good.suites[0].evaluation.candidate.mse,0);
  assert.equal(good.suites[0].evaluation.backend,'wasm');
  assert.equal('model' in good.suites[0].evaluation,false);
  input.suites.push({name:'holdout',dataset:{inputs:[1],targets:[0],rows:1},policy});
  const mixed=await runRegressionGate(input);
  assert.equal(mixed.accepted,false); assert.equal(mixed.suites[0].evaluation.accepted,true);
  assert.equal(mixed.suites[1].evaluation.candidateHash,mixed.candidateHash);
});
test('branch candidates remain independent of one another and of the base', async () => {
  const input=await bundle(), bad=structuredClone(input.patch); bad.edits[0].value=-1;
  const result=await evaluatePatchBranches(input.model,[bad,input.patch],data(),policy);
  assert.deepEqual(result.branches.map(b=>b.evaluation.accepted),[false,true]);
  assert.equal(input.model.layers[0].weights[0],0);
  result.branches[0].evaluation.model.layers[0].weights[0]=50;
  assert.equal(result.branches[1].evaluation.model.layers[0].weights[0],2);
});
test('snapshot isolation precedes async execution', async () => {
  const input=await bundle(); const pending=runRegressionGate(input);
  input.patch.edits[0].value=100; input.suites[0].dataset.targets[0]=-100;
  assert.equal((await pending).accepted,true);
});
test('reject malformed fields, forged base, accessors, sparse arrays and aggregate budgets', async () => {
  for (const mutate of [x=>x.extra=true,x=>x.suites=[],x=>x.suites.push(x.suites[0]),x=>x.patch.baseHash='0'.repeat(64),
    x=>x.suites[0].dataset.inputs=[1,Infinity],x=>x.suites=new Array(2),x=>Object.defineProperty(x,'patch',{get(){throw Error('getter executed');}})]) {
    const input=await bundle(); mutate(input); await assert.rejects(runRegressionGate(input));
  }
  const input=await bundle(); let calls=0; Object.defineProperty(input,'model',{get(){calls++;}});
  await assert.rejects(runRegressionGate(input),/accessors/); assert.equal(calls,0);
  const large=model(); large.layers[0]={inputs:1,outputs:4096,weights:Array(4096).fill(0),bias:Array(4096).fill(0),activation:'Linear',steepness:1};
  const patch={schema:'ruv-fann.weight-patch/1',baseHash:await fingerprintModel(large),edits:[{layer:0,parameter:'weights',index:0,value:1}]};
  await assert.rejects(runRegressionGate({model:large,patch,suites:[{name:'large',dataset:{inputs:[],targets:[],rows:1024},policy}]}),/aggregate/);
});
