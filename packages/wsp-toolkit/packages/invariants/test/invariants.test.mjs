import test from 'node:test';
import assert from 'node:assert/strict';
import { fingerprintModel } from 'ruv-fann-wsp';
import { checkModelInvariants, createBooleanInvariant, estimateInvariantWork, INVARIANT_LIMITS } from '../src/index.js';
const linear = (weight = 2, bias = 0) => ({schema:'ruv-fann.dense-layers/1',layers:[
  {inputs:1,outputs:1,weights:[weight],bias:[bias],activation:'Linear',steepness:1}
]});
const xor = (weights = [1,1]) => ({schema:'ruv-fann.dense-layers/1',layers:[
  {inputs:2,outputs:2,weights:[1,-1,-1,1],bias:[0,0],activation:'ReLU',steepness:1},
  {inputs:2,outputs:1,weights,bias:[0],activation:'Linear',steepness:1}
]});
const suite = item => ({schema:'ruv-fann.invariants/1',cases:[item]});
const affineCase = () => ({name:'affine',inputs:[0,1,2],rows:3,minimum:[0],maximum:[4],expected:[0,2,4]});

for (const simd of [false,true]) test(`full Boolean-domain oracle and independent regression rejection (SIMD=${simd})`, async () => {
  const invariant = createBooleanInvariant({inputs:2,outputs:1,targets:[0,1,1,0]});
  assert.deepEqual(invariant.cases[0].inputs,[0,0,0,1,1,0,1,1]);
  const accepted = await checkModelInvariants(xor(),invariant,{simd});
  assert.equal(accepted.accepted,true);
  assert.equal(accepted.backend,simd?'wasm-simd':'wasm');
  assert.equal(accepted.checks,4); assert.equal(accepted.violationCount,0);
  assert.equal(accepted.work,36); assert.equal(accepted.modelHash,await fingerprintModel(xor()));
  const failed = await checkModelInvariants(xor([1,0]),invariant,{simd});
  assert.equal(failed.accepted,false);
  assert.deepEqual(failed.violations,[{case:'boolean-domain',row:1,output:0,constraint:'expected',actual:0,bound:1}]);
  assert.equal(failed.suiteHash,accepted.suiteHash);
});

test('affine bounds are inclusive; expected tolerance and multi-output batch ordering match an independent oracle', async () => {
  const model={schema:'ruv-fann.dense-layers/1',layers:[{inputs:2,outputs:2,weights:[2,-1,0.5,3],bias:[1,-2],activation:'Linear',steepness:1}]};
  const inputs=Array.from({length:66},(_,index)=>index%7-3);
  const expected=Array.from({length:33},(_,row)=>[2*inputs[2*row]-inputs[2*row+1]+1,0.5*inputs[2*row]+3*inputs[2*row+1]-2]).flat();
  const minimum=[Math.min(...expected.filter((_,i)=>i%2===0)),Math.min(...expected.filter((_,i)=>i%2===1))];
  const maximum=[Math.max(...expected.filter((_,i)=>i%2===0)),Math.max(...expected.filter((_,i)=>i%2===1))];
  const result=await checkModelInvariants(model,suite({name:'affine-oracle',inputs,rows:33,minimum,maximum,expected,tolerance:0}),{simd:false});
  assert.equal(result.accepted,true); assert.equal(result.checks,198); assert.equal(result.work,198);
  const tolerant=await checkModelInvariants(linear(),suite({...affineCase(),expected:[0,2.001,4],tolerance:0.0011}));
  assert.equal(tolerant.accepted,true);
  const failed=await checkModelInvariants(linear(),suite({...affineCase(),expected:[0,2.001,4],tolerance:0.0009}));
  assert.equal(failed.accepted,false);assert.equal(failed.violationCount,1);
});

test('independent cases catch a candidate whose development slice contains no weight information', async () => {
  const development=suite({name:'development',inputs:[0],rows:1,expected:[0]});
  assert.equal((await checkModelInvariants(linear(99),development)).accepted,true);
  const holdout=suite({name:'holdout',inputs:[1],rows:1,minimum:[1.9],maximum:[2.1]});
  const result=await checkModelInvariants(linear(99),holdout);
  assert.equal(result.accepted,false); assert.equal(result.violations[0].constraint,'maximum');
});

test('suite/model hashes normalize key order, absent zero bias, negative zero and default tolerance', async () => {
  const first=linear(); const second={layers:[{activation:'Linear',steepness:1,weights:[2],outputs:1,inputs:1}],schema:'ruv-fann.dense-layers/1'};
  const a=await checkModelInvariants(first,suite(affineCase()));
  const b=await checkModelInvariants(second,{cases:[{expected:[-0,2,4],maximum:[4],minimum:[-0],rows:3,inputs:[-0,1,2],name:'affine',tolerance:0}],schema:'ruv-fann.invariants/1'});
  assert.equal(a.modelHash,b.modelHash);assert.equal(a.suiteHash,b.suiteHash);
  const changed=await checkModelInvariants(first,suite({...affineCase(),expected:[0,2,3.9]}));
  assert.notEqual(a.suiteHash,changed.suiteHash);
});

test('all caller data is snapshotted before asynchronous work starts', async () => {
  const model=linear(), value=suite(affineCase()), options={simd:false};
  const pending=checkModelInvariants(model,value,options);
  model.layers[0].weights[0]=100;value.cases[0].inputs[1]=100;value.cases[0].expected[0]=100;options.simd=true;
  const result=await pending;assert.equal(result.accepted,true);assert.equal(result.backend,'wasm');
  assert.equal(result.modelHash,await fingerprintModel(linear()));
});

test('violation retention is bounded while every scalar comparison is checked', async () => {
  const result=await checkModelInvariants(linear(0),suite({name:'many',inputs:Array(1024).fill(0),rows:1024,expected:Array(1024).fill(1)}));
  assert.equal(result.accepted,false);assert.equal(result.checks,1024);assert.equal(result.violationCount,1024);
  assert.equal(result.violations.length,INVARIANT_LIMITS.maxViolations);assert.equal(result.truncatedViolations,true);
  assert.equal(JSON.parse(JSON.stringify(result)).violationCount,1024);
});

test('plain data validation rejects unknown fields, getters and hostile arrays without invoking them', async () => {
  let calls=0;
  const withGetter=suite(affineCase());Object.defineProperty(withGetter.cases[0],'inputs',{get(){calls++;return [0,1,2];}});
  await assert.rejects(checkModelInvariants(linear(),withGetter),/accessors/);
  const values=[0,1,2];Object.defineProperty(values,'0',{get(){calls++;return 0;}});
  await assert.rejects(checkModelInvariants(linear(),suite({...affineCase(),inputs:values})),/dense data/);
  const model=linear();Object.defineProperty(model.layers[0],'weights',{get(){calls++;return [2];}});
  await assert.rejects(checkModelInvariants(model,suite(affineCase())),/accessors/);
  const unknown=[0,1,2];unknown[Symbol.iterator]=function*(){calls++;yield 0;};
  await assert.rejects(checkModelInvariants(linear(),suite({...affineCase(),inputs:unknown})),/unknown array/);
  assert.equal(calls,0);
  await assert.rejects(checkModelInvariants(linear(),{...suite(affineCase()),provider:'remote'}),/unknown/);
  await assert.rejects(checkModelInvariants(linear(),suite(affineCase()),{simd:false,maxWork:Infinity}),/unknown/);
  await assert.rejects(checkModelInvariants(linear(),suite({...affineCase(),inputs:new Array(3)})),/dense data/);
});

test('malformed constraints, dimensions and nonfinite data fail closed', async () => {
  for(const item of [
    {name:'empty',inputs:[0],rows:1}, {...affineCase(),minimum:[5]}, {...affineCase(),maximum:[Infinity]},
    {...affineCase(),expected:[0]}, {...affineCase(),minimum:[0,1]}, {...affineCase(),inputs:[0,NaN,2]},
    {...affineCase(),tolerance:-1}, {name:'tol',inputs:[0],rows:1,minimum:[0],tolerance:1},
    {...affineCase(),name:'invalid name'}, {...affineCase(),rows:0}, {...affineCase(),expected:null}
  ])await assert.rejects(checkModelInvariants(linear(),suite(item)));
  await assert.rejects(checkModelInvariants(linear(),{schema:'ruv-fann.invariants/1',cases:[affineCase(),affineCase()]}),/unique/);
  await assert.rejects(checkModelInvariants(linear(),{schema:'ruv-fann.invariants/1',cases:[]}),/cases/);
  await assert.rejects(checkModelInvariants(linear(),suite(affineCase()),{simd:'yes'}),/simd/);
  await assert.rejects(checkModelInvariants({...linear(),schema:'other'},suite(affineCase())),/schema/);
});

test('aggregate rows, elements and work reject before reading adversarial input elements', async () => {
  let calls=0;const hostile=[0];Object.defineProperty(hostile,'0',{get(){calls++;return 0;}});
  const cases=Array.from({length:16},(_,i)=>({name:`case${i}`,inputs:hostile,rows:65,minimum:[0]}));
  await assert.rejects(checkModelInvariants(linear(),{schema:'ruv-fann.invariants/1',cases}),/aggregate row/);
  const wide={schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:10000,weights:Array(10000).fill(0),bias:Array(10000).fill(0),activation:'Linear',steepness:1}]};
  await assert.rejects(checkModelInvariants(wide,suite({name:'work',inputs:hostile,rows:501,minimum:[0]})),/aggregate work/);
  const inputWide={schema:'ruv-fann.dense-layers/1',layers:[{inputs:128,outputs:1,weights:Array(128).fill(0),bias:[0],activation:'Linear',steepness:1}]};
  const elementCases=Array.from({length:4},(_,i)=>({name:`elements${i}`,inputs:hostile,rows:256,minimum:[0]}));
  await assert.rejects(checkModelInvariants(inputWide,{schema:'ruv-fann.invariants/1',cases:elementCases}),/aggregate element/);
  assert.equal(calls,0);
});

test('canonical model and suite byte limits reject finite but oversized data', async () => {
  const long=Number.MAX_VALUE;
  const model={schema:'ruv-fann.dense-layers/1',layers:[{inputs:65535,outputs:1,weights:Array(65535).fill(long),bias:[0],activation:'Linear',steepness:1}]};
  await assert.rejects(checkModelInvariants(model,suite({name:'unused',inputs:[0],rows:1,expected:[0]})),/model exceeds.*byte/);
  const dense={schema:'ruv-fann.dense-layers/1',layers:[{inputs:64,outputs:64,weights:Array(4096).fill(0),bias:Array(64).fill(0),activation:'Linear',steepness:1}]};
  await assert.rejects(checkModelInvariants(dense,suite({name:'large',inputs:Array(65536).fill(long),rows:1024,expected:Array(65536).fill(long)})),/suite exceeds byte/);
});

test('overflow rejects, disposes prepared inference and subsequent checks stay healthy', async () => {
  await assert.rejects(checkModelInvariants(linear(Number.MAX_VALUE,Number.MAX_VALUE),suite({name:'overflow',inputs:[1],rows:1,minimum:[0]})),/finite|overflow/);
  await assert.rejects(checkModelInvariants(linear(Number.MAX_VALUE),suite({name:'error-overflow',inputs:[1],rows:1,expected:[-Number.MAX_VALUE]})),/finite/);
  assert.equal((await checkModelInvariants(linear(),suite(affineCase()))).accepted,true);
});

test('Boolean helper enforces a complete small binary domain and owns caller targets', () => {
  const targets=[0,1,1,0];const value=createBooleanInvariant({inputs:2,outputs:1,targets});targets[0]=1;
  assert.equal(value.cases[0].expected[0],0);
  for(const table of [{inputs:0,outputs:1,targets:[0]},{inputs:7,outputs:1,targets:[]},{inputs:1,outputs:17,targets:[]},
    {inputs:2,outputs:1,targets:[0,1]},{inputs:1,outputs:1,targets:[0,0.5]},{inputs:1,outputs:1,targets:[0,1],extra:true},
    {inputs:1,outputs:1,targets:[0,1],tolerance:Infinity}])assert.throws(()=>createBooleanInvariant(table));
  let calls=0;const hostile=[0,1];Object.defineProperty(hostile,'0',{get(){calls++;return 0;}});
  assert.throws(()=>createBooleanInvariant({inputs:1,outputs:1,targets:hostile}));assert.equal(calls,0);
});


test('synchronous preflight reserves exact work and rejects malformed trusted criteria before inference', () => {
  const cases = {schema:'ruv-fann.invariants/1',cases:[affineCase(),{name:'second',inputs:[3,4],rows:2,minimum:[0]}]};
  assert.equal(estimateInvariantWork(linear(),cases),10); // one weight + one bias, five total rows
  // This model overflows actual inference, but planning performs no inference or metric calculation.
  assert.equal(estimateInvariantWork(linear(Number.MAX_VALUE,Number.MAX_VALUE),suite({name:'plan-only',inputs:[1],rows:1,expected:[0]})),2);
  for(const item of [{...affineCase(),name:''},{...affineCase(),name:'bad:name'},
    {...affineCase(),expected:[0,2]},{...affineCase(),minimum:[0,1]},
    {...affineCase(),tolerance:-1},{name:'unbound',inputs:[1],rows:1},
    {name:'tolerance-only',inputs:[1],rows:1,minimum:[0],tolerance:1}])
    assert.throws(()=>estimateInvariantWork(linear(),suite(item)));
  assert.throws(()=>estimateInvariantWork(linear(),{schema:'ruv-fann.invariants/1',cases:[affineCase(),affineCase()]}),/unique/);
  const wide={schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:10000,weights:Array(10000).fill(0),bias:Array(10000).fill(0),activation:'Linear',steepness:1}]};
  assert.throws(()=>estimateInvariantWork(wide,suite({name:'work',inputs:[],rows:501,minimum:[]})),/aggregate work/);
});
