import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fingerprintModel, evaluateWeightPatch } from 'ruv-fann-wsp';
import { runProposalSwarm, estimateSwarmWork, SWARM_LIMITS } from '../src/index.js';
const makeModel = () => ({ schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 2, outputs: 2, weights: [1,-1,-1,1], bias:[0,0], activation:'ReLU', steepness:1 },
  { inputs: 2, outputs: 1, weights: [0,0], bias:[0], activation:'Linear', steepness:1 }
] });
const dataset = () => ({inputs:[0,0,0,1,1,0,1,1], targets:[0,1,1,0], rows:4});
const policy = () => ({maxMSE:0,maxRegressionMSE:0,maxAbsoluteError:0});
const job = (id='search',simd=false) => ({id,kind:'search',model:makeModel(),dataset:dataset(),policy:policy(),
  searchOptions:{locations:[0,1].map(index=>({layer:1,parameter:'weights',index,values:[0,1]})),maxEvaluations:8,maxPasses:2,simd}});
const patchJob = id => ({id,kind:'patch',patch:{schema:'ruv-fann.weight-patch/1',baseHash:'a'.repeat(64),edits:[{layer:0,parameter:'weights',index:0,value:2}]}});
function concurrency(results) {
  const events=results.flatMap(r=>r.startedAtMs===undefined?[]:[{time:r.startedAtMs,delta:1},{time:r.finishedAtMs,delta:-1}]);
  events.sort((a,b)=>a.time-b.time||a.delta-b.delta);
  let active=0,max=0;
  for(const event of events){active+=event.delta;max=Math.max(max,active);}
  assert.equal(active,0);
  return max;
}

test('real worker threads overlap within configured concurrency and repair XOR on scalar/SIMD WASM', {timeout:20000}, async()=>{
  const jobs=[job('scalar',false),job('simd',true),job('scalar-next',false),job('simd-next',true)];
  assert.equal(estimateSwarmWork(jobs),4*396);
  const results=await runProposalSwarm(jobs,{concurrency:2,maxTotalWork:4*396});
  assert.deepEqual(results.map(r=>r.id),jobs.map(j=>j.id));
  assert.ok(results.every(r=>r.status==='ok'),JSON.stringify(results));
  assert.equal(new Set(results.map(r=>r.workerThreadId)).size,4);
  assert.ok(results.every(r=>Number.isInteger(r.workerThreadId)&&r.workerThreadId>0));
  assert.equal(concurrency(results),2);
  assert.deepEqual(results.map(r=>r.search.backend),['wasm','wasm-simd','wasm','wasm-simd']);
  for(let i=0;i<results.length;i++){
    assert.equal(results[i].search.reservedWork,396);
    const evaluation=await evaluateWeightPatch(jobs[i].model,results[i].patch,jobs[i].dataset,jobs[i].policy,{simd:false});
    assert.equal(evaluation.candidate.mse,0);
    assert.equal(evaluation.accepted,true);
  }
});

test('an already optimal checkpoint returns a successful worker with no mutation proposal',async()=>{
  const input=job('optimal');input.model.layers[1].weights=[1,1];
  const [result]=await runProposalSwarm([input]);
  assert.equal(result.status,'ok');assert.equal(result.patch,null);
  assert.equal(result.search.inferencePasses,result.search.evaluations+1);
});

test('all caller job and configuration data are snapshotted before yielding',async()=>{
  const input=[job('owned'),job('owned-next')], options={concurrency:1};
  const baseHash=await fingerprintModel(input[0].model);
  const pending=runProposalSwarm(input,options);
  input[0].id='mutated';input[0].model.layers[0].weights[0]=1000;
  input[0].dataset.targets.fill(10);input[0].searchOptions.locations[0].values[1]=500;
  options.concurrency=4;
  const results=await pending;
  assert.deepEqual(results.map(r=>r.id),['owned','owned-next']);
  assert.ok(results.every(r=>r.status==='ok'));
  assert.equal(results[0].patch.baseHash,baseHash);
  assert.deepEqual(results[0].patch.edits.map(e=>e.value),[1,1]);
  assert.equal(concurrency(results),1);
});

test('patch workers only transport snapshots and cannot authorize the supplied hash',async()=>{
  const input=patchJob('transport'), pending=runProposalSwarm([input],{maxTotalWork:0});
  input.patch.edits[0].value=20;
  const [result]=await pending;
  assert.equal(result.status,'ok');assert.equal(result.patch.edits[0].value,2);
  assert.equal(estimateSwarmWork([patchJob('zero')]),0);
  await assert.rejects(evaluateWeightPatch(makeModel(),result.patch,dataset(),policy()),/baseHash|layer|index/);
});

test('numerical worker failures remain bounded results and do not poison following jobs',async()=>{
  const overflow=job('overflow');
  overflow.model={schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:[1e308],bias:[0],activation:'Linear',steepness:1}]};
  overflow.dataset={inputs:[2],targets:[0],rows:1};
  overflow.searchOptions={locations:[{layer:0,parameter:'bias',index:0,values:[1]}],maxEvaluations:1,maxPasses:1};
  const results=await runProposalSwarm([overflow,patchJob('next')],{concurrency:1});
  assert.equal(results[0].status,'error');assert.ok(results[0].error.length<=512);
  assert.ok(results[0].error.length>0);
  assert.equal(results[1].status,'ok');
});

test('per-worker timeout stays responsive and cleans up before resolving',async()=>{
  let tick=false;const timer=setTimeout(()=>{tick=true;},0);
  const results=await runProposalSwarm([job('deadline'),job('deadline-next')],{concurrency:1,timeoutMs:1});
  clearTimeout(timer);
  assert.equal(tick,true);
  assert.ok(results.every(r=>r.status==='timeout'),JSON.stringify(results));
  assert.ok(results.every(r=>r.workerThreadId>0&&r.finishedAtMs>=r.startedAtMs));
  assert.equal(concurrency(results),1);
  // A new worker still runs after terminated searches.
  assert.equal((await runProposalSwarm([patchJob('after-timeout')]))[0].status,'ok');
});

test('abort terminates active workers, never starts queued jobs and ignores late work',async()=>{
  const controller=new AbortController();
  const jobs=Array.from({length:6},(_,i)=>job(`abort-${i}`));
  const pending=runProposalSwarm(jobs,{concurrency:2,signal:controller.signal});
  controller.abort();
  const results=await pending;
  assert.ok(results.every(r=>r.status==='cancelled'),JSON.stringify(results));
  assert.equal(results.filter(r=>r.workerThreadId!==undefined).length,2);
  assert.ok(results.slice(2).every(r=>r.startedAtMs===undefined));
  const already=new AbortController();already.abort();
  const queued=await runProposalSwarm([patchJob('never-started')],{signal:already.signal});
  assert.equal(queued[0].status,'cancelled');assert.equal(queued[0].workerThreadId,undefined);
});

test('native abort state cannot be hidden by shadow getters or forged signal prototypes',async()=>{
  const controller=new AbortController();controller.abort();let getters=0;
  Object.defineProperty(controller.signal,'aborted',{get(){getters++;return false;}});
  Object.defineProperty(controller.signal,'addEventListener',{get(){getters++;throw new Error('shadow event getter');}});
  const [result]=await runProposalSwarm([patchJob('pre-aborted')],{signal:controller.signal});
  assert.equal(result.status,'cancelled');assert.equal(result.workerThreadId,undefined);assert.equal(getters,0);
  await assert.rejects(runProposalSwarm([patchJob('forged')],{signal:Object.create(AbortSignal.prototype)}),/signal|AbortSignal/);
});

test('unknown operations/options, executable values, accessors and malformed data reject without executing getters',async()=>{
  let invoked=0;
  for(const mutate of [
    x=>x.kind='shell', x=>x.command='touch should-not-exist', x=>x.searchOptions.objective=()=>{invoked++;return 0;},
    x=>x.searchOptions.objective=0, x=>x.searchOptions.locations[0].values=[1,Infinity],
    x=>x.searchOptions.locations.push(x.searchOptions.locations[0]),
    x=>x.model.layers[0].activation='Unknown',x=>x.model.layers[1].inputs=3,
    x=>x.dataset.inputs=new Array(8),x=>x.dataset.rows=1025,
    x=>Object.defineProperty(x,'model',{get(){invoked++;return makeModel();}})
  ]){
    const input=job();mutate(input);await assert.rejects(runProposalSwarm([input]));
  }
  const array=[job()];array[Symbol.iterator]=function*(){invoked++;yield job();};
  await assert.rejects(runProposalSwarm(array));
  for(const options of [{concurrency:5},{concurrency:0},{timeoutMs:0},{timeoutMs:60001},{maxTotalWork:100000001},{signal:{}},{code:'1+1'}])
    await assert.rejects(runProposalSwarm([job()],options));
  await assert.rejects(runProposalSwarm([job('duplicate'),job('duplicate')]));
  await assert.rejects(runProposalSwarm([]));
  await assert.rejects(runProposalSwarm(Array.from({length:65},(_,i)=>patchJob(`too-many-${i}`))));
  assert.equal(invoked,0);
});

test('aggregate byte and inference budgets reject before work can start',async()=>{
  await assert.rejects(runProposalSwarm([job()],{maxTotalWork:395}),/aggregate work/);
  const huge=Array.from({length:11},(_,i)=>({...patchJob(`large-${i}`),padding:'x'.repeat(1024*1024)}));
  await assert.rejects(runProposalSwarm(huge),/byte limit/);
  const many=Array.from({length:64},(_,i)=>({
    id:`work-${i}`,kind:'search',
    model:{schema:'ruv-fann.dense-layers/1',layers:[
      {inputs:1,outputs:256,weights:Array(256).fill(0),bias:Array(256).fill(0),activation:'Linear',steepness:1},
      {inputs:256,outputs:1,weights:Array(256).fill(0),bias:[0],activation:'Linear',steepness:1}
    ]},
    dataset:{inputs:Array(512).fill(0),targets:Array(512).fill(0),rows:512},policy:policy(),
    searchOptions:{locations:[{layer:0,parameter:'weights',index:0,values:[0,1]}],maxEvaluations:1,maxPasses:1}
  }));
  assert.throws(()=>estimateSwarmWork(many),/aggregate work/);
  assert.equal(SWARM_LIMITS.maxConcurrency,4);
});

test('packed-consumer style node --input-type=module -e cannot leak flags into file workers', {timeout:15000},()=>{
  const source=`import {runProposalSwarm} from ${JSON.stringify(new URL('../src/index.js',import.meta.url).href)}; console.log(JSON.stringify(await runProposalSwarm(${JSON.stringify([job('subprocess')])})));`;
  const results=JSON.parse(execFileSync(process.execPath,['--input-type=module','-e',source],{encoding:'utf8',timeout:10000}));
  assert.equal(results[0].status,'ok',JSON.stringify(results));
  assert.equal(results[0].search.backend,'wasm');
});
