import { performance } from 'node:perf_hooks';
import assert from 'node:assert/strict';
import { fingerprintModel, applyWeightPatch, evaluateWeightPatch } from 'ruv-fann-wsp';
import { searchWeightPatch } from 'ruv-fann-wsp-search';
// Compare identical coordinate-search proposals. Reference repeats the baseline per screen.
const model={schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:[0],bias:[0],activation:'Linear',steepness:1}]};
const data={inputs:Array.from({length:32},(_,i)=>i/31),targets:Array.from({length:32},(_,i)=>2*i/31),rows:32};
const policy={maxMSE:0,maxRegressionMSE:0}, values=Array.from({length:16},(_,i)=>(i+1)/8);
const options={locations:[{layer:0,parameter:'weights',index:0,values}],maxEvaluations:16,maxPasses:1,simd:false};
const baseHash=await fingerprintModel(model);
async function reference() {
  let best=Infinity, selected;
  for(const value of values) {
    const patch={schema:'ruv-fann.weight-patch/1',baseHash,edits:[{layer:0,parameter:'weights',index:0,value}]};
    const result=await evaluateWeightPatch(model,patch,data,policy,{simd:false});
    if(result.candidate.mse<best){best=result.candidate.mse;selected=result;}
  }
  return {evaluation:selected,inferencePasses:values.length*2};
}
const run=()=>searchWeightPatch(model,data,policy,options);
await reference();await run(); // Exclude module loading/WASM initialization from both timings.
const durations={reference:[],screened:[]};let baseline,optimized;
for(let repetition=0;repetition<9;repetition++) {
  // Alternate execution order to reduce systematic warm-cache bias.
  for(const variant of repetition%2?['screened','reference']:['reference','screened']) {
    const start=performance.now();const result=await(variant==='reference'?reference():run());
    durations[variant].push(performance.now()-start);
    if(variant==='reference')baseline=result;else optimized=result;
  }
}
assert.equal(optimized.evaluation.candidateHash,baseline.evaluation.candidateHash);
assert.equal(optimized.evaluation.candidate.mse,0);
assert.equal(optimized.inferencePasses,19);assert.equal(baseline.inferencePasses,32);
assert.equal((await applyWeightPatch(model,optimized.patch)).layers[0].weights[0],2);
const median=values=>[...values].sort((a,b)=>a-b)[Math.floor(values.length/2)];
console.log(JSON.stringify({schema:'ruv-fann.search-benchmark/1',node:process.version,backend:optimized.backend,rows:32,candidateCount:16,repetitions:9,
  equivalentCandidateHash:optimized.evaluation.candidateHash,candidateMSE:optimized.evaluation.candidate.mse,
  referenceInferencePasses:baseline.inferencePasses,screenedInferencePasses:optimized.inferencePasses,
  inferencePassReductionPercent:100*(1-optimized.inferencePasses/baseline.inferencePasses),
  referenceMedianMs:median(durations.reference),screenedMedianMs:median(durations.screened),
  note:'Identical 16-candidate monotonic one-coordinate sequence. Timing is a local observation; work reduction includes final authoritative baseline/candidate revalidation. Not an accuracy or generalization benchmark.'},null,2));
