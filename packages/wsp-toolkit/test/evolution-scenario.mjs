import { join } from 'node:path';
import { fingerprintModel } from 'ruv-fann-wsp';
import { createBooleanInvariant } from 'ruv-fann-wsp-invariants';
import { runEvolutionLoop } from 'ruv-fann-wsp-loop';
const check=(ok,message)=>{if(!ok)throw new Error(`evolution scenario: ${message}`);};
export function evolutionModels() {
  return {xor:{schema:'ruv-fann.dense-layers/1',layers:[
    {inputs:2,outputs:2,weights:[1,-1,-1,1],bias:[0,0],activation:'ReLU',steepness:1},
    {inputs:2,outputs:1,weights:[0,0],bias:[0],activation:'Linear',steepness:1}]},
  linear:{schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:[0],bias:[0],activation:'Linear',steepness:1}]}};
}
export function evolutionModules() {
  const policy={maxMSE:0,maxRegressionMSE:0,maxAbsoluteError:0};
  return [{id:'xor',development:{dataset:{inputs:[0,1],targets:[1],rows:1},policy},
    holdouts:[{name:'complete-xor',dataset:{inputs:[0,0,0,1,1,0,1,1],targets:[0,1,1,0],rows:4},policy}],
    invariants:createBooleanInvariant({inputs:2,outputs:1,targets:[0,1,1,0]}),
    strategies:[
      {name:'slice-search',kind:'search',options:{locations:[{layer:1,parameter:'weights',index:1,values:[0,1]}],maxEvaluations:4,maxPasses:1}},
      {name:'bias-shortcut',kind:'search',options:{locations:[{layer:1,parameter:'bias',index:0,values:[0,1]}],maxEvaluations:4,maxPasses:1}},
      {name:'symbolic-repair',kind:'patch',edits:[0,1].map(index=>({layer:1,parameter:'weights',index,value:1}))}
    ]},
    {id:'linear',development:{dataset:{inputs:[1],targets:[2],rows:1},policy},
      holdouts:[{name:'independent-linear',dataset:{inputs:[0,2,3],targets:[0,4,6],rows:3},policy}],
      invariants:{schema:'ruv-fann.invariants/1',cases:[{name:'operating-cases',inputs:[0,2,3],rows:3,minimum:[0],maximum:[6],expected:[0,4,6]}]},
      strategies:[
        {name:'weight-search',kind:'search',options:{locations:[{layer:0,parameter:'weights',index:0,values:[0,1,2]}],maxEvaluations:4,maxPasses:1}},
        {name:'bias-shortcut',kind:'search',options:{locations:[{layer:0,parameter:'bias',index:0,values:[0,2]}],maxEvaluations:4,maxPasses:1}}
      ]}];
}
export async function runEvolutionScenario(openRegistry,directory) {
  const filename=join(directory,'evolution.sqlite'),models=evolutionModels();
  let registry=await openRegistry(filename);
  try {
    await registry.create('xor',models.xor);await registry.create('linear',models.linear);
    const originalHashes=Object.fromEntries(await Promise.all(Object.entries(models).map(async([id,model])=>[id,await fingerprintModel(model)])));
    const report=await runEvolutionLoop({registry,modules:evolutionModules(),maxRounds:3,maxProposals:12,maxWork:100000,concurrency:3,jobTimeoutMs:5000,maxDurationMs:30000,simd:false});
    check(report.stopReason==='goals-met'&&report.promotions===2,'two independent improvements');
    check(report.workerThreadIds.length>=2,'actual concurrent worker execution');
    check(report.events.filter(e=>e.type==='rejected'&&e.reason==='regression-gate').length>=3,'independent holdouts reject development shortcuts');
    check(report.events.filter(e=>e.type==='validated').every(e=>e.invariantSuiteHash?.length===64),'invariants bind validation');
    const promoted=await Promise.all(['xor','linear'].map(id=>registry.get(id)));
    check(promoted.every(p=>p.revision===2),'each module promoted exactly once');
    await registry.close();registry=await openRegistry(filename);
    const reopened=await Promise.all(['xor','linear'].map(id=>registry.get(id)));
    check(reopened.every((p,i)=>p.revision===2&&p.hash===promoted[i].hash),'durable restart resumes exact checkpoints');
    const rollback=[];
    for(const id of ['xor','linear']) {
      const restored=await registry.rollback(id,1,2);
      check(restored.revision===3&&restored.hash===originalHashes[id],'monotonic exact rollback');
      rollback.push({id,revision:restored.revision,hash:restored.hash,historyEntries:restored.history.length});
    }
    await registry.close();registry=await openRegistry(filename);
    check((await registry.get('xor')).revision===3,'rollback persists across another restart');
    return {schema:'ruv-fann.evolution-demo/1',report,durability:{reopened:true,rollbackPersisted:true},rollback};
  } finally {await registry.close();}
}
