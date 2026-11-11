import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { openDemoRegistry } from '../../../tooling/agentdb/runtime.mjs';
import { fingerprintModel } from 'ruv-fann-wsp';
import { createAgentDBRegistry } from 'ruv-fann-wsp-registry';
import { runEvolutionLoop } from '../src/index.js';
import { evolutionModels,evolutionModules,runEvolutionScenario } from '../../../test/evolution-scenario.mjs';
async function setup(t) {
  const dir=await mkdtemp(join(tmpdir(),'wsp-loop-')),registry=await openDemoRegistry(join(dir,'state.sqlite'));
  await registry.create('xor',evolutionModels().xor);
  t.after(async()=>{await registry.close();await rm(dir,{recursive:true,force:true});});
  return registry;
}

test('real swarm loop rejects shortcuts, persists improvements and rolls back after restart',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'wsp-loop-e2e-'));
  try {const result=await runEvolutionScenario(openDemoRegistry,dir);assert.equal(result.report.promotions,2);assert.equal(result.durability.rollbackPersisted,true);}
  finally {await rm(dir,{recursive:true,force:true});}
});
test('proposal and work budgets stop without unreviewed model changes',async t=>{
  const registry=await setup(t),before=await registry.get('xor');
  const module=evolutionModules()[0];
  const limited=await runEvolutionLoop({registry,modules:[module],maxProposals:1,maxRounds:3,simd:false});
  assert.equal(limited.stopReason,'proposal-limit');assert.equal(limited.budget.proposals,1);
  assert.equal(limited.promotions,0);assert.equal((await registry.get('xor')).hash,before.hash);
  const work=await runEvolutionLoop({registry,modules:[module],maxWork:1});
  assert.equal(work.stopReason,'work-limit');assert.equal(work.budget.proposals,0);assert.equal(work.budget.reservedWork,0);
});
test('finite-domain invariants independently reject a patch accepted by supplied MSE suites',async t=>{
  const registry=await setup(t),module=evolutionModules()[0];
  module.holdouts=[{name:'slice-agreement',...module.development}];module.strategies=[module.strategies[0]];
  const result=await runEvolutionLoop({registry,modules:[module],simd:false});
  assert.equal(result.promotions,0);assert.ok(result.events.some(e=>e.type==='rejected'&&e.reason==='invariant-gate'));
  assert.equal((await registry.get('xor')).revision,1);
});
test('configuration snapshots precede queueing and worker execution',async t=>{
  const registry=await setup(t),modules=[evolutionModules()[0]],pending=runEvolutionLoop({registry,modules,simd:false});
  modules[0].holdouts[0].dataset.targets.fill(100);modules[0].strategies[2].edits[0].value=100;
  assert.equal((await pending).promotions,1);assert.equal((await registry.get('xor')).model.layers[1].weights[0],1);
});
test('abort and deadline terminate workers and prevent new promotion',async t=>{
  const registry=await setup(t),controller=new AbortController();controller.abort();
  const pre=await runEvolutionLoop({registry,modules:[evolutionModules()[0]],signal:controller.signal});
  assert.equal(pre.stopReason,'aborted');assert.equal(pre.budget.proposals,0);
  const deadline=await runEvolutionLoop({registry,modules:[evolutionModules()[0]],maxDurationMs:1});
  assert.equal(deadline.stopReason,'deadline');assert.equal(deadline.promotions,0);assert.equal((await registry.get('xor')).revision,1);
});
test('newer revision wins when a competing coordinator changes state during proposal',async t=>{
  const registry=await setup(t);let changed=false;
  const authority={
    get:id=>registry.get(id),
    async propose(...args){if(!changed){changed=true;await registry.rollback('xor',1,1);}return registry.propose(...args);},
    promote:(...args)=>registry.promote(...args)
  };
  const result=await runEvolutionLoop({registry:authority,modules:[evolutionModules()[0]],simd:false});
  assert.equal(result.promotions,0);assert.ok(result.events.some(e=>e.type==='stale'));
  const current=await registry.get('xor');assert.equal(current.revision,2);assert.equal(current.hash,await fingerprintModel(evolutionModels().xor));assert.equal(result.modules[0].finalRevision,2);
});
test('durability failure rejects the run instead of claiming successful evolution',async t=>{
  const registry=await setup(t),failure=Object.assign(new Error('disk full'),{code:'WSP_REGISTRY_PERSISTENCE'});
  const broken={get:id=>registry.get(id),propose:(...args)=>registry.propose(...args),promote:async()=>{throw failure;}};
  await assert.rejects(runEvolutionLoop({registry:broken,modules:[evolutionModules()[0]],simd:false}),e=>e===failure);
  assert.equal((await registry.get('xor')).revision,1);
});
test('unknown code callbacks, invalid limits, malformed suites and accessors fail before dispatch',async t=>{
  const registry=await setup(t);
  for(const mutate of [m=>m.strategies[0].options.objective=()=>0,m=>m.holdouts=[],m=>m.invariants.cases[0].rows=2048,
    m=>m.holdouts[0].policy.maxRegressionMSE=-1,m=>m.strategies[0].options.maxWork=Infinity]) {
    const module=evolutionModules()[0];mutate(module);await assert.rejects(runEvolutionLoop({registry,modules:[module]}));
  }
  let calls=0;const module=evolutionModules()[0];Object.defineProperty(module,'strategies',{get(){calls++;return [];}});
  await assert.rejects(runEvolutionLoop({registry,modules:[module]}));assert.equal(calls,0);
  await assert.rejects(runEvolutionLoop({registry,modules:[evolutionModules()[0]],maxRounds:17}));
  assert.equal((await registry.get('xor')).revision,1);
});
test('continuous generations rebase proposals onto each newly persisted revision',async t=>{
  const registry=await setup(t);await registry.create('linear',evolutionModels().linear);
  const module=evolutionModules()[1];
  module.development.policy={maxMSE:10,maxRegressionMSE:0};module.holdouts[0].policy={maxMSE:10,maxRegressionMSE:0};
  delete module.invariants.cases[0].expected;
  module.strategies=[{name:'one-step',kind:'search',options:{locations:[{layer:0,parameter:'weights',index:0,values:[1,2]}],maxEvaluations:1,maxPasses:1}}];
  const result=await runEvolutionLoop({registry,modules:[module],maxRounds:3,simd:false});
  assert.equal(result.stopReason,'goals-met');assert.equal(result.rounds,2);assert.equal(result.promotions,2);
  assert.deepEqual(result.events.filter(e=>e.type==='promoted').map(e=>[e.revision,e.nextRevision]),[[1,2],[2,3]]);
  assert.equal((await registry.get('linear')).model.layers[0].weights[0],2);
});
test('two actual concurrent loops share one AgentDB CAS authority',async t=>{
  const registry=await setup(t),configuration={registry,modules:[evolutionModules()[0]],simd:false};
  const reports=await Promise.all([runEvolutionLoop(configuration),runEvolutionLoop(configuration)]);
  assert.equal(reports.reduce((sum,r)=>sum+r.promotions,0),1);
  assert.ok(reports.some(r=>r.events.some(e=>e.type==='stale')));
  const current=await registry.get('xor');assert.equal(current.revision,2);assert.deepEqual([...current.model.layers[1].weights],[1,1]);
});
test('completed goals are invalidated when another actor rolls a module back',async t=>{
  const registry=await setup(t);await registry.create('linear',evolutionModels().linear);
  const modules=evolutionModules();
  modules[0].strategies=[modules[0].strategies[2]];
  modules[1].development.policy={maxMSE:10,maxRegressionMSE:0};modules[1].holdouts[0].policy={maxMSE:10,maxRegressionMSE:0};
  delete modules[1].invariants.cases[0].expected;
  modules[1].strategies=[{name:'one-step',kind:'search',options:{locations:[{layer:0,parameter:'weights',index:0,values:[1,2]}],maxEvaluations:1,maxPasses:1}}];
  let rolled=false;
  const authority={get:id=>registry.get(id),propose:(...args)=>registry.propose(...args),async promote(...args){
    const next=await registry.promote(...args);
    if(args[0]==='linear'&&!rolled){rolled=true;await registry.rollback('xor',1,2);}
    return next;
  }};
  const result=await runEvolutionLoop({registry:authority,modules,maxRounds:3,simd:false});
  assert.equal(result.stopReason,'goals-met');assert.equal(result.promotions,4);
  assert.equal((await registry.get('xor')).revision,4);assert.deepEqual([...(await registry.get('xor')).model.layers[1].weights],[1,1]);
  assert.equal(result.modules.find(m=>m.id==='xor').finalRevision,4);
});
test('same-content external revisions update report lineage while preserving valid goal evidence',async t=>{
  const registry=await setup(t),module=evolutionModules()[0];module.strategies=[module.strategies[2]];
  const authority={get:id=>registry.get(id),propose:(...args)=>registry.propose(...args),async promote(...args){
    const next=await registry.promote(...args);await registry.rollback('xor',next.revision,next.revision);return next;
  }};
  const result=await runEvolutionLoop({registry:authority,modules:[module],simd:false});
  assert.equal(result.stopReason,'goals-met');assert.equal(result.modules[0].goalReached,true);
  assert.equal(result.modules[0].finalRevision,3);assert.equal((await registry.get('xor')).revision,3);
});
test('trusted invariant and holdout configuration fails before any proposal or mutation',async t=>{
  const registry=await setup(t);
  for(const mutate of [m=>m.holdouts[0].name='invalid:name',m=>m.invariants.cases[0].name='',
    m=>m.invariants.cases[0].expected=[0],m=>m.invariants.cases[0].minimum=[0,1],
    m=>{delete m.invariants.cases[0].expected;m.invariants.cases[0].minimum=[0];m.invariants.cases[0].tolerance=1;}]) {
    const module=evolutionModules()[0];mutate(module);
    await assert.rejects(runEvolutionLoop({registry,modules:[module],simd:false}));
  }
  const current=await registry.get('xor');assert.equal(current.revision,1);assert.equal(current.proposals.length,0);
});
test('real AgentDB promotion followed by failed disk flush never acknowledges a new checkpoint',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'wsp-loop-disk-failure-'));let registry,calls=0;
  try {
    registry=await openDemoRegistry(join(dir,'state.sqlite'),(programs,options)=>createAgentDBRegistry(programs,{...options,flush:()=>{
      if(++calls===3)throw new Error('simulated disk full after promotion transaction');return options.flush();
    }}));
    await registry.create('xor',evolutionModels().xor);
    await assert.rejects(runEvolutionLoop({registry,modules:[evolutionModules()[0]],simd:false}),error=>error.code==='WSP_REGISTRY_PERSISTENCE');
    assert.equal(calls,3);assert.equal(registry.status().poisoned,true);
    await registry.close();registry=await openDemoRegistry(join(dir,'state.sqlite'));
    const restored=await registry.get('xor');assert.equal(restored.revision,1);assert.deepEqual([...restored.model.layers[1].weights],[0,0]);
  } finally {await registry?.close();await rm(dir,{recursive:true,force:true});}
});
test('cancellation during final registry refresh preserves commits and reports cancellation',async t=>{
  const registry=await setup(t),controller=new AbortController(),module=evolutionModules()[0];
  module.strategies=[module.strategies[2]];let reads=0;
  const authority={async get(id){const value=await registry.get(id);if(++reads===3)controller.abort();return value;},
    propose:(...args)=>registry.propose(...args),promote:(...args)=>registry.promote(...args)};
  const report=await runEvolutionLoop({registry:authority,modules:[module],signal:controller.signal,simd:false});
  assert.equal(report.stopReason,'aborted');assert.equal(report.promotions,1);
  assert.equal((await registry.get('xor')).revision,2);assert.ok(report.events.some(e=>e.type==='promoted'));
});
