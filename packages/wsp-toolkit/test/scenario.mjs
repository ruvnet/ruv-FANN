import { createWeightWorkspace, traceWeightModel, analyzeWeightSensitivity, createNeuronAblationPatch, applyWeightPatch } from 'ruv-fann-wsp';
import { runRegressionGate, evaluatePatchBranches } from 'ruv-fann-wsp-lifecycle';
import { searchWeightPatch, compileBooleanTable, createHebbianPatch } from 'ruv-fann-wsp-search';
import { loadHubModel } from 'ruv-fann-wsp-huggingface';
import { runWasmCanary } from 'ruv-fann-wsp-ruos';
export const xorModel = () => ({ schema: 'ruv-fann.dense-layers/1', layers: [
  {inputs:2,outputs:2,weights:[1,-1,-1,1],bias:[0,0],activation:'ReLU',steepness:1},
  {inputs:2,outputs:1,weights:[0,0],bias:[0],activation:'Linear',steepness:1}
] });
export const xorData = () => ({inputs:[0,0,0,1,1,0,1,1],targets:[0,1,1,0],rows:4});
export const strictPolicy = () => ({maxMSE:0,maxRegressionMSE:0,maxAbsoluteError:0});
const check = (ok, message) => { if (!ok) throw new Error(message); };
export async function runScenario(simd=false) {
  const model=xorModel(), data=xorData(), policy=strictPolicy();
  const search=await searchWeightPatch(model,data,policy,{simd,locations:[0,1].map(index=>({layer:1,parameter:'weights',index,values:[0,1]})),maxEvaluations:8,maxPasses:2});
  check(search.evaluation?.accepted && search.evaluation.candidate.mse===0,'search must repair XOR');
  check(search.patch.edits.length===2,'two surgical weight edits');
  const gate=await runRegressionGate({model,patch:search.patch,suites:[{name:'exhaustive-xor',dataset:data,policy}]},{simd});
  check(gate.accepted,'gate acceptance');
  const workspace=await createWeightWorkspace(model,{simd});
  const proposal=await workspace.propose(search.patch,data,policy);
  const repaired=workspace.promote(proposal.id,1);
  const trace=await traceWeightModel(repaired.model,{inputs:data.inputs,rows:data.rows},{simd});
  check(trace.layers[1].activations.every((x,i)=>x===data.targets[i]),'independent XOR oracle');
  const sensitivity=await analyzeWeightSensitivity(repaired.model,data,[{layer:1,parameter:'weights',index:0}],{epsilon:0.001,simd});
  check(Math.abs(sensitivity.entries[0].derivative)<1e-12,'optimum sensitivity');
  const ablation=await createNeuronAblationPatch(repaired.model,0,0);
  const rejected=await workspace.propose(ablation,data,policy);
  check(!rejected.evaluation.accepted,'ablation regression rejected');
  let denied=false; try {workspace.promote(rejected.id,2);} catch {denied=true;}
  check(denied && workspace.snapshot().hash===repaired.hash,'rejection preserves live state');
  const branches=await evaluatePatchBranches(repaired.model,[ablation,{...ablation,edits:ablation.edits.map(e=>({...e,value:1}))}],data,policy,{simd});
  check(!branches.branches[0].evaluation.accepted && branches.branches[1].evaluation.accepted,'forks isolated');
  const hebbian=await createHebbianPatch(repaired.model,{inputs:data.inputs,rows:data.rows},{layer:1,indices:[0],learningRate:0.1,simd,limits:{maxAbsDelta:0.01}});
  const proposedPlasticity=await applyWeightPatch(repaired.model,hebbian,{maxAbsDelta:0.01});
  check(Math.abs(proposedPlasticity.layers[1].weights[0]-1)<=0.0100000001,'bounded plasticity proposal');
  check(workspace.snapshot().hash===repaired.hash,'plasticity does not promote');
  const table=compileBooleanTable({inputs:2,outputs:1,targets:data.targets});
  const compiled=await traceWeightModel(table,{inputs:data.inputs,rows:4},{simd});
  check(compiled.layers.at(-1).activations.every((v,i)=>v===data.targets[i]),'truth table compiler oracle');
  const bytes=new TextEncoder().encode(JSON.stringify(repaired.model));
  const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
  const loaded=await loadHubModel({repo:'fixture/xor',revision:'a'.repeat(40),path:'xor.json',sha256:digest},{fetch:async()=>new Response(bytes)});
  check(loaded.inspection.baseHash===repaired.hash,'Hub contract fixture fingerprint');
  const rolled=workspace.rollback(1,2);
  check(rolled.revision===3 && rolled.hash===search.baseHash,'rollback exact');
  const canary=await runWasmCanary();
  check(canary.restoredHash===rolled.hash,'platform canary agrees');
  return {schema:'ruv-fann.toolkit-e2e/1',backend:trace.backend,baseHash:search.baseHash,candidateHash:repaired.hash,
    baselineMSE:search.evaluation.baseline.mse,candidateMSE:0,gateAccepted:true,regressionRejected:true,
    rollbackRevision:rolled.revision,compiledXor:true,plasticityBounded:true,hubFixtureVerified:true,
    searchEvaluations:search.evaluations,searchInferencePasses:search.inferencePasses};
}
