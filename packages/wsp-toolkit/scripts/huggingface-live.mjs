import { loadHubLinearHead } from 'ruv-fann-wsp-huggingface';
import { traceWeightModel, evaluateWeightPatch } from 'ruv-fann-wsp';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const source=JSON.parse(await readFile(new URL('../docs/huggingface-source.json',import.meta.url),'utf8'));
const loaded=await loadHubLinearHead(source,{timeoutMs:30000});
const layer=loaded.model.layers[0];
const inputs=Array.from({length:layer.inputs*4},(_,i)=>Math.sin(i)/layer.inputs);
// Independently compute the declared affine interpretation, not the source task's outputs.
const targets=Array.from({length:4},(_,row)=>layer.weights.reduce((sum,w,i)=>sum+w*inputs[row*layer.inputs+i],layer.bias[0]));
const trace=await traceWeightModel(loaded.model,{inputs,rows:4},{simd:false});
const maxDifference=Math.max(...trace.layers[0].activations.map((v,i)=>Math.abs(v-targets[i])));
assert.ok(maxDifference<1e-12);
const patch={schema:'ruv-fann.weight-patch/1',baseHash:loaded.inspection.baseHash,
  edits:[{layer:0,parameter:'bias',index:0,value:layer.bias[0]+0.1}]};
const evaluation=await evaluateWeightPatch(loaded.model,patch,{inputs,targets,rows:4},{maxMSE:0.011,maxRegressionMSE:0.011},{simd:false});
assert.equal(evaluation.accepted,true);
assert.ok(Math.abs(evaluation.candidate.mse-0.01)<1e-12);
patch.edits[0].value=layer.bias[0]+1;
const rejected=await evaluateWeightPatch(loaded.model,patch,{inputs,targets,rows:4},{maxMSE:0.011,maxRegressionMSE:0.011},{simd:false});
assert.equal(rejected.accepted,false);
console.log(JSON.stringify({schema:'ruv-fann.hub-live-validation/1',source:loaded.provenance,interpretation:loaded.interpretation,
  modelHash:loaded.inspection.baseHash,parameters:loaded.inspection.parameterCount,backend:evaluation.backend,
  maxAffineOracleDifference:maxDifference,acceptedControlledMSE:evaluation.candidate.mse,rejectedControlledMSE:rejected.candidate.mse,
  taskAccuracyMeasured:false,note:'Live public Hub download and real WASM affine-head contract. Controlled synthetic inputs are not a WiFi sensing accuracy benchmark.'},null,2));
