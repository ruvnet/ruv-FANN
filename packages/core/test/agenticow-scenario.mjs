/** Independent numerical gate plus actual native RVF branch/persistence acceptance. */
export async function qualifyAgenticow({memory,wsp,lifecycle,core},directory) {
  const assert=(await import('node:assert/strict')).default;
  const {mkdir}=await import('node:fs/promises');
  const {join}=await import('node:path');
  await mkdir(directory,{recursive:true});
  const initial={schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:[0],bias:[0],activation:'Linear',steepness:1}]};
  const development={inputs:[0,1,2],targets:[1,3,5],rows:3};
  const holdout={inputs:[0.5,1.5],targets:[2,4],rows:2};
  const policy={maxMSE:0,maxRegressionMSE:0,maxAbsoluteError:0};
  const hash=await wsp.fingerprintModel(initial);
  const patch=weight=>({schema:'ruv-fann.weight-patch/1',baseHash:hash,edits:[{layer:0,parameter:'weights',index:0,value:weight},{layer:0,parameter:'bias',index:0,value:1}]});
  const workspace=await wsp.createWeightWorkspace(initial);
  const proposal=await workspace.propose(patch(2),development,policy);
  assert.equal(proposal.evaluation.accepted,true);
  const candidate=workspace.promote(proposal.id,1).model;
  const encode=async model=>JSON.stringify({schema:'ruv-fann.weight-memory/1',hash:await wsp.fingerprintModel(model),model});
  const vector=[1,0]; // Caller-supplied task vector, never rounded model weights.
  const query=store=>store.query(vector,1,{forceExact:true})[0];
  let base=memory.open(join(directory,'base.rvf'),{dimension:2,metric:'l2'}),good,bad,reopened,plain;
  try {
    assert.ok(base instanceof memory.AgenticMemory);
    assert.equal(base.ingest([{id:1,vector,text:await encode(initial)}]).accepted,1);
    good=base.branch('candidate');
    bad=base.branch('rejected');
    good.ingest([{id:1,vector,text:await encode(candidate)}]);
    assert.equal(JSON.parse(query(base).text).hash,hash);
    assert.notEqual(JSON.parse(query(good).text).hash,hash);
    assert.deepEqual(good.diff().overridden,[1]);
    const rejectedGate=await lifecycle.runRegressionGate({model:initial,patch:patch(-2),suites:[{name:'independent',dataset:holdout,policy}]});
    assert.equal(rejectedGate.accepted,false);
    const badModel={...initial,layers:[{...initial.layers[0],weights:[-2],bias:[1]}]};
    bad.ingest([{id:1,vector,text:await encode(badModel)},{id:2,vector:[0,1],text:'rejected experiment'}]);
    assert.deepEqual(JSON.parse(query(bad).text).model,badModel);
    assert.equal(JSON.parse(query(base).text).hash,hash);
    assert.deepEqual(base.query(vector,10,{forceExact:true}).map(x=>x.id),[1]);
    bad.close();bad=null; // Rejected edits are discarded; never promoted.
    const gate=await lifecycle.runRegressionGate({model:initial,patch:patch(2),suites:[{name:'independent',dataset:holdout,policy}]});
    assert.equal(gate.accepted,true);
    const checkpoint=base.checkpoint('before-promotion');
    assert.deepEqual(good.promote(base),{ingested:1,deleted:0});
    const accepted=JSON.parse(query(base).text);
    assert.equal(accepted.hash,await wsp.fingerprintModel(accepted.model));
    assert.deepEqual(accepted.model,candidate);
    const runtime=await core.loadRuntime();
    const prepared=runtime.prepare(accepted.model);
    try {assert.deepEqual([...prepared.predict(Float64Array.of(0.5,1.5))],holdout.targets);}
    finally {prepared.dispose();}
    base.rollback(checkpoint.id);
    assert.equal(JSON.parse(query(base).text).hash,hash);
    const rollbackManifest=join(directory,'rollback.json');base.save(rollbackManifest);
    good.close();good=null;base.close();base=null;
    base=memory.AgenticMemory.load(rollbackManifest);
    assert.equal(JSON.parse(query(base).text).hash,hash);
    good=base.branch('after-rollback');
    good.ingest([{id:1,vector,text:await encode(candidate)}]);
    good.promote(base);
    const workingPath=base.lineage()[0].path;
    const manifest=join(directory,'memory.json');base.save(manifest);
    good.close();good=null;base.close();base=null;
    reopened=memory.AgenticMemory.load(manifest);
    assert.deepEqual(JSON.parse(query(reopened).text).model,candidate);
    reopened.close();reopened=null;
    // Prove RVF metadata survives a plain reopen as well as a manifest load.
    plain=memory.open(workingPath,{metric:'l2'});
    assert.deepEqual(JSON.parse(query(plain).text).model,candidate);
    return {version:'0.3.0',native:true,branchIsolation:true,rejectedCandidateIsolated:true,promoted:true,checkpointRollback:true,reopened:true};
  } finally {plain?.close();reopened?.close();bad?.close();good?.close();base?.close();}
}
