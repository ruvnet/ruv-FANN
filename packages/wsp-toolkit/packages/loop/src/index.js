import { inspectWeightModel, applyWeightPatch, fingerprintModel } from 'ruv-fann-wsp';
import { runRegressionGate } from 'ruv-fann-wsp-lifecycle';
import { runProposalSwarm } from 'ruv-fann-wsp-swarm';
import { checkModelInvariants, estimateInvariantWork } from 'ruv-fann-wsp-invariants';
import { record, integer, finite, identifier, snapshot, dataset, policy, limits } from './data.js';

export const LOOP_LIMITS=Object.freeze({maxModules:16,maxRounds:16,maxProposals:64,maxWork:100_000_000,maxConcurrency:4,maxDurationMs:300_000});
function modulesSnapshot(value) {
  const modules=snapshot(value);
  if(!Array.isArray(modules)||modules.length<1||modules.length>LOOP_LIMITS.maxModules)throw new RangeError('modules must contain 1..16 entries');
  const ids=new Set();
  return modules.map(value=>{
    const item=record(value,['id','development','holdouts','invariants','strategies','minImprovement','goalMSE','limits'],['id','development','holdouts','invariants','strategies'],'module');
    const id=identifier(item.id,'module id');if(ids.has(id))throw new TypeError('duplicate module id');ids.add(id);
    const dev=record(item.development,['dataset','policy'],['dataset','policy'],'development');
    const development={dataset:dataset(dev.dataset,'development dataset'),policy:policy(dev.policy)};
    if(!Array.isArray(item.holdouts)||item.holdouts.length<1||item.holdouts.length>15)throw new RangeError('each module requires 1..15 trusted holdout suites');
    const names=new Set(['development']);
    const holdouts=item.holdouts.map(value=>{
      const suite=record(value,['name','dataset','policy'],['name','dataset','policy'],'holdout');
      const name=suite.name;
      if(typeof name!=='string'||!/^[A-Za-z0-9_.-]{1,64}$/.test(name))throw new TypeError('invalid holdout suite name');
      if(names.has(name))throw new TypeError('duplicate suite name');names.add(name);
      return {name,dataset:dataset(suite.dataset,'holdout dataset'),policy:policy(suite.policy)};
    });
    const invariant=record(item.invariants,['schema','cases'],['schema','cases'],'invariant suite');
    if(invariant.schema!=='ruv-fann.invariants/1'||!Array.isArray(invariant.cases)||invariant.cases.length<1||invariant.cases.length>16)throw new TypeError('invalid invariant suite');
    let invariantRows=0;
    for(const value of invariant.cases) {
      const c=record(value,['name','inputs','rows','minimum','maximum','expected','tolerance'],['name','inputs','rows'],'invariant case');
      invariantRows+=integer(c.rows,1,1024,'invariant rows');
      if(!Array.isArray(c.inputs)||c.inputs.length<1||c.inputs.length>65536)throw new RangeError('invalid invariant inputs');
      if(c.minimum===undefined&&c.maximum===undefined&&c.expected===undefined)throw new TypeError('invariant case needs an output constraint');
    }
    if(invariantRows>1024)throw new RangeError('aggregate invariant rows exceed limit');
    if(!Array.isArray(item.strategies)||item.strategies.length<1||item.strategies.length>16)throw new RangeError('strategies must contain 1..16 entries');
    const strategyNames=new Set();
    const strategies=item.strategies.map(value=>{
      const strategy=record(value,['name','kind','options','edits'],['name','kind'],'strategy');
      const name=identifier(strategy.name,'strategy name');if(strategyNames.has(name))throw new TypeError('duplicate strategy name');strategyNames.add(name);
      if(strategy.kind==='patch') {
        if(strategy.options!==undefined||!Array.isArray(strategy.edits)||strategy.edits.length<1||strategy.edits.length>128)throw new TypeError('patch strategy needs 1..128 edits only');
        return {name,kind:'patch',edits:strategy.edits};
      }
      if(strategy.kind!=='search'||strategy.edits!==undefined)throw new TypeError('unknown proposal strategy');
      const options=record(strategy.options,['locations','maxEvaluations','maxPasses','maxWork','simd','limits'],['locations'],'search strategy');
      if(!Array.isArray(options.locations)||options.locations.length<1||options.locations.length>16)throw new RangeError('search locations must contain 1..16 entries');
      for(const location of options.locations) {
        const entry=record(location,['layer','parameter','index','values'],['layer','parameter','index','values'],'search location');
        integer(entry.layer,0,63,'location layer');integer(entry.index,0,65535,'location index');
        if(!['weights','bias'].includes(entry.parameter)||!Array.isArray(entry.values)||entry.values.length<1||entry.values.length>32)throw new TypeError('invalid search location');
      }
      integer(options.maxEvaluations??64,1,256,'maxEvaluations');integer(options.maxPasses??4,1,16,'maxPasses');
      if(options.maxWork!==undefined)integer(options.maxWork,1,10_000_000,'search maxWork');
      if(options.simd!==undefined&&![true,false,'auto'].includes(options.simd))throw new TypeError('invalid search SIMD option');
      if(options.limits!==undefined)limits(options.limits);
      return {name,kind:'search',options};
    });
    return {id,development,holdouts,invariants:invariant,invariantRows,strategies,
      minImprovement:finite(item.minImprovement??0,0,'minImprovement'),goalMSE:finite(item.goalMSE??0,0,'goalMSE'),limits:limits(item.limits)};
  });
}
function searchWork(strategy,parameters,rows) {
  if(strategy.kind!=='search')return 0;
  const opt=strategy.options;
  const candidates=Math.min(opt.maxEvaluations??64,(opt.maxPasses??4)*opt.locations.reduce((sum,loc)=>sum+loc.values.length,0));
  const work=(candidates+3)*parameters*rows;
  if(work>(opt.maxWork??10_000_000))throw new RangeError('search exceeds its work limit');
  return work;
}
function gateWork(module,parameters,invariantWork) {
  // Every candidate reserves independent validation and a possible AgentDB re-evaluation.
  const suiteRows=module.development.dataset.rows+module.holdouts.reduce((sum,s)=>sum+s.dataset.rows,0);
  const work=2*suiteRows*parameters;
  if(work>10_000_000)throw new RangeError('regression suites exceed aggregate work limit');
  if(invariantWork>10_000_000)throw new RangeError('invariants exceed aggregate work limit');
  return work+invariantWork+2*module.development.dataset.rows*parameters;
}
function cleanError(error) {return String(error?.message??error).slice(0,256);}
function registryCapabilities(registry) {
  if(!registry||typeof registry!=='object')throw new TypeError('registry capabilities are required');
  const methods=Object.create(null);
  for(const name of ['get','propose','promote']) {
    let owner=registry,descriptor;
    while(owner&&owner!==Object.prototype) {
      descriptor=Object.getOwnPropertyDescriptor(owner,name);
      if(descriptor)break;
      owner=Object.getPrototypeOf(owner);
    }
    if(!descriptor||!Object.hasOwn(descriptor,'value')||typeof descriptor.value!=='function')
      throw new TypeError(`registry.${name} must be a data method`);
    methods[name]=descriptor.value.bind(registry);
  }
  return methods;
}
function programSnapshot(value) {
  if(value===null)return null;
  if(!value||typeof value!=='object'||Array.isArray(value))throw new TypeError('registry program must be a data record');
  const program=Object.create(null);
  // Ignore registry metadata while detaching the fields that define CAS authority.
  // A registry may return a live record that another coordinator updates in place.
  for(const key of ['revision','hash','model']) {
    const descriptor=Object.getOwnPropertyDescriptor(value,key);
    if(!descriptor||!Object.hasOwn(descriptor,'value'))throw new TypeError(`registry program ${key} must be own data`);
    program[key]=descriptor.value;
  }
  integer(program.revision,1,Number.MAX_SAFE_INTEGER,'registry revision');
  if(typeof program.hash!=='string'||!/^[a-f0-9]{64}$/.test(program.hash))throw new TypeError('invalid registry model hash');
  program.model=snapshot(program.model);
  return program;
}
function abortCapability(source) {
  if(source===undefined)return undefined;
  if(!(source instanceof AbortSignal))throw new TypeError('signal must be an AbortSignal');
  const aborted=Object.getOwnPropertyDescriptor(AbortSignal.prototype,'aborted').get;
  const reason=Object.getOwnPropertyDescriptor(AbortSignal.prototype,'reason').get;
  aborted.call(source); // Do not trust shadow properties or forged prototype chains.
  return {get aborted(){return aborted.call(source);},get reason(){return reason.call(source);},
    addEventListener:EventTarget.prototype.addEventListener.bind(source),
    removeEventListener:EventTarget.prototype.removeEventListener.bind(source)};
}

/** Execute a finite local loop. Only trusted coordinator data can define acceptance criteria. */
export async function runEvolutionLoop(configuration) {
  const raw=record(configuration,['registry','modules','maxRounds','maxProposals','maxWork','concurrency','jobTimeoutMs','maxDurationMs','simd','signal'],['registry','modules'],'loop configuration');
  const registry=registryCapabilities(raw.registry);
  const modules=modulesSnapshot(raw.modules);
  const budget={maxRounds:integer(raw.maxRounds??4,1,LOOP_LIMITS.maxRounds,'maxRounds'),maxProposals:integer(raw.maxProposals??32,1,LOOP_LIMITS.maxProposals,'maxProposals'),
    maxWork:integer(raw.maxWork??LOOP_LIMITS.maxWork,1,LOOP_LIMITS.maxWork,'maxWork'),concurrency:integer(raw.concurrency??2,1,LOOP_LIMITS.maxConcurrency,'concurrency'),
    jobTimeoutMs:integer(raw.jobTimeoutMs??10000,1,60000,'jobTimeoutMs'),maxDurationMs:integer(raw.maxDurationMs??60000,1,LOOP_LIMITS.maxDurationMs,'maxDurationMs')};
  const simd=raw.simd??'auto';if(![true,false,'auto'].includes(simd))throw new TypeError('simd must be boolean or auto');
  const signal=abortCapability(raw.signal);
  const controller=new AbortController();let abortReason='aborted';
  const abort=()=>controller.abort(signal?.reason);
  if(signal?.aborted)abort();else signal?.addEventListener('abort',abort,{once:true});
  const started=performance.now();
  const timer=setTimeout(()=>{abortReason='deadline';controller.abort(new Error('loop deadline'));},budget.maxDurationMs);
  const events=[],states=new Map(),threadIds=new Set();let proposals=0,reservedWork=0,rounds=0,promotions=0,stopReason='round-limit';
  let exhausted;
  const interrupted=()=>{
    if(!controller.signal.aborted&&performance.now()-started>=budget.maxDurationMs){abortReason='deadline';controller.abort(new Error('loop deadline'));}
    return controller.signal.aborted;
  };
  const emit=event=>events.push(event);
  const observe=(id,current)=>{
    const state=states.get(id);
    if(state&&current){if(state.finalHash!==current.hash)state.goalReached=false;state.finalRevision=current.revision;state.finalHash=current.hash;}
  };
  try {
    outer: for(let round=1;round<=budget.maxRounds;round++) {
      if(interrupted()){stopReason=abortReason;break;}
      rounds=round;let roundPromotions=0;
      for(const module of modules) {
        if(interrupted()){stopReason=abortReason;break outer;}
        const source=programSnapshot(await registry.get(module.id));
        if(interrupted()){stopReason=abortReason;break outer;}
        if(!source)throw new Error(`module ${module.id} is not registered`);
        const info=await inspectWeightModel(source.model);
        if(info.baseHash!==source.hash||!Number.isSafeInteger(source.revision)||source.revision<1)throw new Error('registry returned inconsistent model lineage');
        if(!states.has(module.id))states.set(module.id,{id:module.id,initialRevision:source.revision,initialHash:source.hash,finalRevision:source.revision,finalHash:source.hash,promotions:0,goalReached:false});
        else observe(module.id,source);
        if(states.get(module.id).goalReached)continue;
        for(const suite of [module.development,...module.holdouts]) {
          if(suite.dataset.inputs.length!==suite.dataset.rows*info.inputSize||suite.dataset.targets.length!==suite.dataset.rows*info.outputSize)throw new RangeError('trusted suite dimensions disagree with module');
        }
        const invariantWork=estimateInvariantWork(source.model,module.invariants);
        const verificationWork=gateWork(module,info.parameterCount,invariantWork),jobs=[],strategies=[];
        for(const strategy of module.strategies) {
          if(proposals>=budget.maxProposals){exhausted='proposal-limit';break;}
          const work=searchWork(strategy,info.parameterCount,module.development.dataset.rows)+verificationWork;
          if(reservedWork+work>budget.maxWork){exhausted='work-limit';break;}
          proposals++;reservedWork+=work;
          const id=`r${round}:${module.id}:${strategy.name}`;
          jobs.push(strategy.kind==='patch'?{id,kind:'patch',patch:{schema:'ruv-fann.weight-patch/1',baseHash:source.hash,edits:strategy.edits}}:
            {id,kind:'search',model:source.model,dataset:module.development.dataset,policy:module.development.policy,
              searchOptions:{...strategy.options,simd:strategy.options.simd??simd}});
          strategies.push(strategy.name);
        }
        if(!jobs.length){stopReason=exhausted??'no-progress';break outer;}
        const results=await runProposalSwarm(jobs,{concurrency:budget.concurrency,timeoutMs:budget.jobTimeoutMs,signal:controller.signal});
        if(interrupted()){stopReason=abortReason;break outer;}
        if(!Array.isArray(results)||results.length!==jobs.length)throw new Error('swarm returned malformed result count');
        const candidates=[],seen=new Set();
        for(let index=0;index<results.length;index++) {
          if(interrupted()){stopReason=abortReason;break outer;}
          const result=results[index],strategy=strategies[index];
          const context={round,moduleId:module.id,revision:source.revision,strategy};
          if(result.id!==jobs[index].id)throw new Error('swarm result identity mismatch');
          if(Number.isInteger(result.workerThreadId)&&result.workerThreadId>0)threadIds.add(result.workerThreadId);
          if(result.status!=='ok'){emit({...context,type:'worker-error',reason:cleanError(result.error??result.status)});continue;}
          if(!result.patch){emit({...context,type:'no-candidate'});continue;}
          emit({...context,type:'proposed'});
          try {
            // Never use a worker's score, acceptance flag or model as promotion evidence.
            const model=await applyWeightPatch(source.model,result.patch,module.limits);
            const candidateHash=await fingerprintModel(model);
            if(seen.has(candidateHash)){emit({...context,type:'duplicate',candidateHash});continue;}
            seen.add(candidateHash);
            if(interrupted()){stopReason=abortReason;break outer;}
            const gate=await runRegressionGate({model:source.model,patch:result.patch,
              suites:[{name:'development',...module.development},...module.holdouts]},{simd,limits:module.limits});
            const development=gate.suites[0].evaluation;
            if(!gate.accepted){emit({...context,type:'rejected',reason:'regression-gate',candidateHash,
              failedSuites:gate.suites.filter(s=>!s.evaluation.accepted).map(s=>s.name)});continue;}
            if(!(development.baseline.mse-development.candidate.mse>module.minImprovement)) {
              emit({...context,type:'rejected',reason:'no-strict-improvement',candidateHash});continue;
            }
            if(interrupted()){stopReason=abortReason;break outer;}
            const invariants=await checkModelInvariants(model,module.invariants,{simd});
            if(invariants.modelHash!==candidateHash||gate.candidateHash!==candidateHash||gate.baseHash!==source.hash)throw new Error('independent evidence hash mismatch');
            if(!invariants.accepted){emit({...context,type:'rejected',reason:'invariant-gate',candidateHash});continue;}
            emit({...context,type:'validated',candidateHash,baselineMSE:development.baseline.mse,candidateMSE:development.candidate.mse,
              suiteHashes:gate.suites.map(s=>({name:s.name,datasetHash:s.evaluation.datasetHash,policyHash:s.evaluation.policyHash})),invariantSuiteHash:invariants.suiteHash});
            candidates.push({strategy,patch:result.patch,candidateHash,score:development.candidate.mse});
          } catch(error) {emit({...context,type:'rejected',reason:cleanError(error)});}
        }
        candidates.sort((a,b)=>a.score-b.score||a.candidateHash.localeCompare(b.candidateHash)||a.strategy.localeCompare(b.strategy));
        const winner=candidates[0];
        if(winner&&!interrupted()) {
          const context={round,moduleId:module.id,revision:source.revision,strategy:winner.strategy};
          let commitEntered=false;
          try {
            const current=programSnapshot(await registry.get(module.id));
            if(current?.revision!==source.revision||current?.hash!==source.hash){observe(module.id,current);emit({...context,type:'stale'});continue;}
            if(interrupted()){stopReason=abortReason;break outer;}
            const proposal=await registry.propose(module.id,winner.patch,module.development.dataset,module.development.policy);
            if(interrupted()){stopReason=abortReason;break outer;}
            if(proposal.revision!==source.revision||!proposal.evaluation?.accepted||proposal.evaluation.candidateHash!==winner.candidateHash)
              throw new Error('registry proposal disagrees with independent evidence');
            // Abort prevents entering a commit. A commit already entered is awaited and reported.
            commitEntered=true;
            const next=await registry.promote(module.id,proposal.proposalId,source.revision);
            if(next.hash!==winner.candidateHash||next.revision!==source.revision+1)throw new Error('registry promotion returned inconsistent lineage');
            const state=states.get(module.id);state.finalRevision=next.revision;state.finalHash=next.hash;state.promotions++;
            state.goalReached=winner.score<=module.goalMSE;promotions++;roundPromotions++;
            emit({...context,type:'promoted',nextRevision:next.revision,candidateHash:next.hash,candidateMSE:winner.score});
          } catch(error) {
            if(error?.code==='WSP_REGISTRY_PERSISTENCE'||(commitEntered&&error?.code!=='WSP_REGISTRY_STALE'))throw error;
            const current=programSnapshot(await registry.get(module.id));
            if(error?.code==='WSP_REGISTRY_STALE'||current?.revision!==source.revision||current?.hash!==source.hash){observe(module.id,current);emit({...context,type:'stale',reason:cleanError(error)});}
            else throw error; // Durability/runtime failures cannot be hidden as rejected mutations.
          }
        }
      }
      if(interrupted()){stopReason=abortReason;break;}
      if(exhausted){stopReason=exhausted;break;}
      // Refresh behavioral goal evidence against active content before claiming convergence.
      for(const state of states.values()) {
        if(interrupted()){stopReason=abortReason;break outer;}
        const current=programSnapshot(await registry.get(state.id));
        if(interrupted()){stopReason=abortReason;break outer;}
        if(!current)throw new Error('registered module disappeared during loop');
        observe(state.id,current);
      }
      if(states.size===modules.length&&[...states.values()].every(s=>s.goalReached)){stopReason='goals-met';break;}
      if(roundPromotions===0){stopReason='no-progress';break;}
    }
    return {schema:'ruv-fann.evolution-loop/1',stopReason,rounds,promotions,
      budget:{...budget,proposals,reservedWork,elapsedMs:performance.now()-started},
      workerThreadIds:[...threadIds].sort((a,b)=>a-b),modules:[...states.values()],events};
  } finally {clearTimeout(timer);signal?.removeEventListener('abort',abort);}
}
