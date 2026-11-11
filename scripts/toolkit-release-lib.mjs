// Shared, side-effect-free validation for the exact-artifact toolkit release lane.
import { createHash } from 'node:crypto';
export const schema = 'ruv-fann.toolkit';
export const releaseTag = 'ruv-fann-v0.3.0';
// Only core changes generation. Its published runtime/toolkit dependencies are never repacked.
export const packageSpecs = [{name:'ruv-fann',version:'0.3.0',directory:'packages/core'}];
export const coreDependencies = {
  'ruv-fann-wasm':'0.1.2','ruv-fann-wsp':'0.2.0',
  ...Object.fromEntries(['lifecycle','search','huggingface','cloudflare','ruos','invariants','swarm','registry','loop'].map(name=>[`ruv-fann-wsp-${name}`,'0.1.0'])),
  'ruvnet-constellation':'0.1.0','ruvnet-constellation-contracts':'0.1.0','ruvnet-constellation-telemetry':'0.1.0',
  agenticow:'0.3.0'
};
export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
export const integrity = bytes => `sha512-${createHash('sha512').update(bytes).digest('base64')}`;
export const tarballURL = row => `https://registry.npmjs.org/${row.name}/-/${row.name}-${row.version}.tgz`;
export function safeFilename(name) {
  if (typeof name !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,180}$/.test(name) || name.includes('..')) throw Error('Unsafe candidate filename');
  return name;
}
export function assertSourceSha(value) {if(typeof value!=='string'||!/^[a-f0-9]{40}$/.test(value))throw Error('Expected full source SHA');return value;}
export function topologicalOrder(rows) {
  const byName=new Map(rows.map(row=>[row.name,row]));
  if(byName.size!==rows.length)throw Error('Duplicate candidate package');
  const done=new Set(),active=new Set(),ordered=[];
  function visit(row) {
    if(done.has(row.name))return;
    if(active.has(row.name))throw Error('Candidate dependency cycle');
    active.add(row.name);
    for(const [name,version]of Object.entries(row.dependencies??{})) {
      const dependency=byName.get(name);
      // Legacy core0.1.2 stays nested under published WSP0.2.0 and satellites.
      if(dependency && version===dependency.version)visit(dependency);
      else if(dependency && !(name==='ruv-fann'&&version==='0.1.2'))throw Error(`Internal dependency must pin candidate version: ${row.name} -> ${name}`);
    }
    active.delete(row.name);done.add(row.name);ordered.push(row);
  }
  rows.forEach(visit);return ordered;
}
export function validateRows(rows) {
  if(!Array.isArray(rows)||rows.length!==packageSpecs.length)throw Error('Expected only the core candidate package');
  for(const spec of packageSpecs) {
    const found=rows.filter(row=>row.name===spec.name);
    if(found.length!==1||found[0].version!==spec.version)throw Error(`Candidate identity mismatch: ${spec.name}`);
    const row=found[0];safeFilename(row.tarball);
    if(!/^[a-f0-9]{64}$/.test(row.sha256)||!/^sha512-[A-Za-z0-9+/]{86}==$/.test(row.integrity))throw Error('Candidate digest missing');
    if(!row.dependencies||typeof row.dependencies!=='object'||Array.isArray(row.dependencies))throw Error('Candidate dependencies missing');
    if(Object.keys(row.dependencies).length!==Object.keys(coreDependencies).length||Object.entries(coreDependencies).some(([name,version])=>row.dependencies[name]!==version))throw Error('Core candidate must pin all fifteen published dependencies');
  }
  const ordered=topologicalOrder(rows);
  if(JSON.stringify(ordered.map(r=>r.name))!==JSON.stringify(rows.map(r=>r.name)))throw Error('Candidate packages are not in dependency order');
  return rows;
}
export function validateCoreReceipt(receipt,sourceSha) {
  if(receipt?.schema!=='ruv-fann.core-consumer/1'||receipt.status!=='passed'||receipt.sourceSha!==sourceSha||
     !Number.isInteger(receipt.testCount)||receipt.testCount<66||receipt.skipped!==0||receipt.todo!==0||
     receipt.browser?.engine!=='Chromium'||receipt.browser.realWasm!==true||receipt.types?.strict!==true||
     receipt.graph?.core!=='0.3.0'||receipt.graph.legacyCore!=='0.1.2'||receipt.graph.finite!==true||
     receipt.types.subpaths!==15||receipt.browser.pageErrors?.length!==0||
     receipt.agenticow?.version!=='0.3.0'||!['native','branchIsolation','rejectedCandidateIsolated','promoted','checkpointRollback','reopened'].every(key=>receipt.agenticow[key]===true)||
     !['wasm','wasm-simd'].every(name=>[receipt.backendChecks,receipt.browser.checks].every(checks=>
       checks?.some(row=>row.backend===name&&row.checks>0&&['inference','training','search','gate','invariants','rejectedPromotion'].every(key=>row[key]===true)&&row.rollbackRevision===3))))throw Error('Installed core receipt must prove nonzero tests, strict types, finite graph, real WASM/Chromium and native Agenticow isolation/promotion/rollback/reopen');
}
export function validateQualification(receipt,sourceSha,manifestHash) {
  if(receipt?.schema!==`${schema}.qualification/1`||receipt.status!=='passed'||receipt.sourceSha!==sourceSha||receipt.manifestSha256!==manifestHash)throw Error('Qualification source/archive binding mismatch');
  const report=receipt.validation;
  if(report?.typecheck!=='passed'||!(report.tests?.tests>0)||report.tests.pass!==report.tests.tests||
     ['fail','cancelled','skipped','todo'].some(key=>report.tests[key]!==0)||report.audit?.vulnerabilities?.total!==0||report.agentdbIntegrationAudit?.vulnerabilities?.total!==0||receipt.candidateAudit?.vulnerabilities?.total!==0)
    throw Error('Qualification requires nonzero no-skip tests, types and zero-advisory audits');
  validateCoreReceipt(receipt.core,sourceSha);
  validateToolkitProof(receipt.toolkit);
  if(report.benchmark?.schema!=='ruv-fann.search-benchmark/1'||report.benchmark.candidateMSE!==0||
     report.benchmark.referenceInferencePasses!==32||report.benchmark.screenedInferencePasses!==19||
     !/^[a-f0-9]{64}$/.test(report.benchmark.equivalentCandidateHash))throw Error('Search benchmark must preserve the exact candidate and measured work counts');
  if(receipt.runtime!==report.node||receipt.runtime!==receipt.core.runtime)throw Error('Qualification runtimes disagree');
}
export function validateToolkitProof(proof) {
  if(!proof||!['wasm','wasm-simd'].every(backend=>proof.results?.some(row=>row.backend===backend&&row.candidateMSE===0&&row.regressionRejected===true&&row.compiledXor===true&&row.rollbackRevision===3))||
     proof.evolution?.report?.promotions!==2||proof.evolution.durability?.reopened!==true||proof.evolution.durability.rollbackPersisted!==true||!Array.isArray(proof.evolution.report.workerThreadIds)||new Set(proof.evolution.report.workerThreadIds).size<2||
     proof.constellation?.catalogEntries!==24||proof.constellation.scaffoldTests!==3)throw Error('Toolkit installation must prove both real WASM backends, concurrent durable AgentDB promotion/rollback and generated constellation tests');
}
export function publicLock(candidateLock,rows) {
  const lock=structuredClone(candidateLock),root=lock.packages?.[''];
  if(!root)throw Error('Installed candidate lock missing root');
  root.dependencies=Object.fromEntries(rows.map(row=>[row.name,row.version]));
  delete root.devDependencies;
  for(const row of rows) {
    const entry=lock.packages[`node_modules/${row.name}`];
    if(!entry||entry.version!==row.version||entry.link)throw Error(`Candidate not installed at root: ${row.name}`);
    entry.resolved=tarballURL(row);entry.integrity=row.integrity;
    delete entry.inBundle;
  }
  for(const [path,entry]of Object.entries(lock.packages)) {
    if(path&&(entry.link||!entry.resolved?.startsWith('https://registry.npmjs.org/')||!/^sha512-[A-Za-z0-9+/]{86}==$/.test(entry.integrity??'')))throw Error('Public lock contains a link, unpinned integrity or non-registry dependency');
  }
  return lock;
}
