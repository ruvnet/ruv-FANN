import test from 'node:test';
import assert from 'node:assert/strict';
import {collectPublicMetrics,analyzeConstellation,renderConstellationDashboard}from'../src/index.js';
import{catalog,observations,fixtureFetch,start,end,days}from'./fixtures.mjs';

test('complete-window sums and independent weekday/correlation oracles',()=>{
  const report=analyzeConstellation(catalog(),observations()),hub=report.packages[0];
  assert.equal(hub.latestDayDownloads,42);assert.equal(hub.last7Downloads,273);assert.equal(hub.last30Downloads,825);
  assert.equal(hub.weekdayBaseline,24.5);assert.ok(Math.abs(hub.weekdayDeltaPercent-100*17.5/24.5)<1e-10);
  assert.equal(report.totals.last7Downloads,819);assert.equal(report.totals.last30Downloads,2475);
  assert.equal(report.correlations.pairs[0].r,1);assert.equal(report.correlations.pairs[0].samples,42);
  assert.equal(report.declaredLinks[0].target,'core');assert.equal(hub.runtimeDependencyCount,1);assert.equal(report.packages[2].runtimeDependencyCount,null);assert.equal(report.attach.status,'unavailable');assert.equal(report.attach.percentage,null);
  assert.equal(report.repositories[0].openIssuesAndPRs,3);assert.match(report.definitions.repositories,/pull requests/);
});
test('missing days, sources and candidates remain unknown rather than zero-filled',()=>{
  const input=observations();input.packages[0].days.splice(39,1);
  const report=analyzeConstellation(catalog(),input);
  assert.equal(report.packages[0].last7Downloads,null);assert.equal(report.totals.last7Downloads,null);
  assert.equal(report.totals.observedLast7Downloads,546);assert.equal(report.totals.coverage.packagesWithLast7,1);
  assert.equal(report.dailySeries[39].downloads,null);assert.equal(report.packages[2].latestDayDownloads,null);
  input.packages=[];const missing=analyzeConstellation(catalog(),input);
  assert.equal(missing.totals.last30Downloads,null);assert.equal(missing.totals.coverage.unavailablePackages,2);
});
test('zero baselines, short windows and constant series have no invented percentages or correlations',()=>{
  const input=observations();for(const p of input.packages)for(const d of p.days)d.downloads=0;
  const report=analyzeConstellation(catalog(),input);
  assert.equal(report.packages[0].last30Downloads,0);assert.equal(report.packages[0].weekdayDeltaPercent,null);assert.deepEqual(report.correlations.pairs,[]);
  input.start='2024-02-05';for(const p of input.packages)p.days=p.days.slice(-7);
  const short=analyzeConstellation(catalog(),input);assert.equal(short.packages[0].last30Downloads,null);assert.equal(short.packages[0].weekdayBaseline,null);
});
test('explicit adoption deduplicates latest project snapshots and exposes only aggregate project attach',()=>{
  const adoption={schema:'ruvnet.constellation-adoption/1',consent:true,samples:[
    {sampleId:'a'.repeat(64),observedAt:'2024-02-10T12:00:00Z',packages:['hub']},
    {sampleId:'a'.repeat(64),observedAt:'2024-02-11T12:00:00Z',packages:['hub','core']},
    {sampleId:'b'.repeat(64),observedAt:'2024-02-11T12:00:00Z',packages:['hub']},
    {sampleId:'c'.repeat(64),observedAt:'2024-02-11T12:00:00Z',packages:['core']}
  ]};
  const report=analyzeConstellation(catalog(),observations(),{adoption});
  assert.equal(report.attach.unit,'consenting-project-snapshots');assert.equal(report.attach.denominator,2);assert.equal(report.attach.numerator,1);assert.equal(report.attach.percentage,50);
  assert.equal(report.attach.projects,3);assert.equal(report.attach.source.duplicateSnapshotsRemoved,1);assert.equal(report.attach.perPrimitive[0].percentage,50);
  assert.ok(!JSON.stringify(report).includes('a'.repeat(64)));assert.ok(!JSON.stringify(report).includes('sampleId'));
  adoption.samples[1].packages=[];assert.equal(report.attach.percentage,50);
});
test('consent, identifiers, duplicate conflicts and empty denominators are explicit',()=>{
  const sample={sampleId:'a'.repeat(64),observedAt:'2024-02-11T12:00:00Z',packages:['hub','core']};
  for(const adoption of [
    {schema:'ruvnet.constellation-adoption/1',consent:false,samples:[]},
    {schema:'ruvnet.constellation-adoption/1',consent:true,samples:[{...sample,sampleId:'personal@example.test'}]},
    {schema:'ruvnet.constellation-adoption/1',consent:true,samples:[sample,{...sample,observedAt:'2024-02-11T12:00:00.000Z',packages:['hub']}]}
  ])assert.throws(()=>analyzeConstellation(catalog(),observations(),{adoption}));
  const report=analyzeConstellation(catalog(),observations(),{adoption:{schema:'ruvnet.constellation-adoption/1',consent:true,samples:[{...sample,packages:['core']}]}});
  assert.equal(report.attach.status,'unavailable');assert.equal(report.attach.percentage,null);assert.equal(report.attach.denominator,0);
});
test('analysis rejects duplicate/out-of-range points, unsafe counters, unknown fields and getters',()=>{
  for(const mutate of [x=>x.packages[0].days.push(x.packages[0].days[0]),x=>x.packages[0].days[0].day='2024-02-30',
    x=>x.packages[0].days[0].downloads=Infinity,x=>x.packages[2].days=[{day:start,downloads:0}],x=>x.extra=true]){
    const input=observations();mutate(input);assert.throws(()=>analyzeConstellation(catalog(),input));
  }
  let calls=0;const input=observations();Object.defineProperty(input,'packages',{get(){calls++;return [];}});
  assert.throws(()=>analyzeConstellation(catalog(),input),/accessors/);assert.equal(calls,0);
  const overflow=observations();overflow.packages[0].days[0].downloads=Number.MAX_SAFE_INTEGER;assert.throws(()=>analyzeConstellation(catalog(),overflow),/total/);
});
test('public collection uses fixed public sources, bounded runners and never queries candidates',async()=>{
  const calls=[],input=catalog();const pending=collectPublicMetrics(input,{start,end,fetch:fixtureFetch(calls),concurrency:2});input.packages[0].name='changed';
  const result=await pending;assert.equal(result.packages[0].name,'hub');assert.equal(result.packages[0].status,'ok');assert.equal(result.packages[2].status,'candidate');
  assert.equal(result.repositories[0].openIssuesAndPRs,3);assert.equal(calls.length,5);
  assert.ok(calls.every(c=>['api.npmjs.org','registry.npmjs.org','api.github.com'].includes(new URL(c.url).hostname)));
  assert.ok(calls.every(c=>c.options.credentials==='omit'&&c.options.redirect==='error'&&!('Authorization'in c.options.headers)));
  assert.ok(!calls.some(c=>c.url.includes('candidate')));
});
test('metadata failure does not erase independent downloads; source failures stay unavailable',async()=>{
  const good=fixtureFetch();const result=await collectPublicMetrics(catalog(),{start,end,fetch:async(url,options)=>new URL(url).hostname==='registry.npmjs.org'?new Response('',{status:503}):good(url,options)});
  assert.equal(result.packages[0].status,'ok');assert.equal(result.packages[0].metadataStatus,'unavailable');assert.equal(result.packages[0].days.length,42);
  const report=analyzeConstellation(catalog(),result);assert.equal(report.packages[0].runtimeDependencyCount,null);assert.equal(report.declaredLinksCoverage.observedManifests,0);assert.equal(report.declaredLinks.length,0);
  const failed=await collectPublicMetrics(catalog(),{start,end,fetch:async()=>new Response('',{status:429})});
  assert.equal(failed.packages[0].status,'unavailable');assert.equal(failed.packages[0].days.length,0);assert.equal(failed.repositories[0].status,'unavailable');
  assert.match(failed.packages[0].error,/429/);
});
test('malformed names, traversal, dates, raised limits and options reject before fetching',async()=>{
  let requests=0;const fetch=async()=>{requests++;return Response.json({});};
  for(const mutate of [x=>x.packages[0].name='hub?token=secret',x=>x.packages[0].repository='owner/../../internal',x=>x.packages[0].directory='../other']){
    const input=catalog();mutate(input);await assert.rejects(collectPublicMetrics(input,{start,end,fetch}));
  }
  for(const extra of [{start:'2024-02-30'}, {concurrency:9},{timeoutMs:60001},{url:'https://internal.test'}])await assert.rejects(collectPublicMetrics(catalog(),{start,end,fetch,...extra}));
  assert.equal(requests,0);
});
test('deadline bounds uncooperative fetches and no later task issues requests',async()=>{
  let calls=0;const started=performance.now();const result=await collectPublicMetrics(catalog(),{start,end,timeoutMs:10,concurrency:1,fetch:()=>{calls++;return new Promise(()=>{});}});
  assert.ok(performance.now()-started<1000);assert.equal(calls,1);assert.match(result.packages[0].error,/deadline/);assert.equal(result.repositories[0].status,'unavailable');
});
test('response byte limits reject declared and streamed overflow, origin changes and redirects',async()=>{
  for(const fetch of [async()=>new Response('{}',{headers:{'content-length':'3000000'}}),
    async()=>new Response(new Uint8Array(2*1024*1024+1)),
    async()=>{const response=Response.json({});Object.defineProperty(response,'url',{value:'https://internal.test/'});return response;},
    async()=>new Response('',{status:302,headers:{location:'https://internal.test/'}})]){
    const result=await collectPublicMetrics(catalog(),{start,end,fetch});assert.equal(result.packages[0].status,'unavailable');assert.equal(result.repositories[0].status,'unavailable');
  }
});
test('standalone dashboard escapes hostile text, uses safe DOM and disables network collection',()=>{
  const input=catalog();input.packages[0].description='</script><img src=x onerror="window.pwned=1"> & \u2028';
  const report=analyzeConstellation(input,observations()),html=renderConstellationDashboard(report);
  assert.ok(!html.includes('<img src=x'));assert.match(html,/\\u003c\/script\\u003e/);assert.match(html,/connect-src 'none'/);assert.match(html,/textContent/);
  assert.ok(!/<script[^>]+src=/.test(html));assert.ok(!html.includes('innerHTML'));assert.match(html,/id="period-filter"/);assert.match(html,/id="package-table"/);
  assert.throws(()=>renderConstellationDashboard({schema:'other'}));
});
test('out-of-order contradictions reject at any prior instant with normalized fractional timestamps',()=>{
  const first={sampleId:'a'.repeat(64),observedAt:'2024-02-10T12:00:00.1Z',packages:['hub','core']};
  const newer={...first,observedAt:'2024-02-11T12:00:00Z',packages:['hub']};
  const conflicting={...first,observedAt:'2024-02-10T12:00:00.100Z',packages:['hub']};
  const run=samples=>analyzeConstellation(catalog(),observations(),{adoption:{schema:'ruvnet.constellation-adoption/1',consent:true,samples}});
  for(const samples of [[first,newer,conflicting],[newer,first,conflicting],[conflicting,newer,first]])assert.throws(()=>run(samples),/conflicting duplicate project snapshot/);
  const consistent=run([first,newer,{...first,observedAt:'2024-02-10T12:00:00.100Z',packages:['core','hub']}]);
  assert.equal(consistent.attach.projects,1);assert.equal(consistent.attach.denominator,1);assert.equal(consistent.attach.percentage,0);
  assert.equal(consistent.attach.source.duplicateSnapshotsRemoved,2);
  assert.throws(()=>run([{...first,observedAt:'2023-12-31T12:00:00Z'},newer,{...conflicting,observedAt:'2023-12-31T12:00:00.000Z'}]),/conflicting duplicate project snapshot/);
});
