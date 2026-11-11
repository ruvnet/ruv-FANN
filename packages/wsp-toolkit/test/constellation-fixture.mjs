import {readCatalog} from 'ruvnet-constellation';
import {analyzeConstellation} from 'ruvnet-constellation-telemetry';
export async function constellationFixture({adoption=false,hostile=false,candidates=false}={}) {
  const catalog=await readCatalog();
  // Explicit source-only test variant; the default fixture retains release declarations.
  if(candidates)for(const pkg of catalog.packages)if(pkg.directory)pkg.status='candidate';
  if(hostile)catalog.packages[0].description='</script><script>window.constellationPwned=true</script><img src=x onerror="window.constellationPwned=true">';
  const start='2026-09-30',end='2026-11-10';
  const observations={schema:'ruvnet.constellation-observations/1',observedAt:'2026-11-11T12:00:00.000Z',start,end,
    packages:catalog.packages.map((p,index)=>p.status==='candidate'?{name:p.name,status:'candidate',days:[]}:
      {name:p.name,status:'ok',version:p.version,metadataStatus:'ok',dependencies:p.name==='agentdb'?['ruv-fann-wsp']:[],peerDependencies:[],optionalDependencies:[],
        days:Array.from({length:42},(_,day)=>({day:new Date(Date.parse(start+'T00:00:00Z')+day*86400000).toISOString().slice(0,10),downloads:(index+1)*100+day}))}),
    repositories:[...new Set(catalog.packages.filter(p=>p.status==='published').map(p=>p.repository))].map(repository=>({repository,status:'ok',stars:100,openIssuesAndPRs:3}))};
  const input={schema:'ruvnet.constellation-adoption/1',consent:true,samples:[
    {sampleId:'1'.repeat(64),observedAt:end+'T01:00:00.000Z',packages:['metaharness','agentdb','ruvector']},
    {sampleId:'2'.repeat(64),observedAt:end+'T02:00:00.000Z',packages:['ruflo','ruvector']},
    {sampleId:'3'.repeat(64),observedAt:end+'T03:00:00.000Z',packages:['metaharness']},
    {sampleId:'4'.repeat(64),observedAt:end+'T04:00:00.000Z',packages:['ruv-fann-wsp']}
  ]};
  const report=analyzeConstellation(catalog,observations,adoption?{adoption:input}:undefined);
  return {catalog,observations,report};
}
