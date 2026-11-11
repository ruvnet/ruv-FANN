export const start='2024-01-01',end='2024-02-11';
export const days=Array.from({length:42},(_,i)=>({day:new Date(Date.parse(start+'T00:00:00Z')+i*86400000).toISOString().slice(0,10),downloads:i+1}));
export function catalog(){return {schema:'ruvnet.constellation-catalog/1',repository:'owner/repo',packages:[
  {name:'hub',role:'nucleus',family:'entry',status:'published',version:'1.0.0',repository:'owner/repo',description:'Entry'},
  {name:'core',role:'primitive',family:'compute',status:'published',version:'2.0.0',repository:'owner/repo',description:'Core'},
  {name:'candidate',role:'satellite',family:'compute',status:'candidate',version:'0.1.0',repository:'owner/repo',description:'Local source'}
],profiles:[{id:'all',description:'Both',packages:['hub','core']}]};}
export function observations(){return {schema:'ruvnet.constellation-observations/1',observedAt:'2024-02-12T00:00:00Z',start,end,packages:[
  {name:'hub',status:'ok',days:structuredClone(days),version:'1.1.0',metadataStatus:'ok',dependencies:['core']},
  {name:'core',status:'ok',days:days.map(d=>({...d,downloads:d.downloads*2})),version:'2.0.0',metadataStatus:'ok',dependencies:[]},
  {name:'candidate',status:'candidate',days:[]}
],repositories:[{repository:'owner/repo',status:'ok',stars:10,openIssuesAndPRs:3}]};}
export function fixtureFetch(calls=[]){return async(url,options)=>{calls.push({url,options});const u=new URL(url);if(u.hostname==='api.npmjs.org'){const name=decodeURIComponent(u.pathname.split('/').at(-1));return Response.json({package:name,start,end,downloads:days.map(d=>({...d,downloads:d.downloads*(name==='core'?2:1)}))});}if(u.hostname==='registry.npmjs.org'){const name=decodeURIComponent(u.pathname.slice(1,-'/latest'.length));return Response.json({name,version:'1.0.0',dependencies:name==='hub'?{core:'^2'}:{}});}return Response.json({full_name:'owner/repo',stargazers_count:10,open_issues_count:3});};}
