import {readFile,writeFile} from 'node:fs/promises';
import {resolve,dirname,relative,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {validateCatalog,validateModuleManifest,CONTRACTS} from 'ruvnet-constellation-contracts';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const catalog=validateCatalog(JSON.parse(await readFile(join(root,'packages/constellation/catalog.json'),'utf8')));
const check=process.argv.slice(2).includes('--check');
if(process.argv.slice(2).some(arg=>arg!=='--check'))throw new Error('usage: node scripts/constellation-links.mjs [--check]');
const entries=catalog.packages.filter(p=>p.status==='candidate'&&p.repository===catalog.repository&&p.directory?.startsWith('packages/wsp-toolkit/packages/'));
const start='<!-- constellation-links:start -->',end='<!-- constellation-links:end -->';
let changes=0;
for(const entry of entries) {
  const directory=resolve(root,'../..',entry.directory);
  if(!directory.startsWith(join(root,'packages')+'/'))throw new Error('candidate directory escapes workspace');
  const readmePath=join(directory,'README.md'),manifestPath=join(directory,'package.json');
  const manifest=JSON.parse(await readFile(manifestPath,'utf8'));
  if(manifest.name!==entry.name||manifest.version!==entry.version)throw new Error('catalog package identity differs from source');
  const sourceLink=target=>relative(directory,join(root,'packages',target,'README.md')).split('\\').join('/');
  const block=`${start}\n\nExplore the [constellation onramp](${sourceLink('constellation')}) and [shared contracts](${sourceLink('contracts')}). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).\n\nThese links describe integrations; they do not add install-time dependencies. Workspace package candidates and namespace proposals are listed explicitly in the catalog.\n\n${end}`;
  const before=await readFile(readmePath,'utf8');
  if((before.includes(start)||before.includes(end))&&!(before.includes(start)&&before.includes(end)))throw new Error('incomplete constellation README markers');
  const after=before.includes(start)?before.slice(0,before.indexOf(start))+block+before.slice(before.indexOf(end)+end.length):before.trimEnd()+'\n\n'+block+'\n';
  const dependencies={...manifest.dependencies,...manifest.peerDependencies,...manifest.optionalDependencies};
  const module=validateModuleManifest({schema:CONTRACTS.module,name:entry.name,version:entry.version,role:entry.role,
    contracts:[...new Set([CONTRACTS.module,...(entry.contracts??[])])],integrations:Object.keys(dependencies).filter(name=>catalog.packages.some(p=>p.name===name)).sort()});
  const modulePath=join(directory,'constellation.json');let oldModule;
  try{oldModule=await readFile(modulePath,'utf8');}catch(error){if(error.code!=='ENOENT')throw error;}
  const moduleJSON=JSON.stringify(module,null,2)+'\n';
  const nextManifest={...manifest,files:[...new Set([...(manifest.files??[]),'constellation.json'])]};
  const changed=after!==before||oldModule!==moduleJSON||!manifest.files?.includes('constellation.json');
  if(changed) {
    changes++;
    if(!check){await writeFile(readmePath,after);await writeFile(modulePath,moduleJSON);await writeFile(manifestPath,JSON.stringify(nextManifest,null,2)+'\n');}
  }
}
if(check&&changes)throw new Error(`${changes} constellation metadata/README blocks need regeneration`);
console.log(JSON.stringify({packages:entries.length,changes,mode:check?'check':'write'}));
