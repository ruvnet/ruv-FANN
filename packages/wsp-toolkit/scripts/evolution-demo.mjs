import { mkdtemp,mkdir,writeFile } from 'node:fs/promises';
import { resolve,join } from 'node:path';
import { openDemoRegistry } from '../tooling/agentdb/runtime.mjs';
import { runEvolutionScenario } from '../test/evolution-scenario.mjs';
const root=resolve(new URL('..',import.meta.url).pathname);
await mkdir(join(root,'.artifacts'),{recursive:true});
const directory=await mkdtemp(join(root,'.artifacts','evolution-'));
const result=await runEvolutionScenario(openDemoRegistry,directory);
const filename=join(directory,'report.json');
await writeFile(filename,JSON.stringify({...result,database:join(directory,'evolution.sqlite')},null,2)+'\n');
console.log(JSON.stringify({stopReason:result.report.stopReason,rounds:result.report.rounds,promotions:result.report.promotions,
  proposalWorkers:result.report.workerThreadIds.length,independentRejections:result.report.events.filter(e=>e.type==='rejected').length,
  reservedWork:result.report.budget.reservedWork,durability:result.durability,rollback:result.rollback,report:filename},null,2));
