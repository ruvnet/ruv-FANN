#!/usr/bin/env node
import {readCatalog,planProfile,scaffoldSatellite} from './index.js';
const help=`constellation — explicit local discovery, planning and satellite scaffolding

Usage:
  constellation help
  constellation catalog [--json]
  constellation plan <profile> [--json]
  constellation scaffold <new-directory> --name <npm-name> [--description <text>] [--integration <npm-name> ...] [--json]

Plans describe verified published pins and local candidates separately. No command
installs dependencies, starts an agent, collects telemetry or publishes a package.
Scaffolding requires Linux with /proc/self/fd, creates a new directory only,
and refuses existing targets and symlinks. Catalog and planning remain portable.

Nucleus: https://www.npmjs.com/package/metaharness
         https://www.npmjs.com/package/@claude-flow/cli
Memory:  https://www.npmjs.com/package/agentdb
Vectors: https://www.npmjs.com/package/ruvector
Weights: https://www.npmjs.com/package/ruv-fann-wsp
`;
try {
  const args=process.argv.slice(2),command=args.shift()??'help';
  if(command==='help'||command==='--help'||command==='-h'){
    if(args.length)throw new Error('help accepts no arguments');
    process.stdout.write(help);
  }else if(command==='catalog'){
    if(args.length&&!(args.length===1&&args[0]==='--json'))throw new Error('catalog accepts only --json');
    process.stdout.write(JSON.stringify(await readCatalog(),null,2)+'\n');
  }else if(command==='plan'){
    const profile=args.shift();if(!profile||(args.length&&!(args.length===1&&args[0]==='--json')))throw new Error('usage: constellation plan <profile> [--json]');
    process.stdout.write(JSON.stringify(planProfile(await readCatalog(),profile),null,2)+'\n');
  }else if(command==='scaffold'){
    const directory=args.shift();if(!directory||directory.startsWith('--'))throw new Error('scaffold requires an explicit new directory');
    const config={directory},integrations=[];const seen=new Set();
    while(args.length){
      const flag=args.shift();
      if(flag==='--json'){if(seen.has(flag))throw new Error('duplicate --json');seen.add(flag);continue;}
      if(!['--name','--description','--integration'].includes(flag))throw new Error('unknown scaffold option');
      const value=args.shift();if(!value||value.startsWith('--'))throw new Error(`missing value for ${flag}`);
      if(flag==='--integration')integrations.push(value);
      else {if(seen.has(flag))throw new Error(`duplicate ${flag}`);seen.add(flag);config[flag.slice(2)]=value;}
    }
    if(integrations.length)config.integrations=integrations;
    process.stdout.write(JSON.stringify(await scaffoldSatellite(config),null,2)+'\n');
  }else throw new Error('unknown command; use constellation help');
}catch(error){process.stderr.write(`constellation: ${error.message}\n`);process.exitCode=2;}
