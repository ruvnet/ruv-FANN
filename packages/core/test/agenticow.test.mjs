import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import * as core from '../src/index.js';
import * as memory from '../src/capabilities/agenticow.js';
import * as wsp from '../src/capabilities/wsp.js';
import * as lifecycle from '../src/capabilities/lifecycle.js';
import {qualifyAgenticow} from './agenticow-scenario.mjs';

test('native Agenticow branches isolate rejected weight edits, gate promotion and persist rollback',async()=>{
  const directory=await mkdtemp(join(tmpdir(),'fann-agenticow-'));
  try {
    assert.equal(memory.default.AgenticMemory,memory.AgenticMemory);
    const proof=await qualifyAgenticow({core,memory,wsp,lifecycle},directory);
    assert.deepEqual(proof,{version:'0.3.0',native:true,branchIsolation:true,rejectedCandidateIsolated:true,promoted:true,checkpointRollback:true,reopened:true});
  } finally {await rm(directory,{recursive:true,force:true});}
});
