#!/usr/bin/env node
/** Actual extracted npm consumer proof. No dependency downloads or lifecycle scripts. */
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { externalPath } from '../scripts/paths.mjs';

const directory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
if (process.argv.length !== 4 || process.argv[2] !== '--output') throw new Error('Usage: packed-consumer.mjs --output <NEW_EXTERNAL_DIR>');
const output = await externalPath(process.argv[3], resolve(directory, '../..'));
await mkdir(output, { recursive: true, mode: 0o700 });
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const build = JSON.parse(await readFile(join(directory, 'build-info.json')));
const before = hash(await readFile(join(directory, 'ruv_fann.node')));
const report = { schema: 'ruv-fann.napi-consumer/1', status: 'failed', nativeLoaded: false,
  node: process.version, platform: process.platform, arch: process.arch, binarySha256: before,
  buildSourceHashes: build.inputsBefore, commands: [] };
async function run(label, program, args, cwd) {
  const child = spawnSync(program, args, { cwd, encoding: 'utf8', timeout: 120000, maxBuffer: 4 * 1024 * 1024 });
  report.commands.push({ label, program, args, exitCode: child.status });
  if (child.stdout) await writeFile(join(output, label + '.stdout.log'), child.stdout);
  if (child.stderr) await writeFile(join(output, label + '.stderr.log'), child.stderr);
  if (child.error || child.status !== 0) throw new Error(`${label} failed: ${child.error?.message ?? child.stderr}`);
  return child.stdout;
}
try {
  const packs = JSON.parse(await run('pack', 'npm', ['pack', '--json', '--ignore-scripts', '--pack-destination', output], directory));
  if (packs.length !== 1) throw new Error('Expected one actual package artifact');
  const artifact = join(output, packs[0].filename);
  const consumer = join(output, 'consumer'); await mkdir(consumer);
  await writeFile(join(consumer, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  await run('install-local', 'npm', ['install', '--ignore-scripts', '--offline', '--no-audit', '--no-fund', artifact], consumer);
  const script = `
    import assert from 'node:assert/strict';
    import {createRequire} from 'node:module';
    import {readFileSync} from 'node:fs';
    import {createHash} from 'node:crypto';
    import {NativePreparedNetwork,NativeTrainableNetwork} from '@ruv-fann/napi';
    import {trainingProof} from ${JSON.stringify(new URL('./training-proof.mjs', import.meta.url).href)};
    const require=createRequire(import.meta.url), cjs=require('@ruv-fann/napi'), raw=require('@ruv-fann/napi/native');
    assert.equal(raw.NativePreparedNetwork,NativePreparedNetwork);assert.equal(cjs.NativePreparedNetwork,NativePreparedNetwork);
    const json=JSON.stringify({schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:[2],bias:[1],activation:'Linear',steepness:1}]});
    const p=raw.NativePreparedNetwork.from_dense_layers_json(json);assert.deepEqual([...p.run(new Float64Array([3]))],[7]);
    const alias=new Float64Array([4]);p.run_into(alias,alias);assert.deepEqual([...alias],[9]);
    const output=new Float64Array([77]);assert.throws(()=>p.run_into(new Float64Array([NaN]),output));assert.deepEqual([...output],[77]);
    assert.throws(()=>p.run(new Float64Array(new SharedArrayBuffer(8))));
    const batch=p.create_batch(2);p.free();p.free();const result=new Float64Array(2);batch.run_into(new Float64Array([1,2]),result,2);assert.deepEqual([...result],[3,5]);batch.free();
    const training=await trainingProof(NativeTrainableNetwork);
    const binary=require.resolve('@ruv-fann/napi/native');assert.ok(binary.endsWith('.node'));
    const binarySha256=createHash('sha256').update(readFileSync(binary)).digest('hex');
    console.log(JSON.stringify({nativeLoaded:true,binarySha256,training,checks:{esm:true,cjs:true,directBinding:true,alias:true,atomic:true,lifecycle:true}}));
  `;
  await writeFile(join(consumer, 'verify.mjs'), script);
  const actual = JSON.parse(await run('consume', process.execPath, ['verify.mjs'], consumer));
  if (actual.binarySha256 !== before) throw new Error('Packed native binary differs from built binary');
  const after = hash(await readFile(join(directory, 'ruv_fann.node')));
  if (after !== before) throw new Error('Native binary changed during consumer checks');
  Object.assign(report, actual, { status: 'passed', unchangedBinary: true,
    artifact: { filename: packs[0].filename, sha256: hash(await readFile(artifact)), size: packs[0].size },
    scope: 'Actual offline installed native addon, raw binding gateway and real core training; no model providers or fallback.' });
} catch (error) { report.error = error.message; throw error; }
finally { await writeFile(join(output, 'receipt.json'), JSON.stringify(report, null, 2) + '\n'); }
process.stdout.write(JSON.stringify({ status: report.status, receiptPath: join(output, 'receipt.json'), binarySha256: report.binarySha256 }) + '\n');
