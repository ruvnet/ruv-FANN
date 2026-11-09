#!/usr/bin/env node
/** Run serially in the coordinator's quiet window: actual Darwin against native FANN. */
import { createRequire } from 'node:module';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { externalPath } from '../scripts/paths.mjs';
import { tuneBatchRows } from '../../../integrations/ruv-fann/src/metaharness.mjs';
import { dense } from './fixtures.mjs';
if (process.argv.length !== 4 || process.argv[2] !== '--output') throw new Error('Usage: tuner-compat.mjs --output <NEW_EXTERNAL_JSON>');
const output = await externalPath(process.argv[3], fileURLToPath(new URL('../../../', import.meta.url)));
const require = createRequire(import.meta.url), binary = require.resolve('../ruv_fann.node');
const { NativePreparedNetwork } = require(binary);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const before = hash(await readFile(binary));
const prepared = NativePreparedNetwork.from_dense_layers_json(dense());
let receipt;
try {
  const inputs = Float64Array.from({ length: 256 }, (_, index) => (index % 11 - 5) / 9);
  receipt = await tuneBatchRows({ prepared, inputs, modelId: 'native-real-darwin-compatibility', baselineBatchRows: 128,
    generations: 1, children: 1, samples: 5, seed: 7, timeoutMs: 30000 });
} finally { prepared.free(); }
const after = hash(await readFile(binary));
if (after !== before || receipt.status !== 'passed' || receipt.confirmation?.verified !== true) throw new Error('Native Darwin compatibility failed');
const report = { schema: 'ruv-fann.napi-tuner/1', status: 'passed', nativeLoaded: true,
  node: process.version, binarySha256: before, unchangedBinary: true, actualDarwin: true, receipt };
await writeFile(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
process.stdout.write(JSON.stringify({ status: 'passed', receiptPath: output, confirmation: receipt.confirmation.status }) + '\n');
