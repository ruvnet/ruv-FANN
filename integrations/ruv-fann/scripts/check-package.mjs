/** Check an extracted npm pack using the reviewed installed dependency graph. */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, realpath, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { packageRoot, externalEvidencePath } from '../test/e2e/fixture.mjs';
import { runAcceptance } from '../test/e2e/acceptance.mjs';

if (process.argv.length !== 4 || process.argv[2] !== '--output') throw new Error('Use --output NEW_EXTERNAL_DIRECTORY');
const output = await externalEvidencePath(process.argv[3]);
await mkdir(output, { mode: 0o700 });
const packed = JSON.parse(execFileSync('npm', ['pack', '--ignore-scripts', '--json', '--pack-destination', output],
  { cwd: packageRoot, encoding: 'utf8', timeout: 30000, maxBuffer: 1048576 }))[0];
const tarball = join(output, packed.filename);
const entries = execFileSync('tar', ['-tzf', tarball], { encoding: 'utf8', timeout: 10000, maxBuffer: 1048576 }).trim().split('\n');
assert(entries.length < 100 && entries.every(name => name.startsWith('package/') && !name.split('/').includes('..')));
assert(!entries.some(name => /node_modules|\.swarm|\.env|test\/|examples\//.test(name)));
execFileSync('tar', ['-xzf', tarball, '-C', output], { timeout: 10000 });
// This check verifies packaged runtime isolation, reusing the already audited
// exact dependency install. It does not fetch or resolve a second graph.
await symlink(await realpath(join(packageRoot, 'node_modules')), join(output, 'node_modules'), 'dir');
const extracted = join(output, 'package');
const manifest = JSON.parse(await readFile(join(extracted, 'package.json'), 'utf8'));
for (const spec of Object.values(manifest.exports)) {
  await readFile(join(extracted, spec.import)); await readFile(join(extracted, spec.types));
}
const api = await import(pathToFileURL(join(extracted, manifest.exports['.'].import)));
const vectorApi = await import(pathToFileURL(join(extracted, manifest.exports['./ruvector'].import)));
const tuner = await import(pathToFileURL(join(extracted, manifest.exports['./metaharness'].import)));
const node = await import(pathToFileURL(join(extracted, manifest.exports['./node'].import)));
assert((await node.ruvectorWasmBytes()).length > 8);
const sourceEvidence = await runAcceptance({ api, vectorApi, tuner });
const receipt = { schema: 'ruv-fann.integration-package/1', status: 'passed', version: manifest.version,
  actualConsumers: 1, actualNeuralInference: true, actualRuVector: true, actualDarwin: true,
  coreAndSimd: true, sourceEvidence,
  dependencyScope: 'Reviewed installed dependency graph reused; no second installation or registry publication.',
  tarballSha256: createHash('sha256').update(await readFile(tarball)).digest('hex'), entries };
await writeFile(join(output, 'receipt.json'), JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ status: 'passed', receipt: join(output, 'receipt.json'), actualConsumers: 1 }));
