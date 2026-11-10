#!/usr/bin/env node
// Fetch the sole candidate artifact produced by the successful qualification run for an exact SHA.
import { execFileSync } from 'node:child_process';
import { readFile, readdir, mkdir, mkdtemp, rename, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, basename } from 'node:path';
import { createHash } from 'node:crypto';

const args = process.argv.slice(2);
const get = name => {
  const i = args.indexOf(`--${name}`);
  if (i < 0 || !args[i + 1] || args.indexOf(`--${name}`, i + 1) >= 0) throw new Error(`Expected one --${name}`);
  return args[i + 1];
};
const sha = get('expected-sha');
const output = resolve(get('output'));
if (!/^[0-9a-f]{40}$/.test(sha)) throw new Error('Invalid expected source SHA');
const run = (cmd, argv) => execFileSync(cmd, argv, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'], timeout: 5 * 60_000 }).trim();
const repo = process.env.GITHUB_REPOSITORY;
if (repo !== 'ruvnet/ruv-FANN' || process.env.GITHUB_ACTIONS !== 'true') throw new Error('Candidate download is restricted to the upstream GitHub Actions workflow');
const runs = JSON.parse(run('gh', ['api', `repos/${repo}/actions/workflows/wsp-modules.yml/runs?head_sha=${sha}&status=completed&per_page=100`])).workflow_runs;
const candidates = runs.filter(item => item.head_sha === sha && item.head_repository?.full_name === repo &&
  item.conclusion === 'success' && ['pull_request', 'workflow_dispatch'].includes(item.event));
if (!candidates.length) throw new Error(`No successful upstream qualification run for ${sha}`);
candidates.sort((a,b) => b.run_number - a.run_number);
const artifactName = `wsp-candidate-${sha}`;
let runId;
for (const candidate of candidates) {
  const artifacts = JSON.parse(run('gh', ['api', `repos/${repo}/actions/runs/${candidate.id}/artifacts?per_page=100`])).artifacts;
  if (artifacts.filter(item => item.name === artifactName && !item.expired).length === 1) {
    runId = String(candidate.id);
    break;
  }
}
if (!runId) throw new Error('No successful qualification run contains the exact unexpired candidate artifact');
const temp = await mkdtemp(join(tmpdir(), 'wsp-candidate-download-'));
try {
  run('gh', ['run', 'download', runId, '-R', repo, '-n', artifactName, '-D', temp]);
  const manifest = JSON.parse(await readFile(join(temp, 'package-manifest.json'), 'utf8'));
  const seal = JSON.parse(await readFile(join(temp, 'candidate-seal.json'), 'utf8'));
  const bytes = await readFile(join(temp, manifest.tarball));
  const digest = createHash('sha256').update(bytes).digest('hex');
  if (manifest.sourceSha !== sha || seal.sourceSha !== sha || seal.status !== 'passed' || seal.tarball !== manifest.tarball ||
      basename(manifest.tarball) !== manifest.tarball || seal.sha256 !== digest || manifest.sha256 !== digest ||
      manifest.name !== 'ruv-fann-wsp' || manifest.version !== '0.2.0' || seal.releaseTag !== 'ruv-fann-wsp-v0.2.0')
    throw new Error('Downloaded candidate is not the exact successful sealed archive requested');
  await mkdir(output, { recursive: true });
  for (const name of await readdir(temp)) {
    try { await rename(join(temp, name), join(output, name)); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
} finally { await rm(temp, { recursive: true, force: true }); }
console.log(JSON.stringify({ runId, sourceSha: sha, artifact: artifactName, status: 'downloaded-and-verified' }));
