import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
const repo = 'ruvnet/ruv-FANN';
const sha = process.env.EXPECTED_SHA;
if (process.env.GITHUB_ACTIONS !== 'true' || process.env.GITHUB_WORKFLOW !== 'protected-release' || process.env.GITHUB_REF !== 'refs/heads/main' || process.env.GITHUB_SHA !== sha || !/^[0-9a-f]{40}$/.test(sha)) throw new Error('Exact-main protected publication required');
const api = path => JSON.parse(execFileSync('gh', ['api', path], { encoding: 'utf8' }));
const runs = api(`repos/${repo}/actions/workflows/npm-modules.yml/runs?head_sha=${sha}&status=success&per_page=100`).workflow_runs;
const expectedName = `npm-module-candidate-${sha}`;
let candidate;
for (const run of runs) {
  if (run.head_sha !== sha || run.head_repository?.full_name !== repo || run.conclusion !== 'success' || run.status !== 'completed') continue;
  const artifacts = api(`repos/${repo}/actions/runs/${run.id}/artifacts?per_page=100`).artifacts;
  candidate = artifacts.find(artifact => artifact.name === expectedName && !artifact.expired);
  if (candidate) break;
}
if (!candidate) throw new Error(`No successful qualification artifact for exact source ${sha}`);
const out = resolve(process.argv[2]); mkdirSync(out, { recursive: true });
const archive = join(out, 'candidate.zip');
writeFileSync(archive, execFileSync('gh', ['api', `repos/${repo}/actions/artifacts/${candidate.id}/zip`], { maxBuffer: 20 * 1024 * 1024 }));
execFileSync('unzip', ['-q', archive, '-d', out]);
const seal = JSON.parse(readFileSync(join(out, 'candidate-seal.json'), 'utf8'));
if (seal.sourceSha !== sha) throw new Error('Downloaded candidate source mismatch');
console.log(`Downloaded ${expectedName} from successful qualification, artifact ${candidate.id}.`);
