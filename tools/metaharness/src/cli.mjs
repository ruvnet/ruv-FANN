#!/usr/bin/env node
/** Private, bounded MetaHarness CLI. Installs and source edits are outside this tool. */
import { lstat, mkdir, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULTS, ExperimentError, InputError, TOOL_ROOT, assertNoScoreModel, boundedProcess, digest, externalOutput,
  fileBytes, finiteJson, inputSnapshot, packagePins, parseArgs, repository } from './guards.mjs';

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** Exact public high-signal files plus public structural signals, never private agent config. */
export async function scoreProjection(repo, output) {
  const projection = join(output, 'score-input');
  await mkdir(projection, { mode: 0o700 });
  const files = {};
  for (const name of ['README.md', 'Cargo.toml', 'package.json']) {
    const bytes = await fileBytes(repo, name, 1048576);
    await writeFile(join(projection, name), bytes, { flag: 'wx' });
    files[name] = digest(bytes);
  }
  const directories = [];
  for (const name of ['tests', 'docs', '.github/workflows']) {
    let current = repo;
    try {
      for (const component of name.split('/')) {
        current = join(current, component);
        const info = await lstat(current);
        if (info.isSymbolicLink() || !info.isDirectory()) throw new InputError('score projection directory must be ordinary');
      }
      await mkdir(join(projection, name), { recursive: true }); directories.push(name);
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  return { path: projection, files, directories, excluded: ['.mcp.json', '.claude', '.swarm', 'credentials', 'source execution'] };
}

export async function main(argv) {
  let output, report;
  try {
    const options = parseArgs(argv);
    options.repo = await repository(options.repo);
    const pins = await packagePins();
    const inputsBefore = await inputSnapshot(options.repo);
    const worstCaseDispatches = 1 + options.children + 2 * options.children * (options.generations - 1);
    if (options.operation === 'plan') {
      process.stdout.write(JSON.stringify({ schema: 'ruv-fann.metaharness/1', status: 'planned', readOnly: true,
        pins, options, defaults: DEFAULTS, inputsBefore, modelCalls: 0, sourceApplied: false,
        trialBudget: { absoluteLimit: 12, worstCaseDispatches, concurrency: 1 },
        scope: 'Actual Darwin numeric evolution of batchRows only; scores require real WASM measurements and fresh paired confirmation.' }, null, 2) + '\n');
      return 0;
    }
    output = await externalOutput(options.output, options.repo); options.output = output;
    report = { schema: 'ruv-fann.metaharness/1', status: 'running', operation: options.operation,
      pins, options, inputsBefore, modelCalls: 0, sourceApplied: false,
      runtime: { node: process.version, arch: process.arch, platform: process.platform } };
    await writeFile(join(output, 'receipt.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
    try {
      if (options.operation === 'score') {
        assertNoScoreModel();
        const projection = await scoreProjection(options.repo, output);
        const child = await boundedProcess([join(TOOL_ROOT, 'node_modules/metaharness/dist/bin.js'),
          'score', projection.path, '--json'], { cwd: output, timeoutMs: options.timeoutMs });
        await writeFile(join(output, 'score.stdout.log'), child.stdout, { flag: 'wx' });
        await writeFile(join(output, 'score.stderr.log'), child.stderr, { flag: 'wx' });
        const scorecard = JSON.parse(child.stdout); finiteJson(scorecard);
        if (scorecard.schema !== 1 || scorecard.error) throw new ExperimentError('actual MetaHarness score CLI failed');
        Object.assign(report, { status: 'passed', scorecard, projection, staticAnalysisOnly: true,
          scope: 'Upstream heuristic score of bounded public metadata; estimates are not measured runtime cost, security, or model performance.' });
      } else {
        const nonce = randomUUID();
        await writeFile(join(output, 'request.json'), JSON.stringify({ nonce, options }) + '\n', { flag: 'wx' });
        const child = await boundedProcess([join(TOOL_ROOT, 'src/evolution.mjs'), '--worker', output, nonce],
          { cwd: output, timeoutMs: options.timeoutMs });
        await writeFile(join(output, 'worker.stdout.log'), child.stdout, { flag: 'wx' });
        await writeFile(join(output, 'worker.stderr.log'), child.stderr, { flag: 'wx' });
        const result = JSON.parse((await fileBytes(output, 'search.json', 2097152)).toString('utf8'));
        finiteJson(result);
        if (result.status !== 'passed' || result.inputVerification !== 'passed' || result.modelCalls !== 0 || result.sourceApplied !== false)
          throw new ExperimentError('numeric worker did not produce verified evidence');
        Object.assign(report, result);
      }
      report.inputsAfter = await inputSnapshot(options.repo);
      if (!same(inputsBefore, report.inputsAfter)) throw new ExperimentError('repository/tool inputs changed during the experiment');
      finiteJson(report);
    } catch (error) {
      report.status = 'failed'; report.error = error.message;
      if (error.result) report.childFailure = error.result;
      try { report.inputsAfter = await inputSnapshot(options.repo); }
      catch (failure) { report.inputCaptureError = failure.message; }
      throw error;
    } finally {
      await writeFile(join(output, 'receipt.json'), JSON.stringify(report, null, 2) + '\n');
    }
    process.stdout.write(JSON.stringify({ status: report.status, receiptPath: join(output, 'receipt.json'),
      ...(report.confirmation ? { recommendation: report.recommendation,
        confirmation: { status: report.confirmation.status, meanSpeedRatio: report.confirmation.meanSpeedRatio, ci95: report.confirmation.ci95 } } : {}) }, null, 2) + '\n');
    return 0;
  } catch (error) {
    process.stdout.write(JSON.stringify({ status: error instanceof InputError ? 'invalid_input' : 'failed',
      error: error.message, ...(output ? { receiptPath: join(output, 'receipt.json') } : {}) }) + '\n');
    return error instanceof InputError ? 2 : 1;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  process.exitCode = await main(process.argv.slice(2));
