/** Actual Darwin numeric engine adapter. It never uses ShellEvaluator or a model provider. */
import { mkdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ExperimentError, InputError, fileBytes, finiteJson, inside, ordinaryAncestry, repository, validateOptions } from './guards.mjs';

const HEX = /^[a-f0-9]{64}$/;
const positive = value => typeof value === 'number' && Number.isFinite(value) && value > 0;

export function checkedGenome(genome) {
  if (!genome || Object.getPrototypeOf(genome) !== Object.prototype ||
      Object.keys(genome).length !== 1 || !Number.isInteger(genome.batchRows) || genome.batchRows < 1 || genome.batchRows > 128)
    throw new ExperimentError('genome must contain only integer batchRows from 1 to 128');
  return { batchRows: genome.batchRows };
}

/** Copies detach evaluator cache entries from the mutable upstream archive. */
export function checkedScore(card, variantId, genome, samples, context) {
  finiteJson(card);
  if (card?.variantId !== variantId || !positive(card.primary) || typeof card.regressed !== 'boolean' ||
      !Number.isFinite(card.noopRate) || card.noopRate < 0 || card.noopRate > 1 ||
      !Number.isFinite(card.costPerWin) || card.costPerWin < 0 || card.evaluatorError)
    throw new ExperimentError('evaluator returned invalid or failed numeric evidence');
  const raw = card.raw;
  if (!raw || raw.maximumAbsoluteError !== 0 || raw.batchRows !== genome.batchRows ||
      !Array.isArray(raw.samplesNs) || raw.samplesNs.length !== samples || !raw.samplesNs.every(positive) ||
      !HEX.test(raw.outputSha256) || typeof raw.cacheHit !== 'boolean' ||
      typeof raw.observationId !== 'string' || !raw.observationId || raw.observationId.length > 128)
    throw new ExperimentError('missing real timing, exact correctness, or observation evidence');
  for (const key of ['variant', 'model', 'totalRows', 'workloadSha256', 'outputSha256',
    'inputSha256', 'weightsSha256', 'sourceSha256', 'binarySha256', 'glueSha256', 'receiptSha256'])
    if (raw[key] !== context[key]) throw new ExperimentError(`numeric score changed its frozen ${key}`);
  if (![raw.meanNs, raw.medianNs, raw.rowsPerSecond, raw.estimatedBytes].every(positive) ||
      card.primary !== raw.rowsPerSecond) throw new ExperimentError('fitness does not match measured throughput');
  return structuredClone(card);
}

export function checkedConfirmation(value, baseline, winner, context, samples) {
  finiteJson(value);
  if (!value || value.verified !== true || !['passed', 'not_improved'].includes(value.status) ||
      typeof value.faster !== 'boolean' || value.baselineBatchRows !== baseline.batchRows ||
      value.candidateBatchRows !== winner.batchRows || value.minimumImprovement !== 0.05 ||
      value.maximumAbsoluteError !== 0 || !positive(value.meanSpeedRatio) ||
      !Array.isArray(value.ci95) || value.ci95.length !== 2 || !value.ci95.every(positive) ||
      value.ci95[0] > value.ci95[1] || value.faster !== (value.status === 'passed'))
    throw new ExperimentError('independent confirmation is missing, nonfinite, or inconsistent');
  for (const name of ['baselineSamplesNs', 'candidateSamplesNs'])
    if (!Array.isArray(value[name]) || value[name].length !== samples || !value[name].every(positive))
      throw new ExperimentError('confirmation needs bounded fresh paired timing samples');
  if (value.baselineSamplesNs.length !== value.candidateSamplesNs.length)
    throw new ExperimentError('confirmation sample pairs have mismatched lengths');
  if (value.workloadSha256 !== context.workloadSha256 ||
      value.faster !== (winner.batchRows !== baseline.batchRows && value.ci95[0] > 1.05))
    throw new ExperimentError('confirmation changed its workload or bypassed the 5% confidence policy');
  const mean = values => values.reduce((sum, item) => sum + item, 0) / values.length;
  const observedRatio = mean(value.baselineSamplesNs) / mean(value.candidateSamplesNs);
  if (Math.abs(value.meanSpeedRatio - observedRatio) > 1e-10 * Math.max(1, observedRatio))
    throw new ExperimentError('confirmation ratio disagrees with the raw paired timing samples');
  return structuredClone(value);
}

function record(value) {
  return { variantId: value.variant.id, genome: structuredClone(value.variant.genome), score: structuredClone(value.score) };
}

/** Tests may inject an evaluator/engine; production always uses the pinned real engine. */
export async function runEvolution(options, output, evaluator, engine) {
  let failure;
  try {
    const config = validateOptions(options);
    for (const name of ['evaluate', 'confirmWinner', 'verifyInputs', 'close'])
      if (typeof evaluator?.[name] !== 'function') throw new InputError(`evaluator must implement ${name}`);
    const context = structuredClone(evaluator.context);
    finiteJson(context);
    if (context?.variant !== config.variant || context?.model !== config.model || context?.totalRows !== config.totalRows ||
        ['workloadSha256', 'outputSha256', 'inputSha256', 'weightsSha256', 'sourceSha256', 'binarySha256', 'glueSha256', 'receiptSha256']
          .some(key => !HEX.test(context[key]))) throw new ExperimentError('evaluator context does not bind the configured real workload');
    const evolveNumeric = engine ?? (await import('@metaharness/darwin')).evolveNumeric;
    if (typeof evolveNumeric !== 'function') throw new ExperimentError('pinned Darwin numeric engine is unavailable');
    const worstCaseDispatches = 1 + config.children + 2 * config.children * (config.generations - 1);
    const trials = [], observations = new Map();
    let active = false, cacheHits = 0, dispatchCount = 0;
    const boundedEvaluator = { evaluate: async (input, variantId) => {
      if (active || ++dispatchCount > worstCaseDispatches || dispatchCount > 12)
        throw new ExperimentError('independent numeric dispatch/concurrency budget exceeded');
      const genome = checkedGenome(input);
      if (typeof variantId !== 'string' || !variantId || variantId.length > 128)
        throw new ExperimentError('invalid numeric variant identity');
      active = true;
      try {
        const score = checkedScore(await evaluator.evaluate(Object.freeze({ ...genome }), variantId), variantId, genome, config.samples, context);
        const { observationId, cacheHit } = score.raw;
        if (cacheHit) {
          if (observations.get(observationId) !== genome.batchRows) throw new ExperimentError('cache evidence has no matching prior observation');
          cacheHits++;
        } else {
          if (observations.has(observationId)) throw new ExperimentError('a new observation reused an existing identity');
          observations.set(observationId, genome.batchRows);
        }
        trials.push({ variantId, genome, score: structuredClone(score) });
        return structuredClone(score);
      } finally { active = false; }
    } };
    const workRoot = join(output, 'darwin');
    await mkdir(workRoot, { mode: 0o700 }); // Existing archives are never loaded or resumed.
    const result = await evolveNumeric({ genomeSpec: { batchRows: { min: 1, max: 128, scale: 'linear', type: 'int', default: config.baselineBatchRows } },
      baselineGenome: { batchRows: config.baselineBatchRows }, evaluator: boundedEvaluator, generations: config.generations,
      childrenPerGeneration: config.children, seed: config.seed, concurrency: 1,
      mutationSigma: 0.25, crossover: false, workRoot });
    if (!result?.baseline || !Array.isArray(result.records) || result.records.length !== trials.length || trials.length === 0)
      throw new ExperimentError('numeric archive does not match independently counted dispatches');
    const baseline = record(result.baseline);
    checkedGenome(baseline.genome); checkedScore(baseline.score, baseline.variantId, baseline.genome, config.samples, context);
    if (baseline.genome.batchRows !== config.baselineBatchRows || baseline.score.regressed)
      throw new ExperimentError('baseline is not independently verified against the configured capacity');
    let winner = result.winner ? record(result.winner) : baseline;
    let winnerRejected = null;
    checkedGenome(winner.genome); checkedScore(winner.score, winner.variantId, winner.genome, config.samples, context);
    if (!trials.some(item => item.variantId === winner.variantId && JSON.stringify(item.score) === JSON.stringify(winner.score)))
      throw new ExperimentError('numeric winner has no matching measured trial');
    if (winner.score.regressed) { winnerRejected = { variantId: winner.variantId, reason: 'regressed archive winner' }; winner = baseline; }
    const confirmation = checkedConfirmation(await evaluator.confirmWinner({ baselineGenome: baseline.genome,
      winnerGenome: winner.genome, variantId: winner.variantId }), baseline.genome, winner.genome, context, config.samples);
    const finalInputs = await evaluator.verifyInputs();
    if (finalInputs?.status !== 'passed' || finalInputs.sourceSha256 !== context.sourceSha256)
      throw new ExperimentError('final source-aware input verification did not pass');
    const answer = { status: 'passed', modelCalls: 0, sourceApplied: false, inputVerification: 'passed',
      trialBudget: { absoluteLimit: 12, worstCaseDispatches, dispatchCount,
        actualMeasurementCount: observations.size, cacheHits, concurrency: 1 },
      trials, baseline, winner, winnerRejected, winnerLineage: result.winnerLineage,
      confirmation, recommendation: confirmation.faster ? { ...winner.genome } : { ...baseline.genome },
      evaluatorContext: context, evaluatorStats: evaluator.stats ?? {},
      scope: 'Runtime batch capacity only; candidate search and fresh paired confirmation are separate. No source or configuration was applied.' };
    finiteJson(answer);
    return answer;
  } catch (error) { failure = error; throw error; }
  finally {
    if (typeof evaluator?.close === 'function') {
      try { await evaluator.close(); } catch (error) { if (!failure) throw error; }
    }
  }
}

async function worker() {
  const output = resolve(process.argv[3] ?? '');
  const request = JSON.parse((await fileBytes(output, 'request.json', 16384)).toString('utf8'));
  if (process.argv[2] !== '--worker' || request.nonce !== process.argv[4] || request.options?.output !== output)
    throw new InputError('invalid private worker request');
  const options = validateOptions(request.options);
  options.repo = await repository(options.repo);
  await ordinaryAncestry(output);
  if (inside(options.repo, output) || inside(output, options.repo)) throw new InputError('worker output must be external');
  const pending = JSON.parse((await fileBytes(output, 'receipt.json', 65536)).toString('utf8'));
  if (pending.schema !== 'ruv-fann.metaharness/1' || pending.status !== 'running' || pending.operation !== 'optimize')
    throw new InputError('worker needs a new coordinator-owned pending receipt');
  const { createEvaluator } = await import('./wasm-evaluator.mjs');
  const evaluator = await createEvaluator(options);
  const result = await runEvolution(options, output, evaluator);
  await writeFile(join(output, 'search.json'), JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  process.stdout.write(JSON.stringify({ status: 'passed', searchFile: 'search.json' }) + '\n');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  worker().catch(error => { process.stderr.write(JSON.stringify({ status: 'failed', error: error.message }) + '\n'); process.exitCode = 1; });
}
