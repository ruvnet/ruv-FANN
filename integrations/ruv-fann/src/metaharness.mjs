// Explicit Node-only subpath. Importing the package root never loads Darwin.
import {createHash} from 'node:crypto';
import {mkdtemp, open, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createEncoder, preparedDimensions, checkedInputs} from './encoder.mjs';
import {calibrate, deadline, mean, measure, median, pairedRatio, positive} from './timing.mjs';

const MAX_BUFFERS = 64 * 1024 * 1024;
const DEFAULTS = Object.freeze({baselineBatchRows: 128, generations: 1, children: 1, samples: 10, seed: 7, timeoutMs: 30000});
const BOUNDS = {baselineBatchRows: [1, 128], generations: [1, 3], children: [1, 2],
  samples: [5, 20], seed: [0, 4294967295], timeoutMs: [100, 120000]};
const hash = values => createHash('sha256').update(Buffer.from(values.buffer, values.byteOffset, values.byteLength)).digest('hex');
const clone = value => structuredClone(value);

async function darwinIdentity(check) {
  const manifest = new URL('../package.json', import.meta.resolve('@metaharness/darwin'));
  const handle = await open(manifest, 'r');
  try {
    const stat = await handle.stat();
    if (!stat.isFile() || stat.size > 65536) throw new Error('Invalid installed Darwin manifest');
    const bytes = Buffer.alloc(65537);
    let used = 0;
    while (used < bytes.length) {
      check();
      const {bytesRead} = await handle.read(bytes, used, bytes.length - used, null);
      if (!bytesRead) break;
      used += bytesRead;
    }
    if (used > 65536) throw new Error('Installed Darwin manifest exceeds its byte limit');
    const installed = JSON.parse(bytes.subarray(0, used).toString('utf8'));
    if (installed.name !== '@metaharness/darwin' || installed.version !== '0.13.0')
      throw new Error('Expected installed @metaharness/darwin 0.13.0');
    return {package: installed.name, version: installed.version, algorithm: 'evolveNumeric'};
  } finally { await handle.close(); }
}

function integer(value, min, max, name) {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new TypeError(`${name} must be an integer from ${min} to ${max}`);
  return value;
}

function options(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      Object.keys(value).some(key => !['prepared', 'inputs', 'modelId', ...Object.keys(DEFAULTS)].includes(key)))
    throw new TypeError('Unsupported tuning options');
  const config = {...DEFAULTS, ...value};
  for (const [key, [min, max]] of Object.entries(BOUNDS)) integer(config[key], min, max, key);
  if (typeof config.modelId !== 'string' || !config.modelId || config.modelId.length > 256 ||
      Buffer.byteLength(config.modelId) > 256) throw new TypeError('modelId must be a nonempty label of at most 256 UTF-8 bytes');
  if (typeof config.prepared?.run !== 'function') throw new TypeError('prepared.run is required');
  return config;
}

function genome(value) {
  if (!value || Object.getPrototypeOf(value) !== Object.prototype ||
      Object.keys(value).length !== 1) throw new Error('Darwin returned an invalid genome');
  return {batchRows: integer(value.batchRows, 1, 128, 'batchRows')};
}

function exact(actual, expected) {
  if (actual.length !== expected.length) throw new Error('Prepared output shape changed');
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i]) || !Object.is(actual[i], expected[i]))
      throw new Error(`Prepared output differs at ${i}`);
  }
}

/** Tune only call batching. Prepared remains caller-owned and must stay alive.
 * Inputs are snapshotted before the first await. Deadline checks are cooperative:
 * an arbitrary caller method or filesystem operation cannot be preempted here.
 */
export async function tuneBatchRows(value) {
  const config = options(value);
  const check = deadline(config.timeoutMs);
  const {prepared, modelId} = config;
  const {inputSize, outputSize} = preparedDimensions(prepared);
  const checked = checkedInputs(config.inputs, inputSize, outputSize, 1024);
  const totalRows = integer(checked.rows, 128, 1024, 'input rows');
  const outputElements = totalRows * outputSize;
  // Snapshot + reference + two momentarily live encoder results + one row,
  // plus encoder input/output staging at the largest searchable capacity.
  // Allocator/GC overhead and opaque caller/WASM storage are not process bounds.
  const encoderStagingBytes = 128 * (inputSize + outputSize) * 8;
  const numericBufferBytes = (totalRows * inputSize + 3 * outputElements + outputSize) * 8 + encoderStagingBytes;
  if (numericBufferBytes > MAX_BUFFERS)
    throw new RangeError('Tuning numeric buffers exceed 64 MiB');
  check();
  const inputs = new Float64Array(checked.values);
  const expected = new Float64Array(outputElements);
  const inputSha256 = hash(inputs);
  function reference(initialize = false) {
    for (let row = 0; row < totalRows; row++) {
      check();
      const input = inputs.subarray(row * inputSize, (row + 1) * inputSize);
      const {values, rows} = checkedInputs(prepared.run(input), outputSize, 1, 1);
      if (rows !== 1 || values.length !== outputSize) throw new Error('Prepared reference output shape changed');
      if (initialize) expected.set(values, row * outputSize);
      else exact(values, expected.subarray(row * outputSize, (row + 1) * outputSize));
    }
  }
  reference(true);
  if (hash(inputs) !== inputSha256) throw new Error('Prepared reference changed tuning inputs');
  const context = Object.freeze({modelId, inputSize, outputSize, totalRows, precision: 'f64',
    inputSha256, outputSha256: hash(expected),
    modelIdScope: 'Caller-supplied label only; not a model-weight or binary attestation.',
    timingScope: 'Whole encoder workload including guards, copies and owned-result allocation; setup/reference checks excluded.'});
  const workloadSha256 = createHash('sha256').update(JSON.stringify(context)).digest('hex');
  const scope = Object.freeze({...context, workloadSha256});
  const verify = () => {
    check();
    if (hash(inputs) !== scope.inputSha256) throw new Error('Frozen tuning inputs changed');
    reference();
    if (hash(expected) !== scope.outputSha256) throw new Error('Reference outputs changed');
  };
  function runner(batchRows) {
    check();
    const encoder = createEncoder(prepared, {batchRows, maxRows: 1024});
    let last;
    const run = () => { last = encoder.encode(inputs); return last[0]; };
    const verifyOutput = () => { check(); exact(last, expected); };
    try { run(); verifyOutput(); }
    catch (error) { encoder.close(); throw error; }
    return {run, verifyOutput, close: () => encoder.close()};
  }
  const cache = new Map(), observed = new Map(), trials = [];
  const worstCaseDispatches = 1 + config.children + 2 * config.children * (config.generations - 1);
  let dispatchCount = 0, cacheHits = 0, active = false, baselinePrimary;
  let evaluationFailure;
  const evaluator = {evaluate: async (input, variantId) => {
    check();
    if (active || ++dispatchCount > worstCaseDispatches || dispatchCount > 12) {
      evaluationFailure = new Error('Darwin dispatch or concurrency budget exceeded');
      throw evaluationFailure;
    }
    const selected = genome(input);
    if (typeof variantId !== 'string' || !variantId || variantId.length > 128) throw new Error('Invalid Darwin variant identity');
    active = true;
    try {
      await Promise.resolve(); // Concurrent callbacks must fail rather than overlap.
      check();
      let score;
      if (cache.has(selected.batchRows)) {
        cacheHits++;
        score = clone(cache.get(selected.batchRows));
        score.variantId = variantId; score.raw.cacheHit = true;
      } else {
        if (observed.has(variantId)) throw new Error('Darwin reused an observation identity');
        verify();
        const trial = runner(selected.batchRows);
        const samplesNs = [];
        let iterations;
        try {
          iterations = calibrate(trial.run, check);
          for (let i = 0; i < config.samples; i++) samplesNs.push(measure(trial.run, iterations, check));
          trial.verifyOutput(); verify();
        } finally { trial.close(); }
        const rowsPerSecond = totalRows * 1e9 / mean(samplesNs);
        if (!positive(rowsPerSecond)) throw new Error('Invalid tuning throughput');
        if (baselinePrimary === undefined) {
          if (selected.batchRows !== config.baselineBatchRows) throw new Error('Darwin did not measure the requested baseline first');
          baselinePrimary = rowsPerSecond;
        }
        score = {variantId, primary: rowsPerSecond, regressed: rowsPerSecond < baselinePrimary,
          noopRate: 0, costPerWin: 0, raw: {...scope, samplesNs, iterations, meanNs: mean(samplesNs),
            medianNs: median(samplesNs), rowsPerSecond, batchRows: selected.batchRows,
            maximumAbsoluteError: 0, cacheHit: false, observationId: variantId}};
        observed.set(variantId, selected.batchRows);
        cache.set(selected.batchRows, clone(score));
      }
      trials.push({variantId, genome: selected, score: clone(score)});
      return clone(score);
    } catch (error) { evaluationFailure = error; throw error; }
    finally { active = false; }
  }};
  let archive, answer;
  try {
    archive = await mkdtemp(join(tmpdir(), 'ruv-fann-darwin-'));
    check();
    const engine = await darwinIdentity(check);
    const {evolveNumeric} = await import('@metaharness/darwin');
    check();
    const result = await evolveNumeric({genomeSpec: {batchRows: {min: 1, max: 128, type: 'int',
      scale: 'linear', default: config.baselineBatchRows}}, baselineGenome: {batchRows: config.baselineBatchRows},
      evaluator, generations: config.generations, childrenPerGeneration: config.children, seed: config.seed,
      concurrency: 1, mutationSigma: 0.25, crossover: false, workRoot: archive});
    check();
    if (evaluationFailure) throw evaluationFailure;
    if (!result?.baseline || !Array.isArray(result.records) || result.records.length !== trials.length ||
        trials.length === 0 || trials.length !== dispatchCount) throw new Error('Darwin archive does not match observed dispatches');
    const record = item => ({variantId: item?.variant?.id, genome: genome(item?.variant?.genome), score: clone(item?.score)});
    const measured = item => {
      if (!trials.some(trial => JSON.stringify(trial) === JSON.stringify(item))) throw new Error('Darwin returned unmeasured evidence');
      return item;
    };
    const baseline = measured(record(result.baseline));
    if (baseline.genome.batchRows !== config.baselineBatchRows || baseline.score.regressed)
      throw new Error('Darwin changed the requested baseline');
    let winner = result.winner ? measured(record(result.winner)) : baseline;
    let winnerRejected = null;
    if (winner.score.regressed || winner.score.primary < baseline.score.primary) {
      winnerRejected = {variantId: winner.variantId, reason: 'regressed search winner'}; winner = baseline;
    }
    verify();
    const counts = [];
    // One encoder alive at a time, including independent confirmation setup.
    for (const item of [baseline, winner]) {
      const trial = runner(item.genome.batchRows);
      try { counts.push(calibrate(trial.run, check)); trial.verifyOutput(); }
      finally { trial.close(); }
    }
    const paired = [[], []], capacities = [baseline.genome.batchRows, winner.genome.batchRows];
    for (let sample = 0; sample < config.samples; sample++) {
      for (let offset = 0; offset < 2; offset++) {
        check(); const index = (sample + offset) % 2;
        const trial = runner(capacities[index]);
        try { paired[index].push(measure(trial.run, counts[index], check)); trial.verifyOutput(); }
        finally { trial.close(); }
      }
    }
    verify();
    const ratio = pairedRatio(paired[0], paired[1], check);
    const faster = capacities[0] !== capacities[1] && ratio.ci95[0] > 1.05;
    const confirmation = {verified: true, status: faster ? 'passed' : 'not_improved', faster, ...ratio,
      baselineBatchRows: capacities[0], candidateBatchRows: capacities[1], minimumImprovement: 0.05,
      baselineSamplesNs: paired[0], candidateSamplesNs: paired[1], maximumAbsoluteError: 0, workloadSha256,
      scope: 'Fresh paired timing blocks after selection; one host and caller input/model, not generalization.'};
    answer = {schema: 'ruv-fann.batch-tuning/1', status: 'passed', engine, modelId, modelIdScope: context.modelIdScope,
      options: Object.fromEntries(Object.keys(DEFAULTS).map(key => [key, config[key]])), context: scope,
      trialBudget: {absoluteLimit: 12, worstCaseDispatches, dispatchCount,
        actualMeasurementCount: observed.size, cacheHits, concurrency: 1}, trials, baseline, winner, winnerRejected,
      confirmation, recommendation: {batchRows: faster ? capacities[1] : capacities[0]},
      modelCalls: 0, sourceApplied: false, limits: {maxNumericBufferBytes: MAX_BUFFERS,
        numericBufferBytes, encoderStagingBytes, cooperativeDeadline: true}};
  } finally { if (archive) await rm(archive, {recursive: true, force: true}); }
  check();
  answer.temporaryArchiveRemoved = true;
  return answer;
}
