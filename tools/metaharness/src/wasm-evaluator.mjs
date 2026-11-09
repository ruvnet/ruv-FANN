// Real generated ruv-FANN WASM; the optimizer changes only call batching.
// Whole-workload timings include public guards/copies and all tail calls.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash, randomUUID} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {verifyWasmBuild} from '../../../ruv-swarm/npm/scripts/wasm-provenance.js';

const MODELS = Object.freeze({tiny: [5, 7, 3], medium: [100, 50, 25, 10], large: [784, 128, 64, 10]});
const hash = data => createHash('sha256').update(data).digest('hex');
const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

function integer(value, min, max, label) {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new Error(`Invalid ${label}`);
  return value;
}

function batchRows(genome) {
  if (!genome || typeof genome !== 'object' || Array.isArray(genome) ||
      Object.keys(genome).length !== 1 || !Object.hasOwn(genome, 'batchRows')) throw new Error('Expected only batchRows');
  return integer(genome.batchRows, 1, 128, 'batchRows');
}

async function containedFile(repo, relative, maxBytes = 8_388_608) {
  let current = repo;
  for (const part of relative.split('/')) {
    current = path.join(current, part);
    const stat = await fs.lstat(current);
    if (stat.isSymbolicLink()) throw new Error(`Symlink rejected: ${relative}`);
  }
  const stat = await fs.stat(current);
  if (!stat.isFile() || stat.size > maxBytes) throw new Error(`Invalid bounded artifact: ${relative}`);
  return current;
}

function checkedOutput(actual, expected) {
  if (actual.length !== expected.length) throw new Error('Output shape changed');
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i]) || !Object.is(actual[i], expected[i])) throw new Error(`Output differs at ${i}`);
  }
}

function block(call, iterations) {
  let checksum = 0;
  const start = performance.now();
  for (let i = 0; i < iterations; i++) checksum += call();
  const ns = (performance.now() - start) * 1e6 / iterations;
  if (!Number.isFinite(ns) || ns <= 0 || !Number.isFinite(checksum)) throw new Error('Invalid timing/checksum');
  return ns;
}

function iterationsFor(call, check) {
  let count = 1;
  for (let attempt = 0; attempt < 5; attempt++) {
    check();
    const ns = block(call, count);
    if (ns * count >= 3e6 || count === 1000) return Math.max(1, Math.min(1000, Math.ceil(3e6 / ns)));
    count = Math.min(1000, count * 8);
  }
  return count;
}

// Paired timing-block bootstrap. This is not uncertainty over other hosts or
// workloads. A fresh confirmation after search avoids reusing the search max.
export function pairedRatio(baseline, candidate) {
  if (baseline.length !== candidate.length || baseline.length < 5 ||
      [...baseline, ...candidate].some(x => !Number.isFinite(x) || x <= 0)) throw new Error('Invalid paired samples');
  let state = 20261109;
  const ratios = [];
  for (let draw = 0; draw < 2000; draw++) {
    let a = 0, b = 0;
    for (let i = 0; i < baseline.length; i++) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const index = Math.floor(state / 4294967296 * baseline.length);
      a += baseline[index]; b += candidate[index];
    }
    ratios.push(a / b);
  }
  ratios.sort((a, b) => a - b);
  return {meanSpeedRatio: mean(baseline) / mean(candidate), ci95: [ratios[50], ratios[1949]]};
}

/** Create one isolated model/precision workload for actual Darwin numeric trials. */
export async function createEvaluator({repo, variant = 'core', model = 'tiny', totalRows = 512, samples = 20, timeoutMs = 60000}) {
  integer(totalRows, 128, 1024, 'totalRows');
  integer(samples, 5, 20, 'samples');
  integer(timeoutMs, 100, 120000, 'timeoutMs');
  if (!['core', 'simd'].includes(variant) || !Object.hasOwn(MODELS, model)) throw new Error('Unsupported workload');
  repo = await fs.realpath(repo);
  const npm = path.join(repo, 'ruv-swarm/npm');
  const binaryName = variant === 'core' ? 'ruv_swarm_wasm_bg.wasm' : 'ruv_swarm_simd.wasm';
  const relative = name => `ruv-swarm/npm/wasm/${name}`;
  const binaryPath = await containedFile(repo, relative(binaryName));
  const gluePath = await containedFile(repo, relative('ruv_swarm_wasm.js'));
  const receiptPath = await containedFile(repo, relative('build-info.json'));
  await verifyWasmBuild(npm, repo, ['ruv_swarm_wasm_bg.wasm', 'ruv_swarm_simd.wasm']);
  const [binary, glue, receiptBytes] = await Promise.all([fs.readFile(binaryPath), fs.readFile(gluePath), fs.readFile(receiptPath)]);
  const receipt = JSON.parse(receiptBytes);
  if (hash(binary) !== receipt.artifacts[binaryName] || hash(glue) !== receipt.artifacts['ruv_swarm_wasm.js']) throw new Error('Artifact changed before loading');
  const url = pathToFileURL(gluePath); url.searchParams.set('evaluation', randomUUID());
  const bindings = await import(url.href);
  await bindings.default({module_or_path: binary});
  if (hash(await fs.readFile(gluePath)) !== hash(glue)) throw new Error('Bindings changed during import');
  const features = JSON.parse(bindings.get_features());
  if (features.simd_support !== (variant === 'simd') || !features.prepared_batching) throw new Error('Wrong WASM capabilities');
  const layers = MODELS[model];
  const weights = Float64Array.from({length: layers.slice(1).reduce((n, size, i) => n + layers[i] * size, 0)}, (_, i) => (i * 17 % 31 - 15) / 64);
  const inputs = Float64Array.from({length: totalRows * layers[0]}, (_, i) =>
    ((i % layers[0]) * 7 % 13 - 6) / 8 + (Math.floor(i / layers[0]) % 5) / 64);
  const expected = new Float64Array(totalRows * layers.at(-1));
  const network = new bindings.WasmNeuralNetwork(Uint32Array.from(layers), bindings.ActivationFunction.Linear);
  let prepared;
  try {
    network.set_weights(weights);
    for (let row = 0; row < totalRows; row++) expected.set(network.run(inputs.subarray(row * layers[0], (row + 1) * layers[0])), row * layers.at(-1));
    prepared = network.prepare();
  } finally { network.free(); }
  if ([...expected].some(x => !Number.isFinite(x))) { prepared.free(); throw new Error('Non-finite reference'); }
  const context = Object.freeze({variant, model, layers: Object.freeze([...layers]), activation: 'Linear', precision: 'f64', totalRows,
    inputSha256: hash(inputs), weightsSha256: hash(weights), outputSha256: hash(expected),
    sourceSha256: receipt.sourceSha256, binarySha256: hash(binary), glueSha256: hash(glue), receiptSha256: hash(receiptBytes),
    timingScope: 'Whole fixed-row workload, including JS guards, copies and tail calls; setup/validation excluded.',
    timer: 'performance.now(); ns derived from ms, not ns clock resolution.'});
  const workloadSha256 = hash(JSON.stringify(context));
  const cache = new Map();
  const stats = {measurementCount: 0, cacheHits: 0};
  let closed = false;
  const deadline = performance.now() + timeoutMs;
  const check = () => { if (closed || performance.now() > deadline) throw new Error('Evaluator closed or deadline exceeded'); };

  function runner(rows) {
    check();
    const start = performance.now();
    const batch = prepared.create_batch(rows);
    const creationNs = (performance.now() - start) * 1e6;
    const output = new Float64Array(expected.length);
    const chunks = [];
    for (let first = 0; first < totalRows; first += rows) {
      const active = Math.min(rows, totalRows - first);
      chunks.push([inputs.subarray(first * layers[0], (first + active) * layers[0]), output.subarray(first * layers.at(-1), (first + active) * layers.at(-1)), active]);
    }
    const run = () => {
      for (const [input, target, active] of chunks) batch.run_into(input, target, active);
      return output[0] + output.at(-1);
    };
    try { run(); checkedOutput(output, expected); }
    catch (error) { batch.free(); throw error; }
    return {run, output, free: () => batch.free(), estimatedBytes: batch.estimated_bytes, creationNs, chunks: chunks.length};
  }

  async function evaluate(genome, variantId) {
    check();
    const rows = batchRows(genome);
    if (typeof variantId !== 'string' || variantId.length > 256) throw new Error('Invalid variant identity');
    if (cache.has(rows)) {
      stats.cacheHits++;
      const cached = structuredClone(cache.get(rows));
      return {...cached, variantId, raw: {...cached.raw, cacheHit: true}};
    }
    const trial = runner(rows);
    try {
      const iterations = iterationsFor(trial.run, check);
      const samplesNs = [];
      for (let i = 0; i < samples; i++) { check(); samplesNs.push(block(trial.run, iterations)); }
      checkedOutput(trial.output, expected);
      const outputSha256 = hash(trial.output);
      if (outputSha256 !== context.outputSha256) throw new Error('Output bytes changed');
      const rowsPerSecond = totalRows * 1e9 / mean(samplesNs);
      if (!Number.isFinite(rowsPerSecond) || rowsPerSecond <= 0) throw new Error('Invalid throughput');
      const score = {variantId, primary: rowsPerSecond, regressed: false, noopRate: 0, costPerWin: 0,
        raw: {...context, outputSha256, workloadSha256, batchRows: rows, samplesNs, iterations, meanNs: mean(samplesNs), medianNs: median(samplesNs),
          rowsPerSecond, maximumAbsoluteError: 0, estimatedBytes: trial.estimatedBytes,
          creationOnceNs: trial.creationNs, callsPerWorkload: trial.chunks, cacheHit: false, observationId: variantId}};
      cache.set(rows, structuredClone(score)); stats.measurementCount++;
      return score;
    } finally { trial.free(); }
  }

  async function confirmWinner({baselineGenome, winnerGenome, variantId}) {
    check();
    const baselineBatchRows = batchRows(baselineGenome), candidateBatchRows = batchRows(winnerGenome);
    const baseline = runner(baselineBatchRows);
    let candidate;
    try {
      candidate = runner(candidateBatchRows);
      const counts = [iterationsFor(baseline.run, check), iterationsFor(candidate.run, check)];
      const raw = [[], []], calls = [baseline.run, candidate.run];
      for (let i = 0; i < samples; i++) {
        for (let offset = 0; offset < 2; offset++) {
          check(); const index = (i + offset) % 2;
          raw[index].push(block(calls[index], counts[index]));
        }
      }
      checkedOutput(baseline.output, expected); checkedOutput(candidate.output, expected);
      const ratio = pairedRatio(raw[0], raw[1]);
      const faster = candidateBatchRows !== baselineBatchRows && ratio.ci95[0] > 1.05;
      return {status: faster ? 'passed' : 'not_improved', verified: true, faster, variantId,
        baselineBatchRows, candidateBatchRows, ...ratio, minimumImprovement: 0.05, maximumAbsoluteError: 0,
        baselineSamplesNs: raw[0], candidateSamplesNs: raw[1], iterations: counts, workloadSha256,
        scope: 'Fresh paired confirmation after search; timing-block bootstrap only, one host/workload, no source auto-apply or universal speed claim.'};
    } finally { baseline.free(); candidate?.free(); }
  }
  async function verifyInputs() {
    await verifyWasmBuild(npm, repo, ['ruv_swarm_wasm_bg.wasm', 'ruv_swarm_simd.wasm']);
    for (const [file, expectedHash] of [[binaryPath, context.binarySha256], [gluePath, context.glueSha256], [receiptPath, context.receiptSha256]]) {
      if (hash(await fs.readFile(file)) !== expectedHash) throw new Error('Evaluation artifact changed');
    }
    return {status: 'passed', sourceSha256: context.sourceSha256};
  }
  return {evaluate, confirmWinner, verifyInputs, context: Object.freeze({...context, workloadSha256}),
    get stats() { return {...stats}; }, close() { if (!closed) { closed = true; prepared.free(); } }};
}
