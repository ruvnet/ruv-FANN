// Offline benchmark helpers. These are not production npm APIs.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join, resolve, isAbsolute } from 'node:path';
export const ROOT = fileURLToPath(new URL('../', import.meta.url));
export const DATA_ROOT = join(ROOT, 'packages/wsp/benchmarks');
export const SEEDS = Object.freeze([17, 43, 101, 211, 509]);
export const PINNED_DATA = Object.freeze({
  iris: { sha256: '6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0', bytes: 4551, rows: 150, features: 4, counts: [50, 50, 50] },
  wine: { sha256: '6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659', bytes: 10782, rows: 178, features: 13, counts: [59, 71, 48] }
});
export function sha256(value) { return createHash('sha256').update(value).digest('hex'); }
export function digestJSON(value) { return sha256(JSON.stringify(value)); }
export function rng(seed) {
  let state = seed >>> 0;
  return () => ((state = (Math.imul(1664525, state) + 1013904223) >>> 0) / 4294967296);
}
export function parseDataset(id, bytes) {
  const spec = PINNED_DATA[id];
  assert.ok(spec, 'unknown dataset identity');
  assert.equal(bytes.length, spec.bytes, 'raw dataset byte count mismatch');
  assert.equal(sha256(bytes), spec.sha256, 'raw dataset hash mismatch');
  const labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'];
  const rows = bytes.toString('utf8').trim().split(/\r?\n/).map((line, index) => {
    const cells = line.split(',');
    assert.equal(cells.length, spec.features + 1, 'invalid dataset column count');
    const label = id === 'iris' ? labels.indexOf(cells.pop()) : Number(cells.shift()) - 1;
    assert.ok(Number.isInteger(label) && label >= 0 && label < 3, 'invalid dataset label');
    const features = cells.map(value => {
      assert.match(value, /^-?(?:\d+(?:\.\d*)?|\.\d+)$/, 'dataset numeric syntax');
      const number = Number(value); assert.ok(Number.isFinite(number) && Math.abs(number) <= 1e6); return number;
    });
    return { id: `${id}:${index + 1}`, features, label };
  });
  assert.equal(rows.length, spec.rows);
  assert.deepEqual([0, 1, 2].map(label => rows.filter(row => row.label === label).length), spec.counts);
  return rows;
}
export async function loadDatasets() {
  const bytes = await readFile(join(DATA_ROOT, 'datasets.lock.json'));
  assert.ok(bytes.length < 16384, 'dataset lock exceeds bound');
  const lock = JSON.parse(bytes);
  assert.equal(lock.schema, 'ruv-fann.benchmark-datasets/1');
  assert.equal(lock.license, 'CC-BY-4.0');
  assert.deepEqual(lock.datasets.map(dataset => dataset.id), ['iris', 'wine']);
  const datasets = [];
  for (const entry of lock.datasets) {
    const spec = PINNED_DATA[entry.id];
    assert.equal(entry.sha256, spec.sha256, 'lock fixture hash is not pinned identity');
    assert.equal(entry.rows, spec.rows); assert.equal(entry.features, spec.features); assert.equal(entry.bytes, spec.bytes);
    assert.equal(new URL(entry.sourceUrl).origin, 'https://archive.ics.uci.edu');
    const raw = await readFile(join(DATA_ROOT, 'data', `${entry.id}.data`));
    datasets.push({ provenance: entry, rows: parseDataset(entry.id, raw) });
  }
  return { lock, lockSha256: sha256(bytes), datasets };
}
export function stratifiedSplit(rows, seed) {
  assert.ok(SEEDS.includes(seed), 'seed must be in fixed benchmark plan');
  const random = rng(seed), partitions = { train: [], validation: [], test: [] };
  for (const label of [0, 1, 2]) {
    const group = rows.filter(row => row.label === label);
    for (let index = group.length - 1; index > 0; index--) {
      const other = Math.floor(random() * (index + 1)); [group[index], group[other]] = [group[other], group[index]];
    }
    const trainingCount = Math.floor(group.length * 0.6), validationCount = Math.floor(group.length * 0.2);
    partitions.train.push(...group.slice(0, trainingCount));
    partitions.validation.push(...group.slice(trainingCount, trainingCount + validationCount));
    partitions.test.push(...group.slice(trainingCount + validationCount));
  }
  for (const group of Object.values(partitions)) group.sort((left, right) => Number(left.id.split(':')[1]) - Number(right.id.split(':')[1]));
  const ids = Object.fromEntries(Object.entries(partitions).map(([key, group]) => [key, group.map(row => row.id)]));
  assert.equal(new Set(Object.values(ids).flat()).size, rows.length, 'split overlap');
  assert.equal(Object.values(ids).flat().length, rows.length, 'split coverage');
  return { ...partitions, ids, hash: digestJSON(ids) };
}
export function standardizer(training) {
  assert.ok(training.length > 0 && training.length <= 200, 'training rows out of bounds');
  const width = training[0].features.length;
  const mean = Array.from({ length: width }, (_, column) => training.reduce((sum, row) => sum + row.features[column], 0) / training.length);
  const scale = mean.map((value, column) => Math.sqrt(training.reduce((sum, row) => sum + (row.features[column] - value) ** 2, 0) / training.length) || 1);
  const transform = rows => rows.map(row => ({ ...row, features: row.features.map((value, column) => (value - mean[column]) / scale[column]) }));
  return { mean, scale, transform, fittedRowIds: training.map(row => row.id) };
}
export function asDataset(rows) {
  return { rows: rows.length, inputs: rows.flatMap(row => row.features), targets: rows.flatMap(row => [0, 1, 2].map(label => Number(label === row.label))) };
}
export function metrics(labels, scores) {
  assert.ok(labels.length > 0 && labels.length <= 200, 'metric rows out of bounds');
  assert.equal(scores.length, labels.length * 3, 'metric shape mismatch');
  const confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  let squaredError = 0;
  for (let row = 0; row < labels.length; row++) {
    assert.ok(Number.isInteger(labels[row]) && labels[row] >= 0 && labels[row] < 3, 'metric class out of bounds');
    let predicted = 0;
    for (let label = 0; label < 3; label++) {
      const score = scores[row * 3 + label]; assert.ok(Number.isFinite(score), 'metric scores must be finite');
      if (score > scores[row * 3 + predicted]) predicted = label;
      squaredError += (score - Number(labels[row] === label)) ** 2;
    }
    confusion[labels[row]][predicted]++;
  }
  const accuracy = confusion.reduce((sum, values, label) => sum + values[label], 0) / labels.length;
  const macroF1 = confusion.reduce((sum, values, label) => {
    const truePositive = values[label];
    const denominator = values.reduce((a, b) => a + b, 0) + confusion.reduce((total, row) => total + row[label], 0);
    return sum + (denominator ? 2 * truePositive / denominator : 0);
  }, 0) / 3;
  return { accuracy, macroF1, mse: squaredError / scores.length, confusion };
}
export function summary(values) {
  assert.equal(values.length, 5, 'predeclared five-seed summary required');
  assert.ok(values.every(Number.isFinite));
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sampleSD = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / 4);
  const halfWidth = 2.7764451051977987 * sampleSD / Math.sqrt(5);
  return { n: 5, mean, sampleSD, ci95: [mean - halfWidth, mean + halfWidth], method: 'Student t, df=4; descriptive across overlapping random splits, not independent population evidence' };
}
export function parseOptions(args) {
  assert.ok(args.length <= 6 && args.length % 2 === 0, 'usage: --output ABSOLUTE_PATH [--wsp-module ABSOLUTE_PATH] [--python ABSOLUTE_PATH]');
  const options = Object.create(null);
  for (let index = 0; index < args.length; index += 2) {
    assert.ok(['--output', '--wsp-module', '--python'].includes(args[index]) && !Object.hasOwn(options, args[index]), 'unknown or duplicate benchmark option');
    const value = args[index + 1];
    assert.ok(typeof value === 'string' && value.length <= 4096 && isAbsolute(value) && !value.includes('\0'), 'benchmark paths must be absolute');
    options[args[index]] = resolve(value);
  }
  assert.ok(options['--output'], '--output is required');
  return { output: options['--output'], wspModule: options['--wsp-module'] ?? join(ROOT, 'packages/wsp/src/index.js'), python: options['--python'] };
}
