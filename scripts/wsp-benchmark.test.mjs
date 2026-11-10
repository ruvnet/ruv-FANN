import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { DATA_ROOT, SEEDS, PINNED_DATA, parseDataset, loadDatasets, stratifiedSplit, standardizer, metrics, summary, parseOptions, sha256 } from './wsp-benchmark-lib.mjs';

test('official raw fixtures and provenance lock have exact pinned SHA-256 identities', async () => {
  const { datasets, lock, lockSha256 } = await loadDatasets();
  assert.match(lockSha256, /^[a-f0-9]{64}$/); assert.equal(lock.license, 'CC-BY-4.0'); assert.equal(lock.rawDataModified, false);
  assert.deepEqual(datasets.map(dataset => dataset.rows.length), [150, 178]);
  assert.deepEqual(datasets[0].rows[0], { id: 'iris:1', features: [5.1, 3.5, 1.4, 0.2], label: 0 });
  assert.deepEqual(datasets[1].rows[0], { id: 'wine:1', features: [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065], label: 0 });
  for (const name of ['iris', 'wine']) {
    const bytes = await readFile(join(DATA_ROOT, 'data', `${name}.data`));
    assert.equal(sha256(bytes), PINNED_DATA[name].sha256);
    const modified = Buffer.from(bytes); modified[0] ^= 1;
    assert.throws(() => parseDataset(name, modified), /hash mismatch/);
    assert.throws(() => parseDataset(name, bytes.subarray(1)), /byte count/);
    assert.throws(() => parseDataset('invented', bytes), /unknown dataset/);
  }
});

test('five predeclared stratified partitions are complete, disjoint and hash-stable', async () => {
  const { datasets } = await loadDatasets();
  const expected = {
    iris: ['844f12ccde3f667ef5b0419a9c6b283a1af21ee71c2bab667963870caf91cfae', 'fc095c2c8cca934ca7074c3ad04ac99e5ced07acc69775a64835939685ec4e37', '310e0da0ade132ffc6d7c727a0b7902e6b709b8539da5f13599484d4037f8fd8', '156db411bacf2566e475e855f7810812b21194930410422613d3e2630ebcabee', '5e8aebc147b649de595463d06485eda54e374ab3f9bb4a825dbb062cbf7e84bd'],
    wine: ['9253ee13b4697bad62124d725acdae4f41a2547646f03dcf6c28a44e6e630e5a', 'c943f17be6f08e583e096679101a042933d0e666e7f14833903c45f2aad0a8e5', 'df87736213eb6b4b85ce9cef61211bf0dd27052096972cec7b8a124f9bec539a', '5901c851d869690fad1782a6978c536ac95625ee881019880bf872eaf6d31b0e', '6aa72ce46017b6bc5638b51ab3a8fe08a4a6e02e48245a027c82759c8b06d2b2']
  };
  for (const { rows, provenance } of datasets) for (const [index, seed] of SEEDS.entries()) {
    const split = stratifiedSplit(rows, seed);
    assert.equal(split.hash, expected[provenance.id][index]);
    assert.equal(Object.values(split.ids).flat().length, rows.length);
    assert.equal(new Set(Object.values(split.ids).flat()).size, rows.length);
    for (const label of [0, 1, 2]) {
      const count = rows.filter(row => row.label === label).length;
      assert.equal(split.train.filter(row => row.label === label).length, Math.floor(count * 0.6));
      assert.equal(split.validation.filter(row => row.label === label).length, Math.floor(count * 0.2));
    }
    assert.throws(() => stratifiedSplit(rows, 999), /fixed benchmark/);
  }
});

test('standardization is fitted only to training rows, independent of extreme held-out values', () => {
  const training = [{ id: 'a', features: [2, 7], label: 0 }, { id: 'b', features: [4, 7], label: 1 }];
  const scaler = standardizer(training);
  assert.deepEqual(scaler.mean, [3, 7]); assert.deepEqual(scaler.scale, [1, 1]); assert.deepEqual(scaler.fittedRowIds, ['a', 'b']);
  assert.deepEqual(scaler.transform(training).map(row => row.features), [[-1, 0], [1, 0]]);
  assert.deepEqual(scaler.transform([{ id: 'held-out', features: [103, 8], label: 2 }])[0].features, [100, 1]);
  assert.deepEqual(scaler.mean, [3, 7]); assert.deepEqual(training[0].features, [2, 7]);
});

test('classification metric oracles use explicit confusion counts, known MSE and stable argmax ties', () => {
  // Independent hand calculation: 2/4 correct, class F1s [1,0,1/2], 4 squared errors / 12 elements.
  const result = metrics([0, 1, 2, 2], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0]);
  assert.deepEqual(result.confusion, [[1, 0, 0], [0, 0, 1], [0, 1, 1]]);
  assert.equal(result.accuracy, 0.5); assert.equal(result.macroF1, 0.5); assert.equal(result.mse, 1 / 3);
  const ties = metrics([0, 1, 2], Array(9).fill(0));
  assert.equal(ties.accuracy, 1 / 3); assert.equal(ties.macroF1, 1 / 6); assert.equal(ties.mse, 1 / 3);
  for (const scores of [[], Array(8).fill(0), Array(9).fill(Infinity), Array(9).fill(NaN)]) assert.throws(() => metrics([0, 1, 2], scores));
  assert.throws(() => metrics([3], [0, 0, 0])); assert.throws(() => metrics([], []));
});

test('descriptive interval reports five seeds and preserves negative deltas', () => {
  const result = summary([-0.2, -0.1, 0, 0.1, 0.2]);
  assert.ok(Math.abs(result.mean) < 1e-16); assert.ok(Math.abs(result.sampleSD - Math.sqrt(0.025)) < 1e-15);
  assert.ok(result.ci95[0] < 0 && result.ci95[1] > 0); assert.equal(result.n, 5);
  assert.throws(() => summary([1, 2])); assert.throws(() => summary([0, 0, NaN, 0, 0]));
});

test('benchmark CLI rejects unknown, repeated, relative, missing and NUL path arguments', () => {
  assert.equal(parseOptions(['--output', '/tmp/report.json']).output, '/tmp/report.json');
  assert.deepEqual({ ...parseOptions(['--output', '/tmp/report.json', '--wsp-module', '/tmp/installed/src/index.js', '--python', '/tmp/venv/bin/python']) },
    { output: '/tmp/report.json', wspModule: '/tmp/installed/src/index.js', python: '/tmp/venv/bin/python' });
  for (const args of [[], ['--output'], ['--dataset', '/tmp/raw'], ['--output', 'relative'], ['--output', '/tmp/ok', '--output', '/tmp/no'],
    ['--output', '/tmp/a\0b'], ['--output', '/tmp/ok', '--python', 'python -c bad']]) assert.throws(() => parseOptions(args));
});
