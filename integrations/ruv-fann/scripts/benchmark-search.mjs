/** Fixed-workload numerical retrieval benchmark; not semantic-model evaluation. */
import { writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { createNeuralIndex } from '../src/index.mjs';
import { createRuvectorStore } from '../src/ruvector.mjs';
import { pairedRatio } from '../src/timing.mjs';
import { externalEvidencePath, loadPrepared, sha, sourceSnapshot, vectorBytes } from '../test/e2e/fixture.mjs';

if (process.argv.length !== 4 || process.argv[2] !== '--output') throw new Error('Use --output NEW_EXTERNAL_JSON');
const output = await externalEvidencePath(process.argv[3]);
const before = await sourceSnapshot(), wasmBytes = await vectorBytes();
const model = await loadPrepared('simd', 'dense');
const results = [];
let state = 411;
const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296 * 4 - 2; };
const queryRows = Array.from({ length: 32 }, () => Float64Array.from({ length: 3 }, random));
const queriesSha256 = sha(Buffer.concat(queryRows.map(x => Buffer.from(x.buffer))));
try {
  for (const records of [128, 2048]) {
    const corpus = Array.from({ length: records }, () => Float64Array.from({ length: 3 }, random));
    const corpusSha256 = sha(Buffer.concat(corpus.map(x => Buffer.from(x.buffer))));
    const indexes = [], buildNs = [];
    try {
      for (const useHnsw of [false, true]) {
        const store = await createRuvectorStore({ wasmBytes, dimensions: model.prepared.output_size,
          metric: 'euclidean', useHnsw, maxEntries: records });
        const index = createNeuralIndex({ prepared: model.prepared, store, modelId: model.modelId, maxEntries: records });
        indexes.push(index);
        const start = performance.now();
        for (let row = 0; row < corpus.length; row++) index.add({ id: `row-${row}`, input: corpus[row], metadata: { row } });
        buildNs.push((performance.now() - start) * 1e6);
      }
      assert.equal(indexes[0].indexType, 'flat'); assert.equal(indexes[1].indexType, 'hnsw');
      let matches = 0, total = 0;
      for (const query of queryRows) {
        const expected = indexes[0].search(query, { k: 10 });
        const found = indexes[1].search(query, { k: 10 });
        assert.equal(found.length, 10);
        const correct = new Set(expected.map(x => x.id));
        for (const item of found) { assert(Number.isFinite(item.distance)); matches += correct.has(item.id) ? 1 : 0; total++; }
      }
      const run = index => {
        let checksum = 0;
        for (const query of queryRows) checksum += index.search(query, { k: 10 })[0].distance;
        assert(Number.isFinite(checksum));
      };
      for (let warm = 0; warm < 4; warm++) for (const index of indexes) run(index);
      const samplesNsPerQuery = [[], []];
      for (let sample = 0; sample < 20; sample++) for (let offset = 0; offset < 2; offset++) {
        const which = (sample + offset) % 2;
        const start = performance.now(); run(indexes[which]);
        samplesNsPerQuery[which].push((performance.now() - start) * 1e6 / queryRows.length);
      }
      results.push({ records, queries: queryRows.length, k: 10, dimensions: { input: 3, neuralOutput: 2 },
        corpusSha256, queriesSha256, strictIdRecallAt10: matches / total, buildOnceNs: { flat: buildNs[0], hnsw: buildNs[1] },
        samplesNsPerQuery: { flat: samplesNsPerQuery[0], hnsw: samplesNsPerQuery[1] },
        hnswVsFlatThroughput: pairedRatio(samplesNsPerQuery[0], samplesNsPerQuery[1]) });
    } finally { for (const index of indexes) index.close(); }
  }
} finally { model.prepared.free(); }
const after = await sourceSnapshot(); assert.deepEqual(after, before);
const result = { schema: 'ruv-fann.neural-search-benchmark/1', status: 'passed', runtime: process.version,
  modelJsonSha256: model.modelJsonSha256, fannBinarySha256: model.binarySha256, vectorBinarySha256: sha(wasmBytes),
  scope: 'Real SIMD FANN inference, f64-to-f32 conversion, public validation, actual RuVector search, result copies and metadata; complete query API, not ANN kernel only.',
  limits: 'One host and fixed small-output native model. Strict ID recall is against flat search and can penalize equal-distance ties; it is not semantic accuracy. Timings are paired blocks, not cross-host guarantees. Ns units derive from performance.now() milliseconds.',
  inputsBefore: before, inputsAfter: after, unchanged: true, results };
await writeFile(output, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ status: 'passed', output, results: results.map(x => ({ records: x.records,
  recallAt10: x.strictIdRecallAt10, speedRatio: x.hnswVsFlatThroughput.meanSpeedRatio, ci95: x.hnswVsFlatThroughput.ci95 })) }));
