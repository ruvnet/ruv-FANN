import assert from 'node:assert/strict';
import { vectorBytes, sourceSnapshot, loadPrepared, workload } from './fixture.mjs';

/** Run against source exports or an independently extracted npm package. */
export async function runAcceptance({ api, vectorApi, tuner } = {}) {
  api ??= await import('../../src/index.mjs');
  vectorApi ??= await import('../../src/ruvector.mjs');
  tuner ??= await import('../../src/metaharness.mjs');
  const before = await sourceSnapshot(), wasmBytes = await vectorBytes();
  const cases = [], darwin = [];
  for (const variant of ['core', 'simd']) {
    for (const fixtureName of ['dense', 'sparse']) {
      const model = await loadPrepared(variant, fixtureName);
      const { prepared, fixture } = model;
      const inputs = workload(fixture);
      try {
        const original = prepared.run(Float64Array.from(fixture.input));
        for (let i = 0; i < original.length; i++) assert(Math.abs(original[i] - fixture.output[i]) < 1e-12);
        const encoder = api.createEncoder(prepared, { batchRows: 7 });
        let encoded;
        try { encoded = encoder.encode(inputs); } finally { encoder.close(); }
        for (let row = 0; row < 131; row++) {
          const expected = prepared.run(inputs.subarray(row * prepared.input_size, (row + 1) * prepared.input_size));
          for (let j = 0; j < expected.length; j++) assert(Object.is(encoded[row * prepared.output_size + j], expected[j]));
        }
        for (const useHnsw of [false, true]) {
          const store = await vectorApi.createRuvectorStore({ wasmBytes, dimensions: prepared.output_size,
            metric: 'euclidean', useHnsw, maxEntries: 128 });
          const index = api.createNeuralIndex({ prepared, store, modelId: model.modelId, batchRows: 7, maxEntries: 128 });
          const expectedIndexType = useHnsw ? 'hnsw' : 'flat';
          try {
            assert.equal(index.indexType, expectedIndexType);
            assert.equal(index.inputSize, 3); assert.equal(index.outputSize, 2);
            for (let row = 0; row < 33; row++) {
              const metadata = { row, source: 'native-ruv-fann', nested: { approved: true } };
              assert.equal(index.add({ id: `row-${row}`, input: inputs.subarray(row * 3, row * 3 + 3), metadata }), `row-${row}`);
              metadata.nested.approved = false;
            }
            assert.equal(index.size, 33);
            const query = inputs.subarray(21, 24);
            const result = index.search(query, { k: 5 });
            assert.equal(result.length, 5); assert.equal(result[0].id, 'row-7');
            assert.equal(result[0].distance, 0); assert.equal(result[0].metadata.nested.approved, true);
            assert(result.every(r => Number.isFinite(r.distance) && Number.isFinite(r.score) && r.similarity === r.score));
            for (let i = 1; i < result.length; i++) assert(result[i - 1].score >= result[i].score);
            assert.throws(() => index.add({ id: 'row-7', input: query }));
            assert.throws(() => index.add({ id: 'bad', input: Float64Array.of(1, NaN, 3) }));
            assert.throws(() => index.search(Float64Array.of(1), { k: 1 }));
            assert.equal(index.size, 33);
            assert.equal(index.delete('row-7'), true);
            assert(!index.search(query, { k: 5 }).some(r => r.id === 'row-7'));
            cases.push({ variant, fixture: fixtureName, indexType: index.indexType, records: 33,
              outputMaximumAbsoluteError: 0, retrievalExpectedId: 'row-7', retrievalActualId: result[0].id,
              queryDistance: result[0].distance, dimensions: { input: index.inputSize, output: index.outputSize },
              modelJsonSha256: model.modelJsonSha256, binarySha256: model.binarySha256 });
          } finally { index.close(); }
          assert.throws(() => index.search(inputs.subarray(0, 3)));
          assert.deepEqual(prepared.run(Float64Array.from(fixture.input)), original);
        }
        if (fixtureName === 'dense') {
          const receipt = await tuner.tuneBatchRows({ prepared, inputs, modelId: model.modelId,
            baselineBatchRows: 128, samples: 5, generations: 1, children: 1, seed: 7, timeoutMs: 30000 });
          assert.equal(receipt.schema, 'ruv-fann.batch-tuning/1'); assert.equal(receipt.status, 'passed');
          assert.deepEqual(receipt.engine, { package: '@metaharness/darwin', version: '0.13.0', algorithm: 'evolveNumeric' });
          assert.equal(receipt.modelCalls, 0); assert.equal(receipt.sourceApplied, false);
          assert.equal(receipt.baseline.genome.batchRows, 128);
          assert(receipt.trialBudget.dispatchCount <= receipt.trialBudget.worstCaseDispatches);
          assert(receipt.trialBudget.worstCaseDispatches <= 12);
          assert.equal(receipt.confirmation.maximumAbsoluteError, 0);
          assert.deepEqual(receipt.recommendation, receipt.confirmation.faster ? receipt.winner.genome : receipt.baseline.genome);
          darwin.push({ variant, receipt });
        }
      } finally { prepared.free(); }
    }
  }
  const after = await sourceSnapshot(); assert.deepEqual(after, before);
  return { schema: 'ruv-fann.integration-e2e/1', status: 'passed', runtime: process.version,
    sourceValidation: 'passed', modelCalls: 0, actualNeuralInference: true, actualRuVector: true,
    actualDarwin: true, coreAndSimd: true, inputsBefore: before, inputsAfter: after,
    unchanged: true, cases, darwin };
}
