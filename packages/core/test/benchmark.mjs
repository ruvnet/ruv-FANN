import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { loadRuntime } from '../src/index.js';
import { supportsSIMD } from 'ruv-fann-wasm';
const inputSize = 16, outputSize = 8, rows = 128, iterations = 500;
const model = { schema: 'ruv-fann.dense-layers/1', layers: [{ inputs: inputSize, outputs: outputSize,
  weights: Array.from({ length: inputSize * outputSize }, (_, index) => Math.sin(index) / inputSize),
  bias: Array(outputSize).fill(0.1), activation: 'Tanh', steepness: 1 }] };
const input = Float64Array.from({ length: rows * inputSize }, (_, index) => Math.cos(index / 10));
for (const simd of [false, ...(supportsSIMD() ? [true] : [])]) {
  const runtime = await loadRuntime({ simd });
  const batch = runtime.prepare(model, { maxRows: rows }), single = runtime.prepare(model, { maxRows: 1 });
  try {
    const output = new Float64Array(rows * outputSize), rowOutput = new Float64Array(outputSize);
    const rowInputs = Array.from({ length: rows }, (_, row) => input.subarray(row * inputSize, (row + 1) * inputSize));
    batch.predictInto(input, output);
    rowInputs.forEach((row, index) => assert.deepEqual(single.predict(row), output.subarray(index * outputSize, (index + 1) * outputSize)));
    const batched = () => batch.predictInto(input, output);
    const sequential = () => { for (const row of rowInputs) single.predictInto(row, rowOutput); };
    for (let warm = 0; warm < 100; warm++) { batched(); sequential(); }
    const samples = { batch: [], row: [] };
    // Alternate ordering and report medians. Timings are measurements, never pass/fail thresholds.
    for (let round = 0; round < 5; round++) for (const key of round % 2 ? ['row', 'batch'] : ['batch', 'row']) {
      const start = performance.now();
      for (let index = 0; index < iterations; index++) (key === 'batch' ? batched : sequential)();
      samples[key].push(performance.now() - start);
    }
    const median = values => values.sort((a, b) => a - b)[2];
    const batchMs = median(samples.batch), rowMs = median(samples.row);
    console.log(JSON.stringify({ backend: runtime.backend, node: process.version, inputSize, outputSize, rows,
      iterations, medianBatchMs: batchMs, medianRowMs: rowMs, rowOverBatch: rowMs / batchMs,
      rowsPerSecondBatch: iterations * rows / batchMs * 1000, estimatedBytes: batch.estimatedBytes,
      exactBatchRowParity: true }));
  } finally { batch.dispose(); single.dispose(); }
}
