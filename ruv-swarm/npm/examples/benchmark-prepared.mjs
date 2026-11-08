// Compare public WASM calls, including their JavaScript checks and copies.
// Run after building: node examples/benchmark-prepared.mjs > results.json
import { RuvSwarm } from '../src/index.js';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import os from 'node:os';

if (!RuvSwarm.detectSIMDSupport()) {
  throw new Error('This comparison requires a runtime supporting the SIMD binary');
}

function summarize(samples) {
  const sorted = [...samples].sort((a, b) => a - b);
  return { mean_ns: samples.reduce((a, b) => a + b, 0) / samples.length,
    median_ns: (sorted[9] + sorted[10]) / 2, samples_ns: samples };
}

const results = [];
for (const useSIMD of [false, true]) {
  const runtime = await RuvSwarm.initialize({ useSIMD });
  for (const layers of [[5, 7, 3], [100, 50, 25, 10], [784, 128, 64, 10]]) {
    const count = layers.slice(1).reduce((n, size, i) => n + layers[i] * size, 0);
    const weights = Float64Array.from({length: count}, (_, i) => (i * 17 % 31 - 15) / 64);
    const input = Float64Array.from({length: layers[0]}, (_, i) => (i * 7 % 13 - 6) / 8);
    const output = new Float64Array(layers.at(-1));
    const network = runtime.createNeuralNetwork(layers, 'Linear');
    network.set_weights(weights);
    let prepared;
    try {
      const setupStarted = process.hrtime.bigint();
      prepared = network.prepare();
      const setupNs = Number(process.hrtime.bigint() - setupStarted);
      const reference = network.run(input);
      const convenience = prepared.run(input);
      prepared.run_into(input, output);
      let maxAbsError = 0;
      for (let i = 0; i < output.length; i++) {
        maxAbsError = Math.max(maxAbsError, Math.abs(reference[i] - output[i]), Math.abs(reference[i] - convenience[i]));
      }
      if (maxAbsError > 1e-12) throw new Error(`Numerical mismatch: ${maxAbsError}`);
      const iterations = count > 10000 ? 100 : count > 1000 ? 1000 : 10000;
      const samples = { legacy_run: [], prepared_run: [], prepared_run_into: [] };
      const calls = {
        legacy_run: () => network.run(input)[0],
        prepared_run: () => prepared.run(input)[0],
        prepared_run_into: () => { prepared.run_into(input, output); return output[0]; },
      };
      let checksum = 0;
      // Warm each API independently before interleaving timed samples.
      for (const call of Object.values(calls)) {
        for (let i = 0; i < iterations; i++) checksum += call();
      }
      const names = Object.keys(calls);
      for (let sample = 0; sample < 20; sample++) {
        for (let offset = 0; offset < names.length; offset++) {
          const name = names[(sample + offset) % names.length];
          const call = calls[name];
          const started = process.hrtime.bigint();
          for (let i = 0; i < iterations; i++) checksum += call();
          samples[name].push(Number(process.hrtime.bigint() - started) / iterations);
        }
      }
      if (!Number.isFinite(checksum)) throw new Error('Invalid timing checksum');
      results.push({ variant: useSIMD ? 'simd' : 'core', layers, parameters: count,
        precision: 'f64', iterations_per_sample: iterations, samples: 20,
        prepare_once_ns: setupNs, estimated_bytes: prepared.estimated_bytes,
        maximum_absolute_output_error: maxAbsError, checksum,
        ...Object.fromEntries(Object.entries(samples).map(([name, values]) => [name, summarize(values)])),
      });
    } finally {
      if (prepared) prepared.free();
      network.free();
    }
  }
}

const artifactHashes = Object.fromEntries(['ruv_swarm_wasm_bg.wasm', 'ruv_swarm_simd.wasm'].map(name => [name,
  createHash('sha256').update(readFileSync(new URL(`../wasm/${name}`, import.meta.url))).digest('hex')]));
console.log(JSON.stringify({ node: process.version, v8: process.versions.v8, cpu: os.cpus()[0]?.model,
  date_utc: new Date().toISOString(), artifact_sha256: artifactHashes,
  scope: 'Synthetic dense f64 models. Public JS/WASM boundary checks, input/output copies and result allocations are included. No accuracy or universal speed claim.',
  setup_scope: 'One observed preparation call per fixture, without destruction; not a setup-cost distribution.',
  method: '20 interleaved samples after one warm-up batch per API. Default release build. Compare measured APIs, not a native f32 benchmark.',
  results }, null, 2));
